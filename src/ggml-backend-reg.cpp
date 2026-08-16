#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-dl.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <cctype>
#include <climits>
#include <cstdio>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#include "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#include "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#include "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#include "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#include "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#include "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#include "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#include "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#include "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#include "ggml-et.h"
#endif

namespace fs = std::filesystem;

static std::string path_str(const fs::path & path) {
    try {
#if defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        const std::u8string u8str = path.u8string();
        return std::string(reinterpret_cast<const char *>(u8str.data()), u8str.size());
#else
        // C++17: u8string() returns std::string
        return path.u8string();
#endif
    } catch (...) {
        return std::string();
    }
}

struct ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

typedef const char * (*openasr_ggml_backend_abi_v1_t)(void);
typedef int (*openasr_ggml_backend_probe_v1_t)(const char * expected_target, char * driver_out, size_t driver_out_capacity);

static bool openasr_backend_abi_matches(dl_handle * handle, const char * expected_abi, const fs::path & path, bool silent) {
    if (expected_abi == nullptr || expected_abi[0] == '\0') {
        return true;
    }
    auto abi_fn = (openasr_ggml_backend_abi_v1_t) dl_get_sym(handle, "openasr_ggml_backend_abi_v1");
    const char * actual = abi_fn != nullptr ? abi_fn() : nullptr;
    if (actual == nullptr || std::strcmp(actual, expected_abi) != 0) {
        if (!silent) {
            GGML_LOG_ERROR("%s: refusing %s before initialization: OpenASR backend ABI mismatch\n",
                __func__, path_str(path).c_str());
        }
        return false;
    }
    return true;
}

static bool openasr_backend_provider_matches(dl_handle * handle, const char * expected_provider, const fs::path & path, bool silent) {
    if (expected_provider == nullptr || expected_provider[0] == '\0') {
        return true;
    }
    auto provider_fn = (openasr_ggml_backend_abi_v1_t) dl_get_sym(handle, "openasr_ggml_backend_provider_v1");
    const char * actual = provider_fn != nullptr ? provider_fn() : nullptr;
    std::string expected = std::string("ggml-") + expected_provider;
    bool matches = actual != nullptr && (expected == actual ||
        (std::strcmp(expected_provider, "cpu") == 0 && std::strncmp(actual, "ggml-cpu", 8) == 0));
    if (!matches) {
        if (!silent) {
            GGML_LOG_ERROR("%s: refusing %s before initialization: expected OpenASR provider %s\n",
                __func__, path_str(path).c_str(), expected_provider);
        }
        return false;
    }
    return true;
}

static bool openasr_parse_driver_version(const char * value, std::vector<unsigned long long> & parts) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    unsigned long long current = 0;
    bool have_digit = false;
    for (const char * cursor = value;; ++cursor) {
        const unsigned char ch = static_cast<unsigned char>(*cursor);
        if (std::isdigit(ch)) {
            const unsigned digit = ch - static_cast<unsigned char>('0');
            if (current > (ULLONG_MAX - digit) / 10) {
                return false;
            }
            current = current * 10 + digit;
            have_digit = true;
            continue;
        }
        if ((*cursor == '.' || *cursor == '\0') && have_digit) {
            parts.push_back(current);
            current = 0;
            have_digit = false;
            if (*cursor == '\0') {
                return true;
            }
            continue;
        }
        return false;
    }
}

static bool openasr_driver_version_at_least(const char * current, const char * minimum) {
    if (minimum == nullptr || minimum[0] == '\0') {
        return true;
    }
    std::vector<unsigned long long> current_parts;
    std::vector<unsigned long long> minimum_parts;
    if (!openasr_parse_driver_version(current, current_parts) ||
        !openasr_parse_driver_version(minimum, minimum_parts)) {
        return false;
    }
    const size_t width = std::max(current_parts.size(), minimum_parts.size());
    current_parts.resize(width, 0);
    minimum_parts.resize(width, 0);
    return current_parts >= minimum_parts;
}

static bool openasr_backend_runtime_matches(
        dl_handle * handle,
        const char * expected_target,
        const char * minimum_driver,
        const fs::path & path,
        bool silent,
        std::string * actual_driver_out = nullptr) {
    if (expected_target == nullptr || expected_target[0] == '\0') {
        return true;
    }
    auto probe_fn = (openasr_ggml_backend_probe_v1_t) dl_get_sym(handle, "openasr_ggml_backend_probe_v1");
    char actual_driver[64] = {};
    const bool matches = probe_fn != nullptr && probe_fn(expected_target, actual_driver, sizeof(actual_driver)) == 1 &&
        actual_driver[0] != '\0' && openasr_driver_version_at_least(actual_driver, minimum_driver);
    if (!matches) {
        if (!silent) {
            GGML_LOG_ERROR("%s: refusing %s before initialization: live target/driver proof failed for %s\n",
                __func__, path_str(path).c_str(), expected_target);
        }
        return false;
    }
    if (actual_driver_out != nullptr) {
        *actual_driver_out = actual_driver;
    }
    return true;
}

static bool openasr_verify_loaded_backend(
        dl_handle * handle,
        const fs::path & path,
        bool silent,
        const char * expected_abi,
        const char * expected_provider,
        const char * expected_target,
        const char * minimum_driver,
        std::string * actual_driver_out = nullptr) {
    return openasr_backend_abi_matches(handle, expected_abi, path, silent) &&
        openasr_backend_provider_matches(handle, expected_provider, path, silent) &&
        openasr_backend_runtime_matches(handle, expected_target, minimum_driver, path, silent, actual_driver_out);
}

struct ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#ifdef GGML_USE_CUDA
        register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_METAL
        register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
        register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
    // Add runtime disable check
    if (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } else {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#endif
#ifdef GGML_USE_WEBGPU
        register_backend(ggml_backend_webgpu_reg());
#endif
#ifdef GGML_USE_ZDNN
        register_backend(ggml_backend_zdnn_reg());
#endif
#ifdef GGML_USE_VIRTGPU_FRONTEND
        register_backend(ggml_backend_virtgpu_reg());
#endif

#ifdef GGML_USE_OPENCL
        register_backend(ggml_backend_opencl_reg());
#endif
#ifdef GGML_USE_ZENDNN
        register_backend(ggml_backend_zendnn_reg());
#endif
#ifdef GGML_USE_HEXAGON
        register_backend(ggml_backend_hexagon_reg());
#endif
#ifdef GGML_USE_CANN
        register_backend(ggml_backend_cann_reg());
#endif
#ifdef GGML_USE_BLAS
        register_backend(ggml_backend_blas_reg());
#endif
#ifdef GGML_USE_RPC
        register_backend(ggml_backend_rpc_reg());
#endif
#ifdef GGML_USE_OPENVINO
        register_backend(ggml_backend_openvino_reg());
#endif
#ifdef GGML_USE_ET
        register_backend(ggml_backend_et_reg());
#endif
#ifdef GGML_USE_CPU
        register_backend(ggml_backend_cpu_reg());
#endif
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        for (auto & entry : backends) {
            if (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        if (!reg) {
            return;
        }

        for (auto & entry : backends) {
            if (entry.reg == reg) {
                return;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    void register_device(ggml_backend_dev_t device) {
        for (auto & dev : devices) {
            if (dev == device) {
                return;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(
            const fs::path & path,
            bool silent,
            const char * expected_abi = nullptr,
            const char * expected_provider = nullptr,
            const char * expected_target = nullptr,
            const char * minimum_driver = nullptr,
            const std::vector<fs::path> & dependency_dirs = {}) {
        dl_handle_ptr handle { dl_load_library(path, dependency_dirs) };
        if (!handle) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            return nullptr;
        }

        if (!openasr_verify_loaded_backend(handle.get(), path, silent, expected_abi, expected_provider,
                                           expected_target, minimum_driver)) {
            return nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        if (score_fn && score_fn() == 0) {
            if (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            return nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        if (!backend_init_fn) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            return nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        if (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            if (!silent) {
                if (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } else {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            return nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        return reg;
    }

    void unload_backend(ggml_backend_reg_t reg, bool silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](const ggml_backend_reg_entry & entry) { return entry.reg == reg; });

        if (it == backends.end()) {
            if (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            return;
        }

        if (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { return ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

void ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
static bool striequals(const char * a, const char * b) {
    for (; *a && *b; a++, b++) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;
        }
    }
    return *a == *b;
}

size_t ggml_backend_reg_count() {
    return get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_count());
    return get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (striequals(ggml_backend_reg_name(reg), name)) {
            return reg;
        }
    }
    return nullptr;
}

// Device enumeration
size_t ggml_backend_dev_count() {
    return get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_count());
    return get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (striequals(ggml_backend_dev_name(dev), name)) {
            return dev;
        }
    }
    return nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == type) {
            return dev;
        }
    }
    return nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(const char * path) {
#ifdef OPENASR_VERIFIED_BACKEND_LOADING_ONLY
    GGML_UNUSED(path);
    return nullptr;
#else
    return get_reg().load_backend(path, false);
#endif
}

ggml_backend_reg_t ggml_backend_load_utf8(const char * path_utf8) {
#ifdef OPENASR_VERIFIED_BACKEND_LOADING_ONLY
    GGML_UNUSED(path_utf8);
    return nullptr;
#else
    if (path_utf8 == nullptr) {
        return nullptr;
    }
    return get_reg().load_backend(fs::u8path(path_utf8), false);
#endif
}

ggml_backend_reg_t ggml_backend_load_verified_utf8(
        const char * path_utf8,
        const char * expected_openasr_abi_v1,
        const char * expected_provider_v1) {
    if (path_utf8 == nullptr || expected_openasr_abi_v1 == nullptr || expected_openasr_abi_v1[0] == '\0' ||
        expected_provider_v1 == nullptr || expected_provider_v1[0] == '\0') {
        return nullptr;
    }
    return get_reg().load_backend(
        fs::u8path(path_utf8), false, expected_openasr_abi_v1, expected_provider_v1);
}

ggml_backend_reg_t ggml_backend_load_verified_v2_utf8(
        const char * path_utf8,
        const char * expected_openasr_abi_v1,
        const char * expected_provider_v1,
        const char * expected_device_target,
        const char * minimum_driver_version) {
    if (path_utf8 == nullptr || expected_openasr_abi_v1 == nullptr || expected_openasr_abi_v1[0] == '\0' ||
        expected_provider_v1 == nullptr || expected_provider_v1[0] == '\0' ||
        expected_device_target == nullptr || expected_device_target[0] == '\0') {
        return nullptr;
    }
    return get_reg().load_backend(fs::u8path(path_utf8), false, expected_openasr_abi_v1,
        expected_provider_v1, expected_device_target, minimum_driver_version);
}

static bool openasr_parse_dependency_dirs(
        const char * const * dependency_dirs_utf8,
        size_t dependency_dir_count,
        std::vector<fs::path> & dependency_dirs) {
    if (dependency_dir_count == 0) {
        return dependency_dirs_utf8 == nullptr;
    }
    if (dependency_dirs_utf8 == nullptr) {
        return false;
    }
    dependency_dirs.reserve(dependency_dir_count);
    for (size_t index = 0; index < dependency_dir_count; ++index) {
        if (dependency_dirs_utf8[index] == nullptr || dependency_dirs_utf8[index][0] == '\0') {
            return false;
        }
        fs::path dependency_dir = fs::u8path(dependency_dirs_utf8[index]);
        if (!dependency_dir.is_absolute()) {
            return false;
        }
        dependency_dirs.push_back(std::move(dependency_dir));
    }
    return true;
}

ggml_backend_reg_t ggml_backend_load_verified_v3_utf8(
        const char * path_utf8,
        const char * const * dependency_dirs_utf8,
        size_t dependency_dir_count,
        const char * expected_openasr_abi_v1,
        const char * expected_provider_v1,
        const char * expected_device_target,
        const char * minimum_driver_version) {
    if (path_utf8 == nullptr || expected_openasr_abi_v1 == nullptr || expected_openasr_abi_v1[0] == '\0' ||
        expected_provider_v1 == nullptr || expected_provider_v1[0] == '\0' ||
        expected_device_target == nullptr || expected_device_target[0] == '\0') {
        return nullptr;
    }
    std::vector<fs::path> dependency_dirs;
    if (!openasr_parse_dependency_dirs(dependency_dirs_utf8, dependency_dir_count, dependency_dirs)) {
        return nullptr;
    }
    return get_reg().load_backend(fs::u8path(path_utf8), false, expected_openasr_abi_v1,
        expected_provider_v1, expected_device_target, minimum_driver_version, dependency_dirs);
}

bool ggml_backend_probe_verified_v2_utf8(
        const char * path_utf8,
        const char * expected_openasr_abi_v1,
        const char * expected_provider_v1,
        const char * expected_device_target,
        const char * minimum_driver_version,
        char * driver_out,
        size_t driver_out_capacity) {
    if (driver_out != nullptr && driver_out_capacity > 0) {
        driver_out[0] = '\0';
    }
    if (path_utf8 == nullptr || expected_openasr_abi_v1 == nullptr || expected_openasr_abi_v1[0] == '\0' ||
        expected_provider_v1 == nullptr || expected_provider_v1[0] == '\0' ||
        expected_device_target == nullptr || expected_device_target[0] == '\0') {
        return false;
    }
    const fs::path path = fs::u8path(path_utf8);
    dl_handle_ptr handle { dl_load_library(path) };
    if (!handle) {
        return false;
    }
    std::string actual_driver;
    if (!openasr_verify_loaded_backend(handle.get(), path, false, expected_openasr_abi_v1,
                                       expected_provider_v1, expected_device_target,
                                       minimum_driver_version, &actual_driver)) {
        return false;
    }
    if (driver_out != nullptr && driver_out_capacity > 0) {
        std::snprintf(driver_out, driver_out_capacity, "%s", actual_driver.c_str());
    }
    return true;
}

bool ggml_backend_probe_verified_v3_utf8(
        const char * path_utf8,
        const char * const * dependency_dirs_utf8,
        size_t dependency_dir_count,
        const char * expected_openasr_abi_v1,
        const char * expected_provider_v1,
        const char * expected_device_target,
        const char * minimum_driver_version,
        char * driver_out,
        size_t driver_out_capacity) {
    if (driver_out != nullptr && driver_out_capacity > 0) {
        driver_out[0] = '\0';
    }
    if (path_utf8 == nullptr || expected_openasr_abi_v1 == nullptr || expected_openasr_abi_v1[0] == '\0' ||
        expected_provider_v1 == nullptr || expected_provider_v1[0] == '\0' ||
        expected_device_target == nullptr || expected_device_target[0] == '\0') {
        return false;
    }
    std::vector<fs::path> dependency_dirs;
    if (!openasr_parse_dependency_dirs(dependency_dirs_utf8, dependency_dir_count, dependency_dirs)) {
        return false;
    }
    const fs::path path = fs::u8path(path_utf8);
    dl_handle_ptr handle { dl_load_library(path, dependency_dirs) };
    if (!handle) {
        return false;
    }
    std::string actual_driver;
    if (!openasr_verify_loaded_backend(handle.get(), path, false, expected_openasr_abi_v1,
                                       expected_provider_v1, expected_device_target,
                                       minimum_driver_version, &actual_driver)) {
        return false;
    }
    if (driver_out != nullptr && driver_out_capacity > 0) {
        std::snprintf(driver_out, driver_out_capacity, "%s", actual_driver.c_str());
    }
    return true;
}

ggml_backend_reg_t ggml_backend_load_best_verified_utf8(
        const char * const * paths_utf8,
        size_t path_count,
        const char * expected_openasr_abi_v1,
        const char * expected_provider_v1) {
    if (paths_utf8 == nullptr || path_count == 0 || expected_openasr_abi_v1 == nullptr ||
        expected_openasr_abi_v1[0] == '\0' || expected_provider_v1 == nullptr ||
        expected_provider_v1[0] == '\0') {
        return nullptr;
    }
    int best_score = 0;
    fs::path best_path;
    for (size_t index = 0; index < path_count; ++index) {
        if (paths_utf8[index] == nullptr || paths_utf8[index][0] == '\0') {
            return nullptr;
        }
        const fs::path path = fs::u8path(paths_utf8[index]);
        if (!path.is_absolute()) {
            return nullptr;
        }
        dl_handle_ptr handle { dl_load_library(path) };
        if (!handle || !openasr_verify_loaded_backend(handle.get(), path, false,
                expected_openasr_abi_v1, expected_provider_v1, nullptr, nullptr)) {
            return nullptr;
        }
        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        const int score = score_fn != nullptr ? score_fn() : 1;
        if (score > best_score) {
            best_score = score;
            best_path = path;
        }
    }
    if (best_score <= 0 || best_path.empty()) {
        return nullptr;
    }
    return get_reg().load_backend(best_path, false, expected_openasr_abi_v1, expected_provider_v1);
}

void ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, true);
}

static fs::path get_executable_path() {
#if defined(__APPLE__)
    // get executable path
    std::vector<char> path;
    uint32_t size;
    while (true) {
        size = path.size();
        if (_NSGetExecutablePath(path.data(), &size) == 0) {
            break;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + "/";
#elif defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<char> path(1024);
    while (true) {
        // get executable path
#    if defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        if (len == -1) {
            break;
        }
        if (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            if (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            break;
        }
        path.resize(path.size() * 2);
    }

    return base_path + "/";
#elif defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(NULL, path.data(), path.size());
    if (len == 0) {
        return {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + L"\\";
#else
    return {};
#endif
}

static fs::path backend_filename_prefix() {
#ifdef _WIN32
    return fs::u8path("ggml-");
#else
    return fs::u8path("libggml-");
#endif
}

static fs::path backend_filename_extension() {
#ifdef _WIN32
    return fs::u8path(".dll");
#else
    return fs::u8path(".so");
#endif
}

static ggml_backend_reg_t ggml_backend_load_best(
        const char * name,
        bool silent,
        const char * user_search_path,
        const char * expected_openasr_abi_v1 = nullptr) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    const fs::path name_path = fs::u8path(name);
    const fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    const fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> search_paths;
    if (user_search_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        search_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        search_paths.push_back(get_executable_path());
        search_paths.push_back(fs::current_path());
    } else {
        search_paths.push_back(fs::u8path(user_search_path));
    }

    int best_score = 0;
    fs::path best_path;
    std::error_code ec;

    for (const auto & search_path : search_paths) {
        if (!fs::exists(search_path, ec)) {
            if (ec) {
                GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(search_path).c_str(), ec.message().c_str());
            } else {
                GGML_LOG_DEBUG("%s: search path %s does not exist\n", __func__, path_str(search_path).c_str());
            }
            continue;
        }
        fs::directory_iterator dir_it(search_path, fs::directory_options::skip_permission_denied);
        for (const auto & entry : dir_it) {
            if (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                if (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    if (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    if (handle) {
                        if (!openasr_backend_abi_matches(handle.get(), expected_openasr_abi_v1, entry.path(), silent)) {
                            continue;
                        }
                        if (!openasr_backend_provider_matches(handle.get(), name, entry.path(), silent)) {
                            continue;
                        }
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        if (score_fn) {
                            int s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            if (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } else {
                            if (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }
    }

    if (best_score == 0) {
        // try to load the base backend
        for (const auto & search_path : search_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = search_path / filename;
            if (std::error_code ec; fs::exists(path, ec)) {
                return get_reg().load_backend(path, silent, expected_openasr_abi_v1, name);
            } else {
                if (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        return nullptr;
    }

    return get_reg().load_backend(best_path, silent, expected_openasr_abi_v1, name);
}

void ggml_backend_load_bundled_from_path(const char * dir_path_utf8) {
#ifdef OPENASR_VERIFIED_BACKEND_LOADING_ONLY
    GGML_UNUSED(dir_path_utf8);
    return;
#else
    if (dir_path_utf8 == nullptr) {
        return;
    }
    const fs::path dir = fs::u8path(dir_path_utf8);
    if (!dir.is_absolute()) {
        GGML_LOG_ERROR("%s: bundled backend directory must be absolute\n", __func__);
        return;
    }
#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif
    const std::string path = path_str(dir);
    ggml_backend_load_best("vulkan", silent, path.c_str());
    ggml_backend_load_best("cpu", silent, path.c_str());
#endif
}

void ggml_backend_load_bundled_verified_from_path(
        const char * dir_path_utf8,
        const char * expected_openasr_abi_v1) {
#ifdef OPENASR_VERIFIED_BACKEND_LOADING_ONLY
    GGML_UNUSED(dir_path_utf8);
    GGML_UNUSED(expected_openasr_abi_v1);
    return;
#else
    if (dir_path_utf8 == nullptr || expected_openasr_abi_v1 == nullptr || expected_openasr_abi_v1[0] == '\0') {
        return;
    }
    const fs::path dir = fs::u8path(dir_path_utf8);
    if (!dir.is_absolute()) {
        GGML_LOG_ERROR("%s: bundled backend directory must be absolute\n", __func__);
        return;
    }
#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif
    const std::string path = path_str(dir);
    ggml_backend_load_best("vulkan", silent, path.c_str(), expected_openasr_abi_v1);
    ggml_backend_load_best("cpu", silent, path.c_str(), expected_openasr_abi_v1);
#endif
}

void ggml_backend_load_all() {
#ifdef OPENASR_VERIFIED_BACKEND_LOADING_ONLY
    return;
#else
    ggml_backend_load_all_from_path(nullptr);
#endif
}

void ggml_backend_load_all_from_path(const char * dir_path) {
#ifdef OPENASR_VERIFIED_BACKEND_LOADING_ONLY
    GGML_UNUSED(dir_path);
    return;
#else
#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif

    ggml_backend_load_best("blas", silent, dir_path);
    ggml_backend_load_best("zendnn", silent, dir_path);
    ggml_backend_load_best("cann", silent, dir_path);
    ggml_backend_load_best("cuda", silent, dir_path);
    ggml_backend_load_best("hip", silent, dir_path);
    ggml_backend_load_best("metal", silent, dir_path);
    ggml_backend_load_best("rpc", silent, dir_path);
    ggml_backend_load_best("sycl", silent, dir_path);
    ggml_backend_load_best("vulkan", silent, dir_path);
    ggml_backend_load_best("virtgpu", silent, dir_path);
    ggml_backend_load_best("opencl", silent, dir_path);
    ggml_backend_load_best("hexagon", silent, dir_path);
    ggml_backend_load_best("musa", silent, dir_path);
    ggml_backend_load_best("openvino", silent, dir_path);
    ggml_backend_load_best("cpu", silent, dir_path);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    const char * backend_path = std::getenv("GGML_BACKEND_PATH");
    if (backend_path) {
        ggml_backend_load(backend_path);
    }
#endif
}
