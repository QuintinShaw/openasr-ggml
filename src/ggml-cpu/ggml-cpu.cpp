#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "repack.h"
#include "traits.h"
#include "ggml-impl.h"
#include "amx/amx.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

#if defined(__APPLE__)
#    include <mach/mach.h>
#    include <sys/sysctl.h>
#    include <sys/types.h>
#elif defined(__linux__)
#    include <sys/sysinfo.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        if (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        if (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        if (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        if (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        return bufts;
    }();

    return bufts;
}

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    static std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        return bufts;
    }();

    return extra_bufts.data();

    GGML_UNUSED(device);
}

static bool ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    for (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra == buft) {
            return true;
        }
    }
    return false;
}

// CPU backend - backend (stream)

struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    bool                use_ref;  // use reference implementation

    uint64_t            memory_generation;
    uint64_t            memory_high_water;
    uint64_t            allocation_failures;
    uint64_t            quarantine_generation;
    uint32_t            memory_health;
    int64_t             last_native_error;
};

static const char * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    return "CPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = new (std::nothrow) ggml_backend_plan_cpu;
    if (cpu_plan == NULL) {
        cpu_ctx->allocation_failures++;
        return NULL;
    }

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = new (std::nothrow) uint8_t[cpu_plan->cplan.work_size];
        if (cpu_plan->cplan.work_data == NULL) {
            cpu_ctx->allocation_failures++;
            delete cpu_plan;
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *) backend->context;
    if (cpu_ctx->memory_health == GGML_BACKEND_MEMORY_QUARANTINED) {
        return GGML_STATUS_BACKEND_POISONED;
    }
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    if (cpu_ctx->memory_health == GGML_BACKEND_MEMORY_QUARANTINED) {
        return GGML_STATUS_BACKEND_POISONED;
    }

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    if (cpu_ctx->work_size < cplan.work_size) {
        uint8_t * replacement = new (std::nothrow) uint8_t[cplan.work_size];
        if (replacement == NULL) {
            cpu_ctx->allocation_failures++;
            return GGML_STATUS_ALLOC_FAILED;
        }
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = replacement;
        cpu_ctx->work_size = cplan.work_size;
        cpu_ctx->memory_high_water = std::max(cpu_ctx->memory_high_water, (uint64_t) cpu_ctx->work_size);
        cpu_ctx->memory_generation++;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;

    return ggml_graph_compute(cgraph, &cplan);
}

static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record_status     = */ NULL,
    /* .event_wait_status       = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_cpu_guid(void) {
    static ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    return &guid;
}

ggml_backend_t ggml_backend_cpu_init(void) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    struct ggml_backend_cpu_context * ctx = new (std::nothrow) ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;
    ctx->use_ref             = false;
    ctx->memory_generation   = 1;
    ctx->memory_high_water   = 0;
    ctx->allocation_failures = 0;
    ctx->quarantine_generation = 0;
    ctx->memory_health       = GGML_BACKEND_MEMORY_HEALTHY;
    ctx->last_native_error   = 0;

    ggml_backend_t cpu_backend = new (std::nothrow) ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return cpu_backend;
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    if (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

// CPU backend - device

struct ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
        }
#elif defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (strncmp(buf, "model name", 10) == 0) {
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            if (RegQueryValueExA(hKey,
                                "ProcessorNameString",
                                NULL,
                                NULL,
                                NULL,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                if (RegQueryValueExA(hKey,
                                    "ProcessorNameString",
                                    NULL,
                                    NULL,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    if (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            RegCloseKey(hKey);
        }
#endif
    }
};

static const char * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    return "CPU";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#elif defined(__APPLE__)
    int64_t memory_size = 0;
    size_t memory_size_len = sizeof(memory_size);
    if (sysctlbyname("hw.memsize", &memory_size, &memory_size_len, NULL, 0) != 0) {
        *free = 0;
        *total = 0;
        return;
    }
    vm_statistics64_data_t vm = {};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &count) != KERN_SUCCESS) {
        *free = 0;
        *total = (size_t) memory_size;
        return;
    }
    const uint64_t page = (uint64_t) vm_kernel_page_size;
    *total = (size_t) memory_size;
    *free = (size_t) ((vm.free_count + vm.inactive_count + vm.speculative_count) * page);
#elif defined(__linux__)
    struct sysinfo status = {};
    if (sysinfo(&status) != 0) {
        *free = 0;
        *total = 0;
        return;
    }
    *total = (size_t) status.totalram * status.mem_unit;
    *free = (size_t) (status.freeram + status.bufferram) * status.mem_unit;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long available_pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;
    *free = available_pages * page_size;
#endif // _WIN32

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        return true;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    for (int i = 0; i < 4; i++) {
        if (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            return buf_extra->supports_op(dev, op);
        }
    }

    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
            return
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            return src1->type == GGML_TYPE_F32 || src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
        case GGML_OP_SOFT_MAX_BACK: {
            if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                return false;
            }
            float max_bias = 0.0f;

            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

            return max_bias == 0.0f;
        }
        case GGML_OP_IM2COL_BACK:
            return src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        case GGML_OP_GET_ROWS_BACK:
            return src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        case GGML_OP_OUT_PROD:
            return (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        default:
            return true;
    }
}

static bool ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

static const char * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    return "CPU";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_cpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_cpu_device_context ctx;
    static ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    return &ggml_backend_cpu_device;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        if (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        if (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        if (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        if (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (ggml_cpu_get_rvv_vlen() > 0) {
            static std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        if (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        if (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static uint64_t ggml_backend_cpu_memory_round_page(uint64_t size) {
#if defined(_WIN32)
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    const uint64_t page = system_info.dwPageSize;
#else
    const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
#endif
    if (page == 0 || size > UINT64_MAX - (page - 1)) {
        return size;
    }
    return ((size + page - 1) / page) * page;
}

static ggml_backend_memory_domain_id_v1 ggml_backend_cpu_memory_domain(uint32_t kind) {
    ggml_backend_memory_domain_id_v1 id = {};
    id.kind = kind;
    return id;
}

static enum ggml_status ggml_backend_cpu_memory_get_domains(
        ggml_backend_dev_t dev, ggml_backend_memory_domain_v1 * domains, uint32_t * inout_count) {
    GGML_UNUSED(dev);
    if (inout_count == NULL) {
        return GGML_STATUS_FAILED;
    }
    const uint32_t capacity = *inout_count;
    *inout_count = 1;
    if (domains == NULL) {
        return GGML_STATUS_SUCCESS;
    }
    if (capacity < 1 || domains[0].struct_size < sizeof(domains[0])) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_memory_domain_v1 domain = {};
    domain.struct_size = sizeof(domain);
    domain.id = ggml_backend_cpu_memory_domain(GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE);
    snprintf(domain.name, sizeof(domain.name), "host/pageable");
    domains[0] = domain;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_cpu_memory_quote(
        const ggml_backend_memory_request_v1 * requests, uint32_t request_count,
        ggml_backend_memory_quote_v1 * quote, ggml_backend_memory_claim_v1 * claims,
        uint32_t * inout_claim_count) {
    if ((request_count > 0 && requests == NULL) || quote == NULL ||
            quote->struct_size < sizeof(*quote) || inout_claim_count == NULL) {
        return GGML_STATUS_FAILED;
    }

    uint32_t individual = 0;
    uint64_t max_workspace = 0;
    ggml_backend_t workspace_backend = NULL;
    ggml_backend_t any_backend = NULL;
    for (uint32_t i = 0; i < request_count; ++i) {
        if (requests[i].struct_size < sizeof(requests[i])) {
            return GGML_STATUS_FAILED;
        }
        if (requests[i].backend != NULL && !ggml_backend_is_cpu(requests[i].backend)) {
            return GGML_STATUS_FAILED;
        }
        if (any_backend != NULL && requests[i].backend != NULL && requests[i].backend != any_backend) {
            return GGML_STATUS_FAILED;
        }
        any_backend = any_backend ? any_backend : requests[i].backend;
        if (requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_BUFFER ||
                requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT) {
            ++individual;
        } else if (requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE) {
            if (requests[i].backend == NULL || requests[i].graph == NULL) {
                return GGML_STATUS_FAILED;
            }
            ggml_backend_cpu_context * ctx = (ggml_backend_cpu_context *) requests[i].backend->context;
            const ggml_cplan plan = ggml_graph_plan(requests[i].graph, ctx->n_threads, ctx->threadpool);
            if (plan.work_size > max_workspace) {
                max_workspace = plan.work_size;
                workspace_backend = requests[i].backend;
            }
        }
    }

    const uint32_t required = individual + (max_workspace > 0 ? 1u : 0u);
    const uint32_t capacity = *inout_claim_count;
    *inout_claim_count = required;

    uint64_t generation = 1;
    if (any_backend != NULL) {
        ggml_backend_cpu_context * ctx = (ggml_backend_cpu_context *) any_backend->context;
        // get_stats() reports the same generation. External RAM pressure is
        // represented by its live free/budget fields, while this generation
        // changes only when this backend mutates retained memory. Binding a
        // token to raw free-page counts would make a quote spuriously stale on
        // every unrelated system allocation.
        generation = ctx->memory_generation;
    }
    quote->flags = 0;
    quote->residual_flags = 0;
    quote->residual_request_count = 0;
    quote->provisional_requested_upper_bytes = 0;
    quote->stats_generation = generation;
    quote->request_fingerprint = ggml_backend_memory_request_fingerprint_v1(requests, request_count);
    quote->quote_token = quote->request_fingerprint ^ generation;

    if (claims == NULL) {
        return GGML_STATUS_SUCCESS;
    }
    if (capacity < required) {
        return GGML_STATUS_FAILED;
    }

    uint32_t out = 0;
    for (uint32_t i = 0; i < request_count; ++i) {
        if (requests[i].kind != GGML_BACKEND_MEMORY_REQUEST_BUFFER &&
                requests[i].kind != GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT) {
            continue;
        }
        const uint64_t requested_committed = ggml_backend_cpu_memory_round_page(requests[i].requested_bytes);
        const uint64_t before = ggml_backend_cpu_memory_round_page(requests[i].currently_allocated_bytes);
        const bool reuse = requests[i].currently_allocated_bytes >= requests[i].requested_bytes;
        const uint64_t after = reuse ? before : requested_committed;
        ggml_backend_memory_claim_v1 claim = {};
        claim.struct_size = sizeof(claim);
        claim.flags = GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER;
        if (requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT) {
            claim.flags |= GGML_BACKEND_MEMORY_CLAIM_FILE_BACKED;
            claim.domain = ggml_backend_cpu_memory_domain(GGML_BACKEND_MEMORY_DOMAIN_FILE_BACKED);
        } else {
            claim.domain = ggml_backend_cpu_memory_domain(GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE);
        }
        claim.request_id = requests[i].request_id;
        claim.payload_requested_bytes = requests[i].requested_bytes;
        claim.committed_before_bytes = before;
        claim.committed_after_upper_bytes = after;
        // Gallocr replacement is transactional: the complete replacement is
        // allocated before the old arena is released.
        claim.commit_peak_extra_upper_bytes = reuse ? 0 : requested_committed;
        claim.resident_after_upper_bytes = after;
        claim.retained_after_use_upper_bytes = after;
        claims[out++] = claim;
    }
    if (max_workspace > 0) {
        ggml_backend_cpu_context * ctx = (ggml_backend_cpu_context *) workspace_backend->context;
        ggml_backend_memory_claim_v1 claim = {};
        claim.struct_size = sizeof(claim);
        claim.flags = GGML_BACKEND_MEMORY_CLAIM_EXACT |
            GGML_BACKEND_MEMORY_CLAIM_REUSABLE_WORKSPACE;
        claim.domain = ggml_backend_cpu_memory_domain(GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE);
        claim.payload_requested_bytes = max_workspace;
        claim.committed_before_bytes = ctx->work_size;
        const uint64_t after = std::max((uint64_t) ctx->work_size, max_workspace);
        claim.committed_after_upper_bytes = after;
        claim.commit_peak_extra_upper_bytes = max_workspace > ctx->work_size ? max_workspace : 0;
        claim.resident_after_upper_bytes = after;
        claim.retained_after_use_upper_bytes = after;
        claim.releasable_after_use_upper_bytes = after;
        claims[out++] = claim;
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_cpu_memory_reserve_private(
        const ggml_backend_memory_request_v1 * requests, uint32_t request_count,
        const ggml_backend_memory_quote_v1 * quote, ggml_backend_memory_claim_v1 * actual,
        uint32_t * inout_actual_count) {
    if ((request_count > 0 && requests == NULL) || quote == NULL ||
            quote->struct_size < sizeof(*quote) || inout_actual_count == NULL ||
            quote->request_fingerprint != ggml_backend_memory_request_fingerprint_v1(requests, request_count)) {
        return GGML_STATUS_FAILED;
    }
    uint64_t max_workspace = 0;
    ggml_backend_t backend = NULL;
    for (uint32_t i = 0; i < request_count; ++i) {
        if (requests[i].struct_size < sizeof(requests[i]) ||
                (requests[i].backend != NULL && !ggml_backend_is_cpu(requests[i].backend))) {
            return GGML_STATUS_FAILED;
        }
        if (backend != NULL && requests[i].backend != NULL && requests[i].backend != backend) {
            return GGML_STATUS_FAILED;
        }
        backend = backend ? backend : requests[i].backend;
        if (requests[i].kind != GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE) {
            continue;
        }
        ggml_backend_cpu_context * ctx = (ggml_backend_cpu_context *) requests[i].backend->context;
        const ggml_cplan plan = ggml_graph_plan(requests[i].graph, ctx->n_threads, ctx->threadpool);
        if (plan.work_size > max_workspace) {
            max_workspace = plan.work_size;
            backend = requests[i].backend;
        }
    }
    const uint32_t required = max_workspace > 0 ? 1u : 0u;
    const uint32_t capacity = *inout_actual_count;
    *inout_actual_count = required;
    if (actual == NULL) {
        return GGML_STATUS_SUCCESS;
    }
    if (capacity < required) {
        return GGML_STATUS_FAILED;
    }
    if (backend == NULL) {
        return GGML_STATUS_SUCCESS;
    }
    ggml_backend_cpu_context * ctx = (ggml_backend_cpu_context *) backend->context;
    if (ctx->memory_health == GGML_BACKEND_MEMORY_QUARANTINED) {
        return GGML_STATUS_BACKEND_POISONED;
    }
    const uint64_t generation = ctx->memory_generation;
    if (quote->stats_generation != generation || quote->quote_token != (quote->request_fingerprint ^ generation)) {
        return GGML_STATUS_FAILED;
    }
    const uint64_t before = ctx->work_size;
    if (ctx->work_size < max_workspace) {
        uint8_t * replacement = new (std::nothrow) uint8_t[max_workspace];
        if (replacement == NULL) {
            ctx->allocation_failures++;
            return GGML_STATUS_ALLOC_FAILED;
        }
        delete[] ctx->work_data;
        ctx->work_data = replacement;
        ctx->work_size = max_workspace;
        ctx->memory_high_water = std::max(ctx->memory_high_water, (uint64_t) ctx->work_size);
        ctx->memory_generation++;
    }
    if (actual != NULL) {
        ggml_backend_memory_claim_v1 claim = {};
        claim.struct_size = sizeof(claim);
        claim.flags = GGML_BACKEND_MEMORY_CLAIM_EXACT |
            GGML_BACKEND_MEMORY_CLAIM_REUSABLE_WORKSPACE;
        claim.domain = ggml_backend_cpu_memory_domain(GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE);
        claim.payload_requested_bytes = max_workspace;
        claim.committed_before_bytes = before;
        claim.committed_after_upper_bytes = ctx->work_size;
        claim.commit_peak_extra_upper_bytes = ctx->work_size > before ? ctx->work_size : 0;
        claim.resident_after_upper_bytes = ctx->work_size;
        claim.retained_after_use_upper_bytes = ctx->work_size;
        claim.releasable_after_use_upper_bytes = ctx->work_size;
        actual[0] = claim;
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_cpu_memory_get_stats(
        ggml_backend_dev_t dev, ggml_backend_t backend,
        ggml_backend_memory_stats_v1 * stats, uint32_t * inout_count) {
    if (dev == NULL || inout_count == NULL) {
        return GGML_STATUS_FAILED;
    }
    const uint32_t capacity = *inout_count;
    *inout_count = 1;
    if (stats == NULL) {
        return GGML_STATUS_SUCCESS;
    }
    if (capacity < 1 || stats[0].struct_size < sizeof(stats[0])) {
        return GGML_STATUS_FAILED;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    ggml_backend_memory_stats_v1 value = {};
    value.struct_size = sizeof(value);
    value.domain = ggml_backend_cpu_memory_domain(GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE);
    value.total_bytes = total_bytes;
    value.budget_bytes = total_bytes;
    value.device_free_bytes = free_bytes;
    value.device_used_bytes = total_bytes >= free_bytes ? total_bytes - free_bytes : 0;
    value.timestamp_monotonic_ns = (uint64_t) ggml_time_us() * 1000;
    value.health = GGML_BACKEND_MEMORY_HEALTHY;
    if (backend != NULL) {
        ggml_backend_cpu_context * ctx = (ggml_backend_cpu_context *) backend->context;
        value.generation = ctx->memory_generation;
        value.backend_owned_workspace_bytes = ctx->work_size;
        value.backend_owned_live_bytes = ctx->work_size;
        value.backend_owned_high_water_bytes = ctx->memory_high_water;
        value.allocation_failure_count = ctx->allocation_failures;
        value.health = ctx->memory_health;
        value.quarantine_generation = ctx->quarantine_generation;
        value.last_native_error = ctx->last_native_error;
    }
    stats[0] = value;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_cpu_memory_trim(ggml_backend_t backend, uint64_t flags) {
    GGML_UNUSED(flags);
    if (backend == NULL) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_cpu_context * ctx = (ggml_backend_cpu_context *) backend->context;
    delete[] ctx->work_data;
    ctx->work_data = NULL;
    ctx->work_size = 0;
    ctx->memory_generation++;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_cpu_memory_quarantine(
        ggml_backend_t backend, const ggml_backend_memory_quarantine_v1 * request) {
    if (backend == NULL || request == NULL || request->struct_size < sizeof(*request)) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_cpu_context * ctx = (ggml_backend_cpu_context *) backend->context;
    ctx->memory_health = GGML_BACKEND_MEMORY_QUARANTINED;
    ctx->last_native_error = request->native_error;
    ctx->quarantine_generation++;
    ctx->memory_generation++;
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_memory_api_v1 * ggml_backend_cpu_memory_get_api_v1(void) {
    static const ggml_backend_memory_api_v1 api = {
        sizeof(ggml_backend_memory_api_v1),
        GGML_BACKEND_MEMORY_ABI_V1,
        0,
        ggml_backend_cpu_memory_get_domains,
        ggml_backend_cpu_memory_quote,
        ggml_backend_cpu_memory_reserve_private,
        ggml_backend_cpu_memory_get_stats,
        ggml_backend_cpu_memory_trim,
        ggml_backend_cpu_memory_quarantine,
    };
    return &api;
}

static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }
    if (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        return (void *)ggml_backend_cpu_set_use_ref;
    }
    if (strcmp(name, GGML_BACKEND_MEMORY_API_V1_PROC) == 0) {
        return (void *) ggml_backend_cpu_memory_get_api_v1;
    }

    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    // init CPU feature detection
    ggml_cpu_init();

    static struct ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
