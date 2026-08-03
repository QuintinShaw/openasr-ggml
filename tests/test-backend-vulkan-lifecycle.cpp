#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static void set_visible_devices(const char * value) {
#ifdef _WIN32
    assert(_putenv_s("GGML_VK_VISIBLE_DEVICES", value != nullptr ? value : "") == 0);
#else
    if (value != nullptr) {
        assert(setenv("GGML_VK_VISIBLE_DEVICES", value, 1) == 0);
    } else {
        assert(unsetenv("GGML_VK_VISIBLE_DEVICES") == 0);
    }
#endif
}

static void test_retry_after_failed_instance_initialization() {
    const char * current = std::getenv("GGML_VK_VISIBLE_DEVICES");
    const std::string saved = current != nullptr ? current : "";
    const bool had_value = current != nullptr;

    set_visible_devices("999999");
    assert(ggml_backend_vk_reg() == nullptr);

    set_visible_devices(had_value ? saved.c_str() : nullptr);
    ggml_backend_reg_t reg = ggml_backend_vk_reg();
    assert(reg != nullptr);
    assert(ggml_backend_reg_dev_count(reg) > 0);
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--retry-after-failure") == 0) {
        test_retry_after_failed_instance_initialization();
        return 0;
    }

    constexpr size_t worker_count = 8;
    constexpr size_t iterations = 8;

    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&ready, &start, iterations]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (size_t iteration = 0; iteration < iterations; ++iteration) {
                ggml_backend_reg_t reg = ggml_backend_vk_reg();
                assert(reg != nullptr);
                assert(ggml_backend_reg_dev_count(reg) > 0);

                ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
                assert(dev != nullptr);

                ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
                assert(buft != nullptr);
                assert(ggml_backend_buft_get_alignment(buft) > 0);

                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                assert(backend != nullptr);
                assert(ggml_backend_is_vk(backend));
                ggml_backend_free(backend);
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != worker_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);

    for (std::thread & worker : workers) {
        worker.join();
    }

    return 0;
}
