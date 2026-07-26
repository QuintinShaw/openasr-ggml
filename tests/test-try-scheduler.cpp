#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace {

void * align(void * allocation, size_t alignment) {
    return reinterpret_cast<void *>(
            (reinterpret_cast<uintptr_t>(allocation) + alignment - 1) & ~(uintptr_t(alignment) - 1));
}

} // namespace

int main() {
    constexpr size_t graph_size = 16;
    const size_t alignment = ggml_context_alignment();
    const size_t context_size = ggml_context_size();
    const size_t scheduler_size = ggml_backend_sched_context_buffer_size(graph_size);

    assert(scheduler_size > 0);
    assert(ggml_backend_sched_context_buffer_size(SIZE_MAX) == 0);

    void * context_allocation = std::malloc(context_size + alignment);
    void * scheduler_allocation = std::malloc(scheduler_size + alignment);
    assert(context_allocation != nullptr);
    assert(scheduler_allocation != nullptr);

    void * context_storage = align(context_allocation, alignment);
    void * scheduler_storage = align(scheduler_allocation, alignment);
    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);

    // A caller allocation failure and undersized storage are rejected at
    // scheduler construction, before graph splitting can reach ggml_init.
    assert(ggml_backend_sched_try_new(
        &backend, nullptr, 1, graph_size, false, true,
        nullptr, scheduler_size, context_storage, context_size) == nullptr);
    assert(ggml_backend_sched_try_new(
        &backend, nullptr, 1, graph_size, false, true,
        scheduler_storage, scheduler_size - 1, context_storage, context_size) == nullptr);

    ggml_backend_sched_t scheduler = ggml_backend_sched_try_new(
        &backend, nullptr, 1, graph_size, false, true,
        scheduler_storage, scheduler_size, context_storage, context_size);
    assert(scheduler != nullptr);
    ggml_backend_sched_free(scheduler);

    // The caller still owns both buffers after scheduler destruction.
    static_cast<unsigned char *>(context_storage)[0] = 0x3c;
    static_cast<unsigned char *>(scheduler_storage)[0] = 0xc3;

    ggml_backend_free(backend);
    std::free(scheduler_allocation);
    std::free(context_allocation);
    return 0;
}
