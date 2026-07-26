#include "ggml.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstdint>

namespace {

void * align(void * allocation, size_t alignment) {
    return reinterpret_cast<void *>(
            (reinterpret_cast<uintptr_t>(allocation) + alignment - 1) & ~(uintptr_t(alignment) - 1));
}

} // namespace

int main() {
    const size_t alignment = ggml_context_alignment();
    const size_t context_size = ggml_context_size();
    const size_t mem_size = 4096;

    void * context_allocation = std::malloc(context_size + alignment);
    void * mem_allocation = std::malloc(mem_size + alignment);
    assert(context_allocation != nullptr);
    assert(mem_allocation != nullptr);

    void * context_buffer = align(context_allocation, alignment);
    void * mem_buffer = align(mem_allocation, alignment);
    const ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ mem_buffer,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_try_init(params, context_buffer, context_size);
    assert(ctx != nullptr);
    ggml_free(ctx);

    assert(ggml_try_init(params, context_buffer, context_size - 1) == nullptr);

    std::free(mem_allocation);
    std::free(context_allocation);
    return 0;
}
