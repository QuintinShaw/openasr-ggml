#include "ggml.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void * aligned_buffer(void * allocation, size_t alignment) {
    return (void *) (((uintptr_t) allocation + alignment - 1) & ~(uintptr_t) (alignment - 1));
}

static void * fallible_allocation(size_t size, bool fail) {
    return fail ? NULL : malloc(size);
}

static struct ggml_init_params params_for(void * mem_buffer, size_t mem_size) {
    return (struct ggml_init_params) {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ mem_buffer,
        /*.no_alloc   =*/ true,
    };
}

int main(void) {
    const size_t alignment = ggml_context_alignment();
    const size_t context_size = ggml_context_size();
    const size_t mem_size = 4 * 1024;

    GGML_ASSERT(alignment == 64);
    GGML_ASSERT((alignment & (alignment - 1)) == 0);
    GGML_ASSERT(context_size > 0);

    void * context_allocation = malloc(context_size + alignment);
    void * mem_allocation = malloc(mem_size + alignment);
    GGML_ASSERT(context_allocation != NULL);
    GGML_ASSERT(mem_allocation != NULL);

    void * context_buffer = aligned_buffer(context_allocation, alignment);
    void * mem_buffer = aligned_buffer(mem_allocation, alignment);

    // Normal caller-owned initialization and ggml_free() leave both allocations owned by us.
    struct ggml_context * ctx = ggml_try_init(params_for(mem_buffer, mem_size), context_buffer, context_size);
    GGML_ASSERT(ctx != NULL);
    GGML_ASSERT(ggml_get_mem_buffer(ctx) == mem_buffer);
    GGML_ASSERT(ggml_get_mem_size(ctx) == mem_size);
    GGML_ASSERT(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1) != NULL);
    ggml_free(ctx);
    ((unsigned char *) context_buffer)[0] = 0x3c;
    ((unsigned char *) mem_buffer)[0] = 0xc3;

    // Undersized and misaligned caller storage fail without aborting.
    GGML_ASSERT(ggml_try_init(params_for(mem_buffer, mem_size), context_buffer, context_size - 1) == NULL);
    GGML_ASSERT(ggml_try_init(params_for(mem_buffer, mem_size), (char *) context_buffer + 1, context_size) == NULL);
    GGML_ASSERT(ggml_try_init(params_for((char *) mem_buffer + 1, mem_size), context_buffer, context_size) == NULL);
    GGML_ASSERT(ggml_try_init(params_for(NULL, SIZE_MAX), context_buffer, context_size) == NULL);

    // A zero pool size has the same legacy normalization, but still requires caller storage.
    GGML_ASSERT(ggml_try_init(params_for(mem_buffer, 0), context_buffer, context_size) != NULL);
    ggml_free((struct ggml_context *) context_buffer);

    // A caller's fallible 768 MiB allocation can fail before ggml receives the buffer.
    void * failed_allocation = fallible_allocation(768ULL * 1024 * 1024, true);
    GGML_ASSERT(failed_allocation == NULL);
    GGML_ASSERT(ggml_try_init(params_for(failed_allocation, 768ULL * 1024 * 1024), context_buffer, context_size) == NULL);

    // Legacy allocating API remains available, including its existing caller-owned pool behavior.
    ctx = ggml_init(params_for(mem_buffer, mem_size));
    GGML_ASSERT(ctx != NULL);
    ggml_free(ctx);
    ((unsigned char *) mem_buffer)[1] = 0x5a;

    // Legacy allocating API also retains its internally-owned allocation path.
    ctx = ggml_init(params_for(NULL, mem_size));
    GGML_ASSERT(ctx != NULL);
    ggml_free(ctx);

    free(mem_allocation);
    free(context_allocation);
    return 0;
}
