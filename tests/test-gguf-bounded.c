#include "gguf.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

_Static_assert(sizeof(struct gguf_parse_limits) == 5 * sizeof(uint64_t),
               "gguf_parse_limits ABI size changed");
_Static_assert(offsetof(struct gguf_parse_limits, max_tensors) == 0,
               "gguf_parse_limits.max_tensors ABI offset changed");
_Static_assert(offsetof(struct gguf_parse_limits, max_kv) == 8,
               "gguf_parse_limits.max_kv ABI offset changed");
_Static_assert(offsetof(struct gguf_parse_limits, max_string_bytes) == 16,
               "gguf_parse_limits.max_string_bytes ABI offset changed");
_Static_assert(offsetof(struct gguf_parse_limits, max_array_elements) == 24,
               "gguf_parse_limits.max_array_elements ABI offset changed");
_Static_assert(offsetof(struct gguf_parse_limits, max_header_bytes) == 32,
               "gguf_parse_limits.max_header_bytes ABI offset changed");
_Static_assert(sizeof(int32_t) == 4, "parse error out-param must be int32_t");

static void put_u32(uint8_t *dst, uint32_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        dst[i] = (uint8_t)(value >> (8 * i));
    }
}

static void put_u64(uint8_t *dst, uint64_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        dst[i] = (uint8_t)(value >> (8 * i));
    }
}

static size_t put_string(uint8_t *dst, size_t offset, const char *value, size_t length) {
    put_u64(dst + offset, (uint64_t)length);
    offset += sizeof(uint64_t);
    memcpy(dst + offset, value, length);
    return offset + length;
}

static size_t put_empty_header(uint8_t *dst, uint64_t n_tensors, uint64_t n_kv) {
    memcpy(dst, GGUF_MAGIC, 4);
    put_u32(dst + 4, GGUF_VERSION);
    put_u64(dst + 8, n_tensors);
    put_u64(dst + 16, n_kv);
    return 24;
}

static struct gguf_parse_limits generous_limits(void) {
    struct gguf_parse_limits limits = {
        UINT64_MAX,
        UINT64_MAX,
        UINT64_MAX,
        UINT64_MAX,
        UINT64_MAX,
    };
    return limits;
}

static struct gguf_context * parse(const uint8_t *data,
                                   size_t size,
                                   struct gguf_parse_limits limits,
                                   int32_t *error) {
    const struct gguf_init_params params = {
        true,
        NULL,
    };
    return gguf_init_from_buffer_with_limits(data, size, params, limits, error);
}

static void expect_invalid(const uint8_t *data,
                           size_t size,
                           struct gguf_parse_limits limits) {
    int32_t error = INT32_C(-1);
    assert(parse(data, size, limits, &error) == NULL);
    assert(error == (int32_t)GGUF_PARSE_ERROR_INVALID_DATA);
}

static void test_empty_gguf_succeeds(void) {
    uint8_t data[24];
    const size_t size = put_empty_header(data, 0, 0);
    int32_t error = INT32_C(-1);
    struct gguf_context *ctx = parse(data, size, generous_limits(), &error);
    assert(ctx != NULL);
    assert(error == (int32_t)GGUF_PARSE_ERROR_NONE);
    gguf_free(ctx);
}

static void test_tensor_limit(void) {
    uint8_t data[24];
    const size_t size = put_empty_header(data, 1, 0);
    struct gguf_parse_limits limits = generous_limits();
    limits.max_tensors = 0;
    expect_invalid(data, size, limits);
}

static void test_kv_limit(void) {
    uint8_t data[24];
    const size_t size = put_empty_header(data, 0, 1);
    struct gguf_parse_limits limits = generous_limits();
    limits.max_kv = 0;
    expect_invalid(data, size, limits);
}

static void test_string_limit(void) {
    uint8_t data[64];
    size_t size = put_empty_header(data, 0, 1);
    size = put_string(data, size, "k", 1);
    put_u32(data + size, GGUF_TYPE_STRING);
    size += sizeof(uint32_t);
    size = put_string(data, size, "vv", 2);

    struct gguf_parse_limits limits = generous_limits();
    limits.max_string_bytes = 1;
    expect_invalid(data, size, limits);
}

static void test_array_limit(void) {
    uint8_t data[64];
    size_t size = put_empty_header(data, 0, 1);
    size = put_string(data, size, "k", 1);
    put_u32(data + size, GGUF_TYPE_ARRAY);
    size += sizeof(uint32_t);
    put_u32(data + size, GGUF_TYPE_UINT8);
    size += sizeof(uint32_t);
    put_u64(data + size, 2);
    size += sizeof(uint64_t);
    data[size++] = 1;
    data[size++] = 2;

    struct gguf_parse_limits limits = generous_limits();
    limits.max_array_elements = 1;
    expect_invalid(data, size, limits);
}

static void test_header_limit(void) {
    uint8_t data[24];
    const size_t size = put_empty_header(data, 0, 0);
    struct gguf_parse_limits limits = generous_limits();
    limits.max_header_bytes = size - 1;
    expect_invalid(data, size, limits);
}

int main(void) {
    size_t structural_bytes = 0;
    assert(gguf_bounded_parser_structural_bytes(0, 0, &structural_bytes));
    assert(structural_bytes != 0);
    assert(gguf_bounded_parser_payload_wire_multiplier() != 0);

    test_empty_gguf_succeeds();
    test_tensor_limit();
    test_kv_limit();
    test_string_limit();
    test_array_limit();
    test_header_limit();
    return 0;
}
