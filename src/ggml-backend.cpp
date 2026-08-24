// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <new>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

static bool ggml_backend_checked_add_size(
        size_t left, size_t right, size_t * out) {
    if (out == NULL || left > SIZE_MAX - right) {
        return false;
    }
    *out = left + right;
    return true;
}

static bool ggml_backend_checked_mul_size(
        size_t left, size_t right, size_t * out) {
    if (out == NULL || (left != 0 && right > SIZE_MAX / left)) {
        return false;
    }
    *out = left * right;
    return true;
}

static uint64_t ggml_backend_memory_hash_u64_v1(uint64_t hash, uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte) {
        hash ^= (uint8_t) (value >> (byte * 8));
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t ggml_backend_memory_request_fingerprint_v1(
        const ggml_backend_memory_request_v1 * requests,
        uint32_t request_count) {
    if (request_count > 0 && requests == NULL) {
        return 0;
    }

    uint64_t hash = UINT64_C(1469598103934665603);
    hash = ggml_backend_memory_hash_u64_v1(hash, GGML_BACKEND_MEMORY_ABI_V1);
    hash = ggml_backend_memory_hash_u64_v1(hash, request_count);
    for (uint32_t index = 0; index < request_count; ++index) {
        const ggml_backend_memory_request_v1 & request = requests[index];
        const uint64_t values[] = {
            index,
            request.kind,
            request.flags,
            request.usage,
            request.request_id,
            (uint64_t) (uintptr_t) request.backend,
            (uint64_t) (uintptr_t) request.peer_backend,
            (uint64_t) (uintptr_t) request.buft,
            (uint64_t) (uintptr_t) request.graph,
            (uint64_t) (uintptr_t) request.host_ptr,
            request.requested_bytes,
            request.currently_allocated_bytes,
            request.max_tensor_bytes,
        };
        for (uint64_t value : values) {
            hash = ggml_backend_memory_hash_u64_v1(hash, value);
        }
    }
    return hash;
}

static int ggml_backend_memory_hex_nibble_v1(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

static bool ggml_backend_memory_parse_hex_byte_v1(
        const char * text,
        size_t digits,
        uint32_t * value) {
    uint32_t parsed = 0;
    for (size_t index = 0; index < digits; ++index) {
        const int nibble = ggml_backend_memory_hex_nibble_v1(text[index]);
        if (nibble < 0) {
            return false;
        }
        parsed = (parsed << 4) | (uint32_t) nibble;
    }
    *value = parsed;
    return true;
}

bool ggml_backend_memory_encode_pci_bdf_v1(
        const char * pci_bus_id,
        uint8_t physical_device_uuid[16]) {
    if (physical_device_uuid == NULL) {
        return false;
    }
    memset(physical_device_uuid, 0, 16);
    if (pci_bus_id == NULL || strlen(pci_bus_id) != 12 ||
            pci_bus_id[4] != ':' || pci_bus_id[7] != ':' || pci_bus_id[10] != '.') {
        return false;
    }

    uint32_t domain = 0;
    uint32_t bus = 0;
    uint32_t device = 0;
    uint32_t function = 0;
    if (!ggml_backend_memory_parse_hex_byte_v1(pci_bus_id, 4, &domain) ||
            !ggml_backend_memory_parse_hex_byte_v1(pci_bus_id + 5, 2, &bus) ||
            !ggml_backend_memory_parse_hex_byte_v1(pci_bus_id + 8, 2, &device) ||
            !ggml_backend_memory_parse_hex_byte_v1(pci_bus_id + 11, 1, &function) ||
            device > 0x1f || function > 0x07) {
        return false;
    }

    physical_device_uuid[0] = 'P';
    physical_device_uuid[1] = 'C';
    physical_device_uuid[2] = 'I';
    physical_device_uuid[3] = 1;
    physical_device_uuid[4] = (uint8_t) (domain >> 8);
    physical_device_uuid[5] = (uint8_t) domain;
    physical_device_uuid[6] = (uint8_t) bus;
    physical_device_uuid[7] = (uint8_t) device;
    physical_device_uuid[8] = (uint8_t) function;
    return true;
}

const struct ggml_backend_memory_api_v1 * ggml_backend_memory_api_for_backend_v1(
        ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return ggml_backend_noexcept_or<const struct ggml_backend_memory_api_v1 *>([&]() {
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        if (device == NULL) {
            return (const struct ggml_backend_memory_api_v1 *) NULL;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if (reg == NULL) {
            return (const struct ggml_backend_memory_api_v1 *) NULL;
        }
        void * proc = ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MEMORY_API_V1_PROC);
        if (proc == NULL) {
            return (const struct ggml_backend_memory_api_v1 *) NULL;
        }
        auto get_api = reinterpret_cast<ggml_backend_memory_get_api_v1_t>(proc);
        return get_api();
    }, NULL);
}

enum ggml_status ggml_backend_memory_api_get_domains_v1(
        const struct ggml_backend_memory_api_v1 * api, ggml_backend_dev_t dev,
        struct ggml_backend_memory_domain_v1 * domains, uint32_t * inout_count) {
    if (api == NULL || api->get_domains == NULL) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_noexcept_status(
        [&]() { return api->get_domains(dev, domains, inout_count); });
}

enum ggml_status ggml_backend_memory_api_quote_v1(
        const struct ggml_backend_memory_api_v1 * api,
        const struct ggml_backend_memory_request_v1 * requests, uint32_t request_count,
        struct ggml_backend_memory_quote_v1 * quote,
        struct ggml_backend_memory_claim_v1 * claims, uint32_t * inout_claim_count) {
    if (api == NULL || api->quote == NULL) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_noexcept_status([&]() {
        return api->quote(requests, request_count, quote, claims, inout_claim_count);
    });
}

enum ggml_status ggml_backend_memory_api_reserve_private_v1(
        const struct ggml_backend_memory_api_v1 * api,
        const struct ggml_backend_memory_request_v1 * requests, uint32_t request_count,
        const struct ggml_backend_memory_quote_v1 * quote,
        struct ggml_backend_memory_claim_v1 * actual, uint32_t * inout_actual_count) {
    if (api == NULL || api->reserve_private == NULL) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_noexcept_status([&]() {
        return api->reserve_private(requests, request_count, quote, actual, inout_actual_count);
    });
}

enum ggml_status ggml_backend_memory_api_get_stats_v1(
        const struct ggml_backend_memory_api_v1 * api, ggml_backend_dev_t dev,
        ggml_backend_t backend, struct ggml_backend_memory_stats_v1 * stats,
        uint32_t * inout_count) {
    if (api == NULL || api->get_stats == NULL) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_noexcept_status(
        [&]() { return api->get_stats(dev, backend, stats, inout_count); });
}

enum ggml_status ggml_backend_memory_api_trim_v1(
        const struct ggml_backend_memory_api_v1 * api, ggml_backend_t backend, uint64_t flags) {
    if (api == NULL || api->trim == NULL) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_noexcept_status(
        [&]() { return api->trim(backend, flags); });
}

enum ggml_status ggml_backend_memory_api_quarantine_v1(
        const struct ggml_backend_memory_api_v1 * api, ggml_backend_t backend,
        const struct ggml_backend_memory_quarantine_v1 * request) {
    if (api == NULL || api->quarantine == NULL) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_noexcept_status(
        [&]() { return api->quarantine(backend, request); });
}


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    if (buft == NULL || buft->iface.get_name == NULL) return "unknown";
    return ggml_backend_noexcept_or<const char *>(
        [&]() { return buft->iface.get_name(buft); }, "unknown");
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (buft == NULL) return NULL;
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    if (buft->iface.alloc_buffer == NULL) return NULL;
    return ggml_backend_noexcept_or<ggml_backend_buffer_t>(
        [&]() { return buft->iface.alloc_buffer(buft, size); }, NULL);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    if (buft == NULL || buft->iface.get_alignment == NULL) return 0;
    return ggml_backend_noexcept_or<size_t>(
        [&]() { return buft->iface.get_alignment(buft); }, 0);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    if (buft == NULL) return 0;
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return ggml_backend_noexcept_or<size_t>(
            [&]() { return buft->iface.get_max_size(buft); }, 0);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    if (buft == NULL || tensor == NULL) return 0;
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = ggml_backend_noexcept_or<size_t>(
            [&]() { return buft->iface.get_alloc_size(buft, tensor); }, 0);
        if (size < ggml_nbytes(tensor)) {
            return 0;
        }
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    if (buft == NULL) return false;
    if (buft->iface.is_host) {
        return ggml_backend_noexcept_or<bool>(
            [&]() { return buft->iface.is_host(buft); }, false);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    return buft == NULL ? NULL : buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    return ggml_backend_noexcept_or<ggml_backend_buffer_t>([&]() {
        return new ggml_backend_buffer {
            /* .interface = */ iface,
            /* .buft      = */ buft,
            /* .context   = */ context,
            /* .size      = */ size,
            /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
        };
    }, NULL);
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    (void) ggml_backend_buffer_free_status(buffer);
}

enum ggml_status ggml_backend_buffer_free_status(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return GGML_STATUS_SUCCESS;
    }

    enum ggml_status status = GGML_STATUS_SUCCESS;
    if (buffer->iface.free_buffer != NULL) {
        status = ggml_backend_noexcept_status([&]() {
            buffer->iface.free_buffer(buffer);
            return GGML_STATUS_SUCCESS;
        });
    }
    const enum ggml_status delete_status = ggml_backend_noexcept_status([&]() {
        delete buffer;
        return GGML_STATUS_SUCCESS;
    });
    return ggml_backend_status_merge(status, delete_status);
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    return buffer == NULL ? 0 : buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) return NULL;
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    return ggml_backend_noexcept_or<void *>(
        [&]() { return buffer->iface.get_base(buffer); }, NULL);
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    if (buffer == NULL || tensor == NULL) return GGML_STATUS_FAILED;
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return ggml_backend_noexcept_status(
            [&]() { return buffer->iface.init_tensor(buffer, tensor); });
    }
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (buffer == NULL) return GGML_STATUS_FAILED;
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return GGML_STATUS_SUCCESS;
    }
    if (buffer->iface.clear == NULL) return GGML_STATUS_FAILED;
    return ggml_backend_noexcept_status([&]() {
        buffer->iface.clear(buffer, value);
        return GGML_STATUS_SUCCESS;
    });
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    if (buffer == NULL) return;
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    return buffer == NULL ? GGML_BACKEND_BUFFER_USAGE_ANY : buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    return buffer == NULL ? NULL : buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    (void) ggml_backend_buffer_reset_status(buffer);
}

enum ggml_status ggml_backend_buffer_reset_status(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return GGML_STATUS_FAILED;
    }
    if (buffer->iface.reset) {
        return ggml_backend_noexcept_status([&]() {
            buffer->iface.reset(buffer);
            return GGML_STATUS_SUCCESS;
        });
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_buffer_copy_tensor_status(
        const struct ggml_tensor * src, struct ggml_tensor * dst, bool * handled) {
    *handled = false;
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        *handled = true;
        return ggml_backend_noexcept_status([&]() {
            if (!dst_buf->iface.cpy_tensor(dst_buf, src, dst)) {
                *handled = false;
            }
            return GGML_STATUS_SUCCESS;
        });
    }
    return GGML_STATUS_SUCCESS;
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    bool handled = false;
    return ggml_backend_buffer_copy_tensor_status(src, dst, &handled) == GGML_STATUS_SUCCESS && handled;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return ggml_backend_noexcept_or<const char *>(
        [&]() { return backend->iface.get_name(backend); }, "unknown");
}

void ggml_backend_free(ggml_backend_t backend) {
    (void) ggml_backend_free_status(backend);
}

enum ggml_status ggml_backend_free_status(ggml_backend_t backend) {
    if (backend == NULL) {
        return GGML_STATUS_SUCCESS;
    }
    if (backend->iface.free == NULL) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_noexcept_status([&]() {
        backend->iface.free(backend);
        return GGML_STATUS_SUCCESS;
    });
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    return backend == NULL ? NULL : ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    return buft == NULL ? NULL : ggml_backend_buft_alloc_buffer(buft, size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    return buft == NULL ? 0 : ggml_backend_buft_get_alignment(buft);
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    return buft == NULL ? 0 : ggml_backend_buft_get_max_size(buft);
}

static enum ggml_status ggml_backend_status_prefer(
        enum ggml_status first, enum ggml_status second) {
    if (first == GGML_STATUS_DEVICE_LOST || second == GGML_STATUS_DEVICE_LOST) return GGML_STATUS_DEVICE_LOST;
    if (first == GGML_STATUS_BACKEND_POISONED || second == GGML_STATUS_BACKEND_POISONED) return GGML_STATUS_BACKEND_POISONED;
    if (first == GGML_STATUS_EXECUTION_FAILED || second == GGML_STATUS_EXECUTION_FAILED) return GGML_STATUS_EXECUTION_FAILED;
    if (first == GGML_STATUS_ABORTED || second == GGML_STATUS_ABORTED) return GGML_STATUS_ABORTED;
    return first != GGML_STATUS_SUCCESS ? first : second;
}

static bool ggml_backend_tensor_transfer_bounds(
        const struct ggml_tensor * tensor, size_t offset, size_t size,
        size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (tensor == NULL) {
        return false;
    }
    if (n_copies == 0 || size == 0) {
        return true;
    }
    size_t copy_offset = 0;
    size_t transfer_end = 0;
    size_t data_offset = 0;
    return ggml_backend_checked_mul_size(
               n_copies - 1, stride_tensor, &copy_offset) &&
        ggml_backend_checked_add_size(offset, copy_offset, &transfer_end) &&
        ggml_backend_checked_add_size(transfer_end, size, &transfer_end) &&
        transfer_end <= ggml_nbytes(tensor) &&
        ggml_backend_checked_mul_size(
               n_copies - 1, stride_data, &data_offset);
}

enum ggml_status ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (backend == NULL || tensor == NULL ||
            (size > 0 && (tensor->data == NULL || data == NULL)) ||
            !ggml_backend_tensor_transfer_bounds(
                tensor, offset, size, 1, 0, 0)) {
        return GGML_STATUS_FAILED;
    }
    if (backend->iface.set_tensor_async != NULL) {
        return ggml_backend_noexcept_status(
            [&]() { return backend->iface.set_tensor_async(backend, tensor, data, offset, size); });
    }
    const enum ggml_status status = ggml_backend_synchronize(backend);
    return status == GGML_STATUS_SUCCESS ? ggml_backend_tensor_set(tensor, data, offset, size) : status;
}

enum ggml_status ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (backend == NULL || tensor == NULL ||
            (size > 0 && (tensor->data == NULL || data == NULL)) ||
            !ggml_backend_tensor_transfer_bounds(
                tensor, offset, size, 1, 0, 0)) {
        return GGML_STATUS_FAILED;
    }
    if (backend->iface.get_tensor_async != NULL) {
        return ggml_backend_noexcept_status(
            [&]() { return backend->iface.get_tensor_async(backend, tensor, data, offset, size); });
    }
    const enum ggml_status status = ggml_backend_synchronize(backend);
    return status == GGML_STATUS_SUCCESS ? ggml_backend_tensor_get(tensor, data, offset, size) : status;
}

enum ggml_status ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (backend == NULL || tensor == NULL ||
            (size > 0 && n_copies > 0 &&
                (tensor->data == NULL || data == NULL)) ||
            !ggml_backend_tensor_transfer_bounds(
                tensor, offset, size, n_copies, stride_tensor, stride_data)) {
        return GGML_STATUS_FAILED;
    }
    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            const enum ggml_status status = ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
            if (status != GGML_STATUS_SUCCESS) return status;
        }
        return GGML_STATUS_SUCCESS;
    }
    if (size == 0) return GGML_STATUS_SUCCESS;
    return ggml_backend_noexcept_status([&]() {
        return backend->iface.set_tensor_2d_async(
            backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    });
}

enum ggml_status ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (backend == NULL || tensor == NULL ||
            (size > 0 && n_copies > 0 &&
                (tensor->data == NULL || data == NULL)) ||
            !ggml_backend_tensor_transfer_bounds(
                tensor, offset, size, n_copies, stride_tensor, stride_data)) {
        return GGML_STATUS_FAILED;
    }
    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            const enum ggml_status status = ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
            if (status != GGML_STATUS_SUCCESS) return status;
        }
        return GGML_STATUS_SUCCESS;
    }
    if (size == 0) return GGML_STATUS_SUCCESS;
    return ggml_backend_noexcept_status([&]() {
        return backend->iface.get_tensor_2d_async(
            backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    });
}

enum ggml_status ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (tensor == NULL || (size > 0 && data == NULL) ||
            !ggml_backend_tensor_transfer_bounds(
                tensor, offset, size, 1, 0, 0)) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buf == NULL || (size > 0 && tensor->data == NULL) ||
            buf->iface.set_tensor == NULL) {
        return GGML_STATUS_FAILED;
    }
    if (size == 0) return GGML_STATUS_SUCCESS;
    return ggml_backend_noexcept_status([&]() {
        buf->iface.set_tensor(buf, tensor, data, offset, size);
        return GGML_STATUS_SUCCESS;
    });
}

enum ggml_status ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (tensor == NULL || (size > 0 && data == NULL) ||
            !ggml_backend_tensor_transfer_bounds(
                tensor, offset, size, 1, 0, 0)) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buf == NULL || (size > 0 && tensor->data == NULL) ||
            buf->iface.get_tensor == NULL) {
        return GGML_STATUS_FAILED;
    }
    if (size == 0) return GGML_STATUS_SUCCESS;
    return ggml_backend_noexcept_status([&]() {
        buf->iface.get_tensor(buf, tensor, data, offset, size);
        return GGML_STATUS_SUCCESS;
    });
}

enum ggml_status ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (tensor == NULL ||
            (size > 0 && n_copies > 0 && data == NULL) ||
            !ggml_backend_tensor_transfer_bounds(
                tensor, offset, size, n_copies, stride_tensor, stride_data)) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buf == NULL || (size > 0 && n_copies > 0 && tensor->data == NULL)) {
        return GGML_STATUS_FAILED;
    }
    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            const enum ggml_status status = ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
            if (status != GGML_STATUS_SUCCESS) return status;
        }
        return GGML_STATUS_SUCCESS;
    }
    if (size == 0) return GGML_STATUS_SUCCESS;
    return ggml_backend_noexcept_status([&]() {
        buf->iface.set_tensor_2d(
            buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
        return GGML_STATUS_SUCCESS;
    });
}

enum ggml_status ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (tensor == NULL ||
            (size > 0 && n_copies > 0 && data == NULL) ||
            !ggml_backend_tensor_transfer_bounds(
                tensor, offset, size, n_copies, stride_tensor, stride_data)) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buf == NULL || (size > 0 && n_copies > 0 && tensor->data == NULL)) {
        return GGML_STATUS_FAILED;
    }
    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            const enum ggml_status status = ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
            if (status != GGML_STATUS_SUCCESS) return status;
        }
        return GGML_STATUS_SUCCESS;
    }
    if (size == 0) return GGML_STATUS_SUCCESS;
    return ggml_backend_noexcept_status([&]() {
        buf->iface.get_tensor_2d(
            buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
        return GGML_STATUS_SUCCESS;
    });
}

enum ggml_status ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (tensor == NULL || !ggml_backend_tensor_transfer_bounds(
            tensor, offset, size, 1, 0, 0)) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return GGML_STATUS_SUCCESS;
    }

    if (buf == NULL || tensor->data == NULL ||
            buf->iface.memset_tensor == NULL) {
        return GGML_STATUS_FAILED;
    }

    return ggml_backend_noexcept_status([&]() {
        buf->iface.memset_tensor(buf, tensor, value, offset, size);
        return GGML_STATUS_SUCCESS;
    });
}

enum ggml_status ggml_backend_synchronize(ggml_backend_t backend) {
    if (backend == NULL) {
        return GGML_STATUS_FAILED;
    }
    if (backend->iface.synchronize == NULL) {
        return GGML_STATUS_SUCCESS;
    }

    return ggml_backend_noexcept_status(
        [&]() { return backend->iface.synchronize(backend); });
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    if (backend == NULL || cgraph == NULL || backend->iface.graph_plan_create == NULL) return NULL;

    return ggml_backend_noexcept_or<ggml_backend_graph_plan_t>(
        [&]() { return backend->iface.graph_plan_create(backend, cgraph); }, NULL);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    if (backend == NULL || backend->iface.graph_plan_free == NULL) return;

    ggml_backend_noexcept_void(
        [&]() { backend->iface.graph_plan_free(backend, plan); });
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    if (backend == NULL || plan == NULL || backend->iface.graph_plan_compute == NULL) {
        return GGML_STATUS_FAILED;
    }

    enum ggml_status submitted = ggml_backend_noexcept_status(
        [&]() { return backend->iface.graph_plan_compute(backend, plan); });
    return ggml_backend_status_merge(submitted, ggml_backend_synchronize(backend));
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status submitted = ggml_backend_graph_compute_async(backend, cgraph);
    return ggml_backend_status_merge(submitted, ggml_backend_synchronize(backend));
}

static ggml_backend_set_abort_callback_t ggml_backend_native_abort_callback(ggml_backend_t backend) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(backend->device);
    if (reg == NULL) {
        return NULL;
    }
    return (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_set_abort_callback");
}

static ggml_backend_set_abort_callback_status_t ggml_backend_native_abort_callback_status(
        ggml_backend_t backend) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(backend->device);
    if (reg == NULL) {
        return NULL;
    }
    return (ggml_backend_set_abort_callback_status_t) ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_set_abort_callback_status");
}

static void ggml_backend_graph_cancel_capability_reset(
        struct ggml_backend_graph_cancel_capability * capability) {
    capability->mechanism = GGML_BACKEND_GRAPH_CANCEL_DISABLED;
    capability->observation_granularity = GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_NONE;
}

static void ggml_backend_graph_cancel_capability_merge(
        struct ggml_backend_graph_cancel_capability * destination,
        const struct ggml_backend_graph_cancel_capability * source) {
    if (source->mechanism == GGML_BACKEND_GRAPH_CANCEL_DISABLED) {
        return;
    }
    if (destination->mechanism == GGML_BACKEND_GRAPH_CANCEL_DISABLED) {
        *destination = *source;
        return;
    }
    if (source->mechanism == GGML_BACKEND_GRAPH_CANCEL_SEGMENTED) {
        destination->mechanism = GGML_BACKEND_GRAPH_CANCEL_SEGMENTED;
    }
    if (source->observation_granularity > destination->observation_granularity) {
        destination->observation_granularity = source->observation_granularity;
    }
}

static enum ggml_status ggml_backend_native_abort_status(
        enum ggml_status status, ggml_abort_callback abort_callback, void * abort_callback_data) {
    if (status != GGML_STATUS_SUCCESS) {
        return status;
    }
    bool requested = false;
    const enum ggml_status callback_status = ggml_backend_noexcept_status([&]() {
        requested = abort_callback(abort_callback_data);
        return GGML_STATUS_SUCCESS;
    });
    if (callback_status != GGML_STATUS_SUCCESS) {
        return callback_status;
    }
    return requested ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_graph_compute_with_abort(
        ggml_backend_t backend, struct ggml_cgraph * cgraph,
        ggml_abort_callback abort_callback, void * abort_callback_data,
        struct ggml_backend_graph_cancel_capability * cancel_capability) {
    if (backend == NULL || cgraph == NULL || cancel_capability == NULL ||
            backend->iface.graph_compute == NULL) {
        return GGML_STATUS_FAILED;
    }

    ggml_backend_graph_cancel_capability_reset(cancel_capability);
    if (abort_callback == NULL) {
        return ggml_backend_graph_compute(backend, cgraph);
    }

    ggml_backend_set_abort_callback_status_t native_status =
        ggml_backend_native_abort_callback_status(backend);
    ggml_backend_set_abort_callback_t native = native_status == NULL
        ? ggml_backend_native_abort_callback(backend) : NULL;

    // A pre-start cancellation still honors the synchronous return contract:
    // no work previously queued on this backend remains in flight.
    const enum ggml_status prestart = ggml_backend_native_abort_status(
        GGML_STATUS_SUCCESS, abort_callback, abort_callback_data);
    if (prestart != GGML_STATUS_SUCCESS) {
        if (prestart != GGML_STATUS_ABORTED) {
            return prestart;
        }
        return ggml_backend_status_merge(GGML_STATUS_ABORTED, ggml_backend_synchronize(backend));
    }

    if (native_status != NULL || native != NULL) {
        const enum ggml_status install_status = native_status != NULL
            ? ggml_backend_noexcept_status([&]() {
                return native_status(
                    backend, abort_callback, abort_callback_data, cancel_capability);
            })
            : ggml_backend_noexcept_status([&]() {
                native(backend, abort_callback, abort_callback_data, cancel_capability);
                return GGML_STATUS_SUCCESS;
            });
        if (install_status != GGML_STATUS_SUCCESS) {
            if (native_status != NULL) {
                (void) ggml_backend_noexcept_status(
                    [&]() { return native_status(backend, NULL, NULL, NULL); });
            } else {
                ggml_backend_noexcept_void(
                    [&]() { native(backend, NULL, NULL, NULL); });
            }
            return install_status;
        }
        enum ggml_status status = ggml_backend_graph_compute(backend, cgraph);
        status = ggml_backend_native_abort_status(status, abort_callback, abort_callback_data);
        const enum ggml_status clear_status = native_status != NULL
            ? ggml_backend_noexcept_status(
                [&]() { return native_status(backend, NULL, NULL, NULL); })
            : ggml_backend_noexcept_status([&]() {
                native(backend, NULL, NULL, NULL);
                return GGML_STATUS_SUCCESS;
            });
        return status != GGML_STATUS_SUCCESS ? status : clear_status;
    }

    // A backend without a native abort hook still receives a real, typed
    // mid-graph cancellation contract. Graph views are already the scheduler's
    // portable unit of partial execution; synchronizing each bounded view makes
    // it safe to stop without leaving queued work that can touch reused buffers.
    // The callback-free path remains exactly the backend's original async path.
    for (int i = 0; i < cgraph->n_nodes; i += GGML_BACKEND_GRAPH_CANCEL_SEGMENT_NODES) {
        const int i_end = std::min(i + GGML_BACKEND_GRAPH_CANCEL_SEGMENT_NODES, cgraph->n_nodes);
        struct ggml_cgraph view = ggml_graph_view(cgraph, i, i_end);
        cancel_capability->mechanism = GGML_BACKEND_GRAPH_CANCEL_SEGMENTED;
        cancel_capability->observation_granularity =
            GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT;
        enum ggml_status submitted = ggml_backend_noexcept_status(
            [&]() { return backend->iface.graph_compute(backend, &view); });
        enum ggml_status completed = ggml_backend_synchronize(backend);
        enum ggml_status status = ggml_backend_status_merge(submitted, completed);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
        const enum ggml_status cancel_status = ggml_backend_native_abort_status(
            GGML_STATUS_SUCCESS, abort_callback, abort_callback_data);
        if (cancel_status != GGML_STATUS_SUCCESS) {
            return cancel_status;
        }
    }

    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    if (backend == NULL || cgraph == NULL || backend->iface.graph_compute == NULL) {
        return GGML_STATUS_FAILED;
    }

    return ggml_backend_noexcept_status(
        [&]() { return backend->iface.graph_compute(backend, cgraph); });
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return backend != NULL && op != NULL &&
        ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    return backend != NULL && buft != NULL &&
        ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return backend != NULL && op != NULL &&
        ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    return backend == NULL ? NULL : backend->device;
}

// backend copy

enum ggml_status ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (src == NULL || dst == NULL || !ggml_are_same_layout(src, dst)) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_buffer_t src_buffer = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t dst_buffer = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (src_buffer == NULL || dst_buffer == NULL) return GGML_STATUS_FAILED;
    if (src == dst) return GGML_STATUS_SUCCESS;
    if (ggml_backend_buffer_is_host(src_buffer)) return ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    if (ggml_backend_buffer_is_host(dst_buffer)) return ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    bool copy_handled = false;
    const enum ggml_status copy_status =
        ggml_backend_buffer_copy_tensor_status(src, dst, &copy_handled);
    if (copy_status != GGML_STATUS_SUCCESS) return copy_status;
    if (copy_handled) return GGML_STATUS_SUCCESS;
#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src_buffer), ggml_backend_buffer_name(dst_buffer));
#endif
    const size_t nbytes = ggml_nbytes(src);
    void * data = malloc(nbytes);
    if (data == NULL) return GGML_STATUS_ALLOC_FAILED;
    const enum ggml_status get_status = ggml_backend_tensor_get(src, data, 0, nbytes);
    const enum ggml_status set_status = get_status == GGML_STATUS_SUCCESS ? ggml_backend_tensor_set(dst, data, 0, nbytes) : GGML_STATUS_SUCCESS;
    free(data);
    return ggml_backend_status_prefer(get_status, set_status);
}

enum ggml_status ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (backend_src == NULL || backend_dst == NULL || src == NULL || dst == NULL ||
            !ggml_are_same_layout(src, dst)) {
        return GGML_STATUS_FAILED;
    }
    if ((src->view_src ? src->view_src->buffer : src->buffer) == NULL ||
            (dst->view_src ? dst->view_src->buffer : dst->buffer) == NULL) {
        return GGML_STATUS_FAILED;
    }
    if (src == dst) return GGML_STATUS_SUCCESS;
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        return ggml_backend_noexcept_status([&]() {
            return backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst);
        });
    }
    const enum ggml_status src_status = ggml_backend_synchronize(backend_src);
    const enum ggml_status dst_status = ggml_backend_synchronize(backend_dst);
    const enum ggml_status status = ggml_backend_status_prefer(src_status, dst_status);
    return status == GGML_STATUS_SUCCESS ? ggml_backend_tensor_copy(src, dst) : status;
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return ggml_backend_noexcept_or<ggml_backend_event_t>(
        [&]() { return device->iface.event_new(device); }, NULL);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    (void) ggml_backend_event_free_status(event);
}

enum ggml_status ggml_backend_event_free_status(ggml_backend_event_t event) {
    if (event == NULL) {
        return GGML_STATUS_SUCCESS;
    }
    if (event->device == NULL || event->device->iface.event_free == NULL) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_noexcept_status([&]() {
        event->device->iface.event_free(event->device, event);
        return GGML_STATUS_SUCCESS;
    });
}

enum ggml_status ggml_backend_event_record_status(ggml_backend_event_t event, ggml_backend_t backend) {
    if (backend == NULL || event == NULL || backend->iface.event_record_status == NULL) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_noexcept_status(
        [&]() { return backend->iface.event_record_status(backend, event); });
}

enum ggml_status ggml_backend_event_synchronize(ggml_backend_event_t event) {
    if (event == NULL || event->device == NULL ||
            event->device->iface.event_synchronize == NULL) return GGML_STATUS_FAILED;

    return ggml_backend_noexcept_status(
        [&]() { return event->device->iface.event_synchronize(event->device, event); });
}

enum ggml_status ggml_backend_event_wait_status(ggml_backend_t backend, ggml_backend_event_t event) {
    if (backend == NULL || event == NULL || backend->iface.event_wait_status == NULL) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_noexcept_status(
        [&]() { return backend->iface.event_wait_status(backend, event); });
}

static enum ggml_status ggml_backend_graph_optimize(
        ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    if (backend == NULL || cgraph == NULL) return GGML_STATUS_FAILED;
    if (backend->iface.graph_optimize != NULL) {
        return ggml_backend_noexcept_status([&]() {
            backend->iface.graph_optimize(backend, cgraph);
            return GGML_STATUS_SUCCESS;
        });
    }
    return GGML_STATUS_SUCCESS;
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    if (device == NULL || device->iface.get_name == NULL) return "unknown";
    return ggml_backend_noexcept_or<const char *>(
        [&]() { return device->iface.get_name(device); }, "unknown");
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    if (device == NULL || device->iface.get_description == NULL) return "unknown";
    return ggml_backend_noexcept_or<const char *>(
        [&]() { return device->iface.get_description(device); }, "unknown");
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    if (free != NULL) {
        *free = 0;
    }
    if (total != NULL) {
        *total = 0;
    }
    if (device == NULL || device->iface.get_memory == NULL) return;
    const bool succeeded = ggml_backend_noexcept_or<bool>([&]() {
        device->iface.get_memory(device, free, total);
        return true;
    }, false);
    if (!succeeded) {
        if (free != NULL) *free = 0;
        if (total != NULL) *total = 0;
    }
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    if (device == NULL || device->iface.get_type == NULL) {
        return GGML_BACKEND_DEVICE_TYPE_UNKNOWN;
    }
    return ggml_backend_noexcept_or<enum ggml_backend_dev_type>(
        [&]() { return device->iface.get_type(device); }, GGML_BACKEND_DEVICE_TYPE_UNKNOWN);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    if (props == NULL) return;
    memset(props, 0, sizeof(*props));
    if (device == NULL || device->iface.get_props == NULL) return;
    const bool succeeded = ggml_backend_noexcept_or<bool>([&]() {
        device->iface.get_props(device, props);
        return true;
    }, false);
    if (!succeeded) {
        memset(props, 0, sizeof(*props));
    }
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    return device == NULL ? NULL : device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    if (device == NULL || device->iface.init_backend == NULL) return NULL;
    return ggml_backend_noexcept_or<ggml_backend_t>(
        [&]() { return device->iface.init_backend(device, params); }, NULL);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    if (device == NULL || device->iface.get_buffer_type == NULL) return NULL;
    return ggml_backend_noexcept_or<ggml_backend_buffer_type_t>(
        [&]() { return device->iface.get_buffer_type(device); }, NULL);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    if (device == NULL || device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return ggml_backend_noexcept_or<ggml_backend_buffer_type_t>(
        [&]() { return device->iface.get_host_buffer_type(device); }, NULL);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    if (device == NULL || device->iface.buffer_from_host_ptr == NULL) {
        return NULL;
    }
    return ggml_backend_noexcept_or<ggml_backend_buffer_t>(
        [&]() { return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size); }, NULL);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    if (device == NULL || op == NULL || device->iface.supports_op == NULL) return false;
    return ggml_backend_noexcept_or<bool>(
        [&]() { return device->iface.supports_op(device, op); }, false);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    if (device == NULL || buft == NULL || device->iface.supports_buft == NULL) return false;
    return ggml_backend_noexcept_or<bool>(
        [&]() { return device->iface.supports_buft(device, buft); }, false);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    if (device == NULL || op == NULL) return false;
    if (device->iface.offload_op != NULL) {
        return ggml_backend_noexcept_or<bool>(
            [&]() { return device->iface.offload_op(device, op); }, false);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    if (reg == NULL || reg->iface.get_name == NULL) return "unknown";
    return ggml_backend_noexcept_or<const char *>(
        [&]() { return reg->iface.get_name(reg); }, "unknown");
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    if (reg == NULL || reg->iface.get_device_count == NULL) return 0;
    return ggml_backend_noexcept_or<size_t>(
        [&]() { return reg->iface.get_device_count(reg); }, 0);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    if (reg == NULL || reg->iface.get_device == NULL) return NULL;
    return ggml_backend_noexcept_or<ggml_backend_dev_t>(
        [&]() { return reg->iface.get_device(reg, index); }, NULL);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (reg == NULL || name == NULL || !reg->iface.get_proc_address) {
        return NULL;
    }
    return ggml_backend_noexcept_or<void *>(
        [&]() { return reg->iface.get_proc_address(reg, name); }, NULL);
}

enum ggml_status ggml_backend_set_n_threads_if_supported(
        ggml_backend_t backend, int n_threads) {
    if (backend == NULL || n_threads <= 0) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_noexcept_status([&]() {
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = device == NULL ? NULL : ggml_backend_dev_backend_reg(device);
        if (reg == NULL) {
            return GGML_STATUS_SUCCESS;
        }
        ggml_backend_set_n_threads_t set_n_threads =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_set_n_threads");
        if (set_n_threads != NULL) {
            set_n_threads(backend, n_threads);
        }
        return GGML_STATUS_SUCCESS;
    });
}

uint32_t ggml_backend_dev_pci_vendor_id(ggml_backend_dev_t device) {
    if (device == NULL) {
        return 0;
    }
    return ggml_backend_noexcept_or<uint32_t>([&]() {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if (reg == NULL) {
            return 0u;
        }
        ggml_backend_device_pci_vendor_id_t query =
            (ggml_backend_device_pci_vendor_id_t) ggml_backend_reg_get_proc_address(
                reg, GGML_BACKEND_DEVICE_PCI_VENDOR_ID_PROC);
        return query == NULL ? 0u : query(device);
    }, 0);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer == NULL || buffer->context == NULL) {
        throw ggml_backend_exception { GGML_STATUS_FAILED, 0 };
    }
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    enum ggml_status status = GGML_STATUS_SUCCESS;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        status = ggml_backend_status_merge(
            status, ggml_backend_buffer_free_status(ctx->buffers[i]));
    }

    free(ctx->buffers);
    free(ctx);
    if (status != GGML_STATUS_SUCCESS) {
        throw ggml_backend_exception { status, 0 };
    }
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (buffer == NULL || buffer->context == NULL) {
        throw ggml_backend_exception { GGML_STATUS_FAILED, 0 };
    }
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        const enum ggml_status status = ggml_backend_buffer_clear(ctx->buffers[i], value);
        if (status != GGML_STATUS_SUCCESS) {
            throw ggml_backend_exception { status, 0 };
        }
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    if (buffers == NULL || n_buffers == 0) {
        return NULL;
    }
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    if (ctx == NULL) {
        return NULL;
    }
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));
    if (ctx->buffers == NULL) {
        free(ctx);
        return NULL;
    }

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        const size_t buffer_size = ggml_backend_buffer_get_size(buffers[i]);
        if (buffer_size > SIZE_MAX - total_size) {
            free(ctx->buffers);
            free(ctx);
            return NULL;
        }
        total_size += buffer_size;
    }

    ggml_backend_buffer_t result = ggml_backend_buffer_init(
        buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
    if (result == NULL) {
        free(ctx->buffers);
        free(ctx);
    }
    return result;
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    return buffer != NULL &&
        buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    if (!ggml_backend_buffer_is_multi_buffer(buffer)) return;
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched_src_rewrite {
    struct ggml_tensor ** slot;
    struct ggml_tensor * original;
};

struct ggml_backend_sched_memory_plan {
    ggml_backend_sched_t sched;
    struct ggml_cgraph * source_graph;
    std::vector<ggml_backend_memory_request_v1> items;
    uint64_t fingerprint;
    uint64_t source_graph_fingerprint;
    int previous_cur_copy;
    bool committed;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;
    bool memory_plan_active;
    ggml_backend_sched_memory_plan * active_memory_plan;
    // Source graph whose tensor bindings currently refer to `galloc` buffers.
    // It is detached before the scheduler can replace those buffers.
    struct ggml_cgraph * allocated_graph;

    // Graph splitting may replace a source edge with a scheduler-context copy.
    // Retained cgraphs need those edges restored before `sched->ctx` is reset.
    ggml_backend_sched_src_rewrite * src_rewrites;
    size_t n_src_rewrites;
    size_t src_rewrites_capacity;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

static size_t ggml_backend_sched_hash_id(
        ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    const size_t id = ggml_hash_find_or_insert(&sched->hash_set, tensor);
    if (id == GGML_HASHSET_FULL) {
        throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
    }
    return id;
}

#define hash_id(tensor) ggml_backend_sched_hash_id(sched, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_record_src_rewrite(
        ggml_backend_sched_t sched,
        struct ggml_tensor ** slot) {
    if (sched->n_src_rewrites == sched->src_rewrites_capacity) {
        size_t next_capacity = 16;
        if (sched->src_rewrites_capacity != 0 &&
                !ggml_backend_checked_mul_size(
                    sched->src_rewrites_capacity, 2, &next_capacity)) {
            throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
        }
        size_t allocation_size = 0;
        if (!ggml_backend_checked_mul_size(
                next_capacity, sizeof(sched->src_rewrites[0]),
                &allocation_size)) {
            throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
        }
        void * grown = realloc(
                sched->src_rewrites,
                allocation_size);
        if (grown == NULL) {
            throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
        }
        sched->src_rewrites = (ggml_backend_sched_src_rewrite *) grown;
        sched->src_rewrites_capacity = next_capacity;
    }
    sched->src_rewrites[sched->n_src_rewrites++] = {slot, *slot};
}

static void ggml_backend_sched_restore_src_rewrites(ggml_backend_sched_t sched) {
    while (sched->n_src_rewrites > 0) {
        const ggml_backend_sched_src_rewrite rewrite =
                sched->src_rewrites[--sched->n_src_rewrites];
        *rewrite.slot = rewrite.original;
    }
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        throw ggml_backend_exception { GGML_STATUS_FAILED, 0 };
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
static void ggml_backend_sched_split_graph_impl(
        ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    if (sched == NULL || graph == NULL || graph->n_nodes < 0 ||
            graph->n_leafs < 0) {
        throw ggml_backend_exception { GGML_STATUS_FAILED, 0 };
    }
    size_t graph_identity_count = 0;
    if (!ggml_backend_checked_add_size(
            (size_t) graph->n_nodes, (size_t) graph->n_leafs,
            &graph_identity_count) || graph_identity_count > sched->hash_set.size) {
        throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
    }
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_try_init(params);
    if (sched->ctx == NULL) {
        throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        if (*cur_backend_id == -1) {
            throw ggml_backend_exception { GGML_STATUS_FAILED, 0 };
        }
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            if (node_backend_id == -1) {
                throw ggml_backend_exception { GGML_STATUS_FAILED, 0 };
            }

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    if (sched->splits_capacity <= 0 ||
                            sched->splits_capacity > INT_MAX / 2) {
                        throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
                    }
                    const int next_capacity = sched->splits_capacity * 2;
                    size_t allocation_size = 0;
                    if (!ggml_backend_checked_mul_size(
                            (size_t) next_capacity, sizeof(sched->splits[0]),
                            &allocation_size)) {
                        throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
                    }
                    void * grown = realloc(sched->splits, allocation_size);
                    if (grown == NULL) {
                        throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
                    }
                    sched->splits = (ggml_backend_sched_split *) grown;
                    sched->splits_capacity = next_capacity;
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                if (src_backend_id == -1) {
                    throw ggml_backend_exception { GGML_STATUS_FAILED, 0 };
                }

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        if (sched->n_graph_inputs >= GGML_SCHED_MAX_SPLIT_INPUTS) {
                            throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        if (split->n_inputs >= GGML_SCHED_MAX_SPLIT_INPUTS) {
                            throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
                        }
                        int n_inputs = split->n_inputs++;
                        split->inputs[n_inputs] = src;
                    }
                    ggml_backend_sched_record_src_rewrite(sched, &node->src[j]);
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    size_t split_graph_size = 0;
    size_t graph_size_value = (size_t) std::max(graph->n_nodes, graph->n_leafs);
    if (!ggml_backend_checked_mul_size(
            (size_t) sched->n_splits,
            (size_t) GGML_SCHED_MAX_SPLIT_INPUTS * 2,
            &split_graph_size) ||
            !ggml_backend_checked_mul_size(
                split_graph_size, (size_t) sched->n_copies,
                &split_graph_size) ||
            !ggml_backend_checked_add_size(
                graph_size_value, split_graph_size, &graph_size_value) ||
            graph_size_value > INT_MAX) {
        throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
    }
    const int graph_size = (int) graph_size_value;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        size_t allocation_size = 0;
        if (!ggml_backend_checked_mul_size(
                graph_size_value, sizeof(struct ggml_tensor *),
                &allocation_size)) {
            throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
        }
        ggml_tensor ** nodes = (ggml_tensor **) malloc(allocation_size);
        ggml_tensor ** leafs = (ggml_tensor **) malloc(allocation_size);
        if (nodes == NULL || leafs == NULL) {
            free(nodes);
            free(leafs);
            throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
        }
        free(sched->graph.nodes);
        free(sched->graph.leafs);
        sched->graph.nodes = nodes;
        sched->graph.leafs = leafs;
        sched->graph.size = graph_size;
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        const enum ggml_status optimize_status =
            ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);
        if (optimize_status != GGML_STATUS_SUCCESS) {
            throw ggml_backend_exception { optimize_status, 0 };
        }

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            if (graph_copy->n_nodes < 0 ||
                    graph_copy->n_nodes > graph_copy->size - 2) {
                throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
            }

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->prev_node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->prev_node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            if (graph_copy->n_nodes < 0 ||
                    graph_copy->n_nodes >= graph_copy->size) {
                throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
            }
            sched->prev_node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->prev_leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                if (graph_copy->n_leafs < 0 ||
                        graph_copy->n_leafs >= graph_copy->size) {
                    throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
                }
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->prev_leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    if (graph_copy->n_leafs < 0 ||
                            graph_copy->n_leafs >= graph_copy->size) {
                        throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
                    }
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->prev_leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        if (graph_copy->n_leafs < 0 ||
                graph_copy->n_leafs >= graph_copy->size) {
            throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 0 };
        }
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // Publish the completed assignment only after every fallible split step
    // succeeds. Until this swap, the prior assignment remains authoritative.
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

enum ggml_status ggml_backend_sched_split_graph_v2(
        ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    const enum ggml_status status = ggml_backend_noexcept_status([&]() {
        ggml_backend_sched_split_graph_impl(sched, graph);
        return GGML_STATUS_SUCCESS;
    });
    if (status != GGML_STATUS_SUCCESS && sched != NULL) {
        ggml_backend_noexcept_void([&]() { ggml_backend_sched_reset(sched); });
    }
    return status;
}

void ggml_backend_sched_split_graph(
        ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    const enum ggml_status status =
        ggml_backend_sched_split_graph_v2(sched, graph);
    if (status != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: graph split failed with status %d\n", __func__, status);
    }
}

static enum ggml_status ggml_backend_sched_alloc_splits(
        ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    enum ggml_status allocation_status = backend_ids_changed
        ? GGML_STATUS_ALLOC_FAILED
        : ggml_gallocr_alloc_graph_v2(sched->galloc, &sched->graph);
    if (allocation_status != GGML_STATUS_SUCCESS &&
            allocation_status != GGML_STATUS_ALLOC_FAILED) {
        return allocation_status;
    }

    // allocate graph
    if (backend_ids_changed || allocation_status == GGML_STATUS_ALLOC_FAILED) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_LOG_ERROR("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        enum ggml_status synchronize_status = GGML_STATUS_SUCCESS;
        for (int i = 0; i < sched->n_backends; i++) {
            synchronize_status = ggml_backend_status_merge(
                synchronize_status,
                ggml_backend_synchronize(sched->backends[i]));
        }
        if (synchronize_status != GGML_STATUS_SUCCESS) {
            return synchronize_status;
        }

        if (!ggml_gallocr_reserve_n(
                sched->galloc, &sched->graph, sched->node_backend_ids,
                sched->leaf_backend_ids)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
        allocation_status =
            ggml_gallocr_alloc_graph_v2(sched->galloc, &sched->graph);
        if (allocation_status != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: failed to allocate graph (status %d)\n",
                __func__, allocation_status);
            return allocation_status;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_sched_compute_splits(
        ggml_backend_sched_t sched, ggml_abort_callback abort_callback, void * abort_callback_data,
        bool native_cancel, struct ggml_backend_graph_cancel_capability * cancel_capability) {
    if (sched == NULL) {
        return GGML_STATUS_FAILED;
    }
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    const auto abort_requested = [&]() {
        return abort_callback != NULL && abort_callback(abort_callback_data);
    };

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        // A scheduler split may first wait for or copy inputs owned by another
        // backend. Poll at every scheduler-controlled boundary so cancellation
        // cannot silently traverse the rest of the transfer phase. An already
        // entered backend wait/copy remains indivisible; the public synchronous
        // wrapper drains every backend before returning ABORTED.
        if (abort_requested()) {
            return GGML_STATUS_ABORTED;
        }

        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (abort_requested()) {
                return GGML_STATUS_ABORTED;
            }

            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                enum ggml_status boundary_status;
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    boundary_status = ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    boundary_status = ggml_backend_synchronize(split_backend);
                }
                if (boundary_status != GGML_STATUS_SUCCESS) {
                    return boundary_status;
                }
                if (abort_requested()) {
                    return GGML_STATUS_ABORTED;
                }
                const enum ggml_status copy_status = ggml_backend_tensor_copy(input, input_cpy);
                if (copy_status != GGML_STATUS_SUCCESS) {
                    return copy_status;
                }
            } else {
                // wait for the split backend to finish using the input before overwriting it
                enum ggml_status boundary_status;
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    boundary_status = ggml_backend_event_wait_status(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    boundary_status = ggml_backend_synchronize(split_backend);
                }
                if (boundary_status != GGML_STATUS_SUCCESS) {
                    return boundary_status;
                }
                if (abort_requested()) {
                    return GGML_STATUS_ABORTED;
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    enum ggml_status input_status = ggml_backend_synchronize(input_backend);
                    if (input_status != GGML_STATUS_SUCCESS) {
                        return input_status;
                    }
                    if (abort_requested()) {
                        return GGML_STATUS_ABORTED;
                    }

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        enum ggml_status ids_status = ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        if (ids_status == GGML_STATUS_SUCCESS) ids_status = ggml_backend_synchronize(ids_backend);
                        if (ids_status != GGML_STATUS_SUCCESS) {
                            return ids_status;
                        }
                        if (abort_requested()) {
                            return GGML_STATUS_ABORTED;
                        }

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                if (id < 0 || id >= n_expert) {
                                    return GGML_STATUS_EXECUTION_FAILED;
                                }
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) -> enum ggml_status {
                        if (abort_requested()) {
                            return GGML_STATUS_ABORTED;
                        }
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        const enum ggml_status copy_status = ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                        if (copy_status != GGML_STATUS_SUCCESS) {
                            return copy_status;
                        }
                        return abort_requested() ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS;
                    };

                    int id = 0;
                    while (id < n_expert && !ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    if (id >= n_expert) {
                        return GGML_STATUS_EXECUTION_FAILED;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        const enum ggml_status copy_status = copy_experts(first_id, last_id);
                        if (copy_status != GGML_STATUS_SUCCESS) {
                            return copy_status;
                        }

                        first_id = id;
                        last_id = id;
                    }
                    const enum ggml_status copy_status = copy_experts(first_id, last_id);
                    if (copy_status != GGML_STATUS_SUCCESS) {
                        return copy_status;
                    }
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    const enum ggml_status async_copy_status = split_backend->iface.cpy_tensor_async != NULL
                        ? ggml_backend_noexcept_status([&]() {
                            return split_backend->iface.cpy_tensor_async(
                                input_backend, split_backend, input, input_cpy);
                        })
                        : GGML_STATUS_FAILED;
                    if (async_copy_status != GGML_STATUS_SUCCESS && async_copy_status != GGML_STATUS_FAILED) {
                        return async_copy_status;
                    }
                    if (async_copy_status == GGML_STATUS_FAILED) {
                        enum ggml_status copy_source_status = ggml_backend_synchronize(input_backend);
                        if (copy_source_status != GGML_STATUS_SUCCESS) {
                            return copy_source_status;
                        }
                        if (abort_requested()) {
                            return GGML_STATUS_ABORTED;
                        }
                        enum ggml_status copy_destination_status;
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            copy_destination_status = ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            copy_destination_status = ggml_backend_synchronize(split_backend);
                        }
                        if (copy_destination_status != GGML_STATUS_SUCCESS) {
                            return copy_destination_status;
                        }
                        if (abort_requested()) {
                            return GGML_STATUS_ABORTED;
                        }
                        const enum ggml_status copy_status = ggml_backend_tensor_copy(input, input_cpy);
                        if (copy_status != GGML_STATUS_SUCCESS) return copy_status;
                    }
                }
            }

            if (abort_requested()) {
                return GGML_STATUS_ABORTED;
            }
        }

        if (abort_requested()) {
            return GGML_STATUS_ABORTED;
        }

        if (!sched->callback_eval) {
            enum ggml_status ec;
            if (abort_callback == NULL || native_cancel) {
                ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            } else {
                struct ggml_backend_graph_cancel_capability split_capability;
                ec = ggml_backend_graph_compute_with_abort(
                    split_backend, &split->graph, abort_callback, abort_callback_data, &split_capability);
                ggml_backend_graph_cancel_capability_merge(cancel_capability, &split_capability);
            }
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec;
                if (abort_callback == NULL || native_cancel) {
                    ec = ggml_backend_graph_compute_async(split_backend, &gv);
                } else {
                    struct ggml_backend_graph_cancel_capability split_capability;
                    ec = ggml_backend_graph_compute_with_abort(
                        split_backend, &gv, abort_callback, abort_callback_data, &split_capability);
                    ggml_backend_graph_cancel_capability_merge(cancel_capability, &split_capability);
                }
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // Callback consumers observe materialized tensor data, so this
                // completion is a terminal boundary rather than a best-effort wait.
                enum ggml_status completed = ggml_backend_synchronize(split_backend);
                if (completed != GGML_STATUS_SUCCESS) {
                    return completed;
                }

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                enum ggml_status event_status = ggml_backend_event_record_status(
                    sched->events[split_backend_id][sched->cur_copy], split_backend);
                if (event_status != GGML_STATUS_SUCCESS) {
                    return event_status;
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    if (backends == NULL || n_backends <= 0 ||
            n_backends > GGML_SCHED_MAX_BACKENDS || graph_size == 0 ||
            graph_size > INT_MAX) {
        return NULL;
    }
    for (int b = 0; b < n_backends; ++b) {
        if (backends[b] == NULL) {
            return NULL;
        }
    }
    ggml_backend_dev_t cpu_device =
        ggml_backend_get_device(backends[n_backends - 1]);
    if (cpu_device == NULL ||
            ggml_backend_dev_type(cpu_device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        return NULL;
    }

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));
    if (sched == NULL) {
        return NULL;
    }

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    if (!ggml_hash_set_try_new(graph_size, &sched->hash_set)) {
        ggml_backend_sched_free(sched);
        return NULL;
    }
    size_t backend_ids_bytes = 0;
    size_t tensor_copies_count = 0;
    size_t tensor_copies_bytes = 0;
    if (!ggml_backend_checked_mul_size(
            sched->hash_set.size, sizeof(sched->hv_tensor_backend_ids[0]),
            &backend_ids_bytes) ||
            !ggml_backend_checked_mul_size(
                sched->hash_set.size, (size_t) sched->n_backends,
                &tensor_copies_count) ||
            !ggml_backend_checked_mul_size(
                tensor_copies_count, (size_t) sched->n_copies,
                &tensor_copies_count) ||
            !ggml_backend_checked_mul_size(
                tensor_copies_count, sizeof(struct ggml_tensor *),
                &tensor_copies_bytes)) {
        ggml_backend_sched_free(sched);
        return NULL;
    }
    sched->hv_tensor_backend_ids = (int *) malloc(backend_ids_bytes);
    sched->hv_tensor_copies = (ggml_tensor **) malloc(tensor_copies_bytes);

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    size_t split_tensor_count = 0;
    size_t global_input_count = 0;
    size_t nodes_size = 0;
    const size_t graph_copy_factor =
        (size_t) std::max(2, sched->n_copies);
    if (!ggml_backend_checked_mul_size(
            ggml_sched_max_splits,
            (size_t) GGML_SCHED_MAX_SPLIT_INPUTS * graph_copy_factor,
            &split_tensor_count) ||
            !ggml_backend_checked_mul_size(
                (size_t) GGML_SCHED_MAX_SPLIT_INPUTS,
                (size_t) sched->n_copies, &global_input_count) ||
            !ggml_backend_checked_add_size(
                graph_size, split_tensor_count, &nodes_size) ||
            !ggml_backend_checked_add_size(
                nodes_size, global_input_count, &nodes_size) ||
            nodes_size > SIZE_MAX / sizeof(sched->node_backend_ids[0])) {
        ggml_backend_sched_free(sched);
        return NULL;
    }
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    size_t context_tensor_count = 0;
    size_t context_tensor_bytes = 0;
    size_t graph_overhead = 0;
    if (!ggml_backend_checked_mul_size(
            ggml_sched_max_splits,
            (size_t) GGML_SCHED_MAX_SPLIT_INPUTS *
                ((size_t) sched->n_copies + 1),
            &context_tensor_count) ||
            !ggml_backend_checked_add_size(
                context_tensor_count, global_input_count,
                &context_tensor_count) ||
            !ggml_backend_checked_mul_size(
                context_tensor_count, ggml_tensor_overhead(),
                &context_tensor_bytes) ||
            !ggml_graph_overhead_custom_try(
                graph_size, false, &graph_overhead) ||
            !ggml_backend_checked_add_size(
                context_tensor_bytes, graph_overhead,
                &sched->context_buffer_size)) {
        ggml_backend_sched_free(sched);
        return NULL;
    }
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    if (sched->hv_tensor_backend_ids == NULL ||
            sched->hv_tensor_copies == NULL ||
            sched->node_backend_ids == NULL ||
            sched->leaf_backend_ids == NULL ||
            sched->prev_node_backend_ids == NULL ||
            sched->prev_leaf_backend_ids == NULL ||
            sched->context_buffer == NULL || sched->splits == NULL) {
        ggml_backend_sched_free(sched);
        return NULL;
    }

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        if (sched->bufts[b] == NULL ||
                !ggml_backend_supports_buft(backends[b], sched->bufts[b])) {
            ggml_backend_sched_free(sched);
            return NULL;
        }

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    if (sched->galloc == NULL) {
        ggml_backend_sched_free(sched);
        return NULL;
    }
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    (void) ggml_backend_sched_free_status(sched);
}

enum ggml_status ggml_backend_sched_free_status(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return GGML_STATUS_SUCCESS;
    }
    enum ggml_status status = GGML_STATUS_SUCCESS;
    if (sched->active_memory_plan != NULL) {
        // The plan may outlive an accidentally-early scheduler free. Detach it
        // so later commit/free is a clean failure rather than a use-after-free.
        sched->active_memory_plan->sched = NULL;
        sched->active_memory_plan = NULL;
        sched->memory_plan_active = false;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            status = ggml_backend_status_merge(
                status, ggml_backend_event_free_status(sched->events[b][c]));
        }
    }
    status = ggml_backend_status_merge(
        status, ggml_gallocr_free_status(sched->galloc));
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->splits);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->src_rewrites);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
    return status;
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    if (sched->memory_plan_active) {
        GGML_LOG_ERROR("%s: scheduler is owned by an uncommitted memory plan\n", __func__);
        return;
    }
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_backend_sched_restore_src_rewrites(sched);
        if (sched->allocated_graph != NULL) {
            ggml_gallocr_detach_graph_tensors_v1(sched->galloc, sched->allocated_graph);
            sched->allocated_graph = NULL;
        }
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    if (sched == NULL || measure_graph == NULL || sizes == NULL ||
            measure_graph->n_nodes < 0 || measure_graph->n_leafs < 0) {
        return;
    }
    size_t graph_identity_count = 0;
    if (!ggml_backend_checked_add_size(
            (size_t) measure_graph->n_nodes,
            (size_t) measure_graph->n_leafs, &graph_identity_count) ||
            graph_identity_count > sched->hash_set.size) {
        memset(sizes, 0, (size_t) sched->n_backends * sizeof(*sizes));
        return;
    }

    ggml_backend_sched_reset(sched);

    if (ggml_backend_sched_synchronize(sched) != GGML_STATUS_SUCCESS) {
        memset(sizes, 0, (size_t) sched->n_backends * sizeof(*sizes));
        return;
    }

    if (ggml_backend_sched_split_graph_v2(sched, measure_graph) !=
            GGML_STATUS_SUCCESS) {
        memset(sizes, 0, (size_t) sched->n_backends * sizeof(*sizes));
        ggml_backend_sched_reset(sched);
        return;
    }

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    if (sched == NULL || measure_graph == NULL ||
            measure_graph->n_nodes < 0 || measure_graph->n_leafs < 0) {
        return false;
    }
    size_t graph_identity_count = 0;
    if (!ggml_backend_checked_add_size(
            (size_t) measure_graph->n_nodes,
            (size_t) measure_graph->n_leafs, &graph_identity_count) ||
            graph_identity_count > sched->hash_set.size) {
        return false;
    }

    if (ggml_backend_sched_synchronize(sched) != GGML_STATUS_SUCCESS) {
        return false;
    }

    if (ggml_backend_sched_split_graph_v2(sched, measure_graph) !=
            GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(sched);
        return false;
    }

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

static uint64_t ggml_backend_sched_memory_hash_u64(uint64_t hash, uint64_t value) {
    // FNV-1a is not a security boundary; it is a stable audit guard against a
    // quote being committed for a different scheduler plan.
    for (unsigned i = 0; i < 8; ++i) {
        hash ^= (uint8_t) (value >> (i * 8));
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t ggml_backend_sched_memory_graph_fingerprint(const struct ggml_cgraph * graph) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = ggml_backend_sched_memory_hash_u64(hash, (uint64_t) graph->n_nodes);
    hash = ggml_backend_sched_memory_hash_u64(hash, (uint64_t) graph->n_leafs);
    const auto hash_tensor = [&hash](const struct ggml_tensor * tensor) {
        hash = ggml_backend_sched_memory_hash_u64(hash, (uintptr_t) tensor);
        hash = ggml_backend_sched_memory_hash_u64(hash, (uint64_t) tensor->type);
        hash = ggml_backend_sched_memory_hash_u64(hash, (uint64_t) tensor->op);
        hash = ggml_backend_sched_memory_hash_u64(hash, (uintptr_t) tensor->data);
        hash = ggml_backend_sched_memory_hash_u64(hash, (uintptr_t) tensor->view_src);
        hash = ggml_backend_sched_memory_hash_u64(hash, (uint64_t) tensor->view_offs);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            hash = ggml_backend_sched_memory_hash_u64(hash, (uint64_t) tensor->ne[d]);
            hash = ggml_backend_sched_memory_hash_u64(hash, (uint64_t) tensor->nb[d]);
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            hash = ggml_backend_sched_memory_hash_u64(hash, (uintptr_t) tensor->src[s]);
        }
    };
    for (int i = 0; i < graph->n_leafs; ++i) hash_tensor(graph->leafs[i]);
    for (int i = 0; i < graph->n_nodes; ++i) hash_tensor(graph->nodes[i]);
    return hash;
}

static void ggml_backend_sched_memory_plan_rollback(
        ggml_backend_sched_memory_plan_t plan) noexcept {
    if (plan == NULL) {
        return;
    }
    if (plan->sched != NULL) {
        plan->sched->cur_copy = plan->previous_cur_copy;
        plan->sched->memory_plan_active = false;
        plan->sched->active_memory_plan = NULL;
        ggml_backend_noexcept_void([&]() { ggml_backend_sched_reset(plan->sched); });
    }
    delete plan;
}

static enum ggml_status ggml_backend_sched_memory_plan_create_v1_impl(
        ggml_backend_sched_t sched, struct ggml_cgraph * graph,
        ggml_backend_sched_memory_plan_t * out_plan) {
    if (sched == NULL || graph == NULL || out_plan == NULL ||
            graph->n_nodes < 0 || graph->n_leafs < 0 ||
            sched->memory_plan_active) {
        return GGML_STATUS_FAILED;
    }
    *out_plan = NULL;

    // Detach the previously allocated graph before every validation exit. In
    // particular, an oversized sibling must not leave the prior cgraph bound
    // after its caller has conservatively invalidated that binding.
    ggml_backend_sched_reset(sched);
    size_t graph_identity_count = 0;
    if (!ggml_backend_checked_add_size(
            (size_t) graph->n_nodes, (size_t) graph->n_leafs,
            &graph_identity_count) || graph_identity_count > sched->hash_set.size) {
        return GGML_STATUS_ALLOC_FAILED;
    }
    const enum ggml_status synchronized = ggml_backend_sched_synchronize(sched);
    if (synchronized != GGML_STATUS_SUCCESS) {
        return synchronized;
    }

    ggml_backend_sched_memory_plan * plan = new (std::nothrow) ggml_backend_sched_memory_plan();
    if (plan == NULL) {
        return GGML_STATUS_ALLOC_FAILED;
    }
    plan->sched = sched;
    plan->source_graph = graph;
    plan->previous_cur_copy = sched->cur_copy;
    plan->committed = false;
    sched->cur_copy = sched->next_copy;

    try {
        ggml_backend_sched_split_graph_impl(sched, graph);
        if (!ggml_gallocr_measure_n_v1(
                    sched->galloc, &sched->graph,
                    sched->node_backend_ids, sched->leaf_backend_ids)) {
            ggml_backend_sched_memory_plan_rollback(plan);
            return GGML_STATUS_ALLOC_FAILED;
        }

        uint64_t request_id = 1;
        const uint32_t n_chunks = ggml_gallocr_measure_get_chunk_count_v1(sched->galloc);
        plan->items.reserve((size_t) n_chunks + (size_t) sched->n_splits * 2);
        for (uint32_t i = 0; i < n_chunks; ++i) {
            ggml_backend_buffer_type_t buft = NULL;
            uint64_t requested = 0;
            uint64_t current = 0;
            if (!ggml_gallocr_measure_get_chunk_v1(sched->galloc, i, &buft, &requested, &current)) {
                ggml_backend_sched_memory_plan_rollback(plan);
                return GGML_STATUS_FAILED;
            }
            ggml_backend_t backend = NULL;
            for (int b = 0; b < sched->n_backends; ++b) {
                if (sched->bufts[b] == buft) {
                    backend = sched->backends[b];
                    break;
                }
            }
            if (backend == NULL) {
                ggml_backend_sched_memory_plan_rollback(plan);
                return GGML_STATUS_FAILED;
            }
            ggml_backend_memory_request_v1 item = {};
            item.struct_size = sizeof(item);
            item.kind = GGML_BACKEND_MEMORY_REQUEST_BUFFER;
            item.usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
            item.request_id = request_id++;
            item.backend = backend;
            item.buft = buft;
            item.requested_bytes = requested;
            item.currently_allocated_bytes = current;
            plan->items.push_back(item);
        }

        for (int i = 0; i < sched->n_splits; ++i) {
            ggml_backend_sched_split * split = &sched->splits[i];
            ggml_backend_memory_request_v1 graph_item = {};
            graph_item.struct_size = sizeof(graph_item);
            graph_item.kind = GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE;
            graph_item.usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
            graph_item.request_id = request_id++;
            graph_item.backend = sched->backends[split->backend_id];
            graph_item.graph = &split->graph;
            plan->items.push_back(graph_item);

            for (int j = 0; j < split->n_inputs; ++j) {
                ggml_backend_memory_request_v1 transfer = {};
                transfer.struct_size = sizeof(transfer);
                transfer.kind = GGML_BACKEND_MEMORY_REQUEST_TRANSFER;
                transfer.usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
                transfer.request_id = request_id++;
                transfer.backend = graph_item.backend;
                transfer.peer_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[j]);
                transfer.requested_bytes = ggml_nbytes(split->inputs[j]);
                plan->items.push_back(transfer);
            }
        }

        plan->fingerprint = ggml_backend_memory_request_fingerprint_v1(
            plan->items.data(), (uint32_t) plan->items.size());
    } catch (const ggml_backend_exception & error) {
        ggml_backend_sched_memory_plan_rollback(plan);
        return error.status;
    } catch (const std::bad_alloc &) {
        ggml_backend_sched_memory_plan_rollback(plan);
        return GGML_STATUS_ALLOC_FAILED;
    } catch (...) {
        ggml_backend_sched_memory_plan_rollback(plan);
        return GGML_STATUS_EXECUTION_FAILED;
    }
    // Splitting a mixed-backend graph legitimately replaces cross-backend
    // sources with scheduler-owned copies. Freeze the graph only after those
    // internal rewrites are complete, so commit rejects caller mutation but
    // does not reject the scheduler's own plan.
    plan->source_graph_fingerprint = ggml_backend_sched_memory_graph_fingerprint(graph);
    sched->memory_plan_active = true;
    sched->active_memory_plan = plan;
    *out_plan = plan;
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_sched_memory_plan_create_v1(
        ggml_backend_sched_t sched, struct ggml_cgraph * graph,
        ggml_backend_sched_memory_plan_t * out_plan) {
    return ggml_backend_noexcept_status([&]() {
        return ggml_backend_sched_memory_plan_create_v1_impl(sched, graph, out_plan);
    });
}

uint32_t ggml_backend_sched_memory_plan_get_item_count_v1(ggml_backend_sched_memory_plan_t plan) {
    return ggml_backend_noexcept_or<uint32_t>(
        [&]() { return plan == NULL ? 0 : (uint32_t) plan->items.size(); }, 0);
}

bool ggml_backend_sched_memory_plan_get_item_v1(
        ggml_backend_sched_memory_plan_t plan, uint32_t index,
        struct ggml_backend_memory_request_v1 * out_item) {
    if (plan == NULL || out_item == NULL || out_item->struct_size < sizeof(*out_item) || index >= plan->items.size()) {
        return false;
    }
    return ggml_backend_noexcept_or<bool>([&]() {
        *out_item = plan->items[index];
        return true;
    }, false);
}

enum ggml_status ggml_backend_sched_memory_plan_commit_v1(ggml_backend_sched_memory_plan_t plan) {
    uint32_t flags = 0;
    return ggml_backend_sched_memory_plan_commit_v2(plan, &flags);
}

static enum ggml_status ggml_backend_sched_memory_plan_commit_v2_impl(
        ggml_backend_sched_memory_plan_t plan, uint32_t * out_flags) {
    if (out_flags == NULL) {
        return GGML_STATUS_FAILED;
    }
    *out_flags = 0;
    if (plan == NULL || plan->committed || plan->sched == NULL || !plan->sched->memory_plan_active) {
        return GGML_STATUS_FAILED;
    }
    if (ggml_backend_sched_memory_graph_fingerprint(plan->source_graph) != plan->source_graph_fingerprint) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_sched_t sched = plan->sched;
    uint32_t allocator_flags = 0;
    const enum ggml_status allocator_status =
        ggml_gallocr_measure_commit_v2(sched->galloc, &allocator_flags);
    if ((allocator_flags & GGML_GALLOCR_MEASURE_COMMIT_MAY_HAVE_MUTATED) != 0) {
        *out_flags |= GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_MAY_HAVE_MUTATED;
    }
    if (allocator_status != GGML_STATUS_SUCCESS) {
        return allocator_status;
    }
    // The measure commit atomically publishes replacement buffers. From this
    // point onward a failure may leave the scheduler's high-water allocation
    // changed even if graph tensor placement cannot be completed.
    *out_flags |= GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_MAY_HAVE_MUTATED;
    sched->allocated_graph = plan->source_graph;
    const enum ggml_status allocation_status = ggml_backend_noexcept_status(
        [&]() { return ggml_gallocr_alloc_graph_v2(sched->galloc, &sched->graph); });
    if (allocation_status != GGML_STATUS_SUCCESS) {
        // A partially-bound graph cannot retain its newly published gallocr
        // buffers behind a refunded admission. Rebuild the allocator in place:
        // reset first detaches every tensor while the old buffers are alive,
        // then status-aware destruction proves whether all provider releases
        // completed. The scheduler stays reusable only when Rust accepts both
        // this proof and the subsequent live device-health receipts.
        ggml_gallocr_t replacement = ggml_gallocr_new_n(sched->bufts, sched->n_backends);
        if (replacement != NULL) {
            sched->memory_plan_active = false;
            sched->active_memory_plan = NULL;
            ggml_backend_sched_reset(sched);
            ggml_gallocr_t retired = sched->galloc;
            sched->galloc = replacement;
            const enum ggml_status release_status = ggml_gallocr_free_status(retired);
            plan->committed = true;
            plan->sched = NULL;
            if (release_status == GGML_STATUS_SUCCESS) {
                *out_flags |= GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_RELEASE_PROVEN;
            }
        }
        return allocation_status;
    }
    sched->is_alloc = true;
    sched->next_copy = (sched->cur_copy + 1) % sched->n_copies;
    sched->memory_plan_active = false;
    sched->active_memory_plan = NULL;
    plan->committed = true;
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_sched_memory_plan_commit_v2(
        ggml_backend_sched_memory_plan_t plan, uint32_t * out_flags) {
    return ggml_backend_noexcept_status(
        [&]() { return ggml_backend_sched_memory_plan_commit_v2_impl(plan, out_flags); });
}

static void ggml_backend_sched_memory_plan_free_v1_impl(ggml_backend_sched_memory_plan_t plan) {
    if (plan == NULL) {
        return;
    }
    if (!plan->committed && plan->sched != NULL) {
        ggml_backend_sched_memory_plan_rollback(plan);
        return;
    }
    delete plan;
}

void ggml_backend_sched_memory_plan_free_v1(ggml_backend_sched_memory_plan_t plan) {
    ggml_backend_noexcept_void(
        [&]() { ggml_backend_sched_memory_plan_free_v1_impl(plan); });
}

static enum ggml_status ggml_backend_sched_alloc_graph_impl(
        ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    if (sched == NULL || graph == NULL || graph->n_nodes < 0 ||
            graph->n_leafs < 0 || sched->memory_plan_active ||
            sched->is_alloc) {
        return GGML_STATUS_FAILED;
    }
    size_t graph_identity_count = 0;
    if (!ggml_backend_checked_add_size(
            (size_t) graph->n_nodes, (size_t) graph->n_leafs,
            &graph_identity_count) || graph_identity_count > sched->hash_set.size) {
        return GGML_STATUS_ALLOC_FAILED;
    }

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    try {
        ggml_backend_sched_split_graph_impl(sched, graph);

        sched->allocated_graph = graph;
        const enum ggml_status allocation_status =
            ggml_backend_sched_alloc_splits(sched);
        if (allocation_status != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_reset(sched);
            return allocation_status;
        }
    } catch (...) {
        ggml_backend_noexcept_void([&]() { ggml_backend_sched_reset(sched); });
        throw;
    }

    sched->is_alloc = true;

    return GGML_STATUS_SUCCESS;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    return ggml_backend_noexcept_or<bool>(
        [&]() {
            return ggml_backend_sched_alloc_graph_impl(sched, graph) ==
                GGML_STATUS_SUCCESS;
        }, false);
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    if (sched == NULL || graph == NULL) {
        return GGML_STATUS_FAILED;
    }
    enum ggml_status submitted = ggml_backend_sched_graph_compute_async(sched, graph);
    return ggml_backend_status_merge(submitted, ggml_backend_sched_synchronize(sched));
}

static enum ggml_status ggml_backend_sched_graph_compute_async_impl(
        ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    if (sched == NULL || graph == NULL) {
        return GGML_STATUS_FAILED;
    }
    if (sched->memory_plan_active) {
        return GGML_STATUS_FAILED;
    }
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        const enum ggml_status allocation_status =
            ggml_backend_sched_alloc_graph_impl(sched, graph);
        if (allocation_status != GGML_STATUS_SUCCESS) {
            return allocation_status;
        }
    }

    return ggml_backend_sched_compute_splits(sched, NULL, NULL, false, NULL);
}

enum ggml_status ggml_backend_sched_graph_compute_async(
        ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    return ggml_backend_noexcept_status(
        [&]() { return ggml_backend_sched_graph_compute_async_impl(sched, graph); });
}

static enum ggml_status ggml_backend_sched_graph_compute_with_abort_impl(
        ggml_backend_sched_t sched, struct ggml_cgraph * graph,
        ggml_abort_callback abort_callback, void * abort_callback_data,
        struct ggml_backend_graph_cancel_capability * cancel_capability) {
    if (sched == NULL || graph == NULL || cancel_capability == NULL) {
        return GGML_STATUS_FAILED;
    }

    ggml_backend_graph_cancel_capability_reset(cancel_capability);
    if (abort_callback == NULL) {
        return ggml_backend_sched_graph_compute(sched, graph);
    }

    bool native_cancel = true;
    std::vector<ggml_backend_set_abort_callback_t> native_callbacks;
    native_callbacks.reserve(sched->n_backends);
    for (int i = 0; i < sched->n_backends; ++i) {
        ggml_backend_set_abort_callback_t callback =
            ggml_backend_native_abort_callback(sched->backends[i]);
        native_callbacks.push_back(callback);
        if (callback == NULL) {
            native_cancel = false;
        }
    }

    // Avoid graph splitting/allocation and its possible backend synchronization
    // when the request was already cancelled before this compute began.
    const enum ggml_status prestart = ggml_backend_native_abort_status(
        GGML_STATUS_SUCCESS, abort_callback, abort_callback_data);
    if (prestart != GGML_STATUS_SUCCESS) {
        if (prestart != GGML_STATUS_ABORTED) {
            return prestart;
        }
        return ggml_backend_status_merge(GGML_STATUS_ABORTED, ggml_backend_sched_synchronize(sched));
    }

    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }
    if (!sched->is_alloc) {
        const enum ggml_status allocation_status =
            ggml_backend_sched_alloc_graph_impl(sched, graph);
        if (allocation_status != GGML_STATUS_SUCCESS) {
            return allocation_status;
        }
    }

    const enum ggml_status post_alloc = ggml_backend_native_abort_status(
        GGML_STATUS_SUCCESS, abort_callback, abort_callback_data);
    if (post_alloc != GGML_STATUS_SUCCESS) {
        if (post_alloc != GGML_STATUS_ABORTED) {
            return post_alloc;
        }
        return ggml_backend_status_merge(GGML_STATUS_ABORTED, ggml_backend_sched_synchronize(sched));
    }

    std::vector<struct ggml_backend_graph_cancel_capability> backend_capabilities;
    int native_callbacks_installed = 0;
    if (native_cancel) {
        backend_capabilities.resize(sched->n_backends);
        for (int i = 0; i < sched->n_backends; ++i) {
            ggml_backend_graph_cancel_capability_reset(&backend_capabilities[i]);
            native_callbacks_installed = i + 1;
            const enum ggml_status install_status = ggml_backend_noexcept_status([&]() {
                native_callbacks[i](
                    sched->backends[i], abort_callback, abort_callback_data,
                    &backend_capabilities[i]);
                return GGML_STATUS_SUCCESS;
            });
            if (install_status != GGML_STATUS_SUCCESS) {
                for (int installed = 0; installed < native_callbacks_installed; ++installed) {
                    ggml_backend_noexcept_void([&]() {
                        native_callbacks[installed](
                            sched->backends[installed], NULL, NULL, NULL);
                    });
                }
                return install_status;
            }
        }
    }

    enum ggml_status submitted = ggml_backend_sched_compute_splits(
        sched, abort_callback, abort_callback_data, native_cancel, cancel_capability);
    enum ggml_status status = ggml_backend_status_merge(
        submitted, ggml_backend_sched_synchronize(sched));

    if (native_cancel) {
        for (const auto & backend_capability : backend_capabilities) {
            ggml_backend_graph_cancel_capability_merge(cancel_capability, &backend_capability);
        }
        status = ggml_backend_native_abort_status(status, abort_callback, abort_callback_data);
        for (int i = 0; i < native_callbacks_installed; ++i) {
            const enum ggml_status clear_status = ggml_backend_noexcept_status([&]() {
                native_callbacks[i](sched->backends[i], NULL, NULL, NULL);
                return GGML_STATUS_SUCCESS;
            });
            if (status == GGML_STATUS_SUCCESS && clear_status != GGML_STATUS_SUCCESS) {
                status = clear_status;
            }
        }
    }

    return status;
}

enum ggml_status ggml_backend_sched_graph_compute_with_abort(
        ggml_backend_sched_t sched, struct ggml_cgraph * graph,
        ggml_abort_callback abort_callback, void * abort_callback_data,
        struct ggml_backend_graph_cancel_capability * cancel_capability) {
    return ggml_backend_noexcept_status([&]() {
        return ggml_backend_sched_graph_compute_with_abort_impl(
            sched, graph, abort_callback, abort_callback_data, cancel_capability);
    });
}

enum ggml_status ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return GGML_STATUS_FAILED;
    }
    enum ggml_status status = GGML_STATUS_SUCCESS;
    for (int i = 0; i < sched->n_backends; i++) {
        status = ggml_backend_status_merge(status, ggml_backend_synchronize(sched->backends[i]));
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
    return status;
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    if (sched == NULL) {
        return;
    }
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    return sched == NULL ? 0 : sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    return sched == NULL ? 0 : sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    return sched == NULL ? 0 : sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    return sched != NULL && i >= 0 && i < sched->n_backends
        ? sched->backends[i] : NULL;
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    if (sched == NULL || backend == NULL) {
        return NULL;
    }
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    return backend_index >= 0 && backend_index < sched->n_backends
        ? sched->bufts[backend_index] : NULL;
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    if (sched == NULL || backend == NULL) {
        return 0;
    }
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    return backend_index >= 0 && backend_index < sched->n_backends
        ? ggml_gallocr_get_buffer_size(sched->galloc, backend_index) : 0;
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    if (sched == NULL || node == NULL || backend == NULL) {
        return;
    }
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    if (backend_index < 0 || backend_index >= sched->n_backends) {
        return;
    }
    ggml_backend_noexcept_void([&]() {
        tensor_backend_id(node) = backend_index;
        SET_CAUSE(node, "usr");
        sched->is_reset = false;
    });
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    if (sched == NULL || node == NULL) {
        return NULL;
    }
    return ggml_backend_noexcept_or<ggml_backend_t>([&]() {
        int backend_index = tensor_backend_id(node);
        return backend_index >= 0 && backend_index < sched->n_backends
            ? sched->backends[backend_index] : NULL;
    }, NULL);
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    if (tensor == NULL || tensor->buffer != NULL || tensor->view_src == NULL ||
            tensor->view_src->buffer == NULL || tensor->view_src->data == NULL ||
            tensor->view_offs > ggml_nbytes(tensor->view_src)) {
        return GGML_STATUS_FAILED;
    }

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    const enum ggml_status status = ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
    if (status != GGML_STATUS_SUCCESS) {
        tensor->buffer = NULL;
        tensor->data = NULL;
    }
    return status;
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    if (buffer == NULL || tensor == NULL || addr == NULL ||
            tensor->buffer != NULL || tensor->data != NULL || tensor->view_src != NULL) {
        return GGML_STATUS_FAILED;
    }
    if (!ggml_backend_buffer_is_meta(buffer)) {
        void * base = ggml_backend_buffer_get_base(buffer);
        const size_t buffer_size = ggml_backend_buffer_get_size(buffer);
        const size_t allocation_size = ggml_backend_buffer_get_alloc_size(buffer, tensor);
        const uintptr_t base_address = (uintptr_t) base;
        const uintptr_t tensor_address = (uintptr_t) addr;
        if (base == NULL || tensor_address < base_address ||
                tensor_address - base_address > buffer_size ||
                allocation_size > buffer_size - (tensor_address - base_address)) {
            return GGML_STATUS_FAILED;
        }
    }

    tensor->buffer = buffer;
    tensor->data = addr;
    const enum ggml_status status = ggml_backend_buffer_init_tensor(buffer, tensor);
    if (status != GGML_STATUS_SUCCESS) {
        tensor->buffer = NULL;
        tensor->data = NULL;
    }
    return status;
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static enum ggml_status graph_copy_init_tensor(
        struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies,
        bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return GGML_STATUS_SUCCESS;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    enum ggml_status status;
    if (dst->view_src != NULL) {
        status = graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
        status = ggml_backend_view_init(dst);
    }
    else {
        status = ggml_backend_tensor_copy(src, dst);
    }
    if (status != GGML_STATUS_SUCCESS) {
        return status;
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        status = graph_copy_init_tensor(hash_set, node_copies, node_init, s);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }
    return GGML_STATUS_SUCCESS;
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const enum ggml_status status =
            graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
        if (status != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: failed to initialize graph copy with status %d\n", __func__, status);
            ggml_hash_set_free(&hash_set);
            free(node_copies);
            free(node_init);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx_allocated);
            ggml_free(ctx_unallocated);
            return {
                /* .buffer           = */ NULL,
                /* .ctx_allocated    = */ NULL,
                /* .ctx_unallocated  = */ NULL,
                /* .graph            = */ NULL,
            };
        }
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
