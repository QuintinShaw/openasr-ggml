#include "ggml-metal.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-context.h"
#include "ggml-metal-ops.h"

#include <algorithm>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
static int g_devices = 1;

// forward declaration
static bool ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

static void ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

static void * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    return ggml_metal_buffer_get_base(ctx);
}

static void ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static bool ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    if (!ggml_backend_buffer_is_metal(src->buffer)) {
        return GGML_STATUS_FAILED;
    }

    return ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

static void ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ NULL,
};

// private buffer

static void ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

static void * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    return ggml_metal_buffer_get_base(ctx);
}

static void ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static bool ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    if (!ggml_backend_buffer_is_metal(src->buffer)) {
        return GGML_STATUS_FAILED;
    }

    return ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

static void ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ NULL,
};

static bool ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

struct ggml_backend_metal_buffer_type {
    int device;
    std::string name;
};

struct ggml_backend_metal_buffer_type_deleter {
    void operator()(ggml_backend_metal_buffer_type * ctx) const {
        delete ctx;
    }
};

typedef std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
static ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, bool shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    return ggml_backend_buffer_init(buft, buf_i, res, size);
}

static size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    switch (tensor->op) {
        case GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
            } break;
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
            {
                res *= 2;
            } break;
        case GGML_OP_TOP_K:
            {
                res = 2*sizeof(int32_t)*ggml_nelements(tensor->src[0]);
            } break;
        default:
            break;
    }

    return res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

static const char * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                new ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .get_alignment    = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size     = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .is_host          = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// default (private) buffer type

static const char * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, false);
}

static size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = new ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .get_alignment    = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size     = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .is_host          = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// mapped buffer type

static const char * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = new ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ ggml_backend_metal_buffer_type_mapped_get_name,
                    /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_mapped_alloc_buffer,
                    /* .get_alignment    = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size     = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .is_host          = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// backend

static const char * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_get_name(ctx);
}

static void ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    free(backend);
}

static enum ggml_status ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_synchronize(ctx);
}

static enum ggml_status ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

static enum ggml_status ggml_backend_metal_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

static enum ggml_status ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        return GGML_STATUS_FAILED;
    }

    if (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        return GGML_STATUS_FAILED;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    return ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

static enum ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_graph_compute(ctx, cgraph);
}

static enum ggml_status ggml_backend_metal_event_record_status(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    return ggml_metal_event_record(ctx, ev);
}

static enum ggml_status ggml_backend_metal_event_wait_status(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    return ggml_metal_event_wait(ctx, ev);
}

static void ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

static void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

static void ggml_backend_metal_set_abort_callback(
        ggml_backend_t backend, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_set_abort_callback(ctx, abort_callback, abort_callback_data);
}

static ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record_status     = */ ggml_backend_metal_event_record_status,
    /* .event_wait_status       = */ ggml_backend_metal_event_wait_status,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

static ggml_guid_t ggml_backend_metal_guid(void) {
    static ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    return &guid;
}

ggml_backend_t ggml_backend_metal_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;
}

bool ggml_backend_is_metal(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_supports_family(ctx, family);
}

void ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

static const char * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return props_dev->name;
}

static const char * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_get_props(ctx_dev)->desc;
}

static void ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, free, total);
}

static enum ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ true,
    };
}

static ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

static ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

static bool ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_supports_op(ctx_dev, op);
}

static bool ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name);

    GGML_UNUSED(dev);
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    return ev;
}

static void ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

static enum ggml_status ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    return ggml_metal_device_event_synchronize(ctx_dev, evt);
}

static ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

struct ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

typedef struct ggml_backend_metal_reg * ggml_backend_metal_reg_t;

static ggml_backend_metal_reg_t ggml_backend_metal_reg_init(void) {
    ggml_backend_metal_reg_t ctx = new struct ggml_backend_metal_reg;

    return ctx;
}

static void ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

struct ggml_backend_metal_reg_deleter {
    void operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

typedef std::unique_ptr<struct ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

static const char * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_metal_reg_device_count(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature g_ggml_backend_metal_features[] = {
#if defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { NULL, NULL },
};

static ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    return g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

static ggml_backend_memory_domain_id_v1 ggml_backend_metal_memory_domain(uint32_t kind) {
    ggml_backend_memory_domain_id_v1 id = {};
    id.kind = kind;
    return id;
}

static uint64_t ggml_backend_metal_memory_round_page(uint64_t size) {
    const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
    return page == 0 || size > UINT64_MAX - (page - 1) ? size : ((size + page - 1) / page) * page;
}

static enum ggml_status ggml_backend_metal_memory_get_domains(
        ggml_backend_dev_t dev, ggml_backend_memory_domain_v1 * domains, uint32_t * inout_count) {
    GGML_UNUSED(dev);
    if (inout_count == NULL) return GGML_STATUS_FAILED;
    const uint32_t capacity = *inout_count;
    *inout_count = 1;
    if (domains == NULL) return GGML_STATUS_SUCCESS;
    if (capacity < 1 || domains[0].struct_size < sizeof(domains[0])) return GGML_STATUS_FAILED;
    ggml_backend_memory_domain_v1 domain = {};
    domain.struct_size = sizeof(domain);
    domain.id = ggml_backend_metal_memory_domain(GGML_BACKEND_MEMORY_DOMAIN_UNIFIED);
    snprintf(domain.name, sizeof(domain.name), "metal/unified-working-set");
    domains[0] = domain;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_metal_memory_quote(
        const ggml_backend_memory_request_v1 * requests, uint32_t request_count,
        ggml_backend_memory_quote_v1 * quote, ggml_backend_memory_claim_v1 * claims,
        uint32_t * inout_claim_count) {
    if ((request_count && requests == NULL) || quote == NULL || quote->struct_size < sizeof(*quote) || inout_claim_count == NULL) {
        return GGML_STATUS_FAILED;
    }
    uint32_t required = 0;
    uint32_t graph_private_count = 0;
    ggml_backend_t backend = NULL;
    for (uint32_t i = 0; i < request_count; ++i) {
        if (requests[i].struct_size < sizeof(requests[i])) return GGML_STATUS_FAILED;
        if (requests[i].backend != NULL && !ggml_backend_is_metal(requests[i].backend)) return GGML_STATUS_FAILED;
        if (backend != NULL && requests[i].backend != NULL && requests[i].backend != backend) return GGML_STATUS_FAILED;
        if (requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE) {
            if (requests[i].backend == NULL || requests[i].graph == NULL) return GGML_STATUS_FAILED;
            ++graph_private_count;
            ++required;
        }
        backend = backend ? backend : requests[i].backend;
        if (requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_BUFFER ||
                requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT ||
                requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_TRANSFER) {
            ++required;
        }
    }
    // The Metal quote depends only on immutable request shapes and buffer-type
    // allocation-size queries. Live working-set pressure is carried by the
    // immediately-fetched stats, not by the quote epoch.
    const uint64_t generation = 1;
    quote->flags = graph_private_count
        ? GGML_BACKEND_MEMORY_QUOTE_OPAQUE_DRIVER_COSTS_REQUIRE_DOMAIN_HEADROOM
        : 0;
    quote->residual_flags = 0;
    quote->residual_request_count = 0;
    quote->provisional_requested_upper_bytes = 0;
    quote->stats_generation = generation;
    quote->request_fingerprint = ggml_backend_memory_request_fingerprint_v1(requests, request_count);
    quote->quote_token = quote->request_fingerprint ^ generation;
    const uint32_t capacity = *inout_claim_count;
    *inout_claim_count = required;
    if (claims == NULL) return GGML_STATUS_SUCCESS;
    if (capacity < required) return GGML_STATUS_FAILED;
    uint32_t out = 0;
    for (uint32_t i = 0; i < request_count; ++i) {
        if (requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE) {
            // Metal does not retain an engine-visible per-graph workspace.
            // Command-buffer and driver-internal allocations are opaque and
            // covered by domain headroom, as declared by the quote flag.
            ggml_backend_memory_claim_v1 claim = {};
            claim.struct_size = sizeof(claim);
            claim.flags = GGML_BACKEND_MEMORY_CLAIM_EXACT;
            claim.request_id = requests[i].request_id;
            claim.domain = ggml_backend_metal_memory_domain(GGML_BACKEND_MEMORY_DOMAIN_UNIFIED);
            claims[out++] = claim;
            continue;
        }
        if (requests[i].kind != GGML_BACKEND_MEMORY_REQUEST_BUFFER &&
                requests[i].kind != GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT &&
                requests[i].kind != GGML_BACKEND_MEMORY_REQUEST_TRANSFER) continue;
        uint64_t committed = ggml_backend_metal_memory_round_page(requests[i].requested_bytes);
        uint64_t before = ggml_backend_metal_memory_round_page(requests[i].currently_allocated_bytes);
        if (requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_BUFFER && requests[i].buft != NULL) {
            if (requests[i].buft->iface.get_name != ggml_backend_metal_buffer_type_private_get_name &&
                    requests[i].buft->iface.get_name != ggml_backend_metal_buffer_type_shared_get_name &&
                    requests[i].buft->iface.get_name != ggml_backend_metal_buffer_type_mapped_get_name) {
                return GGML_STATUS_FAILED;
            }
            ggml_backend_metal_buffer_type * buft_ctx =
                (ggml_backend_metal_buffer_type *) requests[i].buft->context;
            const bool shared = requests[i].buft->iface.get_name !=
                ggml_backend_metal_buffer_type_private_get_name;
            committed = ggml_metal_device_get_buffer_allocation_size(
                ggml_metal_device_get(buft_ctx->device), requests[i].requested_bytes, shared);
            before = ggml_metal_device_get_buffer_allocation_size(
                ggml_metal_device_get(buft_ctx->device), requests[i].currently_allocated_bytes, shared);
        }
        const bool reuse = requests[i].currently_allocated_bytes >= requests[i].requested_bytes;
        const uint64_t after = reuse ? before : committed;
        ggml_backend_memory_claim_v1 claim = {};
        claim.struct_size = sizeof(claim);
        claim.flags = GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER;
        if (requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT) {
            claim.flags |= GGML_BACKEND_MEMORY_CLAIM_FILE_BACKED;
            claim.domain = ggml_backend_metal_memory_domain(GGML_BACKEND_MEMORY_DOMAIN_FILE_BACKED);
        } else {
            claim.domain = ggml_backend_metal_memory_domain(GGML_BACKEND_MEMORY_DOMAIN_UNIFIED);
        }
        if (requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_TRANSFER) claim.flags |= GGML_BACKEND_MEMORY_CLAIM_TRANSIENT;
        claim.request_id = requests[i].request_id;
        claim.payload_requested_bytes = requests[i].requested_bytes;
        claim.committed_before_bytes = before;
        claim.committed_after_upper_bytes = after;
        claim.commit_peak_extra_upper_bytes = reuse ? 0 : committed;
        claim.resident_after_upper_bytes = after;
        claim.retained_after_use_upper_bytes = requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_TRANSFER ? 0 : after;
        claims[out++] = claim;
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_metal_memory_reserve_private(
        const ggml_backend_memory_request_v1 * requests, uint32_t request_count,
        const ggml_backend_memory_quote_v1 * quote, ggml_backend_memory_claim_v1 * actual,
        uint32_t * inout_actual_count) {
    GGML_UNUSED(actual);
    if ((request_count && requests == NULL) || quote == NULL || inout_actual_count == NULL ||
            quote->request_fingerprint != ggml_backend_memory_request_fingerprint_v1(requests, request_count)) return GGML_STATUS_FAILED;
    *inout_actual_count = 0;
    ggml_backend_t backend = NULL;
    for (uint32_t i = 0; i < request_count; ++i) {
        if (requests[i].struct_size < sizeof(requests[i]) ||
                (requests[i].backend != NULL && !ggml_backend_is_metal(requests[i].backend))) return GGML_STATUS_FAILED;
        if (backend != NULL && requests[i].backend != NULL && requests[i].backend != backend) return GGML_STATUS_FAILED;
        backend = backend ? backend : requests[i].backend;
        if (requests[i].backend != NULL && ggml_metal_is_quarantined((ggml_metal_t) requests[i].backend->context)) {
            return GGML_STATUS_BACKEND_POISONED;
        }
    }
    if (backend != NULL) {
        const uint64_t generation = 1;
        if (generation != quote->stats_generation || quote->quote_token != (quote->request_fingerprint ^ generation)) {
            return GGML_STATUS_FAILED;
        }
    }
    // ABI v1 exposes no engine-visible Metal graph-private allocation. This
    // validation-only hook is failure-atomic; opaque driver costs remain under
    // the domain-headroom contract declared by the quote.
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_metal_memory_get_stats(
        ggml_backend_dev_t dev, ggml_backend_t backend,
        ggml_backend_memory_stats_v1 * stats, uint32_t * inout_count) {
    if (dev == NULL || inout_count == NULL) return GGML_STATUS_FAILED;
    const uint32_t capacity = *inout_count;
    *inout_count = 1;
    if (stats == NULL) return GGML_STATUS_SUCCESS;
    if (capacity < 1 || stats[0].struct_size < sizeof(stats[0])) return GGML_STATUS_FAILED;
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    ggml_backend_memory_stats_v1 value = {};
    value.struct_size = sizeof(value);
    value.domain = ggml_backend_metal_memory_domain(GGML_BACKEND_MEMORY_DOMAIN_UNIFIED);
    value.total_bytes = total_bytes;
    value.budget_bytes = total_bytes;
    value.device_free_bytes = std::min(free_bytes, total_bytes);
    value.device_used_bytes = total_bytes - value.device_free_bytes;
    value.generation = 1;
    value.timestamp_monotonic_ns = (uint64_t) ggml_time_us() * 1000;
    value.health = backend != NULL && ggml_metal_is_quarantined((ggml_metal_t) backend->context)
        ? GGML_BACKEND_MEMORY_QUARANTINED : GGML_BACKEND_MEMORY_HEALTHY;
    stats[0] = value;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_metal_memory_trim(ggml_backend_t backend, uint64_t flags) {
    GGML_UNUSED(flags);
    return backend == NULL ? GGML_STATUS_FAILED : ggml_backend_metal_synchronize(backend);
}

static enum ggml_status ggml_backend_metal_memory_quarantine(
        ggml_backend_t backend, const ggml_backend_memory_quarantine_v1 * request) {
    if (backend == NULL || request == NULL || request->struct_size < sizeof(*request)) return GGML_STATUS_FAILED;
    ggml_metal_quarantine((ggml_metal_t) backend->context);
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_memory_api_v1 * ggml_backend_metal_memory_get_api_v1(void) {
    static const ggml_backend_memory_api_v1 api = {
        sizeof(ggml_backend_memory_api_v1), GGML_BACKEND_MEMORY_ABI_V1, 0,
        ggml_backend_metal_memory_get_domains,
        ggml_backend_metal_memory_quote,
        ggml_backend_metal_memory_reserve_private,
        ggml_backend_metal_memory_get_stats,
        ggml_backend_metal_memory_trim,
        ggml_backend_metal_memory_quarantine,
    };
    return &api;
}

static void * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_metal_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_metal_set_abort_callback;
    }
    if (strcmp(name, GGML_BACKEND_MEMORY_API_V1_PROC) == 0) {
        return (void *) ggml_backend_metal_memory_get_api_v1;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_count,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

static ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, int device) {
    return new ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device),
    };
}

static void ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

struct ggml_backend_device_deleter {
    void operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        const char * env = getenv("GGML_METAL_DEVICES");
        if (env) {
            g_devices = atoi(env);
        }

        static std::vector<ggml_backend_device_ptr> devs;

        if (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", true);

            static ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            for (int i = 0; i < g_devices; ++i) {
                auto * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
