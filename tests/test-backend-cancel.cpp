#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif
#include "ggml-impl.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>

struct fake_event_state {
    enum ggml_status completion_status = GGML_STATUS_SUCCESS;
};

struct fake_backend_state {
    const char * name = "fake";
    enum ggml_backend_dev_type type = GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_buffer_type_t buft = nullptr;
    bool pending = false;
    enum ggml_status submit_status = GGML_STATUS_SUCCESS;
    enum ggml_status completion_status = GGML_STATUS_SUCCESS;
    std::vector<enum ggml_status> completion_sequence;
    enum ggml_status event_record_status = GGML_STATUS_SUCCESS;
    enum ggml_status event_wait_status = GGML_STATUS_SUCCESS;
    int synchronize_calls = 0;
    int copy_calls = 0;
    enum ggml_status transfer_submit_status = GGML_STATUS_SUCCESS;
    enum ggml_status transfer_completion_status = GGML_STATUS_SUCCESS;
    enum transfer_kind_t { TRANSFER_NONE, TRANSFER_SET, TRANSFER_GET, TRANSFER_COPY };
    transfer_kind_t transfer_kind = TRANSFER_NONE;
    void * transfer_dst = nullptr;
    const void * transfer_src = nullptr;
    size_t transfer_size = 0;
    bool poisoned = false;
    std::vector<int> graph_sizes;
    ggml_abort_callback abort_callback = nullptr;
    void * abort_callback_data = nullptr;
    struct ggml_backend_graph_cancel_capability * cancel_capability = nullptr;
    enum ggml_backend_graph_cancel_observation_granularity native_granularity =
        GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT;
    int abort_set_calls = 0;
    int abort_clear_calls = 0;
    bool throw_graph_optimize = false;
    bool throw_init_tensor = false;
    bool throw_buffer_free_before_release = false;
    bool throw_buffer_free = false;
    bool throw_buffer_reset = false;
    bool fail_buffer_alloc = false;
    int buffer_free_calls = 0;
    int buffer_reset_calls = 0;
};

static const char * fake_buffer_name(ggml_backend_buffer_type_t buft) {
    return static_cast<fake_backend_state *>(buft->context)->name;
}

static void fake_buffer_free(ggml_backend_buffer_t buffer) {
    auto * state = static_cast<fake_backend_state *>(buffer->buft->context);
    if (state->throw_buffer_free_before_release) {
        throw ggml_backend_exception { GGML_STATUS_EXECUTION_FAILED, 109 };
    }
    ggml_aligned_free(buffer->context, buffer->size);
    state->buffer_free_calls++;
    if (state->throw_buffer_free) {
        throw ggml_backend_exception { GGML_STATUS_EXECUTION_FAILED, 110 };
    }
}

static void * fake_buffer_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static enum ggml_status fake_buffer_init_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor *) {
    auto * state = static_cast<fake_backend_state *>(buffer->buft->context);
    if (state->throw_init_tensor) {
        throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 111 };
    }
    return GGML_STATUS_SUCCESS;
}

static void fake_buffer_memset(
        ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
}

static void fake_buffer_set(
        ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

static void fake_buffer_get(
        ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

static void fake_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    std::memset(buffer->context, value, buffer->size);
}

static void fake_buffer_reset(ggml_backend_buffer_t buffer) {
    auto * state = static_cast<fake_backend_state *>(buffer->buft->context);
    state->buffer_reset_calls++;
    if (state->throw_buffer_reset) {
        throw ggml_backend_exception { GGML_STATUS_EXECUTION_FAILED, 113 };
    }
}

static ggml_backend_buffer_t fake_buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * state = static_cast<fake_backend_state *>(buft->context);
    if (state->fail_buffer_alloc) {
        return nullptr;
    }
    void * data = ggml_aligned_malloc(size);
    assert(data != nullptr);
    const ggml_backend_buffer_i iface = {
        /* .free_buffer   = */ fake_buffer_free,
        /* .get_base      = */ fake_buffer_base,
        /* .init_tensor   = */ fake_buffer_init_tensor,
        /* .memset_tensor = */ fake_buffer_memset,
        /* .set_tensor    = */ fake_buffer_set,
        /* .get_tensor    = */ fake_buffer_get,
        /* .set_tensor_2d = */ nullptr,
        /* .get_tensor_2d = */ nullptr,
        /* .cpy_tensor    = */ nullptr,
        /* .clear         = */ fake_buffer_clear,
        /* .reset         = */ fake_buffer_reset,
    };
    return ggml_backend_buffer_init(buft, iface, data, size);
}

static size_t fake_buffer_alignment(ggml_backend_buffer_type_t) {
    return GGML_MEM_ALIGN;
}

static size_t invalid_buffer_alignment(ggml_backend_buffer_type_t) {
    return 3;
}

static void * null_buffer_base(ggml_backend_buffer_t) {
    return nullptr;
}

static const char * fake_backend_name(ggml_backend_t backend) {
    return static_cast<fake_backend_state *>(backend->context)->name;
}

static enum ggml_status fake_transfer_submit(
        fake_backend_state * state, fake_backend_state::transfer_kind_t kind, void * dst, const void * src, size_t size) {
    if (state->poisoned) {
        return GGML_STATUS_BACKEND_POISONED;
    }
    if (state->transfer_submit_status != GGML_STATUS_SUCCESS) {
        if (state->transfer_submit_status == GGML_STATUS_DEVICE_LOST) {
            state->poisoned = true;
        }
        return state->transfer_submit_status;
    }
    state->transfer_kind = kind;
    state->transfer_dst = dst;
    state->transfer_src = src;
    state->transfer_size = size;
    state->pending = true;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status fake_set_tensor_async(
        ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * state = static_cast<fake_backend_state *>(backend->context);
    return fake_transfer_submit(state, fake_backend_state::TRANSFER_SET,
        static_cast<char *>(tensor->data) + offset, data, size);
}

static enum ggml_status fake_get_tensor_async(
        ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * state = static_cast<fake_backend_state *>(backend->context);
    return fake_transfer_submit(state, fake_backend_state::TRANSFER_GET,
        data, static_cast<const char *>(tensor->data) + offset, size);
}

static enum ggml_status fake_copy_async(
        ggml_backend_t, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    auto * state = static_cast<fake_backend_state *>(backend_dst->context);
    state->copy_calls++;
    return fake_transfer_submit(state, fake_backend_state::TRANSFER_COPY,
        dst->data, src->data, ggml_nbytes(src));
}

static const char * fake_device_name(ggml_backend_dev_t device) {
    return static_cast<fake_backend_state *>(device->context)->name;
}

static enum ggml_backend_dev_type fake_device_type(ggml_backend_dev_t device) {
    return static_cast<fake_backend_state *>(device->context)->type;
}

static ggml_backend_buffer_type_t fake_device_buffer_type(ggml_backend_dev_t device) {
    return static_cast<fake_backend_state *>(device->context)->buft;
}

static bool fake_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return true;
}

static bool fake_device_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return static_cast<fake_backend_state *>(device->context)->buft == buft;
}

static enum ggml_status fake_graph_compute(ggml_backend_t backend, struct ggml_cgraph * graph) {
    auto * state = static_cast<fake_backend_state *>(backend->context);
    if (state->abort_callback != nullptr && state->cancel_capability != nullptr) {
        state->cancel_capability->mechanism = GGML_BACKEND_GRAPH_CANCEL_NATIVE;
        state->cancel_capability->observation_granularity = state->native_granularity;
    }
    assert(!state->pending);
    state->pending = true;
    state->graph_sizes.push_back(graph->n_nodes);
    if (state->submit_status != GGML_STATUS_SUCCESS) {
        return state->submit_status;
    }
    if (state->abort_callback != nullptr && state->abort_callback(state->abort_callback_data)) {
        return GGML_STATUS_ABORTED;
    }
    return GGML_STATUS_SUCCESS;
}

static void fake_graph_optimize(ggml_backend_t backend, ggml_cgraph *) {
    auto * state = static_cast<fake_backend_state *>(backend->context);
    if (state->throw_graph_optimize) {
        throw ggml_backend_exception { GGML_STATUS_DEVICE_LOST, 108 };
    }
}

static void fake_set_abort_callback(
        ggml_backend_t backend, ggml_abort_callback abort_callback, void * abort_callback_data,
        struct ggml_backend_graph_cancel_capability * cancel_capability) {
    auto * state = static_cast<fake_backend_state *>(backend->context);
    state->abort_callback = abort_callback;
    state->abort_callback_data = abort_callback != nullptr ? abort_callback_data : nullptr;
    state->cancel_capability = abort_callback != nullptr ? cancel_capability : nullptr;
    if (abort_callback != nullptr) {
        state->abort_set_calls++;
    } else {
        state->abort_clear_calls++;
    }
}

static void * fake_reg_get_proc_address(ggml_backend_reg_t, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return reinterpret_cast<void *>(fake_set_abort_callback);
    }
    return nullptr;
}

static enum ggml_status fake_event_record_status(ggml_backend_t backend, ggml_backend_event_t) {
    return static_cast<fake_backend_state *>(backend->context)->event_record_status;
}

static enum ggml_status fake_event_wait_status(ggml_backend_t backend, ggml_backend_event_t) {
    return static_cast<fake_backend_state *>(backend->context)->event_wait_status;
}

static enum ggml_status fake_synchronize(ggml_backend_t backend) {
    auto * state = static_cast<fake_backend_state *>(backend->context);
    state->synchronize_calls++;
    if (state->poisoned) {
        state->pending = false;
        return GGML_STATUS_BACKEND_POISONED;
    }
    if (state->transfer_kind != fake_backend_state::TRANSFER_NONE) {
        const enum ggml_status status = state->transfer_completion_status;
        state->pending = false;
        state->transfer_kind = fake_backend_state::TRANSFER_NONE;
        if (status == GGML_STATUS_SUCCESS) {
            std::memcpy(state->transfer_dst, state->transfer_src, state->transfer_size);
        } else if (status == GGML_STATUS_DEVICE_LOST) {
            state->poisoned = true;
        }
        return status;
    }
    state->pending = false;
    if (!state->completion_sequence.empty()) {
        enum ggml_status status = state->completion_sequence.front();
        state->completion_sequence.erase(state->completion_sequence.begin());
        return status;
    }
    return state->completion_status;
}

struct abort_probe {
    fake_backend_state * backend;
    int polls = 0;
    int abort_after;
};

static bool abort_after_polls(void * data) {
    auto * probe = static_cast<abort_probe *>(data);
    assert(!probe->backend->pending);
    return ++probe->polls >= probe->abort_after;
}

static bool never_abort(void *) {
    return false;
}

static void reset(fake_backend_state & state) {
    state.pending = false;
    state.submit_status = GGML_STATUS_SUCCESS;
    state.completion_status = GGML_STATUS_SUCCESS;
    state.completion_sequence.clear();
    state.event_record_status = GGML_STATUS_SUCCESS;
    state.event_wait_status = GGML_STATUS_SUCCESS;
    state.synchronize_calls = 0;
    state.copy_calls = 0;
    state.transfer_submit_status = GGML_STATUS_SUCCESS;
    state.transfer_completion_status = GGML_STATUS_SUCCESS;
    state.transfer_kind = fake_backend_state::TRANSFER_NONE;
    state.transfer_dst = nullptr;
    state.transfer_src = nullptr;
    state.transfer_size = 0;
    state.poisoned = false;
    state.graph_sizes.clear();
    state.abort_callback = nullptr;
    state.abort_callback_data = nullptr;
    state.cancel_capability = nullptr;
    state.native_granularity = GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT;
    state.abort_set_calls = 0;
    state.abort_clear_calls = 0;
    state.throw_graph_optimize = false;
    state.throw_init_tensor = false;
    state.throw_buffer_free_before_release = false;
    state.throw_buffer_free = false;
    state.throw_buffer_reset = false;
    state.fail_buffer_alloc = false;
    state.buffer_free_calls = 0;
    state.buffer_reset_calls = 0;
}

struct scheduler_copy_abort_probe {
    fake_backend_state * copy_backend;
    int polls = 0;
};

static bool abort_after_first_scheduler_copy(void * data) {
    auto * probe = static_cast<scheduler_copy_abort_probe *>(data);
    probe->polls++;
    return probe->copy_backend->copy_calls > 0;
}

static bool native_abort_after_polls(void * data) {
    auto * probe = static_cast<abort_probe *>(data);
    return ++probe->polls >= probe->abort_after;
}

static bool native_abort_after_submission(void * data) {
    auto * probe = static_cast<abort_probe *>(data);
    probe->polls++;
    return probe->backend->pending;
}

struct scheduler_callback_probe {
    int observed = 0;
};

static bool observe_scheduler_node(ggml_tensor *, bool ask, void * user_data) {
    auto * probe = static_cast<scheduler_callback_probe *>(user_data);
    if (ask) {
        return true;
    }
    probe->observed++;
    return true;
}

static ggml_backend_buffer_type make_fake_buffer_type(fake_backend_state * state) {
    return {
        /* .iface   = */ {
            /* .get_name       = */ fake_buffer_name,
            /* .alloc_buffer   = */ fake_buffer_alloc,
            /* .get_alignment  = */ fake_buffer_alignment,
            /* .get_max_size   = */ nullptr,
            /* .get_alloc_size = */ nullptr,
            /* .is_host        = */ nullptr,
        },
        /* .device  = */ nullptr,
        /* .context = */ state,
    };
}

static enum ggml_status fake_event_synchronize(ggml_backend_dev_t, ggml_backend_event_t event) {
    return static_cast<fake_event_state *>(event->context)->completion_status;
}

static ggml_backend_device make_fake_device(fake_backend_state * state) {
    return {
        /* .iface   = */ {
            /* .get_name             = */ fake_device_name,
            /* .get_description      = */ fake_device_name,
            /* .get_memory           = */ nullptr,
            /* .get_type             = */ fake_device_type,
            /* .get_props            = */ nullptr,
            /* .init_backend         = */ nullptr,
            /* .get_buffer_type      = */ fake_device_buffer_type,
            /* .get_host_buffer_type = */ nullptr,
            /* .buffer_from_host_ptr = */ nullptr,
            /* .supports_op          = */ fake_device_supports_op,
            /* .supports_buft        = */ fake_device_supports_buft,
            /* .offload_op           = */ nullptr,
            /* .event_new            = */ nullptr,
            /* .event_free           = */ nullptr,
            /* .event_synchronize    = */ fake_event_synchronize,
        },
        /* .reg     = */ nullptr,
        /* .context = */ state,
    };
}

static ggml_backend make_fake_scheduler_backend(
        fake_backend_state * state, ggml_backend_device * device) {
    return {
        /* .guid    = */ {},
        /* .iface   = */ {
            /* .get_name           = */ fake_backend_name,
            /* .free               = */ nullptr,
            /* .set_tensor_async   = */ fake_set_tensor_async,
            /* .get_tensor_async   = */ fake_get_tensor_async,
            /* .set_tensor_2d_async= */ nullptr,
            /* .get_tensor_2d_async= */ nullptr,
            /* .cpy_tensor_async   = */ fake_copy_async,
            /* .synchronize        = */ fake_synchronize,
            /* .graph_plan_create  = */ nullptr,
            /* .graph_plan_free    = */ nullptr,
            /* .graph_plan_update  = */ nullptr,
            /* .graph_plan_compute = */ nullptr,
            /* .graph_compute       = */ fake_graph_compute,
            /* .event_record_status = */ fake_event_record_status,
            /* .event_wait_status   = */ fake_event_wait_status,
            /* .graph_optimize      = */ fake_graph_optimize,
        },
        /* .device  = */ device,
        /* .context = */ state,
    };
}

static const char * throwing_buft_name(ggml_backend_buffer_type_t) {
    throw std::runtime_error("injected buffer-name failure");
}

static ggml_backend_buffer_t throwing_buffer_alloc(ggml_backend_buffer_type_t, size_t) {
    throw std::bad_alloc();
}

static void * throwing_buffer_base(ggml_backend_buffer_t) {
    throw std::runtime_error("injected buffer-base failure");
}

static void throwing_buffer_set(
        ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) {
    throw ggml_backend_exception { GGML_STATUS_ALLOC_FAILED, 101 };
}

static void throwing_buffer_get(
        ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t) {
    throw std::runtime_error("injected buffer-read failure");
}

static void throwing_buffer_memset(
        ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t) {
    throw ggml_backend_exception { GGML_STATUS_DEVICE_LOST, 102 };
}

static void throwing_buffer_clear(ggml_backend_buffer_t, uint8_t) {
    throw ggml_backend_exception { GGML_STATUS_BACKEND_POISONED, 103 };
}

static void throwing_buffer_reset(ggml_backend_buffer_t) {
    throw ggml_backend_exception { GGML_STATUS_EXECUTION_FAILED, 114 };
}

static bool throwing_buffer_copy(
        ggml_backend_buffer_t, const ggml_tensor *, ggml_tensor *) {
    throw ggml_backend_exception { GGML_STATUS_DEVICE_LOST, 109 };
}

static enum ggml_status throwing_backend_synchronize(ggml_backend_t) {
    throw std::runtime_error("injected synchronize failure");
}

static const char * throwing_backend_name(ggml_backend_t) {
    throw std::runtime_error("injected backend-name failure");
}

static void throwing_backend_free(ggml_backend_t) {
    throw ggml_backend_exception { GGML_STATUS_EXECUTION_FAILED, 115 };
}

static enum ggml_status throwing_backend_compute(ggml_backend_t, ggml_cgraph *) {
    throw ggml_backend_exception { GGML_STATUS_DEVICE_LOST, 104 };
}

static const char * throwing_device_name(ggml_backend_dev_t) {
    throw std::runtime_error("injected device-name failure");
}

static void throwing_device_memory(ggml_backend_dev_t, size_t *, size_t *) {
    throw std::runtime_error("injected device-memory failure");
}

static enum ggml_backend_dev_type throwing_device_type(ggml_backend_dev_t) {
    throw std::runtime_error("injected device-type failure");
}

static void throwing_device_props(ggml_backend_dev_t, ggml_backend_dev_props *) {
    throw std::runtime_error("injected device-props failure");
}

static ggml_backend_t throwing_device_init(ggml_backend_dev_t, const char *) {
    throw std::bad_alloc();
}

static ggml_backend_buffer_type_t throwing_device_buft(ggml_backend_dev_t) {
    throw std::runtime_error("injected device-buft failure");
}

static bool throwing_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    throw std::runtime_error("injected supports-op failure");
}

static bool throwing_device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t) {
    throw std::runtime_error("injected supports-buft failure");
}

static ggml_backend_event_t throwing_event_new(ggml_backend_dev_t) {
    throw std::bad_alloc();
}

static void throwing_event_free(ggml_backend_dev_t, ggml_backend_event_t) {
    throw ggml_backend_exception { GGML_STATUS_EXECUTION_FAILED, 116 };
}

static enum ggml_status throwing_event_synchronize(ggml_backend_dev_t, ggml_backend_event_t) {
    throw ggml_backend_exception { GGML_STATUS_DEVICE_LOST, 105 };
}

static const char * throwing_reg_name(ggml_backend_reg_t) {
    throw std::runtime_error("injected registry-name failure");
}

static size_t throwing_reg_count(ggml_backend_reg_t) {
    throw std::bad_alloc();
}

static ggml_backend_dev_t throwing_reg_get(ggml_backend_reg_t, size_t) {
    throw std::runtime_error("injected registry-device failure");
}

static void * throwing_reg_proc(ggml_backend_reg_t, const char *) {
    throw std::runtime_error("injected registry-proc failure");
}

static void throwing_set_n_threads(ggml_backend_t, int) {
    throw ggml_backend_exception { GGML_STATUS_DEVICE_LOST, 112 };
}

static uint32_t throwing_pci_vendor_id(ggml_backend_dev_t) {
    throw std::runtime_error("injected PCI vendor query failure");
}

static void * callback_throwing_reg_proc(ggml_backend_reg_t, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return reinterpret_cast<void *>(throwing_set_n_threads);
    }
    if (std::strcmp(name, GGML_BACKEND_DEVICE_PCI_VENDOR_ID_PROC) == 0) {
        return reinterpret_cast<void *>(throwing_pci_vendor_id);
    }
    return nullptr;
}

static enum ggml_status throwing_memory_domains(
        ggml_backend_dev_t, ggml_backend_memory_domain_v1 *, uint32_t *) {
    throw ggml_backend_exception { GGML_STATUS_DEVICE_LOST, 106 };
}

static enum ggml_status throwing_memory_quote(
        const ggml_backend_memory_request_v1 *, uint32_t, ggml_backend_memory_quote_v1 *,
        ggml_backend_memory_claim_v1 *, uint32_t *) {
    throw std::bad_alloc();
}

static enum ggml_status throwing_memory_reserve(
        const ggml_backend_memory_request_v1 *, uint32_t,
        const ggml_backend_memory_quote_v1 *, ggml_backend_memory_claim_v1 *, uint32_t *) {
    throw std::runtime_error("injected reserve failure");
}

static enum ggml_status throwing_memory_stats(
        ggml_backend_dev_t, ggml_backend_t, ggml_backend_memory_stats_v1 *, uint32_t *) {
    throw ggml_backend_exception { GGML_STATUS_BACKEND_POISONED, 107 };
}

static void verify_common_noexcept_adapter(ggml_tensor * tensor) {
    assert(std::strcmp(ggml_backend_buft_name(nullptr), "unknown") == 0);
    assert(ggml_backend_buft_alloc_buffer(nullptr, 16) == nullptr);
    assert(ggml_backend_buft_get_alignment(nullptr) == 0);
    assert(ggml_backend_buffer_init_tensor(nullptr, tensor) == GGML_STATUS_FAILED);
    assert(ggml_backend_buffer_clear(nullptr, 0) == GGML_STATUS_FAILED);
    assert(ggml_backend_graph_compute_async(nullptr, nullptr) == GGML_STATUS_FAILED);
    assert(ggml_backend_event_synchronize(nullptr) == GGML_STATUS_FAILED);
    assert(std::strcmp(ggml_backend_dev_name(nullptr), "unknown") == 0);
    assert(std::strcmp(ggml_backend_reg_name(nullptr), "unknown") == 0);
    assert(ggml_backend_tensor_alloc(nullptr, tensor, nullptr) == GGML_STATUS_FAILED);

    ggml_backend_buffer_type buft = {
        {
            throwing_buft_name,
            throwing_buffer_alloc,
            fake_buffer_alignment,
            nullptr,
            nullptr,
            nullptr,
        },
        nullptr,
        nullptr,
    };
    assert(std::strcmp(ggml_backend_buft_name(&buft), "unknown") == 0);
    assert(ggml_backend_buft_alloc_buffer(&buft, 16) == nullptr);

    char storage[sizeof(float)] = {};
    ggml_backend_buffer buffer = {
        {
            nullptr,
            throwing_buffer_base,
            nullptr,
            throwing_buffer_memset,
            throwing_buffer_set,
            throwing_buffer_get,
            nullptr,
            nullptr,
            throwing_buffer_copy,
            throwing_buffer_clear,
            throwing_buffer_reset,
        },
        &buft,
        storage,
        sizeof(storage),
        GGML_BACKEND_BUFFER_USAGE_ANY,
    };
    tensor->buffer = &buffer;
    tensor->data = storage;
    float value = 1.0f;
    assert(ggml_backend_buffer_get_base(&buffer) == nullptr);
    assert(ggml_backend_tensor_set(tensor, &value, 0, sizeof(value)) == GGML_STATUS_ALLOC_FAILED);
    assert(ggml_backend_tensor_get(tensor, &value, 0, sizeof(value)) == GGML_STATUS_EXECUTION_FAILED);
    assert(ggml_backend_tensor_memset(tensor, 0, 0, sizeof(value)) == GGML_STATUS_DEVICE_LOST);
    assert(ggml_backend_buffer_clear(&buffer, 0) == GGML_STATUS_BACKEND_POISONED);
    assert(ggml_backend_buffer_reset_status(&buffer) == GGML_STATUS_EXECUTION_FAILED);
    ggml_backend_buffer_reset(&buffer);
    assert(ggml_backend_tensor_set(tensor, &value, SIZE_MAX, sizeof(value)) ==
           GGML_STATUS_FAILED);
    assert(ggml_backend_tensor_get(tensor, &value, SIZE_MAX, sizeof(value)) ==
           GGML_STATUS_FAILED);
    assert(ggml_backend_tensor_set_async(
               nullptr, tensor, &value, 0, sizeof(value)) ==
           GGML_STATUS_FAILED);
    ggml_tensor copy_src = *tensor;
    ggml_tensor copy_dst = *tensor;
    assert(ggml_backend_tensor_copy(&copy_src, &copy_dst) == GGML_STATUS_DEVICE_LOST);
    ggml_tensor invalid_view = *tensor;
    invalid_view.buffer = nullptr;
    invalid_view.data = nullptr;
    invalid_view.view_src = nullptr;
    assert(ggml_backend_view_init(&invalid_view) == GGML_STATUS_FAILED);

    ggml_backend_device device = {
        {
            throwing_device_name,
            throwing_device_name,
            throwing_device_memory,
            throwing_device_type,
            throwing_device_props,
            throwing_device_init,
            throwing_device_buft,
            throwing_device_buft,
            nullptr,
            throwing_device_supports_op,
            throwing_device_supports_buft,
            nullptr,
            throwing_event_new,
            throwing_event_free,
            throwing_event_synchronize,
        },
        nullptr,
        nullptr,
    };
    size_t free = 1;
    size_t total = 1;
    ggml_backend_dev_props props = {};
    assert(std::strcmp(ggml_backend_dev_name(&device), "unknown") == 0);
    ggml_backend_dev_memory(&device, &free, &total);
    assert(free == 0 && total == 0);
    assert(ggml_backend_dev_type(&device) == GGML_BACKEND_DEVICE_TYPE_UNKNOWN);
    ggml_backend_dev_get_props(&device, &props);
    assert(props.name == nullptr && props.memory_total == 0);
    assert(ggml_backend_dev_init(&device, nullptr) == nullptr);
    assert(ggml_backend_dev_buffer_type(&device) == nullptr);
    assert(!ggml_backend_dev_supports_op(&device, tensor));
    assert(!ggml_backend_dev_supports_buft(&device, &buft));
    assert(ggml_backend_event_new(&device) == nullptr);

    fake_event_state event_state;
    ggml_backend_event event = {&device, &event_state};
    assert(ggml_backend_event_synchronize(&event) == GGML_STATUS_DEVICE_LOST);
    assert(ggml_backend_event_free_status(&event) == GGML_STATUS_EXECUTION_FAILED);

    ggml_backend backend = {
        {},
        {
            throwing_backend_name,
            throwing_backend_free,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            throwing_backend_synchronize,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            throwing_backend_compute,
            nullptr,
            nullptr,
            nullptr,
        },
        &device,
        nullptr,
    };
    assert(std::strcmp(ggml_backend_name(&backend), "unknown") == 0);
    assert(ggml_backend_synchronize(&backend) == GGML_STATUS_EXECUTION_FAILED);
    ggml_cgraph graph = {};
    assert(ggml_backend_graph_compute_async(&backend, &graph) == GGML_STATUS_DEVICE_LOST);

    ggml_backend_reg callback_reg = {
        GGML_BACKEND_API_VERSION,
        {nullptr, nullptr, nullptr, callback_throwing_reg_proc},
        nullptr,
    };
    device.reg = &callback_reg;
    assert(ggml_backend_set_n_threads_if_supported(&backend, 4) == GGML_STATUS_DEVICE_LOST);
    assert(ggml_backend_dev_pci_vendor_id(&device) == 0);
    device.reg = nullptr;

    ggml_backend_reg reg = {
        GGML_BACKEND_API_VERSION,
        {throwing_reg_name, throwing_reg_count, throwing_reg_get, throwing_reg_proc},
        nullptr,
    };
    assert(std::strcmp(ggml_backend_reg_name(&reg), "unknown") == 0);
    assert(ggml_backend_reg_dev_count(&reg) == 0);
    assert(ggml_backend_reg_dev_get(&reg, 0) == nullptr);
    assert(ggml_backend_reg_get_proc_address(&reg, "injected") == nullptr);

    ggml_backend_memory_api_v1 memory_api = {
        sizeof(ggml_backend_memory_api_v1),
        GGML_BACKEND_MEMORY_ABI_V1,
        0,
        throwing_memory_domains,
        throwing_memory_quote,
        throwing_memory_reserve,
        throwing_memory_stats,
        nullptr,
        nullptr,
    };
    uint32_t count = 0;
    ggml_backend_memory_quote_v1 quote = {};
    quote.struct_size = sizeof(quote);
    assert(ggml_backend_memory_api_get_domains_v1(
               &memory_api, &device, nullptr, &count) == GGML_STATUS_DEVICE_LOST);
    assert(ggml_backend_memory_api_quote_v1(
               &memory_api, nullptr, 0, &quote, nullptr, &count) == GGML_STATUS_ALLOC_FAILED);
    assert(ggml_backend_memory_api_reserve_private_v1(
               &memory_api, nullptr, 0, &quote, nullptr, &count) == GGML_STATUS_EXECUTION_FAILED);
    assert(ggml_backend_memory_api_get_stats_v1(
               &memory_api, &device, &backend, nullptr, &count) == GGML_STATUS_BACKEND_POISONED);
    assert(ggml_backend_free_status(&backend) == GGML_STATUS_EXECUTION_FAILED);

    tensor->buffer = nullptr;
    tensor->data = nullptr;
}

int main() {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);
    ggml_init_params overflowing_params = {
        /* .mem_size   = */ SIZE_MAX,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    assert(ggml_try_init(overflowing_params) == nullptr);
    ggml_hash_set impossible_hash = {};
    assert(!ggml_hash_set_try_new(SIZE_MAX, &impossible_hash));
    assert(impossible_hash.keys == nullptr && impossible_hash.used == nullptr);
    ggml_tensor * boundary_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    // This executable proves the shared adapter, not any native provider SDK.
    // CUDA/HIP/Vulkan compilation and real fault injection remain separate
    // target-qualified gates.
    verify_common_noexcept_adapter(boundary_tensor);
    ggml_tensor * value = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * increment = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    for (int i = 0; i < 65; ++i) {
        value = ggml_add(ctx, value, increment);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, value);
    assert(graph->n_nodes == 65);

    ggml_backend_reg reg = {};
    ggml_backend_reg native_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ {
            /* .get_name         = */ nullptr,
            /* .get_device_count = */ nullptr,
            /* .get_device       = */ nullptr,
            /* .get_proc_address = */ fake_reg_get_proc_address,
        },
        /* .context     = */ nullptr,
    };
    ggml_backend_device device = {};
    device.reg = &reg;
    fake_backend_state state;
    ggml_backend backend = {};
    backend.device = &device;
    backend.context = &state;
    backend.iface.graph_compute = fake_graph_compute;
    backend.iface.synchronize = fake_synchronize;

    // API v3 rejects a backend that did not register typed event callbacks;
    // legacy void callbacks cannot silently convert this boundary to success.
    ggml_backend_event missing_callback_event = {&device, nullptr};
    assert(ggml_backend_event_record_status(&missing_callback_event, &backend) == GGML_STATUS_FAILED);
    assert(ggml_backend_event_wait_status(&backend, &missing_callback_event) == GGML_STATUS_FAILED);

    // Callback-free compute stays one async submission followed by the public
    // synchronous wrapper's one synchronization.
    struct ggml_backend_graph_cancel_capability capability = {};
    assert(ggml_backend_graph_compute(&backend, graph) == GGML_STATUS_SUCCESS);
    assert(state.graph_sizes == std::vector<int>({65}));
    assert(state.synchronize_calls == 1);
    assert(!state.pending);

    // A backend that registers the native hook keeps the full graph intact.
    // Armed-but-false work is one submission plus one terminal drain, and the
    // callback is scoped to exactly that compute call.
    device.reg = &native_reg;
    reset(state);
    abort_probe native_false_probe = {&state, 0, 100};
    assert(ggml_backend_graph_compute_with_abort(
               &backend, graph, native_abort_after_polls, &native_false_probe, &capability) == GGML_STATUS_SUCCESS);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    assert(state.graph_sizes == std::vector<int>({65}));
    assert(state.synchronize_calls == 1);
    assert(native_false_probe.polls == 3);
    assert(state.abort_set_calls == 1);
    assert(state.abort_clear_calls == 1);
    assert(state.abort_callback == nullptr);
    assert(!state.pending);

    // Capability is reported by the concrete execution path, not inferred
    // from the backend registration. A warmed monolithic replay therefore
    // remains native while exposing its graph-completion observation boundary.
    reset(state);
    state.native_granularity = GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_GRAPH_COMPLETION;
    abort_probe native_completion_probe = {&state, 0, 100};
    assert(ggml_backend_graph_compute_with_abort(
               &backend, graph, native_abort_after_polls, &native_completion_probe, &capability) == GGML_STATUS_SUCCESS);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_GRAPH_COMPLETION);
    assert(state.graph_sizes == std::vector<int>({65}));
    assert(state.synchronize_calls == 1);
    assert(state.abort_callback == nullptr);
    assert(!state.pending);

    // Native cancellation drains accepted work, clears the borrowed callback,
    // and leaves the backend reusable by the next compute.
    reset(state);
    abort_probe native_cancel_probe = {&state, 0, 2};
    assert(ggml_backend_graph_compute_with_abort(
               &backend, graph, native_abort_after_polls, &native_cancel_probe, &capability) == GGML_STATUS_ABORTED);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    assert(state.graph_sizes == std::vector<int>({65}));
    assert(state.synchronize_calls == 1);
    assert(native_cancel_probe.polls == 2);
    assert(state.abort_callback == nullptr);
    assert(!state.pending);

    reset(state);
    abort_probe native_reuse_probe = {&state, 0, 100};
    assert(ggml_backend_graph_compute_with_abort(
               &backend, graph, native_abort_after_polls, &native_reuse_probe, &capability) == GGML_STATUS_SUCCESS);
    assert(state.graph_sizes == std::vector<int>({65}));
    assert(!state.pending);

    // A terminal device failure remains authoritative when cancellation races
    // the same native submission.
    reset(state);
    state.completion_status = GGML_STATUS_DEVICE_LOST;
    abort_probe native_cancel_vs_device_loss = {&state, 0, 2};
    assert(ggml_backend_graph_compute_with_abort(
               &backend, graph, native_abort_after_polls, &native_cancel_vs_device_loss, &capability) == GGML_STATUS_DEVICE_LOST);
    assert(state.abort_callback == nullptr);
    assert(!state.pending);
    device.reg = &reg;

    // Submission acknowledges queue acceptance only. Completion determines the
    // terminal status and drains pending work even when submission also failed.
    reset(state);
    state.submit_status = GGML_STATUS_FAILED;
    assert(ggml_backend_graph_compute(&backend, graph) == GGML_STATUS_FAILED);
    assert(state.synchronize_calls == 1);
    assert(!state.pending);

    // A later cancellation result cannot erase a concrete submission failure.
    // This same merge rule is used while a scheduler drains multiple backends.
    reset(state);
    state.submit_status = GGML_STATUS_DEVICE_LOST;
    state.completion_status = GGML_STATUS_ABORTED;
    assert(ggml_backend_graph_compute(&backend, graph) == GGML_STATUS_DEVICE_LOST);
    assert(state.synchronize_calls == 1);
    assert(!state.pending);

    // Native scheduler cancellation keeps the scheduler's original async split
    // submission path. A one-backend graph therefore performs one full-graph
    // submission and one final drain instead of synchronizing inside the split
    // and then synchronizing the scheduler again.
    fake_backend_state native_sched_state = {
        /* .name = */ "fake-native-scheduler",
        /* .type = */ GGML_BACKEND_DEVICE_TYPE_CPU,
    };
    ggml_backend_buffer_type native_sched_buft = make_fake_buffer_type(&native_sched_state);
    native_sched_state.buft = &native_sched_buft;
    ggml_backend_device native_sched_device = make_fake_device(&native_sched_state);
    native_sched_device.reg = &native_reg;
    native_sched_buft.device = &native_sched_device;
    ggml_backend native_sched_backend = make_fake_scheduler_backend(&native_sched_state, &native_sched_device);
    ggml_backend_t native_sched_backends[] = {&native_sched_backend};
    ggml_backend_buffer_type_t native_sched_bufts[] = {&native_sched_buft};
    ggml_backend_sched_t native_scheduler = ggml_backend_sched_new(
        native_sched_backends, native_sched_bufts, 1, 128, false, false);
    abort_probe native_scheduler_probe = {&native_sched_state, 0, 100};
    assert(ggml_backend_sched_graph_compute_with_abort(
               native_scheduler,
               graph,
               native_abort_after_polls,
               &native_scheduler_probe,
               &capability) == GGML_STATUS_SUCCESS);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    assert(native_sched_state.graph_sizes == std::vector<int>({65}));
    assert(native_sched_state.synchronize_calls == 1);
    assert(native_sched_state.abort_set_calls == 1);
    assert(native_sched_state.abort_clear_calls == 1);
    assert(native_sched_state.abort_callback == nullptr);
    assert(!native_sched_state.pending);

    reset(native_sched_state);
    abort_probe native_scheduler_cancel_probe = {&native_sched_state, 0, 0};
    assert(ggml_backend_sched_graph_compute_with_abort(
               native_scheduler,
               graph,
               native_abort_after_submission,
               &native_scheduler_cancel_probe,
               &capability) == GGML_STATUS_ABORTED);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    assert(native_sched_state.graph_sizes == std::vector<int>({65}));
    assert(native_sched_state.synchronize_calls == 1);
    assert(native_sched_state.abort_callback == nullptr);
    assert(!native_sched_state.pending);

    reset(native_sched_state);
    abort_probe native_scheduler_reuse_probe = {&native_sched_state, 0, 100};
    assert(ggml_backend_sched_graph_compute_with_abort(
               native_scheduler,
               graph,
               native_abort_after_polls,
               &native_scheduler_reuse_probe,
               &capability) == GGML_STATUS_SUCCESS);
    assert(native_sched_state.graph_sizes == std::vector<int>({65}));
    assert(native_sched_state.synchronize_calls == 1);
    assert(!native_sched_state.pending);

    reset(native_sched_state);
    native_sched_state.native_granularity = GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_GRAPH_COMPLETION;
    abort_probe native_scheduler_completion_probe = {&native_sched_state, 0, 100};
    assert(ggml_backend_sched_graph_compute_with_abort(
               native_scheduler,
               graph,
               native_abort_after_polls,
               &native_scheduler_completion_probe,
               &capability) == GGML_STATUS_SUCCESS);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_GRAPH_COMPLETION);
    assert(native_sched_state.graph_sizes == std::vector<int>({65}));
    assert(native_sched_state.synchronize_calls == 1);
    assert(!native_sched_state.pending);
    ggml_backend_sched_free(native_scheduler);

    // Scheduler drain order must not let a later backend's cancel result hide
    // an earlier backend's concrete failure.
    fake_backend_state native_error_state = {
        /* .name = */ "fake-native-error",
        /* .type = */ GGML_BACKEND_DEVICE_TYPE_GPU,
    };
    ggml_backend_buffer_type native_error_buft = make_fake_buffer_type(&native_error_state);
    native_error_state.buft = &native_error_buft;
    ggml_backend_device native_error_device = make_fake_device(&native_error_state);
    native_error_device.reg = &native_reg;
    native_error_buft.device = &native_error_device;
    ggml_backend native_error_backend = make_fake_scheduler_backend(&native_error_state, &native_error_device);

    fake_backend_state native_abort_state = {
        /* .name = */ "fake-native-abort",
        /* .type = */ GGML_BACKEND_DEVICE_TYPE_CPU,
    };
    ggml_backend_buffer_type native_abort_buft = make_fake_buffer_type(&native_abort_state);
    native_abort_state.buft = &native_abort_buft;
    ggml_backend_device native_abort_device = make_fake_device(&native_abort_state);
    native_abort_device.reg = &native_reg;
    native_abort_buft.device = &native_abort_device;
    ggml_backend native_abort_backend = make_fake_scheduler_backend(&native_abort_state, &native_abort_device);

    ggml_backend_t native_terminal_backends[] = {&native_error_backend, &native_abort_backend};
    ggml_backend_buffer_type_t native_terminal_bufts[] = {&native_error_buft, &native_abort_buft};
    ggml_backend_sched_t native_terminal_scheduler = ggml_backend_sched_new(
        native_terminal_backends, native_terminal_bufts, 2, 128, false, false);
    native_error_state.completion_status = GGML_STATUS_DEVICE_LOST;
    native_abort_state.completion_status = GGML_STATUS_ABORTED;
    assert(ggml_backend_sched_synchronize(native_terminal_scheduler) == GGML_STATUS_DEVICE_LOST);
    assert(native_error_state.synchronize_calls == 1);
    assert(native_abort_state.synchronize_calls == 1);
    ggml_backend_sched_free(native_terminal_scheduler);

#ifdef GGML_USE_BLAS
    // BLAS is an optional scheduler backend in OpenASR. When built, it must
    // expose the same native contract or merely enabling it would downgrade an
    // otherwise native GPU+CPU scheduler to segmented synchronization.
    ggml_backend_t blas_backend = ggml_backend_blas_init();
    assert(blas_backend != nullptr);
    ggml_tensor inert_node = *graph->nodes[0];
    inert_node.flags &= ~GGML_TENSOR_FLAG_COMPUTE;
    ggml_tensor * inert_nodes[] = {&inert_node};
    struct ggml_cgraph inert_graph = ggml_graph_view(graph, 0, 0);
    inert_graph.n_nodes = 1;
    inert_graph.nodes = inert_nodes;
    abort_probe blas_false_probe = {&state, 0, 100};
    assert(ggml_backend_graph_compute_with_abort(
               blas_backend,
               &inert_graph,
               native_abort_after_polls,
               &blas_false_probe,
               &capability) == GGML_STATUS_SUCCESS);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    assert(blas_false_probe.polls == 3);

    abort_probe blas_cancel_probe = {&state, 0, 2};
    assert(ggml_backend_graph_compute_with_abort(
               blas_backend,
               &inert_graph,
               native_abort_after_polls,
               &blas_cancel_probe,
               &capability) == GGML_STATUS_ABORTED);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    assert(blas_cancel_probe.polls == 2);

    abort_probe blas_reuse_probe = {&state, 0, 100};
    assert(ggml_backend_graph_compute_with_abort(
               blas_backend,
               &inert_graph,
               native_abort_after_polls,
               &blas_reuse_probe,
               &capability) == GGML_STATUS_SUCCESS);
    assert(blas_reuse_probe.polls == 3);
    ggml_backend_free(blas_backend);
#endif

    reset(state);
    state.completion_status = GGML_STATUS_EXECUTION_FAILED;
    assert(ggml_backend_graph_compute(&backend, graph) == GGML_STATUS_EXECUTION_FAILED);
    assert(state.synchronize_calls == 1);
    assert(!state.pending);

    // A real completion failure has priority over an already observed abort.
    reset(state);
    state.completion_status = GGML_STATUS_DEVICE_LOST;
    abort_probe pre_cancel_failure = {&state, 0, 1};
    assert(ggml_backend_graph_compute_with_abort(
               &backend, graph, abort_after_polls, &pre_cancel_failure, &capability) == GGML_STATUS_DEVICE_LOST);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_DISABLED);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_NONE);
    assert(state.synchronize_calls == 1);
    assert(!state.pending);

    // A poisoned completion prevents this call from reporting success; resetting
    // the backend state models explicit recreation before a later successful use.
    reset(state);
    state.completion_status = GGML_STATUS_BACKEND_POISONED;
    assert(ggml_backend_graph_compute(&backend, graph) == GGML_STATUS_BACKEND_POISONED);
    assert(!state.pending);
    reset(state);
    assert(ggml_backend_graph_compute(&backend, graph) == GGML_STATUS_SUCCESS);
    assert(!state.pending);

    // Armed but false compute uses bounded graph views on a backend without a
    // native hook and still completes normally.
    reset(state);
    abort_probe false_probe = {&state, 0, 100};
    assert(ggml_backend_graph_compute_with_abort(
               &backend, graph, abort_after_polls, &false_probe, &capability) == GGML_STATUS_SUCCESS);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_SEGMENTED);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    assert(state.graph_sizes == std::vector<int>({32, 32, 1}));
    assert(std::all_of(state.graph_sizes.begin(), state.graph_sizes.end(), [](int n) {
        return n <= GGML_BACKEND_GRAPH_CANCEL_SEGMENT_NODES;
    }));
    assert(false_probe.polls == 4);
    assert(state.synchronize_calls == 3);
    assert(!state.pending);

    // The second poll happens after the first synchronized segment: this is a
    // deterministic mid-graph flip, not a pre-start cancellation.
    reset(state);
    abort_probe cancel_probe = {&state, 0, 2};
    assert(ggml_backend_graph_compute_with_abort(
               &backend, graph, abort_after_polls, &cancel_probe, &capability) == GGML_STATUS_ABORTED);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_SEGMENTED);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    assert(state.graph_sizes == std::vector<int>({32}));
    assert(cancel_probe.polls == 2);
    assert(state.synchronize_calls == 1);
    assert(!state.pending);

    // A segment completion error wins over the cancellation poll after that
    // segment and no later segment is submitted.
    reset(state);
    state.completion_status = GGML_STATUS_EXECUTION_FAILED;
    abort_probe segment_failure = {&state, 0, 2};
    assert(ggml_backend_graph_compute_with_abort(
               &backend, graph, abort_after_polls, &segment_failure, &capability) == GGML_STATUS_EXECUTION_FAILED);
    assert(state.graph_sizes == std::vector<int>({32}));
    assert(!state.pending);

    // Force a real scheduler split across two incompatible fake buffer types.
    // The destination copy flips cancellation while it is in flight; the
    // scheduler's post-copy checkpoint must abort before submitting the GPU
    // split, and the public API must drain that pending copy before returning.
    fake_backend_state gpu_state = {
        /* .name = */ "fake-gpu",
        /* .type = */ GGML_BACKEND_DEVICE_TYPE_GPU,
    };
    fake_backend_state cpu_state = {
        /* .name = */ "fake-cpu",
        /* .type = */ GGML_BACKEND_DEVICE_TYPE_CPU,
    };
    ggml_backend_buffer_type gpu_buft = make_fake_buffer_type(&gpu_state);
    ggml_backend_buffer_type cpu_buft = make_fake_buffer_type(&cpu_state);
    gpu_state.buft = &gpu_buft;
    cpu_state.buft = &cpu_buft;
    ggml_backend_device gpu_device = make_fake_device(&gpu_state);
    ggml_backend_device cpu_device = make_fake_device(&cpu_state);
    gpu_buft.device = &gpu_device;
    cpu_buft.device = &cpu_device;

    fake_event_state event_failure = {GGML_STATUS_EXECUTION_FAILED};
    ggml_backend_event fake_event = {&gpu_device, &event_failure};
    assert(ggml_backend_event_synchronize(&fake_event) == GGML_STATUS_EXECUTION_FAILED);

    ggml_backend gpu_backend = make_fake_scheduler_backend(&gpu_state, &gpu_device);
    ggml_backend cpu_backend = make_fake_scheduler_backend(&cpu_state, &cpu_device);

    // Host-capacity and malformed-provider metadata are ordinary typed
    // failures at the common allocator seam; none may terminate the process.
    ggml_tensor * oversized_tensor =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    ggml_backend_buffer_t tiny_buffer = fake_buffer_alloc(&cpu_buft, 16);
    ggml_tallocr tiny_allocator = ggml_tallocr_new(tiny_buffer);
    assert(ggml_tallocr_alloc(&tiny_allocator, oversized_tensor) ==
           GGML_STATUS_ALLOC_FAILED);
    assert(oversized_tensor->buffer == nullptr &&
           oversized_tensor->data == nullptr);

    ggml_backend_buffer null_base_buffer = *tiny_buffer;
    null_base_buffer.iface.get_base = null_buffer_base;
    ggml_tallocr null_base_allocator = ggml_tallocr_new(&null_base_buffer);
    assert(ggml_tallocr_alloc(&null_base_allocator, oversized_tensor) ==
           GGML_STATUS_FAILED);
    assert(ggml_backend_buffer_free_status(tiny_buffer) ==
           GGML_STATUS_SUCCESS);

    // A provider failure before physical release is observable even though the
    // opaque outer handle is consumed.  The caller must keep its physical
    // accounting quarantined; the test retains the allocation only to clean up
    // the injected failure without leaking the test process.
    ggml_backend_buffer_t pre_release_failure_buffer =
        fake_buffer_alloc(&cpu_buft, 16);
    void * pre_release_failure_data = pre_release_failure_buffer->context;
    cpu_state.throw_buffer_free_before_release = true;
    assert(ggml_backend_buffer_free_status(pre_release_failure_buffer) ==
           GGML_STATUS_EXECUTION_FAILED);
    cpu_state.throw_buffer_free_before_release = false;
    ggml_aligned_free(pre_release_failure_data, 16);

    ggml_backend_buffer_type invalid_alignment_buft = cpu_buft;
    invalid_alignment_buft.iface.get_alignment = invalid_buffer_alignment;
    assert(ggml_gallocr_new(&invalid_alignment_buft) == nullptr);
    assert(ggml_gallocr_new_n(nullptr, 0) == nullptr);

    ggml_backend_t only_cpu_backend[] = {&cpu_backend};
    ggml_backend_buffer_type_t only_cpu_buft[] = {&cpu_buft};
    assert(ggml_backend_sched_new(
               only_cpu_backend, only_cpu_buft, 1, SIZE_MAX, false,
               false) == nullptr);

    ggml_context * init_failure_ctx = ggml_init(params);
    ggml_tensor * init_failure_tensor =
        ggml_new_tensor_1d(init_failure_ctx, GGML_TYPE_F32, 8);
    const int frees_before_init_failure = cpu_state.buffer_free_calls;
    cpu_state.throw_init_tensor = true;
    assert(ggml_backend_alloc_ctx_tensors(
               init_failure_ctx, &cpu_backend) == nullptr);
    assert(init_failure_tensor->buffer == nullptr &&
           init_failure_tensor->data == nullptr);
    assert(cpu_state.buffer_free_calls > frees_before_init_failure);
    cpu_state.throw_init_tensor = false;
    ggml_backend_buffer_t init_retry_buffer =
        ggml_backend_alloc_ctx_tensors(init_failure_ctx, &cpu_backend);
    assert(init_retry_buffer != nullptr);
    assert(ggml_backend_buffer_free_status(init_retry_buffer) ==
           GGML_STATUS_SUCCESS);
    ggml_free(init_failure_ctx);

    // Direct reserve is replacement-first: a failed larger allocation leaves
    // the previous graph binding alive. The typed measure commit separately
    // exposes a retired-buffer release failure instead of reporting success.
    ggml_context * reserve_ctx = ggml_init(params);
    ggml_tensor * reserve_small_input =
        ggml_new_tensor_1d(reserve_ctx, GGML_TYPE_F32, 4);
    ggml_tensor * reserve_small_node =
        ggml_sqr(reserve_ctx, reserve_small_input);
    ggml_cgraph * reserve_small_graph =
        ggml_new_graph_custom(reserve_ctx, 16, false);
    ggml_build_forward_expand(reserve_small_graph, reserve_small_node);
    ggml_tensor * reserve_large_input =
        ggml_new_tensor_1d(reserve_ctx, GGML_TYPE_F32, 1024);
    ggml_tensor * reserve_large_node =
        ggml_sqr(reserve_ctx, reserve_large_input);
    ggml_cgraph * reserve_large_graph =
        ggml_new_graph_custom(reserve_ctx, 16, false);
    ggml_build_forward_expand(reserve_large_graph, reserve_large_node);

    ggml_gallocr_t reserve_allocator = ggml_gallocr_new(&cpu_buft);
    assert(reserve_allocator != nullptr);
    assert(ggml_gallocr_measure_n_v1(
        reserve_allocator, reserve_small_graph, nullptr, nullptr));
    uint32_t allocator_commit_flags = 0;
    assert(ggml_gallocr_measure_commit_v2(
               reserve_allocator, &allocator_commit_flags) ==
           GGML_STATUS_SUCCESS);
    assert((allocator_commit_flags &
            GGML_GALLOCR_MEASURE_COMMIT_MAY_HAVE_MUTATED) != 0);
    assert((allocator_commit_flags &
            GGML_GALLOCR_MEASURE_COMMIT_RELEASE_UNPROVEN) == 0);
    assert(ggml_gallocr_alloc_graph_v2(
               reserve_allocator, reserve_small_graph) ==
           GGML_STATUS_SUCCESS);
    cpu_state.throw_buffer_reset = true;
    ggml_backend_buffer_t reset_failure_buffer = reserve_small_node->buffer;
    assert(ggml_backend_buffer_reset_status(reset_failure_buffer) ==
           GGML_STATUS_EXECUTION_FAILED);
    // The legacy void wrapper remains source-compatible but cannot leak the
    // provider exception across the public C boundary.
    ggml_backend_buffer_reset(reset_failure_buffer);
    assert(ggml_gallocr_alloc_graph_v2(
               reserve_allocator, reserve_small_graph) ==
           GGML_STATUS_EXECUTION_FAILED);
    assert(reserve_small_node->buffer == nullptr &&
           reserve_small_node->data == nullptr);
    cpu_state.throw_buffer_reset = false;
    assert(ggml_gallocr_alloc_graph_v2(
               reserve_allocator, reserve_small_graph) ==
           GGML_STATUS_SUCCESS);
    ggml_backend_buffer_t prior_small_binding = reserve_small_node->buffer;
    void * prior_small_data = reserve_small_node->data;
    cpu_state.fail_buffer_alloc = true;
    assert(!ggml_gallocr_reserve(
        reserve_allocator, reserve_large_graph));
    assert(reserve_small_node->buffer == prior_small_binding &&
           reserve_small_node->data == prior_small_data);
    cpu_state.fail_buffer_alloc = false;

    ggml_gallocr_detach_graph_tensors_v1(
        reserve_allocator, reserve_small_graph);
    assert(ggml_gallocr_measure_n_v1(
        reserve_allocator, reserve_large_graph, nullptr, nullptr));
    cpu_state.throw_buffer_free = true;
    allocator_commit_flags = 0;
    assert(ggml_gallocr_measure_commit_v2(
               reserve_allocator, &allocator_commit_flags) ==
           GGML_STATUS_EXECUTION_FAILED);
    assert((allocator_commit_flags &
            GGML_GALLOCR_MEASURE_COMMIT_MAY_HAVE_MUTATED) != 0);
    assert((allocator_commit_flags &
            GGML_GALLOCR_MEASURE_COMMIT_RELEASE_UNPROVEN) != 0);
    cpu_state.throw_buffer_free = false;
    assert(ggml_gallocr_free_status(reserve_allocator) ==
           GGML_STATUS_SUCCESS);
    ggml_free(reserve_ctx);

    uint8_t src_bytes[4] = {1, 2, 3, 4};
    uint8_t dst_bytes[4] = {0xa5, 0xa5, 0xa5, 0xa5};
    ggml_tensor * src_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 4);
    ggml_tensor * dst_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 4);
    src_tensor->buffer = fake_buffer_alloc(&gpu_buft, 4);
    dst_tensor->buffer = fake_buffer_alloc(&cpu_buft, 4);
    src_tensor->data = ggml_backend_buffer_get_base(src_tensor->buffer);
    dst_tensor->data = ggml_backend_buffer_get_base(dst_tensor->buffer);
    std::memcpy(src_tensor->data, src_bytes, 4);
    std::memcpy(dst_tensor->data, dst_bytes, 4);

    // Fallback drains both endpoints. It cannot read the source or mutate the
    // destination unless both terminal statuses were successful.
    cpu_backend.iface.cpy_tensor_async = nullptr;
    reset(gpu_state); reset(cpu_state);
    gpu_state.completion_status = GGML_STATUS_EXECUTION_FAILED;
    assert(ggml_backend_tensor_copy_async(&gpu_backend, &cpu_backend, src_tensor, dst_tensor) == GGML_STATUS_EXECUTION_FAILED);
    assert(std::memcmp(dst_tensor->data, dst_bytes, 4) == 0);
    reset(gpu_state); reset(cpu_state);
    cpu_state.completion_status = GGML_STATUS_EXECUTION_FAILED;
    assert(ggml_backend_tensor_copy_async(&gpu_backend, &cpu_backend, src_tensor, dst_tensor) == GGML_STATUS_EXECUTION_FAILED);
    assert(std::memcmp(dst_tensor->data, dst_bytes, 4) == 0);
    reset(gpu_state); reset(cpu_state);
    gpu_state.completion_status = GGML_STATUS_EXECUTION_FAILED;
    cpu_state.completion_status = GGML_STATUS_DEVICE_LOST;
    assert(ggml_backend_tensor_copy_async(&gpu_backend, &cpu_backend, src_tensor, dst_tensor) == GGML_STATUS_DEVICE_LOST);
    assert(std::memcmp(dst_tensor->data, dst_bytes, 4) == 0);

    cpu_backend.iface.cpy_tensor_async = fake_copy_async;

    // Transfer callbacks are exercised through the public backend API. Accepted
    // submissions defer their write until synchronize, so a completion fault
    // cannot expose successful readback or mutate the destination.
    const uint8_t write_bytes[4] = {9, 9, 9, 9};
    reset(gpu_state);
    gpu_state.transfer_completion_status = GGML_STATUS_EXECUTION_FAILED;
    assert(ggml_backend_tensor_set_async(&gpu_backend, src_tensor, write_bytes, 0, 4) == GGML_STATUS_SUCCESS);
    assert(ggml_backend_synchronize(&gpu_backend) == GGML_STATUS_EXECUTION_FAILED);
    assert(std::memcmp(src_tensor->data, src_bytes, 4) == 0);

    reset(gpu_state);
    uint8_t readback[4] = {0xa5, 0xa5, 0xa5, 0xa5};
    gpu_state.transfer_completion_status = GGML_STATUS_EXECUTION_FAILED;
    assert(ggml_backend_tensor_get_async(&gpu_backend, src_tensor, readback, 0, 4) == GGML_STATUS_SUCCESS);
    assert(ggml_backend_synchronize(&gpu_backend) == GGML_STATUS_EXECUTION_FAILED);
    assert(std::memcmp(readback, dst_bytes, 4) == 0);

    reset(gpu_state); reset(cpu_state);
    cpu_state.transfer_completion_status = GGML_STATUS_EXECUTION_FAILED;
    assert(ggml_backend_tensor_copy_async(&gpu_backend, &cpu_backend, src_tensor, dst_tensor) == GGML_STATUS_SUCCESS);
    assert(ggml_backend_synchronize(&cpu_backend) == GGML_STATUS_EXECUTION_FAILED);
    assert(std::memcmp(dst_tensor->data, dst_bytes, 4) == 0);

    reset(gpu_state); reset(cpu_state);
    cpu_state.transfer_completion_status = GGML_STATUS_DEVICE_LOST;
    assert(ggml_backend_tensor_copy_async(&gpu_backend, &cpu_backend, src_tensor, dst_tensor) == GGML_STATUS_SUCCESS);
    assert(ggml_backend_synchronize(&cpu_backend) == GGML_STATUS_DEVICE_LOST);
    assert(std::memcmp(dst_tensor->data, dst_bytes, 4) == 0);
    assert(ggml_backend_tensor_copy_async(&gpu_backend, &cpu_backend, src_tensor, dst_tensor) == GGML_STATUS_BACKEND_POISONED);

    ggml_backend_buffer_free(src_tensor->buffer);
    ggml_backend_buffer_free(dst_tensor->buffer);
    reset(gpu_state); reset(cpu_state);
    // Typed record/wait failures are scheduler-visible terminal boundaries. This
    // models GPU completion seams without requiring a platform SDK at test time.
    gpu_state.event_record_status = GGML_STATUS_EXECUTION_FAILED;
    assert(ggml_backend_event_record_status(&fake_event, &gpu_backend) == GGML_STATUS_EXECUTION_FAILED);
    gpu_state.event_record_status = GGML_STATUS_SUCCESS;
    gpu_state.event_wait_status = GGML_STATUS_DEVICE_LOST;
    assert(ggml_backend_event_wait_status(&gpu_backend, &fake_event) == GGML_STATUS_DEVICE_LOST);
    gpu_state.event_wait_status = GGML_STATUS_SUCCESS;

    // Device backends expose their platform-specific fault source through this
    // common seam: Metal record failure, CANN stream reset, WebGPU queue loss,
    // and Vulkan device loss all report the first terminal classification; a
    // recreated backend is required before a later submission can succeed.
    const enum ggml_status device_loss_sources[] = {
        GGML_STATUS_EXECUTION_FAILED, // Metal command/event failure
        GGML_STATUS_DEVICE_LOST,      // CANN device reset
        GGML_STATUS_DEVICE_LOST,      // WebGPU queue/device loss
        GGML_STATUS_DEVICE_LOST,      // Vulkan device loss
    };
    for (enum ggml_status first_fault : device_loss_sources) {
        reset(gpu_state);
        gpu_state.completion_status = first_fault;
        assert(ggml_backend_graph_compute(&gpu_backend, graph) == first_fault);
        assert(!gpu_state.pending);
        gpu_state.completion_status = GGML_STATUS_BACKEND_POISONED;
        assert(ggml_backend_graph_compute(&gpu_backend, graph) == GGML_STATUS_BACKEND_POISONED);
        assert(!gpu_state.pending);
    }
    reset(gpu_state);
    assert(ggml_backend_graph_compute(&gpu_backend, graph) == GGML_STATUS_SUCCESS);
    assert(!gpu_state.pending);
    reset(gpu_state);

    // Transfer submission and completion faults leave no pending work and do
    // not mutate a fallback destination. Platform adapters map their runtime
    // result into these same terminal classifications before poison-repeat.
    reset(gpu_state); reset(cpu_state);
    gpu_state.submit_status = GGML_STATUS_DEVICE_LOST;
    assert(ggml_backend_graph_compute(&gpu_backend, graph) == GGML_STATUS_DEVICE_LOST);
    assert(!gpu_state.pending);
    reset(gpu_state); reset(cpu_state);
    gpu_state.completion_status = GGML_STATUS_DEVICE_LOST;
    assert(ggml_backend_graph_compute(&gpu_backend, graph) == GGML_STATUS_DEVICE_LOST);
    assert(!gpu_state.pending);

    // A terminal fault wins over cancellation at the same segment boundary.
    gpu_state.completion_status = GGML_STATUS_DEVICE_LOST;
    abort_probe cancel_vs_device_loss = {&gpu_state, 0, 2};
    assert(ggml_backend_graph_compute_with_abort(
               &gpu_backend, graph, abort_after_polls, &cancel_vs_device_loss, &capability) == GGML_STATUS_DEVICE_LOST);
    assert(!gpu_state.pending);
    reset(gpu_state);

    ggml_backend_t scheduler_backends[] = {&gpu_backend, &cpu_backend};
    ggml_backend_buffer_type_t scheduler_bufts[] = {&gpu_buft, &cpu_buft};

    // A memory plan freezes the graph after the scheduler has performed its
    // own mixed-backend split. The split legitimately replaces a cross-backend
    // source with a scheduler-owned copy; commit must not misclassify that
    // internal rewrite as caller mutation.
    ggml_backend_sched_t memory_plan_scheduler = ggml_backend_sched_new(
        scheduler_backends, scheduler_bufts, 2, 16, false, false);
    ggml_context * memory_plan_cpu_ctx = ggml_init(params);
    ggml_context * memory_plan_gpu_ctx = ggml_init(params);
    assert(memory_plan_cpu_ctx != nullptr);
    assert(memory_plan_gpu_ctx != nullptr);
    ggml_tensor * memory_plan_input = ggml_new_tensor_1d(memory_plan_cpu_ctx, GGML_TYPE_F32, 1);
    ggml_tensor * memory_plan_cpu_node = ggml_sqr(memory_plan_cpu_ctx, memory_plan_input);
    ggml_tensor * memory_plan_gpu_node = ggml_sqr(memory_plan_gpu_ctx, memory_plan_cpu_node);
    ggml_backend_buffer_t memory_plan_cpu_buffer =
        ggml_backend_alloc_ctx_tensors(memory_plan_cpu_ctx, &cpu_backend);
    ggml_backend_buffer_t memory_plan_gpu_buffer =
        ggml_backend_alloc_ctx_tensors(memory_plan_gpu_ctx, &gpu_backend);
    assert(memory_plan_cpu_buffer != nullptr);
    assert(memory_plan_gpu_buffer != nullptr);
    ggml_cgraph * memory_plan_graph = ggml_new_graph_custom(memory_plan_gpu_ctx, 16, false);
    ggml_build_forward_expand(memory_plan_graph, memory_plan_gpu_node);
    ggml_backend_sched_memory_plan_t memory_plan = nullptr;

    // A provider failure during graph optimization may occur after the split
    // has rewritten cross-backend sources. Creation must roll back those
    // rewrites and leave the scheduler reusable instead of publishing a
    // half-created plan.
    gpu_state.throw_graph_optimize = true;
    assert(ggml_backend_sched_memory_plan_create_v1(
               memory_plan_scheduler, memory_plan_graph, &memory_plan) == GGML_STATUS_DEVICE_LOST);
    assert(memory_plan == nullptr);
    assert(memory_plan_gpu_node->src[0] == memory_plan_cpu_node);
    gpu_state.throw_graph_optimize = false;

    const int gpu_frees_before_recovery = gpu_state.buffer_free_calls;
    const int cpu_frees_before_recovery = cpu_state.buffer_free_calls;
    assert(ggml_backend_sched_memory_plan_create_v1(
               memory_plan_scheduler, memory_plan_graph, &memory_plan) == GGML_STATUS_SUCCESS);
    assert(memory_plan != nullptr);
    assert(memory_plan_gpu_node->src[0] != memory_plan_cpu_node);
    gpu_state.throw_init_tensor = true;
    cpu_state.throw_init_tensor = true;
    uint32_t failed_commit_flags = 0;
    assert(ggml_backend_sched_memory_plan_commit_v2(
               memory_plan, &failed_commit_flags) == GGML_STATUS_ALLOC_FAILED);
    assert((failed_commit_flags & GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_MAY_HAVE_MUTATED) != 0);
    assert((failed_commit_flags & GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_RELEASE_PROVEN) != 0);
    assert(gpu_state.buffer_free_calls > gpu_frees_before_recovery ||
           cpu_state.buffer_free_calls > cpu_frees_before_recovery);
    assert(memory_plan_gpu_node->src[0] == memory_plan_cpu_node);
    ggml_backend_sched_memory_plan_free_v1(memory_plan);
    memory_plan = nullptr;
    gpu_state.throw_init_tensor = false;
    cpu_state.throw_init_tensor = false;

    // The proven recovery installs a fresh empty allocator, so the same
    // scheduler can commit a later plan without rebuilding the backend.
    assert(ggml_backend_sched_memory_plan_create_v1(
               memory_plan_scheduler, memory_plan_graph, &memory_plan) == GGML_STATUS_SUCCESS);
    assert(memory_plan != nullptr);
    // The v1 entrypoint remains ABI-compatible and delegates to the v2 commit
    // contract used by OpenASR's typed admission layer.
    assert(ggml_backend_sched_memory_plan_commit_v1(memory_plan) == GGML_STATUS_SUCCESS);
    ggml_backend_sched_memory_plan_free_v1(memory_plan);
    memory_plan = nullptr;

    // A release callback exception is contained but cannot be treated as proof.
    assert(ggml_backend_sched_memory_plan_create_v1(
               memory_plan_scheduler, memory_plan_graph, &memory_plan) == GGML_STATUS_SUCCESS);
    gpu_state.throw_init_tensor = true;
    cpu_state.throw_init_tensor = true;
    gpu_state.throw_buffer_free = true;
    cpu_state.throw_buffer_free = true;
    uint32_t unproven_commit_flags = 0;
    assert(ggml_backend_sched_memory_plan_commit_v2(
               memory_plan, &unproven_commit_flags) == GGML_STATUS_ALLOC_FAILED);
    assert((unproven_commit_flags & GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_MAY_HAVE_MUTATED) != 0);
    assert((unproven_commit_flags & GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_RELEASE_PROVEN) == 0);
    ggml_backend_sched_memory_plan_free_v1(memory_plan);
    gpu_state.throw_init_tensor = false;
    cpu_state.throw_init_tensor = false;
    gpu_state.throw_buffer_free = false;
    cpu_state.throw_buffer_free = false;
    memory_plan_gpu_node->src[0] = memory_plan_cpu_node;
    ggml_backend_sched_free(memory_plan_scheduler);

    ggml_backend_sched_t release_status_scheduler = ggml_backend_sched_new(
        scheduler_backends, scheduler_bufts, 2, 16, false, false);
    assert(release_status_scheduler != nullptr);
    assert(ggml_backend_sched_memory_plan_create_v1(
               release_status_scheduler, memory_plan_graph, &memory_plan) ==
           GGML_STATUS_SUCCESS);
    assert(ggml_backend_sched_memory_plan_commit_v1(memory_plan) ==
           GGML_STATUS_SUCCESS);
    ggml_backend_sched_memory_plan_free_v1(memory_plan);
    memory_plan = nullptr;
    gpu_state.throw_buffer_free = true;
    cpu_state.throw_buffer_free = true;
    assert(ggml_backend_sched_free_status(release_status_scheduler) ==
           GGML_STATUS_EXECUTION_FAILED);
    gpu_state.throw_buffer_free = false;
    cpu_state.throw_buffer_free = false;
    ggml_backend_buffer_free(memory_plan_gpu_buffer);
    ggml_backend_buffer_free(memory_plan_cpu_buffer);
    ggml_free(memory_plan_gpu_ctx);
    ggml_free(memory_plan_cpu_ctx);

    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
        scheduler_backends, scheduler_bufts, 2, 16, false, false);

    ggml_tensor * scheduler_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * cpu_node = ggml_sqr(ctx, scheduler_input);
    ggml_tensor * gpu_node = ggml_sqr(ctx, cpu_node);
    ggml_cgraph * scheduler_graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(scheduler_graph, gpu_node);
    assert(scheduler_graph->n_nodes == 2);
    ggml_backend_sched_set_tensor_backend(scheduler, cpu_node, &cpu_backend);
    ggml_backend_sched_set_tensor_backend(scheduler, gpu_node, &gpu_backend);
    assert(ggml_backend_sched_alloc_graph(scheduler, scheduler_graph));

    // Use independent roots so this aggregation test does not depend on the
    // fake backend modeling multiple queued copy/compute operations.
    ggml_backend_sched_t mixed_scheduler = ggml_backend_sched_new(
        scheduler_backends, scheduler_bufts, 2, 16, false, false);
    ggml_tensor * mixed_cpu_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * mixed_gpu_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * mixed_cpu_node = ggml_sqr(ctx, mixed_cpu_input);
    ggml_tensor * mixed_gpu_node = ggml_sqr(ctx, mixed_gpu_input);
    ggml_cgraph * mixed_graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(mixed_graph, mixed_cpu_node);
    ggml_build_forward_expand(mixed_graph, mixed_gpu_node);
    assert(mixed_graph->n_nodes == 2);
    ggml_backend_sched_set_tensor_backend(mixed_scheduler, mixed_cpu_input, &cpu_backend);
    ggml_backend_sched_set_tensor_backend(mixed_scheduler, mixed_cpu_node, &cpu_backend);
    ggml_backend_sched_set_tensor_backend(mixed_scheduler, mixed_gpu_input, &gpu_backend);
    ggml_backend_sched_set_tensor_backend(mixed_scheduler, mixed_gpu_node, &gpu_backend);
    assert(ggml_backend_sched_alloc_graph(mixed_scheduler, mixed_graph));

    // A mixed scheduler remains SEGMENTED when any executed split needs the
    // fallback, while its orthogonal granularity conservatively reports the
    // coarsest actual path across all executed splits.
    reset(cpu_state);
    reset(gpu_state);
    gpu_device.reg = &native_reg;
    gpu_state.native_granularity = GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_GRAPH_COMPLETION;
    assert(ggml_backend_sched_graph_compute_with_abort(
               mixed_scheduler,
               mixed_graph,
               never_abort,
               nullptr,
               &capability) == GGML_STATUS_SUCCESS);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_SEGMENTED);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_GRAPH_COMPLETION);
    assert(!cpu_state.pending);
    assert(!gpu_state.pending);
    ggml_backend_sched_free(mixed_scheduler);

    gpu_device.reg = nullptr;
    reset(cpu_state);
    reset(gpu_state);

    scheduler_copy_abort_probe scheduler_probe = {&gpu_state};
    assert(ggml_backend_sched_graph_compute_with_abort(
               scheduler,
               scheduler_graph,
               abort_after_first_scheduler_copy,
               &scheduler_probe,
               &capability) == GGML_STATUS_ABORTED);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_SEGMENTED);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    assert(cpu_state.graph_sizes == std::vector<int>({1}));
    assert(gpu_state.copy_calls == 1);
    assert(gpu_state.graph_sizes.empty());
    assert(scheduler_probe.polls > 0);
    assert(!cpu_state.pending);
    assert(!gpu_state.pending);

    // A provider-terminal async copy is not an "unsupported" result. It must
    // fail the exact scheduler lane instead of silently retrying through the
    // synchronous host fallback.
    reset(cpu_state);
    reset(gpu_state);
    gpu_state.transfer_submit_status = GGML_STATUS_DEVICE_LOST;
    assert(ggml_backend_sched_graph_compute(
               scheduler, scheduler_graph) == GGML_STATUS_DEVICE_LOST);
    assert(gpu_state.copy_calls == 1);
    assert(gpu_state.graph_sizes.empty());
    assert(!cpu_state.pending);
    assert(!gpu_state.pending);

    // callback_eval computes synchronously per observed node. The first terminal
    // completion fails, the scheduler's final drain succeeds, and the original
    // failure remains authoritative with no backend work pending.
    reset(cpu_state);
    reset(gpu_state);
    cpu_state.completion_sequence = {GGML_STATUS_EXECUTION_FAILED, GGML_STATUS_SUCCESS};
    scheduler_callback_probe callback_probe;
    ggml_backend_sched_set_eval_callback(scheduler, observe_scheduler_node, &callback_probe);
    assert(ggml_backend_sched_graph_compute(scheduler, scheduler_graph) == GGML_STATUS_EXECUTION_FAILED);
    assert(callback_probe.observed == 0);
    assert(cpu_state.synchronize_calls >= 2);
    assert(!cpu_state.pending);
    assert(!gpu_state.pending);
    ggml_backend_sched_set_eval_callback(scheduler, nullptr, nullptr);

    ggml_backend_sched_free(scheduler);

    ggml_free(ctx);
    return 0;
}
