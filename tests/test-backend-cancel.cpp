#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif
#include "ggml-impl.h"

#include <algorithm>
#include <cassert>
#include <cstring>
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
    int abort_set_calls = 0;
    int abort_clear_calls = 0;
};

static const char * fake_buffer_name(ggml_backend_buffer_type_t buft) {
    return static_cast<fake_backend_state *>(buft->context)->name;
}

static void fake_buffer_free(ggml_backend_buffer_t buffer) {
    ggml_aligned_free(buffer->context, buffer->size);
}

static void * fake_buffer_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
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

static ggml_backend_buffer_t fake_buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    assert(data != nullptr);
    const ggml_backend_buffer_i iface = {
        /* .free_buffer   = */ fake_buffer_free,
        /* .get_base      = */ fake_buffer_base,
        /* .init_tensor   = */ nullptr,
        /* .memset_tensor = */ fake_buffer_memset,
        /* .set_tensor    = */ fake_buffer_set,
        /* .get_tensor    = */ fake_buffer_get,
        /* .set_tensor_2d = */ nullptr,
        /* .get_tensor_2d = */ nullptr,
        /* .cpy_tensor    = */ nullptr,
        /* .clear         = */ fake_buffer_clear,
        /* .reset         = */ nullptr,
    };
    return ggml_backend_buffer_init(buft, iface, data, size);
}

static size_t fake_buffer_alignment(ggml_backend_buffer_type_t) {
    return GGML_MEM_ALIGN;
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

static void fake_set_abort_callback(
        ggml_backend_t backend, ggml_abort_callback abort_callback, void * abort_callback_data) {
    auto * state = static_cast<fake_backend_state *>(backend->context);
    state->abort_callback = abort_callback;
    state->abort_callback_data = abort_callback != nullptr ? abort_callback_data : nullptr;
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
    state.abort_set_calls = 0;
    state.abort_clear_calls = 0;
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
            /* .graph_optimize      = */ nullptr,
        },
        /* .device  = */ device,
        /* .context = */ state,
    };
}

int main() {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);
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
    enum ggml_backend_graph_cancel_mode mode = GGML_BACKEND_GRAPH_CANCEL_DISABLED;
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
               &backend, graph, native_abort_after_polls, &native_false_probe, &mode) == GGML_STATUS_SUCCESS);
    assert(mode == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(state.graph_sizes == std::vector<int>({65}));
    assert(state.synchronize_calls == 1);
    assert(native_false_probe.polls == 3);
    assert(state.abort_set_calls == 1);
    assert(state.abort_clear_calls == 1);
    assert(state.abort_callback == nullptr);
    assert(!state.pending);

    // Native cancellation drains accepted work, clears the borrowed callback,
    // and leaves the backend reusable by the next compute.
    reset(state);
    abort_probe native_cancel_probe = {&state, 0, 2};
    assert(ggml_backend_graph_compute_with_abort(
               &backend, graph, native_abort_after_polls, &native_cancel_probe, &mode) == GGML_STATUS_ABORTED);
    assert(mode == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(state.graph_sizes == std::vector<int>({65}));
    assert(state.synchronize_calls == 1);
    assert(native_cancel_probe.polls == 2);
    assert(state.abort_callback == nullptr);
    assert(!state.pending);

    reset(state);
    abort_probe native_reuse_probe = {&state, 0, 100};
    assert(ggml_backend_graph_compute_with_abort(
               &backend, graph, native_abort_after_polls, &native_reuse_probe, &mode) == GGML_STATUS_SUCCESS);
    assert(state.graph_sizes == std::vector<int>({65}));
    assert(!state.pending);

    // A terminal device failure remains authoritative when cancellation races
    // the same native submission.
    reset(state);
    state.completion_status = GGML_STATUS_DEVICE_LOST;
    abort_probe native_cancel_vs_device_loss = {&state, 0, 2};
    assert(ggml_backend_graph_compute_with_abort(
               &backend, graph, native_abort_after_polls, &native_cancel_vs_device_loss, &mode) == GGML_STATUS_DEVICE_LOST);
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
               &mode) == GGML_STATUS_SUCCESS);
    assert(mode == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
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
               &mode) == GGML_STATUS_ABORTED);
    assert(mode == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
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
               &mode) == GGML_STATUS_SUCCESS);
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
               &mode) == GGML_STATUS_SUCCESS);
    assert(mode == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(blas_false_probe.polls == 3);

    abort_probe blas_cancel_probe = {&state, 0, 2};
    assert(ggml_backend_graph_compute_with_abort(
               blas_backend,
               &inert_graph,
               native_abort_after_polls,
               &blas_cancel_probe,
               &mode) == GGML_STATUS_ABORTED);
    assert(mode == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(blas_cancel_probe.polls == 2);

    abort_probe blas_reuse_probe = {&state, 0, 100};
    assert(ggml_backend_graph_compute_with_abort(
               blas_backend,
               &inert_graph,
               native_abort_after_polls,
               &blas_reuse_probe,
               &mode) == GGML_STATUS_SUCCESS);
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
               &backend, graph, abort_after_polls, &pre_cancel_failure, &mode) == GGML_STATUS_DEVICE_LOST);
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
               &backend, graph, abort_after_polls, &false_probe, &mode) == GGML_STATUS_SUCCESS);
    assert(mode == GGML_BACKEND_GRAPH_CANCEL_SEGMENTED);
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
               &backend, graph, abort_after_polls, &cancel_probe, &mode) == GGML_STATUS_ABORTED);
    assert(mode == GGML_BACKEND_GRAPH_CANCEL_SEGMENTED);
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
               &backend, graph, abort_after_polls, &segment_failure, &mode) == GGML_STATUS_EXECUTION_FAILED);
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
               &gpu_backend, graph, abort_after_polls, &cancel_vs_device_loss, &mode) == GGML_STATUS_DEVICE_LOST);
    assert(!gpu_state.pending);
    reset(gpu_state);

    ggml_backend_t scheduler_backends[] = {&gpu_backend, &cpu_backend};
    ggml_backend_buffer_type_t scheduler_bufts[] = {&gpu_buft, &cpu_buft};
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

    scheduler_copy_abort_probe scheduler_probe = {&gpu_state};
    assert(ggml_backend_sched_graph_compute_with_abort(
               scheduler,
               scheduler_graph,
               abort_after_first_scheduler_copy,
               &scheduler_probe,
               &mode) == GGML_STATUS_ABORTED);
    assert(mode == GGML_BACKEND_GRAPH_CANCEL_SEGMENTED);
    assert(cpu_state.graph_sizes == std::vector<int>({1}));
    assert(gpu_state.copy_calls == 1);
    assert(gpu_state.graph_sizes.empty());
    assert(scheduler_probe.polls > 0);
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
