#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

struct fake_backend_state {
    const char * name = "fake";
    enum ggml_backend_dev_type type = GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_buffer_type_t buft = nullptr;
    bool pending = false;
    int synchronize_calls = 0;
    int copy_calls = 0;
    std::vector<int> graph_sizes;
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

static bool fake_copy_async(
        ggml_backend_t, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    auto * state = static_cast<fake_backend_state *>(backend_dst->context);
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    state->copy_calls++;
    state->pending = true;
    return true;
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
    return GGML_STATUS_SUCCESS;
}

static void fake_synchronize(ggml_backend_t backend) {
    auto * state = static_cast<fake_backend_state *>(backend->context);
    state->pending = false;
    state->synchronize_calls++;
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
    state.synchronize_calls = 0;
    state.copy_calls = 0;
    state.graph_sizes.clear();
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
            /* .event_synchronize    = */ nullptr,
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
            /* .set_tensor_async   = */ nullptr,
            /* .get_tensor_async   = */ nullptr,
            /* .set_tensor_2d_async= */ nullptr,
            /* .get_tensor_2d_async= */ nullptr,
            /* .cpy_tensor_async   = */ fake_copy_async,
            /* .synchronize        = */ fake_synchronize,
            /* .graph_plan_create  = */ nullptr,
            /* .graph_plan_free    = */ nullptr,
            /* .graph_plan_update  = */ nullptr,
            /* .graph_plan_compute = */ nullptr,
            /* .graph_compute      = */ fake_graph_compute,
            /* .event_record       = */ nullptr,
            /* .event_wait         = */ nullptr,
            /* .graph_optimize     = */ nullptr,
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
    ggml_backend_device device = {};
    device.reg = &reg;
    fake_backend_state state;
    ggml_backend backend = {};
    backend.device = &device;
    backend.context = &state;
    backend.iface.graph_compute = fake_graph_compute;
    backend.iface.synchronize = fake_synchronize;

    // Callback-free compute stays one async submission followed by the public
    // synchronous wrapper's one synchronization.
    enum ggml_backend_graph_cancel_mode mode = GGML_BACKEND_GRAPH_CANCEL_DISABLED;
    assert(ggml_backend_graph_compute(&backend, graph) == GGML_STATUS_SUCCESS);
    assert(state.graph_sizes == std::vector<int>({65}));
    assert(state.synchronize_calls == 1);
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
    ggml_backend gpu_backend = make_fake_scheduler_backend(&gpu_state, &gpu_device);
    ggml_backend cpu_backend = make_fake_scheduler_backend(&cpu_state, &cpu_device);
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

    ggml_backend_sched_free(scheduler);

    ggml_free(ctx);
    return 0;
}
