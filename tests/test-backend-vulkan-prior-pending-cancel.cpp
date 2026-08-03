#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "ggml.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

using gate_create_fn = void * (*)(ggml_backend_t);
using gate_enqueue_wait_fn = bool (*)(ggml_backend_t, void *);
using gate_signal_fn = void (*)(ggml_backend_t, void *);
using gate_free_fn = void (*)(ggml_backend_t, void *);

struct cancel_after_precheck {
    std::atomic<unsigned> calls{0};
};

static bool cancel_on_first_native_poll(void * opaque) {
    cancel_after_precheck * probe = (cancel_after_precheck *) opaque;
    return probe->calls.fetch_add(1, std::memory_order_acq_rel) != 0;
}

static bool never_cancel(void *) {
    return false;
}

template <typename T>
static T get_test_proc(ggml_backend_reg_t reg, const char * name) {
    T proc = (T) ggml_backend_reg_get_proc_address(reg, name);
    assert(proc != nullptr);
    return proc;
}

struct graph_fixture {
    ggml_context * ctx;
    ggml_gallocr_t allocator;
    ggml_cgraph * graph;
    ggml_tensor * output;
};

static graph_fixture make_graph(ggml_backend_t backend) {
    constexpr int node_count = 65;
    constexpr int element_count = 1024;
    ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);
    ggml_tensor * value = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor * increment = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor * output = value;
    for (int i = 0; i < node_count; ++i) {
        output = ggml_add(ctx, output, increment);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, output);
    assert(ggml_graph_n_nodes(graph) == node_count);

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(allocator != nullptr);
    assert(ggml_gallocr_alloc_graph(allocator, graph));
    std::vector<float> values(element_count, 1.0f);
    ggml_backend_tensor_set(value, values.data(), 0, ggml_nbytes(value));
    ggml_backend_tensor_set(increment, values.data(), 0, ggml_nbytes(increment));
    assert(ggml_backend_synchronize(backend) == GGML_STATUS_SUCCESS);
    return { ctx, allocator, graph, output };
}

static void free_graph(graph_fixture & fixture) {
    ggml_gallocr_free(fixture.allocator);
    ggml_free(fixture.ctx);
}

static void run_prior_pending_case(
        ggml_backend_t backend,
        graph_fixture & fixture,
        void * gate,
        gate_enqueue_wait_fn enqueue_wait,
        gate_signal_fn signal_gate) {
    assert(enqueue_wait(backend, gate));

    cancel_after_precheck probe;
    std::atomic<bool> returned{false};
    enum ggml_status status = GGML_STATUS_FAILED;
    enum ggml_backend_graph_cancel_mode mode = GGML_BACKEND_GRAPH_CANCEL_DISABLED;
    std::thread compute([&]() {
        status = ggml_backend_graph_compute_with_abort(
            backend, fixture.graph, cancel_on_first_native_poll, &probe, &mode);
        returned.store(true, std::memory_order_release);
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (probe.calls.load(std::memory_order_acquire) < 2 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    assert(probe.calls.load(std::memory_order_acquire) >= 2);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const bool returned_before_gate = returned.load(std::memory_order_acquire);
    signal_gate(backend, gate);
    compute.join();

    assert(!returned_before_gate);
    assert(status == GGML_STATUS_ABORTED);
    assert(mode == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(ggml_backend_graph_compute(backend, fixture.graph) == GGML_STATUS_SUCCESS);
}

int main() {
    ggml_backend_reg_t reg = ggml_backend_vk_reg();
    assert(reg != nullptr);
    assert(ggml_backend_reg_dev_count(reg) > 0);
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    assert(backend != nullptr);

    gate_create_fn gate_create = get_test_proc<gate_create_fn>(reg, "ggml_backend_vk_test_gate_create");
    gate_enqueue_wait_fn enqueue_wait =
        get_test_proc<gate_enqueue_wait_fn>(reg, "ggml_backend_vk_test_gate_enqueue_wait");
    gate_signal_fn signal_gate = get_test_proc<gate_signal_fn>(reg, "ggml_backend_vk_test_gate_signal");
    gate_free_fn gate_free = get_test_proc<gate_free_fn>(reg, "ggml_backend_vk_test_gate_free");

    graph_fixture fixture = make_graph(backend);

    // A callback-free submission exists before the first armed completion
    // value has ever been created.
    void * initial_gate = gate_create(backend);
    run_prior_pending_case(backend, fixture, initial_gate, enqueue_wait, signal_gate);
    gate_free(backend, initial_gate);

    // Establish a completed historical armed value, then enqueue new
    // callback-free work. The old value must not cover the new submission.
    enum ggml_backend_graph_cancel_mode prime_mode = GGML_BACKEND_GRAPH_CANCEL_DISABLED;
    assert(ggml_backend_graph_compute_with_abort(
        backend, fixture.graph, never_cancel, nullptr, &prime_mode) == GGML_STATUS_SUCCESS);
    assert(prime_mode == GGML_BACKEND_GRAPH_CANCEL_NATIVE);

    void * historical_gate = gate_create(backend);
    run_prior_pending_case(backend, fixture, historical_gate, enqueue_wait, signal_gate);
    gate_free(backend, historical_gate);

    free_graph(fixture);
    ggml_backend_free(backend);
    return 0;
}
