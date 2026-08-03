#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

using steady_clock = std::chrono::steady_clock;

struct graph_fixture {
    ggml_context * ctx;
    ggml_gallocr_t allocator;
    ggml_cgraph * graph;
    ggml_tensor * input;
    ggml_tensor * increment;
    ggml_tensor * output;
    std::vector<float> input_values;
    float expected;
};

static graph_fixture make_graph(ggml_backend_t backend, bool allocate_directly) {
    constexpr int node_count = 2048;
    constexpr int64_t element_count = 4 * 1024 * 1024;
    ggml_init_params params = {
        /* .mem_size   = */ 32 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor * increment = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor * output = input;
    for (int i = 0; i < node_count; ++i) {
        output = ggml_add(ctx, output, increment);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, output);
    assert(ggml_graph_n_nodes(graph) == node_count);

    ggml_gallocr_t allocator = nullptr;
    if (allocate_directly) {
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        assert(allocator != nullptr);
        assert(ggml_gallocr_alloc_graph(allocator, graph));
    }
    std::vector<float> values(element_count, 1.0f);
    return {ctx, allocator, graph, input, increment, output, std::move(values), (float) node_count + 1.0f};
}

static void free_graph(graph_fixture & fixture) {
    if (fixture.allocator != nullptr) {
        ggml_gallocr_free(fixture.allocator);
    }
    ggml_free(fixture.ctx);
}

static void reset_input(ggml_backend_t backend, graph_fixture & fixture) {
    ggml_backend_tensor_set(
        fixture.input, fixture.input_values.data(), 0, ggml_nbytes(fixture.input));
    ggml_backend_tensor_set(
        fixture.increment, fixture.input_values.data(), 0, ggml_nbytes(fixture.increment));
    assert(ggml_backend_synchronize(backend) == GGML_STATUS_SUCCESS);
}

static void assert_output(graph_fixture & fixture) {
    float value = 0.0f;
    ggml_backend_tensor_get(fixture.output, &value, 0, sizeof(value));
    assert(std::fabs(value - fixture.expected) < 0.01f);
}

static bool never_cancel(void *) {
    return false;
}

static bool always_cancel(void *) {
    return true;
}

struct replay_cancel_probe {
    std::atomic<unsigned> polls{0};
    std::atomic<bool> launch_imminent{false};
    std::atomic<bool> cancel{false};
};

static bool cancel_during_replay(void * opaque) {
    auto * probe = static_cast<replay_cancel_probe *>(opaque);
    const unsigned poll = probe->polls.fetch_add(1, std::memory_order_acq_rel) + 1;
    // Direct graph replay polls once in the public precheck, once on backend
    // entry, and once immediately before graph launch. No backend poll occurs
    // again until the terminal graph-completion boundary.
    if (poll == 3) {
        probe->launch_imminent.store(true, std::memory_order_release);
    }
    return probe->cancel.load(std::memory_order_acquire);
}

static void wait_for_launch_and_cancel(
        replay_cancel_probe & probe, std::chrono::nanoseconds delay) {
    const auto deadline = steady_clock::now() + std::chrono::seconds(5);
    while (!probe.launch_imminent.load(std::memory_order_acquire)) {
        assert(steady_clock::now() < deadline);
        std::this_thread::yield();
    }
    const auto cancel_at = steady_clock::now() + delay;
    while (steady_clock::now() < cancel_at) {
        // Windows timer granularity can overshoot a short sleep by an entire
        // replay. A bounded spin keeps this hardware contract test precise.
    }
    probe.cancel.store(true, std::memory_order_release);
}

static std::chrono::nanoseconds run_direct_success(
        ggml_backend_t backend,
        graph_fixture & fixture,
        enum ggml_backend_graph_cancel_observation_granularity expected_granularity) {
    reset_input(backend, fixture);
    struct ggml_backend_graph_cancel_capability capability = {};
    const auto start = steady_clock::now();
    assert(ggml_backend_graph_compute_with_abort(
               backend, fixture.graph, never_cancel, nullptr, &capability) == GGML_STATUS_SUCCESS);
    const auto elapsed = steady_clock::now() - start;
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(capability.observation_granularity == expected_granularity);
    assert_output(fixture);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
}

static void run_direct_replay_cancel(
        ggml_backend_t backend,
        graph_fixture & fixture,
        std::chrono::nanoseconds delay) {
    reset_input(backend, fixture);
    replay_cancel_probe probe;
    std::thread canceler(wait_for_launch_and_cancel, std::ref(probe), delay);
    struct ggml_backend_graph_cancel_capability capability = {};
    const enum ggml_status status = ggml_backend_graph_compute_with_abort(
        backend, fixture.graph, cancel_during_replay, &probe, &capability);
    canceler.join();
    assert(status == GGML_STATUS_ABORTED);
    assert(probe.polls.load(std::memory_order_acquire) >= 4);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(capability.observation_granularity ==
           GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_GRAPH_COMPLETION);
    // A monolithic replay finishes accepted work before ABORTED is returned.
    assert_output(fixture);

    reset_input(backend, fixture);
    assert(ggml_backend_graph_compute(backend, fixture.graph) == GGML_STATUS_SUCCESS);
    assert_output(fixture);
}

static void test_direct_contract() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend != nullptr);
    graph_fixture fixture = make_graph(backend, true);

    struct ggml_backend_graph_cancel_capability capability = {};
    assert(ggml_backend_graph_compute_with_abort(
               backend, fixture.graph, always_cancel, nullptr, &capability) == GGML_STATUS_ABORTED);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_DISABLED);
    assert(capability.observation_granularity == GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_NONE);

    // Stable graph pointer: direct evaluation, capture, then warmed pure replay.
    run_direct_success(
        backend, fixture, GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    run_direct_success(
        backend, fixture, GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    const auto replay_duration = run_direct_success(
        backend, fixture, GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_GRAPH_COMPLETION);

    run_direct_replay_cancel(backend, fixture, std::chrono::milliseconds(1));
    // Exercise a cancellation close to the terminal boundary while retaining
    // margin for normal run-to-run variance on the hardware test graph.
    run_direct_replay_cancel(backend, fixture, replay_duration * 7 / 10);

    free_graph(fixture);
    ggml_backend_free(backend);
}

struct timed_cancel_probe {
    std::atomic<bool> cancel{false};
};

static bool read_cancel_flag(void * opaque) {
    return static_cast<timed_cancel_probe *>(opaque)->cancel.load(std::memory_order_acquire);
}

static void test_scheduler_contract() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    assert(backend != nullptr);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    assert(cpu_backend != nullptr);
    graph_fixture fixture = make_graph(backend, false);
    ggml_backend_t backends[] = {backend, cpu_backend};
    ggml_backend_buffer_type_t buffer_types[] = {
        ggml_backend_get_default_buffer_type(backend),
        ggml_backend_get_default_buffer_type(cpu_backend),
    };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
        backends, buffer_types, 2, 4096, false, false);
    assert(scheduler != nullptr);
    assert(ggml_backend_sched_alloc_graph(scheduler, fixture.graph));

    auto run_success = [&](enum ggml_backend_graph_cancel_observation_granularity expected) {
        reset_input(backend, fixture);
        struct ggml_backend_graph_cancel_capability capability = {};
        assert(ggml_backend_sched_graph_compute_with_abort(
                   scheduler, fixture.graph, never_cancel, nullptr, &capability) == GGML_STATUS_SUCCESS);
        assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
        assert(capability.observation_granularity == expected);
        assert_output(fixture);
    };
    run_success(GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    run_success(GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    run_success(GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_GRAPH_COMPLETION);

    reset_input(backend, fixture);
    timed_cancel_probe probe;
    std::thread canceler([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        probe.cancel.store(true, std::memory_order_release);
    });
    struct ggml_backend_graph_cancel_capability capability = {};
    const enum ggml_status status = ggml_backend_sched_graph_compute_with_abort(
        scheduler, fixture.graph, read_cancel_flag, &probe, &capability);
    canceler.join();
    assert(status == GGML_STATUS_ABORTED);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(capability.observation_granularity ==
           GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_GRAPH_COMPLETION);
    assert_output(fixture);

    reset_input(backend, fixture);
    assert(ggml_backend_sched_graph_compute(scheduler, fixture.graph) == GGML_STATUS_SUCCESS);
    assert_output(fixture);

    ggml_backend_sched_free(scheduler);
    free_graph(fixture);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(backend);
}

int main() {
    assert(ggml_backend_cuda_get_device_count() > 0);
    test_direct_contract();
    test_scheduler_contract();
    return 0;
}
