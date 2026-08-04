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
using has_pending_submission_fn = bool (*)(ggml_backend_t);

struct cancel_probe {
    ggml_backend_t canceled_backend;
    ggml_backend_t tail_backend;
    void * gate;
    gate_enqueue_wait_fn enqueue_wait;
    has_pending_submission_fn has_pending_submission;
    std::atomic<bool> tail_enqueued{false};
};

static bool cancel_after_own_submission(void * opaque) {
    cancel_probe * probe = (cancel_probe *) opaque;
    if (!probe->has_pending_submission(probe->canceled_backend)) {
        return false;
    }

    bool expected = false;
    if (probe->tail_enqueued.compare_exchange_strong(expected, true)) {
        assert(probe->enqueue_wait(probe->tail_backend, probe->gate));
    }
    return true;
}

template <typename T>
static T get_test_proc(ggml_backend_reg_t reg, const char * name) {
    T proc = (T) ggml_backend_reg_get_proc_address(reg, name);
    assert(proc != nullptr);
    return proc;
}

int main() {
    ggml_backend_reg_t reg = ggml_backend_vk_reg();
    assert(reg != nullptr);
    assert(ggml_backend_reg_dev_count(reg) > 0);
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);

    ggml_backend_t tail_backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t canceled_backend = ggml_backend_dev_init(dev, nullptr);
    assert(tail_backend != nullptr);
    assert(canceled_backend != nullptr);

    gate_create_fn gate_create = get_test_proc<gate_create_fn>(reg, "ggml_backend_vk_test_gate_create");
    gate_enqueue_wait_fn gate_enqueue_wait =
        get_test_proc<gate_enqueue_wait_fn>(reg, "ggml_backend_vk_test_gate_enqueue_wait");
    gate_signal_fn gate_signal = get_test_proc<gate_signal_fn>(reg, "ggml_backend_vk_test_gate_signal");
    gate_free_fn gate_free = get_test_proc<gate_free_fn>(reg, "ggml_backend_vk_test_gate_free");
    has_pending_submission_fn has_pending_submission =
        get_test_proc<has_pending_submission_fn>(reg, "ggml_backend_vk_test_has_pending_submission");

    void * gate = gate_create(tail_backend);
    assert(gate != nullptr);

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

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(canceled_backend));
    assert(allocator != nullptr);
    assert(ggml_gallocr_alloc_graph(allocator, graph));
    std::vector<float> values(element_count, 1.0f);
    ggml_backend_tensor_set(value, values.data(), 0, ggml_nbytes(value));
    ggml_backend_tensor_set(increment, values.data(), 0, ggml_nbytes(increment));

    cancel_probe probe = {
        canceled_backend,
        tail_backend,
        gate,
        gate_enqueue_wait,
        has_pending_submission,
    };
    std::atomic<bool> canceled_returned{false};
    std::atomic<bool> gate_released{false};
    std::atomic<bool> watchdog_was_needed{false};
    auto release_gate_once = [&](bool watchdog) {
        bool expected = false;
        if (gate_released.compare_exchange_strong(expected, true)) {
            watchdog_was_needed.store(watchdog, std::memory_order_release);
            gate_signal(tail_backend, gate);
        }
    };
    std::thread watchdog([&]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (!canceled_returned.load(std::memory_order_acquire)) {
            release_gate_once(true);
        }
    });

    struct ggml_backend_graph_cancel_capability capability = {};
    const enum ggml_status canceled = ggml_backend_graph_compute_with_abort(
        canceled_backend, graph, cancel_after_own_submission, &probe, &capability);
    canceled_returned.store(true, std::memory_order_release);
    release_gate_once(false);
    watchdog.join();

    assert(canceled == GGML_STATUS_ABORTED);
    assert(capability.mechanism == GGML_BACKEND_GRAPH_CANCEL_NATIVE);
    assert(capability.observation_granularity ==
           GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);
    assert(probe.tail_enqueued.load(std::memory_order_acquire));
    assert(!watchdog_was_needed.load(std::memory_order_acquire));
    assert(ggml_backend_synchronize(tail_backend) == GGML_STATUS_SUCCESS);

    // Both contexts remain usable after the canceled context returns and the
    // unrelated tail submission is released.
    assert(ggml_backend_graph_compute(canceled_backend, graph) == GGML_STATUS_SUCCESS);
    assert(ggml_backend_graph_compute(tail_backend, graph) == GGML_STATUS_SUCCESS);

    gate_free(tail_backend, gate);
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    ggml_backend_free(canceled_backend);
    ggml_backend_free(tail_backend);
    return 0;
}
