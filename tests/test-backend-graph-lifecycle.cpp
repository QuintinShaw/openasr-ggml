#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

static_assert(GGML_BACKEND_GRAPH_LIFECYCLE_ABI_V1 == 1u);
static_assert(GGML_BACKEND_GRAPH_LIFECYCLE_CAPTURE_SUPPORTED_V1 == (1u << 0));
static_assert(GGML_BACKEND_GRAPH_LIFECYCLE_CAPTURE_ENABLED_V1 == (1u << 1));
static_assert(GGML_BACKEND_GRAPH_LIFECYCLE_EXECUTABLE_PRESENT_V1 == (1u << 2));
static_assert(offsetof(ggml_backend_graph_lifecycle_observation_v1, struct_size) == 0);
static_assert(offsetof(ggml_backend_graph_lifecycle_observation_v1, abi_version) == 4);
static_assert(offsetof(ggml_backend_graph_lifecycle_observation_v1, flags) == 8);
static_assert(offsetof(ggml_backend_graph_lifecycle_observation_v1, last_executable_change) == 12);
static_assert(offsetof(ggml_backend_graph_lifecycle_observation_v1, executable_generation) == 16);
static_assert(sizeof(ggml_backend_graph_lifecycle_observation_v1) == 24);

int main() {
    ggml_backend_graph_lifecycle_observation_v1 observation = {};
    observation.struct_size = sizeof(observation);
    if (ggml_backend_graph_lifecycle_api_for_backend_v1(nullptr) != nullptr) {
        return 1;
    }
    if (ggml_backend_graph_lifecycle_api_observe_v1(
            nullptr, nullptr, nullptr, &observation) != GGML_STATUS_FAILED) {
        return 2;
    }
    if (observation.executable_generation != 0) {
        return 3;
    }
    if (ggml_graph_capture_source(nullptr) != nullptr ||
            ggml_graph_capture_uid(nullptr) != 0) {
        return 4;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return 5;
    }
    ggml_cgraph * owned = ggml_new_graph_custom(ctx, 16, false);
    if (owned == nullptr || ggml_graph_capture_source(owned) != owned ||
            owned->n_nodes != 0 || owned->uid == 0 ||
            ggml_graph_capture_uid(owned) != owned->uid) {
        ggml_free(ctx);
        return 6;
    }
    ggml_cgraph view = ggml_graph_view(owned, 0, 0);
    if (ggml_graph_capture_source(&view) != owned ||
            ggml_graph_capture_uid(&view) != owned->uid) {
        ggml_free(ctx);
        return 7;
    }
    ggml_cgraph nested = ggml_graph_view(&view, 0, 0);
    if (ggml_graph_capture_source(&nested) != owned ||
            ggml_graph_capture_uid(&nested) != owned->uid) {
        ggml_free(ctx);
        return 8;
    }
    ggml_cgraph split = ggml_graph_view(owned, 0, 0);
    split.view_src = nullptr;
    split.uid = ggml_graph_next_uid();
    if (ggml_graph_capture_source(&split) != &split ||
            ggml_graph_capture_uid(&split) == 0 ||
            ggml_graph_capture_uid(&split) == owned->uid) {
        ggml_free(ctx);
        return 9;
    }
    ggml_cgraph * dup = ggml_graph_dup(ctx, owned, false);
    if (dup == nullptr || ggml_graph_capture_source(dup) != dup ||
            ggml_graph_capture_uid(dup) == 0 ||
            ggml_graph_capture_uid(dup) == owned->uid) {
        ggml_free(ctx);
        return 10;
    }
    ggml_free(ctx);
    return 0;
}
