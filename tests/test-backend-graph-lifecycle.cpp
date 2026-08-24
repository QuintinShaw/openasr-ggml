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
    return observation.executable_generation == 0 ? 0 : 1;
}
