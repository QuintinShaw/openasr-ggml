#!/usr/bin/env python3
"""Source-contract checks for native capture lifecycle evidence.

These checks cannot claim CUDA/HIP hardware execution. They keep the optional
ABI tied to the native instantiate/update sites and prevent the observation
path from creating or mutating a graph-cache entry. Real-provider CI remains
authoritative for capture behavior.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BackendGraphLifecycleStaticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (ROOT / "include/ggml-backend.h").read_text()
        cls.common = (ROOT / "src/ggml-cuda/common.cuh").read_text()
        cls.cuda = (ROOT / "src/ggml-cuda/ggml-cuda.cu").read_text()

    def test_versioned_observation_is_diagnostic_only(self) -> None:
        self.assertIn("GGML_BACKEND_GRAPH_LIFECYCLE_API_V1_PROC", self.header)
        self.assertIn("GGML_BACKEND_GRAPH_LIFECYCLE_ABI_V1", self.header)
        self.assertIn("executable generation minted", self.header)
        self.assertIn("uint64_t executable_generation", self.header)
        self.assertIn("uint32_t last_executable_change", self.header)
        self.assertIn("GGML_BACKEND_GRAPH_LIFECYCLE_GRAPH_TRACKED_V1", self.header)

    def test_generation_is_context_monotonic_across_graph_eviction(self) -> None:
        self.assertIn("uint64_t last_cuda_graph_executable_generation = 0", self.common)
        recorder = self.common[
            self.common.index("bool record_cuda_graph_executable_change") :
            self.common.index("// Check if any CUDA graph is enabled")
        ]
        self.assertIn("++last_cuda_graph_executable_generation", recorder)
        self.assertIn("graph->executable_generation", recorder)
        self.assertIn("graph->last_executable_change", recorder)

    def test_successful_native_changes_are_recorded_once_per_capture_cycle(self) -> None:
        evaluator = self.cuda[
            self.cuda.index("static enum ggml_status ggml_cuda_graph_evaluate_and_capture") :
            self.cuda.index("static bool ggml_cuda_graph_set_enabled")
        ]
        self.assertIn("cudaGraphInstantiate", evaluator)
        self.assertIn("GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_INSTANTIATED_V1", evaluator)
        self.assertIn("ggml_cuda_graph_update_executable", evaluator)
        self.assertIn("record_cuda_graph_executable_change", evaluator)

        updater = self.cuda[
            self.cuda.index("static enum ggml_status ggml_cuda_graph_update_executable") :
            self.cuda.index("#endif // USE_CUDA_GRAPH", self.cuda.index("static enum ggml_status ggml_cuda_graph_update_executable"))
        ]
        self.assertIn("GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_UPDATED_V1", updater)
        self.assertIn("GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_REPLACED_V1", updater)
        self.assertIn("instantiate_status == cudaSuccess", updater)
        self.assertIn("stat == cudaSuccess", updater)

    def test_observation_is_side_effect_free_and_exported_by_cuda_and_hip(self) -> None:
        observer = self.cuda[
            self.cuda.index("static enum ggml_status ggml_backend_cuda_graph_lifecycle_observe") :
            self.cuda.index("static const ggml_backend_graph_lifecycle_api_v1", self.cuda.index("static enum ggml_status ggml_backend_cuda_graph_lifecycle_observe"))
        ]
        self.assertIn("find_cuda_graph", observer)
        self.assertNotIn("->cuda_graph(", observer)
        self.assertIn("GGML_BACKEND_GRAPH_LIFECYCLE_GRAPH_TRACKED_V1", observer)
        self.assertIn("GGML_BACKEND_GRAPH_LIFECYCLE_EXECUTABLE_PRESENT_V1", observer)
        self.assertIn("GGML_BACKEND_GRAPH_LIFECYCLE_API_V1_PROC", self.cuda)
        # HIP compiles this same source through vendors/hip.h, so there must be
        # no provider-name branch around the registration.
        registration = self.cuda[
            self.cuda.index("static void * ggml_backend_cuda_reg_get_proc_address") :
            self.cuda.index("static const ggml_backend_reg_i", self.cuda.index("static void * ggml_backend_cuda_reg_get_proc_address"))
        ]
        self.assertIn("ggml_backend_cuda_graph_lifecycle_get_api_v1", registration)


if __name__ == "__main__":
    unittest.main()
