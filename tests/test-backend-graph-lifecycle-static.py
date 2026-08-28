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
        cls.ggml_h = (ROOT / "include/ggml.h").read_text()
        cls.ggml_c = (ROOT / "src/ggml.c").read_text()
        cls.backend = (ROOT / "src/ggml-backend.cpp").read_text()
        cls.common = (ROOT / "src/ggml-cuda/common.cuh").read_text()
        cls.cuda = (ROOT / "src/ggml-cuda/ggml-cuda.cu").read_text()

    def test_versioned_observation_is_diagnostic_only(self) -> None:
        self.assertIn("GGML_BACKEND_GRAPH_LIFECYCLE_API_V1_PROC", self.header)
        self.assertIn("GGML_BACKEND_GRAPH_LIFECYCLE_ABI_V1", self.header)
        self.assertIn("executable generation minted", self.header)
        self.assertIn("uint64_t executable_generation", self.header)
        self.assertIn("uint32_t last_executable_change", self.header)
        self.assertIn("GGML_BACKEND_GRAPH_LIFECYCLE_GRAPH_TRACKED_V1", self.header)

    def test_foreign_callers_use_shared_no_throw_trampolines(self) -> None:
        self.assertIn("ggml_backend_graph_lifecycle_api_for_backend_v1", self.header)
        self.assertIn("ggml_backend_graph_lifecycle_api_observe_v1", self.header)
        resolver = self.backend[
            self.backend.index("ggml_backend_graph_lifecycle_api_for_backend_v1") :
            self.backend.index("ggml_backend_graph_lifecycle_api_observe_v1")
        ]
        self.assertIn("ggml_backend_noexcept_or", resolver)
        self.assertIn("GGML_BACKEND_GRAPH_LIFECYCLE_API_V1_PROC", resolver)
        observer = self.backend[
            self.backend.index("ggml_backend_graph_lifecycle_api_observe_v1") :
            self.backend.index("ggml_backend_memory_api_get_domains_v1")
        ]
        self.assertIn("ggml_backend_noexcept_status", observer)

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

    def test_capture_identity_is_the_owned_cgraph(self) -> None:
        self.assertIn("ggml_graph_capture_source", self.ggml_h)
        self.assertIn("ggml_graph_capture_uid", self.ggml_h)
        view = self.ggml_c[
            self.ggml_c.index("struct ggml_cgraph ggml_graph_view") :
            self.ggml_c.index("void ggml_graph_cpy")
        ]
        self.assertIn("ggml_graph_capture_source", view)
        self.assertIn("view_src", view)
        copy = self.ggml_c[
            self.ggml_c.index("void ggml_graph_cpy") :
            self.ggml_c.index("struct ggml_cgraph * ggml_graph_dup")
        ]
        self.assertIn("Capture identity (uid, view_src) stays with dst", copy)
        splits = self.backend[
            self.backend.index("// set ids for all splits") :
            self.backend.index("enum ggml_status ggml_backend_sched_split_graph_v2")
        ]
        self.assertIn("view_src = NULL", splits)
        self.assertIn("ggml_graph_next_uid", splits)
        getter = self.cuda[
            self.cuda.index("static uint64_t ggml_cuda_graph_get_key") :
            self.cuda.index("static bool ggml_cuda_graph_update_required")
        ]
        self.assertIn("ggml_graph_capture_uid", getter)
        self.assertNotIn("return cgraph->nodes[0];", getter)
        self.assertNotIn("ggml_graph_capture_source(cgraph)", getter)
        props = self.cuda[
            self.cuda.index("static bool ggml_cuda_graph_update_required") :
            self.cuda.index("static enum ggml_status ggml_cuda_graph_update_executable")
        ]
        self.assertIn("property_baseline", props)
        self.assertNotIn("CUDA Graph id", props)
        self.assertIn("bool property_baseline = false", self.common)
        self.assertIn(
            "std::unordered_map<uint64_t, std::unique_ptr<ggml_cuda_graph>>",
            self.common,
        )
        self.assertNotIn("first_node_ptr", self.common)

    def test_capture_is_opt_in_on_owned_cgraph(self) -> None:
        impl = (ROOT / "src/ggml-impl.h").read_text()
        self.assertIn("ggml_graph_set_capture_allowed", impl)
        self.assertIn("ggml_graph_capture_allowed", impl)
        self.assertNotIn("ggml_graph_set_capture_allowed", self.ggml_h)
        self.assertNotIn("ggml_graph_capture_allowed", self.ggml_h)
        impl = self.ggml_c[
            self.ggml_c.index("void ggml_graph_set_capture_allowed") :
            self.ggml_c.index("struct ggml_cgraph ggml_graph_view")
        ]
        self.assertIn("source->capture_allowed = allowed", impl)
        self.assertIn("source != NULL && source->capture_allowed", impl)
        new_graph = self.ggml_c[
            self.ggml_c.index("struct ggml_cgraph * ggml_new_graph_custom") :
            self.ggml_c.index("struct ggml_cgraph * ggml_new_graph(")
        ]
        self.assertIn("capture_allowed =*/ false", new_graph)
        dispatcher = self.cuda[
            self.cuda.index("bool use_cuda_graph             = false;") :
            self.cuda.index("return ggml_cuda_graph_evaluate_and_capture")
        ]
        self.assertIn("ggml_graph_capture_allowed(cgraph)", dispatcher)
        self.assertIn("never enter the uid cache", dispatcher)

    def test_uid_reuse_ignores_input_pointer_churn(self) -> None:
        props = self.cuda[
            self.cuda.index("static bool ggml_cuda_graph_update_required") :
            self.cuda.index("static enum ggml_status ggml_cuda_graph_update_executable")
        ]
        self.assertIn("treating data-pointer churn as a capture miss recaptures", props)
        self.assertIn("Cheap-path: n_nodes + op/type", props)
        cheap = props[
            props.index("if (graph->property_baseline && graph->warmup_complete &&") :
            props.index("The map is keyed by capture uid")
        ]
        self.assertIn("topology_changed", cheap)
        self.assertIn("memcmp(prev.ne, node->ne, sizeof(prev.ne))", cheap)
        self.assertNotIn("prev.data != node->data", cheap)
        self.assertNotIn("node_src_data_ptrs", cheap)

    def test_observation_is_side_effect_free_and_exported_by_cuda_and_hip(self) -> None:
        observer = self.cuda[
            self.cuda.index("static enum ggml_status ggml_backend_cuda_graph_lifecycle_observe") :
            self.cuda.index("static const ggml_backend_graph_lifecycle_api_v1", self.cuda.index("static enum ggml_status ggml_backend_cuda_graph_lifecycle_observe"))
        ]
        self.assertIn("find_cuda_graph", observer)
        self.assertIn("ggml_cuda_graph_get_key", observer)
        self.assertNotIn("->cuda_graph(", observer)
        self.assertNotIn("n_nodes", observer)
        self.assertNotIn("nodes[0]", observer)
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
