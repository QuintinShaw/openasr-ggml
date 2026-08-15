#!/usr/bin/env python3
"""Source-contract checks for GPU backends unavailable on the test host.

These checks do not claim platform compilation. They guard the easy-to-regress
parts of the additive ABI: complete transaction binding, physical identity,
exact/provisional disclosure, and typed failure mapping. Native CUDA/HIP/Vulkan
CI remains authoritative.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def function_body(source: str, name: str, next_name: str) -> str:
    pattern = rf"static enum ggml_status {re.escape(name)}\(.*?(?=static enum ggml_status {re.escape(next_name)}\()"
    match = re.search(pattern, source, flags=re.DOTALL)
    if match is None:
        raise AssertionError(f"could not find {name}")
    return match.group(0)


class BackendMemoryStaticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (ROOT / "include/ggml-backend.h").read_text()
        cls.core = (ROOT / "src/ggml-backend.cpp").read_text()
        cls.cpu = (ROOT / "src/ggml-cpu/ggml-cpu.cpp").read_text()
        cls.blas = (ROOT / "src/ggml-blas/ggml-blas.cpp").read_text()
        cls.metal = (ROOT / "src/ggml-metal/ggml-metal.cpp").read_text()
        cls.cuda = (ROOT / "src/ggml-cuda/ggml-cuda.cu").read_text()
        cls.vulkan = (ROOT / "src/ggml-vulkan/ggml-vulkan.cpp").read_text()

    def test_all_provider_tokens_use_complete_shared_fingerprint(self) -> None:
        helper = self.core[
            self.core.index("uint64_t ggml_backend_memory_request_fingerprint_v1") :
            self.core.index("static int ggml_backend_memory_hex_nibble_v1")
        ]
        for field in (
            "GGML_BACKEND_MEMORY_ABI_V1",
            "request_count",
            "index",
            "request.kind",
            "request.flags",
            "request.usage",
            "request.request_id",
            "request.backend",
            "request.peer_backend",
            "request.buft",
            "request.graph",
            "request.host_ptr",
            "request.requested_bytes",
            "request.currently_allocated_bytes",
            "request.max_tensor_bytes",
        ):
            self.assertIn(field, helper)
        for provider in (self.cpu, self.blas, self.metal, self.cuda, self.vulkan):
            self.assertGreaterEqual(
                provider.count("ggml_backend_memory_request_fingerprint_v1"), 2
            )
        self.assertIn(
            "plan->fingerprint = ggml_backend_memory_request_fingerprint_v1",
            self.core,
        )

    def test_blas_quotes_and_reserves_its_reusable_conversion_workspace(self) -> None:
        quote = function_body(
            self.blas,
            "ggml_backend_blas_memory_quote",
            "ggml_backend_blas_memory_reserve_private",
        )
        reserve = function_body(
            self.blas,
            "ggml_backend_blas_memory_reserve_private",
            "ggml_backend_blas_memory_get_stats",
        )
        self.assertIn("ggml_backend_blas_graph_workspace", quote)
        self.assertIn("GGML_BACKEND_MEMORY_CLAIM_REUSABLE_WORKSPACE", quote)
        self.assertIn("new (std::nothrow) char[maximum_workspace]", reserve)
        self.assertIn("ctx->memory_generation++", reserve)
        self.assertIn("GGML_BACKEND_MEMORY_API_V1_PROC", self.blas)

    def test_reserve_private_contract_is_explicitly_failure_atomic(self) -> None:
        self.assertIn("Failure-atomic transactional hook", self.header)
        self.assertIn("must leave", self.header)
        self.assertIn("validation-only/no-op", self.header)

    def test_metal_graph_private_is_exact_zero_with_opaque_headroom(self) -> None:
        quote = function_body(
            self.metal,
            "ggml_backend_metal_memory_quote",
            "ggml_backend_metal_memory_reserve_private",
        )
        self.assertIn(
            "GGML_BACKEND_MEMORY_QUOTE_OPAQUE_DRIVER_COSTS_REQUIRE_DOMAIN_HEADROOM",
            quote,
        )
        self.assertIn("GGML_BACKEND_MEMORY_CLAIM_EXACT", quote)
        self.assertIn("claim.request_id = requests[i].request_id", quote)
        self.assertNotIn("GGML_BACKEND_MEMORY_QUOTE_PROVISIONAL", quote)
        self.assertNotIn("GGML_BACKEND_MEMORY_QUOTE_HAS_RESIDUAL_UNCERTAINTY", quote)
        self.assertNotIn("GGML_BACKEND_MEMORY_RESIDUAL_BACKEND_PRIVATE", quote)
        self.assertIn("quote->residual_flags = 0", quote)
        self.assertIn("quote->residual_request_count = 0", quote)

    def test_cuda_and_vulkan_share_canonical_pci_bdf_identity(self) -> None:
        cuda_domain = self.cuda[
            self.cuda.index("static ggml_backend_memory_domain_id_v1 ggml_backend_cuda_memory_domain") :
            self.cuda.index("static uint64_t ggml_backend_cuda_memory_generation")
        ]
        self.assertIn("cudaDeviceGetPCIBusId", cuda_domain)
        self.assertIn("ggml_backend_memory_encode_pci_bdf_v1", cuda_domain)
        self.assertNotIn("pciDomainID", cuda_domain)
        self.assertNotIn("pciBusID", cuda_domain)
        self.assertNotIn("pciDeviceID", cuda_domain)
        self.assertNotIn("memcpy(id.physical_device_uuid", cuda_domain)

        vk_domain = self.vulkan[
            self.vulkan.index("static ggml_backend_memory_domain_id_v1 ggml_backend_vk_memory_domain") :
            self.vulkan.index("static bool ggml_backend_vk_memory_type_for_buffer")
        ]
        self.assertIn("ggml_backend_vk_get_device_pci_id", vk_domain)
        self.assertIn("ggml_backend_memory_encode_pci_bdf_v1", vk_domain)
        self.assertNotIn("deviceUUID", vk_domain)

        encoder = self.core[
            self.core.index("bool ggml_backend_memory_encode_pci_bdf_v1") :
            self.core.index("// backend buffer type")
        ]
        self.assertIn("physical_device_uuid[0] = 'P'", encoder)
        self.assertIn("physical_device_uuid[3] = 1", encoder)
        self.assertIn("memset(physical_device_uuid, 0, 16)", encoder)

    def test_cuda_exposes_partial_quote_instead_of_rejecting_graph_private(self) -> None:
        quote = function_body(
            self.cuda,
            "ggml_backend_cuda_memory_quote",
            "ggml_backend_cuda_memory_reserve_private",
        )
        self.assertIn("GGML_BACKEND_MEMORY_QUOTE_PROVISIONAL", quote)
        self.assertIn("GGML_BACKEND_MEMORY_RESIDUAL_BACKEND_PRIVATE", quote)
        self.assertIn("GGML_BACKEND_MEMORY_CLAIM_PROVISIONAL", quote)
        self.assertNotRegex(
            quote,
            r"kind == GGML_BACKEND_MEMORY_REQUEST_(?:BUFFER|GRAPH_PRIVATE)\) return GGML_STATUS_FAILED",
        )
        self.assertIn("GGML_BACKEND_MEMORY_API_V1_PROC", self.cuda)

    def test_cuda_hip_oom_is_typed_and_compute_catches_it(self) -> None:
        self.assertIn("cudaErrorMemoryAllocation ? GGML_STATUS_ALLOC_FAILED", self.cuda)
        self.assertIn("hipErrorOutOfMemory ? GGML_STATUS_ALLOC_FAILED", self.cuda)
        self.assertIn("catch (const ggml_cuda_typed_error & error)", self.cuda)
        self.assertIn("error == CUDA_ERROR_OUT_OF_MEMORY ? GGML_STATUS_ALLOC_FAILED", self.cuda)

    def test_vulkan_quotes_buffer_requirements_and_discloses_private_residual(self) -> None:
        quote = function_body(
            self.vulkan,
            "ggml_backend_vk_memory_quote",
            "ggml_backend_vk_memory_reserve_private",
        )
        self.assertIn("ggml_backend_vk_memory_buffer_commitment", quote)
        self.assertIn("currently_allocated_bytes", quote)
        self.assertIn("GGML_BACKEND_MEMORY_RESIDUAL_BACKEND_PRIVATE", quote)
        self.assertNotIn(
            "kind == GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE) return GGML_STATUS_FAILED",
            quote,
        )
        self.assertIn("GGML_BACKEND_MEMORY_API_V1_PROC", self.vulkan)

    def test_vulkan_quote_and_allocator_share_immutable_heap_binding(self) -> None:
        context = self.vulkan[
            self.vulkan.index("struct ggml_backend_vk_buffer_type_context") :
            self.vulkan.index("struct vk_queue;")
        ]
        self.assertIn("uint32_t allocation_heap_index", context)

        selector = self.vulkan[
            self.vulkan.index("static uint32_t ggml_vk_select_default_allocation_heap") :
            self.vulkan.index("static vk_buffer ggml_vk_create_buffer(")
        ]
        self.assertIn("largest_heap_with", selector)
        self.assertIn("device->uma", selector)
        self.assertIn("getBufferMemoryRequirements(probe)", selector)
        self.assertIn("requirements.memoryTypeBits", selector)

        allocator_start = self.vulkan.index("static vk_buffer ggml_vk_create_buffer(")
        allocator = self.vulkan[
            allocator_start :
            self.vulkan.index("static void ggml_vk_destroy_buffer", allocator_start)
        ]
        self.assertIn("memory_type.heapIndex == allocation_heap_index", self.vulkan)
        self.assertIn(
            "ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags, allocation_heap_index)",
            allocator,
        )
        self.assertIn(
            "ctx->device, size, ctx->allocation_heap_index",
            self.vulkan,
        )

        resolver = self.vulkan[
            self.vulkan.index("static bool ggml_backend_vk_memory_type_for_buffer") :
            self.vulkan.index("static enum ggml_status ggml_backend_vk_memory_buffer_commitment")
        ]
        self.assertIn("allocation_heap_index", resolver)
        self.assertIn("ggml_vk_find_memory_properties", resolver)

        quote = function_body(
            self.vulkan,
            "ggml_backend_vk_memory_quote",
            "ggml_backend_vk_memory_reserve_private",
        )
        self.assertIn(
            "buft_ctx, requests[i].requested_bytes",
            quote,
        )

    def test_vulkan_quote_generation_excludes_live_heap_usage(self) -> None:
        generation = self.vulkan[
            self.vulkan.index("static bool ggml_backend_vk_memory_generation") :
            self.vulkan.index("static enum ggml_status ggml_backend_vk_memory_quote")
        ]
        self.assertIn("*generation = 1", generation)
        self.assertNotIn("heapBudget", generation)
        self.assertNotIn("heapUsage", generation)
        stats = function_body(
            self.vulkan,
            "ggml_backend_vk_memory_get_stats",
            "ggml_backend_vk_memory_trim",
        )
        self.assertIn(
            "ggml_backend_vk_memory_generation(device, &stats_generation)", stats
        )
        self.assertNotIn(
            "has_budget && !ggml_backend_vk_memory_generation", stats
        )

    def test_vulkan_oom_and_device_loss_are_typed(self) -> None:
        mapper = self.vulkan[
            self.vulkan.index("static enum ggml_status ggml_vk_status") :
            self.vulkan.index("#define VK_CHECK")
        ]
        self.assertIn("eErrorOutOfDeviceMemory", mapper)
        self.assertIn("GGML_STATUS_ALLOC_FAILED", mapper)
        self.assertIn("eErrorDeviceLost", mapper)
        self.assertIn("GGML_STATUS_DEVICE_LOST", mapper)


if __name__ == "__main__":
    unittest.main()
