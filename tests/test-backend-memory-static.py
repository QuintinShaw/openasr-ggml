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


def read_source(relative_path: str) -> str:
    """Read repository sources using their declared, platform-independent encoding."""
    return (ROOT / relative_path).read_text(encoding="utf-8")


def function_body(source: str, name: str, next_name: str) -> str:
    pattern = rf"static enum ggml_status {re.escape(name)}\(.*?(?=static enum ggml_status {re.escape(next_name)}\()"
    match = re.search(pattern, source, flags=re.DOTALL)
    if match is None:
        raise AssertionError(f"could not find {name}")
    return match.group(0)


class BackendMemoryStaticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = read_source("include/ggml-backend.h")
        cls.alloc_header = read_source("include/ggml-alloc.h")
        cls.core = read_source("src/ggml-backend.cpp")
        cls.meta = read_source("src/ggml-backend-meta.cpp")
        cls.allocator = read_source("src/ggml-alloc.c")
        cls.impl = read_source("src/ggml-impl.h")
        cls.cpu = read_source("src/ggml-cpu/ggml-cpu.cpp")
        cls.blas = read_source("src/ggml-blas/ggml-blas.cpp")
        cls.metal = read_source("src/ggml-metal/ggml-metal.cpp")
        cls.cuda = read_source("src/ggml-cuda/ggml-cuda.cu")
        cls.vulkan = read_source("src/ggml-vulkan/ggml-vulkan.cpp")

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
        self.assertIn("ggml_backend_vk_unique_nonzero_device_uuid", vk_domain)
        self.assertIn("id_props.deviceUUID.data()", vk_domain)
        self.assertNotIn("memcpy(id.physical_device_uuid, &device->idx", vk_domain)

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
        self.assertRegex(
            self.cuda,
            r"error\s*==\s*cudaErrorMemoryAllocation\s*\?\s*GGML_STATUS_ALLOC_FAILED",
        )
        self.assertIn("catch (const ggml_backend_exception & error)", self.cuda)
        self.assertRegex(
            self.cuda,
            r"error\s*==\s*CUDA_ERROR_OUT_OF_MEMORY\s*\?\s*GGML_STATUS_ALLOC_FAILED",
        )
        self.assertIn("catch (...) {", self.cuda)
        self.assertNotIn('GGML_ABORT(GGML_CUDA_NAME " error")', self.cuda)

        common = read_source("src/ggml-cuda/common.cuh")
        hip = read_source("src/ggml-cuda/vendors/hip.h")
        musa = read_source("src/ggml-cuda/vendors/musa.h")
        self.assertIn("#define GGML_ABORT(...) ggml_cuda_abort", common)
        self.assertIn("#define cudaErrorMemoryAllocation hipErrorOutOfMemory", hip)
        self.assertIn("#define cudaErrorNoDevice hipErrorNoDevice", hip)
        self.assertIn("#define cudaErrorAssert hipErrorAssert", hip)
        self.assertIn("#define cudaErrorNoDevice musaErrorNoDevice", musa)
        self.assertIn("#define cudaErrorAssert musaErrorAssert", musa)
        self.assertIn("ggml_cuda_status(err_)", hip)
        self.assertNotIn("HipVMM Failure", hip)
        self.assertIn("if (err == cudaErrorNoDevice)", self.cuda)
        self.assertIn("CUDA_CHECK(err);", self.cuda)

        stats = function_body(
            self.cuda,
            "ggml_backend_cuda_memory_get_stats",
            "ggml_backend_cuda_memory_trim",
        )
        self.assertIn("value.last_ggml_status", stats)
        self.assertIn("value.last_native_error", stats)
        self.assertIn("GGML_BACKEND_MEMORY_DEVICE_LOST", stats)

    def test_vulkan_provider_failures_are_typed_and_health_is_observable(self) -> None:
        self.assertIn("#define GGML_ABORT(...) ggml_vk_abort", self.vulkan)
        self.assertIn("ggml_vk_status_boundary", self.vulkan)
        self.assertIn("ggml_vk_buffer_boundary", self.vulkan)
        self.assertNotIn("vk_pipeline_stats_filter.clear();\n    throw;", self.vulkan)
        self.assertIn("static std::mutex vk_instance_init_mutex", self.vulkan)
        self.assertIn("static bool vk_instance_initialized = false", self.vulkan)
        self.assertIn("ggml_vk_reset_failed_instance", self.vulkan)
        self.assertNotIn("std::call_once(vk_instance_init_flag", self.vulkan)
        self.assertIn(
            "const enum ggml_status status = ggml_vk_instance_init_impl()",
            self.vulkan,
        )
        self.assertIn("if (status != GGML_STATUS_SUCCESS)", self.vulkan)

        stats = function_body(
            self.vulkan,
            "ggml_backend_vk_memory_get_stats",
            "ggml_backend_vk_memory_trim",
        )
        self.assertIn("value.last_ggml_status", stats)
        self.assertIn("value.last_native_error", stats)
        self.assertIn("value.quarantine_generation", stats)
        self.assertIn("GGML_BACKEND_MEMORY_DEVICE_LOST", stats)

        quarantine = self.vulkan[
            self.vulkan.index("static enum ggml_status ggml_backend_vk_memory_quarantine") :
            self.vulkan.index("static const ggml_backend_memory_api_v1 * ggml_backend_vk_memory_get_api_v1")
        ]
        self.assertIn("ctx->memory_quarantined = true", quarantine)
        self.assertNotIn("ctx->device->poisoned.store(true)", quarantine)

    def test_optional_plugin_callbacks_use_common_noexcept_trampolines(self) -> None:
        self.assertIn("ggml_backend_set_n_threads_if_supported", self.header)
        self.assertIn("ggml_backend_dev_pci_vendor_id", self.header)
        thread_trampoline = self.core[
            self.core.index("enum ggml_status ggml_backend_set_n_threads_if_supported") :
            self.core.index("uint32_t ggml_backend_dev_pci_vendor_id")
        ]
        self.assertIn("ggml_backend_noexcept_status", thread_trampoline)
        pci_trampoline = self.core[
            self.core.index("uint32_t ggml_backend_dev_pci_vendor_id") :
            self.core.index("// multi-buffer buffer")
        ]
        self.assertIn("ggml_backend_noexcept_or<uint32_t>", pci_trampoline)

    def test_reset_and_owner_release_use_status_seams(self) -> None:
        for symbol in (
            "ggml_backend_buffer_reset_status",
            "ggml_backend_free_status",
            "ggml_backend_event_free_status",
            "ggml_backend_sched_free_status",
        ):
            self.assertIn(symbol, self.header)
            self.assertIn(symbol, self.core)

        reset = self.core[
            self.core.index("enum ggml_status ggml_backend_buffer_reset_status") :
            self.core.index("static enum ggml_status ggml_backend_buffer_copy_tensor_status")
        ]
        self.assertIn("ggml_backend_noexcept_status", reset)
        self.assertNotIn("ggml_backend_noexcept_void", reset)

        alloc_graph = self.allocator[
            self.allocator.index("enum ggml_status ggml_gallocr_alloc_graph_v2") :
            self.allocator.index("bool ggml_gallocr_alloc_graph(")
        ]
        self.assertIn("ggml_vbuffer_reset", alloc_graph)
        self.assertIn("return reset_status", alloc_graph)
        self.assertIn("ggml_backend_buffer_reset_status", self.meta)

        scheduler_free = self.core[
            self.core.index("enum ggml_status ggml_backend_sched_free_status") :
            self.core.index("void ggml_backend_sched_reset")
        ]
        self.assertIn("ggml_backend_event_free_status", scheduler_free)
        self.assertIn("ggml_gallocr_free_status", scheduler_free)
        self.assertIn("return status", scheduler_free)

    def test_gpu_provider_entry_validation_is_typed(self) -> None:
        cuda_ranges = (
            ("ggml_backend_cuda_buffer_init_tensor", "ggml_backend_cuda_buffer_memset_tensor"),
            ("ggml_backend_cuda_set_tensor_async", "ggml_backend_cuda_get_tensor_async"),
            ("ggml_backend_cuda_get_tensor_async", "ggml_backend_cuda_set_tensor_2d_async"),
            ("ggml_backend_cuda_set_tensor_2d_async", "ggml_backend_cuda_get_tensor_2d_async"),
            ("ggml_backend_cuda_get_tensor_2d_async", "ggml_backend_cuda_cpy_tensor_async"),
        )
        for current, following in cuda_ranges:
            body = self.cuda[
                self.cuda.index(f"static enum ggml_status {current}") :
                self.cuda.index(following, self.cuda.index(f"static enum ggml_status {current}") + 1)
            ]
            self.assertNotIn("GGML_ASSERT", body)
            self.assertIn("GGML_STATUS_FAILED", body)

        vk_init = self.vulkan[
            self.vulkan.index("static enum ggml_status ggml_backend_vk_buffer_init_tensor") :
            self.vulkan.index("static void ggml_backend_vk_buffer_memset_tensor")
        ]
        self.assertNotIn("GGML_ASSERT", vk_init)
        self.assertIn("GGML_STATUS_FAILED", vk_init)
        for current, following in (
            ("ggml_backend_vk_set_tensor_2d_async", "ggml_backend_vk_set_tensor_async"),
            ("ggml_backend_vk_get_tensor_2d_async", "ggml_backend_vk_get_tensor_async"),
        ):
            body = self.vulkan[
                self.vulkan.index(f"static enum ggml_status {current}") :
                self.vulkan.index(f"static enum ggml_status {following}")
            ]
            self.assertNotIn("GGML_ASSERT", body)
            self.assertIn("GGML_STATUS_FAILED", body)
            self.assertIn("ggml_vk_status_boundary", body)

        self.assertIn("ggml_backend_set_abort_callback_status", self.core)
        self.assertIn("ggml_backend_cuda_set_abort_callback_status", self.cuda)
        self.assertIn("ggml_backend_vk_set_abort_callback_status", self.vulkan)

        cuda_alloc = self.cuda[
            self.cuda.index("static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer") :
            self.cuda.index("static size_t ggml_backend_cuda_buffer_type_get_alignment")
        ]
        self.assertLess(cuda_alloc.index("try {"), cuda_alloc.index("ggml_cuda_set_device"))
        vk_host_alloc = self.vulkan[
            self.vulkan.index("static ggml_backend_buffer_t ggml_backend_vk_host_buffer_type_alloc_buffer") :
            self.vulkan.index("static size_t ggml_backend_vk_host_buffer_type_get_alignment")
        ]
        self.assertLess(vk_host_alloc.index("try {"), vk_host_alloc.index("ggml_backend_cpu_buffer_from_ptr"))
        self.assertLess(vk_host_alloc.index("ggml_backend_cpu_buffer_from_ptr"), vk_host_alloc.index("catch ("))

    def test_scheduler_and_allocator_host_failures_are_fallible(self) -> None:
        constructor = self.core[
            self.core.index("ggml_backend_sched_t ggml_backend_sched_new") :
            self.core.index("void ggml_backend_sched_free")
        ]
        self.assertIn("ggml_hash_set_try_new", constructor)
        self.assertIn("ggml_graph_overhead_custom_try", constructor)
        self.assertIn("ggml_backend_sched_free(sched)", constructor)
        self.assertNotIn("GGML_ASSERT", constructor)
        self.assertIn("ggml_backend_sched_split_graph_v2", self.header)

        reserve = self.allocator[
            self.allocator.index("static bool ggml_gallocr_reserve_n_impl") :
            self.allocator.index("void ggml_gallocr_reserve_n_size")
        ]
        self.assertLess(
            reserve.index("ggml_vbuffer_alloc_v2"),
            reserve.index("ggml_gallocr_detach_graph_tensors_v1"),
        )
        self.assertIn("ggml_vbuffer_free_status(retired)", reserve)
        self.assertIn("ggml_gallocr_measure_commit_v2", self.alloc_header)
        self.assertIn("GGML_GALLOCR_MEASURE_COMMIT_MAY_HAVE_MUTATED", self.alloc_header)
        self.assertIn("GGML_GALLOCR_MEASURE_COMMIT_RELEASE_UNPROVEN", self.alloc_header)

        hash_insert = self.impl[
            self.impl.index(
                "static size_t ggml_hash_find_or_insert(struct ggml_hash_set * hash_set, struct ggml_tensor * key) {"
            ) :
            self.impl.index("// computation graph")
        ]
        self.assertIn("return GGML_HASHSET_FULL", hash_insert)
        self.assertNotIn('GGML_ABORT("fatal error")', hash_insert)

    def test_vulkan_trim_attempts_every_release_and_queue_failure_is_typed(self) -> None:
        trim = function_body(
            self.vulkan,
            "ggml_backend_vk_memory_trim",
            "ggml_backend_vk_memory_quarantine",
        )
        self.assertIn("first_failure", trim)
        self.assertEqual(trim.count("release([&]()"), 5)
        self.assertIn("return first_failure", trim)

        queue_selector = self.vulkan[
            self.vulkan.index("static uint32_t ggml_vk_find_queue_family_index") :
            self.vulkan.index("static void ggml_vk_create_queue")
        ]
        self.assertIn("throw ggml_backend_exception", queue_selector)
        self.assertNotIn("abort()", queue_selector)

        preallocate = self.vulkan[
            self.vulkan.index(
                "static void ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {"
            ) :
            self.vulkan.index(
                "static void ggml_vk_compute_forward",
                self.vulkan.index(
                    "static void ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {"
                ),
            )
        ]
        self.assertNotIn("abort()", preallocate)
        self.assertNotIn('GGML_ABORT("fatal error")', preallocate)

        for function_name, next_name in (
            ("ggml_backend_vk_set_tensor_2d_async", "ggml_backend_vk_set_tensor_async"),
            ("ggml_backend_vk_get_tensor_2d_async", "ggml_backend_vk_get_tensor_async"),
        ):
            transfer = function_body(self.vulkan, function_name, next_name)
            self.assertIn("size > SIZE_MAX / n_copies", transfer)
            self.assertLess(
                transfer.index("size > SIZE_MAX / n_copies"),
                transfer.index("size * n_copies"),
            )

    def test_cuda_release_failures_reach_the_status_bearing_free_seam(self) -> None:
        buffer_context = self.cuda[
            self.cuda.index("struct ggml_backend_cuda_buffer_context") :
            self.cuda.index("static bool ggml_backend_buffer_is_cuda")
        ]
        self.assertIn("void release()", buffer_context)
        self.assertIn("CUDA_CHECK(cudaFree(dev_ptr))", buffer_context)
        self.assertLess(
            buffer_context.index("ctx->release()"),
            buffer_context.index("delete ctx"),
        )

        backend_free = self.cuda[
            self.cuda.index("static void ggml_backend_cuda_free") :
            self.cuda.index("static enum ggml_status ggml_backend_cuda_completion_status")
        ]
        self.assertLess(backend_free.index("cuda_ctx->release()"), backend_free.index("delete cuda_ctx"))

        context_release = self.cuda[
            self.cuda.index("void ggml_backend_cuda_context::release()") :
            self.cuda.index("ggml_backend_cuda_context::~ggml_backend_cuda_context()")
        ]
        self.assertIn("it->second->release()", context_release)
        self.assertIn("concurrent_stream_context.reset()", context_release)
        self.assertIn("pools[i][j]->release()", context_release)
        self.assertLess(context_release.index("pools[i][j]->release()"), context_release.index("cudaStreamDestroy"))

        vk_buffer_release = self.vulkan[
            self.vulkan.index("struct vk_buffer_struct") :
            self.vulkan.index("struct vk_subbuffer")
        ]
        self.assertIn("void release()", vk_buffer_release)
        self.assertIn("device->device.destroyBuffer(buffer)", vk_buffer_release)
        self.assertIn("device->device.freeMemory(device_memory)", vk_buffer_release)

        vk_backend_free = self.vulkan[
            self.vulkan.index("static void ggml_backend_vk_free(ggml_backend_t backend) {") :
            self.vulkan.index("static ggml_backend_buffer_type_t ggml_backend_vk_get_default_buffer_type")
        ]
        self.assertIn("const enum ggml_status status = ggml_vk_cleanup(ctx)", vk_backend_free)
        self.assertIn("throw ggml_backend_exception", vk_backend_free)

        vk_cleanup = self.vulkan[
            self.vulkan.index("static enum ggml_status ggml_vk_cleanup(") :
            self.vulkan.index("static int ggml_vk_get_device_count")
        ]
        self.assertIn("handle = VK_NULL_HANDLE", vk_cleanup)
        self.assertIn("ctx->gc.events.erase", vk_cleanup)
        self.assertIn("ctx->descriptor_pools.erase", vk_cleanup)
        self.assertIn("ctx->transfer_semaphore.s", vk_cleanup)

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
