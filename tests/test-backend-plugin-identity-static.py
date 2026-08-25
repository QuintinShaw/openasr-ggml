#!/usr/bin/env python3
"""Source-contract checks for verified plugin target discovery.

These checks do not claim physical Vulkan execution. They keep the discovery
ABI behind the verified loader and bind the returned ordinal to the same Vulkan
device order used by the backend registry. Real-provider qualification remains
authoritative for the observed target and driver.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BackendPluginIdentityStaticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (ROOT / "include/ggml-backend.h").read_text()
        cls.registry = (ROOT / "src/ggml-backend-reg.cpp").read_text()
        cls.vulkan = (ROOT / "src/ggml-vulkan/ggml-vulkan.cpp").read_text()

    def test_host_discovery_is_abi_and_provider_verified(self) -> None:
        self.assertIn("ggml_backend_probe_identity_verified_v1_utf8", self.header)
        wrapper = self.registry[
            self.registry.index("ggml_backend_probe_identity_verified_v1_utf8") :
            self.registry.index("ggml_backend_load_best_verified_utf8")
        ]
        self.assertIn("openasr_backend_abi_matches", wrapper)
        self.assertIn("openasr_backend_provider_matches", wrapper)
        self.assertIn("openasr_ggml_backend_target_identity_v1", wrapper)
        self.assertIn("ggml_backend_noexcept_or", wrapper)
        self.assertNotIn("openasr_backend_runtime_matches", wrapper)

    def test_vulkan_identity_is_derived_from_provider_device_ordinal(self) -> None:
        identity = self.vulkan[
            self.vulkan.index("static int openasr_ggml_backend_vulkan_target_identity") :
            self.vulkan.index("extern \"C\" GGML_BACKEND_API int openasr_ggml_backend_probe_v1")
        ]
        self.assertIn("vk_instance.device_indices[device_index]", identity)
        self.assertIn("properties.properties.vendorID", identity)
        self.assertIn("properties.properties.deviceID", identity)
        self.assertIn("properties.properties.pipelineCacheUUID", identity)
        self.assertIn('"vk_caps_%08x_%08x_"', identity)
        self.assertIn("properties.properties.driverVersion", identity)
        self.assertIn("openasr_ggml_backend_target_identity_v1", identity)

    def test_exact_probe_reuses_the_same_identity_producer(self) -> None:
        probe = self.vulkan[
            self.vulkan.index("extern \"C\" GGML_BACKEND_API int openasr_ggml_backend_probe_v1") :
            self.vulkan.index("GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)")
        ]
        self.assertIn("openasr_ggml_backend_vulkan_target_identity", probe)
        self.assertIn("std::strcmp(actual_target, expected_target)", probe)


if __name__ == "__main__":
    unittest.main()
