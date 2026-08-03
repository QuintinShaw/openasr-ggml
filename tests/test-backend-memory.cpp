#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <vector>

static_assert(sizeof(ggml_backend_memory_domain_id_v1) == 24);
static_assert(sizeof(ggml_backend_memory_request_v1) == 88);
static_assert(sizeof(ggml_backend_memory_claim_v1) == 96);
static_assert(sizeof(ggml_backend_memory_quote_v1) == 48);
static_assert(sizeof(ggml_backend_memory_stats_v1) == 152);
static_assert(sizeof(ggml_backend_memory_api_v1) == 64);

static void test_canonical_pci_bdf_encoding() {
    uint8_t encoded[16];
    memset(encoded, 0xff, sizeof(encoded));
    assert(ggml_backend_memory_encode_pci_bdf_v1("ABCD:Ef:1F.7", encoded));
    const uint8_t expected[16] = {
        'P', 'C', 'I', 1, 0xab, 0xcd, 0xef, 0x1f, 0x07,
        0, 0, 0, 0, 0, 0, 0,
    };
    assert(memcmp(encoded, expected, sizeof(encoded)) == 0);

    const char * invalid[] = {
        nullptr,
        "abcd:ef:1f",
        "abcd:ef:20.0",
        "abcd:ef:1f.8",
        "abcd:ef:1f.0-v1",
        "abcd-ef-1f.0",
    };
    for (const char * value : invalid) {
        memset(encoded, 0xff, sizeof(encoded));
        assert(!ggml_backend_memory_encode_pci_bdf_v1(value, encoded));
        assert(std::all_of(std::begin(encoded), std::end(encoded), [](uint8_t byte) {
            return byte == 0;
        }));
    }
}

static void test_request_fingerprint_binds_all_semantics() {
    // The empty-list golden binds the ABI version as well as request_count.
    assert(ggml_backend_memory_request_fingerprint_v1(nullptr, 0) == UINT64_C(0x5420115802dc1402));
    assert(ggml_backend_memory_request_fingerprint_v1(nullptr, 1) == 0);

    int markers[12] = {};
    ggml_backend_memory_request_v1 requests[2] = {};
    for (ggml_backend_memory_request_v1 & request : requests) {
        request.struct_size = sizeof(request);
    }
    requests[0].kind = GGML_BACKEND_MEMORY_REQUEST_BUFFER;
    requests[0].flags = 3;
    requests[0].usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
    requests[0].request_id = 41;
    requests[0].backend = reinterpret_cast<ggml_backend_t>(&markers[0]);
    requests[0].peer_backend = reinterpret_cast<ggml_backend_t>(&markers[1]);
    requests[0].buft = reinterpret_cast<ggml_backend_buffer_type_t>(&markers[2]);
    requests[0].graph = reinterpret_cast<ggml_cgraph *>(&markers[3]);
    requests[0].host_ptr = &markers[4];
    requests[0].requested_bytes = 1234;
    requests[0].currently_allocated_bytes = 567;
    requests[0].max_tensor_bytes = 890;
    requests[1] = requests[0];
    requests[1].request_id = 42;
    requests[1].backend = reinterpret_cast<ggml_backend_t>(&markers[5]);

    const uint64_t baseline = ggml_backend_memory_request_fingerprint_v1(requests, 2);
    assert(baseline != ggml_backend_memory_request_fingerprint_v1(requests, 1));
    std::swap(requests[0], requests[1]);
    assert(baseline != ggml_backend_memory_request_fingerprint_v1(requests, 2));
    std::swap(requests[0], requests[1]);

#define ASSERT_FIELD_BINDS(field, value) do {                                      \
        ggml_backend_memory_request_v1 changed[2] = { requests[0], requests[1] };   \
        changed[0].field = (value);                                                  \
        assert(baseline != ggml_backend_memory_request_fingerprint_v1(changed, 2)); \
    } while (false)

    ASSERT_FIELD_BINDS(kind, GGML_BACKEND_MEMORY_REQUEST_TRANSFER);
    ASSERT_FIELD_BINDS(flags, 4);
    ASSERT_FIELD_BINDS(usage, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ASSERT_FIELD_BINDS(request_id, 99);
    ASSERT_FIELD_BINDS(backend, reinterpret_cast<ggml_backend_t>(&markers[6]));
    ASSERT_FIELD_BINDS(peer_backend, reinterpret_cast<ggml_backend_t>(&markers[7]));
    ASSERT_FIELD_BINDS(buft, reinterpret_cast<ggml_backend_buffer_type_t>(&markers[8]));
    ASSERT_FIELD_BINDS(graph, reinterpret_cast<ggml_cgraph *>(&markers[9]));
    ASSERT_FIELD_BINDS(host_ptr, &markers[10]);
    ASSERT_FIELD_BINDS(requested_bytes, 1235);
    ASSERT_FIELD_BINDS(currently_allocated_bytes, 568);
    ASSERT_FIELD_BINDS(max_tensor_bytes, 891);

#undef ASSERT_FIELD_BINDS
}

static const ggml_backend_memory_api_v1 * cpu_memory_api() {
    ggml_backend_reg_t reg = ggml_backend_cpu_reg();
    auto get_api = reinterpret_cast<ggml_backend_memory_get_api_v1_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MEMORY_API_V1_PROC));
    assert(get_api != nullptr);
    const ggml_backend_memory_api_v1 * api = get_api();
    assert(api != nullptr);
    assert(api->struct_size >= sizeof(*api));
    assert(api->abi_version == GGML_BACKEND_MEMORY_ABI_V1);
    assert(api->get_domains != nullptr);
    assert(api->quote != nullptr);
    assert(api->reserve_private != nullptr);
    assert(api->get_stats != nullptr);
    assert(api->trim != nullptr);
    assert(api->quarantine != nullptr);
    return api;
}

int main() {
    test_canonical_pci_bdf_encoding();
    test_request_fingerprint_binds_all_semantics();

    const ggml_backend_memory_api_v1 * api = cpu_memory_api();
    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend != nullptr);
    ggml_backend_dev_t device = ggml_backend_get_device(backend);

    uint32_t domain_count = 0;
    assert(api->get_domains(device, nullptr, &domain_count) == GGML_STATUS_SUCCESS);
    assert(domain_count == 1);
    ggml_backend_memory_domain_v1 domain = {};
    domain.struct_size = sizeof(domain);
    domain_count = 1;
    assert(api->get_domains(device, &domain, &domain_count) == GGML_STATUS_SUCCESS);
    assert(domain.id.kind == GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE);

    ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
    ggml_tensor * out = ggml_mul_mat(ctx, lhs, rhs);
    ggml_set_input(lhs);
    ggml_set_input(rhs);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_t backends[] = {backend};
    ggml_backend_buffer_type_t bufts[] = {ggml_backend_get_default_buffer_type(backend)};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 1, 32, false, false);
    assert(sched != nullptr);

    ggml_backend_sched_memory_plan_t plan = nullptr;
    assert(ggml_backend_sched_memory_plan_create_v1(sched, graph, &plan) == GGML_STATUS_SUCCESS);
    assert(plan != nullptr);
    // A frozen plan has exclusive ownership: ordinary allocation cannot race
    // and commit a different graph between quote and reserve.
    assert(!ggml_backend_sched_alloc_graph(sched, graph));

    const uint32_t item_count = ggml_backend_sched_memory_plan_get_item_count_v1(plan);
    assert(item_count >= 2); // at least one gallocr buffer plus graph-private work
    std::vector<ggml_backend_memory_request_v1> requests(item_count);
    bool saw_buffer = false;
    bool saw_graph_private = false;
    for (uint32_t i = 0; i < item_count; ++i) {
        requests[i].struct_size = sizeof(requests[i]);
        assert(ggml_backend_sched_memory_plan_get_item_v1(plan, i, &requests[i]));
        assert(requests[i].request_id == i + 1);
        assert(requests[i].backend == backend);
        saw_buffer |= requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_BUFFER;
        saw_graph_private |= requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE;
    }
    assert(saw_buffer && saw_graph_private);

    ggml_backend_memory_quote_v1 quote = {};
    quote.struct_size = sizeof(quote);
    uint32_t claim_count = 0;
    assert(api->quote(requests.data(), item_count, &quote, nullptr, &claim_count) == GGML_STATUS_SUCCESS);
    assert(claim_count >= 1);
    assert(quote.stats_generation != 0);
    assert(quote.quote_token != 0);
    assert(quote.request_fingerprint != 0);
    assert(quote.flags == 0);
    assert(quote.residual_flags == 0);
    assert(quote.residual_request_count == 0);

    std::vector<ggml_backend_memory_claim_v1> claims(claim_count);
    for (auto & claim : claims) claim.struct_size = sizeof(claim);
    assert(api->quote(requests.data(), item_count, &quote, claims.data(), &claim_count) == GGML_STATUS_SUCCESS);
    for (const auto & claim : claims) {
        assert(claim.struct_size == sizeof(claim));
        assert(claim.committed_after_upper_bytes >= claim.payload_requested_bytes);
        assert(claim.resident_after_upper_bytes <= claim.committed_after_upper_bytes);
    }

    uint32_t actual_count = 0;
    assert(api->reserve_private(requests.data(), item_count, &quote, nullptr, &actual_count) == GGML_STATUS_SUCCESS);
    std::vector<ggml_backend_memory_claim_v1> actual(std::max(1u, actual_count));
    for (auto & claim : actual) claim.struct_size = sizeof(claim);

    // Any backend-retained-memory mutation invalidates the token. This proves
    // reserve is not accepting a stale enumeration snapshot.
    assert(api->trim(backend, 0) == GGML_STATUS_SUCCESS);
    uint32_t stale_count = actual_count;
    assert(api->reserve_private(requests.data(), item_count, &quote, actual.data(), &stale_count) == GGML_STATUS_FAILED);

    quote = {};
    quote.struct_size = sizeof(quote);
    claim_count = static_cast<uint32_t>(claims.size());
    assert(api->quote(requests.data(), item_count, &quote, claims.data(), &claim_count) == GGML_STATUS_SUCCESS);
    actual_count = static_cast<uint32_t>(actual.size());
    assert(api->reserve_private(requests.data(), item_count, &quote, actual.data(), &actual_count) == GGML_STATUS_SUCCESS);

    assert(ggml_backend_sched_memory_plan_commit_v1(plan) == GGML_STATUS_SUCCESS);
    ggml_backend_sched_memory_plan_free_v1(plan);
    assert(out->data != nullptr);

    // A same-shape graph reuses the resident gallocr arena. Its next quote
    // reports the existing commitment and zero replacement peak instead of
    // charging the arena as a second resident owner.
    ggml_context * ctx2 = ggml_init(params);
    assert(ctx2 != nullptr);
    ggml_tensor * lhs2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, 128, 128);
    ggml_tensor * rhs2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, 128, 128);
    ggml_tensor * out2 = ggml_mul_mat(ctx2, lhs2, rhs2);
    ggml_set_input(lhs2);
    ggml_set_input(rhs2);
    ggml_set_output(out2);
    ggml_cgraph * graph2 = ggml_new_graph_custom(ctx2, 32, false);
    ggml_build_forward_expand(graph2, out2);
    ggml_backend_sched_memory_plan_t reuse_plan = nullptr;
    assert(ggml_backend_sched_memory_plan_create_v1(sched, graph2, &reuse_plan) == GGML_STATUS_SUCCESS);
    const uint32_t reuse_count = ggml_backend_sched_memory_plan_get_item_count_v1(reuse_plan);
    std::vector<ggml_backend_memory_request_v1> reuse_requests(reuse_count);
    for (uint32_t i = 0; i < reuse_count; ++i) {
        reuse_requests[i].struct_size = sizeof(reuse_requests[i]);
        assert(ggml_backend_sched_memory_plan_get_item_v1(reuse_plan, i, &reuse_requests[i]));
    }
    ggml_backend_memory_quote_v1 reuse_quote = {};
    reuse_quote.struct_size = sizeof(reuse_quote);
    uint32_t reuse_claim_count = 0;
    assert(api->quote(reuse_requests.data(), reuse_count, &reuse_quote, nullptr, &reuse_claim_count) == GGML_STATUS_SUCCESS);
    std::vector<ggml_backend_memory_claim_v1> reuse_claims(reuse_claim_count);
    for (auto & claim : reuse_claims) claim.struct_size = sizeof(claim);
    assert(api->quote(reuse_requests.data(), reuse_count, &reuse_quote, reuse_claims.data(), &reuse_claim_count) == GGML_STATUS_SUCCESS);
    bool saw_reused_arena = false;
    for (const auto & claim : reuse_claims) {
        if (claim.request_id != 0 && claim.committed_before_bytes > 0) {
            saw_reused_arena = true;
            assert(claim.committed_after_upper_bytes == claim.committed_before_bytes);
            assert(claim.commit_peak_extra_upper_bytes == 0);
        }
    }
    assert(saw_reused_arena);
    ggml_backend_sched_memory_plan_free_v1(reuse_plan);
    ggml_free(ctx2);

    ggml_backend_memory_stats_v1 stats = {};
    stats.struct_size = sizeof(stats);
    uint32_t stats_count = 1;
    assert(api->get_stats(device, backend, &stats, &stats_count) == GGML_STATUS_SUCCESS);
    assert(stats_count == 1);
    assert(stats.total_bytes >= stats.device_free_bytes);
    assert(stats.health == GGML_BACKEND_MEMORY_HEALTHY);

    ggml_backend_memory_quarantine_v1 quarantine = {};
    quarantine.struct_size = sizeof(quarantine);
    quarantine.ggml_status = GGML_STATUS_DEVICE_LOST;
    assert(api->quarantine(backend, &quarantine) == GGML_STATUS_SUCCESS);
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_BACKEND_POISONED);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
