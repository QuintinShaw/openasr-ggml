#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef GGML_BACKEND_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BACKEND_BUILD
#            define GGML_BACKEND_API __declspec(dllexport) extern
#        else
#            define GGML_BACKEND_API __declspec(dllimport) extern
#        endif
#    else
#        define GGML_BACKEND_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define GGML_BACKEND_API extern
#endif

#ifdef  __cplusplus
extern "C" {
#endif

    typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
    typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
    typedef struct ggml_backend_event * ggml_backend_event_t;
    typedef struct ggml_backend * ggml_backend_t;
    typedef void * ggml_backend_graph_plan_t;
    typedef struct ggml_backend_reg * ggml_backend_reg_t;
    typedef struct ggml_backend_device * ggml_backend_dev_t;


    //
    // Backend buffer type
    //

    GGML_API const char *          ggml_backend_buft_name          (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer  (ggml_backend_buffer_type_t buft, size_t size);
    GGML_API size_t                ggml_backend_buft_get_alignment (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_max_size  (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
    GGML_API bool                  ggml_backend_buft_is_host       (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_dev_t    ggml_backend_buft_get_device    (ggml_backend_buffer_type_t buft);

    //
    // Backend buffer
    //

    enum ggml_backend_buffer_usage {
        GGML_BACKEND_BUFFER_USAGE_ANY = 0,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
        GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,
    };

    // Optional, versioned physical-memory accounting extension. Backends expose
    // the entry point through ggml_backend_reg_get_proc_address() under
    // GGML_BACKEND_MEMORY_API_V1_PROC. Byte counts are fixed-width so quotes
    // can be persisted in audit records across C/Rust ABI boundaries.
    #define GGML_BACKEND_MEMORY_API_V1_PROC "ggml_backend_memory_get_api_v1"
    #define GGML_BACKEND_MEMORY_ABI_V1 1u

    enum ggml_backend_memory_domain_kind_v1 {
        GGML_BACKEND_MEMORY_DOMAIN_DEVICE_LOCAL = 1,
        GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE = 2,
        GGML_BACKEND_MEMORY_DOMAIN_HOST_PINNED = 3,
        GGML_BACKEND_MEMORY_DOMAIN_UNIFIED = 4,
        GGML_BACKEND_MEMORY_DOMAIN_FILE_BACKED = 5,
    };

    enum ggml_backend_memory_request_kind_v1 {
        GGML_BACKEND_MEMORY_REQUEST_BUFFER = 1,
        GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT = 2,
        GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE = 3,
        GGML_BACKEND_MEMORY_REQUEST_TRANSFER = 4,
    };

    enum ggml_backend_memory_claim_flags_v1 {
        GGML_BACKEND_MEMORY_CLAIM_EXACT = 1u << 0,
        GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER = 1u << 1,
        GGML_BACKEND_MEMORY_CLAIM_DRIVER_ESTIMATE = 1u << 2,
        GGML_BACKEND_MEMORY_CLAIM_REUSABLE_WORKSPACE = 1u << 3,
        GGML_BACKEND_MEMORY_CLAIM_TRANSIENT = 1u << 4,
        GGML_BACKEND_MEMORY_CLAIM_FILE_BACKED = 1u << 5,
        // Physical commitment is reconciled from post-allocation stats. The
        // numeric claim is an admission estimate, never an exact/upper claim.
        GGML_BACKEND_MEMORY_CLAIM_PROVISIONAL = 1u << 6,
    };

    enum ggml_backend_memory_quote_flags_v1 {
        GGML_BACKEND_MEMORY_QUOTE_PROVISIONAL = 1u << 0,
        GGML_BACKEND_MEMORY_QUOTE_HAS_RESIDUAL_UNCERTAINTY = 1u << 1,
        // Engine-visible graph-private payload is fully priced, but opaque
        // command-buffer/driver costs are intentionally outside the claim.
        // Admission must cover those costs with the physical domain's policy
        // headroom; this flag does not make the quote provisional.
        GGML_BACKEND_MEMORY_QUOTE_OPAQUE_DRIVER_COSTS_REQUIRE_DOMAIN_HEADROOM = 1u << 2,
    };

    enum ggml_backend_memory_residual_flags_v1 {
        GGML_BACKEND_MEMORY_RESIDUAL_BACKEND_PRIVATE = 1u << 0,
        GGML_BACKEND_MEMORY_RESIDUAL_DRIVER_ACCOUNTING = 1u << 1,
    };

    enum ggml_backend_memory_health_v1 {
        GGML_BACKEND_MEMORY_HEALTHY = 0,
        GGML_BACKEND_MEMORY_DEGRADED = 1,
        GGML_BACKEND_MEMORY_QUARANTINED = 2,
        GGML_BACKEND_MEMORY_DEVICE_LOST = 3,
    };

    enum ggml_backend_memory_stats_flags_v1 {
        GGML_BACKEND_MEMORY_STATS_BUDGET_UNAVAILABLE = 1u << 0,
    };

    struct ggml_backend_memory_domain_id_v1 {
        // Provider-neutral physical identity token. GPU PCI providers use the
        // canonical BDF encoding below; all zero means unknown and must remain
        // provider/logical-device scoped in the caller.
        uint8_t physical_device_uuid[16];
        uint32_t heap_index;
        uint32_t kind;
    };

    struct ggml_backend_memory_domain_v1 {
        uint32_t struct_size;
        uint32_t flags;
        struct ggml_backend_memory_domain_id_v1 id;
        char name[48];
    };

    struct ggml_backend_memory_request_v1 {
        uint32_t struct_size;
        uint32_t kind;
        uint32_t flags;
        uint32_t usage;
        uint64_t request_id;
        ggml_backend_t backend;
        ggml_backend_t peer_backend;
        ggml_backend_buffer_type_t buft;
        const struct ggml_cgraph * graph;
        const void * host_ptr;
        uint64_t requested_bytes;
        // Logical payload already owned by this reusable allocation. This is
        // zero for a new owner and non-zero for an arena replacement/reuse.
        uint64_t currently_allocated_bytes;
        uint64_t max_tensor_bytes;
    };

    struct ggml_backend_memory_claim_v1 {
        uint32_t struct_size;
        uint32_t flags;
        uint64_t request_id; // zero denotes a batch-wide shared cost
        struct ggml_backend_memory_domain_id_v1 domain;
        uint64_t payload_requested_bytes;
        uint64_t committed_before_bytes;
        uint64_t committed_after_upper_bytes;
        uint64_t commit_peak_extra_upper_bytes;
        uint64_t resident_after_upper_bytes;
        uint64_t retained_after_use_upper_bytes;
        uint64_t releasable_after_use_upper_bytes;
    };

    struct ggml_backend_memory_quote_v1 {
        uint32_t struct_size;
        uint32_t flags;
        uint32_t residual_flags;
        uint32_t residual_request_count;
        uint64_t provisional_requested_upper_bytes;
        uint64_t stats_generation;
        uint64_t quote_token;
        uint64_t request_fingerprint;
    };

    // Produces the canonical ABI-v1 transaction fingerprint. The hash binds
    // the ABI version, request count and order, and every semantic request
    // field. Providers must use this helper for both quote and reserve checks.
    GGML_API uint64_t ggml_backend_memory_request_fingerprint_v1(
        const struct ggml_backend_memory_request_v1 * requests,
        uint32_t request_count);

    // Encodes canonical PCI BDF text ("dddd:bb:dd.f") into the 16-byte
    // physical identity token shared by CUDA/HIP and Vulkan:
    //   [0..4)  = {'P', 'C', 'I', 1}
    //   [4..6)  = PCI domain, big endian
    //   [6]     = bus
    //   [7]     = device
    //   [8]     = function
    //   [9..16) = zero
    // The output is zeroed and false is returned when the BDF is unavailable
    // or non-canonical. Provider ordinals must never enter this token.
    GGML_API bool ggml_backend_memory_encode_pci_bdf_v1(
        const char * pci_bus_id,
        uint8_t physical_device_uuid[16]);

    struct ggml_backend_memory_stats_v1 {
        uint32_t struct_size;
        uint32_t flags;
        struct ggml_backend_memory_domain_id_v1 domain;
        // Provider epoch for state that can invalidate a quote's physical
        // layout/cost derivation. This is deliberately NOT a hash of live
        // free/used bytes: capacity is carried by the fresh budget fields,
        // and unrelated allocations must not make a request-shape quote stale.
        uint64_t generation;
        uint64_t timestamp_monotonic_ns;
        uint64_t total_bytes;
        uint64_t budget_bytes;
        uint64_t device_used_bytes;
        uint64_t device_free_bytes;
        uint64_t backend_owned_live_bytes;
        uint64_t backend_owned_cached_bytes;
        uint64_t backend_owned_workspace_bytes;
        uint64_t backend_owned_high_water_bytes;
        uint64_t allocation_count;
        uint64_t allocation_failure_count;
        uint32_t health;
        int32_t last_ggml_status;
        int64_t last_native_error;
        uint64_t quarantine_generation;
    };

    struct ggml_backend_memory_quarantine_v1 {
        uint32_t struct_size;
        uint32_t flags;
        uint32_t reason;
        int32_t ggml_status;
        int64_t native_error;
        char message[96];
    };

    struct ggml_backend_memory_api_v1 {
        uint32_t struct_size;
        uint32_t abi_version;
        uint64_t capabilities;
        enum ggml_status (*get_domains)(ggml_backend_dev_t dev, struct ggml_backend_memory_domain_v1 * domains, uint32_t * inout_count);
        enum ggml_status (*quote)(const struct ggml_backend_memory_request_v1 * requests, uint32_t request_count, struct ggml_backend_memory_quote_v1 * quote, struct ggml_backend_memory_claim_v1 * claims, uint32_t * inout_claim_count);
        // Failure-atomic transactional hook. A non-success return must leave
        // native allocation state unchanged and must not retain replacement
        // buffers, caches, workspaces, or command resources. A size query
        // (actual == NULL) is side-effect free. Providers that cannot prove
        // this contract in ABI v1 must keep this hook validation-only/no-op;
        // the caller then holds its provisional domain gate through first use.
        enum ggml_status (*reserve_private)(const struct ggml_backend_memory_request_v1 * requests, uint32_t request_count, const struct ggml_backend_memory_quote_v1 * quote, struct ggml_backend_memory_claim_v1 * actual, uint32_t * inout_actual_count);
        enum ggml_status (*get_stats)(ggml_backend_dev_t dev, ggml_backend_t backend, struct ggml_backend_memory_stats_v1 * stats, uint32_t * inout_count);
        enum ggml_status (*trim)(ggml_backend_t backend, uint64_t flags);
        enum ggml_status (*quarantine)(ggml_backend_t backend, const struct ggml_backend_memory_quarantine_v1 * request);
    };

    typedef const struct ggml_backend_memory_api_v1 * (*ggml_backend_memory_get_api_v1_t)(void);

    // Exception-safe host trampolines for the optional provider table. Native
    // callers must use these functions instead of invoking plugin-owned C++
    // callbacks directly across an FFI seam.
    GGML_API const struct ggml_backend_memory_api_v1 * ggml_backend_memory_api_for_backend_v1(ggml_backend_t backend);
    GGML_API enum ggml_status ggml_backend_memory_api_get_domains_v1(
        const struct ggml_backend_memory_api_v1 * api, ggml_backend_dev_t dev,
        struct ggml_backend_memory_domain_v1 * domains, uint32_t * inout_count);
    GGML_API enum ggml_status ggml_backend_memory_api_quote_v1(
        const struct ggml_backend_memory_api_v1 * api,
        const struct ggml_backend_memory_request_v1 * requests, uint32_t request_count,
        struct ggml_backend_memory_quote_v1 * quote,
        struct ggml_backend_memory_claim_v1 * claims, uint32_t * inout_claim_count);
    GGML_API enum ggml_status ggml_backend_memory_api_reserve_private_v1(
        const struct ggml_backend_memory_api_v1 * api,
        const struct ggml_backend_memory_request_v1 * requests, uint32_t request_count,
        const struct ggml_backend_memory_quote_v1 * quote,
        struct ggml_backend_memory_claim_v1 * actual, uint32_t * inout_actual_count);
    GGML_API enum ggml_status ggml_backend_memory_api_get_stats_v1(
        const struct ggml_backend_memory_api_v1 * api, ggml_backend_dev_t dev,
        ggml_backend_t backend, struct ggml_backend_memory_stats_v1 * stats,
        uint32_t * inout_count);
    GGML_API enum ggml_status ggml_backend_memory_api_trim_v1(
        const struct ggml_backend_memory_api_v1 * api, ggml_backend_t backend, uint64_t flags);
    GGML_API enum ggml_status ggml_backend_memory_api_quarantine_v1(
        const struct ggml_backend_memory_api_v1 * api, ggml_backend_t backend,
        const struct ggml_backend_memory_quarantine_v1 * request);

    // Optional, versioned observation of backend-native captured graph
    // executables. This is diagnostic evidence only: it cannot enable graph
    // capture or alter backend policy. A caller observes one concrete backend
    // and cgraph after successful compute; providers must return the actual
    // executable generation minted where instantiate/update succeeds rather
    // than infer it from the requested planner mode.
    #define GGML_BACKEND_GRAPH_LIFECYCLE_API_V1_PROC "ggml_backend_graph_lifecycle_get_api_v1"
    #define GGML_BACKEND_GRAPH_LIFECYCLE_ABI_V1 1u

    enum ggml_backend_graph_lifecycle_flags_v1 {
        GGML_BACKEND_GRAPH_LIFECYCLE_CAPTURE_SUPPORTED_V1  = 1u << 0,
        GGML_BACKEND_GRAPH_LIFECYCLE_CAPTURE_ENABLED_V1    = 1u << 1,
        GGML_BACKEND_GRAPH_LIFECYCLE_EXECUTABLE_PRESENT_V1 = 1u << 2,
    };

    enum ggml_backend_graph_executable_change_v1 {
        GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_NONE_V1        = 0,
        GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_INSTANTIATED_V1 = 1,
        GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_UPDATED_V1      = 2,
        GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_REPLACED_V1     = 3,
    };

    struct ggml_backend_graph_lifecycle_observation_v1 {
        uint32_t struct_size;
        uint32_t abi_version;
        uint32_t flags;
        uint32_t last_executable_change;
        // Monotonic within one backend context, including cache eviction and
        // recreation. Zero means no captured executable has been created for
        // this graph in the current context.
        uint64_t executable_generation;
    };

    struct ggml_backend_graph_lifecycle_api_v1 {
        uint32_t struct_size;
        uint32_t abi_version;
        uint64_t capabilities;
        enum ggml_status (*observe)(
            ggml_backend_t backend,
            const struct ggml_cgraph * graph,
            struct ggml_backend_graph_lifecycle_observation_v1 * observation);
    };

    typedef const struct ggml_backend_graph_lifecycle_api_v1 *
        (*ggml_backend_graph_lifecycle_get_api_v1_t)(void);

    GGML_API const char *                   ggml_backend_buffer_name          (ggml_backend_buffer_t buffer);
    GGML_API void                           ggml_backend_buffer_free          (ggml_backend_buffer_t buffer);
    GGML_API enum ggml_status               ggml_backend_buffer_free_status   (ggml_backend_buffer_t buffer);
    GGML_API void *                         ggml_backend_buffer_get_base      (ggml_backend_buffer_t buffer);
    GGML_API size_t                         ggml_backend_buffer_get_size      (ggml_backend_buffer_t buffer);
    GGML_API enum ggml_status               ggml_backend_buffer_init_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    GGML_API size_t                         ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer);
    GGML_API size_t                         ggml_backend_buffer_get_max_size  (ggml_backend_buffer_t buffer);
    GGML_API size_t                         ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor);
    GGML_API enum ggml_status               ggml_backend_buffer_clear         (ggml_backend_buffer_t buffer, uint8_t value);
    GGML_API bool                           ggml_backend_buffer_is_host       (ggml_backend_buffer_t buffer);
    GGML_API void                           ggml_backend_buffer_set_usage     (ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
    GGML_API enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage     (ggml_backend_buffer_t buffer);
    GGML_API ggml_backend_buffer_type_t     ggml_backend_buffer_get_type      (ggml_backend_buffer_t buffer);
    GGML_API void                           ggml_backend_buffer_reset         (ggml_backend_buffer_t buffer);
    GGML_API enum ggml_status               ggml_backend_buffer_reset_status  (ggml_backend_buffer_t buffer);

    // tensor copy between different backends
    GGML_API enum ggml_status ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst);

    //
    // Backend (stream)
    //

    GGML_API ggml_guid_t  ggml_backend_guid(ggml_backend_t backend);
    GGML_API const char * ggml_backend_name(ggml_backend_t backend);
    GGML_API void         ggml_backend_free(ggml_backend_t backend);
    GGML_API enum ggml_status ggml_backend_free_status(ggml_backend_t backend);

    GGML_API ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);
    GGML_API ggml_backend_buffer_t      ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);
    GGML_API size_t                     ggml_backend_get_alignment(ggml_backend_t backend);
    GGML_API size_t                     ggml_backend_get_max_size(ggml_backend_t backend);

    GGML_API enum ggml_status ggml_backend_tensor_set_async   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API enum ggml_status ggml_backend_tensor_get_async   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    GGML_API enum ggml_status ggml_backend_tensor_set_2d_async(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API enum ggml_status ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

    // "offset" refers to the offset in tensor->data for setting/getting data.
    // A non-success result guarantees no fallback operation touched the destination.
    GGML_API enum ggml_status ggml_backend_tensor_set   (      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API enum ggml_status ggml_backend_tensor_get   (const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    GGML_API enum ggml_status ggml_backend_tensor_set_2d(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API enum ggml_status ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API enum ggml_status ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);

    // Completion is terminal: a successful return guarantees this backend has no
    // work from the completed operation still pending.
    GGML_API enum ggml_status ggml_backend_synchronize(ggml_backend_t backend);

    GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API void                      ggml_backend_graph_plan_free  (ggml_backend_t backend, ggml_backend_graph_plan_t plan);

    GGML_API enum ggml_status ggml_backend_graph_plan_compute (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    GGML_API enum ggml_status ggml_backend_graph_compute      (ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);

    // Cooperative graph cancellation reports two orthogonal properties for
    // the execution path actually used by this compute. The mechanism says
    // where polling is implemented; the observation granularity says the
    // coarsest boundary at which a newly raised request can be observed.
    enum ggml_backend_graph_cancel_mechanism {
        GGML_BACKEND_GRAPH_CANCEL_DISABLED  = 0,
        GGML_BACKEND_GRAPH_CANCEL_NATIVE    = 1,
        GGML_BACKEND_GRAPH_CANCEL_SEGMENTED = 2,
    };

    enum ggml_backend_graph_cancel_observation_granularity {
        GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_NONE                  = 0,
        GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT = 1,
        GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_GRAPH_COMPLETION      = 2,
    };

    struct ggml_backend_graph_cancel_capability {
        enum ggml_backend_graph_cancel_mechanism mechanism;
        enum ggml_backend_graph_cancel_observation_granularity observation_granularity;
    };

    // Non-native backends synchronize after at most this many graph nodes while
    // cancellation is armed. 32 is half Metal's historical 64-node main
    // submission: smaller segments would lower worst-case cancellation latency
    // but add more submission/synchronization overhead, while larger segments
    // improve throughput at the cost of slower cancellation observation. The
    // callback-free async path is unchanged.
#define GGML_BACKEND_GRAPH_CANCEL_SEGMENT_NODES 32

    // Synchronous, compute-scoped cancellation. `abort_callback_data` must stay
    // alive only for this call; no job pointer is retained by the shared layer.
    // `cancel_capability` reports the execution path actually used. A request
    // observed before graph submission reports DISABLED/NONE because no backend
    // cancellation mechanism ran. GRAPH_COMPLETION means an already launched
    // monolithic graph cannot be interrupted; cancellation is observed after
    // that graph completes and prevents later upper-layer work from being
    // submitted. ABORTED means the request was observed, accepted work has been
    // completed or drained, and the backend is safe to reuse. Concrete device,
    // execution, or poisoned-backend failures take precedence over ABORTED.
    GGML_API enum ggml_status ggml_backend_graph_compute_with_abort(
            ggml_backend_t backend, struct ggml_cgraph * cgraph,
            ggml_abort_callback abort_callback, void * abort_callback_data,
            struct ggml_backend_graph_cancel_capability * cancel_capability);

    // NOTE: will be removed, use device version instead
    GGML_API bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);
    GGML_API bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
    GGML_API bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op);

    // asynchronous copy
    // the copy is performed after all the currently queued operations in backend_src
    // backend_dst will wait for the copy to complete before performing other operations
    // automatic fallback to sync copy if async is not supported
    GGML_API enum ggml_status ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

    GGML_API ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend);

    //
    // Events
    //

    GGML_API ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device);
    GGML_API void                 ggml_backend_event_free(ggml_backend_event_t event);
    GGML_API enum ggml_status     ggml_backend_event_free_status(ggml_backend_event_t event);
    GGML_API enum ggml_status     ggml_backend_event_record_status(ggml_backend_event_t event, ggml_backend_t backend);
    GGML_API enum ggml_status     ggml_backend_event_synchronize(ggml_backend_event_t event);
    GGML_API enum ggml_status     ggml_backend_event_wait_status(ggml_backend_t backend, ggml_backend_event_t event);

    //
    // Backend device
    //

    enum ggml_backend_dev_type {
        // Provider callback failed before a trustworthy class was available.
        GGML_BACKEND_DEVICE_TYPE_UNKNOWN = -1,
        // CPU device using system memory
        GGML_BACKEND_DEVICE_TYPE_CPU,
        // GPU device using dedicated memory
        GGML_BACKEND_DEVICE_TYPE_GPU,
        // integrated GPU device using host memory
        GGML_BACKEND_DEVICE_TYPE_IGPU,
        // accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
        GGML_BACKEND_DEVICE_TYPE_ACCEL,
        // "meta" device wrapping multiple other devices for tensor parallelism
        GGML_BACKEND_DEVICE_TYPE_META,
    };

    // functionality supported by the device
    struct ggml_backend_dev_caps {
        // asynchronous operations
        bool async;
        // pinned host buffer
        bool host_buffer;
        // creating buffers from host ptr
        bool buffer_from_host_ptr;
        // event synchronization
        bool events;
    };

    // all the device properties
    struct ggml_backend_dev_props {
        // device name
        const char * name;
        // device description
        const char * description;
        // device free memory in bytes
        size_t memory_free;
        // device total memory in bytes
        size_t memory_total;
        // device type
        enum ggml_backend_dev_type type;
        // device id
        //   for PCI devices, this should be the lower-case PCI bus id formatted as "domain:bus:device.function" (e.g. "0000:c1:00.0")
        //   if the id is unknown, this should be NULL
        const char * device_id;
        // device capabilities
        struct ggml_backend_dev_caps caps;
    };

    GGML_API const char *                  ggml_backend_dev_name(ggml_backend_dev_t device);
    GGML_API const char *                  ggml_backend_dev_description(ggml_backend_dev_t device);
    GGML_API void                          ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total);
    GGML_API enum ggml_backend_dev_type    ggml_backend_dev_type(ggml_backend_dev_t device);
    GGML_API void                          ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props);
    GGML_API ggml_backend_reg_t            ggml_backend_dev_backend_reg(ggml_backend_dev_t device);
    GGML_API ggml_backend_t                ggml_backend_dev_init(ggml_backend_dev_t device, const char * params);
    GGML_API ggml_backend_buffer_type_t    ggml_backend_dev_buffer_type(ggml_backend_dev_t device);
    GGML_API ggml_backend_buffer_type_t    ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device);
    GGML_API ggml_backend_buffer_t         ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size);

    GGML_API bool                          ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
    GGML_API bool                          ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft);
    GGML_API bool                          ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op);

    //
    // Backend (reg)
    //

    GGML_API const char *       ggml_backend_reg_name(ggml_backend_reg_t reg);
    GGML_API size_t             ggml_backend_reg_dev_count(ggml_backend_reg_t reg);
    GGML_API ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index);
    GGML_API void *             ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name);

    // Shared no-throw adapters for optional registry procedures. Callers must
    // use these instead of invoking a get_proc_address result across a C ABI.
    // An unavailable optional procedure preserves the historical no-op/unknown
    // semantics; a provider exception is converted to a typed status or zero.
    GGML_API enum ggml_status ggml_backend_set_n_threads_if_supported(
            ggml_backend_t backend, int n_threads);
    GGML_API uint32_t ggml_backend_dev_pci_vendor_id(ggml_backend_dev_t device);

    // Common functions that may be obtained using ggml_backend_reg_get_proc_address

    // Context management and operations for faster communication between backends, used for tensor parallelism (meta backend)
    typedef void * (*ggml_backend_comm_init_t)(ggml_backend_t * backends, size_t n_backends);
    typedef void   (*ggml_backend_comm_free_t)(void * comm_ctx);
    typedef bool   (*ggml_backend_comm_allreduce_tensor_t)(void * comm_ctx, struct ggml_tensor ** tensors);

    // Split buffer type for tensor parallelism (old)
    typedef ggml_backend_buffer_type_t   (*ggml_backend_split_buffer_type_t)(int main_device, const float * tensor_split);
    // Set the number of threads for the backend
    typedef void                         (*ggml_backend_set_n_threads_t)(ggml_backend_t backend, int n_threads);
    // Get additional buffer types provided by the device (returns a NULL-terminated array)
    typedef ggml_backend_buffer_type_t * (*ggml_backend_dev_get_extra_bufts_t)(ggml_backend_dev_t device);
    // Set the compute-scoped abort callback for a native-cancellation backend.
    // The shared layer installs it before graph submission, keeps it alive
    // through the terminal synchronization, and clears it before returning.
    // Implementations must poll at safe submission/completion boundaries,
    // report GGML_STATUS_ABORTED after an observed request (without hiding a
    // device/execute failure), and retain neither pointer after the clear call.
    typedef void                         (*ggml_backend_set_abort_callback_t)(ggml_backend_t backend, ggml_abort_callback abort_callback, void * abort_callback_data, struct ggml_backend_graph_cancel_capability * cancel_capability);
    typedef enum ggml_status             (*ggml_backend_set_abort_callback_status_t)(ggml_backend_t backend, ggml_abort_callback abort_callback, void * abort_callback_data, struct ggml_backend_graph_cancel_capability * cancel_capability);
    // Get a list of feature flags supported by the backend (returns a NULL-terminated array)
    struct ggml_backend_feature {
        const char * name;
        const char * value;
    };
    typedef struct ggml_backend_feature * (*ggml_backend_get_features_t)(ggml_backend_reg_t reg);

    // Optional, stable hardware-vendor fact for one device. Backends that can
    // prove a PCI vendor id expose this through get_proc_address using the name
    // below; absence/zero means unknown and callers must not infer it from a
    // human-readable device name.
    #define GGML_BACKEND_DEVICE_PCI_VENDOR_ID_PROC "ggml_backend_device_pci_vendor_id"
    typedef uint32_t (*ggml_backend_device_pci_vendor_id_t)(ggml_backend_dev_t device);

    //
    // Backend registry
    //

    GGML_API void ggml_backend_register(ggml_backend_reg_t reg);

    GGML_API void ggml_backend_device_register(ggml_backend_dev_t device);

    // Backend (reg) enumeration
    GGML_API size_t             ggml_backend_reg_count(void);
    GGML_API ggml_backend_reg_t ggml_backend_reg_get(size_t index);
    GGML_API ggml_backend_reg_t ggml_backend_reg_by_name(const char * name);

    // Device enumeration
    GGML_API size_t             ggml_backend_dev_count(void);
    GGML_API ggml_backend_dev_t ggml_backend_dev_get(size_t index);
    GGML_API ggml_backend_dev_t ggml_backend_dev_by_name(const char * name);
    GGML_API ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type);

    // Direct backend (stream) initialization
    // = ggml_backend_dev_init(ggml_backend_dev_by_name(name), params)
    GGML_API ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params);
    // = ggml_backend_dev_init(ggml_backend_dev_by_type(type), params)
    GGML_API ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params);
    // = ggml_backend_dev_init(ggml_backend_dev_by_type(GPU) OR ggml_backend_dev_by_type(CPU), NULL)
    GGML_API ggml_backend_t ggml_backend_init_best(void);

    // Load a backend from a dynamic library and register it
    GGML_API ggml_backend_reg_t ggml_backend_load(const char * path);
    // Load an exact backend path encoded as UTF-8. This is distinct from the
    // legacy native-narrow path on Windows so user-profile directories outside
    // the active ANSI code page remain loadable without lossy conversion.
    GGML_API ggml_backend_reg_t ggml_backend_load_utf8(const char * path_utf8);
    // Load one exact UTF-8 path only when its OpenASR ABI-v1 export matches.
    // The comparison happens before ggml_backend_score/ggml_backend_init, so an
    // incompatible plugin cannot register devices or start backend threads.
    GGML_API ggml_backend_reg_t ggml_backend_load_verified_utf8(
            const char * path_utf8,
            const char * expected_openasr_abi_v1,
            const char * expected_provider_v1);
    // As above, additionally requiring the module's side-effect-free hardware
    // probe to attest an exact device target and a driver at or above the
    // signed catalog floor before ggml_backend_score/ggml_backend_init runs.
    GGML_API ggml_backend_reg_t ggml_backend_load_verified_v2_utf8(
            const char * path_utf8,
            const char * expected_openasr_abi_v1,
            const char * expected_provider_v1,
            const char * expected_device_target,
            const char * minimum_driver_version);
    // Version 3 additionally accepts exact, already verified dependency
    // directories. On Windows these directories participate only in this
    // LoadLibraryEx transaction; PATH and the current directory are never
    // searched.
    GGML_API ggml_backend_reg_t ggml_backend_load_verified_v3_utf8(
            const char * path_utf8,
            const char * const * dependency_dirs_utf8,
            size_t dependency_dir_count,
            const char * expected_openasr_abi_v1,
            const char * expected_provider_v1,
            const char * expected_device_target,
            const char * minimum_driver_version);
    // Validate the same contract and return the normalized live driver
    // version without registering the backend. `driver_out` is always NUL
    // terminated when its capacity is non-zero.
    GGML_API bool ggml_backend_probe_verified_v2_utf8(
            const char * path_utf8,
            const char * expected_openasr_abi_v1,
            const char * expected_provider_v1,
            const char * expected_device_target,
            const char * minimum_driver_version,
            char * driver_out,
            size_t driver_out_capacity);
    GGML_API bool ggml_backend_probe_verified_v3_utf8(
            const char * path_utf8,
            const char * const * dependency_dirs_utf8,
            size_t dependency_dir_count,
            const char * expected_openasr_abi_v1,
            const char * expected_provider_v1,
            const char * expected_device_target,
            const char * minimum_driver_version,
            char * driver_out,
            size_t driver_out_capacity);
    // Load only the CPU and Vulkan modules from one host-owned absolute
    // directory. Unlike load_all, this never consults the executable's current
    // directory, GGML_BACKEND_PATH, or any downloaded plugin store.
    GGML_API void               ggml_backend_load_bundled_from_path(const char * dir_path_utf8);
    GGML_API void               ggml_backend_load_bundled_verified_from_path(
            const char * dir_path_utf8,
            const char * expected_openasr_abi_v1);
    // Select the highest-scoring backend only from this already verified,
    // host-owned exact path list. No directory enumeration or environment
    // search participates.
    GGML_API ggml_backend_reg_t ggml_backend_load_best_verified_utf8(
            const char * const * paths_utf8,
            size_t path_count,
            const char * expected_openasr_abi_v1,
            const char * expected_provider_v1);
    // Unload a backend if loaded dynamically and unregister it
    GGML_API void               ggml_backend_unload(ggml_backend_reg_t reg);
    // Load all known backends from dynamic libraries
    GGML_API void               ggml_backend_load_all(void);
    GGML_API void               ggml_backend_load_all_from_path(const char * dir_path);

    //
    // Backend scheduler
    //

    // The backend scheduler allows for multiple backend devices to be used together
    // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
    // The backends are selected based on:
    // - the backend that supports the operation
    // - the location of the pre-allocated tensors (e.g. the weights)
    /*
      Example usage:

        // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
        // preferably to run on the same backend as the buffer
        ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, GGML_DEFAULT_GRAPH_SIZE, false, true);

        // initialize buffers from a max size graph (optional)
        reserve_graph = build_graph(sched, max_batch_size);

        // manually assign nodes to a backend (optional, should not be needed in most cases)
        struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
        ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

        ggml_backend_sched_reserve(sched, reserve_graph);

        // compute
        graph = build_graph(sched); // the graph and its tensors are single-use in terms of allocation, multi-use in terms of computation
        for (int i = 0; i < 10; ++i) {
            ggml_backend_sched_graph_compute(sched, graph); // on the first iteration the graph is allocated automatically
        }

        // if there are graph inputs:
        graph = build_graph(sched); // get a new graph that is not allocated (the metadata for the old graph is freed once ggml_free is called)
        ggml_backend_sched_reset(sched); // clear the allocation of the previous graph
        ggml_backend_sched_alloc_graph(sched, graph); // explicitly allocate the new graph but do not execute it
        ggml_backend_tensor_set(input_tensor, ...); // copy data to the newly allocated graph tensors
        ggml_backend_sched_graph_compute(sched, graph); // execute the graph

        // as an alternative to the above it is also possible to assign the inputs to a dedicated context and
        // allocate them statically via ggml_backend_alloc_ctx_tensors
    }
    */

    typedef struct ggml_backend_sched * ggml_backend_sched_t;
    typedef struct ggml_backend_sched_memory_plan * ggml_backend_sched_memory_plan_t;

    // Evaluation callback for each node in the graph (set with ggml_backend_sched_set_eval_callback)
    // when ask == true, the scheduler wants to know if the user wants to observe this node
    // this allows the scheduler to batch nodes together in order to evaluate them in a single call
    //
    // when ask == false, the scheduler is passing the node tensor to the user for observation
    // if the user returns false, the scheduler will cancel the graph compute
    //
    typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);

    // Initialize a backend scheduler, backends with low index are given priority over backends with high index
    GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
    GGML_API void                 ggml_backend_sched_free(ggml_backend_sched_t sched);
    GGML_API enum ggml_status     ggml_backend_sched_free_status(ggml_backend_sched_t sched);

    // Initialize backend buffers from a measure graph
    GGML_API void                 ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes);
    GGML_API bool                 ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph); // returns success

    // Freeze the exact scheduler split/gallocr measurement used for admission.
    // A plan owns the scheduler until commit/free; the graph and its tensors
    // must remain alive and immutable for that lifetime.
    GGML_API enum ggml_status ggml_backend_sched_memory_plan_create_v1(
            ggml_backend_sched_t sched, struct ggml_cgraph * graph,
            ggml_backend_sched_memory_plan_t * out_plan);
    GGML_API uint32_t ggml_backend_sched_memory_plan_get_item_count_v1(
            ggml_backend_sched_memory_plan_t plan);
    GGML_API bool ggml_backend_sched_memory_plan_get_item_v1(
            ggml_backend_sched_memory_plan_t plan, uint32_t index,
            struct ggml_backend_memory_request_v1 * out_item);
    GGML_API enum ggml_status ggml_backend_sched_memory_plan_commit_v1(
            ggml_backend_sched_memory_plan_t plan);
    // Reports whether the commit may have changed scheduler-owned native
    // allocation state. On failure, callers may safely refund admission only
    // when the bit remains clear. Unknown future bits are conservative.
    enum ggml_backend_sched_memory_plan_commit_flag {
        GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_MAY_HAVE_MUTATED = 1u << 0,
        // A failed commit rebuilt the scheduler allocator and every native
        // buffer release callback completed successfully. Callers may refund
        // the failed candidate only when live device-health evidence is also
        // available and non-terminal.
        GGML_BACKEND_SCHED_MEMORY_PLAN_COMMIT_RELEASE_PROVEN = 1u << 1,
    };
    GGML_API enum ggml_status ggml_backend_sched_memory_plan_commit_v2(
            ggml_backend_sched_memory_plan_t plan, uint32_t * out_flags);
    GGML_API void ggml_backend_sched_memory_plan_free_v1(
            ggml_backend_sched_memory_plan_t plan);

    GGML_API int                  ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched);
    GGML_API ggml_backend_t       ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i);

    // Get the number of splits of the last graph
    GGML_API int                  ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
    GGML_API int                  ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);

    GGML_API ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend);
    GGML_API size_t                     ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);

    GGML_API void                 ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
    GGML_API ggml_backend_t       ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);

    // Split graph without allocating it
    GGML_API void                 ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API enum ggml_status     ggml_backend_sched_split_graph_v2(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

    // Allocate and compute graph on the backend scheduler
    GGML_API bool                 ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph); // returns success
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API enum ggml_status     ggml_backend_sched_synchronize(ggml_backend_sched_t sched);
    // Synchronous cancellation also polls around scheduler-controlled input
    // waits/copies. One backend call already in progress remains indivisible;
    // every backend is synchronized before an observed abort returns.
    GGML_API enum ggml_status ggml_backend_sched_graph_compute_with_abort(
            ggml_backend_sched_t sched, struct ggml_cgraph * graph,
            ggml_abort_callback abort_callback, void * abort_callback_data,
            struct ggml_backend_graph_cancel_capability * cancel_capability);

    // Reset all assignments and allocators - must be called before changing the node backends or allocating a new graph.
    // Scheduler-owned tensor bindings are detached while their buffers are
    // still alive, and scheduler-inserted source copies are restored. A
    // retained graph can therefore be allocated again after another graph;
    // external/static tensor bindings are preserved.
    GGML_API void                 ggml_backend_sched_reset(ggml_backend_sched_t sched);

    // Set a callback to be called for each resulting node during graph compute
    GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);

    //
    // Meta backend
    //

#define GGML_BACKEND_META_MAX_DEVICES 16

    enum ggml_backend_meta_split_axis {
        // tensor split by tensor dimensions:
        GGML_BACKEND_SPLIT_AXIS_0 = 0,
        GGML_BACKEND_SPLIT_AXIS_1 = 1,
        GGML_BACKEND_SPLIT_AXIS_2 = 2,
        GGML_BACKEND_SPLIT_AXIS_3 = 3,

        GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10, // all values on all backends
        GGML_BACKEND_SPLIT_AXIS_PARTIAL  = 11, // each backend has a partial sum

        // for internal bookkeeping only:
        GGML_BACKEND_SPLIT_AXIS_NONE    = 98,
        GGML_BACKEND_SPLIT_AXIS_UNKNOWN = 99,
    };
    GGML_API const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis);

    struct ggml_backend_meta_split_state {
        enum ggml_backend_meta_split_axis axis;

        // for tensors with axis >= 0 && axis < GGML_MAX_DIMS:
        //   - each device has a slice of the tensor along the split axis
        //   - most tensors have n_segments == 1 and a contiguous slice of the tensor data
        //   - some tensors have an inhomogenenous data layout along the split axis,
        //     those tensors are divided into segments which are each individually split across devices
        //   - ne has one entry per segment and device and that segment repeats nr times,
        //     in total when accounting for repetitions the segments add up to ggml_tensor::ne for that axis,
        //     the outer/inner loops are over segments/devices like [seg0_dev0_r0, seg0_dev1_r0, seg0_dev0_r1, seg0_dev1_r1, seg1_dev0_r0, seg1_dev1_r0],
        //   - for example, a transformer may have a fused QKV matrix rather than 3 matrices, those would be 3 separate segments
        //     that each need to be split individually across devices so that each device gets a slice of Q, K, and V,
        //     the Q matrix can be larger than the K and V matrices so this can either be expressed as 3 segments or as 2 segments
        //     where the segment for K/V repeats twice
        int64_t  ne[16*GGML_BACKEND_META_MAX_DEVICES];
        uint32_t nr[16];
        uint32_t n_segments;
    };

    // function to assign split states for statically allocated tensors, compute tensor split states will be assigned to be compatible:
    typedef struct ggml_backend_meta_split_state(*ggml_backend_meta_get_split_state_t)(const struct ggml_tensor * tensor, void * userdata);

    // create a new meta device from "simple" devices, meta buffer type/buffer/backend is then derived from this:
    // TODO: this looks a bit strange - a backend API creates a device. I think we should try
    //       express this as a backend registry functionality instead
    GGML_API ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud);

    //
    // Utils
    //

    struct ggml_backend_graph_copy {
        ggml_backend_buffer_t buffer;
        struct ggml_context * ctx_allocated;
        struct ggml_context * ctx_unallocated;
        struct ggml_cgraph * graph;
    };

    // Copy a graph to a different backend
    GGML_API struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph);
    GGML_API void                           ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy);

    typedef bool (*ggml_backend_eval_callback)(int node_index, struct ggml_tensor * t1, struct ggml_tensor * t2, void * user_data);

    // Compare the output of two backends
    GGML_API bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes);

    // Tensor initialization
    GGML_API enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr);
    GGML_API enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor);

    // CPU buffer types are always available
    GGML_API ggml_backend_buffer_t      ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);
    GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);

#ifdef  __cplusplus
}
#endif
