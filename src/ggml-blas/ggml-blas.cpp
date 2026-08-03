#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-blas.h"
#include "ggml-backend-impl.h"

#include <future>
#include <vector>
#include <cstring>
#include <new>

#if defined(_WIN32)
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#else
#   include <unistd.h>
#endif

#if defined(__APPLE__)
#   include <mach/mach.h>
#   include <sys/sysctl.h>
#   include <sys/types.h>
#elif defined(__linux__)
#   include <sys/sysinfo.h>
#endif

#if defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#elif defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#elif defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#elif defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#else
#   include <cblas.h>
#endif

struct ggml_backend_blas_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
    struct ggml_backend_abort_context abort = {};
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
    uint64_t memory_generation = 1;
    uint64_t memory_high_water = 0;
    uint64_t allocation_failures = 0;
    uint64_t quarantine_generation = 0;
    uint32_t memory_health = GGML_BACKEND_MEMORY_HEALTHY;
    int64_t last_native_error = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<void>> tasks;
#endif
};

static bool ggml_backend_blas_mul_mat_workspace(
        const struct ggml_tensor * dst, uint64_t * workspace) {
    if (dst == nullptr || workspace == nullptr || dst->op != GGML_OP_MUL_MAT ||
            dst->src[0] == nullptr) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    if (src0->type == GGML_TYPE_F32) {
        *workspace = 0;
        return true;
    }
    uint64_t bytes = sizeof(float);
    for (int dim = 0; dim < 4; ++dim) {
        if (src0->ne[dim] < 0 || (uint64_t) src0->ne[dim] > UINT64_MAX / bytes) {
            return false;
        }
        bytes *= (uint64_t) src0->ne[dim];
    }
    *workspace = bytes;
    return true;
}

static bool ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    const int64_t ne_plane = ne01*ne00;
    uint64_t desired_wsize_u64 = 0;
    if (!ggml_backend_blas_mul_mat_workspace(dst, &desired_wsize_u64) ||
            desired_wsize_u64 > SIZE_MAX) {
        ctx->allocation_failures++;
        return false;
    }
    const size_t desired_wsize = (size_t) desired_wsize_u64;

    if (ctx->work_size < desired_wsize) {
        std::unique_ptr<char[]> replacement(new (std::nothrow) char[desired_wsize]);
        if (replacement == nullptr) {
            ctx->allocation_failures++;
            return false;
        }
        ctx->work_data = std::move(replacement);
        ctx->work_size = desired_wsize;
        ctx->memory_high_water = std::max(ctx->memory_high_water, (uint64_t) ctx->work_size);
        ctx->memory_generation++;
    }
    void * wdata = ctx->work_data.get();

    // convert src0 to float
    if (type != GGML_TYPE_F32) {
        const auto * type_traits = ggml_get_type_traits(type);
        ggml_to_float_t const to_float = type_traits->to_float;

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const void  *       x      = (char *)  src0->data + i02*nb02          + i03*nb03;
                      float * const wplane = (float *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                const int min_cols_per_thread = 4096;
                const int min_rows_per_thread = std::max((int)(min_cols_per_thread/ne00), 1);
                const int n_threads = std::max(std::min(ctx->n_threads, (int)(ne01/min_rows_per_thread)), 1);

#ifdef GGML_USE_OPENMP
                #pragma omp parallel for num_threads(n_threads)
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#else
                for (int i = 1; i < n_threads; i++) {
                    const int64_t start =       i*ne01/n_threads;
                    const int64_t end   = (i + 1)*ne01/n_threads;
                    if (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            for (int64_t i01 = start; i01 < end; i01++) {
                                to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    const int64_t start = 0;
                    const int64_t end   = ne01/n_threads;
                    for (int64_t i01 = start; i01 < end; i01++) {
                        to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef GGML_USE_OPENMP
        // wait for all tasks to finish
        for (auto & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#if defined(GGML_BLAS_USE_OPENBLAS)
    openblas_set_num_threads(ctx->n_threads);
#elif defined(GGML_BLAS_USE_BLIS)
    bli_thread_set_num_threads(ctx->n_threads);
#elif defined(GGML_BLAS_USE_NVPL)
    nvpl_blas_set_num_threads(ctx->n_threads);
#elif defined(GGML_BLAS_USE_MKL)
    mkl_set_num_threads(ctx->n_threads);
#endif

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13/r3;
            const int64_t i02 = i12/r2;

            const float * x = (float *) ((char *) src0->data + i02*nb02 + i03*nb03);
            const float * y = (float *) ((char *) src1->data + i12*nb12 + i13*nb13);
                  float * d = (float *) ((char *)  dst->data + i12*nb2  + i13*nb3);

            if (type != GGML_TYPE_F32) {
                x = (float *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }

    return true;
}

static void ggml_backend_blas_out_prod(ggml_backend_blas_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne0  == ne00);
    GGML_ASSERT(ne1  == ne10);
    GGML_ASSERT(ne2  == ne02);
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne3  == ne13);
    GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == sizeof(float));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // Arguments to ggml_compute_forward_out_prod (expressed as major,minor)
    // src0: (k,n)
    // src1: (k,m)
    // dst:  (m,n)
    //
    // Arguments to sgemm (see https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/sgemm.f)
    // Also expressed as (major,minor)
    // a: (m,k): so src1 transposed
    // b: (k,n): so src0
    // c: (m,n)
    //
    // However, if ggml_is_transposed(src1) is true, then
    // src1->data already contains a transposed version, so sgemm mustn't
    // transpose it further.

    int n = src0->ne[0];
    int k = src0->ne[1];
    int m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    int lda;

    if (!ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } else {
        transposeA = CblasNoTrans;
        lda = k;
    }

    float * a = (float *) ((char *) src1->data);
    float * b = (float *) ((char *) src0->data);
    float * c = (float *) ((char *) dst->data);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    GGML_UNUSED(ctx);
}

// backend interface

static const char * ggml_backend_blas_get_name(ggml_backend_t backend) {
    return "BLAS";

    GGML_UNUSED(backend);
}

static void ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    if (ctx->memory_health == GGML_BACKEND_MEMORY_QUARANTINED) {
        return GGML_STATUS_BACKEND_POISONED;
    }

    ggml_backend_abort_context_mark_native(
        &ctx->abort, GGML_BACKEND_GRAPH_CANCEL_OBSERVATION_SUBMISSION_CHECKPOINT);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (ggml_backend_abort_context_requested(&ctx->abort)) {
            return GGML_STATUS_ABORTED;
        }
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                if (!ggml_backend_blas_mul_mat(ctx, node)) {
                    return GGML_STATUS_ALLOC_FAILED;
                }
                break;

            case GGML_OP_OUT_PROD:
                ggml_backend_blas_out_prod(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static uint64_t ggml_backend_blas_memory_round_page(uint64_t size) {
#if defined(_WIN32)
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    const uint64_t page = system_info.dwPageSize;
#else
    const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
#endif
    if (page == 0 || size > UINT64_MAX - (page - 1)) {
        return size;
    }
    return ((size + page - 1) / page) * page;
}

static ggml_backend_memory_domain_id_v1 ggml_backend_blas_memory_domain(void) {
    ggml_backend_memory_domain_id_v1 id = {};
    id.kind = GGML_BACKEND_MEMORY_DOMAIN_HOST_PAGEABLE;
    return id;
}

static bool ggml_backend_blas_graph_workspace(
        const struct ggml_cgraph * graph, uint64_t * workspace) {
    if (graph == nullptr || workspace == nullptr) {
        return false;
    }
    uint64_t maximum = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || node->op != GGML_OP_MUL_MAT) {
            continue;
        }
        uint64_t bytes = 0;
        if (!ggml_backend_blas_mul_mat_workspace(node, &bytes)) {
            return false;
        }
        maximum = std::max(maximum, bytes);
    }
    *workspace = maximum;
    return true;
}

static enum ggml_status ggml_backend_blas_memory_get_domains(
        ggml_backend_dev_t dev, ggml_backend_memory_domain_v1 * domains, uint32_t * inout_count) {
    GGML_UNUSED(dev);
    if (inout_count == nullptr) {
        return GGML_STATUS_FAILED;
    }
    const uint32_t capacity = *inout_count;
    *inout_count = 1;
    if (domains == nullptr) {
        return GGML_STATUS_SUCCESS;
    }
    if (capacity < 1 || domains[0].struct_size < sizeof(domains[0])) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_memory_domain_v1 domain = {};
    domain.struct_size = sizeof(domain);
    domain.id = ggml_backend_blas_memory_domain();
    snprintf(domain.name, sizeof(domain.name), "host/pageable");
    domains[0] = domain;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_blas_memory_quote(
        const ggml_backend_memory_request_v1 * requests, uint32_t request_count,
        ggml_backend_memory_quote_v1 * quote, ggml_backend_memory_claim_v1 * claims,
        uint32_t * inout_claim_count) {
    if ((request_count > 0 && requests == nullptr) || quote == nullptr ||
            quote->struct_size < sizeof(*quote) || inout_claim_count == nullptr) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_t backend = nullptr;
    uint32_t individual = 0;
    uint64_t maximum_workspace = 0;
    for (uint32_t i = 0; i < request_count; ++i) {
        if (requests[i].struct_size < sizeof(requests[i]) ||
                (requests[i].backend != nullptr && !ggml_backend_is_blas(requests[i].backend)) ||
                (backend != nullptr && requests[i].backend != nullptr && requests[i].backend != backend)) {
            return GGML_STATUS_FAILED;
        }
        backend = backend != nullptr ? backend : requests[i].backend;
        if (requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_BUFFER ||
                requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT) {
            individual++;
        } else if (requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE) {
            uint64_t workspace = 0;
            if (requests[i].backend == nullptr ||
                    !ggml_backend_blas_graph_workspace(requests[i].graph, &workspace)) {
                return GGML_STATUS_FAILED;
            }
            maximum_workspace = std::max(maximum_workspace, workspace);
        }
    }
    const uint32_t required = individual + (maximum_workspace > 0 ? 1u : 0u);
    const uint32_t capacity = *inout_claim_count;
    *inout_claim_count = required;
    const ggml_backend_blas_context * ctx = backend == nullptr
        ? nullptr : (const ggml_backend_blas_context *) backend->context;
    const uint64_t generation = ctx == nullptr ? 1 : ctx->memory_generation;
    quote->flags = 0;
    quote->residual_flags = 0;
    quote->residual_request_count = 0;
    quote->provisional_requested_upper_bytes = 0;
    quote->stats_generation = generation;
    quote->request_fingerprint = ggml_backend_memory_request_fingerprint_v1(requests, request_count);
    quote->quote_token = quote->request_fingerprint ^ generation;
    if (claims == nullptr) {
        return GGML_STATUS_SUCCESS;
    }
    if (capacity < required) {
        return GGML_STATUS_FAILED;
    }
    uint32_t out = 0;
    for (uint32_t i = 0; i < request_count; ++i) {
        if (requests[i].kind != GGML_BACKEND_MEMORY_REQUEST_BUFFER &&
                requests[i].kind != GGML_BACKEND_MEMORY_REQUEST_HOST_IMPORT) {
            continue;
        }
        ggml_backend_memory_claim_v1 claim = {};
        claim.struct_size = sizeof(claim);
        claim.flags = GGML_BACKEND_MEMORY_CLAIM_CONSERVATIVE_UPPER;
        claim.domain = ggml_backend_blas_memory_domain();
        claim.request_id = requests[i].request_id;
        claim.payload_requested_bytes = requests[i].requested_bytes;
        claim.committed_before_bytes = ggml_backend_blas_memory_round_page(requests[i].currently_allocated_bytes);
        const bool reuse = requests[i].currently_allocated_bytes >= requests[i].requested_bytes;
        const uint64_t requested = ggml_backend_blas_memory_round_page(requests[i].requested_bytes);
        const uint64_t after = reuse ? claim.committed_before_bytes : requested;
        claim.committed_after_upper_bytes = after;
        claim.commit_peak_extra_upper_bytes = reuse ? 0 : requested;
        claim.resident_after_upper_bytes = after;
        claim.retained_after_use_upper_bytes = after;
        claims[out++] = claim;
    }
    if (maximum_workspace > 0) {
        const uint64_t before = ctx == nullptr ? 0 : ctx->work_size;
        const uint64_t after = std::max(before, maximum_workspace);
        ggml_backend_memory_claim_v1 claim = {};
        claim.struct_size = sizeof(claim);
        claim.flags = GGML_BACKEND_MEMORY_CLAIM_EXACT |
            GGML_BACKEND_MEMORY_CLAIM_REUSABLE_WORKSPACE;
        claim.domain = ggml_backend_blas_memory_domain();
        claim.payload_requested_bytes = maximum_workspace;
        claim.committed_before_bytes = before;
        claim.committed_after_upper_bytes = after;
        claim.commit_peak_extra_upper_bytes = after > before ? after : 0;
        claim.resident_after_upper_bytes = after;
        claim.retained_after_use_upper_bytes = after;
        claim.releasable_after_use_upper_bytes = after;
        claims[out++] = claim;
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_blas_memory_reserve_private(
        const ggml_backend_memory_request_v1 * requests, uint32_t request_count,
        const ggml_backend_memory_quote_v1 * quote, ggml_backend_memory_claim_v1 * actual,
        uint32_t * inout_actual_count) {
    if ((request_count > 0 && requests == nullptr) || quote == nullptr ||
            quote->struct_size < sizeof(*quote) || inout_actual_count == nullptr ||
            quote->request_fingerprint != ggml_backend_memory_request_fingerprint_v1(requests, request_count)) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_t backend = nullptr;
    uint64_t maximum_workspace = 0;
    for (uint32_t i = 0; i < request_count; ++i) {
        if (requests[i].struct_size < sizeof(requests[i]) ||
                (requests[i].backend != nullptr && !ggml_backend_is_blas(requests[i].backend)) ||
                (backend != nullptr && requests[i].backend != nullptr && requests[i].backend != backend)) {
            return GGML_STATUS_FAILED;
        }
        backend = backend != nullptr ? backend : requests[i].backend;
        if (requests[i].kind == GGML_BACKEND_MEMORY_REQUEST_GRAPH_PRIVATE) {
            uint64_t workspace = 0;
            if (!ggml_backend_blas_graph_workspace(requests[i].graph, &workspace)) {
                return GGML_STATUS_FAILED;
            }
            maximum_workspace = std::max(maximum_workspace, workspace);
        }
    }
    const uint32_t required = maximum_workspace > 0 ? 1u : 0u;
    const uint32_t capacity = *inout_actual_count;
    *inout_actual_count = required;
    if (actual == nullptr) {
        return GGML_STATUS_SUCCESS;
    }
    if (capacity < required) {
        return GGML_STATUS_FAILED;
    }
    if (backend == nullptr) {
        return GGML_STATUS_SUCCESS;
    }
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *) backend->context;
    if (ctx->memory_health == GGML_BACKEND_MEMORY_QUARANTINED) {
        return GGML_STATUS_BACKEND_POISONED;
    }
    const uint64_t generation = ctx->memory_generation;
    if (quote->stats_generation != generation ||
            quote->quote_token != (quote->request_fingerprint ^ generation)) {
        return GGML_STATUS_FAILED;
    }
    const uint64_t before = ctx->work_size;
    if (ctx->work_size < maximum_workspace) {
        std::unique_ptr<char[]> replacement(new (std::nothrow) char[maximum_workspace]);
        if (replacement == nullptr) {
            ctx->allocation_failures++;
            return GGML_STATUS_ALLOC_FAILED;
        }
        ctx->work_data = std::move(replacement);
        ctx->work_size = maximum_workspace;
        ctx->memory_high_water = std::max(ctx->memory_high_water, (uint64_t) ctx->work_size);
        ctx->memory_generation++;
    }
    if (required > 0) {
        ggml_backend_memory_claim_v1 claim = {};
        claim.struct_size = sizeof(claim);
        claim.flags = GGML_BACKEND_MEMORY_CLAIM_EXACT |
            GGML_BACKEND_MEMORY_CLAIM_REUSABLE_WORKSPACE;
        claim.domain = ggml_backend_blas_memory_domain();
        claim.payload_requested_bytes = maximum_workspace;
        claim.committed_before_bytes = before;
        claim.committed_after_upper_bytes = ctx->work_size;
        claim.commit_peak_extra_upper_bytes = ctx->work_size > before ? ctx->work_size : 0;
        claim.resident_after_upper_bytes = ctx->work_size;
        claim.retained_after_use_upper_bytes = ctx->work_size;
        claim.releasable_after_use_upper_bytes = ctx->work_size;
        actual[0] = claim;
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_blas_memory_get_stats(
        ggml_backend_dev_t dev, ggml_backend_t backend,
        ggml_backend_memory_stats_v1 * stats, uint32_t * inout_count) {
    if (dev == nullptr || inout_count == nullptr) {
        return GGML_STATUS_FAILED;
    }
    const uint32_t capacity = *inout_count;
    *inout_count = 1;
    if (stats == nullptr) {
        return GGML_STATUS_SUCCESS;
    }
    if (capacity < 1 || stats[0].struct_size < sizeof(stats[0])) {
        return GGML_STATUS_FAILED;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    ggml_backend_memory_stats_v1 value = {};
    value.struct_size = sizeof(value);
    value.domain = ggml_backend_blas_memory_domain();
    value.total_bytes = total_bytes;
    value.budget_bytes = total_bytes;
    value.device_free_bytes = free_bytes;
    value.device_used_bytes = total_bytes >= free_bytes ? total_bytes - free_bytes : 0;
    value.timestamp_monotonic_ns = (uint64_t) ggml_time_us() * 1000;
    value.health = GGML_BACKEND_MEMORY_HEALTHY;
    if (backend != nullptr) {
        ggml_backend_blas_context * ctx = (ggml_backend_blas_context *) backend->context;
        value.generation = ctx->memory_generation;
        value.backend_owned_workspace_bytes = ctx->work_size;
        value.backend_owned_live_bytes = ctx->work_size;
        value.backend_owned_high_water_bytes = ctx->memory_high_water;
        value.allocation_failure_count = ctx->allocation_failures;
        value.health = ctx->memory_health;
        value.quarantine_generation = ctx->quarantine_generation;
        value.last_native_error = ctx->last_native_error;
    }
    stats[0] = value;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_blas_memory_trim(ggml_backend_t backend, uint64_t flags) {
    GGML_UNUSED(flags);
    if (backend == nullptr || !ggml_backend_is_blas(backend)) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *) backend->context;
    ctx->work_data.reset();
    ctx->work_size = 0;
    ctx->memory_generation++;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_blas_memory_quarantine(
        ggml_backend_t backend, const ggml_backend_memory_quarantine_v1 * request) {
    if (backend == nullptr || !ggml_backend_is_blas(backend) || request == nullptr ||
            request->struct_size < sizeof(*request)) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *) backend->context;
    ctx->memory_health = GGML_BACKEND_MEMORY_QUARANTINED;
    ctx->last_native_error = request->native_error;
    ctx->quarantine_generation++;
    ctx->memory_generation++;
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_memory_api_v1 ggml_backend_blas_memory_api = {
    sizeof(ggml_backend_memory_api_v1), GGML_BACKEND_MEMORY_ABI_V1, 0,
    ggml_backend_blas_memory_get_domains, ggml_backend_blas_memory_quote,
    ggml_backend_blas_memory_reserve_private, ggml_backend_blas_memory_get_stats,
    ggml_backend_blas_memory_trim, ggml_backend_blas_memory_quarantine,
};

static struct ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record_status     = */ NULL,
    /* .event_wait_status       = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_blas_guid(void) {
    static ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    return &guid;
}

ggml_backend_t ggml_backend_blas_init(void) {
    ggml_backend_blas_context * ctx = new ggml_backend_blas_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#if defined(GGML_BLAS_USE_OPENBLAS) && defined(GGML_USE_OPENMP)
    if (openblas_get_parallel() != OPENBLAS_OPENMP) {
        GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#endif

#if defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    return backend;
}

bool ggml_backend_is_blas(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

void ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

static void ggml_backend_blas_set_abort_callback(
        ggml_backend_t backend, ggml_abort_callback abort_callback, void * abort_callback_data,
        struct ggml_backend_graph_cancel_capability * cancel_capability) {
    GGML_ASSERT(ggml_backend_is_blas(backend));
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *) backend->context;
    ggml_backend_abort_context_set(
        &ctx->abort, abort_callback, abort_callback_data, cancel_capability);
}

// device interface

static const char * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    return "BLAS";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    #if defined(GGML_BLAS_USE_ACCELERATE)
        return "Accelerate";
    #elif defined(GGML_BLAS_USE_MKL)
        return "MKL";
    #elif defined(GGML_BLAS_USE_BLIS)
        return "BLIS";
    #elif defined(GGML_BLAS_USE_NVPL)
        return "NVPL";
    #elif defined(GGML_BLAS_USE_OPENBLAS)
        return "OpenBLAS";
    #else
        return "BLAS";
    #endif

    GGML_UNUSED(dev);
}

static void ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#elif defined(__APPLE__)
    int64_t memory_size = 0;
    size_t memory_size_len = sizeof(memory_size);
    if (sysctlbyname("hw.memsize", &memory_size, &memory_size_len, NULL, 0) != 0) {
        *free = 0;
        *total = 0;
        return;
    }
    vm_statistics64_data_t vm = {};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &count) != KERN_SUCCESS) {
        *free = 0;
        *total = (size_t) memory_size;
        return;
    }
    const uint64_t page = (uint64_t) vm_kernel_page_size;
    *total = (size_t) memory_size;
    *free = (size_t) ((vm.free_count + vm.inactive_count + vm.speculative_count) * page);
#elif defined(__linux__)
    struct sysinfo status = {};
    if (sysinfo(&status) != 0) {
        *free = 0;
        *total = 0;
        return;
    }
    *total = (size_t) status.totalram * status.mem_unit;
    *free = (size_t) (status.freeram + status.bufferram) * status.mem_unit;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long available_pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages > 0 && page_size > 0 ? (size_t) pages * (size_t) page_size : 0;
    *free = available_pages > 0 && page_size > 0
        ? (size_t) available_pages * (size_t) page_size : 0;
#endif

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_blas_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            const int64_t ne10 = src1->ne[0];

            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            const int64_t min_batch = 32;

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                return false;
            }

            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
        }

        case GGML_OP_OUT_PROD:
            return op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_matrix(src0) &&
                   ggml_is_matrix(src1) &&
                   ggml_is_contiguous(src0) &&
                   (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);

        default:
            return false;

    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    return "BLAS";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_blas_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_blas_set_n_threads;
    }
    if (std::strcmp(name, GGML_BACKEND_MEMORY_API_V1_PROC) == 0) {
        return (void *) +[]() -> const ggml_backend_memory_api_v1 * {
            return &ggml_backend_blas_memory_api;
        };
    }
    if (std::strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_blas_set_abort_callback;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_count,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(void) {
    static struct ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
