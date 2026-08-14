#include "common.cuh"

static constexpr bool ggml_cuda_ssm_conv_supports_d_conv(int64_t d_conv) {
    switch (d_conv) {
        case 3:
        case 4:
        case 5:
        case 9:
        case 15:
        case 20:
            return true;
        default:
            return false;
    }
}

static inline bool ggml_cuda_ssm_conv_supports_shape(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return src0->ne[0] == src1->ne[0] - 1 + dst->ne[1] &&
           src0->ne[1] == src1->ne[1] &&
           src0->ne[1] == dst->ne[0] &&
           src0->ne[2] == dst->ne[2] &&
           src0->ne[3] == 1 &&
           src1->ne[2] == 1 &&
           src1->ne[3] == 1 &&
           dst->ne[3] == 1 &&
           dst->nb[0] == sizeof(float);
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);
