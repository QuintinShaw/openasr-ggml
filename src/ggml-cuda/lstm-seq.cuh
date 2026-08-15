#pragma once

#include "common.cuh"

#include <cstdint>
#include <limits>

// GGML_OP_LSTM_SEQ uses src[0..4] = x, W, R, bias, gate and cell workspace and two int32 parameters:
// gate_order and reverse. Gate order 0 is IOFC and 1 is IFGO.
static inline bool ggml_cuda_lstm_seq_supported(const ggml_tensor * dst) {
    if (dst->op != GGML_OP_LSTM_SEQ || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const ggml_tensor * x    = dst->src[0];
    const ggml_tensor * w    = dst->src[1];
    const ggml_tensor * r    = dst->src[2];
    const ggml_tensor * bias = dst->src[3];
    const ggml_tensor * cell = dst->src[4];
    if (x == nullptr || w == nullptr || r == nullptr || bias == nullptr || cell == nullptr) {
        return false;
    }
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        if (dst->src[i] != nullptr) {
            return false;
        }
    }

    const int32_t gate_order = dst->op_params[0];
    const int32_t reverse    = dst->op_params[1];
    if ((reverse != 0 && reverse != 1) || (gate_order != 0 && gate_order != 1)) {
        return false;
    }

    const int64_t input  = x->ne[0];
    const int64_t seq    = x->ne[1];
    const int64_t batch  = x->ne[2];
    const int64_t hidden = dst->ne[0];
    if (input < 1 || seq < 1 || batch < 1 || hidden < 1 || hidden > 256 ||
        x->ne[3] != 1 || w->ne[2] != 1 || w->ne[3] != 1 ||
        r->ne[2] != 1 || r->ne[3] != 1 ||
        bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1 ||
        cell->ne[0] != 5 * hidden || cell->ne[1] != batch || cell->ne[2] != 1 || cell->ne[3] != 1 ||
        dst->ne[1] != seq || dst->ne[2] != batch || dst->ne[3] != 1 ||
        w->ne[0] != input || w->ne[1] != 4 * hidden ||
        r->ne[0] != hidden || r->ne[1] != 4 * hidden ||
        (bias->ne[0] != 4 * hidden && bias->ne[0] != 8 * hidden)) {
        return false;
    }
    if (static_cast<uint64_t>(batch) >
        std::numeric_limits<int>::max() / (4 * static_cast<uint64_t>(hidden))) {
        return false;
    }

    const ggml_tensor * tensors[] = { x, w, r, bias, cell, dst };
    for (const ggml_tensor * tensor : tensors) {
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
            tensor->nb[0] != sizeof(float) || tensor->view_offs % alignof(float) != 0 ||
            (tensor->data != nullptr && reinterpret_cast<uintptr_t>(tensor->data) % alignof(float) != 0)) {
            return false;
        }
    }

    return true;
}

void ggml_cuda_op_lstm_seq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
