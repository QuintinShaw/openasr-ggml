#include "lstm-seq.cuh"

static __device__ __forceinline__ float ggml_cuda_lstm_sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }

    const float z = expf(x);
    return z / (1.0f + z);
}

static __global__ void ggml_cuda_lstm_seq_gates_f32(
        const float * x,
        const float * w,
        const float * r,
        const float * bias,
        float * workspace,
        const float * dst,
        int64_t input,
        int64_t seq,
        int64_t hidden,
        int64_t step,
        bool reverse,
        bool split_bias) {
    const int lane = threadIdx.x;
    const int64_t gates_per_batch = 4 * hidden;
    const int64_t block = static_cast<int64_t>(blockIdx.x);
    const int64_t batch = block / gates_per_batch;
    const int64_t gate_unit = block % gates_per_batch;
    const int64_t time = reverse ? seq - 1 - step : step;

    __shared__ float partials[32];

    float sum = 0.0f;
    const float * x_row = x + (batch * seq + time) * input;
    for (int64_t k = lane; k < input; k += blockDim.x) {
        sum += x_row[k] * w[gate_unit * input + k];
    }

    if (step != 0) {
        const int64_t previous_time = reverse ? seq - step : step - 1;
        const float * h_previous = dst + (batch * seq + previous_time) * hidden;
        for (int64_t k = lane; k < hidden; k += blockDim.x) {
            sum += h_previous[k] * r[gate_unit * hidden + k];
        }
    }

    partials[lane] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (lane < offset) {
            partials[lane] += partials[lane + offset];
        }
        __syncthreads();
    }

    if (lane == 0) {
        float value = partials[0] + bias[gate_unit];
        if (split_bias) {
            value += bias[gates_per_batch + gate_unit];
        }
        workspace[batch * (5 * hidden) + gate_unit] = value;
    }
}

static __global__ void ggml_cuda_lstm_seq_update_f32(
        float * workspace,
        float * dst,
        int64_t seq,
        int64_t hidden,
        int64_t step,
        int32_t gate_order,
        bool reverse,
        int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }

    const int64_t batch = index / hidden;
    const int64_t unit = index % hidden;
    const int64_t workspace_base = batch * (5 * hidden);
    const int64_t forget_gate = gate_order == 0 ? 2 : 1;
    const int64_t cell_gate = gate_order == 0 ? 3 : 2;
    const int64_t output_gate = gate_order == 0 ? 1 : 3;
    const int64_t cell_offset = workspace_base + 4 * hidden + unit;

    const float input_value = ggml_cuda_lstm_sigmoid(workspace[workspace_base + unit]);
    const float forget_value = ggml_cuda_lstm_sigmoid(workspace[workspace_base + forget_gate * hidden + unit]);
    const float cell_value = tanhf(workspace[workspace_base + cell_gate * hidden + unit]);
    const float output_value = ggml_cuda_lstm_sigmoid(workspace[workspace_base + output_gate * hidden + unit]);
    const float previous_cell = step == 0 ? 0.0f : workspace[cell_offset];
    const float cell = forget_value * previous_cell + input_value * cell_value;
    const int64_t time = reverse ? seq - 1 - step : step;

    workspace[cell_offset] = cell;
    dst[(batch * seq + time) * hidden + unit] = output_value * tanhf(cell);
}

void ggml_cuda_op_lstm_seq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_lstm_seq_supported(dst));

    const ggml_tensor * x    = dst->src[0];
    const ggml_tensor * w    = dst->src[1];
    const ggml_tensor * r    = dst->src[2];
    const ggml_tensor * bias = dst->src[3];
    const ggml_tensor * workspace = dst->src[4];

    const int64_t hidden = dst->ne[0];
    const int64_t seq    = dst->ne[1];
    const int64_t batch  = dst->ne[2];
    const int32_t gate_order = dst->op_params[0];
    const bool reverse    = dst->op_params[1] != 0;
    const bool split_bias = bias->ne[0] == 8 * hidden;
    const int64_t gates_per_batch = 4 * hidden;
    const int64_t gate_blocks = gates_per_batch * batch;
    const int64_t update_elements = hidden * batch;
    const cudaStream_t stream = ctx.stream();

    const ggml_cuda_kernel_launch_params gate_launch(
        dim3(static_cast<unsigned int>(gate_blocks), 1, 1), dim3(32, 1, 1), 0, stream);
    const ggml_cuda_kernel_launch_params update_launch(
        dim3(static_cast<unsigned int>((update_elements + 255) / 256), 1, 1), dim3(256, 1, 1), 0, stream);

    for (int64_t step = 0; step < seq; ++step) {
        ggml_cuda_kernel_launch(ggml_cuda_lstm_seq_gates_f32, gate_launch,
            static_cast<const float *>(x->data),
            static_cast<const float *>(w->data),
            static_cast<const float *>(r->data),
            static_cast<const float *>(bias->data),
            static_cast<float *>(workspace->data),
            static_cast<const float *>(dst->data),
            x->ne[0], seq, hidden, step, reverse, split_bias);
        ggml_cuda_kernel_launch(ggml_cuda_lstm_seq_update_f32, update_launch,
            static_cast<float *>(workspace->data),
            static_cast<float *>(dst->data),
            seq, hidden, step, gate_order, reverse, update_elements);
    }
}
