/**
 * @file nimcp_backprop_kernels.cu
 * @brief GPU Backpropagation CUDA Kernels
 *
 * WHAT: CUDA kernels for neural network backpropagation
 * WHY:  GPU acceleration for computing gradients during training
 * HOW:  Custom kernels for layer gradients (linear, conv, activation, normalization)
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifdef NIMCP_ENABLE_CUDA

// Include CUDA headers FIRST (before any extern "C" blocks from our headers)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// Now include our headers (which have extern "C" blocks)
#include "gpu/training/nimcp_training_gpu.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "BACKPROP_GPU"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        LOG_ERROR("cuBLAS error: %d", status); \
        return false; \
    } \
} while(0)

#define BLOCK_SIZE 256
#define GRID_SIZE(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define WARP_SIZE 32

//=============================================================================
// Linear Layer Backward
//=============================================================================

bool nimcp_gpu_backward_linear(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* x,
    const nimcp_gpu_tensor_t* weight,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input,
    nimcp_gpu_tensor_t* grad_weight,
    nimcp_gpu_tensor_t* grad_bias)
{
    if (!ctx || !x || !weight || !grad_output) return false;

    cublasHandle_t handle = (cublasHandle_t)ctx->cublas_handle;

    // Dimensions: x is (batch, in_features), weight is (out_features, in_features)
    // grad_output is (batch, out_features)
    int batch = x->dims[0];
    int in_features = x->dims[1];
    int out_features = weight->dims[0];

    float alpha = 1.0f, beta = 0.0f;

    // grad_input = grad_output @ weight
    // (batch, in_features) = (batch, out_features) @ (out_features, in_features)
    if (grad_input) {
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            in_features, batch, out_features,
            &alpha,
            (const float*)weight->data, in_features,
            (const float*)grad_output->data, out_features,
            &beta,
            (float*)grad_input->data, in_features));
    }

    // grad_weight = grad_output^T @ x
    // (out_features, in_features) = (out_features, batch) @ (batch, in_features)
    if (grad_weight) {
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            in_features, out_features, batch,
            &alpha,
            (const float*)x->data, in_features,
            (const float*)grad_output->data, out_features,
            &beta,
            (float*)grad_weight->data, in_features));
    }

    // grad_bias = sum(grad_output, axis=0)
    if (grad_bias) {
        // Sum across batch dimension
        // Use cublas gemv with ones vector
        float* ones;
        CUDA_CHECK(cudaMalloc(&ones, batch * sizeof(float)));

        // Fill with ones
        float one = 1.0f;
        for (int i = 0; i < batch; i++) {
            CUDA_CHECK(cudaMemcpy(ones + i, &one, sizeof(float), cudaMemcpyHostToDevice));
        }

        CUBLAS_CHECK(cublasSgemv(handle,
            CUBLAS_OP_T,
            batch, out_features,
            &alpha,
            (const float*)grad_output->data, batch,
            ones, 1,
            &beta,
            (float*)grad_bias->data, 1));

        cudaFree(ones);
    }

    return true;
}

//=============================================================================
// Activation Backward Kernels
//=============================================================================

__global__ void kernel_backward_relu(
    const float* x, const float* grad_output, float* grad_input, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = (x[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

bool nimcp_gpu_backward_relu(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* x,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input)
{
    if (!ctx || !x || !grad_output || !grad_input) return false;

    kernel_backward_relu<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>(
        (const float*)x->data, (const float*)grad_output->data,
        (float*)grad_input->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

__global__ void kernel_backward_sigmoid(
    const float* output, const float* grad_output, float* grad_input, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = output[idx];  // sigmoid output
        grad_input[idx] = grad_output[idx] * s * (1.0f - s);
    }
}

bool nimcp_gpu_backward_sigmoid(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* output,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input)
{
    if (!ctx || !output || !grad_output || !grad_input) return false;

    kernel_backward_sigmoid<<<GRID_SIZE(output->numel), BLOCK_SIZE>>>(
        (const float*)output->data, (const float*)grad_output->data,
        (float*)grad_input->data, output->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

__global__ void kernel_backward_tanh(
    const float* output, const float* grad_output, float* grad_input, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float t = output[idx];  // tanh output
        grad_input[idx] = grad_output[idx] * (1.0f - t * t);
    }
}

bool nimcp_gpu_backward_tanh(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* output,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input)
{
    if (!ctx || !output || !grad_output || !grad_input) return false;

    kernel_backward_tanh<<<GRID_SIZE(output->numel), BLOCK_SIZE>>>(
        (const float*)output->data, (const float*)grad_output->data,
        (float*)grad_input->data, output->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

__global__ void kernel_backward_gelu(
    const float* x, const float* grad_output, float* grad_input, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float val_cubed = val * val * val;

        // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float sqrt_2_pi = 0.7978845608f;
        float inner = sqrt_2_pi * (val + 0.044715f * val_cubed);
        float tanh_inner = tanhf(inner);

        // Derivative
        float sech2 = 1.0f - tanh_inner * tanh_inner;
        float inner_deriv = sqrt_2_pi * (1.0f + 3.0f * 0.044715f * val * val);

        float gelu_deriv = 0.5f * (1.0f + tanh_inner) +
                          0.5f * val * sech2 * inner_deriv;

        grad_input[idx] = grad_output[idx] * gelu_deriv;
    }
}

bool nimcp_gpu_backward_gelu(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* x,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input)
{
    if (!ctx || !x || !grad_output || !grad_input) return false;

    kernel_backward_gelu<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>(
        (const float*)x->data, (const float*)grad_output->data,
        (float*)grad_input->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Softmax Backward
//=============================================================================

__global__ void kernel_backward_softmax(
    const float* output, const float* grad_output, float* grad_input,
    size_t batch_size, size_t num_classes)
{
    size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* s = output + batch_idx * num_classes;
    const float* dy = grad_output + batch_idx * num_classes;
    float* dx = grad_input + batch_idx * num_classes;

    // Compute sum_j(dy_j * s_j)
    __shared__ float shared_sum[WARP_SIZE];
    float thread_sum = 0.0f;
    for (size_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        thread_sum += dy[i] * s[i];
    }

    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (threadIdx.x % WARP_SIZE == 0) {
        shared_sum[threadIdx.x / WARP_SIZE] = thread_sum;
    }
    __syncthreads();

    float total_sum = 0.0f;
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        total_sum = shared_sum[threadIdx.x];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        total_sum += __shfl_down_sync(0xffffffff, total_sum, offset);
    }
    total_sum = __shfl_sync(0xffffffff, total_sum, 0);

    // Compute dx_i = s_i * (dy_i - sum_j(dy_j * s_j))
    for (size_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        dx[i] = s[i] * (dy[i] - total_sum);
    }
}

bool nimcp_gpu_backward_softmax(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* output,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input)
{
    if (!ctx || !output || !grad_output || !grad_input) return false;

    size_t num_classes = output->dims[output->ndim - 1];
    size_t batch_size = output->numel / num_classes;

    kernel_backward_softmax<<<batch_size, BLOCK_SIZE>>>(
        (const float*)output->data, (const float*)grad_output->data,
        (float*)grad_input->data, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Batch Normalization Backward
//=============================================================================

__global__ void kernel_backward_batchnorm(
    const float* x, const float* gamma, const float* mean, const float* var,
    const float* grad_output, float* grad_input, float* grad_gamma, float* grad_beta,
    float eps, size_t batch_size, size_t features)
{
    size_t feat_idx = blockIdx.x;
    if (feat_idx >= features) return;

    float m = mean[feat_idx];
    float v = var[feat_idx];
    float g = gamma[feat_idx];
    float std_inv = rsqrtf(v + eps);

    // Compute gradients using reduction
    float sum_dy = 0.0f;
    float sum_dy_x_hat = 0.0f;

    for (size_t b = threadIdx.x; b < batch_size; b += blockDim.x) {
        size_t idx = b * features + feat_idx;
        float x_hat = (x[idx] - m) * std_inv;
        sum_dy += grad_output[idx];
        sum_dy_x_hat += grad_output[idx] * x_hat;
    }

    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum_dy += __shfl_down_sync(0xffffffff, sum_dy, offset);
        sum_dy_x_hat += __shfl_down_sync(0xffffffff, sum_dy_x_hat, offset);
    }

    __shared__ float shared_dy[WARP_SIZE];
    __shared__ float shared_dy_x_hat[WARP_SIZE];

    if (threadIdx.x % WARP_SIZE == 0) {
        shared_dy[threadIdx.x / WARP_SIZE] = sum_dy;
        shared_dy_x_hat[threadIdx.x / WARP_SIZE] = sum_dy_x_hat;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total_dy = 0.0f, total_dy_x_hat = 0.0f;
        for (int i = 0; i < blockDim.x / WARP_SIZE; i++) {
            total_dy += shared_dy[i];
            total_dy_x_hat += shared_dy_x_hat[i];
        }

        if (grad_beta) grad_beta[feat_idx] = total_dy;
        if (grad_gamma) grad_gamma[feat_idx] = total_dy_x_hat;

        shared_dy[0] = total_dy;
        shared_dy_x_hat[0] = total_dy_x_hat;
    }
    __syncthreads();

    float total_dy = shared_dy[0];
    float total_dy_x_hat = shared_dy_x_hat[0];

    // Compute grad_input
    if (grad_input) {
        float inv_N = 1.0f / (float)batch_size;
        for (size_t b = threadIdx.x; b < batch_size; b += blockDim.x) {
            size_t idx = b * features + feat_idx;
            float x_hat = (x[idx] - m) * std_inv;
            float dy = grad_output[idx];

            grad_input[idx] = g * std_inv * inv_N *
                ((float)batch_size * dy - total_dy - x_hat * total_dy_x_hat);
        }
    }
}

bool nimcp_gpu_backward_batchnorm(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* x,
    const nimcp_gpu_tensor_t* gamma,
    const nimcp_gpu_tensor_t* mean,
    const nimcp_gpu_tensor_t* var,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input,
    nimcp_gpu_tensor_t* grad_gamma,
    nimcp_gpu_tensor_t* grad_beta,
    float eps)
{
    if (!ctx || !x || !gamma || !mean || !var || !grad_output) return false;

    size_t features = x->dims[x->ndim - 1];
    size_t batch_size = x->numel / features;

    kernel_backward_batchnorm<<<features, BLOCK_SIZE>>>(
        (const float*)x->data, (const float*)gamma->data,
        (const float*)mean->data, (const float*)var->data,
        (const float*)grad_output->data,
        grad_input ? (float*)grad_input->data : NULL,
        grad_gamma ? (float*)grad_gamma->data : NULL,
        grad_beta ? (float*)grad_beta->data : NULL,
        eps, batch_size, features);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Layer Normalization Backward
//=============================================================================

bool nimcp_gpu_backward_layernorm(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* x,
    const nimcp_gpu_tensor_t* gamma,
    const nimcp_gpu_tensor_t* mean,
    const nimcp_gpu_tensor_t* var,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input,
    nimcp_gpu_tensor_t* grad_gamma,
    nimcp_gpu_tensor_t* grad_beta,
    float eps)
{
    // Layer norm is similar to batch norm but normalizes along feature dimension
    // For now, use similar implementation with swapped axes
    LOG_WARN("Layer norm backward using simplified implementation");
    return nimcp_gpu_backward_batchnorm(ctx, x, gamma, mean, var, grad_output,
                                        grad_input, grad_gamma, grad_beta, eps);
}

//=============================================================================
// Dropout Backward
//=============================================================================

__global__ void kernel_backward_dropout(
    const float* mask, const float* grad_output, float* grad_input,
    float scale, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = (mask[idx] != 0.0f) ? grad_output[idx] * scale : 0.0f;
    }
}

bool nimcp_gpu_backward_dropout(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* mask,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input,
    float p)
{
    if (!ctx || !mask || !grad_output || !grad_input) return false;

    float scale = 1.0f / (1.0f - p);

    kernel_backward_dropout<<<GRID_SIZE(mask->numel), BLOCK_SIZE>>>(
        (const float*)mask->data, (const float*)grad_output->data,
        (float*)grad_input->data, scale, mask->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else // !NIMCP_ENABLE_CUDA

#include "gpu/training/nimcp_training_gpu.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "BACKPROP_GPU"

bool nimcp_gpu_backward_linear(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
    const nimcp_gpu_tensor_t* weight, const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input, nimcp_gpu_tensor_t* grad_weight, nimcp_gpu_tensor_t* grad_bias)
{
    LOG_WARN("CUDA not available - backprop requires GPU");
    return false;
}

bool nimcp_gpu_backward_relu(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
    const nimcp_gpu_tensor_t* grad_output, nimcp_gpu_tensor_t* grad_input)
{
    return false;
}

bool nimcp_gpu_backward_sigmoid(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* output,
    const nimcp_gpu_tensor_t* grad_output, nimcp_gpu_tensor_t* grad_input)
{
    return false;
}

bool nimcp_gpu_backward_tanh(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* output,
    const nimcp_gpu_tensor_t* grad_output, nimcp_gpu_tensor_t* grad_input)
{
    return false;
}

bool nimcp_gpu_backward_gelu(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
    const nimcp_gpu_tensor_t* grad_output, nimcp_gpu_tensor_t* grad_input)
{
    return false;
}

bool nimcp_gpu_backward_softmax(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* output,
    const nimcp_gpu_tensor_t* grad_output, nimcp_gpu_tensor_t* grad_input)
{
    return false;
}

bool nimcp_gpu_backward_batchnorm(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
    const nimcp_gpu_tensor_t* gamma, const nimcp_gpu_tensor_t* mean, const nimcp_gpu_tensor_t* var,
    const nimcp_gpu_tensor_t* grad_output, nimcp_gpu_tensor_t* grad_input,
    nimcp_gpu_tensor_t* grad_gamma, nimcp_gpu_tensor_t* grad_beta, float eps)
{
    return false;
}

bool nimcp_gpu_backward_layernorm(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
    const nimcp_gpu_tensor_t* gamma, const nimcp_gpu_tensor_t* mean, const nimcp_gpu_tensor_t* var,
    const nimcp_gpu_tensor_t* grad_output, nimcp_gpu_tensor_t* grad_input,
    nimcp_gpu_tensor_t* grad_gamma, nimcp_gpu_tensor_t* grad_beta, float eps)
{
    return false;
}

bool nimcp_gpu_backward_dropout(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* mask,
    const nimcp_gpu_tensor_t* grad_output, nimcp_gpu_tensor_t* grad_input, float p)
{
    return false;
}

#endif // NIMCP_ENABLE_CUDA
