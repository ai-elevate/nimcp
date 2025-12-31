/**
 * @file nimcp_conv_kernels.cu
 * @brief GPU CNN CUDA Kernels (Convolution, Pooling, Normalization)
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifdef NIMCP_ENABLE_CUDA

// Include CUDA headers FIRST (before any extern "C" blocks from our headers)
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Now include our headers (which have extern "C" blocks)
#include "gpu/cnn/nimcp_cnn_gpu.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "CNN_GPU"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

#define BLOCK_SIZE 16
#define WARP_SIZE 32

//=============================================================================
// Conv2D Forward Kernel
//=============================================================================

__global__ void kernel_conv2d_forward(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kH, int kW, int stride_h, int stride_w,
    int pad_h, int pad_w, int dilation_h, int dilation_w, int groups)
{
    int n = blockIdx.z;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    int h_out = blockIdx.x * blockDim.x + threadIdx.x / W_out;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x % W_out;

    if (c_out >= C_out || h_out >= H_out || w_out >= W_out) return;

    int group = c_out / (C_out / groups);
    int C_per_group = C_in / groups;

    float sum = 0.0f;

    for (int c_in = 0; c_in < C_per_group; c_in++) {
        int c = group * C_per_group + c_in;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                int w_in = w_out * stride_w - pad_w + kw * dilation_w;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int in_idx = n * C_in * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in;
                    int w_idx = c_out * C_per_group * kH * kW + c_in * kH * kW + kh * kW + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    if (bias) {
        sum += bias[c_out];
    }

    int out_idx = n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out;
    output[out_idx] = sum;
}

bool nimcp_gpu_conv2d_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* weight,
    const nimcp_gpu_tensor_t* bias,
    nimcp_gpu_tensor_t* output,
    const nimcp_conv_params_t* params)
{
    if (!ctx || !input || !weight || !output || !params) return false;

    int N = input->dims[0];
    int C_in = input->dims[1];
    int H_in = input->dims[2];
    int W_in = input->dims[3];

    int C_out = weight->dims[0];
    int kH = params->kernel_h;
    int kW = params->kernel_w;

    int H_out = (H_in + 2 * params->pad_h - params->dilation_h * (kH - 1) - 1) / params->stride_h + 1;
    int W_out = (W_in + 2 * params->pad_w - params->dilation_w * (kW - 1) - 1) / params->stride_w + 1;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((H_out * W_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (C_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
              N);

    kernel_conv2d_forward<<<grid, block>>>(
        (const float*)input->data, (const float*)weight->data,
        bias ? (const float*)bias->data : NULL, (float*)output->data,
        N, C_in, H_in, W_in, C_out, H_out, W_out,
        kH, kW, params->stride_h, params->stride_w,
        params->pad_h, params->pad_w, params->dilation_h, params->dilation_w,
        params->groups);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_conv2d_backward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* weight,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input,
    nimcp_gpu_tensor_t* grad_weight,
    nimcp_gpu_tensor_t* grad_bias,
    const nimcp_conv_params_t* params)
{
    // TODO: Implement conv2d backward
    LOG_WARN("Conv2D backward not yet implemented");
    return false;
}

//=============================================================================
// Conv1D Forward (Audio/Speech)
//=============================================================================

__global__ void kernel_conv1d_forward(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C_in, int L_in, int C_out, int L_out,
    int K, int stride, int padding, int dilation)
{
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int l_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (l_out >= L_out) return;

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int k = 0; k < K; k++) {
            int l_in = l_out * stride - padding + k * dilation;
            if (l_in >= 0 && l_in < L_in) {
                sum += input[n * C_in * L_in + c_in * L_in + l_in] *
                       weight[c_out * C_in * K + c_in * K + k];
            }
        }
    }

    if (bias) sum += bias[c_out];
    output[n * C_out * L_out + c_out * L_out + l_out] = sum;
}

bool nimcp_gpu_conv1d_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* weight,
    const nimcp_gpu_tensor_t* bias,
    nimcp_gpu_tensor_t* output,
    uint32_t kernel_size, uint32_t stride, uint32_t padding, uint32_t dilation)
{
    if (!ctx || !input || !weight || !output) return false;

    int N = input->dims[0];
    int C_in = input->dims[1];
    int L_in = input->dims[2];
    int C_out = weight->dims[0];
    int L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    dim3 block(256);
    dim3 grid((L_out + 255) / 256, C_out, N);

    kernel_conv1d_forward<<<grid, block>>>(
        (const float*)input->data, (const float*)weight->data,
        bias ? (const float*)bias->data : NULL, (float*)output->data,
        N, C_in, L_in, C_out, L_out, kernel_size, stride, padding, dilation);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Depthwise Conv2D
//=============================================================================

__global__ void kernel_depthwise_conv2d(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int kH, int kW, int stride_h, int stride_w, int pad_h, int pad_w)
{
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = hw / W_out;
    int w_out = hw % W_out;

    if (h_out >= H_out || w_out >= W_out) return;

    float sum = 0.0f;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int h_in = h_out * stride_h - pad_h + kh;
            int w_in = w_out * stride_w - pad_w + kw;
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                sum += input[n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in] *
                       weight[c * kH * kW + kh * kW + kw];
            }
        }
    }

    if (bias) sum += bias[c];
    output[n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out] = sum;
}

bool nimcp_gpu_depthwise_conv2d(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* weight,
    const nimcp_gpu_tensor_t* bias,
    nimcp_gpu_tensor_t* output,
    const nimcp_conv_params_t* params)
{
    if (!ctx || !input || !weight || !output || !params) return false;

    int N = input->dims[0];
    int C = input->dims[1];
    int H_in = input->dims[2];
    int W_in = input->dims[3];
    int kH = params->kernel_h;
    int kW = params->kernel_w;
    int H_out = (H_in + 2 * params->pad_h - kH) / params->stride_h + 1;
    int W_out = (W_in + 2 * params->pad_w - kW) / params->stride_w + 1;

    dim3 block(256);
    dim3 grid((H_out * W_out + 255) / 256, C, N);

    kernel_depthwise_conv2d<<<grid, block>>>(
        (const float*)input->data, (const float*)weight->data,
        bias ? (const float*)bias->data : NULL, (float*)output->data,
        N, C, H_in, W_in, H_out, W_out,
        kH, kW, params->stride_h, params->stride_w, params->pad_h, params->pad_w);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Pooling Operations
//=============================================================================

__global__ void kernel_maxpool2d(
    const float* input, float* output, int* indices,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int kH, int kW, int stride_h, int stride_w, int pad_h, int pad_w)
{
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = hw / W_out;
    int w_out = hw % W_out;

    if (h_out >= H_out || w_out >= W_out) return;

    float max_val = -FLT_MAX;
    int max_idx = 0;

    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int h_in = h_out * stride_h - pad_h + kh;
            int w_in = w_out * stride_w - pad_w + kw;
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                int idx = n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in;
                float val = input[idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = h_in * W_in + w_in;
                }
            }
        }
    }

    int out_idx = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out;
    output[out_idx] = max_val;
    if (indices) indices[out_idx] = max_idx;
}

bool nimcp_gpu_maxpool2d(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output,
    nimcp_gpu_tensor_t* indices,
    const nimcp_pool_params_t* params)
{
    if (!ctx || !input || !output || !params) return false;

    int N = input->dims[0];
    int C = input->dims[1];
    int H_in = input->dims[2];
    int W_in = input->dims[3];
    int H_out = (H_in + 2 * params->pad_h - params->kernel_h) / params->stride_h + 1;
    int W_out = (W_in + 2 * params->pad_w - params->kernel_w) / params->stride_w + 1;

    dim3 block(256);
    dim3 grid((H_out * W_out + 255) / 256, C, N);

    kernel_maxpool2d<<<grid, block>>>(
        (const float*)input->data, (float*)output->data,
        indices ? (int*)indices->data : NULL,
        N, C, H_in, W_in, H_out, W_out,
        params->kernel_h, params->kernel_w, params->stride_h, params->stride_w,
        params->pad_h, params->pad_w);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

__global__ void kernel_avgpool2d(
    const float* input, float* output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int kH, int kW, int stride_h, int stride_w, int pad_h, int pad_w)
{
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = hw / W_out;
    int w_out = hw % W_out;

    if (h_out >= H_out || w_out >= W_out) return;

    float sum = 0.0f;
    int count = 0;

    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int h_in = h_out * stride_h - pad_h + kh;
            int w_in = w_out * stride_w - pad_w + kw;
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                sum += input[n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in];
                count++;
            }
        }
    }

    output[n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out] =
        count > 0 ? sum / count : 0.0f;
}

bool nimcp_gpu_avgpool2d(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output,
    const nimcp_pool_params_t* params)
{
    if (!ctx || !input || !output || !params) return false;

    int N = input->dims[0];
    int C = input->dims[1];
    int H_in = input->dims[2];
    int W_in = input->dims[3];
    int H_out = (H_in + 2 * params->pad_h - params->kernel_h) / params->stride_h + 1;
    int W_out = (W_in + 2 * params->pad_w - params->kernel_w) / params->stride_w + 1;

    dim3 block(256);
    dim3 grid((H_out * W_out + 255) / 256, C, N);

    kernel_avgpool2d<<<grid, block>>>(
        (const float*)input->data, (float*)output->data,
        N, C, H_in, W_in, H_out, W_out,
        params->kernel_h, params->kernel_w, params->stride_h, params->stride_w,
        params->pad_h, params->pad_w);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

__global__ void kernel_global_avgpool(
    const float* input, float* output, int N, int C, int HW)
{
    int n = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float sum = 0.0f;
    for (int i = 0; i < HW; i++) {
        sum += input[n * C * HW + c * HW + i];
    }
    output[n * C + c] = sum / HW;
}

bool nimcp_gpu_global_avgpool(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output)
{
    if (!ctx || !input || !output) return false;

    int N = input->dims[0];
    int C = input->dims[1];
    int HW = input->dims[2] * input->dims[3];

    dim3 block(256);
    dim3 grid((C + 255) / 256, N);

    kernel_global_avgpool<<<grid, block>>>(
        (const float*)input->data, (float*)output->data, N, C, HW);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_adaptive_avgpool2d(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output,
    uint32_t output_h, uint32_t output_w)
{
    // Compute adaptive pooling params
    nimcp_pool_params_t params;
    params.stride_h = input->dims[2] / output_h;
    params.stride_w = input->dims[3] / output_w;
    params.kernel_h = input->dims[2] - (output_h - 1) * params.stride_h;
    params.kernel_w = input->dims[3] - (output_w - 1) * params.stride_w;
    params.pad_h = 0;
    params.pad_w = 0;

    return nimcp_gpu_avgpool2d(ctx, input, output, &params);
}

//=============================================================================
// Batch Normalization
//=============================================================================

__global__ void kernel_batchnorm2d_forward(
    const float* input, const float* gamma, const float* beta,
    float* output, float* running_mean, float* running_var,
    int N, int C, int HW, float momentum, float eps, bool training)
{
    int c = blockIdx.x;
    if (c >= C) return;

    // Compute mean and variance if training
    float mean = 0.0f, var = 0.0f;
    int total = N * HW;

    if (training) {
        // Compute mean
        for (int n = 0; n < N; n++) {
            for (int hw = threadIdx.x; hw < HW; hw += blockDim.x) {
                mean += input[n * C * HW + c * HW + hw];
            }
        }
        __syncthreads();
        // Reduce mean (simplified)
        mean /= total;

        // Compute variance
        for (int n = 0; n < N; n++) {
            for (int hw = threadIdx.x; hw < HW; hw += blockDim.x) {
                float diff = input[n * C * HW + c * HW + hw] - mean;
                var += diff * diff;
            }
        }
        __syncthreads();
        var /= total;

        // Update running stats
        if (threadIdx.x == 0) {
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;
        }
    } else {
        mean = running_mean[c];
        var = running_var[c];
    }

    float inv_std = rsqrtf(var + eps);
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;

    // Normalize
    for (int n = 0; n < N; n++) {
        for (int hw = threadIdx.x; hw < HW; hw += blockDim.x) {
            int idx = n * C * HW + c * HW + hw;
            output[idx] = g * (input[idx] - mean) * inv_std + b;
        }
    }
}

bool nimcp_gpu_batchnorm2d_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* gamma,
    const nimcp_gpu_tensor_t* beta,
    nimcp_gpu_tensor_t* output,
    nimcp_gpu_tensor_t* running_mean,
    nimcp_gpu_tensor_t* running_var,
    float momentum, float eps, bool training)
{
    if (!ctx || !input || !output) return false;

    int N = input->dims[0];
    int C = input->dims[1];
    int HW = input->dims[2] * input->dims[3];

    kernel_batchnorm2d_forward<<<C, 256>>>(
        (const float*)input->data,
        gamma ? (const float*)gamma->data : NULL,
        beta ? (const float*)beta->data : NULL,
        (float*)output->data,
        running_mean ? (float*)running_mean->data : NULL,
        running_var ? (float*)running_var->data : NULL,
        N, C, HW, momentum, eps, training);

    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_layernorm_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* gamma,
    const nimcp_gpu_tensor_t* beta,
    nimcp_gpu_tensor_t* output,
    float eps)
{
    // TODO: Implement layer norm
    LOG_WARN("Layer norm forward not yet optimized");
    return false;
}

bool nimcp_gpu_instancenorm_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* gamma,
    const nimcp_gpu_tensor_t* beta,
    nimcp_gpu_tensor_t* output,
    float eps)
{
    // TODO: Implement instance norm
    LOG_WARN("Instance norm forward not yet implemented");
    return false;
}

#else // !NIMCP_ENABLE_CUDA

#include "gpu/cnn/nimcp_cnn_gpu.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "CNN_GPU"

bool nimcp_gpu_conv2d_forward(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* weight, const nimcp_gpu_tensor_t* bias,
    nimcp_gpu_tensor_t* output, const nimcp_conv_params_t* params)
{
    LOG_WARN("CUDA not available - CNN requires GPU");
    return false;
}

bool nimcp_gpu_conv2d_backward(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* weight, const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input, nimcp_gpu_tensor_t* grad_weight,
    nimcp_gpu_tensor_t* grad_bias, const nimcp_conv_params_t* params)
{
    return false;
}

bool nimcp_gpu_conv1d_forward(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* weight, const nimcp_gpu_tensor_t* bias,
    nimcp_gpu_tensor_t* output, uint32_t ks, uint32_t s, uint32_t p, uint32_t d)
{
    return false;
}

bool nimcp_gpu_depthwise_conv2d(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* weight, const nimcp_gpu_tensor_t* bias,
    nimcp_gpu_tensor_t* output, const nimcp_conv_params_t* params)
{
    return false;
}

bool nimcp_gpu_maxpool2d(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output, nimcp_gpu_tensor_t* indices, const nimcp_pool_params_t* params)
{
    return false;
}

bool nimcp_gpu_avgpool2d(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output, const nimcp_pool_params_t* params)
{
    return false;
}

bool nimcp_gpu_global_avgpool(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output)
{
    return false;
}

bool nimcp_gpu_adaptive_avgpool2d(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output, uint32_t h, uint32_t w)
{
    return false;
}

bool nimcp_gpu_batchnorm2d_forward(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* gamma, const nimcp_gpu_tensor_t* beta,
    nimcp_gpu_tensor_t* output, nimcp_gpu_tensor_t* rm, nimcp_gpu_tensor_t* rv,
    float momentum, float eps, bool training)
{
    return false;
}

bool nimcp_gpu_layernorm_forward(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* gamma, const nimcp_gpu_tensor_t* beta,
    nimcp_gpu_tensor_t* output, float eps)
{
    return false;
}

bool nimcp_gpu_instancenorm_forward(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* gamma, const nimcp_gpu_tensor_t* beta,
    nimcp_gpu_tensor_t* output, float eps)
{
    return false;
}

#endif // NIMCP_ENABLE_CUDA
