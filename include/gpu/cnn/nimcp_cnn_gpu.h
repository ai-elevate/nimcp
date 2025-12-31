//=============================================================================
// nimcp_cnn_gpu.h - GPU CNN Kernels (Convolution, Pooling, Normalization)
//=============================================================================
/**
 * @file nimcp_cnn_gpu.h
 * @brief GPU-accelerated CNN operations using CUDA
 *
 * WHAT: CUDA kernels for convolutional neural networks
 * WHY:  Enables GPU acceleration for visual, audio, and speech processing
 * HOW:  Custom kernels for conv, pooling, normalization layers
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifndef NIMCP_CNN_GPU_H
#define NIMCP_CNN_GPU_H

#include "gpu/context/nimcp_gpu_context.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "common/nimcp_export.h"
#include "gpu/tensor/nimcp_tensor_gpu.h"
#include <stdbool.h>
#include <stdint.h>

//=============================================================================
// Convolution Types
//=============================================================================

typedef enum {
    NIMCP_PAD_VALID = 0,    /**< No padding */
    NIMCP_PAD_SAME = 1,     /**< Pad to maintain size */
    NIMCP_PAD_CAUSAL = 2    /**< Causal padding (for temporal) */
} nimcp_padding_mode_t;

typedef struct {
    uint32_t kernel_h, kernel_w;  /**< Kernel size */
    uint32_t stride_h, stride_w;  /**< Stride */
    uint32_t pad_h, pad_w;        /**< Padding */
    uint32_t dilation_h, dilation_w;  /**< Dilation */
    uint32_t groups;              /**< Groups for grouped conv */
} nimcp_conv_params_t;

typedef struct {
    uint32_t kernel_h, kernel_w;  /**< Pool size */
    uint32_t stride_h, stride_w;  /**< Stride */
    uint32_t pad_h, pad_w;        /**< Padding */
} nimcp_pool_params_t;

//=============================================================================
// 2D Convolution (Visual Processing)
//=============================================================================

NIMCP_EXPORT bool nimcp_gpu_conv2d_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,   // (N, C_in, H, W)
    const nimcp_gpu_tensor_t* weight,  // (C_out, C_in/groups, kH, kW)
    const nimcp_gpu_tensor_t* bias,    // (C_out) or NULL
    nimcp_gpu_tensor_t* output,        // (N, C_out, H_out, W_out)
    const nimcp_conv_params_t* params
);

NIMCP_EXPORT bool nimcp_gpu_conv2d_backward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* weight,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input,
    nimcp_gpu_tensor_t* grad_weight,
    nimcp_gpu_tensor_t* grad_bias,
    const nimcp_conv_params_t* params
);

//=============================================================================
// 1D Convolution (Audio/Speech Processing)
//=============================================================================

NIMCP_EXPORT bool nimcp_gpu_conv1d_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,   // (N, C_in, L)
    const nimcp_gpu_tensor_t* weight,  // (C_out, C_in, K)
    const nimcp_gpu_tensor_t* bias,
    nimcp_gpu_tensor_t* output,
    uint32_t kernel_size, uint32_t stride, uint32_t padding, uint32_t dilation
);

//=============================================================================
// Depthwise Separable Convolution
//=============================================================================

NIMCP_EXPORT bool nimcp_gpu_depthwise_conv2d(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* weight,  // (C, 1, kH, kW)
    const nimcp_gpu_tensor_t* bias,
    nimcp_gpu_tensor_t* output,
    const nimcp_conv_params_t* params
);

//=============================================================================
// Pooling Operations
//=============================================================================

NIMCP_EXPORT bool nimcp_gpu_maxpool2d(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output,
    nimcp_gpu_tensor_t* indices,  // For backward pass
    const nimcp_pool_params_t* params
);

NIMCP_EXPORT bool nimcp_gpu_avgpool2d(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output,
    const nimcp_pool_params_t* params
);

NIMCP_EXPORT bool nimcp_gpu_global_avgpool(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output
);

NIMCP_EXPORT bool nimcp_gpu_adaptive_avgpool2d(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* output,
    uint32_t output_h, uint32_t output_w
);

//=============================================================================
// Normalization
//=============================================================================

NIMCP_EXPORT bool nimcp_gpu_batchnorm2d_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* gamma,
    const nimcp_gpu_tensor_t* beta,
    nimcp_gpu_tensor_t* output,
    nimcp_gpu_tensor_t* running_mean,
    nimcp_gpu_tensor_t* running_var,
    float momentum, float eps, bool training
);

NIMCP_EXPORT bool nimcp_gpu_layernorm_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* gamma,
    const nimcp_gpu_tensor_t* beta,
    nimcp_gpu_tensor_t* output,
    float eps
);

NIMCP_EXPORT bool nimcp_gpu_instancenorm_forward(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* gamma,
    const nimcp_gpu_tensor_t* beta,
    nimcp_gpu_tensor_t* output,
    float eps
);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_CNN_GPU_H
