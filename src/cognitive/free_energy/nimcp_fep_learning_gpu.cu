/**
 * @file nimcp_fep_learning_gpu.cu
 * @brief GPU-accelerated FEP Learning Kernels
 * @version 1.0.0
 * @date 2026-01-01
 *
 * WHAT: CUDA kernels for accelerating FEP matrix learning operations
 * WHY:  Matrix operations in FEP learning benefit greatly from GPU parallelism
 * HOW:  GPU kernels for matvec, gradient computation, and batch updates
 *
 * HOT PATHS ACCELERATED:
 * 1. Matrix-vector multiply: prediction = A * state (O(n^2) -> O(n) parallel)
 * 2. Outer product gradient: grad = error ⊗ state^T (O(n^2) -> O(n) parallel)
 * 3. Matrix update with regularization: A -= lr * (grad + λA) (O(n^2) -> O(n) parallel)
 * 4. Batch gradient accumulation (fused reduce)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// Include GPU context for stream management
// Note: Include OUTSIDE extern "C" as GPU context header may include C++ CUDA types
#include "gpu/context/nimcp_gpu_context.h"

// Forward declaration for cleanup function used in error paths
extern "C" void fep_learning_gpu_destroy(struct fep_learning_gpu_context_s* ctx);

//=============================================================================
// Constants
//=============================================================================

#define FEP_GPU_BLOCK_SIZE 256
#define FEP_GPU_MIN_SIZE_FOR_GPU 64  // Minimum elements before using GPU

//=============================================================================
// Kernel: Matrix-Vector Multiply (prediction = A * x)
//=============================================================================

/**
 * @brief GPU kernel for matrix-vector multiplication
 *
 * Each thread computes one element of the output vector
 * output[i] = sum_j(A[i,j] * x[j])
 *
 * @param A Matrix (dim x dim), row-major
 * @param x Input vector (dim)
 * @param output Output vector (dim)
 * @param dim Dimension
 */
__global__ void kernel_fep_matvec(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ output,
    uint32_t dim)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) return;

    float sum = 0.0f;
    const float* row_ptr = A + row * dim;

    #pragma unroll 4
    for (uint32_t j = 0; j < dim; j++) {
        sum += row_ptr[j] * x[j];
    }

    output[row] = sum;
}

//=============================================================================
// Kernel: Outer Product Gradient (grad = -error ⊗ x^T + λA)
//=============================================================================

/**
 * @brief GPU kernel for computing gradient with regularization
 *
 * grad[i,j] = -error[i] * x[j] + lambda * A[i,j]
 *
 * @param error Prediction error vector (out_dim)
 * @param x Input state vector (in_dim)
 * @param A Current matrix (out_dim x in_dim)
 * @param grad Output gradient (out_dim x in_dim)
 * @param lambda L2 regularization coefficient
 * @param out_dim Output dimension (rows)
 * @param in_dim Input dimension (cols)
 */
__global__ void kernel_fep_gradient(
    const float* __restrict__ error,
    const float* __restrict__ x,
    const float* __restrict__ A,
    float* __restrict__ grad,
    float lambda,
    uint32_t out_dim,
    uint32_t in_dim)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = out_dim * in_dim;
    if (idx >= total) return;

    uint32_t i = idx / in_dim;  // row
    uint32_t j = idx % in_dim;  // col

    float g = -error[i] * x[j] + lambda * A[idx];
    grad[idx] = g;
}

//=============================================================================
// Kernel: Matrix Update with Momentum (A = A - lr * (momentum * v + grad))
//=============================================================================

/**
 * @brief GPU kernel for matrix update with optional momentum
 *
 * If momentum enabled:
 *   v[i] = beta * v[i] + grad[i]
 *   A[i] = A[i] - lr * v[i]
 * Else:
 *   A[i] = A[i] - lr * grad[i]
 *
 * @param A Matrix to update (in-place)
 * @param grad Gradient
 * @param velocity Momentum velocity buffer (can be NULL)
 * @param lr Learning rate
 * @param beta Momentum coefficient
 * @param size Total elements
 */
__global__ void kernel_fep_update(
    float* __restrict__ A,
    const float* __restrict__ grad,
    float* __restrict__ velocity,
    float lr,
    float beta,
    uint32_t size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = grad[idx];

    if (velocity != NULL) {
        float v = beta * velocity[idx] + g;
        velocity[idx] = v;
        g = v;
    }

    A[idx] = A[idx] - lr * g;
}

//=============================================================================
// Kernel: Compute MSE Loss
//=============================================================================

/**
 * @brief GPU kernel to compute MSE loss: 0.5 * ||target - pred||^2
 *
 * Uses parallel reduction for efficient summation
 *
 * @param target Target values
 * @param pred Predicted values
 * @param partial_sums Output partial sums (one per block)
 * @param n Number of elements
 */
__global__ void kernel_fep_mse_loss(
    const float* __restrict__ target,
    const float* __restrict__ pred,
    float* __restrict__ partial_sums,
    uint32_t n)
{
    extern __shared__ float sdata[];

    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute squared difference
    float val = 0.0f;
    if (idx < n) {
        float diff = target[idx] - pred[idx];
        val = diff * diff;
    }
    sdata[tid] = val;
    __syncthreads();

    // Parallel reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block sum
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

//=============================================================================
// Kernel: Batch Gradient Accumulation
//=============================================================================

/**
 * @brief GPU kernel to accumulate gradients from multiple samples
 *
 * Adds gradient contribution to batch accumulator
 *
 * @param batch_grad Accumulated gradient (atomically updated)
 * @param sample_grad Single sample gradient
 * @param size Matrix size
 */
__global__ void kernel_fep_accumulate_gradient(
    float* __restrict__ batch_grad,
    const float* __restrict__ sample_grad,
    uint32_t size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    atomicAdd(&batch_grad[idx], sample_grad[idx]);
}

/**
 * @brief GPU kernel to scale accumulated gradient by 1/N
 *
 * @param grad Gradient to scale (in-place)
 * @param scale Scale factor (1.0 / N)
 * @param size Matrix size
 */
__global__ void kernel_fep_scale_gradient(
    float* __restrict__ grad,
    float scale,
    uint32_t size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    grad[idx] *= scale;
}

/**
 * @brief GPU kernel for vector subtraction: error = target - prediction
 *
 * @param target Target vector
 * @param prediction Prediction vector
 * @param error Output error vector
 * @param n Number of elements
 */
__global__ void kernel_fep_compute_error(
    const float* __restrict__ target,
    const float* __restrict__ prediction,
    float* __restrict__ error,
    uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    error[idx] = target[idx] - prediction[idx];
}

//=============================================================================
// Host API - GPU Context Structure for FEP Learning
//=============================================================================

/**
 * @brief GPU context for FEP learning operations
 */
typedef struct fep_learning_gpu_context_s {
    nimcp_gpu_context_t* gpu_ctx;   // Main GPU context

    // Device buffers for transition learner
    float* d_A;                      // Transition matrix on GPU
    float* d_A_grad;                 // Gradient buffer
    float* d_A_velocity;             // Momentum velocity
    float* d_state_t;                // Current state
    float* d_state_t1;               // Next state
    float* d_prediction;             // Prediction buffer
    float* d_error;                  // Error buffer

    // Device buffers for likelihood learner
    float* d_B;                      // Likelihood matrix on GPU
    float* d_B_grad;                 // Gradient buffer
    float* d_B_velocity;             // Momentum velocity
    float* d_observation;            // Observation vector

    // Batch learning buffers
    float* d_batch_grad;             // Batch gradient accumulator
    float* d_loss_partial;           // Partial sums for loss reduction

    // Dimensions
    uint32_t state_dim;
    uint32_t obs_dim;
    uint32_t max_batch_size;

    // Status
    bool initialized;
    bool matrices_on_gpu;
} fep_learning_gpu_context_t;

//=============================================================================
// Host API Functions
//=============================================================================

extern "C" {

/**
 * @brief Create GPU context for FEP learning
 *
 * @param gpu_ctx Main GPU context
 * @param state_dim State dimension
 * @param obs_dim Observation dimension (0 if not using likelihood)
 * @param max_batch_size Maximum batch size for learning
 * @return FEP learning GPU context
 */
fep_learning_gpu_context_t* fep_learning_gpu_create(
    nimcp_gpu_context_t* gpu_ctx,
    uint32_t state_dim,
    uint32_t obs_dim,
    uint32_t max_batch_size)
{
    if (!gpu_ctx || !nimcp_gpu_context_is_valid(gpu_ctx)) {
        return NULL;
    }

    if (state_dim == 0) {
        return NULL;
    }

    fep_learning_gpu_context_t* ctx =
        (fep_learning_gpu_context_t*)malloc(sizeof(fep_learning_gpu_context_t));
    if (!ctx) return NULL;

    memset(ctx, 0, sizeof(*ctx));
    ctx->gpu_ctx = gpu_ctx;
    ctx->state_dim = state_dim;
    ctx->obs_dim = obs_dim;
    ctx->max_batch_size = max_batch_size > 0 ? max_batch_size : 32;

    // Allocate transition buffers
    size_t matrix_size = state_dim * state_dim * sizeof(float);
    size_t vec_size = state_dim * sizeof(float);

    ctx->d_A = (float*)nimcp_gpu_malloc(gpu_ctx, matrix_size);
    ctx->d_A_grad = (float*)nimcp_gpu_malloc(gpu_ctx, matrix_size);
    ctx->d_A_velocity = (float*)nimcp_gpu_malloc(gpu_ctx, matrix_size);
    ctx->d_state_t = (float*)nimcp_gpu_malloc(gpu_ctx, vec_size);
    ctx->d_state_t1 = (float*)nimcp_gpu_malloc(gpu_ctx, vec_size);
    ctx->d_prediction = (float*)nimcp_gpu_malloc(gpu_ctx, vec_size);
    ctx->d_error = (float*)nimcp_gpu_malloc(gpu_ctx, vec_size);
    ctx->d_batch_grad = (float*)nimcp_gpu_malloc(gpu_ctx, matrix_size);

    if (!ctx->d_A || !ctx->d_A_grad || !ctx->d_state_t || !ctx->d_state_t1 ||
        !ctx->d_prediction || !ctx->d_error || !ctx->d_batch_grad) {
        fep_learning_gpu_destroy(ctx);
        return NULL;
    }

    // Zero velocity buffer
    nimcp_gpu_memset(gpu_ctx, ctx->d_A_velocity, 0, matrix_size);

    // Allocate likelihood buffers if needed
    if (obs_dim > 0) {
        size_t B_size = obs_dim * state_dim * sizeof(float);
        size_t obs_vec_size = obs_dim * sizeof(float);

        ctx->d_B = (float*)nimcp_gpu_malloc(gpu_ctx, B_size);
        ctx->d_B_grad = (float*)nimcp_gpu_malloc(gpu_ctx, B_size);
        ctx->d_B_velocity = (float*)nimcp_gpu_malloc(gpu_ctx, B_size);
        ctx->d_observation = (float*)nimcp_gpu_malloc(gpu_ctx, obs_vec_size);

        if (!ctx->d_B || !ctx->d_B_grad || !ctx->d_observation) {
            fep_learning_gpu_destroy(ctx);
            return NULL;
        }

        nimcp_gpu_memset(gpu_ctx, ctx->d_B_velocity, 0, B_size);
    }

    // Allocate loss reduction buffer
    uint32_t max_blocks = (state_dim + FEP_GPU_BLOCK_SIZE - 1) / FEP_GPU_BLOCK_SIZE;
    ctx->d_loss_partial = (float*)nimcp_gpu_malloc(gpu_ctx, max_blocks * sizeof(float));

    ctx->initialized = true;
    return ctx;
}

/**
 * @brief Destroy FEP learning GPU context
 */
void fep_learning_gpu_destroy(fep_learning_gpu_context_t* ctx)
{
    if (!ctx) return;

    if (ctx->gpu_ctx) {
        if (ctx->d_A) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_A);
        if (ctx->d_A_grad) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_A_grad);
        if (ctx->d_A_velocity) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_A_velocity);
        if (ctx->d_state_t) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_state_t);
        if (ctx->d_state_t1) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_state_t1);
        if (ctx->d_prediction) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_prediction);
        if (ctx->d_error) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_error);
        if (ctx->d_batch_grad) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_batch_grad);
        if (ctx->d_loss_partial) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_loss_partial);

        if (ctx->d_B) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_B);
        if (ctx->d_B_grad) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_B_grad);
        if (ctx->d_B_velocity) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_B_velocity);
        if (ctx->d_observation) nimcp_gpu_free(ctx->gpu_ctx, ctx->d_observation);
    }

    free(ctx);
}

/**
 * @brief Copy transition matrix to GPU
 */
int fep_learning_gpu_upload_transition(
    fep_learning_gpu_context_t* ctx,
    const float* A,
    uint32_t dim)
{
    if (!ctx || !ctx->initialized || !A) return -1;
    if (dim != ctx->state_dim) return -1;

    size_t size = dim * dim * sizeof(float);
    int err = nimcp_gpu_memcpy(ctx->gpu_ctx, ctx->d_A, A, size,
                                GPU_MEMCPY_HOST_TO_DEVICE);
    if (err == 0) {
        ctx->matrices_on_gpu = true;
    }
    return err;
}

/**
 * @brief Copy transition matrix from GPU
 */
int fep_learning_gpu_download_transition(
    fep_learning_gpu_context_t* ctx,
    float* A,
    uint32_t dim)
{
    if (!ctx || !ctx->initialized || !A) return -1;
    if (dim != ctx->state_dim) return -1;

    size_t size = dim * dim * sizeof(float);
    return nimcp_gpu_memcpy(ctx->gpu_ctx, A, ctx->d_A, size,
                            GPU_MEMCPY_DEVICE_TO_HOST);
}

/**
 * @brief GPU-accelerated transition learning step
 *
 * @param ctx FEP learning GPU context
 * @param state_t Current state (host)
 * @param state_t1 Next state (host)
 * @param lr Learning rate
 * @param lambda L2 regularization
 * @param momentum Momentum coefficient (0 to disable)
 * @param loss_out Output loss value (can be NULL)
 * @return 0 on success
 */
int fep_learn_transition_gpu(
    fep_learning_gpu_context_t* ctx,
    const float* state_t,
    const float* state_t1,
    float lr,
    float lambda,
    float momentum,
    float* loss_out)
{
    if (!ctx || !ctx->initialized || !state_t || !state_t1) return -1;
    if (!ctx->matrices_on_gpu) return -1;

    uint32_t dim = ctx->state_dim;
    cudaStream_t stream = (cudaStream_t)nimcp_gpu_get_compute_stream(ctx->gpu_ctx);

    // Copy states to GPU
    size_t vec_size = dim * sizeof(float);
    cudaMemcpyAsync(ctx->d_state_t, state_t, vec_size,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ctx->d_state_t1, state_t1, vec_size,
                    cudaMemcpyHostToDevice, stream);

    // Compute prediction: A * state_t
    uint32_t block_size = FEP_GPU_BLOCK_SIZE;
    uint32_t grid_size = (dim + block_size - 1) / block_size;

    kernel_fep_matvec<<<grid_size, block_size, 0, stream>>>(
        ctx->d_A, ctx->d_state_t, ctx->d_prediction, dim);

    // Compute error: error = state_t1 - prediction
    kernel_fep_compute_error<<<grid_size, block_size, 0, stream>>>(
        ctx->d_state_t1, ctx->d_prediction, ctx->d_error, dim);

    // Compute gradient: grad = -error ⊗ state^T + λA
    uint32_t total_elements = dim * dim;
    uint32_t grad_grid = (total_elements + block_size - 1) / block_size;

    kernel_fep_gradient<<<grad_grid, block_size, 0, stream>>>(
        ctx->d_error,     // Proper error = state_t1 - prediction
        ctx->d_state_t,
        ctx->d_A,
        ctx->d_A_grad,
        lambda,
        dim, dim);

    // Apply update: A = A - lr * grad
    float* velocity_ptr = (momentum > 0.0f) ? ctx->d_A_velocity : NULL;

    kernel_fep_update<<<grad_grid, block_size, 0, stream>>>(
        ctx->d_A,
        ctx->d_A_grad,
        velocity_ptr,
        lr,
        momentum,
        total_elements);

    // Compute loss if requested
    if (loss_out) {
        uint32_t loss_grid = (dim + block_size - 1) / block_size;
        size_t shared_size = block_size * sizeof(float);

        kernel_fep_mse_loss<<<loss_grid, block_size, shared_size, stream>>>(
            ctx->d_state_t1, ctx->d_prediction, ctx->d_loss_partial, dim);

        // Copy partial sums and reduce on CPU
        float* partial = (float*)malloc(loss_grid * sizeof(float));
        cudaMemcpyAsync(partial, ctx->d_loss_partial,
                        loss_grid * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        float total_loss = 0.0f;
        for (uint32_t i = 0; i < loss_grid; i++) {
            total_loss += partial[i];
        }
        // MSE loss: sum(error^2) / dim
        *loss_out = total_loss / (float)dim;
        free(partial);
    }

    return 0;
}

/**
 * @brief GPU-accelerated batch transition learning
 *
 * @param ctx FEP learning GPU context
 * @param states Flattened state sequence (n_transitions+1 states)
 * @param n_transitions Number of transitions
 * @param lr Learning rate
 * @param lambda L2 regularization
 * @param momentum Momentum coefficient
 * @param avg_loss_out Average loss output (can be NULL)
 * @return 0 on success
 */
int fep_learn_transition_batch_gpu(
    fep_learning_gpu_context_t* ctx,
    const float* states,
    uint32_t n_transitions,
    float lr,
    float lambda,
    float momentum,
    float* avg_loss_out)
{
    if (!ctx || !ctx->initialized || !states || n_transitions == 0) return -1;
    if (!ctx->matrices_on_gpu) return -1;

    uint32_t dim = ctx->state_dim;
    size_t matrix_size = dim * dim * sizeof(float);
    cudaStream_t stream = (cudaStream_t)nimcp_gpu_get_compute_stream(ctx->gpu_ctx);

    // Zero batch gradient accumulator
    cudaMemsetAsync(ctx->d_batch_grad, 0, matrix_size, stream);

    float total_loss = 0.0f;
    uint32_t block_size = FEP_GPU_BLOCK_SIZE;

    // Process each transition
    for (uint32_t t = 0; t < n_transitions; t++) {
        const float* state_t = states + t * dim;
        const float* state_t1 = states + (t + 1) * dim;

        float step_loss = 0.0f;

        // Copy states
        size_t vec_size = dim * sizeof(float);
        cudaMemcpyAsync(ctx->d_state_t, state_t, vec_size,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ctx->d_state_t1, state_t1, vec_size,
                        cudaMemcpyHostToDevice, stream);

        // Compute prediction
        uint32_t grid_size = (dim + block_size - 1) / block_size;
        kernel_fep_matvec<<<grid_size, block_size, 0, stream>>>(
            ctx->d_A, ctx->d_state_t, ctx->d_prediction, dim);

        // Compute error: error = state_t1 - prediction
        kernel_fep_compute_error<<<grid_size, block_size, 0, stream>>>(
            ctx->d_state_t1, ctx->d_prediction, ctx->d_error, dim);

        // Compute gradient: grad = -error ⊗ state^T + λA
        uint32_t total_elements = dim * dim;
        uint32_t grad_grid = (total_elements + block_size - 1) / block_size;

        kernel_fep_gradient<<<grad_grid, block_size, 0, stream>>>(
            ctx->d_error, ctx->d_state_t, ctx->d_A, ctx->d_A_grad,
            lambda, dim, dim);

        // Accumulate gradient
        kernel_fep_accumulate_gradient<<<grad_grid, block_size, 0, stream>>>(
            ctx->d_batch_grad, ctx->d_A_grad, total_elements);

        // Compute loss for this step
        if (avg_loss_out) {
            uint32_t loss_grid = (dim + block_size - 1) / block_size;
            size_t shared_size = block_size * sizeof(float);

            kernel_fep_mse_loss<<<loss_grid, block_size, shared_size, stream>>>(
                ctx->d_state_t1, ctx->d_prediction, ctx->d_loss_partial, dim);

            float* partial = (float*)malloc(loss_grid * sizeof(float));
            cudaMemcpyAsync(partial, ctx->d_loss_partial,
                            loss_grid * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            for (uint32_t i = 0; i < loss_grid; i++) {
                step_loss += partial[i];
            }
            free(partial);
            // MSE loss per step: sum(error^2) / dim
            total_loss += step_loss / (float)dim;
        }
    }

    // Average and apply gradient
    uint32_t total_elements = dim * dim;
    uint32_t grad_grid = (total_elements + block_size - 1) / block_size;
    float scale = 1.0f / (float)n_transitions;

    kernel_fep_scale_gradient<<<grad_grid, block_size, 0, stream>>>(
        ctx->d_batch_grad, scale, total_elements);

    float* velocity_ptr = (momentum > 0.0f) ? ctx->d_A_velocity : NULL;

    kernel_fep_update<<<grad_grid, block_size, 0, stream>>>(
        ctx->d_A, ctx->d_batch_grad, velocity_ptr, lr, momentum, total_elements);

    cudaStreamSynchronize(stream);

    if (avg_loss_out) {
        *avg_loss_out = total_loss / (float)n_transitions;
    }

    return 0;
}

/**
 * @brief Check if GPU should be used for FEP learning
 *
 * @param dim State dimension
 * @return true if GPU acceleration is beneficial
 */
bool fep_learning_should_use_gpu(uint32_t dim)
{
    // For small matrices, CPU is faster due to kernel launch overhead
    return dim >= FEP_GPU_MIN_SIZE_FOR_GPU;
}

/**
 * @brief Copy likelihood matrix to GPU
 */
int fep_learning_gpu_upload_likelihood(
    fep_learning_gpu_context_t* ctx,
    const float* B,
    uint32_t obs_dim,
    uint32_t state_dim)
{
    if (!ctx || !ctx->initialized || !B) return -1;
    if (obs_dim != ctx->obs_dim || state_dim != ctx->state_dim) return -1;

    size_t size = obs_dim * state_dim * sizeof(float);
    return nimcp_gpu_memcpy(ctx->gpu_ctx, ctx->d_B, B, size,
                            GPU_MEMCPY_HOST_TO_DEVICE);
}

/**
 * @brief Copy likelihood matrix from GPU
 */
int fep_learning_gpu_download_likelihood(
    fep_learning_gpu_context_t* ctx,
    float* B,
    uint32_t obs_dim,
    uint32_t state_dim)
{
    if (!ctx || !ctx->initialized || !B) return -1;
    if (obs_dim != ctx->obs_dim || state_dim != ctx->state_dim) return -1;

    size_t size = obs_dim * state_dim * sizeof(float);
    return nimcp_gpu_memcpy(ctx->gpu_ctx, B, ctx->d_B, size,
                            GPU_MEMCPY_DEVICE_TO_HOST);
}

/**
 * @brief GPU-accelerated likelihood learning step
 */
int fep_learn_likelihood_gpu(
    fep_learning_gpu_context_t* ctx,
    const float* observation,
    const float* state,
    float lr,
    float lambda,
    float momentum,
    float* loss_out)
{
    if (!ctx || !ctx->initialized || !observation || !state) return -1;
    if (!ctx->d_B) return -1;  // Likelihood not initialized

    uint32_t obs_dim = ctx->obs_dim;
    uint32_t state_dim = ctx->state_dim;
    cudaStream_t stream = (cudaStream_t)nimcp_gpu_get_compute_stream(ctx->gpu_ctx);

    // Copy inputs to GPU
    cudaMemcpyAsync(ctx->d_observation, observation,
                    obs_dim * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ctx->d_state_t, state,
                    state_dim * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Compute prediction: B * state (obs_dim x state_dim) * (state_dim) = (obs_dim)
    uint32_t block_size = FEP_GPU_BLOCK_SIZE;
    uint32_t grid_size = (obs_dim + block_size - 1) / block_size;

    // Reuse prediction buffer (ensure it's large enough for obs_dim)
    // Note: d_prediction was allocated for state_dim, may need separate buffer
    // For now assume obs_dim <= state_dim or allocate larger buffer

    kernel_fep_matvec<<<grid_size, block_size, 0, stream>>>(
        ctx->d_B, ctx->d_state_t, ctx->d_error, obs_dim);  // Using d_error as temp

    // Compute gradient: -error ⊗ state^T + λB
    uint32_t total_elements = obs_dim * state_dim;
    uint32_t grad_grid = (total_elements + block_size - 1) / block_size;

    kernel_fep_gradient<<<grad_grid, block_size, 0, stream>>>(
        ctx->d_observation, ctx->d_state_t, ctx->d_B, ctx->d_B_grad,
        lambda, obs_dim, state_dim);

    // Apply update
    float* velocity_ptr = (momentum > 0.0f) ? ctx->d_B_velocity : NULL;

    kernel_fep_update<<<grad_grid, block_size, 0, stream>>>(
        ctx->d_B, ctx->d_B_grad, velocity_ptr, lr, momentum, total_elements);

    cudaStreamSynchronize(stream);

    return 0;
}

} // extern "C"
