/**
 * @file nimcp_fep_learning_gpu.h
 * @brief GPU-accelerated FEP Learning API
 * @version 1.0.0
 * @date 2026-01-01
 *
 * WHAT: GPU acceleration for FEP learning operations
 * WHY:  Matrix operations benefit greatly from GPU parallelism
 * HOW:  CUDA kernels for matvec, gradient, and batch updates
 */

#ifndef NIMCP_FEP_LEARNING_GPU_H
#define NIMCP_FEP_LEARNING_GPU_H

#include <stdint.h>
#include <stdbool.h>
#include "gpu/context/nimcp_gpu_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque GPU context for FEP learning
 */
typedef struct fep_learning_gpu_context_s fep_learning_gpu_context_t;

/**
 * @brief Create GPU context for FEP learning
 *
 * @param gpu_ctx Main GPU context
 * @param state_dim State dimension
 * @param obs_dim Observation dimension (0 if not using likelihood)
 * @param max_batch_size Maximum batch size for learning
 * @return FEP learning GPU context, NULL on failure
 */
fep_learning_gpu_context_t* fep_learning_gpu_create(
    nimcp_gpu_context_t* gpu_ctx,
    uint32_t state_dim,
    uint32_t obs_dim,
    uint32_t max_batch_size);

/**
 * @brief Destroy FEP learning GPU context
 *
 * @param ctx Context to destroy (NULL safe)
 */
void fep_learning_gpu_destroy(fep_learning_gpu_context_t* ctx);

/**
 * @brief Upload transition matrix to GPU
 *
 * @param ctx FEP learning GPU context
 * @param A Transition matrix (dim x dim, row-major)
 * @param dim State dimension
 * @return 0 on success, negative on error
 */
int fep_learning_gpu_upload_transition(
    fep_learning_gpu_context_t* ctx,
    const float* A,
    uint32_t dim);

/**
 * @brief Download transition matrix from GPU
 *
 * @param ctx FEP learning GPU context
 * @param A Output buffer for matrix
 * @param dim State dimension
 * @return 0 on success, negative on error
 */
int fep_learning_gpu_download_transition(
    fep_learning_gpu_context_t* ctx,
    float* A,
    uint32_t dim);

/**
 * @brief GPU-accelerated transition learning step
 *
 * Performs one step of gradient descent on the transition matrix A:
 *   prediction = A * state_t
 *   error = state_t1 - prediction
 *   grad = -error ⊗ state_t^T + lambda * A
 *   A = A - lr * grad  (with optional momentum)
 *
 * @param ctx FEP learning GPU context
 * @param state_t Current state (host memory)
 * @param state_t1 Next state (host memory)
 * @param lr Learning rate
 * @param lambda L2 regularization coefficient
 * @param momentum Momentum coefficient (0 to disable)
 * @param loss_out Output loss value (can be NULL)
 * @return 0 on success, negative on error
 */
int fep_learn_transition_gpu(
    fep_learning_gpu_context_t* ctx,
    const float* state_t,
    const float* state_t1,
    float lr,
    float lambda,
    float momentum,
    float* loss_out);

/**
 * @brief GPU-accelerated batch transition learning
 *
 * Accumulates gradients over multiple transitions, then applies averaged update.
 *
 * @param ctx FEP learning GPU context
 * @param states Flattened state sequence [(n_transitions+1) * dim]
 * @param n_transitions Number of transitions
 * @param lr Learning rate
 * @param lambda L2 regularization
 * @param momentum Momentum coefficient
 * @param avg_loss_out Average loss output (can be NULL)
 * @return 0 on success, negative on error
 */
int fep_learn_transition_batch_gpu(
    fep_learning_gpu_context_t* ctx,
    const float* states,
    uint32_t n_transitions,
    float lr,
    float lambda,
    float momentum,
    float* avg_loss_out);

/**
 * @brief Upload likelihood matrix to GPU
 *
 * @param ctx FEP learning GPU context
 * @param B Likelihood matrix (obs_dim x state_dim)
 * @param obs_dim Observation dimension
 * @param state_dim State dimension
 * @return 0 on success, negative on error
 */
int fep_learning_gpu_upload_likelihood(
    fep_learning_gpu_context_t* ctx,
    const float* B,
    uint32_t obs_dim,
    uint32_t state_dim);

/**
 * @brief Download likelihood matrix from GPU
 *
 * @param ctx FEP learning GPU context
 * @param B Output buffer for matrix
 * @param obs_dim Observation dimension
 * @param state_dim State dimension
 * @return 0 on success, negative on error
 */
int fep_learning_gpu_download_likelihood(
    fep_learning_gpu_context_t* ctx,
    float* B,
    uint32_t obs_dim,
    uint32_t state_dim);

/**
 * @brief GPU-accelerated likelihood learning step
 *
 * @param ctx FEP learning GPU context
 * @param observation Current observation (host memory)
 * @param state Current state (host memory)
 * @param lr Learning rate
 * @param lambda L2 regularization coefficient
 * @param momentum Momentum coefficient (0 to disable)
 * @param loss_out Output loss value (can be NULL)
 * @return 0 on success, negative on error
 */
int fep_learn_likelihood_gpu(
    fep_learning_gpu_context_t* ctx,
    const float* observation,
    const float* state,
    float lr,
    float lambda,
    float momentum,
    float* loss_out);

/**
 * @brief Check if GPU should be used for FEP learning
 *
 * Returns true if the matrix size is large enough to benefit from GPU.
 * For small matrices (< 64 elements), CPU may be faster due to
 * kernel launch overhead.
 *
 * @param dim State dimension
 * @return true if GPU acceleration is beneficial
 */
bool fep_learning_should_use_gpu(uint32_t dim);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_FEP_LEARNING_GPU_H */
