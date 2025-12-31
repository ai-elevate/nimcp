//=============================================================================
// nimcp_lnn_gpu.h - GPU LNN Kernels (ODE Solvers for Liquid Neural Networks)
//=============================================================================
/**
 * @file nimcp_lnn_gpu.h
 * @brief GPU-accelerated LNN operations using CUDA
 *
 * WHAT: CUDA kernels for Liquid Neural Network (LNN) operations
 * WHY:  Enables GPU acceleration for continuous-time neural dynamics
 * HOW:  Custom kernels for ODE integration (Euler, Heun, RK4, DOPRI5)
 *
 * INTEGRATION:
 * - Uses existing lnn_ode_method_t enum from nimcp_lnn_types.h
 * - Extends CPU ODE solvers from nimcp_lnn_ode.h to GPU
 * - Integrates with GPU tensor library for efficient computation
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifndef NIMCP_LNN_GPU_H
#define NIMCP_LNN_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include "common/nimcp_export.h"
#include "gpu/context/nimcp_gpu_context.h"
#include "gpu/tensor/nimcp_tensor_gpu.h"
#include "lnn/nimcp_lnn_types.h"
#include <stdbool.h>
#include <stdint.h>

//=============================================================================
// LNN ODE Configuration
//=============================================================================

/**
 * @brief GPU ODE solver configuration
 */
typedef struct {
    lnn_ode_method_t method;       /**< ODE solver method */
    float dt;                       /**< Time step */
    float dt_min;                   /**< Minimum adaptive dt */
    float dt_max;                   /**< Maximum adaptive dt */
    float error_tolerance;          /**< Error tolerance for adaptive methods */
    uint32_t max_steps;             /**< Maximum ODE steps per call */
    bool adaptive_stepping;         /**< Enable adaptive time stepping */
} nimcp_lnn_ode_config_t;

/**
 * @brief LNN layer GPU state
 *
 * GPU-resident state tensors for a single LNN layer
 */
typedef struct {
    nimcp_gpu_tensor_t* x;          /**< State [n_neurons] */
    nimcp_gpu_tensor_t* dx_dt;      /**< Derivative [n_neurons] */
    nimcp_gpu_tensor_t* tau;        /**< Time constants [n_neurons] */
    nimcp_gpu_tensor_t* tau_base;   /**< Base time constants [n_neurons] */

    // Weights
    nimcp_gpu_tensor_t* W_in;       /**< Input weights [n_neurons, n_inputs] */
    nimcp_gpu_tensor_t* W_rec;      /**< Recurrent weights [n_neurons, n_neurons] */
    nimcp_gpu_tensor_t* W_tau;      /**< Tau modulation [n_neurons, n_inputs + n_neurons] */
    nimcp_gpu_tensor_t* b_in;       /**< Input bias [n_neurons] */
    nimcp_gpu_tensor_t* b_tau;      /**< Tau bias [n_neurons] */

    // Sparse wiring (CSR format on GPU)
    nimcp_gpu_tensor_t* row_ptr;    /**< CSR row pointers [n_neurons + 1] */
    nimcp_gpu_tensor_t* col_idx;    /**< CSR column indices [n_edges] */
    nimcp_gpu_tensor_t* edge_weights; /**< Edge weights [n_edges] (or NULL) */

    // Configuration
    uint32_t n_neurons;
    uint32_t n_inputs;
    uint32_t n_edges;
    lnn_activation_t activation;
} nimcp_lnn_layer_gpu_t;

//=============================================================================
// Default Configuration
//=============================================================================

/**
 * @brief Get default LNN ODE configuration
 *
 * @return Default configuration (RK4, dt=1.0ms, no adaptive stepping)
 */
NIMCP_EXPORT nimcp_lnn_ode_config_t nimcp_lnn_ode_default_config(void);

//=============================================================================
// ODE Solver Kernels
//=============================================================================

/**
 * @brief Euler step: x_new = x + dt * f(t, x)
 *
 * WHAT: First-order explicit Euler integration
 * WHY:  Fast, simple, single derivative evaluation
 * HOW:  Parallel computation over all neurons
 *
 * @param ctx GPU context
 * @param x Current state tensor [n_neurons]
 * @param dx_dt Computed derivative tensor [n_neurons]
 * @param dt Time step
 * @param x_new Output: new state tensor [n_neurons]
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_lnn_euler_step(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* x,
    const nimcp_gpu_tensor_t* dx_dt,
    float dt,
    nimcp_gpu_tensor_t* x_new
);

/**
 * @brief Heun (RK2) step
 *
 * WHAT: Second-order predictor-corrector method
 * WHY:  Better accuracy than Euler, two derivative evaluations
 * HOW:  k1 = f(t, x); k2 = f(t+dt, x+dt*k1); x_new = x + 0.5*dt*(k1+k2)
 *
 * @param ctx GPU context
 * @param layer LNN layer GPU state (contains weights, biases)
 * @param input External input tensor [n_inputs]
 * @param dt Time step
 * @param config ODE configuration
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_lnn_heun_step(
    nimcp_gpu_context_t* ctx,
    nimcp_lnn_layer_gpu_t* layer,
    const nimcp_gpu_tensor_t* input,
    float dt,
    const nimcp_lnn_ode_config_t* config
);

/**
 * @brief RK4 step
 *
 * WHAT: Fourth-order Runge-Kutta integration
 * WHY:  High accuracy, standard for smooth ODE systems
 * HOW:  Four derivative evaluations with weighted average
 *       k1 = f(t, x)
 *       k2 = f(t + dt/2, x + dt/2 * k1)
 *       k3 = f(t + dt/2, x + dt/2 * k2)
 *       k4 = f(t + dt, x + dt * k3)
 *       x_new = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
 *
 * @param ctx GPU context
 * @param layer LNN layer GPU state
 * @param input External input tensor [n_inputs]
 * @param dt Time step
 * @param config ODE configuration
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_lnn_rk4_step(
    nimcp_gpu_context_t* ctx,
    nimcp_lnn_layer_gpu_t* layer,
    const nimcp_gpu_tensor_t* input,
    float dt,
    const nimcp_lnn_ode_config_t* config
);

/**
 * @brief DOPRI5 adaptive step (Dormand-Prince)
 *
 * WHAT: 5th-order adaptive Runge-Kutta with embedded error estimation
 * WHY:  Automatic step size control for accuracy
 * HOW:  7 derivative evaluations with 4th/5th order pair for error
 *
 * @param ctx GPU context
 * @param layer LNN layer GPU state
 * @param input External input tensor
 * @param dt_ptr In/Out: time step (adjusted based on error)
 * @param config ODE configuration (tolerance, min/max dt)
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_lnn_dopri5_step(
    nimcp_gpu_context_t* ctx,
    nimcp_lnn_layer_gpu_t* layer,
    const nimcp_gpu_tensor_t* input,
    float* dt_ptr,
    const nimcp_lnn_ode_config_t* config
);

/**
 * @brief Unified ODE step with method selection
 *
 * @param ctx GPU context
 * @param layer LNN layer GPU state
 * @param input External input tensor
 * @param config ODE configuration (specifies method)
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_lnn_ode_step(
    nimcp_gpu_context_t* ctx,
    nimcp_lnn_layer_gpu_t* layer,
    const nimcp_gpu_tensor_t* input,
    const nimcp_lnn_ode_config_t* config
);

//=============================================================================
// LTC Dynamics Computation
//=============================================================================

/**
 * @brief Compute LTC neuron derivative: dx/dt = -x/tau(x,I) + f(W_in*I + W_rec*x + b)
 *
 * WHAT: Compute time derivative for all neurons in layer
 * WHY:  Required for any ODE integration method
 * HOW:  Parallel CUDA kernels for tau computation, weighted sums, activation
 *
 * @param ctx GPU context
 * @param layer LNN layer GPU state (contains x, W_in, W_rec, b, tau params)
 * @param input External input [n_inputs]
 * @param dx_dt Output: derivative tensor [n_neurons]
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_lnn_compute_derivative(
    nimcp_gpu_context_t* ctx,
    nimcp_lnn_layer_gpu_t* layer,
    const nimcp_gpu_tensor_t* input,
    nimcp_gpu_tensor_t* dx_dt
);

/**
 * @brief Update time constants: tau(x, I) = tau_base * sigmoid(W_tau * [x; I] + b_tau)
 *
 * WHAT: Compute input-dependent time constants
 * WHY:  LTC neurons adapt temporal response based on context
 * HOW:  Concatenate x and I, apply W_tau, sigmoid, scale by tau_base
 *
 * @param ctx GPU context
 * @param layer LNN layer GPU state
 * @param input External input [n_inputs]
 * @return true on success (tau tensor updated in-place)
 */
NIMCP_EXPORT bool nimcp_gpu_lnn_update_tau(
    nimcp_gpu_context_t* ctx,
    nimcp_lnn_layer_gpu_t* layer,
    const nimcp_gpu_tensor_t* input
);

//=============================================================================
// Sparse Operations (for NCP wiring)
//=============================================================================

/**
 * @brief Sparse matrix-vector product using CSR format
 *
 * WHAT: y = A * x where A is sparse (CSR format)
 * WHY:  Efficient recurrent computation with sparse connectivity
 * HOW:  CUDA kernel optimized for sparse row access patterns
 *
 * @param ctx GPU context
 * @param row_ptr CSR row pointers [n_rows + 1]
 * @param col_idx CSR column indices [nnz]
 * @param values CSR values [nnz] (NULL for binary matrix)
 * @param x Input vector [n_cols]
 * @param y Output vector [n_rows]
 * @param n_rows Number of rows
 * @param alpha Scalar multiplier
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_sparse_matvec(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* row_ptr,
    const nimcp_gpu_tensor_t* col_idx,
    const nimcp_gpu_tensor_t* values,
    const nimcp_gpu_tensor_t* x,
    nimcp_gpu_tensor_t* y,
    uint32_t n_rows,
    float alpha
);

/**
 * @brief Sparse matrix addition: C = alpha * A + beta * B (CSR + CSR)
 *
 * @param ctx GPU context
 * @param A_row_ptr A row pointers
 * @param A_col_idx A column indices
 * @param A_values A values
 * @param B_row_ptr B row pointers
 * @param B_col_idx B column indices
 * @param B_values B values
 * @param C_row_ptr Output C row pointers
 * @param C_col_idx Output C column indices
 * @param C_values Output C values
 * @param n_rows Number of rows
 * @param alpha Scalar for A
 * @param beta Scalar for B
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_sparse_add(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* A_row_ptr,
    const nimcp_gpu_tensor_t* A_col_idx,
    const nimcp_gpu_tensor_t* A_values,
    const nimcp_gpu_tensor_t* B_row_ptr,
    const nimcp_gpu_tensor_t* B_col_idx,
    const nimcp_gpu_tensor_t* B_values,
    nimcp_gpu_tensor_t* C_row_ptr,
    nimcp_gpu_tensor_t* C_col_idx,
    nimcp_gpu_tensor_t* C_values,
    uint32_t n_rows,
    float alpha,
    float beta
);

//=============================================================================
// Gradient Computation (Adjoint Method)
//=============================================================================

/**
 * @brief Initialize adjoint state for backward pass
 *
 * WHAT: Set up adjoint variables λ for reverse-mode autodiff through ODE
 * WHY:  Adjoint method provides O(1) memory gradient computation
 * HOW:  λ(T) = dL/dx(T), then integrate backward
 *
 * @param ctx GPU context
 * @param layer LNN layer GPU state
 * @param grad_output Gradient from next layer (dL/dx at time T)
 * @param adjoint Output: adjoint state tensor [n_neurons]
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_lnn_adjoint_init(
    nimcp_gpu_context_t* ctx,
    nimcp_lnn_layer_gpu_t* layer,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* adjoint
);

/**
 * @brief Single adjoint ODE step (backward in time)
 *
 * WHAT: Integrate adjoint state backward: dλ/dt = -λ * (df/dx)
 * WHY:  Compute gradients w.r.t. parameters via adjoint sensitivity
 * HOW:  Reverse-time RK4 integration of adjoint equations
 *
 * @param ctx GPU context
 * @param layer LNN layer GPU state
 * @param adjoint In/Out: adjoint state tensor [n_neurons]
 * @param x_at_t State at time t (from forward pass)
 * @param input_at_t Input at time t
 * @param dt Time step (positive, but integration goes backward)
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_lnn_adjoint_step(
    nimcp_gpu_context_t* ctx,
    nimcp_lnn_layer_gpu_t* layer,
    nimcp_gpu_tensor_t* adjoint,
    const nimcp_gpu_tensor_t* x_at_t,
    const nimcp_gpu_tensor_t* input_at_t,
    float dt
);

/**
 * @brief Accumulate parameter gradients from adjoint
 *
 * WHAT: Add contributions to dL/dW, dL/db from current adjoint state
 * WHY:  Adjoint method accumulates gradients over time integration
 * HOW:  grad_W += λ * (∂f/∂W), grad_b += λ * (∂f/∂b)
 *
 * @param ctx GPU context
 * @param layer LNN layer GPU state (gradients updated in-place)
 * @param adjoint Current adjoint state
 * @param x_at_t State at time t
 * @param input_at_t Input at time t
 * @param dt Time step
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_lnn_accumulate_gradients(
    nimcp_gpu_context_t* ctx,
    nimcp_lnn_layer_gpu_t* layer,
    const nimcp_gpu_tensor_t* adjoint,
    const nimcp_gpu_tensor_t* x_at_t,
    const nimcp_gpu_tensor_t* input_at_t,
    float dt
);

//=============================================================================
// Layer Lifecycle
//=============================================================================

/**
 * @brief Create GPU LNN layer from CPU layer
 *
 * @param ctx GPU context
 * @param cpu_layer CPU LNN layer (lnn_layer_t from nimcp_lnn_types.h)
 * @return GPU layer structure, or NULL on failure
 */
NIMCP_EXPORT nimcp_lnn_layer_gpu_t* nimcp_lnn_layer_gpu_create(
    nimcp_gpu_context_t* ctx,
    const lnn_layer_t* cpu_layer
);

/**
 * @brief Destroy GPU LNN layer
 *
 * @param layer Layer to destroy
 */
NIMCP_EXPORT void nimcp_lnn_layer_gpu_destroy(nimcp_lnn_layer_gpu_t* layer);

/**
 * @brief Copy GPU layer state back to CPU
 *
 * @param gpu_layer Source GPU layer
 * @param cpu_layer Destination CPU layer
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_lnn_layer_gpu_to_cpu(
    const nimcp_lnn_layer_gpu_t* gpu_layer,
    lnn_layer_t* cpu_layer
);

/**
 * @brief Zero layer gradients
 *
 * @param ctx GPU context
 * @param layer GPU layer
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_lnn_layer_gpu_zero_grad(
    nimcp_gpu_context_t* ctx,
    nimcp_lnn_layer_gpu_t* layer
);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_LNN_GPU_H
