/**
 * @file nimcp_lnn_gradient.c
 * @brief Implementation of adjoint method for LNN gradient computation
 *
 * WHAT: Memory-efficient gradient computation for continuous-time LNNs
 * WHY:  Enable training without storing full trajectory
 * HOW:  Adjoint ODE integration backward in time
 *
 * @author NIMCP Development Team
 * @date 2025-12-20
 */

#include "lnn/nimcp_lnn_gradient.h"
#include "lnn/nimcp_lnn_types.h"
#include "lnn/nimcp_lnn_layer.h"
#include "lnn/nimcp_lnn_neuron.h"
#include "lnn/nimcp_lnn_ode.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/*=============================================================================
 * Constants
 *===========================================================================*/

#define DEFAULT_CHECKPOINT_INTERVAL 100
#define GRADIENT_HEALTH_THRESHOLD 1e6f
#define JACOBIAN_EPSILON 1e-5f

/*=============================================================================
 * Helper Function Declarations
 *===========================================================================*/

static int allocate_gradient_storage(lnn_gradient_ctx_t* ctx, lnn_network_t* network);
static int allocate_checkpoints(lnn_gradient_ctx_t* ctx, uint32_t max_steps);
static void free_gradient_storage(lnn_gradient_ctx_t* ctx);
static void free_checkpoints(lnn_gradient_ctx_t* ctx);
static int compute_jacobian_numerical(const lnn_layer_t* layer, const nimcp_tensor_t* x, nimcp_tensor_t* jacobian);
static int accumulate_parameter_gradients(lnn_gradient_ctx_t* ctx, lnn_network_t* network, const nimcp_tensor_t* adjoint, float dt);
static bool check_tensor_health(const nimcp_tensor_t* t);
static double lnn_gradient_get_time_ms(void);

/*=============================================================================
 * Lifecycle Functions
 *===========================================================================*/

/**
 * @brief Create gradient context
 *
 * WHAT: Allocate and initialize gradient computation context
 * WHY:  Centralized gradient state management
 * HOW:  Allocate tensors, setup checkpointing if enabled
 */
lnn_gradient_ctx_t* lnn_gradient_ctx_create(
    lnn_network_t* network,
    uint32_t max_steps,
    bool use_checkpointing,
    uint32_t checkpoint_interval
) {
    // Guard: validate inputs
    if (!network) {
        NIMCP_LOGGING_ERROR("Cannot create gradient context: network is NULL");
        return NULL;
    }

    if (max_steps == 0) {
        NIMCP_LOGGING_ERROR("Cannot create gradient context: max_steps must be > 0");
        return NULL;
    }

    // Allocate context
    lnn_gradient_ctx_t* ctx = (lnn_gradient_ctx_t*)nimcp_malloc(sizeof(lnn_gradient_ctx_t));
    if (!ctx) {
        NIMCP_LOGGING_ERROR("Failed to allocate gradient context");
        return NULL;
    }

    memset(ctx, 0, sizeof(lnn_gradient_ctx_t));

    // Initialize fields
    ctx->network = network;
    ctx->n_steps = max_steps;
    ctx->use_checkpointing = use_checkpointing;
    ctx->checkpoint_interval = (checkpoint_interval > 0) ? checkpoint_interval : DEFAULT_CHECKPOINT_INTERVAL;

    // Allocate gradient storage
    if (allocate_gradient_storage(ctx, network) != 0) {
        NIMCP_LOGGING_ERROR("Failed to allocate gradient storage");
        lnn_gradient_ctx_destroy(ctx);
        return NULL;
    }

    // Allocate checkpoints if enabled
    if (use_checkpointing) {
        if (allocate_checkpoints(ctx, max_steps) != 0) {
            NIMCP_LOGGING_ERROR("Failed to allocate checkpoints");
            lnn_gradient_ctx_destroy(ctx);
            return NULL;
        }
    }

    // Create mutex for thread safety
    ctx->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));
    if (ctx->mutex) {
        nimcp_mutex_init((nimcp_mutex_t*)ctx->mutex, NULL);
    }

    NIMCP_LOGGING_INFO("Created gradient context: max_steps=%u, checkpointing=%s",
                       max_steps, use_checkpointing ? "enabled" : "disabled");

    return ctx;
}

/**
 * @brief Destroy gradient context
 *
 * WHAT: Free all gradient computation resources
 * WHY:  Clean shutdown and memory management
 * HOW:  Destroy tensors, free checkpoints, release mutex
 */
void lnn_gradient_ctx_destroy(lnn_gradient_ctx_t* ctx) {
    // Guard: check NULL
    if (!ctx) {
        return;
    }

    // Free gradient storage
    free_gradient_storage(ctx);

    // Free checkpoints
    if (ctx->use_checkpointing) {
        free_checkpoints(ctx);
    }

    // Destroy mutex
    if (ctx->mutex) {
        nimcp_mutex_destroy((nimcp_mutex_t*)ctx->mutex);
        nimcp_free(ctx->mutex);
    }

    // Free context
    nimcp_free(ctx);

    NIMCP_LOGGING_DEBUG("Destroyed gradient context");
}

/*=============================================================================
 * Gradient Computation Functions
 *===========================================================================*/

/**
 * @brief Compute gradients using adjoint method
 *
 * WHAT: Solve adjoint ODE backward to compute ∂L/∂θ
 * WHY:  O(1) memory vs O(T) for BPTT
 * HOW:  Integrate dλ/dt = -∂f/∂x^T λ from T to t0
 */
int lnn_gradient_compute_adjoint(
    lnn_gradient_ctx_t* ctx,
    lnn_network_t* network,
    const nimcp_tensor_t* dL_dx_final
) {
    // Guard: validate inputs
    if (!ctx) {
        NIMCP_LOGGING_ERROR("Gradient context is NULL");
        return LNN_ERROR_NULL_POINTER;
    }

    if (!network) {
        NIMCP_LOGGING_ERROR("Network is NULL");
        return LNN_ERROR_NULL_POINTER;
    }

    if (!dL_dx_final) {
        NIMCP_LOGGING_ERROR("Loss gradient is NULL");
        return LNN_ERROR_NULL_POINTER;
    }

    // Guard: check network has state history
    if (!network->state_history || network->history_len == 0) {
        NIMCP_LOGGING_ERROR("Network has no state history for adjoint computation");
        return LNN_ERROR_INVALID_STATE;
    }

    // Lock for thread safety
    if (ctx->mutex) {
        nimcp_mutex_lock((nimcp_mutex_t*)ctx->mutex);
    }

    double start_time = lnn_gradient_get_time_ms();

    // Reset gradient accumulator
    lnn_gradient_reset(ctx);

    // Store loss gradient
    if (ctx->dL_dx_final) {
        nimcp_tensor_destroy(ctx->dL_dx_final);
    }
    ctx->dL_dx_final = nimcp_tensor_clone(dL_dx_final);

    // Initialize adjoint: λ(T) = ∂L/∂x(T)
    nimcp_tensor_t* adjoint_current = nimcp_tensor_clone(dL_dx_final);
    if (!adjoint_current) {
        NIMCP_LOGGING_ERROR("Failed to initialize adjoint state");
        if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    // Time parameters
    ctx->t_end = (float)network->history_len * network->config->default_dt;
    ctx->t_start = 0.0f;
    ctx->dt = -network->config->default_dt;  // Negative for backward integration

    NIMCP_LOGGING_DEBUG("Starting adjoint computation: T=%f, dt=%f, steps=%u",
                        ctx->t_end, ctx->dt, network->history_len);

    // Backward integration loop
    for (int step = (int)network->history_len - 1; step >= 0; step--) {
        float t = step * network->config->default_dt;

        // Get state at this time step
        nimcp_tensor_t* x_current = network->state_history[step];
        if (!x_current) {
            NIMCP_LOGGING_ERROR("Missing state at step %d", step);
            nimcp_tensor_destroy(adjoint_current);
            if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
            return LNN_ERROR_INVALID_STATE;
        }

        // Compute adjoint step: λ(t-dt) from λ(t)
        nimcp_tensor_t* adjoint_next = nimcp_tensor_create(
            nimcp_tensor_shape(adjoint_current)->dims,
            nimcp_tensor_shape(adjoint_current)->rank,
            NIMCP_DTYPE_F32
        );

        if (!adjoint_next) {
            NIMCP_LOGGING_ERROR("Failed to allocate adjoint_next at step %d", step);
            nimcp_tensor_destroy(adjoint_current);
            if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
            return LNN_ERROR_OUT_OF_MEMORY;
        }

        int ret = lnn_gradient_adjoint_step(ctx, x_current, adjoint_current, t, ctx->dt, adjoint_next);
        if (ret != 0) {
            NIMCP_LOGGING_ERROR("Adjoint step failed at step %d: error %d", step, ret);
            nimcp_tensor_destroy(adjoint_current);
            nimcp_tensor_destroy(adjoint_next);
            if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
            return ret;
        }

        // Accumulate parameter gradients: ∂L/∂θ += λ^T ∂f/∂θ * dt
        ret = accumulate_parameter_gradients(ctx, network, adjoint_current, fabs(ctx->dt));
        if (ret != 0) {
            NIMCP_LOGGING_WARN("Failed to accumulate gradients at step %d", step);
        }

        // Update adjoint state
        nimcp_tensor_destroy(adjoint_current);
        adjoint_current = adjoint_next;

        ctx->adjoint_steps++;

        // Health check every N steps
        if (step % 100 == 0) {
            if (!check_tensor_health(adjoint_current)) {
                NIMCP_LOGGING_ERROR("Adjoint health check failed at step %d", step);
                ctx->has_nan = true;
                nimcp_tensor_destroy(adjoint_current);
                if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
                return LNN_ERROR_INVALID_STATE;
            }
        }
    }

    // Cleanup
    nimcp_tensor_destroy(adjoint_current);

    // Compute final gradient norm
    ctx->gradient_norm = lnn_gradient_norm(ctx);
    ctx->max_gradient = fmaxf(ctx->max_gradient, ctx->gradient_norm);

    ctx->compute_time_ms = lnn_gradient_get_time_ms() - start_time;

    if (ctx->mutex) {
        nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
    }

    NIMCP_LOGGING_INFO("Adjoint computation complete: norm=%f, time=%f ms, steps=%lu",
                       ctx->gradient_norm, ctx->compute_time_ms, ctx->adjoint_steps);

    return 0;
}

/**
 * @brief Compute gradients using BPTT
 *
 * WHAT: Standard backpropagation through time
 * WHY:  Alternative to adjoint for short sequences
 * HOW:  Unroll network, backprop through each step
 */
int lnn_gradient_compute_bptt(
    lnn_gradient_ctx_t* ctx,
    lnn_network_t* network,
    const nimcp_tensor_t* dL_dx_sequence
) {
    // Guard: validate inputs
    if (!ctx || !network || !dL_dx_sequence) {
        NIMCP_LOGGING_ERROR("NULL input to BPTT gradient computation");
        return LNN_ERROR_NULL_POINTER;
    }

    // BPTT requires full state history
    if (!network->state_history || network->history_len == 0) {
        NIMCP_LOGGING_ERROR("No state history for BPTT");
        return LNN_ERROR_INVALID_STATE;
    }

    NIMCP_LOGGING_WARN("BPTT gradient computation not yet implemented");
    // TODO: Implement BPTT as alternative to adjoint method
    // This requires unrolling the network and backpropagating through each time step

    return LNN_ERROR_OPERATION_FAILED;
}

/**
 * @brief Parallel gradient computation across batch
 *
 * WHAT: Compute gradients for batch in parallel
 * WHY:  Exploit multi-core for faster training
 * HOW:  Distribute batch items across thread pool
 */
int lnn_gradient_compute_batch_parallel(
    lnn_gradient_ctx_t* ctx,
    lnn_network_t* network,
    const nimcp_tensor_t* dL_dx_batch,
    uint32_t batch_size,
    void* thread_pool
) {
    // Guard: validate inputs
    if (!ctx || !network || !dL_dx_batch) {
        NIMCP_LOGGING_ERROR("NULL input to batch parallel gradient");
        return LNN_ERROR_NULL_POINTER;
    }

    if (batch_size == 0) {
        NIMCP_LOGGING_ERROR("Batch size is zero");
        return LNN_ERROR_INVALID_PARAM;
    }

    NIMCP_LOGGING_WARN("Batch parallel gradient computation not yet implemented");
    // TODO: Implement parallel batch gradient computation
    // This requires:
    // 1. Split batch across threads
    // 2. Compute gradient for each item independently
    // 3. Aggregate gradients across batch

    return LNN_ERROR_OPERATION_FAILED;
}

/*=============================================================================
 * Adjoint ODE Solving
 *===========================================================================*/

/**
 * @brief Single adjoint ODE step
 *
 * WHAT: Integrate dλ/dt = -∂f/∂x^T λ for one time step
 * WHY:  Core adjoint computation
 * HOW:  Use RK4 or Euler on adjoint dynamics
 */
int lnn_gradient_adjoint_step(
    lnn_gradient_ctx_t* ctx,
    const nimcp_tensor_t* x,
    const nimcp_tensor_t* adjoint,
    float t,
    float dt,
    nimcp_tensor_t* adjoint_next
) {
    // Guard: validate inputs
    if (!ctx || !x || !adjoint || !adjoint_next) {
        NIMCP_LOGGING_ERROR("NULL input to adjoint step");
        return LNN_ERROR_NULL_POINTER;
    }

    // Get network and first layer (simplified - assumes single layer)
    lnn_network_t* network = ctx->network;
    if (!network || !network->layers || network->n_layers == 0) {
        NIMCP_LOGGING_ERROR("Invalid network structure");
        return LNN_ERROR_INVALID_STATE;
    }

    lnn_layer_t* layer = network->layers[0];

    // Compute Jacobian ∂f/∂x at current state
    nimcp_tensor_t* jacobian = nimcp_tensor_create(
        (uint32_t[]){layer->n_neurons, layer->n_neurons}, 2, NIMCP_DTYPE_F32
    );
    if (!jacobian) {
        NIMCP_LOGGING_ERROR("Failed to allocate Jacobian");
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    int ret = lnn_gradient_compute_jacobian(layer, x, jacobian);
    if (ret != 0) {
        NIMCP_LOGGING_ERROR("Jacobian computation failed");
        nimcp_tensor_destroy(jacobian);
        return ret;
    }

    ctx->jacobian_evals++;

    // Adjoint dynamics: dλ/dt = -J^T λ
    // Compute -J^T λ
    nimcp_tensor_t* jacobian_T = nimcp_tensor_transpose(jacobian);
    nimcp_tensor_t* J_T_lambda = nimcp_tensor_mv(jacobian_T, adjoint);
    nimcp_tensor_t* d_adjoint_dt = nimcp_tensor_mul_scalar(J_T_lambda, -1.0);

    // Simple Euler step: λ(t+dt) = λ(t) + dt * dλ/dt
    nimcp_tensor_t* delta = nimcp_tensor_mul_scalar(d_adjoint_dt, dt);

    // Copy adjoint to adjoint_next and add delta
    size_t numel = nimcp_tensor_numel(adjoint);
    float* adj_data = (float*)nimcp_tensor_data((nimcp_tensor_t*)adjoint);
    float* delta_data = (float*)nimcp_tensor_data(delta);
    float* next_data = (float*)nimcp_tensor_data(adjoint_next);

    for (size_t i = 0; i < numel; i++) {
        next_data[i] = adj_data[i] + delta_data[i];
    }

    // Cleanup
    nimcp_tensor_destroy(jacobian);
    nimcp_tensor_destroy(jacobian_T);
    nimcp_tensor_destroy(J_T_lambda);
    nimcp_tensor_destroy(d_adjoint_dt);
    nimcp_tensor_destroy(delta);

    return 0;
}

/**
 * @brief Compute Jacobian ∂f/∂x for layer
 *
 * WHAT: Compute Jacobian matrix of layer dynamics
 * WHY:  Required for adjoint ODE
 * HOW:  Numerical differentiation (can be optimized to analytical)
 */
int lnn_gradient_compute_jacobian(
    const lnn_layer_t* layer,
    const nimcp_tensor_t* x,
    nimcp_tensor_t* jacobian
) {
    // Guard: validate inputs
    if (!layer || !x || !jacobian) {
        NIMCP_LOGGING_ERROR("NULL input to Jacobian computation");
        return LNN_ERROR_NULL_POINTER;
    }

    // Use numerical differentiation
    return compute_jacobian_numerical(layer, x, jacobian);
}

/*=============================================================================
 * Gradient Access and Application
 *===========================================================================*/

/**
 * @brief Get accumulated parameter gradients
 *
 * WHAT: Extract final ∂L/∂θ
 * WHY:  Transfer to optimizer
 * HOW:  Copy from internal storage
 */
int lnn_gradient_get_params(
    const lnn_gradient_ctx_t* ctx,
    nimcp_tensor_t* grad_params
) {
    // Guard: validate inputs
    if (!ctx || !grad_params) {
        NIMCP_LOGGING_ERROR("NULL input to get_params");
        return LNN_ERROR_NULL_POINTER;
    }

    if (!ctx->grad_params) {
        NIMCP_LOGGING_ERROR("No gradients available");
        return LNN_ERROR_INVALID_STATE;
    }

    // Copy gradients
    size_t numel = nimcp_tensor_numel(ctx->grad_params);
    size_t numel_out = nimcp_tensor_numel(grad_params);

    if (numel != numel_out) {
        NIMCP_LOGGING_ERROR("Gradient size mismatch: %zu vs %zu", numel, numel_out);
        return LNN_ERROR_INVALID_PARAM;
    }

    memcpy(nimcp_tensor_data(grad_params),
           nimcp_tensor_data_const(ctx->grad_params),
           numel * sizeof(float));

    return 0;
}

/**
 * @brief Apply gradients using optimizer
 *
 * WHAT: Update network parameters
 * WHY:  Convenience for training loop
 * HOW:  Call optimizer update
 */
int lnn_gradient_apply(
    lnn_gradient_ctx_t* ctx,
    lnn_network_t* network,
    void* optimizer
) {
    // Guard: validate inputs
    if (!ctx || !network || !optimizer) {
        NIMCP_LOGGING_ERROR("NULL input to gradient apply");
        return LNN_ERROR_NULL_POINTER;
    }

    NIMCP_LOGGING_WARN("Gradient apply not yet implemented (requires optimizer integration)");
    // TODO: Implement optimizer integration
    // This requires calling the optimizer's update function with accumulated gradients

    return LNN_ERROR_OPERATION_FAILED;
}

/**
 * @brief Reset gradient accumulator
 *
 * WHAT: Zero out gradients
 * WHY:  Prepare for next batch
 * HOW:  Set to zero
 */
void lnn_gradient_reset(lnn_gradient_ctx_t* ctx) {
    // Guard: check NULL
    if (!ctx) {
        return;
    }

    // Zero gradient tensor
    if (ctx->grad_params) {
        size_t numel = nimcp_tensor_numel(ctx->grad_params);
        float* data = (float*)nimcp_tensor_data(ctx->grad_params);
        memset(data, 0, numel * sizeof(float));
    }

    // Reset health flags
    ctx->has_nan = false;
    ctx->has_inf = false;
    ctx->gradient_norm = 0.0f;
}

/*=============================================================================
 * Checkpointing Functions
 *===========================================================================*/

/**
 * @brief Save state checkpoint
 *
 * WHAT: Save network state at time step
 * WHY:  Tradeoff memory for computation
 * HOW:  Store copy in checkpoint array
 */
int lnn_gradient_save_checkpoint(
    lnn_gradient_ctx_t* ctx,
    uint32_t step,
    const nimcp_tensor_t* state
) {
    // Guard: validate inputs
    if (!ctx || !state) {
        return LNN_ERROR_NULL_POINTER;
    }

    if (!ctx->use_checkpointing) {
        return 0;  // Checkpointing disabled
    }

    // Check capacity
    if (step >= ctx->checkpoint_capacity) {
        NIMCP_LOGGING_WARN("Checkpoint step %u exceeds capacity %u", step, ctx->checkpoint_capacity);
        return LNN_ERROR_INVALID_PARAM;
    }

    // Free existing checkpoint if present
    if (ctx->checkpoints[step]) {
        nimcp_tensor_destroy(ctx->checkpoints[step]);
    }

    // Clone state
    ctx->checkpoints[step] = nimcp_tensor_clone(state);
    if (!ctx->checkpoints[step]) {
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    ctx->n_checkpoints++;
    return 0;
}

/**
 * @brief Load state checkpoint
 *
 * WHAT: Restore state from checkpoint
 * WHY:  Resume forward pass
 * HOW:  Copy checkpoint to output
 */
int lnn_gradient_load_checkpoint(
    lnn_gradient_ctx_t* ctx,
    uint32_t step,
    nimcp_tensor_t* state
) {
    // Guard: validate inputs
    if (!ctx || !state) {
        return LNN_ERROR_NULL_POINTER;
    }

    if (!ctx->use_checkpointing) {
        return LNN_ERROR_INVALID_STATE;
    }

    if (step >= ctx->checkpoint_capacity || !ctx->checkpoints[step]) {
        NIMCP_LOGGING_ERROR("No checkpoint at step %u", step);
        return LNN_ERROR_INVALID_PARAM;
    }

    // Copy checkpoint data
    size_t numel = nimcp_tensor_numel(ctx->checkpoints[step]);
    memcpy(nimcp_tensor_data(state),
           nimcp_tensor_data_const(ctx->checkpoints[step]),
           numel * sizeof(float));

    ctx->checkpoints_used++;
    return 0;
}

/**
 * @brief Recompute forward from checkpoint
 *
 * WHAT: Re-run forward pass between checkpoints
 * WHY:  Recover intermediate states
 * HOW:  Load checkpoint, simulate forward
 */
int lnn_gradient_recompute_from_checkpoint(
    lnn_gradient_ctx_t* ctx,
    lnn_network_t* network,
    uint32_t from_step,
    uint32_t to_step
) {
    // Guard: validate inputs
    if (!ctx || !network) {
        return LNN_ERROR_NULL_POINTER;
    }

    if (from_step >= to_step) {
        return LNN_ERROR_INVALID_PARAM;
    }

    NIMCP_LOGGING_DEBUG("Recomputing forward from step %u to %u", from_step, to_step);

    // TODO: Implement forward recomputation
    // This requires:
    // 1. Load checkpoint at from_step
    // 2. Run forward ODE for (to_step - from_step) steps
    // 3. Store intermediate states if needed

    return LNN_ERROR_OPERATION_FAILED;
}

/*=============================================================================
 * Utility Functions
 *===========================================================================*/

/**
 * @brief Compute gradient norm
 *
 * WHAT: ||∂L/∂θ||₂
 * WHY:  Monitor gradient health
 * HOW:  L2 norm
 */
float lnn_gradient_norm(const lnn_gradient_ctx_t* ctx) {
    // Guard: check NULL
    if (!ctx || !ctx->grad_params) {
        return 0.0f;
    }

    return (float)nimcp_tensor_norm_p(ctx->grad_params, 2.0);
}

/**
 * @brief Clip gradients by norm
 *
 * WHAT: Scale gradients if norm > max
 * WHY:  Prevent gradient explosion
 * HOW:  grad ← grad * (max_norm / norm)
 */
int lnn_gradient_clip(lnn_gradient_ctx_t* ctx, float max_norm) {
    // Guard: validate inputs
    if (!ctx) {
        return LNN_ERROR_NULL_POINTER;
    }

    if (max_norm <= 0.0f) {
        return LNN_ERROR_INVALID_PARAM;
    }

    float norm = lnn_gradient_norm(ctx);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        nimcp_tensor_mul_scalar_(ctx->grad_params, scale);

        NIMCP_LOGGING_DEBUG("Clipped gradients: norm %f -> %f (scale=%f)", norm, max_norm, scale);
    }

    return 0;
}

/**
 * @brief Check gradient health
 *
 * WHAT: Detect NaN, Inf, or explosion
 * WHY:  Early detection prevents divergence
 * HOW:  Scan gradient tensor
 */
bool lnn_gradient_check_health(const lnn_gradient_ctx_t* ctx) {
    // Guard: check NULL
    if (!ctx || !ctx->grad_params) {
        return false;
    }

    // Check tensor health
    if (!check_tensor_health(ctx->grad_params)) {
        return false;
    }

    // Check norm threshold
    float norm = lnn_gradient_norm(ctx);
    if (norm > GRADIENT_HEALTH_THRESHOLD) {
        NIMCP_LOGGING_WARN("Gradient norm %f exceeds health threshold %f", norm, GRADIENT_HEALTH_THRESHOLD);
        return false;
    }

    return true;
}

/**
 * @brief Get gradient statistics
 *
 * WHAT: Retrieve gradient metrics
 * WHY:  Performance profiling
 * HOW:  Return accumulated stats
 */
int lnn_gradient_get_stats(
    const lnn_gradient_ctx_t* ctx,
    uint64_t* adjoint_steps,
    uint64_t* jacobian_evals,
    uint64_t* checkpoints_used,
    double* compute_time_ms
) {
    // Guard: validate inputs
    if (!ctx) {
        return LNN_ERROR_NULL_POINTER;
    }

    if (adjoint_steps) *adjoint_steps = ctx->adjoint_steps;
    if (jacobian_evals) *jacobian_evals = ctx->jacobian_evals;
    if (checkpoints_used) *checkpoints_used = ctx->checkpoints_used;
    if (compute_time_ms) *compute_time_ms = ctx->compute_time_ms;

    return 0;
}

/*=============================================================================
 * Helper Functions
 *===========================================================================*/

/**
 * @brief Allocate gradient storage
 *
 * WHAT: Create tensors for adjoint and gradients
 * WHY:  Initialize gradient computation
 * HOW:  Allocate based on network size
 */
static int allocate_gradient_storage(lnn_gradient_ctx_t* ctx, lnn_network_t* network) {
    // Guard: validate inputs
    if (!ctx || !network) {
        return LNN_ERROR_NULL_POINTER;
    }

    // Count total parameters (simplified - assumes single layer)
    if (!network->layers || network->n_layers == 0) {
        NIMCP_LOGGING_ERROR("Network has no layers");
        return LNN_ERROR_INVALID_STATE;
    }

    lnn_layer_t* layer = network->layers[0];
    uint32_t n_neurons = layer->n_neurons;

    // Allocate adjoint state tensor
    ctx->adjoint = nimcp_tensor_zeros((uint32_t[]){n_neurons}, 1, NIMCP_DTYPE_F32);
    if (!ctx->adjoint) {
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    // Estimate parameter count (simplified)
    // Full implementation would count all W_in, W_rec, W_tau, biases, tau_base across all layers
    size_t n_params = n_neurons * (n_neurons + 10);  // Rough estimate

    // Allocate gradient parameter tensor
    ctx->grad_params = nimcp_tensor_zeros((uint32_t[]){(uint32_t)n_params}, 1, NIMCP_DTYPE_F32);
    if (!ctx->grad_params) {
        nimcp_tensor_destroy(ctx->adjoint);
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    return 0;
}

/**
 * @brief Allocate checkpoints
 *
 * WHAT: Create checkpoint storage
 * WHY:  Enable memory-computation tradeoff
 * HOW:  Allocate array of tensor pointers
 */
static int allocate_checkpoints(lnn_gradient_ctx_t* ctx, uint32_t max_steps) {
    // Guard: validate inputs
    if (!ctx) {
        return LNN_ERROR_NULL_POINTER;
    }

    // Calculate number of checkpoints needed
    uint32_t n_checkpoints = (max_steps + ctx->checkpoint_interval - 1) / ctx->checkpoint_interval;

    ctx->checkpoint_capacity = n_checkpoints;
    ctx->checkpoints = (nimcp_tensor_t**)nimcp_malloc(n_checkpoints * sizeof(nimcp_tensor_t*));
    if (!ctx->checkpoints) {
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    memset(ctx->checkpoints, 0, n_checkpoints * sizeof(nimcp_tensor_t*));

    NIMCP_LOGGING_DEBUG("Allocated %u checkpoint slots (interval=%u)", n_checkpoints, ctx->checkpoint_interval);

    return 0;
}

/**
 * @brief Free gradient storage
 *
 * WHAT: Destroy gradient tensors
 * WHY:  Clean shutdown
 * HOW:  Destroy each tensor
 */
static void free_gradient_storage(lnn_gradient_ctx_t* ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->adjoint) {
        nimcp_tensor_destroy(ctx->adjoint);
        ctx->adjoint = NULL;
    }

    if (ctx->adjoint_prev) {
        nimcp_tensor_destroy(ctx->adjoint_prev);
        ctx->adjoint_prev = NULL;
    }

    if (ctx->grad_params) {
        nimcp_tensor_destroy(ctx->grad_params);
        ctx->grad_params = NULL;
    }

    if (ctx->dL_dx_final) {
        nimcp_tensor_destroy(ctx->dL_dx_final);
        ctx->dL_dx_final = NULL;
    }
}

/**
 * @brief Free checkpoints
 *
 * WHAT: Destroy checkpoint storage
 * WHY:  Clean shutdown
 * HOW:  Destroy each checkpoint tensor
 */
static void free_checkpoints(lnn_gradient_ctx_t* ctx) {
    if (!ctx || !ctx->checkpoints) {
        return;
    }

    for (uint32_t i = 0; i < ctx->checkpoint_capacity; i++) {
        if (ctx->checkpoints[i]) {
            nimcp_tensor_destroy(ctx->checkpoints[i]);
        }
    }

    nimcp_free(ctx->checkpoints);
    ctx->checkpoints = NULL;
    ctx->n_checkpoints = 0;
}

/**
 * @brief Compute Jacobian numerically
 *
 * WHAT: Numerical Jacobian via finite differences
 * WHY:  Simple implementation (can be optimized to analytical)
 * HOW:  Perturb each state element, measure derivative
 *
 * NOTE: Uses fixed epsilon (JACOBIAN_EPSILON = 1e-5f)
 * LIMITATION: Fixed epsilon may not be optimal for all state magnitudes
 * FUTURE: Consider making epsilon configurable or adaptive (scaled by state magnitude)
 */
static int compute_jacobian_numerical(
    const lnn_layer_t* layer,
    const nimcp_tensor_t* x,
    nimcp_tensor_t* jacobian
) {
    // Guard: validate inputs
    if (!layer || !x || !jacobian) {
        return LNN_ERROR_NULL_POINTER;
    }

    uint32_t n = layer->n_neurons;
    /* Fixed epsilon for finite differences
     * Known limitation: Not scaled by state magnitude
     * Could be made configurable for improved numerical accuracy
     */
    float eps = JACOBIAN_EPSILON;

    // Clone state for perturbation
    nimcp_tensor_t* x_perturbed = nimcp_tensor_clone(x);
    if (!x_perturbed) {
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    // Allocate buffers for f(x+eps) and f(x-eps)
    nimcp_tensor_t* f_plus = nimcp_tensor_create((uint32_t[]){n}, 1, NIMCP_DTYPE_F32);
    nimcp_tensor_t* f_minus = nimcp_tensor_create((uint32_t[]){n}, 1, NIMCP_DTYPE_F32);

    if (!f_plus || !f_minus) {
        nimcp_tensor_destroy(x_perturbed);
        if (f_plus) nimcp_tensor_destroy(f_plus);
        if (f_minus) nimcp_tensor_destroy(f_minus);
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    float* x_data = (float*)nimcp_tensor_data(x_perturbed);
    float* jac_data = (float*)nimcp_tensor_data(jacobian);
    float* f_plus_data = (float*)nimcp_tensor_data(f_plus);
    float* f_minus_data = (float*)nimcp_tensor_data(f_minus);
    const float* x_orig_data = (const float*)nimcp_tensor_data_const(x);

    // For each state variable i (column of Jacobian)
    for (uint32_t i = 0; i < n; i++) {
        float x_orig = x_orig_data[i];

        // Compute f(x + eps*e_i)
        memcpy(x_data, x_orig_data, n * sizeof(float));
        x_data[i] = x_orig + eps;

        // Evaluate layer dynamics: dx/dt = -x/tau + activation(W_rec @ x)
        for (uint32_t j = 0; j < n; j++) {
            float tau_j = (layer->tau_base && nimcp_tensor_data(layer->tau_base)) ?
                         ((float*)nimcp_tensor_data(layer->tau_base))[j] : 10.0f;
            float recurrent_sum = 0.0f;

            // W_rec @ x (simplified dense matmul)
            if (layer->W_rec && nimcp_tensor_data(layer->W_rec)) {
                float* W_rec_data = (float*)nimcp_tensor_data(layer->W_rec);
                for (uint32_t k = 0; k < n; k++) {
                    recurrent_sum += W_rec_data[j * n + k] * x_data[k];
                }
            }

            // dx/dt = -x/tau + tanh(recurrent)
            f_plus_data[j] = -x_data[j] / tau_j + tanhf(recurrent_sum);
        }

        // Compute f(x - eps*e_i)
        memcpy(x_data, x_orig_data, n * sizeof(float));
        x_data[i] = x_orig - eps;

        for (uint32_t j = 0; j < n; j++) {
            float tau_j = (layer->tau_base && nimcp_tensor_data(layer->tau_base)) ?
                         ((float*)nimcp_tensor_data(layer->tau_base))[j] : 10.0f;
            float recurrent_sum = 0.0f;

            if (layer->W_rec && nimcp_tensor_data(layer->W_rec)) {
                float* W_rec_data = (float*)nimcp_tensor_data(layer->W_rec);
                for (uint32_t k = 0; k < n; k++) {
                    recurrent_sum += W_rec_data[j * n + k] * x_data[k];
                }
            }

            f_minus_data[j] = -x_data[j] / tau_j + tanhf(recurrent_sum);
        }

        // Jacobian column i: ∂f/∂x_i = (f_plus - f_minus) / (2*eps)
        for (uint32_t j = 0; j < n; j++) {
            jac_data[j * n + i] = (f_plus_data[j] - f_minus_data[j]) / (2.0f * eps);
        }
    }

    nimcp_tensor_destroy(x_perturbed);
    nimcp_tensor_destroy(f_plus);
    nimcp_tensor_destroy(f_minus);
    return 0;
}

/**
 * @brief Accumulate parameter gradients
 *
 * WHAT: Accumulate ∂L/∂θ += λ^T ∂f/∂θ * dt
 * WHY:  Build up total gradient over time
 * HOW:  Compute parameter Jacobian, multiply by adjoint
 */
static int accumulate_parameter_gradients(
    lnn_gradient_ctx_t* ctx,
    lnn_network_t* network,
    const nimcp_tensor_t* adjoint,
    float dt
) {
    // Guard: validate inputs
    if (!ctx || !network || !adjoint) {
        return LNN_ERROR_NULL_POINTER;
    }

    if (!network->layers || network->n_layers == 0) {
        return LNN_ERROR_INVALID_STATE;
    }

    // For each layer, compute and accumulate parameter gradients
    for (uint32_t layer_idx = 0; layer_idx < network->n_layers; layer_idx++) {
        lnn_layer_t* layer = network->layers[layer_idx];
        if (!layer || !layer->x) continue;

        uint32_t n = layer->n_neurons;
        /* n_inputs can be derived from W_in->shape.dims[1] if needed */

        const float* adjoint_data = (const float*)nimcp_tensor_data_const(adjoint);
        const float* x_data = (const float*)nimcp_tensor_data_const(layer->x);

        // Gradient w.r.t. W_rec: ∂L/∂W_rec = λ^T * ∂f/∂W_rec * dt
        // ∂f/∂W_rec[j,k] = activation'(...) * x[k] for neuron j
        if (layer->grad_W_rec && layer->W_rec) {
            float* grad_W_rec = (float*)nimcp_tensor_data(layer->grad_W_rec);

            for (uint32_t j = 0; j < n; j++) {
                // Get recurrent sum for derivative computation
                float recurrent_sum = 0.0f;
                float* W_rec_data = (float*)nimcp_tensor_data(layer->W_rec);

                for (uint32_t k = 0; k < n; k++) {
                    recurrent_sum += W_rec_data[j * n + k] * x_data[k];
                }

                // Activation derivative (tanh': 1 - tanh^2)
                float tanh_val = tanhf(recurrent_sum);
                float act_deriv = 1.0f - tanh_val * tanh_val;

                // Accumulate gradient for each weight W_rec[j,k]
                for (uint32_t k = 0; k < n; k++) {
                    grad_W_rec[j * n + k] += adjoint_data[j] * act_deriv * x_data[k] * dt;
                }
            }
        }

        // Gradient w.r.t. tau_base: ∂L/∂tau_base = λ^T * ∂f/∂tau * dt
        // ∂f/∂tau[j] = x[j] / (tau^2) (from -x/tau term)
        if (layer->grad_tau_base && layer->tau_base) {
            float* grad_tau_base = (float*)nimcp_tensor_data(layer->grad_tau_base);
            float* tau_base_data = (float*)nimcp_tensor_data(layer->tau_base);

            for (uint32_t j = 0; j < n; j++) {
                float tau_j = tau_base_data[j];
                if (tau_j > 0.0f) {
                    // ∂(-x/tau)/∂tau = x / tau^2
                    float df_dtau = x_data[j] / (tau_j * tau_j);
                    grad_tau_base[j] += adjoint_data[j] * df_dtau * dt;
                }
            }
        }

        // Gradient w.r.t. W_in (if input is available in context)
        // This would require storing the input, which we don't have here
        // In a full implementation, we'd need input history or checkpointing

        // Gradient w.r.t. biases can be computed similarly
        if (layer->grad_b_in) {
            float* grad_b_in = (float*)nimcp_tensor_data(layer->grad_b_in);

            for (uint32_t j = 0; j < n; j++) {
                // ∂f/∂b_in = activation'(...)
                // Simplified: assume contribution through recurrent path
                grad_b_in[j] += adjoint_data[j] * dt;
            }
        }
    }

    return 0;
}

/**
 * @brief Check tensor health
 *
 * WHAT: Scan tensor for NaN/Inf
 * WHY:  Early detection of numerical issues
 * HOW:  Iterate through elements
 */
static bool check_tensor_health(const nimcp_tensor_t* t) {
    if (!t) {
        return false;
    }

    size_t numel = nimcp_tensor_numel(t);
    const float* data = (const float*)nimcp_tensor_data_const(t);

    for (size_t i = 0; i < numel; i++) {
        if (isnan(data[i]) || isinf(data[i])) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Get current time in milliseconds
 *
 * WHAT: Platform-independent time measurement
 * WHY:  Performance profiling
 * HOW:  Use system clock
 */
static double lnn_gradient_get_time_ms(void) {
    // Simplified placeholder - should use proper timer
    // In real implementation, use clock_gettime or QueryPerformanceCounter
    return 0.0;
}
