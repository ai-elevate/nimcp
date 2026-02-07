#include <stddef.h>  /* for NULL */
//=============================================================================
// nimcp_snn_backprop.c - SNN Backpropagation Training Implementation
//=============================================================================
/**
 * @file nimcp_snn_backprop.c
 * @brief Implementation of gradient-based SNN training algorithms
 *
 * IMPLEMENTATION NOTES:
 * - Uses surrogate gradients to approximate spike derivatives
 * - Supports multiple training algorithms (BPTT, E-prop, RTRL, etc.)
 * - Integrates with NIMCP gradient manager and loss functions
 * - Memory-efficient temporal unrolling with optional truncation
 *
 * MEMORY LAYOUT:
 * - Activation buffers: [timesteps][batch_size][n_neurons]
 * - Gradient buffers: [n_synapses] accumulated over time
 * - Eligibility traces: [n_synapses] for E-prop/RTRL
 *
 * @author NIMCP Development Team
 * @date 2025-12-20
 * @version 1.0.0
 */

#include "training/nimcp_snn_backprop.h"
#include "utils/validation/nimcp_common.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(snn_backprop)

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Activation recording buffer
 *
 * WHAT: Store network activations during forward pass
 * WHY:  Required for backpropagation through time
 * HOW:  Ring buffer of membrane potentials, spikes, currents
 */
typedef struct {
    nimcp_tensor_t* membrane_v;      /**< Membrane potentials [T][B][N] */
    nimcp_tensor_t* spikes;          /**< Spike outputs [T][B][N] */
    nimcp_tensor_t* currents;        /**< Synaptic currents [T][B][N] */
    nimcp_tensor_t* thresholds;      /**< Adaptive thresholds [T][B][N] */
    uint32_t timesteps;              /**< Number of timesteps recorded */
    uint32_t batch_size;             /**< Batch size */
    uint32_t n_neurons;              /**< Number of neurons */
} activation_buffer_t;

/**
 * @brief Gradient accumulation buffer
 *
 * WHAT: Accumulate weight gradients during backward pass
 * WHY:  Gradients from multiple timesteps/samples must be summed
 * HOW:  Tensor of gradients per synapse
 */
typedef struct {
    nimcp_tensor_t* weight_grads;    /**< Weight gradients [n_synapses] */
    nimcp_tensor_t* threshold_grads; /**< Threshold gradients [n_neurons] */
    nimcp_tensor_t* tau_grads;       /**< Time constant gradients [n_neurons] */
    uint32_t accumulation_count;     /**< Number of gradients accumulated */
} gradient_buffer_t;

/**
 * @brief Eligibility trace buffer (for E-prop/RTRL)
 *
 * WHAT: Local eligibility traces for online learning
 * WHY:  Approximate gradient computation without full BPTT
 * HOW:  Exponentially decaying traces per synapse
 */
typedef struct {
    nimcp_tensor_t* eligibility;     /**< Eligibility traces [n_synapses] */
    float tau;                       /**< Decay time constant (ms) */
    float dt;                        /**< Simulation timestep (ms) */
} eligibility_buffer_t;

/**
 * @brief SNN backprop trainer context
 */
struct snn_backprop_ctx_s {
    /* Configuration */
    snn_backprop_config_t config;

    /* Network reference */
    snn_network_t* network;          /**< SNN network being trained */

    /* Activation recording */
    activation_buffer_t* activations; /**< Forward pass activations */

    /* Gradient buffers */
    gradient_buffer_t* gradients;    /**< Accumulated gradients */
    eligibility_buffer_t* eligibility; /**< Eligibility traces (E-prop/RTRL) */

    /* Integration */
    nimcp_gradient_manager_ctx_t* grad_manager; /**< Gradient manager */
    nimcp_loss_context_t* loss_ctx;  /**< Loss function context */

    /* Statistics */
    snn_backprop_stats_t stats;

    /* Memory pool */
    void* memory_pool;               /**< Memory pool for buffers */

    /* Thread safety */
    void* mutex;                     /**< nimcp_mutex_t* */
};

/**
 * @brief Training batch structure
 */
struct snn_batch_s {
    float* inputs;                   /**< Input data [batch_size × n_inputs] */
    float* targets;                  /**< Target data [batch_size × n_outputs] */
    uint32_t batch_size;             /**< Number of samples */
    uint32_t n_inputs;               /**< Input dimension */
    uint32_t n_outputs;              /**< Output dimension */
};

//=============================================================================
// Default Configuration Functions
//=============================================================================

snn_surrogate_config_t snn_surrogate_default_config(void) {
    snn_surrogate_config_t config = {
        .method = SNN_SURROGATE_SUPERSPIKE,
        .beta = SNN_SURROGATE_BETA_DEFAULT,
        .width = 1.0f,
        .adaptive_beta = false,
        .beta_min = 0.1f,
        .beta_max = 10.0f
    };
    return config;
}

snn_bptt_config_t snn_bptt_default_config(uint32_t sequence_length) {
    // Guard: Use 50-step default if sequence too short
    if (sequence_length == 0) {
        sequence_length = 50;
    }

    snn_bptt_config_t config = {
        .unroll_steps = sequence_length,
        .truncate = true,
        .truncation_length = (sequence_length > 100) ? 100 : sequence_length,
        .detach_spike_grad = false,
        .accumulate_over_time = true
    };
    return config;
}

snn_eprop_config_t snn_eprop_default_config(void) {
    snn_eprop_config_t config = {
        .eligibility_tau = SNN_ELIGIBILITY_TAU_DEFAULT,
        .learning_signal_delay = 0.0f,
        .use_symmetric_eprop = true,
        .adaptive_learning_signal = false,
        .kappa = 0.1f
    };
    return config;
}

snn_loss_config_t snn_loss_default_config(snn_loss_type_t loss_type) {
    snn_loss_config_t config = {
        .type = loss_type,
        .reduction = NIMCP_LOSS_REDUCE_MEAN,
        .target_rate = 10.0f,
        .rate_regularization = 0.001f,
        .timing_precision = 1.0f,
        .earliest_spike_time = 0.0f,
        .latest_spike_time = 100.0f,
        .tau_vr = 10.0f,
        .cost_vp = 1.0f,
        .custom_forward = NULL,
        .custom_backward = NULL,
        .custom_user_data = NULL
    };
    return config;
}

snn_backprop_config_t snn_backprop_default_config(snn_train_algorithm_t algorithm) {
    snn_backprop_config_t config;
    memset(&config, 0, sizeof(config));

    config.algorithm = algorithm;
    config.temporal_mode = SNN_TEMPORAL_BATCH;
    config.surrogate = snn_surrogate_default_config();

    // Algorithm-specific defaults
    config.bptt = snn_bptt_default_config(50);
    config.eprop = snn_eprop_default_config();
    config.rtrl.sparse_jacobian = true;
    config.rtrl.sparsity_threshold = 0.01f;
    config.rtrl.max_jacobian_rank = 100;

    config.loss = snn_loss_default_config(SNN_LOSS_RATE_CODED_MSE);

    config.learning_rate = 0.001f;
    config.weight_decay = 0.0001f;
    config.use_gradient_clipping = true;
    config.gradient_clip_norm = SNN_GRADIENT_CLIP_DEFAULT;

    config.batch_size = 32;
    config.sequence_length = 100;
    config.shuffle_batches = true;

    config.spike_regularization = 0.0f;
    config.membrane_regularization = 0.0f;
    config.use_homeostatic = false;
    config.target_population_rate = 10.0f;

    config.use_gradient_manager = true;
    config.grad_manager_config = nimcp_gradient_manager_default_config();

    config.preallocate_buffers = true;
    config.max_memory_bytes = 1024 * 1024 * 1024; // 1GB

    config.track_gradient_stats = true;
    config.verbose = false;

    return config;
}

//=============================================================================
// Surrogate Gradient Implementations
//=============================================================================

/**
 * @brief Compute SuperSpike surrogate gradient
 *
 * WHAT: σ'(x) = 1 / (β|x| + 1)²
 * WHY:  Smooth, bounded approximation of Dirac delta
 * HOW:  Reciprocal of quadratic in distance from threshold
 *
 * REFERENCE: Zenke & Ganguli 2018
 */
static inline float surrogate_superspike(float x, float beta) {
    float abs_x = fabsf(x);
    float denom = beta * abs_x + 1.0f;
    return 1.0f / (denom * denom);
}

/**
 * @brief Compute fast sigmoid surrogate gradient
 *
 * WHAT: σ'(x) = x / (1 + |x|)²
 * WHY:  Faster computation than SuperSpike
 * HOW:  Variant with x in numerator
 */
static inline float surrogate_fast_sigmoid(float x, float beta) {
    (void)beta; // Unused for fast sigmoid
    float abs_x = fabsf(x);
    float denom = 1.0f + abs_x;
    return x / (denom * denom);
}

/**
 * @brief Compute sigmoid surrogate gradient
 *
 * WHAT: σ'(x) = β·σ(βx)·(1 - σ(βx))
 * WHY:  Standard sigmoid derivative
 * HOW:  Logistic function derivative
 */
static inline float surrogate_sigmoid(float x, float beta) {
    float exp_term = expf(-beta * x);
    float sigmoid = 1.0f / (1.0f + exp_term);
    return beta * sigmoid * (1.0f - sigmoid);
}

/**
 * @brief Compute arctan surrogate gradient
 *
 * WHAT: σ'(x) = 1 / (1 + (πx)²)
 * WHY:  Smooth bell curve
 * HOW:  Derivative of arctan
 */
static inline float surrogate_arctan(float x, float beta) {
    (void)beta;
    float pi_x = M_PI * x;
    return 1.0f / (1.0f + pi_x * pi_x);
}

/**
 * @brief Compute triangular surrogate gradient
 *
 * WHAT: σ'(x) = max(0, 1 - |x|/a)
 * WHY:  Piecewise linear, simple
 * HOW:  Triangle function
 */
static inline float surrogate_triangular(float x, float width) {
    float abs_x = fabsf(x);
    if (abs_x >= width) {
        return 0.0f;
    }
    return 1.0f - abs_x / width;
}

/**
 * @brief Compute rectangular surrogate gradient (STE)
 *
 * WHAT: σ'(x) = 1 if |x| < a else 0
 * WHY:  Straight-through estimator
 * HOW:  Box function
 */
static inline float surrogate_rectangular(float x, float width) {
    return (fabsf(x) < width) ? 1.0f : 0.0f;
}

/**
 * @brief Compute exponential surrogate gradient
 *
 * WHAT: σ'(x) = β·exp(-β|x|)
 * WHY:  Exponential decay from threshold
 * HOW:  Double-sided exponential
 */
static inline float surrogate_exponential(float x, float beta) {
    return beta * expf(-beta * fabsf(x));
}

//=============================================================================
// Public Surrogate Gradient Functions
//=============================================================================

float snn_surrogate_gradient(
    const snn_backprop_ctx_t* ctx,
    float membrane_v
) {
    // Guard: Null check
    if (!ctx) {
        return 0.0f;
    }

    const snn_surrogate_config_t* cfg = &ctx->config.surrogate;

    switch (cfg->method) {
        case SNN_SURROGATE_SUPERSPIKE:
            return surrogate_superspike(membrane_v, cfg->beta);

        case SNN_SURROGATE_FAST_SIGMOID:
            return surrogate_fast_sigmoid(membrane_v, cfg->beta);

        case SNN_SURROGATE_SIGMOID:
            return surrogate_sigmoid(membrane_v, cfg->beta);

        case SNN_SURROGATE_ARCTAN:
            return surrogate_arctan(membrane_v, cfg->beta);

        case SNN_SURROGATE_TRIANGULAR:
            return surrogate_triangular(membrane_v, cfg->width);

        case SNN_SURROGATE_RECTANGULAR:
            return surrogate_rectangular(membrane_v, cfg->width);

        case SNN_SURROGATE_EXPONENTIAL:
            return surrogate_exponential(membrane_v, cfg->beta);

        default:
            NIMCP_LOGGING_ERROR("Unknown surrogate method");
            return 0.0f;
    }
}

nimcp_tensor_t* snn_surrogate_gradient_tensor(
    const snn_backprop_ctx_t* ctx,
    const nimcp_tensor_t* membrane_v
) {
    // Guard: Null checks
    if (!ctx || !membrane_v) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_surrogate_gradient_tensor: required parameter is NULL (ctx, membrane_v)");
        return NULL;
    }

    // Create output tensor with same shape
    const nimcp_tensor_shape_t* shape = nimcp_tensor_shape(membrane_v);
    if (!shape) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "shape is NULL");

        return NULL;
    }

    nimcp_tensor_t* grad = nimcp_tensor_create(
        shape->dims,
        shape->rank,
        NIMCP_DTYPE_F32
    );

    if (!grad) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "grad is NULL");


        return NULL;
    }

    // Apply surrogate gradient element-wise
    size_t numel = nimcp_tensor_numel(membrane_v);
    const float* v_data = (const float*)nimcp_tensor_data_const(membrane_v);
    float* grad_data = (float*)nimcp_tensor_data(grad);

    for (size_t i = 0; i < numel; i++) {
        grad_data[i] = snn_surrogate_gradient(ctx, v_data[i]);
    }

    return grad;
}

int snn_backprop_set_surrogate(
    snn_backprop_ctx_t* ctx,
    snn_surrogate_method_t method
) {
    // Guard: Null check
    if (!ctx) {
        return SNN_ERROR_NULL_POINTER;
    }

    // Guard: Validate method
    if (method >= SNN_SURROGATE_COUNT) {
        return SNN_ERROR_INVALID_STATE;
    }

    ctx->config.surrogate.method = method;
    return SNN_SUCCESS;
}

//=============================================================================
// Lifecycle Functions (Stubs)
//=============================================================================

snn_backprop_ctx_t* snn_backprop_create(
    snn_network_t* network,
    const snn_backprop_config_t* config
) {
    // Guard: Null checks
    if (!network || !config) {
        NIMCP_LOGGING_ERROR("Null network or config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_backprop_create: required parameter is NULL (network, config)");
        return NULL;
    }

    // Guard: Validate config
    if (snn_backprop_validate_config(config) != SNN_SUCCESS) {
        NIMCP_LOGGING_ERROR("Invalid configuration");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_backprop_create: validation failed");
        return NULL;
    }

    // TODO: Full implementation
    NIMCP_LOGGING_INFO("SNN backprop trainer creation (stub)");
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_IMPLEMENTED,
        "snn_backprop_create: stub - full implementation pending");

    return NULL; // Stub return
}

void snn_backprop_destroy(snn_backprop_ctx_t* ctx) {
    // Guard: Null check
    if (!ctx) {
        return;
    }

    // TODO: Free all resources
    NIMCP_LOGGING_INFO("SNN backprop trainer destruction (stub)");
}

int snn_backprop_reset(snn_backprop_ctx_t* ctx) {
    // Guard: Null check
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_backprop_reset: ctx is NULL");
        return SNN_ERROR_NULL_POINTER;
    }

    // TODO: Reset state
    NIMCP_LOGGING_INFO("SNN backprop trainer reset (stub)");

    return SNN_SUCCESS;
}

//=============================================================================
// Forward/Backward Pass Functions (Stubs)
//=============================================================================

int snn_backprop_forward(
    snn_backprop_ctx_t* ctx,
    const float* inputs,
    uint32_t batch_size,
    float duration_ms,
    float* outputs
) {
    // Guard: Null checks
    if (!ctx || !inputs) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_backprop_forward: ctx or inputs is NULL");
        return SNN_ERROR_NULL_POINTER;
    }

    // Guard: Validate batch size
    if (batch_size == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "snn_backprop_forward: batch_size is 0");
        return SNN_ERROR_INVALID_STATE;
    }

    // TODO: Full forward pass with recording
    NIMCP_LOGGING_INFO("SNN forward pass (stub)");

    return SNN_SUCCESS;
}

int snn_backprop_backward(
    snn_backprop_ctx_t* ctx,
    const float* targets,
    uint32_t batch_size
) {
    // Guard: Null checks
    if (!ctx || !targets) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_backprop_backward: ctx or targets is NULL");
        return SNN_ERROR_NULL_POINTER;
    }

    // TODO: Full backward pass
    NIMCP_LOGGING_INFO("SNN backward pass (stub)");

    return SNN_SUCCESS;
}

int snn_backprop_step(
    snn_backprop_ctx_t* ctx,
    float learning_rate
) {
    // Guard: Null check
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_backprop_step: ctx is NULL");
        return SNN_ERROR_NULL_POINTER;
    }

    // Use config learning rate if not specified
    if (learning_rate == 0.0f) {
        learning_rate = ctx->config.learning_rate;
    }

    // TODO: Apply gradients
    NIMCP_LOGGING_INFO("SNN weight update (stub)");

    return 0; // Return number of weights updated
}

int snn_backprop_zero_grad(snn_backprop_ctx_t* ctx) {
    // Guard: Null check
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_backprop_zero_grad: ctx is NULL");
        return SNN_ERROR_NULL_POINTER;
    }

    // TODO: Zero gradients
    return SNN_SUCCESS;
}

//=============================================================================
// Complete Training Step (Stub)
//=============================================================================

int snn_backprop_train_step(
    snn_backprop_ctx_t* ctx,
    const float* inputs,
    const float* targets,
    uint32_t batch_size,
    float duration_ms,
    snn_train_result_t* result
) {
    // Guard: Null checks
    if (!ctx || !inputs || !targets) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_backprop_train_step: ctx, inputs, or targets is NULL");
        return SNN_ERROR_NULL_POINTER;
    }

    // TODO: Complete training step
    // 1. Forward pass
    // 2. Compute loss
    // 3. Backward pass
    // 4. Update weights
    // 5. Fill result structure

    if (result) {
        memset(result, 0, sizeof(snn_train_result_t));
    }

    return SNN_SUCCESS;
}

//=============================================================================
// Utility Functions
//=============================================================================

const char* snn_train_algorithm_name(snn_train_algorithm_t algorithm) {
    switch (algorithm) {
        case SNN_TRAIN_BPTT: return "BPTT";
        case SNN_TRAIN_TRUNCATED_BPTT: return "Truncated-BPTT";
        case SNN_TRAIN_EPROP: return "E-prop";
        case SNN_TRAIN_RTRL: return "RTRL";
        case SNN_TRAIN_SLAYER: return "SLAYER";
        case SNN_TRAIN_DECOLLE: return "DECOLLE";
        case SNN_TRAIN_HYBRID: return "Hybrid";
        default: return "Unknown";
    }
}

const char* snn_surrogate_method_name(snn_surrogate_method_t method) {
    switch (method) {
        case SNN_SURROGATE_SUPERSPIKE: return "SuperSpike";
        case SNN_SURROGATE_FAST_SIGMOID: return "FastSigmoid";
        case SNN_SURROGATE_SIGMOID: return "Sigmoid";
        case SNN_SURROGATE_ARCTAN: return "Arctan";
        case SNN_SURROGATE_TRIANGULAR: return "Triangular";
        case SNN_SURROGATE_RECTANGULAR: return "Rectangular";
        case SNN_SURROGATE_EXPONENTIAL: return "Exponential";
        default: return "Unknown";
    }
}

const char* snn_loss_type_name(snn_loss_type_t type) {
    switch (type) {
        case SNN_LOSS_SPIKE_COUNT: return "SpikeCount";
        case SNN_LOSS_FIRST_SPIKE_TIME: return "FirstSpikeTime";
        case SNN_LOSS_RATE_CODED_MSE: return "RateCodedMSE";
        case SNN_LOSS_RATE_CODED_CROSS_ENTROPY: return "RateCodedCrossEntropy";
        case SNN_LOSS_TEMPORAL_CROSS_ENTROPY: return "TemporalCrossEntropy";
        case SNN_LOSS_VAN_ROSSUM: return "vanRossum";
        case SNN_LOSS_VICTOR_PURPURA: return "VictorPurpura";
        case SNN_LOSS_MEMBRANE_POTENTIAL: return "MembranePotential";
        case SNN_LOSS_CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

int snn_backprop_validate_config(const snn_backprop_config_t* config) {
    // Guard: Null check
    if (!config) {
        return SNN_ERROR_NULL_POINTER;
    }

    // Validate algorithm
    if (config->algorithm >= SNN_TRAIN_MODE_COUNT) {
        NIMCP_LOGGING_ERROR("Invalid training algorithm");
        return SNN_ERROR_INVALID_STATE;
    }

    // Validate surrogate method
    if (config->surrogate.method >= SNN_SURROGATE_COUNT) {
        NIMCP_LOGGING_ERROR("Invalid surrogate method");
        return SNN_ERROR_INVALID_STATE;
    }

    // Validate loss type
    if (config->loss.type >= SNN_LOSS_TYPE_COUNT) {
        NIMCP_LOGGING_ERROR("Invalid loss type");
        return SNN_ERROR_INVALID_STATE;
    }

    // Validate learning rate
    if (config->learning_rate <= 0.0f || config->learning_rate > 1.0f) {
        NIMCP_LOGGING_ERROR("Invalid learning rate");
        return SNN_ERROR_INVALID_STATE;
    }

    // Validate batch size
    if (config->batch_size == 0) {
        NIMCP_LOGGING_ERROR("Invalid batch size");
        return SNN_ERROR_INVALID_STATE;
    }

    // Validate sequence length
    if (config->sequence_length == 0) {
        NIMCP_LOGGING_ERROR("Invalid sequence length");
        return SNN_ERROR_INVALID_STATE;
    }

    return SNN_SUCCESS;
}

//=============================================================================
// Additional Stub Functions
//=============================================================================

int snn_backprop_forward_tensor(
    snn_backprop_ctx_t* ctx,
    const nimcp_tensor_t* input_tensor,
    nimcp_tensor_t* output_tensor
) {
    (void)ctx; (void)input_tensor; (void)output_tensor;
    NIMCP_LOGGING_INFO("SNN forward tensor (stub)");
    return SNN_SUCCESS;
}

int snn_backprop_backward_tensor(
    snn_backprop_ctx_t* ctx,
    const nimcp_tensor_t* target_tensor
) {
    (void)ctx; (void)target_tensor;
    NIMCP_LOGGING_INFO("SNN backward tensor (stub)");
    return SNN_SUCCESS;
}

int snn_backprop_train_step_tensor(
    snn_backprop_ctx_t* ctx,
    const nimcp_tensor_t* input_tensor,
    const nimcp_tensor_t* target_tensor,
    snn_train_result_t* result
) {
    (void)ctx; (void)input_tensor; (void)target_tensor; (void)result;
    NIMCP_LOGGING_INFO("SNN train step tensor (stub)");
    return SNN_SUCCESS;
}

snn_batch_t* snn_batch_create(
    const float* inputs,
    const float* targets,
    uint32_t batch_size,
    uint32_t n_inputs,
    uint32_t n_outputs
) {
    (void)inputs; (void)targets; (void)batch_size;
    (void)n_inputs; (void)n_outputs;
    NIMCP_LOGGING_INFO("SNN batch create (stub)");
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_batch_create: operation failed");
    return NULL;
}

void snn_batch_destroy(snn_batch_t* batch) {
    (void)batch;
    NIMCP_LOGGING_INFO("SNN batch destroy (stub)");
}

int snn_backprop_train_batch(
    snn_backprop_ctx_t* ctx,
    const snn_batch_t* batch,
    float duration_ms,
    snn_train_result_t* result
) {
    (void)ctx; (void)batch; (void)duration_ms; (void)result;
    NIMCP_LOGGING_INFO("SNN train batch (stub)");
    return SNN_SUCCESS;
}

float snn_backprop_compute_loss(
    snn_backprop_ctx_t* ctx,
    const float* outputs,
    const float* targets,
    uint32_t batch_size
) {
    (void)ctx; (void)outputs; (void)targets; (void)batch_size;
    return 0.0f;
}

int snn_backprop_compute_loss_grad(
    snn_backprop_ctx_t* ctx,
    const float* outputs,
    const float* targets,
    uint32_t batch_size,
    float* gradients
) {
    (void)ctx; (void)outputs; (void)targets;
    (void)batch_size; (void)gradients;
    return SNN_SUCCESS;
}

int snn_backprop_connect_gradient_manager(
    snn_backprop_ctx_t* ctx,
    nimcp_gradient_manager_ctx_t* grad_manager
) {
    (void)ctx; (void)grad_manager;
    NIMCP_LOGGING_INFO("Connect gradient manager (stub)");
    return SNN_SUCCESS;
}

nimcp_gradient_manager_ctx_t* snn_backprop_get_gradient_manager(
    snn_backprop_ctx_t* ctx
) {
    return ctx ? ctx->grad_manager : NULL;
}

int snn_backprop_get_stats(
    const snn_backprop_ctx_t* ctx,
    snn_backprop_stats_t* stats
) {
    if (!ctx || !stats) {
        return SNN_ERROR_NULL_POINTER;
    }

    *stats = ctx->stats;
    return SNN_SUCCESS;
}

void snn_backprop_reset_stats(snn_backprop_ctx_t* ctx) {
    if (!ctx) {
        return;
    }

    memset(&ctx->stats, 0, sizeof(snn_backprop_stats_t));
}

float snn_backprop_get_gradient_norm(const snn_backprop_ctx_t* ctx) {
    (void)ctx;
    return 0.0f; // Stub
}

float snn_backprop_get_weight_norm(const snn_backprop_ctx_t* ctx) {
    (void)ctx;
    return 0.0f; // Stub
}
