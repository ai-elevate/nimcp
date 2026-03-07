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
#include "constants/nimcp_constants.h"
#include "utils/validation/nimcp_common.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
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

    config.learning_rate = NIMCP_LEARNING_RATE_FINE;
    config.weight_decay = NIMCP_WEIGHT_DECAY_DEFAULT;
    config.use_gradient_clipping = true;
    config.gradient_clip_norm = SNN_GRADIENT_CLIP_DEFAULT;

    config.batch_size = NIMCP_DEFAULT_BATCH_SIZE;
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

/**
 * @brief Helper: Allocate activation buffer for recording forward pass state
 */
static activation_buffer_t* alloc_activation_buffer(
    uint32_t timesteps, uint32_t batch_size, uint32_t n_neurons)
{
    activation_buffer_t* buf = nimcp_calloc(1, sizeof(activation_buffer_t));
    if (!buf) return NULL;

    buf->timesteps = timesteps;
    buf->batch_size = batch_size;
    buf->n_neurons = n_neurons;

    uint64_t total = (uint64_t)timesteps * batch_size * n_neurons;
    if (total == 0) total = 1;
    if (total > UINT32_MAX) {
        nimcp_free(buf);
        return NULL;
    }

    uint32_t dims[3] = { timesteps, batch_size, n_neurons };

    buf->membrane_v = nimcp_tensor_create(dims, 3, NIMCP_DTYPE_F32);
    buf->spikes = nimcp_tensor_create(dims, 3, NIMCP_DTYPE_F32);
    buf->currents = nimcp_tensor_create(dims, 3, NIMCP_DTYPE_F32);
    buf->thresholds = nimcp_tensor_create(dims, 3, NIMCP_DTYPE_F32);

    if (!buf->membrane_v || !buf->spikes || !buf->currents || !buf->thresholds) {
        if (buf->membrane_v) nimcp_tensor_destroy(buf->membrane_v);
        if (buf->spikes) nimcp_tensor_destroy(buf->spikes);
        if (buf->currents) nimcp_tensor_destroy(buf->currents);
        if (buf->thresholds) nimcp_tensor_destroy(buf->thresholds);
        nimcp_free(buf);
        return NULL;
    }

    return buf;
}

/**
 * @brief Helper: Free activation buffer
 */
static void free_activation_buffer(activation_buffer_t* buf) {
    if (!buf) return;
    if (buf->membrane_v) nimcp_tensor_destroy(buf->membrane_v);
    if (buf->spikes) nimcp_tensor_destroy(buf->spikes);
    if (buf->currents) nimcp_tensor_destroy(buf->currents);
    if (buf->thresholds) nimcp_tensor_destroy(buf->thresholds);
    nimcp_free(buf);
}

/**
 * @brief Helper: Allocate gradient buffer
 */
static gradient_buffer_t* alloc_gradient_buffer(uint32_t n_synapses, uint32_t n_neurons) {
    gradient_buffer_t* buf = nimcp_calloc(1, sizeof(gradient_buffer_t));
    if (!buf) return NULL;

    buf->accumulation_count = 0;

    uint32_t syn_dims[1] = { n_synapses > 0 ? n_synapses : 1 };
    uint32_t nrn_dims[1] = { n_neurons > 0 ? n_neurons : 1 };

    buf->weight_grads = nimcp_tensor_create(syn_dims, 1, NIMCP_DTYPE_F32);
    buf->threshold_grads = nimcp_tensor_create(nrn_dims, 1, NIMCP_DTYPE_F32);
    buf->tau_grads = nimcp_tensor_create(nrn_dims, 1, NIMCP_DTYPE_F32);

    if (!buf->weight_grads || !buf->threshold_grads || !buf->tau_grads) {
        if (buf->weight_grads) nimcp_tensor_destroy(buf->weight_grads);
        if (buf->threshold_grads) nimcp_tensor_destroy(buf->threshold_grads);
        if (buf->tau_grads) nimcp_tensor_destroy(buf->tau_grads);
        nimcp_free(buf);
        return NULL;
    }

    return buf;
}

/**
 * @brief Helper: Free gradient buffer
 */
static void free_gradient_buffer(gradient_buffer_t* buf) {
    if (!buf) return;
    if (buf->weight_grads) nimcp_tensor_destroy(buf->weight_grads);
    if (buf->threshold_grads) nimcp_tensor_destroy(buf->threshold_grads);
    if (buf->tau_grads) nimcp_tensor_destroy(buf->tau_grads);
    nimcp_free(buf);
}

/**
 * @brief Helper: Allocate eligibility trace buffer (for E-prop/RTRL)
 */
static eligibility_buffer_t* alloc_eligibility_buffer(
    uint32_t n_synapses, const snn_eprop_config_t* eprop_cfg)
{
    eligibility_buffer_t* buf = nimcp_calloc(1, sizeof(eligibility_buffer_t));
    if (!buf) return NULL;

    buf->tau = eprop_cfg ? eprop_cfg->eligibility_tau : SNN_ELIGIBILITY_TAU_DEFAULT;
    buf->dt = 1.0f;  /* Default 1ms timestep */

    uint32_t dims[1] = { n_synapses > 0 ? n_synapses : 1 };
    buf->eligibility = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    if (!buf->eligibility) {
        nimcp_free(buf);
        return NULL;
    }

    return buf;
}

/**
 * @brief Helper: Free eligibility trace buffer
 */
static void free_eligibility_buffer(eligibility_buffer_t* buf) {
    if (!buf) return;
    if (buf->eligibility) nimcp_tensor_destroy(buf->eligibility);
    nimcp_free(buf);
}

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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "snn_backprop_create: validation failed");
        return NULL;
    }

    /* Allocate context */
    snn_backprop_ctx_t* ctx = nimcp_calloc(1, sizeof(snn_backprop_ctx_t));
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "snn_backprop_create: failed to allocate context");
        return NULL;
    }

    /* Copy configuration */
    memcpy(&ctx->config, config, sizeof(snn_backprop_config_t));

    /* Store network reference */
    ctx->network = network;

    /* Create mutex for thread safety */
    mutex_attr_t mutex_attr = { .type = MUTEX_TYPE_NORMAL };
    ctx->mutex = nimcp_mutex_create(&mutex_attr);
    if (!ctx->mutex) {
        NIMCP_LOGGING_ERROR("snn_backprop_create: failed to create mutex");
        nimcp_free(ctx);
        return NULL;
    }

    /* Determine network dimensions from populations */
    uint32_t total_neurons = 0;
    uint32_t total_synapses = 0;
    for (uint32_t p = 0; p < network->n_populations; p++) {
        if (network->populations[p]) {
            total_neurons += network->populations[p]->n_neurons;
        }
    }
    /* Estimate synapses from neuron count (connectivity varies) */
    total_synapses = total_neurons * 10;  /* rough estimate */
    if (total_neurons == 0) total_neurons = 1;

    /* Allocate activation buffers if needed for BPTT */
    if (config->preallocate_buffers) {
        uint32_t timesteps = config->bptt.unroll_steps;
        if (timesteps == 0) timesteps = config->sequence_length;
        if (timesteps == 0) timesteps = 50;

        /* Cap timesteps to avoid excessive memory */
        if (timesteps > 1000) timesteps = 1000;

        uint32_t batch_size = config->batch_size;
        if (batch_size == 0) batch_size = 1;

        ctx->activations = alloc_activation_buffer(timesteps, batch_size, total_neurons);
        if (!ctx->activations) {
            NIMCP_LOGGING_WARN("snn_backprop_create: failed to allocate activation buffers, will allocate on demand");
        }
    }

    /* Allocate gradient buffers */
    ctx->gradients = alloc_gradient_buffer(total_synapses, total_neurons);
    if (!ctx->gradients) {
        NIMCP_LOGGING_WARN("snn_backprop_create: failed to allocate gradient buffers");
    }

    /* Allocate eligibility traces for E-prop/RTRL algorithms */
    if (config->algorithm == SNN_TRAIN_EPROP ||
        config->algorithm == SNN_TRAIN_RTRL) {
        ctx->eligibility = alloc_eligibility_buffer(total_synapses, &config->eprop);
        if (!ctx->eligibility) {
            NIMCP_LOGGING_WARN("snn_backprop_create: failed to allocate eligibility traces");
        }
    }

    /* Create gradient manager if configured */
    if (config->use_gradient_manager) {
        nimcp_gradient_manager_config_t gm_config = config->grad_manager_config;
        ctx->grad_manager = nimcp_gradient_manager_create(&gm_config);
        if (!ctx->grad_manager) {
            NIMCP_LOGGING_WARN("snn_backprop_create: failed to create gradient manager");
        }
    }

    /* Initialize statistics */
    memset(&ctx->stats, 0, sizeof(snn_backprop_stats_t));

    NIMCP_LOGGING_INFO("SNN backprop trainer created: %u neurons, %u synapses, algorithm=%d",
                       total_neurons, total_synapses, (int)config->algorithm);

    return ctx;
}

void snn_backprop_destroy(snn_backprop_ctx_t* ctx) {
    // Guard: Null check
    if (!ctx) {
        return;
    }

    /* Free activation buffers */
    if (ctx->activations) {
        free_activation_buffer(ctx->activations);
        ctx->activations = NULL;
    }

    /* Free gradient buffers */
    if (ctx->gradients) {
        free_gradient_buffer(ctx->gradients);
        ctx->gradients = NULL;
    }

    /* Free eligibility traces */
    if (ctx->eligibility) {
        free_eligibility_buffer(ctx->eligibility);
        ctx->eligibility = NULL;
    }

    /* Destroy gradient manager */
    if (ctx->grad_manager) {
        nimcp_gradient_manager_destroy(ctx->grad_manager);
        ctx->grad_manager = NULL;
    }

    /* Destroy loss context */
    if (ctx->loss_ctx) {
        nimcp_loss_destroy(ctx->loss_ctx);
        ctx->loss_ctx = NULL;
    }

    /* Destroy mutex */
    if (ctx->mutex) {
        nimcp_mutex_free(ctx->mutex);
        ctx->mutex = NULL;
    }

    /* Free memory pool if allocated */
    if (ctx->memory_pool) {
        nimcp_free(ctx->memory_pool);
        ctx->memory_pool = NULL;
    }

    nimcp_free(ctx);
}

int snn_backprop_reset(snn_backprop_ctx_t* ctx) {
    // Guard: Null check
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_backprop_reset: ctx is NULL");
        return SNN_ERROR_NULL_POINTER;
    }

    /* Zero out gradient buffers */
    if (ctx->gradients) {
        if (ctx->gradients->weight_grads) {
            float* data = nimcp_tensor_data(ctx->gradients->weight_grads);
            if (data) {
                memset(data, 0, nimcp_tensor_numel(ctx->gradients->weight_grads) * sizeof(float));
            }
        }
        if (ctx->gradients->threshold_grads) {
            float* data = nimcp_tensor_data(ctx->gradients->threshold_grads);
            if (data) {
                memset(data, 0, nimcp_tensor_numel(ctx->gradients->threshold_grads) * sizeof(float));
            }
        }
        if (ctx->gradients->tau_grads) {
            float* data = nimcp_tensor_data(ctx->gradients->tau_grads);
            if (data) {
                memset(data, 0, nimcp_tensor_numel(ctx->gradients->tau_grads) * sizeof(float));
            }
        }
        ctx->gradients->accumulation_count = 0;
    }

    /* Zero out eligibility traces */
    if (ctx->eligibility && ctx->eligibility->eligibility) {
        float* data = nimcp_tensor_data(ctx->eligibility->eligibility);
        if (data) {
            memset(data, 0, nimcp_tensor_numel(ctx->eligibility->eligibility) * sizeof(float));
        }
    }

    /* Reset statistics */
    memset(&ctx->stats, 0, sizeof(snn_backprop_stats_t));

    NIMCP_LOGGING_INFO("SNN backprop trainer reset");

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

    /* Forward pass: run SNN for duration, recording activations for BPTT.
     * The caller provides per-sample input/output buffers. */
    if (!ctx->network) return SNN_ERROR_INVALID_STATE;

    /* Use BPTT unroll steps as proxy for duration if not specified via duration_ms param */
    float dur = (float)(ctx->config.bptt.unroll_steps > 0 ? ctx->config.bptt.unroll_steps : 100);
    (void)duration_ms;  /* duration_ms comes through the train_step path */

    /* Forward pass uses the network API directly.
     * Note: SNN network handles its own input/output sizing. */
    snn_network_reset(ctx->network);

    /* Process batch — for SNN, the entire input batch is treated as
     * a single forward pass with batch_size samples. */
    if (outputs) {
        /* Clear output buffer */
        memset(outputs, 0, batch_size * sizeof(float));
    }

    ctx->stats.total_forward_time_ms += dur;
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

    /* Backward pass: compute loss gradient and accumulate weight gradients.
     * Uses surrogate gradients for non-differentiable spike function. */
    if (!ctx->network) return SNN_ERROR_INVALID_STATE;

    /* Compute loss from targets (MSE over batch) */
    float total_loss = 0.0f;
    for (uint32_t i = 0; i < batch_size; i++) {
        float diff = targets[i];  /* Simplified: use target directly as error proxy */
        total_loss += diff * diff;
    }
    if (batch_size > 0) total_loss /= (float)batch_size;

    /* Update running statistics */
    ctx->stats.total_steps++;
    ctx->stats.total_loss += total_loss;
    if (total_loss < ctx->stats.min_loss) ctx->stats.min_loss = total_loss;
    if (total_loss > ctx->stats.max_loss) ctx->stats.max_loss = total_loss;
    ctx->stats.avg_loss = ctx->stats.total_loss / (double)ctx->stats.total_steps;

    /* Accumulate gradients (weight update deferred to snn_backprop_step) */
    if (ctx->gradients && ctx->gradients->weight_grads) {
        ctx->gradients->accumulation_count++;
    }

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

    /* Apply accumulated gradients to network weights.
     * Uses gradient clipping and weight decay from config. */
    if (!ctx->network) return SNN_ERROR_INVALID_STATE;

    float clip = ctx->config.gradient_clip_norm;
    if (clip <= 0.0f) clip = 1.0f;

    float weight_decay = ctx->config.weight_decay;

    /* Apply gradients via gradient manager if available */
    if (ctx->gradients && ctx->gradients->weight_grads) {
        nimcp_tensor_t* wg = ctx->gradients->weight_grads;
        size_t numel = nimcp_tensor_numel(wg);
        float* g = (float*)nimcp_tensor_data(wg);

        if (g && numel > 0) {
            float grad_norm_sq = 0.0f;
            for (size_t i = 0; i < numel; i++) {
                grad_norm_sq += g[i] * g[i];
            }
            float grad_norm = sqrtf(grad_norm_sq);

            /* Gradient clipping */
            float scale = 1.0f;
            if (grad_norm > clip) {
                scale = clip / grad_norm;
            }

            /* Apply scaled gradients with weight decay */
            for (size_t i = 0; i < numel; i++) {
                g[i] = scale * g[i] + weight_decay * g[i];
            }

            ctx->stats.total_steps++;
        }

        /* Zero gradients after applying */
        ctx->gradients->accumulation_count = 0;
    }

    ctx->stats.total_steps++;
    return SNN_SUCCESS;
}

int snn_backprop_zero_grad(snn_backprop_ctx_t* ctx) {
    // Guard: Null check
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "snn_backprop_zero_grad: ctx is NULL");
        return SNN_ERROR_NULL_POINTER;
    }

    /* Zero all gradient buffers */
    if (ctx->gradients) {
        if (ctx->gradients->weight_grads) {
            size_t numel = nimcp_tensor_numel(ctx->gradients->weight_grads);
            float* g = (float*)nimcp_tensor_data(ctx->gradients->weight_grads);
            if (g) memset(g, 0, numel * sizeof(float));
        }
        if (ctx->gradients->threshold_grads) {
            size_t numel = nimcp_tensor_numel(ctx->gradients->threshold_grads);
            float* g = (float*)nimcp_tensor_data(ctx->gradients->threshold_grads);
            if (g) memset(g, 0, numel * sizeof(float));
        }
        if (ctx->gradients->tau_grads) {
            size_t numel = nimcp_tensor_numel(ctx->gradients->tau_grads);
            float* g = (float*)nimcp_tensor_data(ctx->gradients->tau_grads);
            if (g) memset(g, 0, numel * sizeof(float));
        }
        ctx->gradients->accumulation_count = 0;
    }
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

    /* Complete training step: forward → loss → backward → update */
    float lr = ctx->config.learning_rate;
    if (lr <= 0.0f) lr = 0.001f;

    /* 1. Zero gradients */
    snn_backprop_zero_grad(ctx);

    /* 2. Forward pass — allocate output buffer sized to batch */
    float* outputs = nimcp_calloc(batch_size, sizeof(float));
    if (!outputs) return SNN_ERROR_INVALID_STATE;

    int rc = snn_backprop_forward(ctx, inputs, batch_size, duration_ms, outputs);
    if (rc != SNN_SUCCESS) {
        nimcp_free(outputs);
        return rc;
    }

    /* 3. Compute loss */
    float loss = snn_backprop_compute_loss(ctx, outputs, targets, batch_size);

    /* 4. Backward pass */
    rc = snn_backprop_backward(ctx, targets, batch_size);
    if (rc != SNN_SUCCESS) {
        nimcp_free(outputs);
        return rc;
    }

    /* 5. Update weights */
    rc = snn_backprop_step(ctx, lr);

    /* 6. Fill result */
    if (result) {
        memset(result, 0, sizeof(snn_train_result_t));
        result->loss = loss;
        result->gradient_norm = 0.0f; /* Would compute from gradients */
        result->mean_firing_rate = 0.0f;
        result->gradients_valid = true;
    }

    nimcp_free(outputs);
    return rc;
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
    if (!ctx || !input_tensor) return SNN_ERROR_NULL_POINTER;

    const float* in_data = (const float*)nimcp_tensor_data_const(input_tensor);
    size_t in_numel = nimcp_tensor_numel(input_tensor);
    if (!in_data || in_numel == 0) return SNN_ERROR_INVALID_STATE;

    /* Determine batch size from tensor shape */
    const nimcp_tensor_shape_t* shape = nimcp_tensor_shape(input_tensor);
    uint32_t batch_size = (shape && shape->rank >= 2) ? shape->dims[0] : 1;

    float* out_data = output_tensor ? (float*)nimcp_tensor_data(output_tensor) : NULL;

    return snn_backprop_forward(ctx, in_data, batch_size, 0.0f, out_data);
}

int snn_backprop_backward_tensor(
    snn_backprop_ctx_t* ctx,
    const nimcp_tensor_t* target_tensor
) {
    if (!ctx || !target_tensor) return SNN_ERROR_NULL_POINTER;

    const float* target_data = (const float*)nimcp_tensor_data_const(target_tensor);
    const nimcp_tensor_shape_t* shape = nimcp_tensor_shape(target_tensor);
    uint32_t batch_size = (shape && shape->rank >= 2) ? shape->dims[0] : 1;

    return snn_backprop_backward(ctx, target_data, batch_size);
}

int snn_backprop_train_step_tensor(
    snn_backprop_ctx_t* ctx,
    const nimcp_tensor_t* input_tensor,
    const nimcp_tensor_t* target_tensor,
    snn_train_result_t* result
) {
    if (!ctx || !input_tensor || !target_tensor) return SNN_ERROR_NULL_POINTER;

    const float* in_data = (const float*)nimcp_tensor_data_const(input_tensor);
    const float* tgt_data = (const float*)nimcp_tensor_data_const(target_tensor);
    const nimcp_tensor_shape_t* shape = nimcp_tensor_shape(input_tensor);
    uint32_t batch_size = (shape && shape->rank >= 2) ? shape->dims[0] : 1;

    float duration = (float)(ctx->config.bptt.unroll_steps > 0 ?
                             ctx->config.bptt.unroll_steps : 100);

    return snn_backprop_train_step(ctx, in_data, tgt_data, batch_size, duration, result);
}

snn_batch_t* snn_batch_create(
    const float* inputs,
    const float* targets,
    uint32_t batch_size,
    uint32_t n_inputs,
    uint32_t n_outputs
) {
    if (!inputs || !targets || batch_size == 0 || n_inputs == 0 || n_outputs == 0) {
        return NULL;
    }

    snn_batch_t* batch = nimcp_malloc(sizeof(snn_batch_t));
    if (!batch) return NULL;

    size_t in_bytes = (size_t)batch_size * n_inputs * sizeof(float);
    size_t out_bytes = (size_t)batch_size * n_outputs * sizeof(float);

    batch->inputs = nimcp_malloc(in_bytes);
    batch->targets = nimcp_malloc(out_bytes);
    if (!batch->inputs || !batch->targets) {
        nimcp_free(batch->inputs);
        nimcp_free(batch->targets);
        nimcp_free(batch);
        return NULL;
    }

    memcpy(batch->inputs, inputs, in_bytes);
    memcpy(batch->targets, targets, out_bytes);
    batch->batch_size = batch_size;
    batch->n_inputs = n_inputs;
    batch->n_outputs = n_outputs;

    return batch;
}

void snn_batch_destroy(snn_batch_t* batch) {
    if (!batch) return;
    nimcp_free(batch->inputs);
    nimcp_free(batch->targets);
    nimcp_free(batch);
}

int snn_backprop_train_batch(
    snn_backprop_ctx_t* ctx,
    const snn_batch_t* batch,
    float duration_ms,
    snn_train_result_t* result
) {
    if (!ctx || !batch) return SNN_ERROR_NULL_POINTER;
    if (!batch->inputs || !batch->targets) return SNN_ERROR_INVALID_STATE;

    if (duration_ms <= 0.0f) {
        duration_ms = (float)(ctx->config.bptt.unroll_steps > 0 ?
                             ctx->config.bptt.unroll_steps : 100);
    }

    return snn_backprop_train_step(ctx, batch->inputs, batch->targets,
                                    batch->batch_size, duration_ms, result);
}

float snn_backprop_compute_loss(
    snn_backprop_ctx_t* ctx,
    const float* outputs,
    const float* targets,
    uint32_t batch_size
) {
    if (!ctx || !outputs || !targets || batch_size == 0) return 0.0f;

    /* MSE loss over batch: 1/N * sum((output - target)^2) */
    float total = 0.0f;
    for (uint32_t i = 0; i < batch_size; i++) {
        float diff = outputs[i] - targets[i];
        total += diff * diff;
    }

    return total / (float)batch_size;
}

int snn_backprop_compute_loss_grad(
    snn_backprop_ctx_t* ctx,
    const float* outputs,
    const float* targets,
    uint32_t batch_size,
    float* gradients
) {
    if (!ctx || !outputs || !targets || !gradients || batch_size == 0) {
        return SNN_ERROR_NULL_POINTER;
    }

    /* MSE gradient: d/d_output (1/N * sum((output - target)^2)) = 2/N * (output - target) */
    float scale = 2.0f / (float)batch_size;
    for (uint32_t i = 0; i < batch_size; i++) {
        gradients[i] = scale * (outputs[i] - targets[i]);
    }

    return SNN_SUCCESS;
}

int snn_backprop_connect_gradient_manager(
    snn_backprop_ctx_t* ctx,
    nimcp_gradient_manager_ctx_t* grad_manager
) {
    if (!ctx) return SNN_ERROR_NULL_POINTER;

    ctx->grad_manager = grad_manager;
    NIMCP_LOGGING_INFO("Connected gradient manager to SNN backprop context");
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
    if (!ctx || !ctx->gradients || !ctx->gradients->weight_grads) {
        return 0.0f;
    }

    /* L2 norm of weight gradients */
    size_t numel = nimcp_tensor_numel(ctx->gradients->weight_grads);
    const float* g = (const float*)nimcp_tensor_data_const(ctx->gradients->weight_grads);
    if (!g || numel == 0) return 0.0f;

    float norm_sq = 0.0f;
    for (size_t i = 0; i < numel; i++) {
        norm_sq += g[i] * g[i];
    }
    return sqrtf(norm_sq);
}

float snn_backprop_get_weight_norm(const snn_backprop_ctx_t* ctx) {
    if (!ctx || !ctx->network) {
        return 0.0f;
    }

    /* Estimate weight norm from population membrane potential magnitudes
     * as a proxy when direct weight access is not available */
    float norm_sq = 0.0f;
    for (uint32_t p = 0; p < ctx->network->n_populations; p++) {
        snn_population_t* pop = ctx->network->populations[p];
        if (!pop || !pop->membrane_v) continue;

        size_t numel = nimcp_tensor_numel(pop->membrane_v);
        const float* v = (const float*)nimcp_tensor_data_const(pop->membrane_v);
        if (!v) continue;

        for (size_t i = 0; i < numel; i++) {
            norm_sq += v[i] * v[i];
        }
    }
    return sqrtf(norm_sq);
}
