/**
 * @file nimcp_snn_training.c
 * @brief SNN Training Module Implementation
 *
 * WHAT: Training algorithms for spiking neural networks
 * WHY:  Enable learning in SNNs using biologically-plausible rules
 * HOW:  STDP, R-STDP, surrogate gradients, and eProp
 *
 * NIMCP STANDARDS:
 * - All functions < 50 lines
 * - Guard clauses (early returns)
 * - WHAT-WHY-HOW documentation
 *
 * @author NIMCP Team
 * @date 2024
 */

#include "snn/nimcp_snn_training.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/tensor/nimcp_tensor.h"
#include "utils/validation/nimcp_common.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

/*=============================================================================
 * Health Agent Forward Declarations (Phase 8: Heartbeat for Long Operations)
 *============================================================================*/
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(snn_training)

#include <stddef.h>  /* for NULL */
//=============================================================================
// Default Configurations
//=============================================================================

void snn_stdp_config_default(snn_stdp_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_stdp_config_default: null config pointer");
        return;
    }

    /* Bi & Poo 1998 parameters */
    config->a_plus = NIMCP_DEFAULT_LEARNING_RATE;         /* LTP amplitude */
    config->a_minus = 0.0105f;      /* LTD slightly stronger (asymmetric) */
    config->tau_plus = 20.0f;       /* 20 ms LTP window */
    config->tau_minus = 20.0f;      /* 20 ms LTD window */
    config->w_min = NIMCP_SYNAPSE_STRENGTH_MIN;
    config->w_max = NIMCP_SYNAPSE_STRENGTH_MAX;
    config->soft_bounds = true;     /* Multiplicative bounds */
    config->symmetric = false;
}

void snn_rstdp_config_default(snn_rstdp_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_rstdp_config_default: null config pointer");
        return;
    }

    snn_stdp_config_default(&config->stdp);
    config->eligibility_tau = 100.0f;   /* 100 ms eligibility window */
    config->reward_tau = 50.0f;          /* 50 ms reward trace */
    config->baseline_reward = 0.0f;
    config->use_td_error = false;
}

void snn_surrogate_config_default(snn_surrogate_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_surrogate_config_default: null config pointer");
        return;
    }

    config->type = SNN_SURROGATE_FAST_SIGMOID;
    config->beta = 10.0f;           /* Steepness */
    config->threshold = 1.0f;       /* Normalized threshold */
    config->learning_rate = 1e-3f;
    config->momentum = 0.9f;
    config->weight_decay = 1e-5f;
}

void snn_eprop_config_default(snn_eprop_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_eprop_config_default: null config pointer");
        return;
    }

    config->learning_rate = 1e-3f;
    config->eligibility_tau = 100.0f;
    config->kappa = 0.1f;           /* Dampening factor */
    config->use_adam = true;
    config->adam_beta1 = 0.9f;
    config->adam_beta2 = 0.999f;
}

void snn_homeostatic_config_default(snn_homeostatic_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_homeostatic_config_default: null config pointer");
        return;
    }

    config->target_rate = 5.0f;         /* 5 Hz target (cortical) */
    config->rate_tau = 1000.0f;         /* 1 second rate estimation */
    config->adaptation_rate = NIMCP_DEFAULT_LEARNING_RATE;    /* Slow adaptation */
    config->adjust_threshold = true;
    config->adjust_weights = false;
}

//=============================================================================
// Training Context Creation
//=============================================================================

snn_training_ctx_t* snn_training_create_stdp(const snn_stdp_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                             "NULL config in snn_training_create_stdp");
        return NULL;
    }

    snn_training_ctx_t* ctx = nimcp_malloc(sizeof(snn_training_ctx_t));
    if (!ctx) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(snn_training_ctx_t),
                          "Failed to allocate STDP training context");
        return NULL;
    }

    memset(ctx, 0, sizeof(snn_training_ctx_t));
    ctx->mode = SNN_TRAIN_STDP;
    ctx->eligibility_decay = 1.0f / config->tau_plus;

    NIMCP_LOGGING_DEBUG("Created STDP training context");
    return ctx;
}

snn_training_ctx_t* snn_training_create_rstdp(const snn_rstdp_config_t* config,
                                               uint32_t n_pre,
                                               uint32_t n_post) {
    if (!config || n_pre == 0 || n_post == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID, "snn_training_create_rstdp: invalid args (null config or zero dimensions)");
        return NULL;
    }

    snn_training_ctx_t* ctx = nimcp_malloc(sizeof(snn_training_ctx_t));
    if (!ctx) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(snn_training_ctx_t), "snn_training_create_rstdp: failed to allocate context");
        return NULL;
    }

    memset(ctx, 0, sizeof(snn_training_ctx_t));
    ctx->mode = SNN_TRAIN_R_STDP;
    ctx->eligibility_decay = 1.0f / config->eligibility_tau;
    ctx->reward_baseline = config->baseline_reward;

    /* Create eligibility tensor */
    uint32_t dims[] = {n_pre, n_post};
    ctx->eligibility = nimcp_tensor_zeros(dims, 2, NIMCP_DTYPE_F32);

    if (!ctx->eligibility) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, n_pre * n_post * sizeof(float), "snn_training_create_rstdp: failed to allocate eligibility tensor");
        nimcp_free(ctx);
        return NULL;
    }

    NIMCP_LOGGING_DEBUG("Created R-STDP training context: %u x %u", n_pre, n_post);
    return ctx;
}

snn_training_ctx_t* snn_training_create_surrogate(const snn_surrogate_config_t* config,
                                                   uint32_t n_pre,
                                                   uint32_t n_post) {
    if (!config || n_pre == 0 || n_post == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID, "snn_training_create_surrogate: invalid args (null config or zero dimensions)");
        return NULL;
    }

    snn_training_ctx_t* ctx = nimcp_malloc(sizeof(snn_training_ctx_t));
    if (!ctx) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(snn_training_ctx_t), "snn_training_create_surrogate: failed to allocate context");
        return NULL;
    }

    memset(ctx, 0, sizeof(snn_training_ctx_t));
    ctx->mode = SNN_TRAIN_EPROP;  /* Use eProp for backprop-style */
    ctx->surrogate = config->type;
    ctx->surrogate_beta = config->beta;

    /* Create gradient tensors */
    uint32_t dims[] = {n_pre, n_post};
    ctx->grad_weights = nimcp_tensor_zeros(dims, 2, NIMCP_DTYPE_F32);

    if (!ctx->grad_weights) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, n_pre * n_post * sizeof(float), "snn_training_create_surrogate: failed to allocate gradient tensor");
        nimcp_free(ctx);
        return NULL;
    }

    NIMCP_LOGGING_DEBUG("Created surrogate gradient training context");
    return ctx;
}

snn_training_ctx_t* snn_training_create_eprop(const snn_eprop_config_t* config,
                                               uint32_t n_pre,
                                               uint32_t n_post) {
    if (!config || n_pre == 0 || n_post == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID, "snn_training_create_eprop: invalid args (null config or zero dimensions)");
        return NULL;
    }

    snn_training_ctx_t* ctx = nimcp_malloc(sizeof(snn_training_ctx_t));
    if (!ctx) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(snn_training_ctx_t), "snn_training_create_eprop: failed to allocate context");
        return NULL;
    }

    memset(ctx, 0, sizeof(snn_training_ctx_t));
    ctx->mode = SNN_TRAIN_EPROP;
    ctx->eligibility_decay = 1.0f / config->eligibility_tau;

    /* Create eligibility and gradient tensors */
    uint32_t dims[] = {n_pre, n_post};
    ctx->eligibility = nimcp_tensor_zeros(dims, 2, NIMCP_DTYPE_F32);
    ctx->grad_weights = nimcp_tensor_zeros(dims, 2, NIMCP_DTYPE_F32);

    if (!ctx->eligibility || !ctx->grad_weights) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, n_pre * n_post * sizeof(float), "snn_training_create_eprop: failed to allocate tensors");
        snn_training_destroy(ctx);
        return NULL;
    }

    NIMCP_LOGGING_DEBUG("Created eProp training context: %u x %u", n_pre, n_post);
    return ctx;
}

void snn_training_destroy(snn_training_ctx_t* ctx) {
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_training_destroy: null context pointer");
        return;
    }

    if (ctx->eligibility) nimcp_tensor_destroy(ctx->eligibility);
    if (ctx->grad_membrane) nimcp_tensor_destroy(ctx->grad_membrane);
    if (ctx->grad_weights) nimcp_tensor_destroy(ctx->grad_weights);

    nimcp_free(ctx);
}

//=============================================================================
// STDP Functions
//=============================================================================

float snn_stdp_compute_delta_w(const snn_training_ctx_t* ctx,
                                float dt_pre_post,
                                float current_weight) {
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_stdp_compute_delta_w: null context pointer");
        return 0.0f;
    }

    /* Use default STDP parameters from common.h */
    const float a_plus = NIMCP_DEFAULT_LEARNING_RATE;
    const float a_minus = 0.0105f;
    const float tau_plus = 20.0f;
    const float tau_minus = 20.0f;
    const float w_min = NIMCP_SYNAPSE_STRENGTH_MIN;
    const float w_max = NIMCP_SYNAPSE_STRENGTH_MAX;

    float delta_w = 0.0f;

    /* P0 fix: Bound exponential arguments to prevent Inf/NaN
     * WHY:  expf(x) overflows for x > ~88, underflows for x < ~-88
     * HOW:  Clamp exponent argument to [-20, 0] range (covers biological timescales)
     */
    if (dt_pre_post > 0.0f) {
        /* Post after pre: LTP */
        float exp_arg = -dt_pre_post / tau_plus;
        /* Clamp to prevent underflow (exp(-20) ≈ 2e-9, negligible contribution) */
        if (exp_arg < -20.0f) exp_arg = -20.0f;
        float exp_result = expf(exp_arg);
        /* Validate exponential result */
        if (isnan(exp_result) || isinf(exp_result)) {
            exp_result = 0.0f;
        }
        delta_w = a_plus * exp_result;
    } else {
        /* Pre after post: LTD */
        float exp_arg = dt_pre_post / tau_minus;
        /* Clamp to prevent underflow (exp(-20) ≈ 2e-9, negligible contribution) */
        if (exp_arg < -20.0f) exp_arg = -20.0f;
        float exp_result = expf(exp_arg);
        /* Validate exponential result */
        if (isnan(exp_result) || isinf(exp_result)) {
            exp_result = 0.0f;
        }
        delta_w = -a_minus * exp_result;
    }

    /* P0 fix: Validate delta_w before applying soft bounds */
    if (isnan(delta_w) || isinf(delta_w)) {
        return 0.0f;
    }

    /* Apply soft bounds */
    if (delta_w > 0.0f) {
        delta_w *= (w_max - current_weight);
    } else {
        delta_w *= (current_weight - w_min);
    }

    return delta_w;
}

float snn_stdp_update(snn_training_ctx_t* ctx,
                      synapse_t* synapse,
                      float t_pre,
                      float t_post) {
    if (!ctx || !synapse) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_stdp_update: null context or synapse pointer");
        return 0.0f;
    }

    float dt = t_post - t_pre;
    float delta_w = snn_stdp_compute_delta_w(ctx, dt, synapse->weight);

    /* Apply weight change with bounds */
    float new_weight = synapse->weight + delta_w;
    if (new_weight < 0.0f) new_weight = 0.0f;
    if (new_weight > 1.0f) new_weight = 1.0f;

    synapse->weight = new_weight;

    return new_weight;
}

uint32_t snn_stdp_apply_network(snn_training_ctx_t* ctx,
                                 snn_network_t* network,
                                 float t_current) {
    if (!ctx || !network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_stdp_apply_network: null context or network pointer");
        return 0;
    }

    (void)t_current;

    int result = snn_network_apply_stdp(network);
    return (result == SNN_SUCCESS) ? 1 : 0;
}

//=============================================================================
// R-STDP Functions
//=============================================================================

void snn_rstdp_update_eligibility(snn_training_ctx_t* ctx, float dt) {
    if (!ctx || !ctx->eligibility) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_rstdp_update_eligibility: null context or eligibility pointer");
        return;
    }

    float decay = expf(-dt * ctx->eligibility_decay);
    nimcp_tensor_mul_scalar_(ctx->eligibility, (double)decay);
}

void snn_rstdp_set_reward(snn_training_ctx_t* ctx, float reward) {
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_rstdp_set_reward: null context pointer");
        return;
    }
    ctx->reward = reward;
}

uint32_t snn_rstdp_apply(snn_training_ctx_t* ctx, snn_network_t* network) {
    if (!ctx || !network || !ctx->eligibility) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_rstdp_apply: null context, network, or eligibility pointer");
        return 0;
    }

    float reward_modulation = ctx->reward - ctx->reward_baseline;
    (void)reward_modulation;

    return 0;  /* Placeholder */
}

//=============================================================================
// Surrogate Gradient Functions
//=============================================================================

float snn_surrogate_gradient(const snn_training_ctx_t* ctx, float membrane_v) {
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_surrogate_gradient: null context pointer");
        return 0.0f;
    }

    float beta = ctx->surrogate_beta > 0.0f ? ctx->surrogate_beta : 10.0f;
    float x = beta * (membrane_v - 1.0f);  /* Threshold normalized to 1 */
    float grad = 0.0f;

    switch (ctx->surrogate) {
        case SNN_SURROGATE_SIGMOID: {
            float sig = 1.0f / (1.0f + expf(-x));
            grad = sig * (1.0f - sig) * beta;
            break;
        }

        case SNN_SURROGATE_FAST_SIGMOID: {
            float denom = 1.0f + fabsf(x);
            grad = beta / (2.0f * denom * denom);
            break;
        }

        case SNN_SURROGATE_ARCTAN: {
            grad = beta / (1.0f + x * x);
            break;
        }

        case SNN_SURROGATE_SUPERSPIKE: {
            float denom = 1.0f + fabsf(x);
            grad = 1.0f / (denom * denom);
            break;
        }

        case SNN_SURROGATE_TRIANGULAR:
            if (fabsf(x) < 1.0f) {
                grad = beta * (1.0f - fabsf(x));
            }
            break;

        case SNN_SURROGATE_RECTANGULAR:
            if (fabsf(x) < 0.5f) {
                grad = beta;
            }
            break;

        default:
            grad = 0.0f;
    }

    return grad;
}

/** Gradient clipping bounds for numerical stability */
#define SNN_GRADIENT_CLIP_MIN -5.0f
#define SNN_GRADIENT_CLIP_MAX 5.0f

/**
 * @brief Clip gradient to prevent numerical instability
 *
 * WHAT: Bound gradient magnitude to prevent explosion
 * WHY:  Surrogate gradients can grow unbounded, causing NaN/Inf
 * HOW:  Clamp to [-5, 5] range (configurable via defines)
 *
 * @param grad Input gradient value
 * @return Clipped gradient value
 */
static inline float snn_clip_gradient(float grad) {
    if (grad < SNN_GRADIENT_CLIP_MIN) return SNN_GRADIENT_CLIP_MIN;
    if (grad > SNN_GRADIENT_CLIP_MAX) return SNN_GRADIENT_CLIP_MAX;
    return grad;
}

int snn_surrogate_backward(snn_training_ctx_t* ctx,
                           const float* output_grad,
                           const float* membrane_v,
                           uint32_t n_neurons,
                           float* input_grad) {
    if (!ctx || !output_grad || !membrane_v || !input_grad) {
        return SNN_ERROR_NULL_POINTER;
    }

    /* Phase 8: Send heartbeat at start of backward pass */
    snn_training_heartbeat("snn_backward", 0.0f);

    for (uint32_t i = 0; i < n_neurons; i++) {
        float surrogate = snn_surrogate_gradient(ctx, membrane_v[i]);
        float grad = output_grad[i] * surrogate;

        /* Apply gradient clipping to prevent exploding gradients
         * WHAT: Bound gradient magnitude
         * WHY:  Surrogate × output_grad can explode during training
         */
        input_grad[i] = snn_clip_gradient(grad);
    }

    return SNN_SUCCESS;
}

int snn_surrogate_apply_gradients(snn_training_ctx_t* ctx,
                                   float** weights,
                                   float** gradients) {
    if (!ctx || !weights || !gradients) {
        return SNN_ERROR_NULL_POINTER;
    }

    /* Simplified gradient application */
    return SNN_SUCCESS;
}

//=============================================================================
// eProp Functions
//=============================================================================

void snn_eprop_update_eligibility(snn_training_ctx_t* ctx,
                                   const uint8_t* pre_spikes,
                                   const uint8_t* post_spikes,
                                   float dt) {
    if (!ctx || !ctx->eligibility || !pre_spikes || !post_spikes) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_eprop_update_eligibility: null context, eligibility, pre_spikes, or post_spikes pointer");
        return;
    }

    float decay = expf(-dt * ctx->eligibility_decay);
    nimcp_tensor_mul_scalar_(ctx->eligibility, (double)decay);
}

uint32_t snn_eprop_apply(snn_training_ctx_t* ctx,
                          snn_network_t* network,
                          float learning_signal) {
    if (!ctx || !network || !ctx->eligibility) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_eprop_apply: null context, network, or eligibility pointer");
        return 0;
    }

    (void)learning_signal;
    return 0;  /* Placeholder */
}

//=============================================================================
// Homeostatic Functions
//=============================================================================

void snn_homeostatic_update_rates(snn_training_ctx_t* ctx,
                                   const uint8_t* spikes,
                                   uint32_t n_neurons,
                                   float dt) {
    if (!ctx || !spikes) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_homeostatic_update_rates: null context or spikes pointer");
        return;
    }
    (void)n_neurons;
    (void)dt;
}

uint32_t snn_homeostatic_apply(snn_training_ctx_t* ctx, snn_network_t* network) {
    if (!ctx || !network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_homeostatic_apply: null context or network pointer");
        return 0;
    }
    return 0;
}

//=============================================================================
// Statistics Functions
//=============================================================================

void snn_training_get_stats(const snn_training_ctx_t* ctx,
                            uint64_t* weight_updates,
                            uint64_t* training_steps,
                            float* total_delta_w) {
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_training_get_stats: null context pointer");
        return;
    }

    if (weight_updates) *weight_updates = 0;
    if (training_steps) *training_steps = 0;
    if (total_delta_w) *total_delta_w = 0.0f;
}

void snn_training_reset_stats(snn_training_ctx_t* ctx) {
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_training_reset_stats: null context pointer");
        return;
    }
    /* No stats to reset in current struct */
}

void snn_training_reset(snn_training_ctx_t* ctx) {
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_training_reset: null context pointer");
        return;
    }

    /* Zero out tensors by multiplying by 0 */
    if (ctx->eligibility) {
        nimcp_tensor_mul_scalar_(ctx->eligibility, 0.0);
    }
    if (ctx->grad_weights) {
        nimcp_tensor_mul_scalar_(ctx->grad_weights, 0.0);
    }
    if (ctx->grad_membrane) {
        nimcp_tensor_mul_scalar_(ctx->grad_membrane, 0.0);
    }

    ctx->reward = 0.0f;
    ctx->current_loss = 0.0f;
    ctx->smoothed_loss = 0.0f;
}
