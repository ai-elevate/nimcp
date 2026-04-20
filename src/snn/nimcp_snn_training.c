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
#include "snn/nimcp_snn_network.h"
#include "snn/nimcp_snn_types.h"
#include "snn/nimcp_snn_synapse.h"  /* for snn_csr_storage_t, snn_csr_synapse_t */
#include "constants/nimcp_constants.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/tensor/nimcp_tensor.h"
#include "utils/tensor/nimcp_tensor_internal.h"
#include "utils/validation/nimcp_common.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "core/neuralnet/nimcp_sparse_synapse.h"
#include "core/neuralnet/nimcp_neuralnet.h"
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
    config->tau_plus = NIMCP_STDP_TAU_PLUS_MS;       /* 20 ms LTP window */
    config->tau_minus = NIMCP_STDP_TAU_MINUS_MS;      /* 20 ms LTD window */
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
    config->learning_rate = NIMCP_LEARNING_RATE_FINE;
    config->momentum = NIMCP_MOMENTUM_DEFAULT;
    config->weight_decay = 1e-5f;  /* Module-specific: lighter than default */
}

void snn_eprop_config_default(snn_eprop_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_eprop_config_default: null config pointer");
        return;
    }

    config->learning_rate = NIMCP_LEARNING_RATE_FINE;
    config->eligibility_tau = 100.0f;
    config->kappa = NIMCP_EXPLORATION_RATE_DEFAULT;           /* Dampening factor */
    config->use_adam = true;
    config->adam_beta1 = NIMCP_ADAM_BETA1_DEFAULT;
    config->adam_beta2 = NIMCP_ADAM_BETA2_DEFAULT;
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
    ctx->mode = SNN_TRAIN_SURROGATE;  /* Surrogate gradient backprop */
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

    /* Clamp non-finite weight to bounds */
    if (!isfinite(current_weight)) {
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

    /* R-STDP: weight update = lr * (reward - baseline) * eligibility_trace
     * Reward modulates accumulated STDP eligibility traces.
     * Positive reward strengthens recently co-active synapses. */
    float reward_modulation = ctx->reward - ctx->reward_baseline;

    /* Diagnostic: log every call so we can see if reward signal is reaching us */
    static uint64_t _rstdp_call_count = 0;
    _rstdp_call_count++;
    if ((_rstdp_call_count % 50) == 0) {
        NIMCP_LOGGING_INFO(
            "[R-STDP] call=%llu reward=%.4f baseline=%.4f mod=%.4f%s",
            (unsigned long long)_rstdp_call_count,
            ctx->reward, ctx->reward_baseline, reward_modulation,
            (fabsf(reward_modulation) < 1e-8f) ? " — SKIP (no signal)" : "");
    }

    if (fabsf(reward_modulation) < 1e-8f) return 0;

    /* R-STDP learning rate.
     * Halved 0.001 → 0.0005 after observing runaway saturation once
     * richer Claude/Phi-3 parent narrations started driving stronger
     * BERT embeddings through the network. Slower weight growth lets
     * homeostasis keep up; combined with the now wider emergency
     * scaling range in snn_homeostatic_apply (±10%), the network can
     * be held in biological firing regime. */
    float lr = 0.0005f;
    float scale = lr * reward_modulation;

    /* Apply eligibility-modulated update to network synapses */
    uint32_t updates = 0;
    if (network->neural_net) {
        uint32_t n_neurons = neural_network_get_num_neurons(network->neural_net);
        float* elig_data = (float*)ctx->eligibility->data;
        uint32_t elig_rows = ctx->eligibility->shape.dims[0];
        uint32_t elig_cols = ctx->eligibility->shape.dims[ctx->eligibility->shape.rank - 1];

        for (uint32_t i = 0; i < n_neurons; i++) {
            neuron_t* n = neural_network_get_neuron(network->neural_net, i);
            if (!n) continue;
            uint32_t syn_count = sparse_synapse_count(&n->outgoing);
            for (uint32_t s = 0; s < syn_count; s++) {
                synapse_handle_t* h = sparse_synapse_get(&n->outgoing, s);
                if (!h) continue;
                /* Look up eligibility for this synapse (i, target).
                 * Only synapses within the trace dimensions get reward-modulated
                 * updates; the rest rely on local STDP in the SNN step. */
                uint32_t j = h->target_neuron_id;
                if (i < elig_rows && j < elig_cols && elig_data) {
                    float e = elig_data[i * elig_cols + j];
                    if (fabsf(e) > 1e-10f) {
                        h->weight += scale * e;
                        /* Clamp weight to [-2, 2] */
                        if (h->weight > 2.0f) h->weight = 2.0f;
                        if (h->weight < -2.0f) h->weight = -2.0f;
                        updates++;
                    }
                }
            }
        }
    }

    /* === LIGHTWEIGHT CSR R-STDP ===
     * The hierarchical SNN uses lightweight CSR populations (n_hidden=0 in
     * neural_net), so the loop above updates zero synapses. For CSR pops,
     * apply a direct Hebbian rule modulated by reward using current spike
     * outputs. For each incoming CSR entry on each post-synaptic neuron:
     *   pre_spike  = source pop's spike_output[src_neuron]
     *   post_spike = this pop's spike_output[j]
     *   Δw = scale × (pre × post - decay × weight)
     * Only updates synapses where both pre and post fired recently —
     * effectively Hebbian + reward + light weight decay. */
    for (uint32_t p = 0; p < network->n_populations; p++) {
        snn_population_t* dst_pop = network->populations[p];
        if (!dst_pop || !dst_pop->lightweight || !dst_pop->incoming_csr) continue;
        if (!dst_pop->spike_output) continue;
        /* Warm-up gate: skip R-STDP on populations that haven't collected
         * enough firing-rate samples for homeostasis to engage. During the
         * first 100 SNN steps, R-STDP with no homeostatic brake can amplify
         * transient hyperactivity into runaway — classic Hebbian compounding.
         * Matches the same rate_samples < 100 gate in snn_homeostatic_apply. */
        if (dst_pop->rate_samples < 100) continue;
        snn_csr_storage_t* csr = dst_pop->incoming_csr;
        if (!csr->entries || csr->n_synapses == 0) continue;

        const float* post_spikes = (const float*)nimcp_tensor_data_const(dst_pop->spike_output);
        if (!post_spikes) continue;

        /* For each neuron j in this population, iterate its incoming synapses */
        for (uint32_t j = 0; j < csr->n_neurons; j++) {
            float post = post_spikes[j];
            if (post < 0.5f) continue;  /* Post didn't spike — no Hebbian update */

            uint32_t row_start = csr->row_ptr[j];
            uint32_t row_end = csr->row_ptr[j + 1];
            for (uint32_t e = row_start; e < row_end; e++) {
                snn_csr_synapse_t* entry = &csr->entries[e];
                /* Look up pre-synaptic spike from source population */
                if (entry->src_pop >= network->n_populations) continue;
                snn_population_t* src_pop = network->populations[entry->src_pop];
                if (!src_pop || !src_pop->spike_output) continue;
                if (entry->src_neuron >= src_pop->n_neurons) continue;

                const float* src_spikes = (const float*)nimcp_tensor_data_const(src_pop->spike_output);
                if (!src_spikes) continue;
                float pre = src_spikes[entry->src_neuron];
                if (pre < 0.5f) continue;  /* Pre didn't spike */

                /* Reward-modulated Hebbian + standard L2 weight decay.
                 *
                 * EXCITATORY (weight > 0): standard rule
                 *   Δw = scale - decay × w
                 *   + reward → weight grows, strengthening excitation
                 *
                 * INHIBITORY (weight < 0): SIGN-FLIPPED rule (inhibitory plasticity)
                 *   Δw = -scale - decay × w
                 *   + reward → weight becomes more negative, strengthening inhibition
                 *   This implements separate E/I plasticity: inhibitory synapses
                 *   that "helped" (by suppressing wrong output when reward came)
                 *   also strengthen, maintaining E/I balance.
                 *
                 * RATE-DEPENDENT INHIBITORY STRENGTHENING:
                 * If postsynaptic neuron is over-firing (rate_ema > 0.10),
                 * inhibitory synapses onto it strengthen proportionally —
                 * exactly what's needed to clamp runaway excitation at the
                 * local circuit level, not just via slow synaptic scaling. */
                float decay_rate = 1e-5f;
                float delta;
                if (entry->weight >= 0.0f) {
                    delta = scale - decay_rate * entry->weight;
                } else {
                    delta = -scale - decay_rate * entry->weight;
                    /* Extra inhibitory strengthening proportional to post rate */
                    if (dst_pop->neuron_rate_ema) {
                        float rate = dst_pop->neuron_rate_ema[j];
                        if (rate > 0.10f) {
                            /* Push weight more negative: additional 0.1 × (rate - 0.10)
                             * per apply. At rate=0.5, adds -0.04 per call. */
                            delta -= 0.1f * (rate - 0.10f);
                        }
                    }
                }
                entry->weight += delta;
                if (entry->weight > 2.0f) entry->weight = 2.0f;
                if (entry->weight < -4.0f) entry->weight = -4.0f;  /* inhib cap −4 */
                updates++;
            }
        }
    }

    /* Decay reward baseline toward recent rewards (EMA) */
    ctx->reward_baseline = 0.99f * ctx->reward_baseline + 0.01f * ctx->reward;

    if ((_rstdp_call_count % 50) == 0 && updates > 0) {
        NIMCP_LOGGING_INFO("[R-STDP] updated %u synapses (delta sign=%s)",
                           updates, (scale > 0) ? "+" : "-");
    }

    return updates;
}

//=============================================================================
// Surrogate Gradient Functions
//=============================================================================

static float snn_surrogate_gradient_local(const snn_training_ctx_t* ctx, float membrane_v) {
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

/**
 * @brief Public wrapper for snn_surrogate_gradient_local
 *
 * WHAT: Public API for computing surrogate gradient with snn_training_ctx_t
 * WHY:  Exposes functionality renamed from snn_surrogate_gradient() to avoid
 *       ODR conflict with snn_backprop's snn_surrogate_gradient()
 * HOW:  Delegates to static snn_surrogate_gradient_local()
 */
float snn_training_surrogate_gradient(const snn_training_ctx_t* ctx, float membrane_v) {
    return snn_surrogate_gradient_local(ctx, membrane_v);
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
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "surrogate_backward: training context is NULL");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!output_grad) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "surrogate_backward: gradient buffer is NULL (no forward pass?)");
        return SNN_ERROR_NULL_POINTER;
    }
    if (!membrane_v || !input_grad) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "surrogate_backward: membrane_v or input_grad is NULL");
        return SNN_ERROR_NULL_POINTER;
    }

    /* Phase 8: Send heartbeat at start of backward pass */
    snn_training_heartbeat("snn_backward", 0.0f);

    for (uint32_t i = 0; i < n_neurons; i++) {
        float surrogate = snn_surrogate_gradient_local(ctx, membrane_v[i]);
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

    /* eProp: weight update = lr * learning_signal * eligibility_trace
     * learning_signal is the error signal broadcast to all neurons.
     * eProp uses only local information (eligibility) + global signal. */
    float lr = 0.001f;  /* eProp learning rate (ctx has no lr field) */
    float scale = lr * learning_signal;
    if (fabsf(scale) < 1e-10f) return 0;

    uint32_t updates = 0;
    if (network->neural_net) {
        uint32_t n_neurons = neural_network_get_num_neurons(network->neural_net);
        float* elig_data = (float*)ctx->eligibility->data;
        uint32_t elig_rows = ctx->eligibility->shape.dims[0];
        uint32_t elig_cols = ctx->eligibility->shape.dims[ctx->eligibility->shape.rank - 1];

        for (uint32_t i = 0; i < n_neurons; i++) {
            neuron_t* n = neural_network_get_neuron(network->neural_net, i);
            if (!n) continue;
            uint32_t syn_count = sparse_synapse_count(&n->outgoing);
            for (uint32_t s = 0; s < syn_count; s++) {
                synapse_handle_t* h = sparse_synapse_get(&n->outgoing, s);
                if (!h) continue;
                uint32_t j = h->target_neuron_id;
                if (i < elig_rows && j < elig_cols && elig_data) {
                    float e = elig_data[i * elig_cols + j];
                    if (fabsf(e) > 1e-10f) {
                        h->weight -= scale * e;
                        if (h->weight > 2.0f) h->weight = 2.0f;
                        if (h->weight < -2.0f) h->weight = -2.0f;
                        updates++;
                    }
                }
            }
        }
    }

    return updates;
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

/* Synaptic scaling (Turrigiano 2008, biological homeostatic plasticity).
 * For each lightweight population, compare its EMA firing rate to the
 * biological target and multiplicatively scale all incoming CSR weights
 * to pull the rate back toward target. Prevents R-STDP runaway into
 * either silence or saturation.
 *
 * Called from the training loop every N learn_vector calls (not every
 * SNN step — too expensive to iterate 1.45B synapses that often). */
uint32_t snn_homeostatic_apply(snn_training_ctx_t* ctx, snn_network_t* network) {
    if (!ctx || !network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snn_homeostatic_apply: null context or network pointer");
        return 0;
    }

    const float target_rate   = 0.03f;   /* biological target: 3% firing */
    /* Scale bounds: tight [0.98, 1.02] always. The earlier emergency band
     * [0.90, 1.10] caused bang-bang oscillation — repeated 1.10× pushes
     * compounded the weights (1.10^20 ≈ 6.7×) so the SNN swung from silent
     * collapse (<1K spikes) to hyperactive (>1M spikes) within hundreds
     * of steps, degrading training. Tight bounds recover slower
     * (~30 applies from extreme) but don't overshoot. */
    const float rate_floor    = 1e-4f;   /* avoid divide-by-near-zero */
    const float w_cap         = 10.0f;   /* same cap as init path */

    uint32_t n_scaled = 0;
    uint32_t n_skipped_warmup = 0;

    for (uint32_t p = 0; p < network->n_populations; p++) {
        snn_population_t* pop = network->populations[p];
        if (!pop || !pop->lightweight || !pop->incoming_csr) continue;

        /* Warm-up gate: don't scale until EMA has enough samples to be
         * trustworthy. Prevents scaling on transient startup activity. */
        if (pop->rate_samples < 100) {
            n_skipped_warmup++;
            continue;
        }

        float cur_rate = pop->firing_rate_ema;
        if (cur_rate < rate_floor) cur_rate = rate_floor;

        /* Tight bounds always — see header comment. */
        const float min_scale = 0.98f;
        const float max_scale = 1.02f;

        /* Target / current gives the pull direction. */
        float scale = target_rate / cur_rate;
        if (scale < min_scale) scale = min_scale;
        if (scale > max_scale) scale = max_scale;

        /* Skip near-unity scaling — saves cycles when population is on
         * target. 0.5% deadband avoids thrashing around the setpoint. */
        if (scale > 0.995f && scale < 1.005f) continue;

        snn_csr_storage_t* csr = pop->incoming_csr;
        for (uint32_t e = 0; e < csr->n_synapses; e++) {
            float w = csr->entries[e].weight * scale;
            if (w > w_cap) w = w_cap;
            if (w < -w_cap) w = -w_cap;
            csr->entries[e].weight = w;
        }

        /* Per-neuron metabolic budget: cap sum(|w|) of incoming synapses.
         * Biology enforces a metabolic ceiling. Budget scales with fan_in —
         * cap = 0.8 × fan_in, so average |w| per synapse can be up to 0.8.
         * That's roughly 2-3× typical init |w| (0.15-0.3), giving R-STDP
         * room to grow synapses that matter without allowing a neuron to
         * accumulate arbitrarily strong total drive. */
        for (uint32_t j = 0; j < csr->n_neurons; j++) {
            uint32_t rs = csr->row_ptr[j];
            uint32_t re = csr->row_ptr[j + 1];
            uint32_t fan_in = re - rs;
            if (fan_in == 0) continue;
            float cap = 0.8f * (float)fan_in;
            float sum_abs = 0.0f;
            for (uint32_t e = rs; e < re; e++) sum_abs += fabsf(csr->entries[e].weight);
            if (sum_abs > cap) {
                float rescale = cap / sum_abs;
                for (uint32_t e = rs; e < re; e++) csr->entries[e].weight *= rescale;
            }
        }

        /* Keep the flat weights[] array (used by the GPU kernel) in sync
         * with entries[]. With persistent GPU residency (V2), we must also
         * push the updated host weights to the device copy, otherwise the
         * kernel reads stale values. */
        if (csr->gpu_ready && csr->weights) {
            for (uint32_t e = 0; e < csr->n_synapses; e++) {
                csr->weights[e] = csr->entries[e].weight;
            }
            if (csr->gpu_resident) {
                snn_csr_sync_weights_to_gpu(csr);
            }
        }

        NIMCP_LOGGING_INFO("homeostatic: pop %u '%s' rate=%.4f target=%.4f "
                           "scale=%.4f syns=%u",
                           pop->id, pop->name, pop->firing_rate_ema,
                           target_rate, scale, csr->n_synapses);
        n_scaled++;
    }

    if (n_skipped_warmup > 0) {
        NIMCP_LOGGING_DEBUG("homeostatic: %u pops still in warm-up",
                            n_skipped_warmup);
    }
    return n_scaled;
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
