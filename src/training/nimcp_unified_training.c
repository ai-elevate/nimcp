/**
 * @file nimcp_unified_training.c
 * @brief Unified Training Manager — lifecycle, registration, and training step
 *
 * WHAT: Single manager for all trainable networks with composite loss and shared infrastructure
 * WHY:  Eliminates per-network duplication, enables cross-network gradient flow
 * HOW:  Vtable-based polymorphism, topology-ordered forward/backward, unified gradient management
 *
 * @author NIMCP Development Team
 * @date 2026-03-11
 */

#include "training/nimcp_unified_training.h"
#include "middleware/training/nimcp_training_plasticity_bridge.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

/* Fractal + geometry + quantum integration */
#include "cognitive/memory/core/nimcp_fractal.h"
#include "physics/geometry/nimcp_information_geometry.h"
#include "utils/math/nimcp_complex_math.h"
#include "optimization/quantum_annealing/nimcp_quantum_annealing.h"

#include <math.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>

/* C1: GPU acceleration — forward declarations gated behind NIMCP_ENABLE_CUDA.
 * These are called only when mgr->gpu_ctx is non-NULL at runtime. */
#ifdef NIMCP_ENABLE_CUDA
extern bool nimcp_gpu_loss_mse(void* gpu_ctx, const float* pred, const float* target,
                                uint32_t n, float* loss);
extern bool nimcp_gpu_optim_adamw(void* gpu_ctx, float* params, float* grads, float* m, float* v,
                                    uint32_t n, float lr, float beta1, float beta2, float eps, float wd,
                                    uint64_t step);
#endif

//=============================================================================
// Default Configuration
//=============================================================================

void nimcp_utm_default_config(nimcp_unified_training_config_t* config) {
    if (!config) return;
    memset(config, 0, sizeof(nimcp_unified_training_config_t));

    config->optimizer_type = 5; /* NIMCP_OPTIMIZER_ADAM */
    config->learning_rate = 0.01f;
    config->weight_decay = 0.0f;

    config->anti_collapse.diversity_loss_weight = NIMCP_UTM_DEFAULT_DIVERSITY_WEIGHT;
    config->anti_collapse.diversity_buffer_size = NIMCP_UTM_DEFAULT_DIVERSITY_BUFFER;
    config->anti_collapse.use_gradient_normalization = true;
    config->anti_collapse.gradient_target_norm = NIMCP_UTM_DEFAULT_GRADIENT_TARGET;
    config->anti_collapse.gradient_clip_value = 5.0f;
    config->anti_collapse.adaptive_gradient_target = true;

    config->batch_size = 1;
    config->unified_optimizer = false;
    config->loss_type = 0; /* NIMCP_LOSS_MSE */
    config->use_composite_loss = true;
    config->enable_cross_network_gradients = false;

    config->lr_schedule.type = NIMCP_LR_SCHEDULE_COSINE;
    config->lr_schedule.warmup_steps = 1000;
    config->lr_schedule.total_steps = 100000;
    config->lr_schedule.min_lr_ratio = 0.01f;
    config->lr_schedule.step_decay_factor = 0.5f;
    config->lr_schedule.step_decay_interval = 10000;

    /* DFA health monitoring (default: enabled) */
    config->loss_history_size = 256;
    config->health_check_interval = 64;
    config->dfa_auto_adjust_lr = true;

    /* Quantum annealing (default: enabled) */
    config->enable_quantum_anneal = true;
    config->plateau_anneal_threshold = 3;

    /* Quantum Shannon bottleneck (default: enabled) */
    config->bottleneck_check_interval = 128;

    /* Natural gradient (default: enabled) */
    config->enable_natural_gradient = true;
    config->fisher_update_interval = 16;
    config->natural_grad_max_params = 4096;

    /* Manifold tracking (default: enabled) */
    config->enable_manifold_tracking = true;
}

//=============================================================================
// Lifecycle
//=============================================================================

nimcp_unified_training_manager_t* nimcp_utm_create(
    const nimcp_unified_training_config_t* config) {

    nimcp_unified_training_manager_t* mgr = (nimcp_unified_training_manager_t*)
        nimcp_calloc(1, sizeof(nimcp_unified_training_manager_t));
    if (!mgr) {
        NIMCP_LOGGING_ERROR("nimcp_utm_create: Failed to allocate manager");
        return NULL;
    }

    if (config) {
        mgr->config = *config;
    } else {
        nimcp_utm_default_config(&mgr->config);
    }

    mgr->current_lr = mgr->config.learning_rate;

    /* Initialize per-network anti-collapse state */
    for (uint32_t i = 0; i < NIMCP_UTM_MAX_NETWORKS; i++) {
        nimcp_anti_collapse_init(&mgr->anti_collapse[i], &mgr->config.anti_collapse);
    }

    /* Initialize AdamW optimizer state */
    mgr->adam_m = NULL;
    mgr->adam_v = NULL;
    mgr->adam_sizes = NULL;
    mgr->adam_num_groups = 0;
    mgr->adam_beta1_t = 1.0f;
    mgr->adam_beta2_t = 1.0f;

    /* Initialize fractal LR multipliers */
    for (uint32_t i = 0; i < NIMCP_UTM_MAX_NETWORKS; i++) {
        mgr->fractal_lr_multiplier[i] = 1.0f;
    }
    mgr->fractal_enabled = false;

    /* Initialize DFA health monitoring */
    mgr->loss_history_size = mgr->config.loss_history_size;
    mgr->health_check_interval = mgr->config.health_check_interval;
    mgr->dfa_auto_adjust_lr = mgr->config.dfa_auto_adjust_lr;
    if (mgr->loss_history_size > 0) {
        mgr->loss_history = (float*)nimcp_calloc(mgr->loss_history_size, sizeof(float));
    }
    mgr->training_health = NIMCP_TRAINING_HEALTH_UNKNOWN;

    /* Initialize quantum annealing */
    mgr->enable_quantum_anneal = mgr->config.enable_quantum_anneal;
    mgr->plateau_anneal_threshold = mgr->config.plateau_anneal_threshold;
    mgr->plateau_consecutive_count = 0;

    /* Initialize bottleneck detection */
    mgr->bottleneck_check_interval = mgr->config.bottleneck_check_interval;

    /* Initialize natural gradient */
    mgr->natural_gradient_enabled = mgr->config.enable_natural_gradient;
    mgr->fisher_update_interval = (mgr->config.fisher_update_interval > 0) ?
                                    mgr->config.fisher_update_interval : 16;
    mgr->natural_grad_max_params = (mgr->config.natural_grad_max_params > 0) ?
                                    mgr->config.natural_grad_max_params : 4096;
    /* Natural gradient handles are created lazily when param groups are known */

    /* Initialize manifold tracking */
    mgr->manifold_tracking_enabled = mgr->config.enable_manifold_tracking;
    /* Manifold handles are created lazily when output dims are known */

    /* Initialize phase coherence */
    mgr->cross_network_coherence = 0.0f;
    mgr->per_network_loss_history = NULL;

    NIMCP_LOGGING_INFO("Unified training manager created (lr=%.4f, diversity_w=%.2f, grad_norm=%s, "
                       "dfa=%s, quantum=%s, nat_grad=%s, manifold=%s)",
                       mgr->current_lr,
                       mgr->config.anti_collapse.diversity_loss_weight,
                       mgr->config.anti_collapse.use_gradient_normalization ? "normalize" : "clip",
                       mgr->loss_history_size > 0 ? "on" : "off",
                       mgr->enable_quantum_anneal ? "on" : "off",
                       mgr->natural_gradient_enabled ? "on" : "off",
                       mgr->manifold_tracking_enabled ? "on" : "off");

    return mgr;
}

void nimcp_utm_destroy(nimcp_unified_training_manager_t* mgr) {
    if (!mgr) return;

    /* Destroy adapters via their vtable destroy() */
    for (uint32_t i = 0; i < mgr->num_networks; i++) {
        if (mgr->networks[i].ops && mgr->networks[i].ops->destroy) {
            mgr->networks[i].ops->destroy(mgr->networks[i].ctx);
        }
        mgr->networks[i].ctx = NULL;
        mgr->networks[i].ops = NULL;
    }

    /* Destroy bridges */
    for (uint32_t i = 0; i < mgr->num_bridges; i++) {
        nimcp_cross_network_bridge_t* b = &mgr->bridges[i];
        nimcp_free(b->transform_weights);
        nimcp_free(b->transform_bias);
        nimcp_free(b->weight_grad);
        nimcp_free(b->bias_grad);
        nimcp_free(b->last_source_output);
        nimcp_free(b->last_target_input);
    }

    /* Destroy per-network anti-collapse state */
    for (uint32_t i = 0; i < NIMCP_UTM_MAX_NETWORKS; i++) {
        nimcp_anti_collapse_destroy(&mgr->anti_collapse[i]);
    }

    /* Free Adam optimizer state */
    if (mgr->adam_m) {
        for (uint32_t i = 0; i < mgr->adam_num_groups; i++) {
            nimcp_free(mgr->adam_m[i]);
            nimcp_free(mgr->adam_v[i]);
        }
        nimcp_free(mgr->adam_m);
        nimcp_free(mgr->adam_v);
        nimcp_free(mgr->adam_sizes);
    }

    /* Free DFA loss history */
    nimcp_free(mgr->loss_history);
    nimcp_free(mgr->per_network_loss_history);

    /* Destroy natural gradient handles */
    for (uint32_t i = 0; i < NIMCP_UTM_MAX_NETWORKS; i++) {
        if (mgr->natural_grad[i]) {
            nimcp_natural_grad_destroy(mgr->natural_grad[i]);
        }
        if (mgr->fisher[i]) {
            nimcp_fisher_destroy(mgr->fisher[i]);
        }
        if (mgr->output_manifold[i]) {
            nimcp_manifold_destroy(mgr->output_manifold[i]);
        }
    }

    nimcp_free(mgr);
    NIMCP_LOGGING_INFO("Unified training manager destroyed");
}

//=============================================================================
// Network Registration
//=============================================================================

int nimcp_utm_register_network(nimcp_unified_training_manager_t* mgr,
                               const nimcp_trainable_network_ops_t* ops,
                               void* ctx,
                               float loss_weight) {
    if (!mgr || !ops || !ctx) return -1;

    if (mgr->num_networks >= NIMCP_UTM_MAX_NETWORKS) {
        NIMCP_LOGGING_ERROR("nimcp_utm_register_network: Max networks (%d) reached",
                           NIMCP_UTM_MAX_NETWORKS);
        return -1;
    }

    uint32_t idx = mgr->num_networks;
    mgr->networks[idx].ops = ops;
    mgr->networks[idx].ctx = ctx;
    mgr->networks[idx].loss_weight = (loss_weight > 0.0f) ? loss_weight : 1.0f;
    mgr->networks[idx].enabled = true;
    mgr->num_networks++;

    NIMCP_LOGGING_INFO("Registered trainable network '%s' at index %u (weight=%.2f)",
                       ops->name ? ops->name : "unnamed", idx, mgr->networks[idx].loss_weight);

    return (int)idx;
}

int nimcp_utm_set_network_enabled(nimcp_unified_training_manager_t* mgr,
                                  uint32_t network_idx, bool enabled) {
    if (!mgr || network_idx >= mgr->num_networks) return -1;
    mgr->networks[network_idx].enabled = enabled;
    return 0;
}

void nimcp_utm_set_plasticity_bridge(nimcp_unified_training_manager_t* mgr,
                                      tpb_context_t* tpb) {
    if (!mgr) return;
    mgr->plasticity_bridge = tpb;
}

//=============================================================================
// Bridge API
//=============================================================================

int nimcp_utm_add_bridge(nimcp_unified_training_manager_t* mgr,
                         uint32_t source_idx, uint32_t target_idx,
                         nimcp_bridge_type_t type) {
    if (!mgr) return -1;
    if (source_idx >= mgr->num_networks || target_idx >= mgr->num_networks) return -1;
    if (mgr->num_bridges >= NIMCP_UTM_MAX_BRIDGES) return -1;

    uint32_t bidx = mgr->num_bridges;
    nimcp_cross_network_bridge_t* b = &mgr->bridges[bidx];
    memset(b, 0, sizeof(nimcp_cross_network_bridge_t));

    b->source_idx = source_idx;
    b->target_idx = target_idx;
    b->type = type;
    b->enabled = true;

    /* B7: Set configurable bridge parameter defaults */
    b->surrogate_beta = 1.0f;
    b->spike_rate_alpha = 0.3f;
    b->spike_gain = 5.0f;
    b->spike_threshold = 0.5f;

    /* Get dimensions from registered networks */
    nimcp_trainable_network_t* src = &mgr->networks[source_idx];
    nimcp_trainable_network_t* tgt = &mgr->networks[target_idx];

    b->source_dim = src->ops->get_output_dim(src->ctx);
    b->target_dim = tgt->ops->get_input_dim(tgt->ctx);

    if (type == NIMCP_BRIDGE_LINEAR && b->source_dim > 0 && b->target_dim > 0) {
        size_t w_size = (size_t)b->target_dim * b->source_dim;
        b->transform_weights = (float*)nimcp_calloc(w_size, sizeof(float));
        b->weight_grad = (float*)nimcp_calloc(w_size, sizeof(float));
        b->transform_bias = (float*)nimcp_calloc(b->target_dim, sizeof(float));
        b->bias_grad = (float*)nimcp_calloc(b->target_dim, sizeof(float));

        if (!b->transform_weights || !b->weight_grad) {
            NIMCP_LOGGING_ERROR("nimcp_utm_add_bridge: Failed to allocate transform");
            return -1;
        }

        /* Xavier initialization */
        float scale = sqrtf(2.0f / (float)(b->source_dim + b->target_dim));
        for (size_t i = 0; i < w_size; i++) {
            b->transform_weights[i] = scale * ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        }
    }

    b->last_source_output = (float*)nimcp_calloc(b->source_dim, sizeof(float));
    b->last_target_input = (float*)nimcp_calloc(b->target_dim, sizeof(float));

    mgr->num_bridges++;

    NIMCP_LOGGING_INFO("Added %s bridge: network[%u] (%ux) -> network[%u] (%ux)",
                       type == NIMCP_BRIDGE_LINEAR ? "linear" :
                       type == NIMCP_BRIDGE_IDENTITY ? "identity" : "encoded",
                       source_idx, b->source_dim, target_idx, b->target_dim);

    return (int)bidx;
}

//=============================================================================
// Bridge Forward/Backward (internal)
//=============================================================================

static int bridge_forward(nimcp_cross_network_bridge_t* b,
                          const float* source_output,
                          float* target_input) {
    if (!b || !source_output || !target_input) return -1;

    /* Cache source output for backward */
    memcpy(b->last_source_output, source_output, b->source_dim * sizeof(float));

    switch (b->type) {
        case NIMCP_BRIDGE_IDENTITY:
            /* Direct copy (dims must match) */
            {
                uint32_t copy_dim = (b->source_dim < b->target_dim) ?
                                     b->source_dim : b->target_dim;
                memcpy(target_input, source_output, copy_dim * sizeof(float));
                /* Zero-pad if target is larger */
                for (uint32_t i = copy_dim; i < b->target_dim; i++) {
                    target_input[i] = 0.0f;
                }
            }
            break;

        case NIMCP_BRIDGE_LINEAR:
            if (!b->transform_weights) return -1; /* B8: null guard */
            /* target = W @ source + bias */
            for (uint32_t t = 0; t < b->target_dim; t++) {
                float sum = b->transform_bias ? b->transform_bias[t] : 0.0f;
                for (uint32_t s = 0; s < b->source_dim; s++) {
                    sum += b->transform_weights[t * b->source_dim + s] * source_output[s];
                }
                target_input[t] = sum;
            }
            break;

        case NIMCP_BRIDGE_RATE_TO_SPIKE:
            bridge_rate_to_spike_forward(b, source_output, target_input);
            break;

        case NIMCP_BRIDGE_SPIKE_TO_RATE:
            bridge_spike_to_rate_forward(b, source_output, target_input);
            break;

        case NIMCP_BRIDGE_CONTINUOUS_TO_SPIKE:
            bridge_continuous_to_spike_forward(b, source_output, target_input);
            break;
    }

    /* Cache transformed output */
    memcpy(b->last_target_input, target_input, b->target_dim * sizeof(float));
    return 0;
}

static int bridge_backward(nimcp_cross_network_bridge_t* b,
                            const float* dl_dtarget_input,
                            float* dl_dsource_output) {
    if (!b || !dl_dtarget_input) return -1;

    switch (b->type) {
        case NIMCP_BRIDGE_IDENTITY:
            if (dl_dsource_output) {
                uint32_t copy_dim = (b->source_dim < b->target_dim) ?
                                     b->source_dim : b->target_dim;
                memcpy(dl_dsource_output, dl_dtarget_input, copy_dim * sizeof(float));
            }
            break;

        case NIMCP_BRIDGE_LINEAR:
            /* Gradient w.r.t. weights: dL/dW = dL/dtarget @ source.T */
            if (b->weight_grad) {
                for (uint32_t t = 0; t < b->target_dim; t++) {
                    for (uint32_t s = 0; s < b->source_dim; s++) {
                        b->weight_grad[t * b->source_dim + s] +=
                            dl_dtarget_input[t] * b->last_source_output[s];
                    }
                }
            }
            /* Gradient w.r.t. bias */
            if (b->bias_grad) {
                for (uint32_t t = 0; t < b->target_dim; t++) {
                    b->bias_grad[t] += dl_dtarget_input[t];
                }
            }
            /* Gradient w.r.t. source output: dL/dsource = W.T @ dL/dtarget */
            if (dl_dsource_output) {
                for (uint32_t s = 0; s < b->source_dim; s++) {
                    float sum = 0.0f;
                    for (uint32_t t = 0; t < b->target_dim; t++) {
                        sum += b->transform_weights[t * b->source_dim + s] * dl_dtarget_input[t];
                    }
                    dl_dsource_output[s] = sum;
                }
            }
            break;

        case NIMCP_BRIDGE_RATE_TO_SPIKE:
            bridge_rate_to_spike_backward(b, dl_dtarget_input, dl_dsource_output);
            break;

        case NIMCP_BRIDGE_SPIKE_TO_RATE:
            bridge_spike_to_rate_backward(b, dl_dtarget_input, dl_dsource_output);
            break;

        case NIMCP_BRIDGE_CONTINUOUS_TO_SPIKE:
            bridge_continuous_to_spike_backward(b, dl_dtarget_input, dl_dsource_output);
            break;
    }

    return 0;
}

//=============================================================================
// Learning Rate Schedule
//=============================================================================

float nimcp_utm_get_scheduled_lr(const nimcp_unified_training_manager_t* mgr) {
    if (!mgr) return 0.01f;

    float base_lr = mgr->config.learning_rate;
    const nimcp_lr_schedule_config_t* sched = &mgr->config.lr_schedule;
    uint64_t step = mgr->step_count;

    /* Linear warmup */
    if (sched->warmup_steps > 0 && step < sched->warmup_steps) {
        float warmup_ratio = (float)(step + 1) / (float)sched->warmup_steps;
        return base_lr * warmup_ratio;
    }

    float min_lr = base_lr * sched->min_lr_ratio;

    switch (sched->type) {
        case NIMCP_LR_SCHEDULE_COSINE: {
            if (sched->total_steps <= sched->warmup_steps) return base_lr;
            uint64_t decay_steps = sched->total_steps - sched->warmup_steps;
            uint64_t decay_step = step - sched->warmup_steps;
            if (decay_step >= decay_steps) return min_lr;
            float progress = (float)decay_step / (float)decay_steps;
            /* Cosine annealing: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress)) */
            return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cosf((float)M_PI * progress));
        }
        case NIMCP_LR_SCHEDULE_STEP: {
            if (sched->step_decay_interval == 0) return base_lr;
            uint64_t num_decays = step / sched->step_decay_interval;
            float lr = base_lr;
            for (uint64_t d = 0; d < num_decays && d < 20; d++) {
                lr *= sched->step_decay_factor;
            }
            return fmaxf(lr, min_lr);
        }
        default:
            return base_lr;
    }
}

//=============================================================================
// Unified Training Step
//=============================================================================

int nimcp_utm_step(nimcp_unified_training_manager_t* mgr,
                   const float* input, uint32_t input_dim,
                   const float* target, uint32_t target_dim,
                   nimcp_utm_step_result_t* result) {
    if (!mgr || !input || !target) return -1;
    if (mgr->num_networks == 0) return -1;

    nimcp_utm_step_result_t local_result;
    memset(&local_result, 0, sizeof(local_result));

    /* ------------------------------------------------------------------ */
    /* Step 1: Zero all gradients (only on first sample of batch)         */
    /* ------------------------------------------------------------------ */
    uint32_t effective_batch = (mgr->config.batch_size > 0) ? mgr->config.batch_size : 1;
    bool is_first_in_batch = (mgr->batch_accumulation_count == 0);

    if (is_first_in_batch) {
        for (uint32_t i = 0; i < mgr->num_networks; i++) {
            if (!mgr->networks[i].enabled) continue;
            const nimcp_trainable_network_ops_t* ops = mgr->networks[i].ops;
            if (ops->zero_grad) {
                ops->zero_grad(mgr->networks[i].ctx);
            }
        }

        /* Zero bridge gradients */
        for (uint32_t i = 0; i < mgr->num_bridges; i++) {
            nimcp_cross_network_bridge_t* b = &mgr->bridges[i];
            if (b->weight_grad) memset(b->weight_grad, 0,
                (size_t)b->target_dim * b->source_dim * sizeof(float));
            if (b->bias_grad) memset(b->bias_grad, 0,
                b->target_dim * sizeof(float));
        }
    }

    /* ------------------------------------------------------------------ */
    /* Step 2: Forward pass through networks (topology order)             */
    /* ------------------------------------------------------------------ */
    /* Allocate per-network output buffers */
    float* net_outputs[NIMCP_UTM_MAX_NETWORKS];
    uint32_t net_output_dims[NIMCP_UTM_MAX_NETWORKS];
    memset(net_outputs, 0, sizeof(net_outputs));
    memset(net_output_dims, 0, sizeof(net_output_dims));

    for (uint32_t i = 0; i < mgr->num_networks; i++) {
        if (!mgr->networks[i].enabled) continue;

        nimcp_trainable_network_t* net = &mgr->networks[i];
        uint32_t out_dim = net->ops->get_output_dim(net->ctx);
        net_output_dims[i] = out_dim;

        net_outputs[i] = (float*)nimcp_calloc(out_dim, sizeof(float));
        if (!net_outputs[i]) goto cleanup;

        /* Determine input: either from a bridge or from the raw input */
        const float* net_input = input;
        uint32_t net_input_dim = input_dim;

        /* Check if any bridge feeds into this network (use first matching) */
        for (uint32_t b = 0; b < mgr->num_bridges; b++) {
            if (mgr->bridges[b].target_idx == i && mgr->bridges[b].enabled) {
                uint32_t src = mgr->bridges[b].source_idx;
                if (net_outputs[src]) {
                    /* Bridge from source output to this network's input */
                    uint32_t bridged_dim = net->ops->get_input_dim(net->ctx);
                    float* bridged_input = (float*)nimcp_calloc(bridged_dim, sizeof(float));
                    if (bridged_input) {
                        bridge_forward(&mgr->bridges[b], net_outputs[src], bridged_input);
                        net_input = bridged_input;
                        net_input_dim = bridged_dim;
                        break; /* A1: Use first bridge only — prevents memory leak */
                    }
                }
            }
        }

        int rc = net->ops->forward(net->ctx, net_input, net_input_dim,
                                   net_outputs[i], out_dim);

        /* Free bridged input if we allocated one */
        if (net_input != input) {
            nimcp_free((void*)net_input);
        }

        if (rc != 0) {
            NIMCP_LOGGING_WARN("utm_step: Forward failed for network '%s' (rc=%d)",
                              net->ops->name, rc);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Step 3: Compute composite loss                                     */
    /* ------------------------------------------------------------------ */
    float composite_loss = 0.0f;
    float total_weight = 0.0f;

    /* Per-network loss against target (B2: MSE/MAE/cross-entropy) */
    uint32_t loss_type = mgr->config.loss_type;
    for (uint32_t i = 0; i < mgr->num_networks; i++) {
        if (!mgr->networks[i].enabled || !net_outputs[i]) continue;
        nimcp_trainable_network_t* net = &mgr->networks[i];

        uint32_t cmp_dim = (net_output_dims[i] < target_dim) ?
                            net_output_dims[i] : target_dim;
        float loss_val = 0.0f;
        float n = (float)(cmp_dim > 0 ? cmp_dim : 1);
        switch (loss_type) {
            case 1: /* MAE: Σ|out-tgt|/n */
                for (uint32_t j = 0; j < cmp_dim; j++) {
                    loss_val += fabsf(net_outputs[i][j] - target[j]);
                }
                loss_val /= n;
                break;
            case 2: /* Cross-entropy: -Σ tgt*log(max(out,ε))/n */
                for (uint32_t j = 0; j < cmp_dim; j++) {
                    float o = net_outputs[i][j];
                    if (o < 1e-7f) o = 1e-7f;
                    loss_val -= target[j] * logf(o);
                }
                loss_val /= n;
                break;
            default: /* MSE: Σ(out-tgt)²/n */
                for (uint32_t j = 0; j < cmp_dim; j++) {
                    float diff = net_outputs[i][j] - target[j];
                    loss_val += diff * diff;
                }
                loss_val /= n;
                break;
        }

        /* Auxiliary loss from the network (e.g., spike regularization) */
        float aux = 0.0f;
        if (net->ops->compute_auxiliary_loss) {
            aux = net->ops->compute_auxiliary_loss(net->ctx);
        }

        float net_loss = loss_val + aux;
        local_result.per_network_loss[i] = net_loss;
        composite_loss += net_loss * net->loss_weight;
        total_weight += net->loss_weight;
    }

    if (total_weight > 0.0f) {
        composite_loss /= total_weight;
    }
    local_result.composite_loss = composite_loss;

    /* ------------------------------------------------------------------ */
    /* Step 4: Backward pass (reverse order, with bridge gradient flow)   */
    /* ------------------------------------------------------------------ */

    /* Phase 5: Suppress biological plasticity during backprop */
    if (mgr->plasticity_bridge) {
        nimcp_tpb_set_backprop_active(mgr->plasticity_bridge, true);
    }

    /* Store per-network input gradients for cross-network gradient flow.
     * Backward runs in reverse order so downstream networks compute their
     * dl_dinput before upstream networks need it for bridge backward. */
    float* dl_dinputs[NIMCP_UTM_MAX_NETWORKS];
    memset(dl_dinputs, 0, sizeof(dl_dinputs));

    for (int i = (int)mgr->num_networks - 1; i >= 0; i--) {
        if (!mgr->networks[i].enabled || !net_outputs[i]) continue;
        nimcp_trainable_network_t* net = &mgr->networks[i];

        /* Compute dL/dOutput for this network (B2: loss-type-aware gradient) */
        uint32_t out_dim = net_output_dims[i];
        float* dl_dout = (float*)nimcp_calloc(out_dim, sizeof(float));
        if (!dl_dout) continue;

        uint32_t cmp_dim = (out_dim < target_dim) ? out_dim : target_dim;
        float n_cmp = (float)(cmp_dim > 0 ? cmp_dim : 1);
        switch (loss_type) {
            case 1: /* MAE gradient: sign(out-tgt)/n */
                for (uint32_t j = 0; j < cmp_dim; j++) {
                    float diff = net_outputs[i][j] - target[j];
                    dl_dout[j] = (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f)) / n_cmp;
                }
                break;
            case 2: /* Cross-entropy gradient: -tgt/max(out,ε)/n */
                for (uint32_t j = 0; j < cmp_dim; j++) {
                    float o = net_outputs[i][j];
                    if (o < 1e-7f) o = 1e-7f;
                    dl_dout[j] = -target[j] / o / n_cmp;
                }
                break;
            default: /* MSE gradient: 2(y-target)/n */
                for (uint32_t j = 0; j < cmp_dim; j++) {
                    dl_dout[j] = 2.0f * (net_outputs[i][j] - target[j]) / n_cmp;
                }
                break;
        }

        /* Add gradient from any downstream bridge using the target network's
         * input gradient (dl_dinputs[tgt]) instead of dl_dout approximation */
        if (mgr->config.enable_cross_network_gradients) {
            for (uint32_t b = 0; b < mgr->num_bridges; b++) {
                if (mgr->bridges[b].source_idx == (uint32_t)i && mgr->bridges[b].enabled) {
                    uint32_t tgt = mgr->bridges[b].target_idx;
                    /* Use target network's input gradient if available */
                    const float* tgt_grad = dl_dinputs[tgt];
                    if (tgt_grad) {
                        float* bridge_grad = (float*)nimcp_calloc(out_dim, sizeof(float));
                        if (bridge_grad) {
                            bridge_backward(&mgr->bridges[b], tgt_grad, bridge_grad);
                            for (uint32_t j = 0; j < out_dim; j++) {
                                dl_dout[j] += bridge_grad[j];
                            }
                            nimcp_free(bridge_grad);
                        }
                    }
                }
            }
        }

        /* Add diversity loss gradient (per-network anti-collapse) */
        float div_loss = nimcp_anti_collapse_diversity_loss(
            &mgr->anti_collapse[i], net_outputs[i], dl_dout, out_dim);
        local_result.diversity_loss += div_loss;

        /* Backward through the network */
        uint32_t in_dim = net->ops->get_input_dim(net->ctx);
        float* dl_din = (float*)nimcp_calloc(in_dim, sizeof(float));
        if (!dl_din) { nimcp_free(dl_dout); dl_dinputs[i] = NULL; continue; }

        net->ops->backward(net->ctx, dl_dout, out_dim, dl_din, in_dim);

        /* Store input gradient for upstream bridge backward (don't free yet) */
        dl_dinputs[i] = dl_din;

        nimcp_free(dl_dout);
    }

    /* Phase 5: Re-enable biological plasticity after backprop */
    if (mgr->plasticity_bridge) {
        nimcp_tpb_set_backprop_active(mgr->plasticity_bridge, false);
    }

    /* Free all stored input gradients */
    for (uint32_t i = 0; i < mgr->num_networks; i++) {
        nimcp_free(dl_dinputs[i]);
    }

    /* ------------------------------------------------------------------ */
    /* Step 4.5: Mini-batch accumulation check                            */
    /* ------------------------------------------------------------------ */
    mgr->batch_accumulation_count++;
    bool batch_complete = (mgr->batch_accumulation_count >= effective_batch);

    if (!batch_complete) {
        /* Accumulating — skip gradient normalization and optimizer step */
        local_result.composite_loss = composite_loss;
        local_result.step = mgr->step_count;
        if (result) *result = local_result;
        goto cleanup;
    }

    /* B4: Track which networks actually ran backward successfully */
    bool backward_ran[NIMCP_UTM_MAX_NETWORKS];
    memset(backward_ran, 0, sizeof(backward_ran));
    for (uint32_t i = 0; i < mgr->num_networks; i++) {
        if (mgr->networks[i].enabled && net_outputs[i]) {
            backward_ran[i] = true;
        }
    }

    /* Batch complete — divide accumulated gradients by batch_size */
    if (effective_batch > 1) {
        float inv_batch = 1.0f / (float)effective_batch;
        for (uint32_t i = 0; i < mgr->num_networks; i++) {
            if (!mgr->networks[i].enabled || !backward_ran[i]) continue;
            nimcp_utm_param_group_t* groups = NULL;
            uint32_t num_groups = 0;
            if (mgr->networks[i].ops->get_param_groups &&
                mgr->networks[i].ops->get_param_groups(mgr->networks[i].ctx, &groups, &num_groups) == 0) {
                for (uint32_t g = 0; g < num_groups; g++) {
                    if (groups[g].gradients && groups[g].count > 0) {
                        for (size_t j = 0; j < groups[g].count; j++) {
                            groups[g].gradients[j] *= inv_batch;
                        }
                    }
                }
                nimcp_free(groups);
            }
        }
    }
    mgr->batch_accumulation_count = 0;

    /* ------------------------------------------------------------------ */
    /* Step 5: Gradient normalization / clipping (unified, single pass)   */
    /* ------------------------------------------------------------------ */
    {
        /* B3: Per-network gradient normalization, then lighter global safety clip */
        for (uint32_t i = 0; i < mgr->num_networks; i++) {
            if (!mgr->networks[i].enabled) continue;
            nimcp_trainable_network_t* net = &mgr->networks[i];

            float* net_grads[32];
            size_t net_sizes[32];
            uint32_t net_arrays = 0;

            nimcp_utm_param_group_t* groups = NULL;
            uint32_t num_groups = 0;
            if (net->ops->get_param_groups &&
                net->ops->get_param_groups(net->ctx, &groups, &num_groups) == 0) {
                for (uint32_t g = 0; g < num_groups && net_arrays < 32; g++) {
                    if (groups[g].gradients && groups[g].count > 0) {
                        net_grads[net_arrays] = groups[g].gradients;
                        net_sizes[net_arrays] = groups[g].count;
                        net_arrays++;
                    }
                }
                nimcp_free(groups);
            }

            if (net_arrays > 0) {
                nimcp_anti_collapse_normalize_gradients(
                    &mgr->anti_collapse[i], net_grads, net_sizes, net_arrays);
            }
        }

        /* Global safety clip at 10x target norm using anti_collapse[0] */
        float* all_grads[NIMCP_UTM_MAX_NETWORKS * 32];
        size_t all_sizes[NIMCP_UTM_MAX_NETWORKS * 32];
        uint32_t total_arrays = 0;

        for (uint32_t i = 0; i < mgr->num_networks; i++) {
            if (!mgr->networks[i].enabled) continue;
            nimcp_trainable_network_t* net = &mgr->networks[i];
            nimcp_utm_param_group_t* groups = NULL;
            uint32_t num_groups = 0;
            if (net->ops->get_param_groups &&
                net->ops->get_param_groups(net->ctx, &groups, &num_groups) == 0) {
                for (uint32_t g = 0; g < num_groups && total_arrays < NIMCP_UTM_MAX_NETWORKS * 32; g++) {
                    if (groups[g].gradients && groups[g].count > 0) {
                        all_grads[total_arrays] = groups[g].gradients;
                        all_sizes[total_arrays] = groups[g].count;
                        total_arrays++;
                    }
                }
                nimcp_free(groups);
            }
        }

        /* Include bridge weight gradients */
        for (uint32_t b = 0; b < mgr->num_bridges && total_arrays < NIMCP_UTM_MAX_NETWORKS * 32; b++) {
            nimcp_cross_network_bridge_t* br = &mgr->bridges[b];
            if (br->weight_grad && br->source_dim > 0 && br->target_dim > 0) {
                all_grads[total_arrays] = br->weight_grad;
                all_sizes[total_arrays] = (size_t)br->target_dim * br->source_dim;
                total_arrays++;
            }
        }

        /* Global safety clip (10x target norm) */
        float scale = nimcp_anti_collapse_normalize_gradients(
            &mgr->anti_collapse[0], all_grads, all_sizes, total_arrays);

        float effective_target = mgr->anti_collapse[0].config.gradient_target_norm;
        if (mgr->anti_collapse[0].config.adaptive_gradient_target && effective_target <= 0.0f) {
            effective_target = sqrtf(mgr->anti_collapse[0].ema_gradient_norm > 0.0f ?
                                     mgr->anti_collapse[0].ema_gradient_norm : 1.0f);
        }
        local_result.gradient_norm = (scale != 0.0f) ?
            effective_target / scale : 0.0f;
        local_result.gradient_scale = scale;
    }

    /* ------------------------------------------------------------------ */
    /* Step 6: Optimizer step (AdamW with LR schedule)                    */
    /* ------------------------------------------------------------------ */

    /* Update learning rate from schedule */
    mgr->current_lr = nimcp_utm_get_scheduled_lr(mgr);

    /* AdamW constants */
    const float adam_beta1 = 0.9f;
    const float adam_beta2 = 0.999f;
    const float adam_eps = 1e-8f;

    /* Update bias correction terms: beta^t */
    mgr->adam_beta1_t *= adam_beta1;
    mgr->adam_beta2_t *= adam_beta2;

    /* Collect all param groups across all networks to assign moment indices */
    {
        /* First pass: count total param groups */
        uint32_t total_groups = 0;
        for (uint32_t i = 0; i < mgr->num_networks; i++) {
            if (!mgr->networks[i].enabled) continue;
            nimcp_trainable_network_t* net = &mgr->networks[i];
            nimcp_utm_param_group_t* groups = NULL;
            uint32_t num_groups = 0;
            if (net->ops->get_param_groups &&
                net->ops->get_param_groups(net->ctx, &groups, &num_groups) == 0) {
                total_groups += num_groups;
                nimcp_free(groups);
            }
        }

        /* Lazy-allocate or re-allocate Adam moment arrays if group count changed */
        if (total_groups > 0 && (mgr->adam_m == NULL || total_groups != mgr->adam_num_groups)) {
            /* Free old state if group count changed */
            if (mgr->adam_m) {
                for (uint32_t i = 0; i < mgr->adam_num_groups; i++) {
                    nimcp_free(mgr->adam_m[i]);
                    nimcp_free(mgr->adam_v[i]);
                }
                nimcp_free(mgr->adam_m);
                nimcp_free(mgr->adam_v);
                nimcp_free(mgr->adam_sizes);
            }

            mgr->adam_m = (float**)nimcp_calloc(total_groups, sizeof(float*));
            mgr->adam_v = (float**)nimcp_calloc(total_groups, sizeof(float*));
            mgr->adam_sizes = (size_t*)nimcp_calloc(total_groups, sizeof(size_t));
            mgr->adam_num_groups = total_groups;
            /* Reset bias correction on realloc */
            mgr->adam_beta1_t = adam_beta1;
            mgr->adam_beta2_t = adam_beta2;
        }

        /* Second pass: apply AdamW update */
        uint32_t group_idx = 0;
        for (uint32_t i = 0; i < mgr->num_networks; i++) {
            if (!mgr->networks[i].enabled) continue;
            nimcp_trainable_network_t* net = &mgr->networks[i];

            nimcp_utm_param_group_t* groups = NULL;
            uint32_t num_groups = 0;
            if (net->ops->get_param_groups &&
                net->ops->get_param_groups(net->ctx, &groups, &num_groups) == 0) {
                for (uint32_t g = 0; g < num_groups; g++) {
                    if (!groups[g].params || !groups[g].gradients || groups[g].count == 0) {
                        group_idx++;
                        continue;
                    }

                    /* Lazy-allocate moment vectors for this group */
                    if (group_idx < mgr->adam_num_groups) {
                        if (!mgr->adam_m[group_idx] || mgr->adam_sizes[group_idx] != groups[g].count) {
                            nimcp_free(mgr->adam_m[group_idx]);
                            nimcp_free(mgr->adam_v[group_idx]);
                            mgr->adam_m[group_idx] = (float*)nimcp_calloc(groups[g].count, sizeof(float));
                            mgr->adam_v[group_idx] = (float*)nimcp_calloc(groups[g].count, sizeof(float));
                            mgr->adam_sizes[group_idx] = groups[g].count;
                        }
                    }

                    /* Feature 1: Hub-aware fractal LR scaling */
                    float fractal_scale = mgr->fractal_enabled ?
                        mgr->fractal_lr_multiplier[i] : 1.0f;
                    float lr = mgr->current_lr * groups[g].lr_scale * fractal_scale;
                    float wd = groups[g].weight_decay;
                    float bc1 = 1.0f - mgr->adam_beta1_t;  /* 1 - beta1^t */
                    float bc2 = 1.0f - mgr->adam_beta2_t;  /* 1 - beta2^t */

                    float* m = (group_idx < mgr->adam_num_groups) ? mgr->adam_m[group_idx] : NULL;
                    float* v = (group_idx < mgr->adam_num_groups) ? mgr->adam_v[group_idx] : NULL;

                    /* Feature 7: Natural gradient optimizer for small param groups */
                    bool use_ng = mgr->natural_gradient_enabled &&
                                  groups[g].count <= mgr->natural_grad_max_params &&
                                  groups[g].count > 0;

                    if (use_ng) {
                        /* Lazy-create Fisher + NG handles for this network */
                        if (!mgr->fisher[i]) {
                            nimcp_fisher_config_t f_cfg = {0};
                            f_cfg.param_dim = (uint32_t)groups[g].count;
                            f_cfg.sample_size = 1;
                            f_cfg.regularization = 1e-4f;
                            f_cfg.use_empirical = true;
                            f_cfg.enable_damping = true;
                            f_cfg.initial_damping = 1e-3f;
                            mgr->fisher[i] = nimcp_fisher_create(&f_cfg);
                        }
                        if (!mgr->natural_grad[i]) {
                            nimcp_natural_grad_config_t ng_cfg = {0};
                            ng_cfg.learning_rate = lr;
                            ng_cfg.momentum = 0.9f;
                            ng_cfg.gradient_clip = 5.0f;
                            ng_cfg.use_preconditioner = true;
                            ng_cfg.enable_warmup = false;
                            mgr->natural_grad[i] = nimcp_natural_grad_create(
                                &ng_cfg, (uint32_t)groups[g].count);
                        }

                        /* Periodically recompute Fisher */
                        if (mgr->fisher[i] &&
                            mgr->step_count % mgr->fisher_update_interval == 0) {
                            nimcp_fisher_compute(mgr->fisher[i],
                                groups[g].gradients, 1, (uint32_t)groups[g].count);
                            if (mgr->natural_grad[i]) {
                                nimcp_natural_grad_update_fisher(
                                    mgr->natural_grad[i], mgr->fisher[i]);
                            }
                        }

                        /* Apply weight decay before NG step */
                        if (wd > 0.0f) {
                            for (size_t j = 0; j < groups[g].count; j++) {
                                groups[g].params[j] -= lr * wd * groups[g].params[j];
                            }
                        }

                        /* Natural gradient step */
                        if (mgr->natural_grad[i]) {
                            nimcp_natural_grad_step(mgr->natural_grad[i],
                                groups[g].params, groups[g].gradients,
                                (uint32_t)groups[g].count);
                        }
                    } else {
                        /* Standard AdamW */
                        for (size_t j = 0; j < groups[g].count; j++) {
                            float grad = groups[g].gradients[j];

                            if (m && v) {
                                /* AdamW: update moments */
                                m[j] = adam_beta1 * m[j] + (1.0f - adam_beta1) * grad;
                                v[j] = adam_beta2 * v[j] + (1.0f - adam_beta2) * grad * grad;

                                /* Bias-corrected estimates */
                                float m_hat = m[j] / bc1;
                                float v_hat = v[j] / bc2;

                                /* AdamW: decoupled weight decay */
                                if (wd > 0.0f) {
                                    groups[g].params[j] -= lr * wd * groups[g].params[j];
                                }

                                /* Adam update */
                                groups[g].params[j] -= lr * m_hat / (sqrtf(v_hat) + adam_eps);
                            } else {
                                /* Fallback SGD */
                                if (wd > 0.0f) {
                                    groups[g].params[j] -= lr * wd * groups[g].params[j];
                                }
                                groups[g].params[j] -= lr * grad;
                            }
                        }
                    }

                    group_idx++;
                }
                nimcp_free(groups);
            }
        }
    }

    /* Phase 4: Sync cached params back to underlying networks after AdamW */
    for (uint32_t i = 0; i < mgr->num_networks; i++) {
        if (!mgr->networks[i].enabled) continue;
        if (mgr->networks[i].ops->sync_params) {
            mgr->networks[i].ops->sync_params(mgr->networks[i].ctx);
        }
    }

    /* Apply bridge weight updates */
    for (uint32_t b = 0; b < mgr->num_bridges; b++) {
        nimcp_cross_network_bridge_t* br = &mgr->bridges[b];
        if (!br->enabled || !br->transform_weights || !br->weight_grad) continue;

        size_t w_size = (size_t)br->target_dim * br->source_dim;
        for (size_t j = 0; j < w_size; j++) {
            br->transform_weights[j] -= mgr->current_lr * br->weight_grad[j];
        }
        if (br->transform_bias && br->bias_grad) {
            for (uint32_t j = 0; j < br->target_dim; j++) {
                br->transform_bias[j] -= mgr->current_lr * br->bias_grad[j];
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* Step 7: Update state                                               */
    /* ------------------------------------------------------------------ */
    mgr->step_count++;
    mgr->last_composite_loss = composite_loss;
    memcpy(mgr->per_network_loss, local_result.per_network_loss, sizeof(mgr->per_network_loss));

    local_result.step = mgr->step_count;

    /* ------------------------------------------------------------------ */
    /* Step 8: DFA health monitoring + fractal analysis                    */
    /* ------------------------------------------------------------------ */
    if (mgr->loss_history && mgr->loss_history_size > 0) {
        /* Record loss in ring buffer */
        mgr->loss_history[mgr->loss_history_pos] = composite_loss;
        mgr->loss_history_pos = (mgr->loss_history_pos + 1) % mgr->loss_history_size;
        if (mgr->loss_history_count < mgr->loss_history_size) {
            mgr->loss_history_count++;
        }

        /* Record per-network losses for phase coherence */
        if (mgr->num_networks >= 2 && mgr->per_network_loss_history == NULL) {
            mgr->per_network_loss_history = (float*)nimcp_calloc(
                (size_t)mgr->loss_history_size * mgr->num_networks, sizeof(float));
        }
        if (mgr->per_network_loss_history) {
            uint32_t pos = (mgr->loss_history_pos == 0) ?
                            mgr->loss_history_size - 1 : mgr->loss_history_pos - 1;
            for (uint32_t n = 0; n < mgr->num_networks; n++) {
                mgr->per_network_loss_history[pos * mgr->num_networks + n] =
                    local_result.per_network_loss[n];
            }
        }

        /* Run health check every N steps when enough data */
        if (mgr->health_check_interval > 0 &&
            mgr->step_count % mgr->health_check_interval == 0 &&
            mgr->loss_history_count >= 32) {

            /* Linearize ring buffer for fractal analysis */
            uint32_t count = mgr->loss_history_count;
            float* linear = (float*)nimcp_malloc(count * sizeof(float));
            if (linear) {
                uint32_t start = (mgr->loss_history_count < mgr->loss_history_size) ?
                    0 : mgr->loss_history_pos;
                for (uint32_t k = 0; k < count; k++) {
                    linear[k] = mgr->loss_history[(start + k) % mgr->loss_history_size];
                }

                /* Feature 2: DFA + Hurst */
                fractal_config_t fcfg = fractal_config_default();
                fractal_result_t dfa_result = {0};
                fractal_result_t hurst_result = {0};

                if (fractal_dfa(linear, count, &fcfg, &dfa_result) == 0) {
                    mgr->dfa_exponent = dfa_result.dfa_exponent;
                }
                if (fractal_hurst_rs(linear, count, &fcfg, &hurst_result) == 0) {
                    mgr->hurst_exponent = hurst_result.hurst_exponent;
                }

                /* Classify health */
                float alpha = mgr->dfa_exponent;
                float H = mgr->hurst_exponent;
                if (alpha < 0.3f) {
                    mgr->training_health = NIMCP_TRAINING_HEALTH_OSCILLATING;
                } else if (alpha < 0.6f) {
                    mgr->training_health = NIMCP_TRAINING_HEALTH_NOISY;
                } else if (alpha > 1.3f) {
                    mgr->training_health = NIMCP_TRAINING_HEALTH_DRIFTING;
                } else if (H > 0.8f) {
                    mgr->training_health = NIMCP_TRAINING_HEALTH_PLATEAU;
                } else {
                    mgr->training_health = NIMCP_TRAINING_HEALTH_OPTIMAL;
                }

                /* Feature 2: Auto-adjust LR based on health */
                if (mgr->dfa_auto_adjust_lr) {
                    switch (mgr->training_health) {
                        case NIMCP_TRAINING_HEALTH_NOISY:
                            mgr->current_lr *= 0.8f;
                            break;
                        case NIMCP_TRAINING_HEALTH_DRIFTING:
                            mgr->current_lr *= 0.5f;
                            break;
                        case NIMCP_TRAINING_HEALTH_OSCILLATING:
                            mgr->current_lr *= 0.3f;
                            break;
                        case NIMCP_TRAINING_HEALTH_PLATEAU:
                            mgr->current_lr *= 1.5f;
                            break;
                        default:
                            break;
                    }
                }

                /* Feature 3: Extended fractal analysis */
                /* Lacunarity */
                if (count >= 8) {
                    mgr->lacunarity_value = fractal_lacunarity(linear, count, count / 8);
                }

                /* Multifractal spectrum */
                multifractal_spectrum_t* spectrum = NULL;
                if (fractal_multifractal_spectrum(linear, count,
                        -3.0f, 3.0f, 13, &spectrum) == 0 && spectrum) {
                    mgr->is_multifractal = spectrum->is_multifractal;
                    mgr->multifractal_width = spectrum->width;
                    multifractal_spectrum_destroy(spectrum);
                }

                /* Spectral exponent (supplementary) */
                fractal_result_t spectral_result = {0};
                fractal_spectral_exponent(linear, count, &spectral_result);

                /* Feature 5: Quantum annealing for plateau escape */
                if (mgr->training_health == NIMCP_TRAINING_HEALTH_PLATEAU) {
                    mgr->plateau_consecutive_count++;
                } else {
                    mgr->plateau_consecutive_count = 0;
                }

                if (mgr->enable_quantum_anneal &&
                    mgr->plateau_consecutive_count >= mgr->plateau_anneal_threshold) {

                    NIMCP_LOGGING_INFO("UTM: Triggering quantum annealing (plateau for %u checks)",
                                       mgr->plateau_consecutive_count);

                    quantum_annealing_config_t qa_cfg = quantum_annealing_default_config();
                    qa_cfg.num_iterations = 50;
                    qa_cfg.quantum_strength = 0.1f;
                    qa_cfg.enable_tunneling = true;
                    quantum_annealer_t annealer = quantum_annealer_create(&qa_cfg);
                    if (annealer) {
                        /* Perturb each network's first small param group */
                        for (uint32_t n = 0; n < mgr->num_networks; n++) {
                            if (!mgr->networks[n].enabled) continue;
                            nimcp_utm_param_group_t* pgroups = NULL;
                            uint32_t npg = 0;
                            if (mgr->networks[n].ops->get_param_groups &&
                                mgr->networks[n].ops->get_param_groups(
                                    mgr->networks[n].ctx, &pgroups, &npg) == 0 && npg > 0) {
                                /* Anneal first group if small enough */
                                if (pgroups[0].params && pgroups[0].count > 0 &&
                                    pgroups[0].count <= 4096) {
                                    float* optimized = (float*)nimcp_malloc(
                                        pgroups[0].count * sizeof(float));
                                    if (optimized) {
                                        quantum_anneal(annealer, NULL,
                                            pgroups[0].params, optimized,
                                            (uint32_t)pgroups[0].count, NULL);
                                        memcpy(pgroups[0].params, optimized,
                                            pgroups[0].count * sizeof(float));
                                        nimcp_free(optimized);
                                    }
                                }
                                nimcp_free(pgroups);
                            }
                        }
                        quantum_annealer_destroy(annealer);
                        mgr->plateau_consecutive_count = 0;
                    }
                }

                /* Feature 8: Phase coherence (cross-network mode collapse detection) */
                if (mgr->num_networks >= 2 && mgr->per_network_loss_history && count >= 16) {
                    float total_sync = 0.0f;
                    uint32_t pair_count = 0;

                    /* Allocate phasor buffers */
                    neural_phasor_t* phasors_a = (neural_phasor_t*)nimcp_malloc(
                        count * sizeof(neural_phasor_t));
                    neural_phasor_t* phasors_b = (neural_phasor_t*)nimcp_malloc(
                        count * sizeof(neural_phasor_t));
                    float* net_series = (float*)nimcp_malloc(count * sizeof(float));

                    if (phasors_a && phasors_b && net_series) {
                        for (uint32_t na = 0; na < mgr->num_networks; na++) {
                            /* Extract network na's loss history */
                            for (uint32_t k = 0; k < count; k++) {
                                uint32_t idx = (start + k) % mgr->loss_history_size;
                                net_series[k] = mgr->per_network_loss_history[
                                    idx * mgr->num_networks + na];
                            }
                            if (!phasor_hilbert_transform(net_series, phasors_a, count))
                                continue;

                            for (uint32_t nb = na + 1; nb < mgr->num_networks; nb++) {
                                for (uint32_t k = 0; k < count; k++) {
                                    uint32_t idx = (start + k) % mgr->loss_history_size;
                                    net_series[k] = mgr->per_network_loss_history[
                                        idx * mgr->num_networks + nb];
                                }
                                if (!phasor_hilbert_transform(net_series, phasors_b, count))
                                    continue;

                                float sync = phasor_array_synchrony(phasors_a, phasors_b, count);
                                total_sync += sync;
                                pair_count++;
                            }
                        }
                    }
                    nimcp_free(phasors_a);
                    nimcp_free(phasors_b);
                    nimcp_free(net_series);

                    if (pair_count > 0) {
                        mgr->cross_network_coherence = total_sync / (float)pair_count;
                        /* Override health if networks are locked together */
                        if (mgr->cross_network_coherence > 0.9f) {
                            mgr->training_health = NIMCP_TRAINING_HEALTH_OSCILLATING;
                        }
                    }
                }

                nimcp_free(linear);
            }
        }

        /* Feature 6: Bottleneck detection (less frequent) */
        if (mgr->bottleneck_check_interval > 0 &&
            mgr->step_count % mgr->bottleneck_check_interval == 0 &&
            mgr->num_bridges > 0) {
            for (uint32_t b = 0; b < mgr->num_bridges; b++) {
                nimcp_cross_network_bridge_t* br = &mgr->bridges[b];
                if (!br->enabled || !br->last_source_output || !br->last_target_input)
                    continue;

                /* Simple bottleneck metric: cosine similarity between source and target */
                float dot = 0.0f, norm_s = 0.0f, norm_t = 0.0f;
                uint32_t min_dim = (br->source_dim < br->target_dim) ?
                                    br->source_dim : br->target_dim;
                for (uint32_t j = 0; j < min_dim; j++) {
                    dot += br->last_source_output[j] * br->last_target_input[j];
                    norm_s += br->last_source_output[j] * br->last_source_output[j];
                    norm_t += br->last_target_input[j] * br->last_target_input[j];
                }
                float denom = sqrtf(norm_s) * sqrtf(norm_t);
                float similarity = (denom > 1e-7f) ? (dot / denom) : 0.0f;
                /* Low similarity = information bottleneck */
                mgr->bridge_bottleneck_severity[b] = 1.0f - fabsf(similarity);

                if (mgr->bridge_bottleneck_severity[b] > 0.7f) {
                    NIMCP_LOGGING_WARN("UTM: Bridge %u bottleneck severity %.2f",
                                        b, mgr->bridge_bottleneck_severity[b]);
                }
            }
        }
    }

    /* Feature 9: Manifold dimensionality tracking */
    if (mgr->manifold_tracking_enabled &&
        mgr->health_check_interval > 0 &&
        mgr->step_count % mgr->health_check_interval == 0) {
        for (uint32_t i = 0; i < mgr->num_networks; i++) {
            if (!mgr->networks[i].enabled || !net_outputs[i]) continue;
            uint32_t out_dim = net_output_dims[i];
            if (out_dim == 0) continue;

            /* Lazy-create manifold handle */
            if (!mgr->output_manifold[i]) {
                nimcp_manifold_config_t m_cfg = {0};
                m_cfg.intrinsic_dim = 0; /* auto-detect */
                m_cfg.num_samples = 64;
                m_cfg.neighborhood_radius = 0.1f;
                m_cfg.compute_curvature = false;
                m_cfg.enable_embedding = false;
                mgr->output_manifold[i] = nimcp_manifold_create(&m_cfg, out_dim);
            }

            /* Add current output as a sample */
            if (mgr->output_manifold[i]) {
                nimcp_manifold_add_samples(mgr->output_manifold[i],
                    net_outputs[i], 1, out_dim);

                /* Estimate intrinsic dimensionality */
                uint32_t dim = 0;
                if (nimcp_manifold_estimate_dim(mgr->output_manifold[i], &dim) == 0) {
                    mgr->manifold_intrinsic_dim[i] = dim;
                    if (dim < 2 && dim > 0) {
                        NIMCP_LOGGING_WARN("UTM: Network %u manifold dim collapsed to %u", i, dim);
                    }
                }
            }
        }
    }

    /* Populate extended result fields */
    local_result.training_health = mgr->training_health;
    local_result.dfa_exponent = mgr->dfa_exponent;
    local_result.hurst_exponent = mgr->hurst_exponent;
    local_result.lacunarity_value = mgr->lacunarity_value;
    local_result.is_multifractal = mgr->is_multifractal;
    local_result.cross_network_coherence = mgr->cross_network_coherence;
    memcpy(local_result.manifold_intrinsic_dim, mgr->manifold_intrinsic_dim,
           sizeof(mgr->manifold_intrinsic_dim));

    if (result) {
        *result = local_result;
    }

cleanup:
    /* Free output buffers */
    for (uint32_t i = 0; i < mgr->num_networks; i++) {
        nimcp_free(net_outputs[i]);
    }

    return 0;
}

//=============================================================================
// Fractal-Aware Training API
//=============================================================================

void nimcp_utm_set_fractal_lr(nimcp_unified_training_manager_t* mgr,
                               uint32_t net_idx, float scale) {
    if (!mgr || net_idx >= NIMCP_UTM_MAX_NETWORKS) return;
    mgr->fractal_lr_multiplier[net_idx] = scale;
    mgr->fractal_enabled = true;
}

nimcp_training_health_t nimcp_utm_get_health(const nimcp_unified_training_manager_t* mgr) {
    if (!mgr) return NIMCP_TRAINING_HEALTH_UNKNOWN;
    return mgr->training_health;
}

float nimcp_utm_get_dfa_exponent(const nimcp_unified_training_manager_t* mgr) {
    if (!mgr) return 0.0f;
    return mgr->dfa_exponent;
}

void nimcp_utm_set_natural_gradient(nimcp_unified_training_manager_t* mgr,
                                     uint32_t net_idx, bool enabled) {
    if (!mgr || net_idx >= NIMCP_UTM_MAX_NETWORKS) return;
    /* Toggle per-network by destroying/allowing lazy creation */
    if (!enabled && mgr->natural_grad[net_idx]) {
        nimcp_natural_grad_destroy(mgr->natural_grad[net_idx]);
        mgr->natural_grad[net_idx] = NULL;
    }
    if (!enabled && mgr->fisher[net_idx]) {
        nimcp_fisher_destroy(mgr->fisher[net_idx]);
        mgr->fisher[net_idx] = NULL;
    }
    /* If enabling, handles will be lazy-created in the optimizer loop */
}
