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
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <math.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>

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

    config->loss_type = 0; /* NIMCP_LOSS_MSE */
    config->use_composite_loss = true;
    config->enable_cross_network_gradients = false;
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

    /* Initialize anti-collapse state */
    nimcp_anti_collapse_init(&mgr->anti_collapse, &mgr->config.anti_collapse);

    NIMCP_LOGGING_INFO("Unified training manager created (lr=%.4f, diversity_w=%.2f, grad_norm=%s)",
                       mgr->current_lr,
                       mgr->config.anti_collapse.diversity_loss_weight,
                       mgr->config.anti_collapse.use_gradient_normalization ? "normalize" : "clip");

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

    /* Destroy anti-collapse state */
    nimcp_anti_collapse_destroy(&mgr->anti_collapse);

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
    /* Step 1: Zero all gradients                                         */
    /* ------------------------------------------------------------------ */
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

        /* Check if any bridge feeds into this network */
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
                        /* Note: bridged_input will leak if multiple bridges feed same network.
                         * In practice, networks have at most one incoming bridge. */
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

    /* Per-network MSE loss against target */
    /* In composite mode, the final network's output is compared to target.
     * Each network can also contribute auxiliary loss. */
    for (uint32_t i = 0; i < mgr->num_networks; i++) {
        if (!mgr->networks[i].enabled || !net_outputs[i]) continue;
        nimcp_trainable_network_t* net = &mgr->networks[i];

        /* MSE loss: 1/n Σ(output - target)² */
        uint32_t cmp_dim = (net_output_dims[i] < target_dim) ?
                            net_output_dims[i] : target_dim;
        float mse = 0.0f;
        for (uint32_t j = 0; j < cmp_dim; j++) {
            float diff = net_outputs[i][j] - target[j];
            mse += diff * diff;
        }
        mse /= (float)(cmp_dim > 0 ? cmp_dim : 1);

        /* Auxiliary loss from the network (e.g., spike regularization) */
        float aux = 0.0f;
        if (net->ops->compute_auxiliary_loss) {
            aux = net->ops->compute_auxiliary_loss(net->ctx);
        }

        float net_loss = mse + aux;
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
    for (int i = (int)mgr->num_networks - 1; i >= 0; i--) {
        if (!mgr->networks[i].enabled || !net_outputs[i]) continue;
        nimcp_trainable_network_t* net = &mgr->networks[i];

        /* Compute dL/dOutput for this network */
        uint32_t out_dim = net_output_dims[i];
        float* dl_dout = (float*)nimcp_calloc(out_dim, sizeof(float));
        if (!dl_dout) continue;

        /* MSE gradient: dL/dy = 2(y - target) / n */
        uint32_t cmp_dim = (out_dim < target_dim) ? out_dim : target_dim;
        for (uint32_t j = 0; j < cmp_dim; j++) {
            dl_dout[j] = 2.0f * (net_outputs[i][j] - target[j]) / (float)cmp_dim;
        }

        /* Add gradient from any downstream bridge */
        if (mgr->config.enable_cross_network_gradients) {
            for (uint32_t b = 0; b < mgr->num_bridges; b++) {
                if (mgr->bridges[b].source_idx == (uint32_t)i && mgr->bridges[b].enabled) {
                    /* This network is a source — add gradient from bridge backward */
                    float* bridge_grad = (float*)nimcp_calloc(out_dim, sizeof(float));
                    if (bridge_grad) {
                        /* The bridge backward needs dL/d(target_input), which is
                         * the input gradient of the downstream network. For now,
                         * use the downstream network's output gradient as approximation.
                         * Full implementation in Phase 4. */
                        bridge_backward(&mgr->bridges[b], dl_dout, bridge_grad);
                        for (uint32_t j = 0; j < out_dim; j++) {
                            dl_dout[j] += bridge_grad[j];
                        }
                        nimcp_free(bridge_grad);
                    }
                }
            }
        }

        /* Add diversity loss gradient (shared anti-collapse) */
        float div_loss = nimcp_anti_collapse_diversity_loss(
            &mgr->anti_collapse, net_outputs[i], dl_dout, out_dim);
        local_result.diversity_loss += div_loss;

        /* Backward through the network */
        uint32_t in_dim = net->ops->get_input_dim(net->ctx);
        float* dl_din = (float*)nimcp_calloc(in_dim, sizeof(float));

        net->ops->backward(net->ctx, dl_dout, out_dim, dl_din, in_dim);

        nimcp_free(dl_dout);
        nimcp_free(dl_din);
    }

    /* ------------------------------------------------------------------ */
    /* Step 5: Gradient normalization / clipping (unified, single pass)   */
    /* ------------------------------------------------------------------ */
    {
        /* Collect all gradient arrays from all networks */
        float* all_grads[NIMCP_UTM_MAX_NETWORKS * 32]; /* generous upper bound */
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

        float scale = nimcp_anti_collapse_normalize_gradients(
            &mgr->config.anti_collapse, all_grads, all_sizes, total_arrays);

        /* Compute gradient norm for reporting (before normalization was applied) */
        local_result.gradient_norm = (scale != 0.0f) ?
            mgr->config.anti_collapse.gradient_target_norm / scale : 0.0f;
        local_result.gradient_scale = scale;
    }

    /* ------------------------------------------------------------------ */
    /* Step 6: Optimizer step (apply gradients to parameters)             */
    /* ------------------------------------------------------------------ */
    for (uint32_t i = 0; i < mgr->num_networks; i++) {
        if (!mgr->networks[i].enabled) continue;
        nimcp_trainable_network_t* net = &mgr->networks[i];

        nimcp_utm_param_group_t* groups = NULL;
        uint32_t num_groups = 0;
        if (net->ops->get_param_groups &&
            net->ops->get_param_groups(net->ctx, &groups, &num_groups) == 0) {
            for (uint32_t g = 0; g < num_groups; g++) {
                if (!groups[g].params || !groups[g].gradients || groups[g].count == 0) continue;

                float lr = mgr->current_lr * groups[g].lr_scale;
                float wd = groups[g].weight_decay;

                /* SGD with weight decay: param -= lr * (grad + wd * param) */
                for (size_t j = 0; j < groups[g].count; j++) {
                    float grad = groups[g].gradients[j];
                    if (wd > 0.0f) {
                        grad += wd * groups[g].params[j];
                    }
                    groups[g].params[j] -= lr * grad;
                }
            }
            nimcp_free(groups);
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
