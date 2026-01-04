/**
 * @file nimcp_omni_logic_bridge.c
 * @brief Implementation of Omnidirectional Inference to Logic Gate Bridge
 */

#include "cognitive/logic/nimcp_omni_logic_bridge.h"
#include "cognitive/jepa/nimcp_jepa_bidirectional.h"
#include "cognitive/predictive/nimcp_predictive_hierarchy.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/logging/nimcp_logging.h"

#include <string.h>
#include <math.h>

/* ============================================================================
 * Static Helpers
 * ============================================================================ */

static omni_infer_direction_t compute_recommended_direction(
    const omni_logic_conditions_t* conds,
    const omni_logic_config_t* config) {

    /* If goal is specified and causes unknown → backward chaining */
    if (conds->goal_specified && !conds->causes_known) {
        return OMNI_INFER_BACKWARD;
    }

    /* If causes known and goal unknown → forward chaining */
    if (conds->causes_known && !conds->goal_specified) {
        return OMNI_INFER_FORWARD;
    }

    /* If lateral modality is relevant → lateral inference */
    if (conds->lateral_relevant) {
        return OMNI_INFER_LATERAL;
    }

    /* If conflicting evidence → weighted combination */
    if (conds->conflicting_evidence) {
        return OMNI_INFER_WEIGHTED_COMBINE;
    }

    /* Default: use direction with highest confidence */
    if (conds->forward_confident && !conds->backward_confident) {
        return OMNI_INFER_FORWARD;
    }
    if (conds->backward_confident && !conds->forward_confident) {
        return OMNI_INFER_BACKWARD;
    }

    /* Both confident or neither → weighted combination */
    return OMNI_INFER_WEIGHTED_COMBINE;
}

static omni_logic_type_t compute_suggested_logic_type(
    const omni_logic_conditions_t* conds) {

    if (conds->goal_specified && !conds->causes_known) {
        return OMNI_LOGIC_ABDUCTIVE;
    }
    if (conds->causes_known && !conds->goal_specified) {
        return OMNI_LOGIC_DEDUCTIVE;
    }
    if (conds->lateral_relevant) {
        return OMNI_LOGIC_ANALOGICAL;
    }
    return OMNI_LOGIC_COMBINED;
}

/* ============================================================================
 * Configuration API
 * ============================================================================ */

int omni_logic_default_config(omni_logic_config_t* config) {
    if (!config) return NIMCP_ERROR_INVALID_PARAM;

    memset(config, 0, sizeof(omni_logic_config_t));

    config->confidence_threshold = OMNI_LOGIC_DEFAULT_CONFIDENCE_THRESHOLD;
    config->forward_confidence_min = 0.5f;
    config->backward_confidence_min = 0.5f;

    config->default_forward_weight = 1.0f;
    config->default_backward_weight = 1.0f;
    config->default_lateral_weight = 0.8f;

    config->enable_chaining = true;
    config->max_chain_depth = 5;
    config->max_iterations = 10;

    config->create_default_gates = true;

    config->enable_bio_async = true;
    config->enable_logging = false;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

omni_logic_bridge_t* omni_logic_bridge_create(const omni_logic_config_t* config) {
    omni_logic_bridge_t* bridge = nimcp_calloc(1, sizeof(omni_logic_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        memcpy(&bridge->config, config, sizeof(omni_logic_config_t));
    } else {
        omni_logic_default_config(&bridge->config);
    }

    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate rules array */
    bridge->max_rules = OMNI_LOGIC_MAX_RULES;
    bridge->rules = nimcp_calloc(bridge->max_rules, sizeof(omni_logic_rule_t));
    if (!bridge->rules) {
        nimcp_mutex_destroy(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }
    bridge->num_rules = 0;

    /* Initialize default conditions */
    memset(&bridge->conditions, 0, sizeof(omni_logic_conditions_t));
    bridge->conditions.available_time_ms = 1000.0f;

    /* Initialize default direction weights */
    bridge->logic_effects.direction_weight[0] = bridge->config.default_forward_weight;
    bridge->logic_effects.direction_weight[1] = bridge->config.default_backward_weight;
    bridge->logic_effects.direction_weight[2] = bridge->config.default_lateral_weight;
    bridge->logic_effects.direction_weight[3] = 0.5f;  /* Hierarchical up */
    bridge->logic_effects.direction_weight[4] = 0.5f;  /* Hierarchical down */

    bridge->logic_effects.max_iterations = bridge->config.max_iterations;
    bridge->logic_effects.confidence_threshold = bridge->config.confidence_threshold;
    bridge->logic_effects.enable_chaining = bridge->config.enable_chaining;

    memset(&bridge->stats, 0, sizeof(omni_logic_stats_t));

    return bridge;
}

void omni_logic_bridge_destroy(omni_logic_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->rules) {
        nimcp_free(bridge->rules);
    }

    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int omni_logic_connect_jepa(omni_logic_bridge_t* bridge,
                             jepa_bidirectional_t* jepa) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->jepa = jepa;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_logic_connect_pred_hier(omni_logic_bridge_t* bridge,
                                  predictive_hierarchy_t* pred_hier) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->pred_hier = pred_hier;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_logic_connect_logic_network(omni_logic_bridge_t* bridge,
                                      neural_logic_network_t* logic_net) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->logic_net = logic_net;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Update API
 * ============================================================================ */

int omni_logic_update(omni_logic_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    /* Update conditions from connected systems */
    if (bridge->jepa) {
        jepa_bidir_stats_t stats;
        if (jepa_bidir_get_stats(bridge->jepa, &stats) == NIMCP_SUCCESS) {
            bridge->conditions.forward_pe = stats.avg_forward_error;
            bridge->conditions.backward_pe = stats.avg_backward_error;
            bridge->conditions.forward_confident =
                (stats.avg_forward_error < bridge->config.forward_confidence_min);
            bridge->conditions.backward_confident =
                (stats.avg_backward_error < bridge->config.backward_confidence_min);
        }
    }

    /* Compute omni → logic effects */
    bridge->omni_effects.forward_confidence =
        1.0f / (1.0f + bridge->conditions.forward_pe);
    bridge->omni_effects.backward_confidence =
        1.0f / (1.0f + bridge->conditions.backward_pe);
    bridge->omni_effects.lateral_confidence =
        1.0f / (1.0f + bridge->conditions.lateral_pe);
    bridge->omni_effects.hierarchical_confidence = 0.7f;

    bridge->omni_effects.suggested_type = compute_suggested_logic_type(&bridge->conditions);
    bridge->omni_effects.recommended_dir =
        compute_recommended_direction(&bridge->conditions, &bridge->config);

    /* Update logic → omni effects based on conditions */
    bridge->logic_effects.direction = bridge->omni_effects.recommended_dir;
    bridge->logic_effects.force_direction = false;

    /* Update statistics */
    bridge->stats.total_updates++;
    float n = (float)bridge->stats.total_updates;
    bridge->stats.avg_forward_confidence =
        (bridge->stats.avg_forward_confidence * (n - 1) + bridge->omni_effects.forward_confidence) / n;
    bridge->stats.avg_backward_confidence =
        (bridge->stats.avg_backward_confidence * (n - 1) + bridge->omni_effects.backward_confidence) / n;

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_logic_apply_to_logic(omni_logic_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    bridge->stats.gate_evaluations++;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_logic_apply_to_omni(omni_logic_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Inference Direction API
 * ============================================================================ */

int omni_logic_get_direction(const omni_logic_bridge_t* bridge,
                              omni_infer_direction_t* direction) {
    if (!bridge || !direction) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_logic_bridge_t*)bridge)->mutex);
    *direction = bridge->logic_effects.direction;
    ((omni_logic_bridge_t*)bridge)->stats.direction_decisions++;
    nimcp_mutex_unlock(((omni_logic_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_logic_set_condition(omni_logic_bridge_t* bridge,
                              omni_logic_condition_t condition,
                              bool value) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    switch (condition) {
        case OMNI_COND_GOAL_SPECIFIED:
            bridge->conditions.goal_specified = value;
            break;
        case OMNI_COND_CAUSES_KNOWN:
            bridge->conditions.causes_known = value;
            break;
        case OMNI_COND_FORWARD_CONFIDENT:
            bridge->conditions.forward_confident = value;
            break;
        case OMNI_COND_BACKWARD_CONFIDENT:
            bridge->conditions.backward_confident = value;
            break;
        case OMNI_COND_LATERAL_RELEVANT:
            bridge->conditions.lateral_relevant = value;
            break;
        case OMNI_COND_CONFLICTING_EVIDENCE:
            bridge->conditions.conflicting_evidence = value;
            break;
        case OMNI_COND_TIME_CONSTRAINED:
            bridge->conditions.time_constrained = value;
            break;
        default:
            nimcp_mutex_unlock(bridge->mutex);
            return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_logic_get_conditions(const omni_logic_bridge_t* bridge,
                               omni_logic_conditions_t* conditions) {
    if (!bridge || !conditions) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_logic_bridge_t*)bridge)->mutex);
    memcpy(conditions, &bridge->conditions, sizeof(omni_logic_conditions_t));
    nimcp_mutex_unlock(((omni_logic_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

bool omni_logic_should_forward_chain(omni_logic_bridge_t* bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->mutex);
    bool result = (bridge->conditions.causes_known && !bridge->conditions.goal_specified);
    bridge->stats.forward_selected += result ? 1 : 0;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

bool omni_logic_should_backward_chain(omni_logic_bridge_t* bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->mutex);
    bool result = (bridge->conditions.goal_specified && !bridge->conditions.causes_known);
    bridge->stats.backward_selected += result ? 1 : 0;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

bool omni_logic_should_switch_direction(omni_logic_bridge_t* bridge) {
    if (!bridge) return false;
    nimcp_mutex_lock(bridge->mutex);
    bool result = bridge->conditions.conflicting_evidence;
    nimcp_mutex_unlock(bridge->mutex);
    return result;
}

/* ============================================================================
 * Rule Management API
 * ============================================================================ */

int omni_logic_add_rule(omni_logic_bridge_t* bridge,
                         const char* name,
                         omni_logic_type_t type,
                         uint32_t gate_id,
                         uint32_t* rule_id) {
    if (!bridge || !name) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    if (bridge->num_rules >= bridge->max_rules) {
        nimcp_mutex_unlock(bridge->mutex);
        return NIMCP_ERROR_CAPACITY;
    }

    omni_logic_rule_t* rule = &bridge->rules[bridge->num_rules];
    rule->rule_id = bridge->num_rules + 1;
    strncpy(rule->name, name, sizeof(rule->name) - 1);
    rule->name[sizeof(rule->name) - 1] = '\0';
    rule->type = type;
    rule->gate_id = gate_id;
    rule->confidence_threshold = bridge->config.confidence_threshold;
    rule->active = true;

    if (rule_id) *rule_id = rule->rule_id;
    bridge->num_rules++;

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int omni_logic_remove_rule(omni_logic_bridge_t* bridge,
                            uint32_t rule_id) {
    if (!bridge || rule_id == 0) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    for (uint32_t i = 0; i < bridge->num_rules; i++) {
        if (bridge->rules[i].rule_id == rule_id) {
            bridge->rules[i].active = false;
            nimcp_mutex_unlock(bridge->mutex);
            return NIMCP_SUCCESS;
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_ERROR_NOT_FOUND;
}

int omni_logic_evaluate_rule(omni_logic_bridge_t* bridge,
                              uint32_t rule_id,
                              bool* result) {
    if (!bridge || !result || rule_id == 0) return NIMCP_ERROR_INVALID_PARAM;

    nimcp_mutex_lock(bridge->mutex);

    for (uint32_t i = 0; i < bridge->num_rules; i++) {
        if (bridge->rules[i].rule_id == rule_id && bridge->rules[i].active) {
            /* Simple evaluation based on rule type and current state */
            float confidence = 0.0f;
            switch (bridge->rules[i].type) {
                case OMNI_LOGIC_DEDUCTIVE:
                    confidence = bridge->omni_effects.forward_confidence;
                    break;
                case OMNI_LOGIC_ABDUCTIVE:
                    confidence = bridge->omni_effects.backward_confidence;
                    break;
                case OMNI_LOGIC_ANALOGICAL:
                    confidence = bridge->omni_effects.lateral_confidence;
                    break;
                default:
                    confidence = (bridge->omni_effects.forward_confidence +
                                  bridge->omni_effects.backward_confidence) / 2.0f;
                    break;
            }
            *result = (confidence >= bridge->rules[i].confidence_threshold);
            nimcp_mutex_unlock(bridge->mutex);
            return NIMCP_SUCCESS;
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_ERROR_NOT_FOUND;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int omni_logic_get_omni_effects(const omni_logic_bridge_t* bridge,
                                 omni_to_logic_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_logic_bridge_t*)bridge)->mutex);
    memcpy(effects, &bridge->omni_effects, sizeof(omni_to_logic_effects_t));
    nimcp_mutex_unlock(((omni_logic_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_logic_get_logic_effects(const omni_logic_bridge_t* bridge,
                                  logic_to_omni_effects_t* effects) {
    if (!bridge || !effects) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_logic_bridge_t*)bridge)->mutex);
    memcpy(effects, &bridge->logic_effects, sizeof(logic_to_omni_effects_t));
    nimcp_mutex_unlock(((omni_logic_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_logic_get_stats(const omni_logic_bridge_t* bridge,
                          omni_logic_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(((omni_logic_bridge_t*)bridge)->mutex);
    memcpy(stats, &bridge->stats, sizeof(omni_logic_stats_t));
    nimcp_mutex_unlock(((omni_logic_bridge_t*)bridge)->mutex);
    return NIMCP_SUCCESS;
}

int omni_logic_reset_stats(omni_logic_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(omni_logic_stats_t));
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async API
 * ============================================================================ */

int omni_logic_connect_bio_async(omni_logic_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    return NIMCP_SUCCESS;
}

int omni_logic_disconnect_bio_async(omni_logic_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;
    return NIMCP_SUCCESS;
}

bool omni_logic_is_bio_async_connected(const omni_logic_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->config.enable_bio_async;
}

/* ============================================================================
 * String Conversion API
 * ============================================================================ */

const char* omni_logic_type_to_string(omni_logic_type_t type) {
    switch (type) {
        case OMNI_LOGIC_DEDUCTIVE: return "DEDUCTIVE";
        case OMNI_LOGIC_ABDUCTIVE: return "ABDUCTIVE";
        case OMNI_LOGIC_INDUCTIVE: return "INDUCTIVE";
        case OMNI_LOGIC_ANALOGICAL: return "ANALOGICAL";
        case OMNI_LOGIC_COMBINED: return "COMBINED";
        default: return "UNKNOWN";
    }
}

const char* omni_logic_direction_to_string(omni_infer_direction_t direction) {
    switch (direction) {
        case OMNI_INFER_FORWARD: return "FORWARD";
        case OMNI_INFER_BACKWARD: return "BACKWARD";
        case OMNI_INFER_LATERAL: return "LATERAL";
        case OMNI_INFER_HIERARCHICAL_UP: return "HIERARCHICAL_UP";
        case OMNI_INFER_HIERARCHICAL_DOWN: return "HIERARCHICAL_DOWN";
        case OMNI_INFER_WEIGHTED_COMBINE: return "WEIGHTED_COMBINE";
        case OMNI_INFER_SEQUENTIAL: return "SEQUENTIAL";
        default: return "UNKNOWN";
    }
}

const char* omni_logic_condition_to_string(omni_logic_condition_t condition) {
    switch (condition) {
        case OMNI_COND_GOAL_SPECIFIED: return "GOAL_SPECIFIED";
        case OMNI_COND_CAUSES_KNOWN: return "CAUSES_KNOWN";
        case OMNI_COND_FORWARD_CONFIDENT: return "FORWARD_CONFIDENT";
        case OMNI_COND_BACKWARD_CONFIDENT: return "BACKWARD_CONFIDENT";
        case OMNI_COND_LATERAL_RELEVANT: return "LATERAL_RELEVANT";
        case OMNI_COND_CONFLICTING_EVIDENCE: return "CONFLICTING_EVIDENCE";
        case OMNI_COND_TIME_CONSTRAINED: return "TIME_CONSTRAINED";
        default: return "UNKNOWN";
    }
}
