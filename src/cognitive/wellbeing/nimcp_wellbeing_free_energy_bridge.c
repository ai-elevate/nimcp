/**
 * @file nimcp_wellbeing_free_energy_bridge.c
 * @brief Free Energy - Wellbeing Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Bidirectional integration between free energy principle and wellbeing
 * WHY:  High prediction errors and low precision cause uncertainty distress;
 *       good model fit promotes epistemic wellbeing and flourishing
 * HOW:  Map free energy metrics to distress/wellbeing, track model coherence
 *
 * @author NIMCP Development Team
 */

#include "cognitive/wellbeing/nimcp_wellbeing_free_energy_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <math.h>
#include <string.h>

/* ============================================================================
 * Lifecycle API Implementation
 * ============================================================================ */

/**
 * WHAT: Get default free energy bridge configuration
 * WHY:  Provide sensible defaults based on biological evidence
 * HOW:  Return struct with evidence-based parameters
 */
int free_energy_bridge_default_config(free_energy_bridge_config_t* config) {
    if (!config) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    config->enable_prediction_error_effects = true;
    config->enable_precision_effects = true;
    config->enable_model_coherence_effects = true;
    config->enable_active_inference_effects = true;

    config->prediction_error_sensitivity = 1.0f;
    config->precision_sensitivity = 1.0f;
    config->coherence_sensitivity = 1.0f;

    config->high_fe_threshold = FREE_ENERGY_HIGH_THRESHOLD;
    config->low_precision_threshold = PRECISION_LOW_THRESHOLD;
    config->critical_coherence_threshold = MODEL_COHERENCE_CRITICAL;

    return 0;
}

/**
 * WHAT: Create free energy bridge
 * WHY:  Initialize free energy-wellbeing integration
 * HOW:  Allocate structure, initialize state
 */
free_energy_bridge_t* free_energy_bridge_create(
    const free_energy_bridge_config_t* config
) {
    free_energy_bridge_t* bridge = nimcp_malloc(sizeof(free_energy_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate free energy bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(free_energy_bridge_t));

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        free_energy_bridge_default_config(&bridge->config);
    }

    /* Initialize state */
    bridge->fe_state.free_energy = 0.5f;
    bridge->fe_state.prediction_error = 0.0f;
    bridge->fe_state.precision = 0.7f;
    bridge->fe_state.model_evidence = 0.0f;
    bridge->fe_state.expected_free_energy = 0.5f;
    bridge->fe_state.epistemic_value = 0.5f;
    bridge->fe_state.pragmatic_value = 0.5f;

    /* Initialize coherence tracking */
    bridge->model_coherence = 0.8f;
    bridge->identity_stability = 0.8f;
    bridge->coherence_trend = 0.0f;

    /* Initialize prediction tracking */
    bridge->prediction_success_rate = 0.8f;
    bridge->predictions_made = 0;
    bridge->predictions_correct = 0;

    /* Create mutex */
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_WARN("Failed to create mutex for free energy bridge");
    }

    NIMCP_LOGGING_INFO("Free energy bridge created");
    return bridge;
}

/**
 * WHAT: Destroy free energy bridge
 * WHY:  Clean up bridge resources
 * HOW:  Free structure
 */
void free_energy_bridge_destroy(free_energy_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    if (bridge->base.mutex) {
        nimcp_mutex_free(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_DEBUG("Free energy bridge destroyed");
}

/* ============================================================================
 * Update API Implementation
 * ============================================================================ */

/**
 * WHAT: Update free energy state
 * WHY:  Receive input from free energy computations
 * HOW:  Store state, trigger effects computation
 */
int free_energy_bridge_set_state(
    free_energy_bridge_t* bridge,
    const free_energy_state_t* state
) {
    if (!bridge || !state) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (bridge->base.mutex) {
        nimcp_mutex_lock(bridge->base.mutex);
    }

    bridge->fe_state = *state;

    if (bridge->base.mutex) {
        nimcp_mutex_unlock(bridge->base.mutex);
    }

    return 0;
}

/**
 * WHAT: Update free energy wellbeing effects
 * WHY:  Compute wellbeing effects from free energy state
 * HOW:  Analyze FE, precision, coherence; compute effects
 */
int free_energy_bridge_update_effects(free_energy_bridge_t* bridge) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (bridge->base.mutex) {
        nimcp_mutex_lock(bridge->base.mutex);
    }

    /* Compute prediction error distress */
    if (bridge->config.enable_prediction_error_effects) {
        bridge->effects.prediction_error_distress = compute_prediction_error_distress(
            bridge->fe_state.prediction_error,
            bridge->config.prediction_error_sensitivity
        );
    } else {
        bridge->effects.prediction_error_distress = 0.0f;
    }

    /* Compute precision-based uncertainty */
    if (bridge->config.enable_precision_effects) {
        bridge->effects.uncertainty_distress = compute_precision_uncertainty(
            bridge->fe_state.precision,
            bridge->config.low_precision_threshold,
            bridge->config.precision_sensitivity
        );
        bridge->effects.precision_level = bridge->fe_state.precision;
        bridge->effects.confidence_level = bridge->fe_state.precision;
    } else {
        bridge->effects.uncertainty_distress = 0.0f;
    }

    /* Compute model coherence identity effect */
    if (bridge->config.enable_model_coherence_effects) {
        bridge->identity_stability = compute_coherence_identity_effect(
            bridge->model_coherence,
            bridge->config.critical_coherence_threshold
        );
        bridge->effects.identity_stability = bridge->identity_stability;
        bridge->effects.model_coherence = bridge->model_coherence;
    }

    /* Compute epistemic wellbeing */
    bridge->effects.epistemic_wellbeing = compute_epistemic_wellbeing(
        bridge->fe_state.free_energy,
        bridge->fe_state.precision,
        bridge->prediction_success_rate
    );

    /* Compute free energy level */
    bridge->effects.free_energy_level = bridge->fe_state.free_energy;
    bridge->effects.high_uncertainty = bridge->fe_state.free_energy > bridge->config.high_fe_threshold;
    bridge->effects.prediction_success_rate = bridge->prediction_success_rate;

    /* Track high FE events */
    if (bridge->fe_state.free_energy > bridge->config.high_fe_threshold) {
        bridge->high_fe_events++;
    }

    /* Track low precision events */
    if (bridge->fe_state.precision < bridge->config.low_precision_threshold) {
        bridge->low_precision_events++;
    }

    /* Update statistics */
    bridge->total_updates++;
    float alpha = 0.1f;
    bridge->avg_free_energy = alpha * bridge->fe_state.free_energy +
                              (1.0f - alpha) * bridge->avg_free_energy;
    bridge->avg_precision = alpha * bridge->fe_state.precision +
                            (1.0f - alpha) * bridge->avg_precision;

    if (bridge->base.mutex) {
        nimcp_mutex_unlock(bridge->base.mutex);
    }

    return 0;
}

/**
 * WHAT: Update model coherence
 * WHY:  Model coherence affects identity stability
 * HOW:  Track prediction success, compute coherence metric
 */
int free_energy_bridge_update_coherence(
    free_energy_bridge_t* bridge,
    bool prediction_correct
) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (bridge->base.mutex) {
        nimcp_mutex_lock(bridge->base.mutex);
    }

    /* Update prediction counts */
    bridge->predictions_made++;
    if (prediction_correct) {
        bridge->predictions_correct++;
    }

    /* Compute success rate with exponential moving average */
    float new_success = prediction_correct ? 1.0f : 0.0f;
    float alpha = 0.1f;  /* Smoothing factor */
    bridge->prediction_success_rate = alpha * new_success +
                                      (1.0f - alpha) * bridge->prediction_success_rate;

    /* Update model coherence based on success rate */
    float old_coherence = bridge->model_coherence;
    bridge->model_coherence = 0.3f * bridge->prediction_success_rate +
                              0.7f * bridge->model_coherence;

    /* Update coherence trend */
    bridge->coherence_trend = bridge->model_coherence - old_coherence;

    /* Check for coherence warnings */
    if (bridge->model_coherence < bridge->config.critical_coherence_threshold) {
        bridge->coherence_warnings++;
        NIMCP_LOGGING_WARN("Model coherence critically low: %.3f",
                           bridge->model_coherence);
    }

    if (bridge->base.mutex) {
        nimcp_mutex_unlock(bridge->base.mutex);
    }

    return 0;
}

/* ============================================================================
 * Effect Computation API Implementation
 * ============================================================================ */

/**
 * WHAT: Compute prediction error distress
 * WHY:  High prediction errors cause uncertainty distress
 * HOW:  Map prediction error to distress with sensitivity scaling
 */
float compute_prediction_error_distress(float prediction_error, float sensitivity) {
    if (prediction_error <= 0.0f) {
        return 0.0f;
    }
    if (prediction_error >= 1.0f) {
        return fminf(1.0f, sensitivity);
    }

    /* Sigmoid-like function for smooth distress onset */
    float x = prediction_error * 3.0f;  /* Scale for steeper curve */
    float distress = prediction_error * prediction_error * sensitivity;

    return fminf(1.0f, fmaxf(0.0f, distress));
}

/**
 * WHAT: Compute precision-based uncertainty
 * WHY:  Low precision means high uncertainty → distress
 * HOW:  Inverse map precision to uncertainty distress
 */
float compute_precision_uncertainty(
    float precision,
    float threshold,
    float sensitivity
) {
    if (precision >= threshold) {
        return 0.0f;
    }
    if (precision <= 0.0f) {
        return fminf(1.0f, sensitivity);
    }

    /* Linear scaling below threshold */
    float uncertainty = (threshold - precision) / threshold;
    uncertainty *= sensitivity;

    return fminf(1.0f, fmaxf(0.0f, uncertainty));
}

/**
 * WHAT: Compute model coherence identity effect
 * WHY:  Coherent self-model → stable identity
 * HOW:  Map coherence to identity stability score
 */
float compute_coherence_identity_effect(
    float coherence,
    float critical_threshold
) {
    if (coherence >= 0.8f) {
        return 1.0f;  /* Fully stable identity */
    }
    if (coherence <= critical_threshold) {
        /* Critical zone - identity at risk */
        return coherence / critical_threshold * 0.3f;
    }

    /* Linear interpolation in middle zone */
    float range = 0.8f - critical_threshold;
    float position = (coherence - critical_threshold) / range;
    return 0.3f + position * 0.7f;
}

/**
 * WHAT: Compute epistemic wellbeing
 * WHY:  Good model fit promotes satisfaction
 * HOW:  Combine prediction success, low FE, high coherence
 */
float compute_epistemic_wellbeing(
    float free_energy,
    float precision,
    float prediction_success_rate
) {
    /* Low free energy contributes to wellbeing */
    float fe_contribution = 1.0f - free_energy;

    /* High precision contributes to wellbeing */
    float precision_contribution = precision;

    /* High prediction success contributes to wellbeing */
    float success_contribution = prediction_success_rate;

    /* Weighted combination */
    float wellbeing = 0.3f * fe_contribution +
                      0.3f * precision_contribution +
                      0.4f * success_contribution;

    return fminf(1.0f, fmaxf(0.0f, wellbeing));
}

/* ============================================================================
 * Query API Implementation
 * ============================================================================ */

/**
 * WHAT: Get current free energy effects
 * WHY:  External access to computed effects
 * HOW:  Copy effects structure
 */
int free_energy_bridge_get_effects(
    const free_energy_bridge_t* bridge,
    free_energy_wellbeing_effects_t* effects
) {
    if (!bridge || !effects) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (bridge->base.mutex) {
        nimcp_mutex_lock((nimcp_mutex_t*)bridge->base.mutex);
    }

    *effects = bridge->effects;

    if (bridge->base.mutex) {
        nimcp_mutex_unlock((nimcp_mutex_t*)bridge->base.mutex);
    }

    return 0;
}

/**
 * WHAT: Get model coherence
 * WHY:  Query coherence metric
 * HOW:  Return model_coherence field
 */
float free_energy_bridge_get_coherence(const free_energy_bridge_t* bridge) {
    if (!bridge) {
        return 0.0f;
    }
    return bridge->model_coherence;
}

/**
 * WHAT: Get identity stability
 * WHY:  Query identity stability from coherence
 * HOW:  Return identity_stability field
 */
float free_energy_bridge_get_identity_stability(
    const free_energy_bridge_t* bridge
) {
    if (!bridge) {
        return 0.0f;
    }
    return bridge->identity_stability;
}

/**
 * WHAT: Get epistemic wellbeing
 * WHY:  Query epistemic wellbeing score
 * HOW:  Return from effects structure
 */
float free_energy_bridge_get_epistemic_wellbeing(
    const free_energy_bridge_t* bridge
) {
    if (!bridge) {
        return 0.0f;
    }
    return bridge->effects.epistemic_wellbeing;
}

/**
 * WHAT: Check if high uncertainty
 * WHY:  Query uncertainty state
 * HOW:  Check precision below threshold
 */
bool free_energy_bridge_is_high_uncertainty(const free_energy_bridge_t* bridge) {
    if (!bridge) {
        return false;
    }
    return bridge->fe_state.precision < bridge->config.low_precision_threshold;
}

/**
 * WHAT: Check if identity at risk
 * WHY:  Query identity stability state
 * HOW:  Check coherence below critical threshold
 */
bool free_energy_bridge_is_identity_at_risk(const free_energy_bridge_t* bridge) {
    if (!bridge) {
        return false;
    }
    return bridge->model_coherence < bridge->config.critical_coherence_threshold;
}

/**
 * WHAT: Get statistics
 * WHY:  Monitoring and diagnostics
 * HOW:  Return accumulated statistics
 */
int free_energy_bridge_get_stats(
    const free_energy_bridge_t* bridge,
    uint64_t* total_updates,
    uint32_t* high_fe_events,
    uint32_t* low_precision_events,
    float* avg_free_energy
) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (total_updates) *total_updates = bridge->total_updates;
    if (high_fe_events) *high_fe_events = bridge->high_fe_events;
    if (low_precision_events) *low_precision_events = bridge->low_precision_events;
    if (avg_free_energy) *avg_free_energy = bridge->avg_free_energy;

    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Free Energy Wellbeing Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int free_energy_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Free_Energy_Wellbeing_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("Free Energy Wellbeing Bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Free_Energy_Wellbeing_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Free_Energy_Wellbeing_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
