/**
 * @file nimcp_mirror_neurons_fep_bridge.c
 * @brief Free Energy Principle - Mirror Neurons Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-12
 */

#include "cognitive/mirror_neurons/nimcp_mirror_neurons_fep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static inline float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int mirror_neurons_fep_bridge_default_config(mirror_neurons_fep_config_t* config) {
    if (!config) return -1;

    config->precision_gain_factor = 1.0f;
    config->min_mirror_gain = FEP_PRECISION_MIRROR_GAIN_MIN;
    config->max_mirror_gain = FEP_PRECISION_MIRROR_GAIN_MAX;

    config->goal_belief_coupling_rate = 0.3f;
    config->belief_goal_coupling_rate = 0.3f;
    config->confidence_threshold = GOAL_BELIEF_CONFIDENCE_THRESHOLD;

    config->motor_evidence_weight = 0.5f;
    config->motor_evidence_threshold = MOTOR_EVIDENCE_THRESHOLD;

    config->motor_pe_sensitivity = 1.0f;
    config->goal_pe_sensitivity = 1.0f;

    config->enable_precision_modulation = true;
    config->enable_goal_belief_coupling = true;
    config->enable_motor_evidence = true;
    config->enable_pe_propagation = true;

    return 0;
}

mirror_neurons_fep_bridge_t* mirror_neurons_fep_bridge_create(
    const mirror_neurons_fep_config_t* config
) {
    mirror_neurons_fep_bridge_t* bridge = (mirror_neurons_fep_bridge_t*)nimcp_calloc(
        1, sizeof(mirror_neurons_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate mirror neurons-FEP bridge");
        return NULL;
    }

    /* Apply configuration */
    mirror_neurons_fep_config_t default_cfg;
    if (!config) {
        mirror_neurons_fep_bridge_default_config(&default_cfg);
        config = &default_cfg;
    }
    bridge->config = *config;

    /* Initialize state */
    memset(&bridge->state, 0, sizeof(mirror_neurons_fep_state_t));
    bridge->state.current_precision = FEP_PRECISION_DEFAULT;

    /* Initialize effects */
    memset(&bridge->effects, 0, sizeof(mirror_neurons_fep_effects_t));
    bridge->effects.mirror_gain_modulation = 1.0f;

    /* Initialize stats */
    memset(&bridge->stats, 0, sizeof(mirror_neurons_fep_stats_t));
    bridge->stats.avg_mirror_gain = 1.0f;
    bridge->stats.min_mirror_gain_applied = 1.0f;
    bridge->stats.max_mirror_gain_applied = 1.0f;

    /* Create mutex */
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        mirror_neurons_fep_bridge_destroy(bridge);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Mirror neurons-FEP bridge created");
    return bridge;
}

void mirror_neurons_fep_bridge_destroy(mirror_neurons_fep_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->base.bio_async_enabled) {
        mirror_neurons_fep_bridge_disconnect_bio_async(bridge);
    }

    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Mirror neurons-FEP bridge destroyed");
}

/* ============================================================================
 * Connection Implementation
 * ============================================================================ */

int mirror_neurons_fep_bridge_connect_fep(
    mirror_neurons_fep_bridge_t* bridge,
    fep_system_t* fep
) {
    if (!bridge) return -1;
    /* Allow NULL fep to disconnect/reset FEP connection */

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Mirror neurons-FEP bridge connected to FEP system");
    return 0;
}

int mirror_neurons_fep_bridge_connect_mirror_neurons(
    mirror_neurons_fep_bridge_t* bridge,
    mirror_hierarchy_t mirror
) {
    if (!bridge || !mirror) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->mirror_system = mirror;
    nimcp_platform_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Mirror neurons-FEP bridge connected to mirror neuron system");
    return 0;
}

/* ============================================================================
 * FEP → Mirror Neurons Implementation
 * ============================================================================ */

int mirror_neurons_fep_apply_precision_modulation(
    mirror_neurons_fep_bridge_t* bridge
) {
    if (!bridge || !bridge->fep_system || !bridge->mirror_system) return -1;
    if (!bridge->config.enable_precision_modulation) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get average precision across FEP hierarchy */
    fep_system_t* fep = bridge->fep_system;
    float total_precision = 0.0f;
    uint32_t precision_count = 0;

    for (uint32_t l = 0; l < fep->num_levels; l++) {
        fep_hierarchy_level_t* level = &fep->levels[l];
        for (uint32_t i = 0; i < level->errors.dim; i++) {
            total_precision += level->errors.precision[i];
            precision_count++;
        }
    }

    float avg_precision = precision_count > 0 ? total_precision / precision_count : 1.0f;
    bridge->state.current_precision = avg_precision;

    /* Map precision to mirror gain */
    float gain = clamp_f(
        avg_precision * bridge->config.precision_gain_factor,
        bridge->config.min_mirror_gain,
        bridge->config.max_mirror_gain
    );

    bridge->effects.mirror_gain_modulation = gain;

    /* Update statistics */
    bridge->stats.avg_mirror_gain =
        (bridge->stats.avg_mirror_gain * 0.99f) + (gain * 0.01f);
    if (gain < bridge->stats.min_mirror_gain_applied) {
        bridge->stats.min_mirror_gain_applied = gain;
    }
    if (gain > bridge->stats.max_mirror_gain_applied) {
        bridge->stats.max_mirror_gain_applied = gain;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_neurons_fep_update_goals_from_beliefs(
    mirror_neurons_fep_bridge_t* bridge
) {
    if (!bridge || !bridge->fep_system || !bridge->mirror_system) return -1;
    if (!bridge->config.enable_goal_belief_coupling) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    fep_system_t* fep = bridge->fep_system;

    /* Use top-level FEP beliefs as goal activations */
    if (fep->num_levels > 0) {
        fep_belief_t* beliefs = &fep->levels[fep->num_levels - 1].beliefs;

        /* Map belief means to goal activations (simplified) */
        uint32_t max_goals = beliefs->dim < 32 ? beliefs->dim : 32;

        for (uint32_t g = 0; g < max_goals; g++) {
            float belief_activation = beliefs->mean[g];

            /* Only update if above confidence threshold */
            if (belief_activation >= bridge->config.confidence_threshold) {
                /* Goal activation boost from FEP beliefs */
                bridge->effects.goal_activation_boost =
                    belief_activation * bridge->config.belief_goal_coupling_rate;

                bridge->state.belief_goal_updates++;
                bridge->stats.total_belief_updates++;
            }
        }
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_neurons_fep_propagate_prediction_errors(
    mirror_neurons_fep_bridge_t* bridge
) {
    if (!bridge || !bridge->fep_system || !bridge->mirror_system) return -1;
    if (!bridge->config.enable_pe_propagation) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    fep_system_t* fep = bridge->fep_system;

    /* Compute average prediction errors */
    float total_pe = 0.0f;
    uint32_t pe_count = 0;

    for (uint32_t l = 0; l < fep->num_levels; l++) {
        total_pe += fep->levels[l].errors.magnitude;
        pe_count++;
    }

    float avg_pe = pe_count > 0 ? total_pe / pe_count : 0.0f;

    /* Scale by sensitivity */
    bridge->effects.motor_prediction_error = avg_pe * bridge->config.motor_pe_sensitivity;
    bridge->state.avg_motor_pe =
        (bridge->state.avg_motor_pe * 0.9f) + (bridge->effects.motor_prediction_error * 0.1f);

    /* Update statistics */
    bridge->stats.avg_motor_pe = bridge->state.avg_motor_pe;
    if (avg_pe > MOTOR_PE_THRESHOLD_HIGH) {
        bridge->stats.high_motor_pe_count++;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Mirror Neurons → FEP Implementation
 * ============================================================================ */

int mirror_neurons_fep_transfer_goals_to_beliefs(
    mirror_neurons_fep_bridge_t* bridge
) {
    if (!bridge || !bridge->fep_system || !bridge->mirror_system) return -1;
    if (!bridge->config.enable_goal_belief_coupling) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    mirror_hierarchy_t mirror = bridge->mirror_system;
    fep_system_t* fep = bridge->fep_system;

    /* Get mirror neuron statistics to extract goal info */
    mirror_hierarchy_stats_t mirror_stats;
    if (mirror_hierarchy_get_stats(mirror, &mirror_stats)) {
        bridge->state.active_goals = mirror_stats.active_goals;
        bridge->state.max_goal_activation = mirror_stats.mean_goal_activation;

        /* Transfer to FEP top-level beliefs (simplified) */
        if (fep->num_levels > 0 && mirror_stats.active_goals > 0) {
            fep_belief_t* beliefs = &fep->levels[fep->num_levels - 1].beliefs;

            /* Update belief means based on goal activations */
            float coupling_rate = bridge->config.goal_belief_coupling_rate;

            for (uint32_t i = 0; i < beliefs->dim && i < 32; i++) {
                /* Simplified: blend current belief with goal-derived evidence */
                beliefs->mean[i] = beliefs->mean[i] * (1.0f - coupling_rate) +
                                  (mirror_stats.mean_goal_activation * coupling_rate);
            }

            bridge->state.goal_belief_updates++;
            bridge->stats.total_goal_updates++;
        }
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_neurons_fep_provide_motor_evidence(
    mirror_neurons_fep_bridge_t* bridge
) {
    if (!bridge || !bridge->fep_system || !bridge->mirror_system) return -1;
    if (!bridge->config.enable_motor_evidence) return 0;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    mirror_hierarchy_t mirror = bridge->mirror_system;

    /* Get motor activation statistics */
    mirror_hierarchy_stats_t mirror_stats;
    if (mirror_hierarchy_get_stats(mirror, &mirror_stats)) {
        bridge->state.active_motors = mirror_stats.active_motors;
        bridge->state.max_motor_activation = mirror_stats.mean_motor_activation;

        /* Provide motor activations as sensory evidence if above threshold */
        if (mirror_stats.mean_motor_activation >= bridge->config.motor_evidence_threshold) {
            bridge->state.motor_evidence_samples++;
            bridge->stats.total_motor_evidence++;
        }
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mirror_neurons_fep_set_precision_from_resonance(
    mirror_neurons_fep_bridge_t* bridge
) {
    if (!bridge || !bridge->fep_system || !bridge->mirror_system) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get mirror neuron resonance strength */
    mirror_hierarchy_stats_t mirror_stats;
    if (mirror_hierarchy_get_stats(bridge->mirror_system, &mirror_stats)) {
        /* Use mean goal/motor activation as resonance strength proxy */
        float resonance = (mirror_stats.mean_goal_activation +
                          mirror_stats.mean_motor_activation) * 0.5f;

        /* Map resonance to FEP precision */
        fep_system_t* fep = bridge->fep_system;
        float precision_scale = clamp_f(resonance, 0.5f, 1.5f);

        /* Apply to lowest FEP level (sensory) */
        if (fep->num_levels > 0) {
            fep_hierarchy_level_t* sensory = &fep->levels[0];
            for (uint32_t i = 0; i < sensory->errors.dim; i++) {
                sensory->errors.precision[i] *= precision_scale;
            }
        }
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Update Cycle Implementation
 * ============================================================================ */

int mirror_neurons_fep_bridge_update(
    mirror_neurons_fep_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) return -1;

    /* FEP → Mirror Neurons */
    mirror_neurons_fep_apply_precision_modulation(bridge);
    mirror_neurons_fep_update_goals_from_beliefs(bridge);
    mirror_neurons_fep_propagate_prediction_errors(bridge);

    /* Mirror Neurons → FEP */
    mirror_neurons_fep_transfer_goals_to_beliefs(bridge);
    mirror_neurons_fep_provide_motor_evidence(bridge);
    mirror_neurons_fep_set_precision_from_resonance(bridge);

    /* Update current free energy for stats */
    if (bridge->fep_system) {
        bridge->state.current_free_energy = bridge->fep_system->free_energy.total;
    }

    return 0;
}

/* ============================================================================
 * State/Stats Implementation
 * ============================================================================ */

int mirror_neurons_fep_bridge_get_state(
    const mirror_neurons_fep_bridge_t* bridge,
    mirror_neurons_fep_state_t* state
) {
    if (!bridge || !state) return -1;
    *state = bridge->state;
    return 0;
}

int mirror_neurons_fep_bridge_get_stats(
    const mirror_neurons_fep_bridge_t* bridge,
    mirror_neurons_fep_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Bio-Async Implementation
 * ============================================================================ */

int mirror_neurons_fep_bridge_connect_bio_async(
    mirror_neurons_fep_bridge_t* bridge
) {
    if (!bridge) return -1;
    if (bridge->base.bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_MIRROR_NEURONS_BRIDGE,
        .module_name = "mirror_neurons_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Mirror neurons-FEP bridge connected to bio-async");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    }
    return 0;
}

int mirror_neurons_fep_bridge_disconnect_bio_async(
    mirror_neurons_fep_bridge_t* bridge
) {
    if (!bridge) return -1;
    if (!bridge->base.bio_async_enabled) return 0;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }
    bridge->base.bio_async_enabled = false;
    return 0;
}

bool mirror_neurons_fep_bridge_is_bio_async_connected(
    const mirror_neurons_fep_bridge_t* bridge
) {
    return bridge && bridge->base.bio_async_enabled;
}

/* ============================================================================
 * KG Self-Awareness Integration
 * ============================================================================ */

int mirror_neurons_fep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Mirror_Neurons_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("Mirror neurons FEP bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Mirror_Neurons_FEP_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Mirror_Neurons_FEP_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
