/**
 * @file nimcp_remorse_fep_bridge.c
 * @brief Free Energy Principle - Remorse/Regret Integration Bridge Implementation
 *
 * WHAT: Implements bidirectional integration between FEP and remorse/regret system
 * WHY:  Remorse arises from counterfactual inference - comparing actual outcomes
 *       to predicted better alternatives
 * HOW:  FEP counterfactual reasoning generates remorse; remorse modulates
 *       policy learning to avoid future violations
 */

#include "cognitive/remorse/nimcp_remorse_fep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>

#define LOG_MODULE "remorse_fep_bridge"

/* ============================================================================
 * Default Configuration
 * ============================================================================ */

int remorse_fep_default_config(remorse_fep_config_t* config) {
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_POINTER, "config is NULL");

    /* FEP -> Remorse */
    config->counterfactual_pe_gain = 1.0f;
    config->moral_violation_precision_boost = 1.5f;
    config->enable_counterfactual_pe = true;
    config->enable_moral_precision = true;

    /* Remorse -> FEP */
    config->remorse_policy_learning_gain = 2.0f;  /* Remorse strongly affects learning */
    config->guilt_value_update_gain = 1.5f;
    config->enable_remorse_policy_learning = true;
    config->enable_guilt_value_update = true;

    /* Sensitivity */
    config->fe_sensitivity = 1.0f;
    config->emotion_sensitivity = 1.0f;

    return 0;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

remorse_fep_bridge_t* remorse_fep_create(
    const remorse_fep_config_t* config
) {
    remorse_fep_bridge_t* bridge = nimcp_malloc(sizeof(remorse_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate remorse FEP bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    memset(bridge, 0, sizeof(remorse_fep_bridge_t));

    /* Set configuration */
    if (config) {
        bridge->config = *config;
    } else {
        remorse_fep_default_config(&bridge->config);
    }

    /* Create mutex */
    if (bridge_base_init(&bridge->base, 0, "remorse_fep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize defaults */
    bridge->emotion_effects.policy_learning_modifier = 1.0f;
    bridge->emotion_effects.value_aversion_strength = 0.0f;
    bridge->emotion_effects.avoidance_precision = 1.0f;

    NIMCP_LOGGING_INFO("Created remorse FEP bridge");
    return bridge;
}

void remorse_fep_destroy(remorse_fep_bridge_t* bridge) {
    if (!bridge) return;

    /* Disconnect bio-async if connected */
    if (bridge->base.bio_async_enabled) {
        remorse_fep_disconnect_bio_async(bridge);
    }

    /* Destroy mutex */
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed remorse FEP bridge");
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int remorse_fep_connect_fep(
    remorse_fep_bridge_t* bridge,
    fep_system_t* fep
) {
    NIMCP_CHECK_THROW(bridge && fep, NIMCP_ERROR_NULL_POINTER, "bridge or fep is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Connected FEP system to remorse bridge");
    return 0;
}

int remorse_fep_connect_remorse(
    remorse_fep_bridge_t* bridge,
    remorse_regret_system_t* remorse
) {
    NIMCP_CHECK_THROW(bridge && remorse, NIMCP_ERROR_NULL_POINTER, "bridge or remorse is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->remorse_system = remorse;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Connected remorse system to FEP bridge");
    return 0;
}

int remorse_fep_disconnect(remorse_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    bridge->remorse_system = NULL;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Disconnected all systems from remorse FEP bridge");
    return 0;
}

/* ============================================================================
 * FEP -> Remorse Direction
 * ============================================================================ */

int remorse_fep_compute_counterfactual_pe(
    remorse_fep_bridge_t* bridge,
    float actual_value,
    float alternative_value
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_counterfactual_pe) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Counterfactual PE = what could have been - what actually happened
     * Positive counterfactual PE = we could have done better -> remorse
     * Negative counterfactual PE = we did better than alternative -> relief
     */
    float counterfactual_pe = alternative_value - actual_value;
    counterfactual_pe *= bridge->config.counterfactual_pe_gain;

    bridge->fep_effects.counterfactual_pe = counterfactual_pe;

    /* Remorse intensity based on positive counterfactual PE */
    float remorse_intensity = 0.0f;
    float moral_precision = 1.0f;

    if (counterfactual_pe > 0.0f) {
        /* We could have done better -> remorse */
        remorse_intensity = counterfactual_pe;
        if (remorse_intensity > 1.0f) remorse_intensity = 1.0f;

        /* Moral violation increases precision on similar future situations */
        moral_precision = 1.0f + (remorse_intensity * bridge->config.moral_violation_precision_boost);
    }

    bridge->fep_effects.remorse_intensity = remorse_intensity;
    bridge->fep_effects.moral_violation_precision = moral_precision;

    /* Update state */
    bridge->state.current_counterfactual_pe = counterfactual_pe;
    bridge->state.remorse_level = remorse_intensity;

    /* Guilt is sustained remorse */
    if (remorse_intensity > 0.5f) {
        bridge->state.guilt_level = (bridge->state.guilt_level * 0.8f) + (remorse_intensity * 0.2f);
    }

    bridge->state.remorseful = (remorse_intensity > 0.3f);

    /* Update stats */
    if (remorse_intensity > 0.1f) {
        bridge->stats.remorse_events++;
    }
    bridge->stats.counterfactual_simulations++;
    bridge->stats.avg_remorse_intensity =
        (bridge->stats.avg_remorse_intensity * 0.9f) + (remorse_intensity * 0.1f);
    bridge->stats.avg_counterfactual_pe =
        (bridge->stats.avg_counterfactual_pe * 0.9f) + (fabsf(counterfactual_pe) * 0.1f);

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Computed counterfactual PE: %f, remorse=%f, moral_precision=%f",
                        counterfactual_pe, remorse_intensity, moral_precision);
    return 0;
}

int remorse_fep_modulate_policy_learning(
    remorse_fep_bridge_t* bridge
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_remorse_policy_learning) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Remorse strongly affects policy learning
     * The brain learns to avoid actions that led to remorse
     * This is a key mechanism for moral learning
     */
    float remorse_level = bridge->state.remorse_level;
    float guilt_level = bridge->state.guilt_level;

    /* Policy learning modifier - remorse increases learning from mistakes */
    float policy_modifier = 1.0f + (remorse_level * bridge->config.remorse_policy_learning_gain);
    bridge->emotion_effects.policy_learning_modifier = policy_modifier;

    /* Value aversion - guilt makes similar actions less attractive */
    float aversion = guilt_level * bridge->config.guilt_value_update_gain;
    bridge->emotion_effects.value_aversion_strength = aversion;

    /* Avoidance precision - remorse increases attention to avoiding similar situations */
    float avoidance_precision = 1.0f + (remorse_level * 0.5f) + (guilt_level * 0.5f);
    bridge->emotion_effects.avoidance_precision = avoidance_precision;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Modulated policy learning: modifier=%f, aversion=%f, avoidance=%f",
                        policy_modifier, aversion, avoidance_precision);
    return 0;
}

/* ============================================================================
 * Update API
 * ============================================================================ */

int remorse_fep_update(
    remorse_fep_bridge_t* bridge,
    uint64_t delta_ms
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    /* Apply policy learning modulation */
    remorse_fep_modulate_policy_learning(bridge);

    nimcp_mutex_lock(bridge->base.mutex);

    /* Remorse decays moderately - lessons are remembered */
    float remorse_decay = 0.998f;
    float guilt_decay = 0.9999f;  /* Guilt decays very slowly */

    bridge->state.remorse_level *= remorse_decay;
    bridge->state.guilt_level *= guilt_decay;

    if (bridge->state.remorse_level < 0.1f) {
        bridge->state.remorseful = false;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Query API
 * ============================================================================ */

int remorse_fep_get_state(
    const remorse_fep_bridge_t* bridge,
    remorse_fep_state_t* state
) {
    NIMCP_CHECK_THROW(bridge && state, NIMCP_ERROR_NULL_POINTER, "bridge or state is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int remorse_fep_get_stats(
    const remorse_fep_bridge_t* bridge,
    remorse_fep_stats_t* stats
) {
    NIMCP_CHECK_THROW(bridge && stats, NIMCP_ERROR_NULL_POINTER, "bridge or stats is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

int remorse_fep_connect_bio_async(
    remorse_fep_bridge_t* bridge
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (bridge->base.bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_REMORSE_BRIDGE,
        .module_name = "remorse_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available");
    }

    return 0;
}

int remorse_fep_disconnect_bio_async(
    remorse_fep_bridge_t* bridge
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->base.bio_async_enabled) return 0;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected from bio-async router");

    return 0;
}

bool remorse_fep_is_bio_async_connected(
    const remorse_fep_bridge_t* bridge
) {
    return bridge ? bridge->base.bio_async_enabled : false;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int remorse_fep_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Remorse_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Remorse_FEP_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Remorse_FEP_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
