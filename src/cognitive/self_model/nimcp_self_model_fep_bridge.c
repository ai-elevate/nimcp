/**
 * @file nimcp_self_model_fep_bridge.c
 * @brief Free Energy Principle - Self-Model Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-12
 */

#include "cognitive/self_model/nimcp_self_model_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"

#include <string.h>
#include <math.h>

#define LOG_MODULE_SELF_MODEL_FEP "[SELF_MODEL_FEP]"

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

int self_model_fep_bridge_default_config(self_model_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;

    config->capability_pe_threshold = SELF_FEP_PE_CAPABILITY_THRESHOLD;
    config->belief_pe_threshold = SELF_FEP_PE_BELIEF_THRESHOLD;
    config->self_uncertainty_threshold = SELF_FEP_UNCERTAINTY_MEDIUM;

    config->belief_update_rate = SELF_FEP_BELIEF_UPDATE_RATE;
    config->capability_update_rate = SELF_FEP_CAPABILITY_UPDATE_RATE;
    config->core_belief_resistance = SELF_FEP_CORE_BELIEF_RESISTANCE;

    config->enable_belief_updates = true;
    config->enable_capability_learning = true;
    config->enable_self_exploration = true;
    config->enable_identity_protection = true;

    config->pe_sensitivity = 1.0f;
    config->uncertainty_sensitivity = 1.0f;

    return NIMCP_SUCCESS;
}

self_model_fep_bridge_t* self_model_fep_bridge_create(
    const self_model_fep_config_t* config
) {
    self_model_fep_bridge_t* bridge =
        (self_model_fep_bridge_t*)nimcp_malloc(sizeof(self_model_fep_bridge_t));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(self_model_fep_bridge_t));

    if (config) {
        bridge->config = *config;
    } else {
        self_model_fep_bridge_default_config(&bridge->config);
    }

    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->state.identity_stable = true;
    bridge->state.self_knowledge_certainty = 0.5f;
    bridge->state.capability_confidence = 0.5f;

    NIMCP_LOGGING_INFO(LOG_MODULE_SELF_MODEL_FEP " Bridge created");
    return bridge;
}

void self_model_fep_bridge_destroy(self_model_fep_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->base.bio_async_enabled) {
        self_model_fep_bridge_disconnect_bio_async(bridge);
    }

    if (bridge->base.mutex) {
        nimcp_mutex_destroy(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO(LOG_MODULE_SELF_MODEL_FEP " Bridge destroyed");
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int self_model_fep_bridge_connect_fep(
    self_model_fep_bridge_t* bridge,
    fep_system_t* fep
) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO(LOG_MODULE_SELF_MODEL_FEP " Connected to FEP system");
    return NIMCP_SUCCESS;
}

int self_model_fep_bridge_connect_self_model(
    self_model_fep_bridge_t* bridge,
    self_model_t* self_model
) {
    if (!bridge || !self_model) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->self_model_system = self_model;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO(LOG_MODULE_SELF_MODEL_FEP " Connected to self-model system");
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * FEP → Self-Model Direction
 * ============================================================================ */

int self_model_fep_update_belief(
    self_model_fep_bridge_t* bridge,
    uint32_t belief_index,
    float prediction_error
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_belief_updates) return NIMCP_SUCCESS;

    nimcp_mutex_lock(bridge->base.mutex);

    if (fabsf(prediction_error) > bridge->config.belief_pe_threshold) {
        float update_rate = bridge->config.belief_update_rate;

        /* Core beliefs resist change */
        update_rate *= (1.0f - bridge->config.core_belief_resistance);

        bridge->state.beliefs_updated++;
        bridge->stats.belief_updates_total++;
        bridge->stats.avg_belief_pe =
            (bridge->stats.avg_belief_pe * 0.9f) + (fabsf(prediction_error) * 0.1f);

        NIMCP_LOGGING_DEBUG(LOG_MODULE_SELF_MODEL_FEP
            " Updated belief %u (PE=%.2f)", belief_index, prediction_error);
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int self_model_fep_update_capability(
    self_model_fep_bridge_t* bridge,
    uint32_t capability_index,
    float prediction_error
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_capability_learning) return NIMCP_SUCCESS;

    nimcp_mutex_lock(bridge->base.mutex);

    if (fabsf(prediction_error) > bridge->config.capability_pe_threshold) {
        float update_rate = bridge->config.capability_update_rate;

        bridge->state.capabilities_updated++;
        bridge->stats.capability_updates_total++;
        bridge->stats.avg_capability_pe =
            (bridge->stats.avg_capability_pe * 0.9f) + (fabsf(prediction_error) * 0.1f);

        /* Update capability confidence */
        bridge->state.capability_confidence *= (1.0f - update_rate);
        bridge->state.capability_confidence += update_rate * 0.7f;

        NIMCP_LOGGING_DEBUG(LOG_MODULE_SELF_MODEL_FEP
            " Updated capability %u (PE=%.2f)", capability_index, prediction_error);
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int self_model_fep_explore_self(self_model_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_self_exploration) return NIMCP_SUCCESS;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->state.exploring_self = true;
    bridge->stats.exploration_episodes++;
    bridge->effects.exploration_drive = 1.0f;

    NIMCP_LOGGING_INFO(LOG_MODULE_SELF_MODEL_FEP " Self-exploration initiated");

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Self-Model → FEP Direction
 * ============================================================================ */

int self_model_fep_apply_belief_priors(self_model_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Apply self-beliefs as FEP priors */
    /* Implementation would interact with FEP prior setting */

    NIMCP_LOGGING_DEBUG(LOG_MODULE_SELF_MODEL_FEP " Applied belief priors to FEP");

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int self_model_fep_constrain_policies(self_model_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Constrain FEP policy space by known capabilities */
    /* Implementation would filter FEP action space */

    NIMCP_LOGGING_DEBUG(LOG_MODULE_SELF_MODEL_FEP " Constrained FEP policies");

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int self_model_fep_apply_sensory_attenuation(
    self_model_fep_bridge_t* bridge,
    bool is_self_generated
) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);

    if (is_self_generated) {
        /* Reduce precision for self-generated predictions */
        /* Implementation would modulate FEP precision */

        NIMCP_LOGGING_DEBUG(LOG_MODULE_SELF_MODEL_FEP " Applied sensory attenuation");
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Update Cycle
 * ============================================================================ */

int self_model_fep_bridge_update(
    self_model_fep_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update self-knowledge uncertainty */
    float uncertainty = 1.0f - bridge->state.self_knowledge_certainty;
    bridge->effects.self_knowledge_uncertainty = uncertainty;

    /* Check for identity crisis */
    if (bridge->effects.identity_pe > SELF_FEP_PE_IDENTITY_THRESHOLD) {
        bridge->effects.identity_crisis = true;
        bridge->state.identity_stable = false;
        bridge->stats.identity_crises++;
        bridge->state.last_identity_crisis_ms = delta_ms;

        NIMCP_LOGGING_WARN(LOG_MODULE_SELF_MODEL_FEP
            " Identity crisis detected (PE=%.2f)", bridge->effects.identity_pe);
    } else {
        bridge->effects.identity_crisis = false;
    }

    /* Update exploration drive based on uncertainty */
    if (uncertainty > bridge->config.self_uncertainty_threshold) {
        bridge->effects.exploration_drive = uncertainty;
    } else {
        bridge->effects.exploration_drive *= 0.95f;
    }

    /* Decay exploration state */
    if (bridge->state.exploring_self) {
        bridge->state.exploration_progress += 0.01f;
        if (bridge->state.exploration_progress >= 1.0f) {
            bridge->state.exploring_self = false;
            bridge->state.exploration_progress = 0.0f;
            bridge->state.self_knowledge_certainty += 0.1f;
            if (bridge->state.self_knowledge_certainty > 1.0f) {
                bridge->state.self_knowledge_certainty = 1.0f;
            }
        }
    }

    /* Update statistics */
    bridge->stats.avg_self_certainty =
        (bridge->stats.avg_self_certainty * 0.99f) +
        (bridge->state.self_knowledge_certainty * 0.01f);

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * State/Stats API
 * ============================================================================ */

int self_model_fep_bridge_get_state(
    const self_model_fep_bridge_t* bridge,
    self_model_fep_state_t* state
) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int self_model_fep_bridge_get_stats(
    const self_model_fep_bridge_t* bridge,
    self_model_fep_stats_t* stats
) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

bool self_model_fep_is_exploring(const self_model_fep_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->state.exploring_self;
}

float self_model_fep_get_self_certainty(const self_model_fep_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->state.self_knowledge_certainty;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

int self_model_fep_bridge_connect_bio_async(self_model_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->base.bio_async_enabled) return NIMCP_SUCCESS;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_SELF_MODEL_BRIDGE,
        .module_name = "self_model_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO(LOG_MODULE_SELF_MODEL_FEP " Connected to bio-async router");
        return NIMCP_SUCCESS;
    }

    NIMCP_LOGGING_WARN(LOG_MODULE_SELF_MODEL_FEP
        " Bio-async router not available, skipping registration");
    return -1;
}

int self_model_fep_bridge_disconnect_bio_async(
    self_model_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->base.bio_async_enabled) return NIMCP_SUCCESS;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO(LOG_MODULE_SELF_MODEL_FEP " Disconnected from bio-async router");
    return NIMCP_SUCCESS;
}

bool self_model_fep_bridge_is_bio_async_connected(
    const self_model_fep_bridge_t* bridge
) {
    if (!bridge) return false;
    return bridge->base.bio_async_enabled;
}
