/**
 * @file nimcp_mental_health_fep_bridge.c
 * @brief Free Energy Principle - Mental Health Integration Bridge Implementation
 */

#include "cognitive/mental_health/nimcp_mental_health_fep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

#define LOG_MODULE "mental_health_fep_bridge"

int mental_health_fep_bridge_default_config(mental_health_fep_config_t* config) {
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_POINTER, "config is NULL");
    config->aberrant_precision_threshold = MENTAL_HEALTH_FEP_ABERRANT_PRECISION_THRESHOLD;
    config->pathological_lr_threshold = MENTAL_HEALTH_FEP_PATHOLOGICAL_LR_THRESHOLD;
    config->negative_prior_threshold = -1.0f;
    config->enable_aberrant_precision_detection = true;
    config->enable_pathological_learning_detection = true;
    config->enable_negative_prior_detection = true;
    config->intervention_precision_correction = 0.5f;
    config->intervention_lr_correction = 0.1f;
    config->enable_precision_intervention = true;
    config->enable_lr_intervention = true;
    config->fe_sensitivity = 1.0f;
    config->mental_health_sensitivity = 1.0f;
    return 0;
}

mental_health_fep_bridge_t* mental_health_fep_bridge_create(const mental_health_fep_config_t* config) {
    mental_health_fep_bridge_t* bridge = nimcp_malloc(sizeof(mental_health_fep_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    memset(bridge, 0, sizeof(mental_health_fep_bridge_t));
    if (config) bridge->config = *config;
    else mental_health_fep_bridge_default_config(&bridge->config);
    if (bridge_base_init(&bridge->base, 0, "mental_health_fep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }
    return bridge;
}

void mental_health_fep_bridge_destroy(mental_health_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.bio_async_enabled) mental_health_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int mental_health_fep_bridge_connect_fep(mental_health_fep_bridge_t* bridge, fep_system_t* fep) {
    NIMCP_CHECK_THROW(bridge && fep, NIMCP_ERROR_NULL_POINTER, "bridge or fep is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_fep_bridge_connect_mental_health(mental_health_fep_bridge_t* bridge, mental_health_monitor_t* mh) {
    NIMCP_CHECK_THROW(bridge && mh, NIMCP_ERROR_NULL_POINTER, "bridge or mental_health is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->mental_health_system = mh;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_fep_bridge_disconnect(mental_health_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    bridge->mental_health_system = NULL;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_fep_detect_aberrant_precision(mental_health_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");
    if (!bridge->config.enable_aberrant_precision_detection) return 0;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->state.current_precision > bridge->config.aberrant_precision_threshold) {
        bridge->fep_effects.aberrant_precision_detected = true;
        bridge->state.pathology_detected = true;
        bridge->stats.aberrant_precision_events++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_fep_detect_pathological_learning(mental_health_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");
    if (!bridge->config.enable_pathological_learning_detection) return 0;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->state.current_learning_rate < bridge->config.pathological_lr_threshold) {
        bridge->fep_effects.pathological_learning_detected = true;
        bridge->state.pathology_detected = true;
        bridge->stats.pathological_learning_events++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_fep_detect_negative_priors(mental_health_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");
    if (!bridge->config.enable_negative_prior_detection) return 0;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_effects.negative_priors_detected = false;
    bridge->stats.negative_prior_events++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_fep_apply_precision_intervention(mental_health_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");
    if (!bridge->config.enable_precision_intervention) return 0;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->fep_effects.aberrant_precision_detected) {
        bridge->mental_health_effects.precision_correction = bridge->config.intervention_precision_correction;
        bridge->mental_health_effects.intervention_active = true;
        bridge->stats.intervention_events++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_fep_apply_lr_intervention(mental_health_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");
    if (!bridge->config.enable_lr_intervention) return 0;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->fep_effects.pathological_learning_detected) {
        bridge->mental_health_effects.lr_correction = bridge->config.intervention_lr_correction;
        bridge->mental_health_effects.intervention_active = true;
        bridge->stats.intervention_events++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_fep_bridge_update(mental_health_fep_bridge_t* bridge, uint64_t delta_ms) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    mental_health_fep_detect_aberrant_precision(bridge);
    mental_health_fep_detect_pathological_learning(bridge);
    mental_health_fep_detect_negative_priors(bridge);
    mental_health_fep_apply_precision_intervention(bridge);
    mental_health_fep_apply_lr_intervention(bridge);
    return 0;
}

int mental_health_fep_bridge_get_state(const mental_health_fep_bridge_t* bridge, mental_health_fep_state_t* state) {
    NIMCP_CHECK_THROW(bridge && state, NIMCP_ERROR_NULL_POINTER, "bridge or state is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_fep_bridge_get_stats(const mental_health_fep_bridge_t* bridge, mental_health_fep_stats_t* stats) {
    NIMCP_CHECK_THROW(bridge && stats, NIMCP_ERROR_NULL_POINTER, "bridge or stats is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_fep_bridge_connect_bio_async(mental_health_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (bridge->base.bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_MENTAL_HEALTH_BRIDGE,
        .module_name = "mental_health_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };
    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) bridge->base.bio_async_enabled = true;
    return 0;
}

int mental_health_fep_bridge_disconnect_bio_async(mental_health_fep_bridge_t* bridge) {
    if (!bridge || !bridge->base.bio_async_enabled) return 0;
    if (bridge->base.bio_ctx) bio_router_unregister_module(bridge->base.bio_ctx);
    bridge->base.bio_ctx = NULL;
    bridge->base.bio_async_enabled = false;
    return 0;
}

bool mental_health_fep_bridge_is_bio_async_connected(const mental_health_fep_bridge_t* bridge) {
    return bridge ? bridge->base.bio_async_enabled : false;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int mental_health_fep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Mental_Health_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Mental_Health_FEP_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Mental_Health_FEP_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
