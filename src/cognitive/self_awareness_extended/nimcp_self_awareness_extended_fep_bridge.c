/**
 * @file nimcp_self_awareness_extended_fep_bridge.c
 * @brief Extended Self-Awareness FEP Bridge Implementation
 */

#include "cognitive/self_awareness_extended/nimcp_self_awareness_extended_fep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

#define LOG_MODULE "self_awareness_extended_fep_bridge"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for self_awareness_extended_fep_bridge module */
static nimcp_health_agent_t* g_self_awareness_extended_fep_bridge_health_agent = NULL;

/**
 * @brief Set health agent for self_awareness_extended_fep_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void self_awareness_extended_fep_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_self_awareness_extended_fep_bridge_health_agent = agent;
}

/** @brief Send heartbeat from self_awareness_extended_fep_bridge module */
static inline void self_awareness_extended_fep_bridge_heartbeat(const char* operation, float progress) {
    if (g_self_awareness_extended_fep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_self_awareness_extended_fep_bridge_health_agent, operation, progress);
    }
}


int self_awareness_extended_fep_bridge_default_config(self_awareness_extended_fep_config_t* config) {
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_POINTER, "config is NULL");
    config->uncertainty_threshold = SELF_AWARENESS_FEP_HIGH_UNCERTAINTY_THRESHOLD;
    config->coherence_factor = SELF_AWARENESS_FEP_COHERENCE_FACTOR;
    config->enable_metacognitive_monitoring = true;
    config->enable_precision_modulation = true;
    config->enable_self_harm_detection = true;
    config->enable_narrative_coherence = true;
    config->metacognition_sensitivity = 0.7f;
    config->temporal_continuity_weight = 0.8f;
    config->enable_agency_attribution = true;
    config->fe_sensitivity = 1.0f;
    config->awareness_sensitivity = 1.0f;
    return 0;
}

self_awareness_extended_fep_bridge_t* self_awareness_extended_fep_bridge_create(const self_awareness_extended_fep_config_t* config) {
    self_awareness_extended_fep_bridge_t* bridge = nimcp_malloc(sizeof(self_awareness_extended_fep_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    memset(bridge, 0, sizeof(self_awareness_extended_fep_bridge_t));
    if (config) bridge->config = *config;
    else self_awareness_extended_fep_bridge_default_config(&bridge->config);
    if (bridge_base_init(&bridge->base, 0, "self_awareness_extended_fep") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) { nimcp_free(bridge); return NULL; }
    return bridge;
}

void self_awareness_extended_fep_bridge_destroy(self_awareness_extended_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.bio_async_enabled) self_awareness_extended_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int self_awareness_extended_fep_bridge_connect_fep(self_awareness_extended_fep_bridge_t* bridge, fep_system_t* fep) {
    NIMCP_CHECK_THROW(bridge && fep, NIMCP_ERROR_NULL_POINTER, "bridge or fep is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_extended_fep_bridge_connect_awareness(self_awareness_extended_fep_bridge_t* bridge, self_awareness_system_t awareness) {
    NIMCP_CHECK_THROW(bridge && awareness, NIMCP_ERROR_NULL_POINTER, "bridge or awareness is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->awareness_system = awareness;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_extended_fep_bridge_disconnect(self_awareness_extended_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    bridge->awareness_system = NULL;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_extended_fep_trigger_monitoring(self_awareness_extended_fep_bridge_t* bridge, float uncertainty) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_metacognitive_monitoring) return 0;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_effects.current_uncertainty = uncertainty;
    if (uncertainty > bridge->config.uncertainty_threshold) {
        bridge->fep_effects.metacognitive_monitoring_triggered = true;
        bridge->state.monitoring_active = true;
        bridge->stats.monitoring_events++;
        NIMCP_LOGGING_INFO("Metacognitive monitoring triggered (uncertainty=%.2f)", uncertainty);
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_extended_fep_check_self_harm(self_awareness_extended_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_self_harm_detection) return 0;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_effects.self_harm_check_active = true;
    bridge->stats.self_harm_detections++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_extended_fep_modulate_depth(self_awareness_extended_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");
    if (!bridge->config.enable_precision_modulation) return 0;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.regulation_actions++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_extended_fep_apply_narrative_coherence(self_awareness_extended_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");
    if (!bridge->config.enable_narrative_coherence) return 0;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->awareness_effects.self_model_updating_beliefs = true;
    bridge->stats.belief_updates++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_extended_fep_update_from_regulation(self_awareness_extended_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->awareness_effects.agency_attribution_active = true;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_extended_fep_bridge_update(self_awareness_extended_fep_bridge_t* bridge, uint64_t delta_ms) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    self_awareness_extended_fep_check_self_harm(bridge);
    self_awareness_extended_fep_modulate_depth(bridge);
    self_awareness_extended_fep_apply_narrative_coherence(bridge);
    self_awareness_extended_fep_update_from_regulation(bridge);
    return 0;
}

int self_awareness_extended_fep_bridge_get_state(const self_awareness_extended_fep_bridge_t* bridge, self_awareness_extended_fep_state_t* state) {
    NIMCP_CHECK_THROW(bridge && state, NIMCP_ERROR_NULL_POINTER, "bridge or state is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_extended_fep_bridge_get_stats(const self_awareness_extended_fep_bridge_t* bridge, self_awareness_extended_fep_stats_t* stats) {
    NIMCP_CHECK_THROW(bridge && stats, NIMCP_ERROR_NULL_POINTER, "bridge or stats is NULL");
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_awareness_extended_fep_bridge_connect_bio_async(self_awareness_extended_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (bridge->base.bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_SELF_AWARENESS_BRIDGE,
        .module_name = "self_awareness_extended_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };
    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    }
    return 0;
}

int self_awareness_extended_fep_bridge_disconnect_bio_async(self_awareness_extended_fep_bridge_t* bridge) {
    if (!bridge || !bridge->base.bio_async_enabled) return 0;
    if (bridge->base.bio_ctx) bio_router_unregister_module(bridge->base.bio_ctx);
    bridge->base.bio_ctx = NULL;
    bridge->base.bio_async_enabled = false;
    return 0;
}

bool self_awareness_extended_fep_bridge_is_bio_async_connected(const self_awareness_extended_fep_bridge_t* bridge) {
    return bridge ? bridge->base.bio_async_enabled : false;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int self_awareness_extended_fep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Self_Awareness_Extended_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Self_Awareness_Extended_FEP_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Self_Awareness_Extended_FEP_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
