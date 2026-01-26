/**
 * @file nimcp_free_energy_thalamic_bridge.c
 * @brief Free Energy Principle-Thalamic Bridge Implementation
 */

#include "cognitive/free_energy/nimcp_free_energy_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for free_energy_thalamic_bridge module */
static nimcp_health_agent_t* g_free_energy_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for free_energy_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void free_energy_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_free_energy_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from free_energy_thalamic_bridge module */
static inline void free_energy_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_free_energy_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_free_energy_thalamic_bridge_health_agent, operation, progress);
    }
}


struct free_energy_thalamic_bridge {
    bridge_base_t base;
    void* free_energy;
    thalamic_router_t* router;
    free_energy_thalamic_config_t config;
    free_energy_thalamic_stats_t stats;
    float attention_weight;
};

free_energy_thalamic_config_t free_energy_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    free_energy_thalamic_bridge_heartbeat("free_energy__free_energy_thalamic", 0.0f);


    free_energy_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_precision_boost = true,
        .min_error_threshold = 0.2f,
        .precision_boost = 0.4f
    };
    return cfg;
}

free_energy_thalamic_bridge_t* free_energy_thalamic_bridge_create(void* free_energy, thalamic_router_t* router, const free_energy_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    free_energy_thalamic_bridge_heartbeat("free_energy__create", 0.0f);


    free_energy_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(free_energy_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    if (bridge_base_init(&bridge->base, 0, "free_energy_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->free_energy = free_energy;
    bridge->router = router;
    bridge->config = config ? *config : free_energy_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void free_energy_thalamic_bridge_destroy(free_energy_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    /* Phase 8: Heartbeat at operation start */
    free_energy_thalamic_bridge_heartbeat("free_energy__destroy", 0.0f);


    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int free_energy_thalamic_bridge_reset(free_energy_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    free_energy_thalamic_bridge_heartbeat("free_energy__reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int free_energy_thalamic_route_prediction_error(free_energy_thalamic_bridge_t* bridge, const free_energy_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* Phase 8: Heartbeat at operation start */
    free_energy_thalamic_bridge_heartbeat("free_energy__free_energy_thalamic", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    /* High-precision errors bypass gating */
    if (bridge->config.enable_attention_gating &&
        signal->prediction_error < bridge->config.min_error_threshold &&
        signal->precision < 0.7f) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.errors_routed++;
    bridge->stats.avg_precision = (bridge->stats.avg_precision * (bridge->stats.errors_routed - 1) +
                                   signal->precision) / bridge->stats.errors_routed;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int free_energy_thalamic_route_model_update(free_energy_thalamic_bridge_t* bridge, const void* model, float importance) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    free_energy_thalamic_bridge_heartbeat("free_energy__free_energy_thalamic", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.model_updates++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int free_energy_thalamic_set_attention(free_energy_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    free_energy_thalamic_bridge_heartbeat("free_energy__free_energy_thalamic", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int free_energy_thalamic_get_attention(const free_energy_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    /* Phase 8: Heartbeat at operation start */
    free_energy_thalamic_bridge_heartbeat("free_energy__free_energy_thalamic", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int free_energy_thalamic_bridge_get_stats(const free_energy_thalamic_bridge_t* bridge, free_energy_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    /* Phase 8: Heartbeat at operation start */
    free_energy_thalamic_bridge_heartbeat("free_energy__get_stats", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Free Energy Thalamic Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int free_energy_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    free_energy_thalamic_bridge_heartbeat("free_energy__query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Free_Energy_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                free_energy_thalamic_bridge_heartbeat("free_energy__loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("FE Thalamic Bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Free_Energy_Thalamic_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Free_Energy_Thalamic_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
