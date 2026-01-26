/**
 * @file nimcp_consolidation_thalamic_bridge.c
 * @brief Memory Consolidation-Thalamic Bridge Implementation
 */

#include "cognitive/consolidation/nimcp_consolidation_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
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

/** Global health agent for consolidation_thalamic_bridge module */
static nimcp_health_agent_t* g_consolidation_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for consolidation_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void consolidation_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_consolidation_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from consolidation_thalamic_bridge module */
static inline void consolidation_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_consolidation_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_consolidation_thalamic_bridge_health_agent, operation, progress);
    }
}


struct consolidation_thalamic_bridge {
    bridge_base_t base;  /* MUST be first - provides mutex protection */
    void* consolidation;
    thalamic_router_t* router;
    consolidation_thalamic_config_t config;
    consolidation_thalamic_stats_t stats;
    float attention_weight;
};

consolidation_thalamic_config_t consolidation_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_consolidation_thalam", 0.0f);


    consolidation_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_emotional_boost = true,
        .min_memory_salience = 0.3f,
        .replay_boost = 0.3f
    };
    return cfg;
}

consolidation_thalamic_bridge_t* consolidation_thalamic_bridge_create(void* consolidation, thalamic_router_t* router, const consolidation_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_create", 0.0f);


    consolidation_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(consolidation_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    /* Initialize mutex for thread safety */
    if (bridge_base_init(&bridge->base, 0, "consolidation_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->consolidation = consolidation;
    bridge->router = router;
    bridge->config = config ? *config : consolidation_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void consolidation_thalamic_bridge_destroy(consolidation_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_destroy", 0.0f);


    if (bridge) {
        if (bridge->base.mutex) {
            bridge_base_cleanup(&bridge->base);
        }
        nimcp_free(bridge);
    }
}

int consolidation_thalamic_bridge_reset(consolidation_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_route_encode(consolidation_thalamic_bridge_t* bridge, const consolidation_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_consolidation_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->memory_salience < bridge->config.min_memory_salience) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.encodes_routed++;
    bridge->stats.avg_memory_salience = (bridge->stats.avg_memory_salience * (bridge->stats.encodes_routed - 1) +
                                         signal->memory_salience) / bridge->stats.encodes_routed;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_route_replay(consolidation_thalamic_bridge_t* bridge, const void* memory, float importance) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_consolidation_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.replays_triggered++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_set_attention(consolidation_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_consolidation_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_get_attention(const consolidation_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_consolidation_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_bridge_get_stats(const consolidation_thalamic_bridge_t* bridge, consolidation_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_get_stats", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int consolidation_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Consolidation_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                consolidation_thalamic_bridge_heartbeat("consolidatio_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Consolidation_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Consolidation_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
