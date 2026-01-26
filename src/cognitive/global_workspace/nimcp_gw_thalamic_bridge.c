/**
 * @file nimcp_gw_thalamic_bridge.c
 * @brief Global Workspace-Thalamic Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/global_workspace/nimcp_gw_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
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

/** Global health agent for gw_thalamic_bridge module */
static nimcp_health_agent_t* g_gw_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for gw_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void gw_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_gw_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from gw_thalamic_bridge module */
static inline void gw_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_gw_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_gw_thalamic_bridge_health_agent, operation, progress);
    }
}


struct gw_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* gw;
    thalamic_router_t* router;
    gw_thalamic_config_t config;
    gw_thalamic_stats_t stats;
    float attention_weight;
    uint64_t update_count;
};

gw_thalamic_config_t gw_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    gw_thalamic_bridge_heartbeat("gw_thalamic__gw_thalamic_default_", 0.0f);


    gw_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_competition_routing = true,
        .min_salience_threshold = 0.3f,
        .ignition_threshold = 0.6f
    };
    return cfg;
}

gw_thalamic_bridge_t* gw_thalamic_bridge_create(void* gw, thalamic_router_t* router, const gw_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    gw_thalamic_bridge_heartbeat("gw_thalamic__create", 0.0f);


    gw_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(gw_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    bridge->gw = gw;
    bridge->router = router;
    bridge->config = config ? *config : gw_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void gw_thalamic_bridge_destroy(gw_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    gw_thalamic_bridge_heartbeat("gw_thalamic__destroy", 0.0f);


    if (bridge) nimcp_free(bridge);
}

int gw_thalamic_bridge_reset(gw_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    gw_thalamic_bridge_heartbeat("gw_thalamic__reset", 0.0f);


    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int gw_thalamic_route_broadcast(gw_thalamic_bridge_t* bridge, const gw_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* Phase 8: Heartbeat at operation start */
    gw_thalamic_bridge_heartbeat("gw_thalamic__gw_thalamic_route_br", 0.0f);


    if (bridge->config.enable_attention_gating && signal->salience < bridge->config.min_salience_threshold) {
        bridge->stats.signals_gated++;
        return 0;
    }
    bridge->stats.broadcasts_routed++;
    bridge->stats.avg_attention = (bridge->stats.avg_attention * (bridge->stats.broadcasts_routed - 1) +
                                   signal->attention_weight) / bridge->stats.broadcasts_routed;
    return 0;
}

int gw_thalamic_route_ignition(gw_thalamic_bridge_t* bridge, const void* content, float strength) {
    if (!bridge) return -1;
    if (strength < bridge->config.ignition_threshold) return 0;
    /* Phase 8: Heartbeat at operation start */
    gw_thalamic_bridge_heartbeat("gw_thalamic__gw_thalamic_route_ig", 0.0f);


    bridge->stats.ignitions_triggered++;
    return 0;
}

int gw_thalamic_set_attention(gw_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    gw_thalamic_bridge_heartbeat("gw_thalamic__gw_thalamic_set_atte", 0.0f);


    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int gw_thalamic_get_attention(const gw_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    /* Phase 8: Heartbeat at operation start */
    gw_thalamic_bridge_heartbeat("gw_thalamic__gw_thalamic_get_atte", 0.0f);


    return 0;
}

int gw_thalamic_bridge_get_stats(const gw_thalamic_bridge_t* bridge, gw_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    /* Phase 8: Heartbeat at operation start */
    gw_thalamic_bridge_heartbeat("gw_thalamic__get_stats", 0.0f);


    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int gw_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    gw_thalamic_bridge_heartbeat("gw_thalamic__query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Global_Workspace_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                gw_thalamic_bridge_heartbeat("gw_thalamic__loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Global_Workspace_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Global_Workspace_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
