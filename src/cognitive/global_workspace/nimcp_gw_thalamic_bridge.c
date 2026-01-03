/**
 * @file nimcp_gw_thalamic_bridge.c
 * @brief Global Workspace-Thalamic Bridge Implementation
 */

#include "cognitive/global_workspace/nimcp_gw_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct gw_thalamic_bridge {
    void* gw;
    thalamic_router_t* router;
    gw_thalamic_config_t config;
    gw_thalamic_stats_t stats;
    float attention_weight;
    uint64_t update_count;
};

gw_thalamic_config_t gw_thalamic_default_config(void) {
    gw_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_competition_routing = true,
        .min_salience_threshold = 0.3f,
        .ignition_threshold = 0.6f
    };
    return cfg;
}

gw_thalamic_bridge_t* gw_thalamic_bridge_create(void* gw, thalamic_router_t* router, const gw_thalamic_config_t* config) {
    gw_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(gw_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->gw = gw;
    bridge->router = router;
    bridge->config = config ? *config : gw_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void gw_thalamic_bridge_destroy(gw_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int gw_thalamic_bridge_reset(gw_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int gw_thalamic_route_broadcast(gw_thalamic_bridge_t* bridge, const gw_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
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
    bridge->stats.ignitions_triggered++;
    return 0;
}

int gw_thalamic_set_attention(gw_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int gw_thalamic_get_attention(const gw_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int gw_thalamic_bridge_get_stats(const gw_thalamic_bridge_t* bridge, gw_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int gw_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Global_Workspace_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
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
