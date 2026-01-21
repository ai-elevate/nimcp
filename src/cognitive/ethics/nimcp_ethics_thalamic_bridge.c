/**
 * @file nimcp_ethics_thalamic_bridge.c
 * @brief Ethics-Thalamic Bridge Implementation
 */

#include "cognitive/ethics/nimcp_ethics_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct ethics_thalamic_bridge {
    void* ethics;
    thalamic_router_t* router;
    ethics_thalamic_config_t config;
    ethics_thalamic_stats_t stats;
    float attention_weight;
};

ethics_thalamic_config_t ethics_thalamic_default_config(void) {
    ethics_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_emotional_boost = true,
        .min_moral_salience = 0.3f,
        .violation_boost = 0.3f
    };
    return cfg;
}

ethics_thalamic_bridge_t* ethics_thalamic_bridge_create(void* ethics, thalamic_router_t* router, const ethics_thalamic_config_t* config) {
    ethics_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(ethics_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->ethics = ethics;
    bridge->router = router;
    bridge->config = config ? *config : ethics_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void ethics_thalamic_bridge_destroy(ethics_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int ethics_thalamic_bridge_reset(ethics_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int ethics_thalamic_route_judgment(ethics_thalamic_bridge_t* bridge, const ethics_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->moral_salience < bridge->config.min_moral_salience) {
        return 0;
    }
    bridge->stats.judgments_routed++;
    bridge->stats.avg_moral_salience = (bridge->stats.avg_moral_salience * (bridge->stats.judgments_routed - 1) +
                                        signal->moral_salience) / bridge->stats.judgments_routed;
    return 0;
}

int ethics_thalamic_route_violation(ethics_thalamic_bridge_t* bridge, const void* violation, float severity) {
    if (!bridge) return -1;
    bridge->stats.violations_flagged++;
    return 0;
}

int ethics_thalamic_set_attention(ethics_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int ethics_thalamic_get_attention(const ethics_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int ethics_thalamic_bridge_get_stats(const ethics_thalamic_bridge_t* bridge, ethics_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Ethics Thalamic Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int ethics_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Ethics_Thalamic_Bridge_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            // No LOG_MODULE defined in this file, use direct printf or skip
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Ethics_Thalamic_Bridge_Module");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Ethics_Thalamic_Bridge_Module");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
