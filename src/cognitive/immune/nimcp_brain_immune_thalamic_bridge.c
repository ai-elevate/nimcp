/**
 * @file nimcp_brain_immune_thalamic_bridge.c
 * @brief Brain Immune-Thalamic Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/immune/nimcp_brain_immune_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "api/nimcp_api_exception.h"
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

/** Global health agent for brain_immune_thalamic_bridge module */
static nimcp_health_agent_t* g_brain_immune_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for brain_immune_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void brain_immune_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_brain_immune_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from brain_immune_thalamic_bridge module */
static inline void brain_immune_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_brain_immune_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_brain_immune_thalamic_bridge_health_agent, operation, progress);
    }
}


struct brain_immune_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* brain_immune;
    thalamic_router_t* router;
    brain_immune_thalamic_config_t config;
    brain_immune_thalamic_stats_t stats;
    float attention_weight;
};

brain_immune_thalamic_config_t brain_immune_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    brain_immune_thalamic_bridge_heartbeat("brain_immune_brain_immune_thalami", 0.0f);


    brain_immune_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_severity_boost = true,
        .min_threat_threshold = 0.2f,
        .inflammation_boost = 0.4f
    };
    return cfg;
}

brain_immune_thalamic_bridge_t* brain_immune_thalamic_bridge_create(void* brain_immune, thalamic_router_t* router, const brain_immune_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    brain_immune_thalamic_bridge_heartbeat("brain_immune_create", 0.0f);


    brain_immune_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(brain_immune_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    bridge->brain_immune = brain_immune;
    bridge->router = router;
    bridge->config = config ? *config : brain_immune_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void brain_immune_thalamic_bridge_destroy(brain_immune_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    brain_immune_thalamic_bridge_heartbeat("brain_immune_destroy", 0.0f);


    if (bridge) nimcp_free(bridge);
}

int brain_immune_thalamic_bridge_reset(brain_immune_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    brain_immune_thalamic_bridge_heartbeat("brain_immune_reset", 0.0f);


    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int brain_immune_thalamic_route_threat(brain_immune_thalamic_bridge_t* bridge, const brain_immune_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* High-severity threats bypass attention gating */
    /* Phase 8: Heartbeat at operation start */
    brain_immune_thalamic_bridge_heartbeat("brain_immune_brain_immune_thalami", 0.0f);


    if (bridge->config.enable_attention_gating &&
        signal->threat_severity < bridge->config.min_threat_threshold &&
        signal->inflammation_level < 0.6f) {
        return 0;
    }
    bridge->stats.threats_routed++;
    bridge->stats.avg_threat_severity = (bridge->stats.avg_threat_severity * (bridge->stats.threats_routed - 1) +
                                         signal->threat_severity) / bridge->stats.threats_routed;
    return 0;
}

int brain_immune_thalamic_route_response(brain_immune_thalamic_bridge_t* bridge, const void* response, float intensity) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    brain_immune_thalamic_bridge_heartbeat("brain_immune_brain_immune_thalami", 0.0f);


    bridge->stats.responses_triggered++;
    return 0;
}

int brain_immune_thalamic_set_attention(brain_immune_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    brain_immune_thalamic_bridge_heartbeat("brain_immune_brain_immune_thalami", 0.0f);


    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int brain_immune_thalamic_get_attention(const brain_immune_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    /* Phase 8: Heartbeat at operation start */
    brain_immune_thalamic_bridge_heartbeat("brain_immune_brain_immune_thalami", 0.0f);


    return 0;
}

int brain_immune_thalamic_bridge_get_stats(const brain_immune_thalamic_bridge_t* bridge, brain_immune_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    /* Phase 8: Heartbeat at operation start */
    brain_immune_thalamic_bridge_heartbeat("brain_immune_get_stats", 0.0f);


    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * @brief Query self-knowledge from knowledge graph
 *
 * WHAT: Query KG for module self-awareness information
 * WHY:  Enable introspective self-knowledge about brain immune thalamic bridge
 * HOW:  Look up entity and relations in KG
 *
 * @param kg Knowledge graph reader handle
 * @return 1 if self-knowledge found, 0 otherwise
 */
int brain_immune_thalamic_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    brain_immune_thalamic_bridge_heartbeat("brain_immune_brain_immune_thalami", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Brain_Immune_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                brain_immune_thalamic_bridge_heartbeat("brain_immune_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Brain immune thalamic bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Brain_Immune_Thalamic_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Brain_Immune_Thalamic_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
