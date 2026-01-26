/**
 * @file nimcp_logic_thalamic_bridge.c
 * @brief Logic-Thalamic Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/logic/nimcp_logic_thalamic_bridge.h"
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

/** Global health agent for logic_thalamic_bridge module */
static nimcp_health_agent_t* g_logic_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for logic_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void logic_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_logic_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from logic_thalamic_bridge module */
static inline void logic_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_logic_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_logic_thalamic_bridge_health_agent, operation, progress);
    }
}


struct logic_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* logic;
    thalamic_router_t* router;
    logic_thalamic_config_t config;
    logic_thalamic_stats_t stats;
    float attention_weight;
};

logic_thalamic_config_t logic_thalamic_default_config(void) {
    logic_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_contradiction_alert = true,
        .min_logical_strength = 0.3f,
        .max_inference_depth = 10
    };
    return cfg;
}

logic_thalamic_bridge_t* logic_thalamic_bridge_create(void* logic, thalamic_router_t* router, const logic_thalamic_config_t* config) {
    logic_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(logic_thalamic_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "logic_thalamic_bridge_create: failed to allocate bridge");
        return NULL;
    }
    bridge->logic = logic;
    bridge->router = router;
    bridge->config = config ? *config : logic_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void logic_thalamic_bridge_destroy(logic_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int logic_thalamic_bridge_reset(logic_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int logic_thalamic_route_inference(logic_thalamic_bridge_t* bridge, const logic_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->logical_strength < bridge->config.min_logical_strength) {
        return 0;
    }
    if (signal->inference_depth > bridge->config.max_inference_depth) {
        return 0;
    }
    bridge->stats.inferences_routed++;
    bridge->stats.avg_inference_depth = (bridge->stats.avg_inference_depth * (bridge->stats.inferences_routed - 1) +
                                         signal->inference_depth) / bridge->stats.inferences_routed;
    if (bridge->config.enable_contradiction_alert && signal->signal_type == LOGIC_SIGNAL_CONTRADICTION) {
        bridge->stats.contradictions_flagged++;
    }
    return 0;
}

int logic_thalamic_route_conclusion(logic_thalamic_bridge_t* bridge, const void* conclusion, float confidence) {
    if (!bridge) return -1;
    bridge->stats.conclusions_routed++;
    return 0;
}

int logic_thalamic_set_attention(logic_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int logic_thalamic_get_attention(const logic_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int logic_thalamic_bridge_get_stats(const logic_thalamic_bridge_t* bridge, logic_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int logic_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Logic_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Logic_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Logic_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
