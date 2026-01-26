/**
 * @file nimcp_llf_thalamic_bridge.c
 * @brief Love/Loyalty/Friendship-Thalamic Bridge Implementation
 */

#include "cognitive/love_loyalty_friendship/nimcp_llf_thalamic_bridge.h"
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

/** Global health agent for llf_thalamic_bridge module */
static nimcp_health_agent_t* g_llf_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for llf_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void llf_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_llf_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from llf_thalamic_bridge module */
static inline void llf_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_llf_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_llf_thalamic_bridge_health_agent, operation, progress);
    }
}


struct llf_thalamic_bridge {
    bridge_base_t base;
    void* llf;
    thalamic_router_t* router;
    llf_thalamic_config_t config;
    llf_thalamic_stats_t stats;
    float attention_weight;
};

llf_thalamic_config_t llf_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    llf_thalamic_bridge_heartbeat("llf_thalamic_llf_thalamic_default", 0.0f);


    llf_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_bond_strengthening = true,
        .min_bond_threshold = 0.3f,
        .trust_threshold = 0.5f
    };
    return cfg;
}

llf_thalamic_bridge_t* llf_thalamic_bridge_create(void* llf, thalamic_router_t* router, const llf_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    llf_thalamic_bridge_heartbeat("llf_thalamic_create", 0.0f);


    llf_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(llf_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    if (bridge_base_init(&bridge->base, 0, "llf_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->llf = llf;
    bridge->router = router;
    bridge->config = config ? *config : llf_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void llf_thalamic_bridge_destroy(llf_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    /* Phase 8: Heartbeat at operation start */
    llf_thalamic_bridge_heartbeat("llf_thalamic_destroy", 0.0f);


    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int llf_thalamic_bridge_reset(llf_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    llf_thalamic_bridge_heartbeat("llf_thalamic_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int llf_thalamic_route_attachment(llf_thalamic_bridge_t* bridge, const llf_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* Phase 8: Heartbeat at operation start */
    llf_thalamic_bridge_heartbeat("llf_thalamic_llf_thalamic_route_a", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->bond_strength < bridge->config.min_bond_threshold) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.attachments_routed++;
    bridge->stats.avg_bond_strength = (bridge->stats.avg_bond_strength * (bridge->stats.attachments_routed - 1) +
                                       signal->bond_strength) / bridge->stats.attachments_routed;
    if (signal->signal_type == LLF_SIGNAL_TRUST) {
        bridge->stats.trust_updates++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int llf_thalamic_route_care(llf_thalamic_bridge_t* bridge, const void* target, float motivation) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    llf_thalamic_bridge_heartbeat("llf_thalamic_llf_thalamic_route_c", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.care_expressions++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int llf_thalamic_set_attention(llf_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    llf_thalamic_bridge_heartbeat("llf_thalamic_llf_thalamic_set_att", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int llf_thalamic_get_attention(const llf_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    /* Phase 8: Heartbeat at operation start */
    llf_thalamic_bridge_heartbeat("llf_thalamic_llf_thalamic_get_att", 0.0f);


    return 0;
}

int llf_thalamic_bridge_get_stats(const llf_thalamic_bridge_t* bridge, llf_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    /* Phase 8: Heartbeat at operation start */
    llf_thalamic_bridge_heartbeat("llf_thalamic_get_stats", 0.0f);


    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int llf_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    llf_thalamic_bridge_heartbeat("llf_thalamic_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "LLF_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                llf_thalamic_bridge_heartbeat("llf_thalamic_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "LLF_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "LLF_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
