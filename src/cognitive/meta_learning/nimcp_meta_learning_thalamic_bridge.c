/**
 * @file nimcp_meta_learning_thalamic_bridge.c
 * @brief Meta-Learning-Thalamic Bridge Implementation
 */

#include "cognitive/meta_learning/nimcp_meta_learning_thalamic_bridge.h"
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

/** Global health agent for meta_learning_thalamic_bridge module */
static nimcp_health_agent_t* g_meta_learning_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for meta_learning_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void meta_learning_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_meta_learning_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from meta_learning_thalamic_bridge module */
static inline void meta_learning_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_meta_learning_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_meta_learning_thalamic_bridge_health_agent, operation, progress);
    }
}


struct meta_learning_thalamic_bridge {
    bridge_base_t base;
    void* meta_learning;
    thalamic_router_t* router;
    meta_learning_thalamic_config_t config;
    meta_learning_thalamic_stats_t stats;
    float attention_weight;
};

meta_learning_thalamic_config_t meta_learning_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_meta_learning_thalam", 0.0f);


    meta_learning_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_strategy_broadcast = true,
        .min_strategy_confidence = 0.3f,
        .transfer_threshold = 0.5f
    };
    return cfg;
}

meta_learning_thalamic_bridge_t* meta_learning_thalamic_bridge_create(void* meta_learning, thalamic_router_t* router, const meta_learning_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_create", 0.0f);


    meta_learning_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(meta_learning_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    if (bridge_base_init(&bridge->base, 0, "meta_learning_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->meta_learning = meta_learning;
    bridge->router = router;
    bridge->config = config ? *config : meta_learning_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void meta_learning_thalamic_bridge_destroy(meta_learning_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_destroy", 0.0f);


    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int meta_learning_thalamic_bridge_reset(meta_learning_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int meta_learning_thalamic_route_strategy(meta_learning_thalamic_bridge_t* bridge, const meta_learning_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_meta_learning_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->strategy_confidence < bridge->config.min_strategy_confidence) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.strategies_routed++;
    bridge->stats.avg_learning_rate = (bridge->stats.avg_learning_rate * (bridge->stats.strategies_routed - 1) +
                                       signal->learning_rate) / bridge->stats.strategies_routed;
    if (signal->signal_type == META_LEARNING_SIGNAL_RATE) {
        bridge->stats.rate_adjustments++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int meta_learning_thalamic_route_transfer(meta_learning_thalamic_bridge_t* bridge, const void* knowledge, float potential) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_meta_learning_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    if (potential >= bridge->config.transfer_threshold) {
        bridge->stats.transfers_initiated++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int meta_learning_thalamic_set_attention(meta_learning_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_meta_learning_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int meta_learning_thalamic_get_attention(const meta_learning_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_meta_learning_thalam", 0.0f);


    return 0;
}

int meta_learning_thalamic_bridge_get_stats(const meta_learning_thalamic_bridge_t* bridge, meta_learning_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_get_stats", 0.0f);


    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int meta_learning_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Meta_Learning_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                meta_learning_thalamic_bridge_heartbeat("meta_learnin_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Meta_Learning_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Meta_Learning_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
