/**
 * @file nimcp_consolidation_thalamic_bridge.c
 * @brief Memory Consolidation-Thalamic Bridge Implementation
 */

#include "cognitive/consolidation/nimcp_consolidation_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct consolidation_thalamic_bridge {
    bridge_base_t base;  /* MUST be first - provides mutex protection */
    void* consolidation;
    thalamic_router_t* router;
    consolidation_thalamic_config_t config;
    consolidation_thalamic_stats_t stats;
    float attention_weight;
};

consolidation_thalamic_config_t consolidation_thalamic_default_config(void) {
    consolidation_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_emotional_boost = true,
        .min_memory_salience = 0.3f,
        .replay_boost = 0.3f
    };
    return cfg;
}

consolidation_thalamic_bridge_t* consolidation_thalamic_bridge_create(void* consolidation, thalamic_router_t* router, const consolidation_thalamic_config_t* config) {
    consolidation_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(consolidation_thalamic_bridge_t));
    if (!bridge) return NULL;

    /* Initialize mutex for thread safety */
    bridge->base.mutex = nimcp_mutex_create(NULL);
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
    if (bridge) {
        if (bridge->base.mutex) {
            nimcp_mutex_destroy(bridge->base.mutex);
        }
        nimcp_free(bridge);
    }
}

int consolidation_thalamic_bridge_reset(consolidation_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_route_encode(consolidation_thalamic_bridge_t* bridge, const consolidation_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
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
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.replays_triggered++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_set_attention(consolidation_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_get_attention(const consolidation_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_bridge_get_stats(const consolidation_thalamic_bridge_t* bridge, consolidation_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}
