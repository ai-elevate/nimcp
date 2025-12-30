/**
 * @file nimcp_meta_learning_thalamic_bridge.c
 * @brief Meta-Learning-Thalamic Bridge Implementation
 */

#include "cognitive/meta_learning/nimcp_meta_learning_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct meta_learning_thalamic_bridge {
    void* meta_learning;
    thalamic_router_t* router;
    meta_learning_thalamic_config_t config;
    meta_learning_thalamic_stats_t stats;
    float attention_weight;
};

meta_learning_thalamic_config_t meta_learning_thalamic_default_config(void) {
    meta_learning_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_strategy_broadcast = true,
        .min_strategy_confidence = 0.3f,
        .transfer_threshold = 0.5f
    };
    return cfg;
}

meta_learning_thalamic_bridge_t* meta_learning_thalamic_bridge_create(void* meta_learning, thalamic_router_t* router, const meta_learning_thalamic_config_t* config) {
    meta_learning_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(meta_learning_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->meta_learning = meta_learning;
    bridge->router = router;
    bridge->config = config ? *config : meta_learning_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void meta_learning_thalamic_bridge_destroy(meta_learning_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int meta_learning_thalamic_bridge_reset(meta_learning_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int meta_learning_thalamic_route_strategy(meta_learning_thalamic_bridge_t* bridge, const meta_learning_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->strategy_confidence < bridge->config.min_strategy_confidence) {
        return 0;
    }
    bridge->stats.strategies_routed++;
    bridge->stats.avg_learning_rate = (bridge->stats.avg_learning_rate * (bridge->stats.strategies_routed - 1) +
                                       signal->learning_rate) / bridge->stats.strategies_routed;
    if (signal->signal_type == META_LEARNING_SIGNAL_RATE) {
        bridge->stats.rate_adjustments++;
    }
    return 0;
}

int meta_learning_thalamic_route_transfer(meta_learning_thalamic_bridge_t* bridge, const void* knowledge, float potential) {
    if (!bridge) return -1;
    if (potential >= bridge->config.transfer_threshold) {
        bridge->stats.transfers_initiated++;
    }
    return 0;
}

int meta_learning_thalamic_set_attention(meta_learning_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int meta_learning_thalamic_get_attention(const meta_learning_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int meta_learning_thalamic_bridge_get_stats(const meta_learning_thalamic_bridge_t* bridge, meta_learning_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
