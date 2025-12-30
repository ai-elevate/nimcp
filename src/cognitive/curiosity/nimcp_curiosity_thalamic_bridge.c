/**
 * @file nimcp_curiosity_thalamic_bridge.c
 * @brief Curiosity-Thalamic Bridge Implementation
 */

#include "cognitive/curiosity/nimcp_curiosity_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct curiosity_thalamic_bridge {
    void* curiosity;
    thalamic_router_t* router;
    curiosity_thalamic_config_t config;
    curiosity_thalamic_stats_t stats;
    float attention_weight;
};

curiosity_thalamic_config_t curiosity_thalamic_default_config(void) {
    curiosity_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_novelty_boost = true,
        .min_novelty_threshold = 0.3f,
        .exploration_threshold = 0.5f
    };
    return cfg;
}

curiosity_thalamic_bridge_t* curiosity_thalamic_bridge_create(void* curiosity, thalamic_router_t* router, const curiosity_thalamic_config_t* config) {
    curiosity_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(curiosity_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->curiosity = curiosity;
    bridge->router = router;
    bridge->config = config ? *config : curiosity_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void curiosity_thalamic_bridge_destroy(curiosity_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int curiosity_thalamic_bridge_reset(curiosity_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int curiosity_thalamic_route_novelty(curiosity_thalamic_bridge_t* bridge, const curiosity_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->novelty_value < bridge->config.min_novelty_threshold) {
        return 0;
    }
    bridge->stats.novelties_detected++;
    bridge->stats.avg_novelty_value = (bridge->stats.avg_novelty_value * (bridge->stats.novelties_detected - 1) +
                                       signal->novelty_value) / bridge->stats.novelties_detected;
    if (signal->signal_type == CURIOSITY_SIGNAL_INFORMATION) {
        bridge->stats.information_gains++;
    }
    return 0;
}

int curiosity_thalamic_route_exploration(curiosity_thalamic_bridge_t* bridge, const void* target, float drive) {
    if (!bridge) return -1;
    if (drive >= bridge->config.exploration_threshold) {
        bridge->stats.explorations_initiated++;
    }
    return 0;
}

int curiosity_thalamic_set_attention(curiosity_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int curiosity_thalamic_get_attention(const curiosity_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int curiosity_thalamic_bridge_get_stats(const curiosity_thalamic_bridge_t* bridge, curiosity_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
