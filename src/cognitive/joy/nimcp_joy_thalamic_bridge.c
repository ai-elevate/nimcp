/**
 * @file nimcp_joy_thalamic_bridge.c
 * @brief Joy-Thalamic Bridge Implementation
 */

#include "cognitive/joy/nimcp_joy_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct joy_thalamic_bridge {
    void* joy;
    thalamic_router_t* router;
    joy_thalamic_config_t config;
    joy_thalamic_stats_t stats;
    float attention_weight;
};

joy_thalamic_config_t joy_thalamic_default_config(void) {
    joy_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_savoring_boost = true,
        .min_hedonic_threshold = 0.3f,
        .anticipation_threshold = 0.5f
    };
    return cfg;
}

joy_thalamic_bridge_t* joy_thalamic_bridge_create(void* joy, thalamic_router_t* router, const joy_thalamic_config_t* config) {
    joy_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(joy_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->joy = joy;
    bridge->router = router;
    bridge->config = config ? *config : joy_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void joy_thalamic_bridge_destroy(joy_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int joy_thalamic_bridge_reset(joy_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int joy_thalamic_route_pleasure(joy_thalamic_bridge_t* bridge, const joy_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->hedonic_value < bridge->config.min_hedonic_threshold) {
        return 0;
    }
    bridge->stats.pleasures_routed++;
    bridge->stats.avg_joy_intensity = (bridge->stats.avg_joy_intensity * (bridge->stats.pleasures_routed - 1) +
                                       signal->joy_intensity) / bridge->stats.pleasures_routed;
    if (signal->signal_type == JOY_SIGNAL_ANTICIPATION) {
        bridge->stats.anticipations_triggered++;
    }
    return 0;
}

int joy_thalamic_route_savoring(joy_thalamic_bridge_t* bridge, const void* experience, float duration) {
    if (!bridge) return -1;
    if (bridge->config.enable_savoring_boost) {
        bridge->stats.savoring_episodes++;
    }
    return 0;
}

int joy_thalamic_set_attention(joy_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int joy_thalamic_get_attention(const joy_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int joy_thalamic_bridge_get_stats(const joy_thalamic_bridge_t* bridge, joy_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
