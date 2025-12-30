/**
 * @file nimcp_shadow_thalamic_bridge.c
 * @brief Shadow-Thalamic Bridge Implementation
 */

#include "cognitive/shadow/nimcp_shadow_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct shadow_thalamic_bridge {
    void* shadow;
    thalamic_router_t* router;
    shadow_thalamic_config_t config;
    shadow_thalamic_stats_t stats;
    float attention_weight;
};

shadow_thalamic_config_t shadow_thalamic_default_config(void) {
    shadow_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_gradual_emergence = true,
        .min_emergence_threshold = 0.3f,
        .integration_threshold = 0.5f
    };
    return cfg;
}

shadow_thalamic_bridge_t* shadow_thalamic_bridge_create(void* shadow, thalamic_router_t* router, const shadow_thalamic_config_t* config) {
    shadow_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(shadow_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->shadow = shadow;
    bridge->router = router;
    bridge->config = config ? *config : shadow_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void shadow_thalamic_bridge_destroy(shadow_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int shadow_thalamic_bridge_reset(shadow_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int shadow_thalamic_route_emergence(shadow_thalamic_bridge_t* bridge, const shadow_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->emergence_strength < bridge->config.min_emergence_threshold) {
        return 0;
    }
    bridge->stats.emergences_routed++;
    bridge->stats.avg_emergence_strength = (bridge->stats.avg_emergence_strength * (bridge->stats.emergences_routed - 1) +
                                            signal->emergence_strength) / bridge->stats.emergences_routed;
    if (signal->signal_type == SHADOW_SIGNAL_PROJECTION) {
        bridge->stats.projections_detected++;
    }
    return 0;
}

int shadow_thalamic_route_integration(shadow_thalamic_bridge_t* bridge, const void* content, float readiness) {
    if (!bridge) return -1;
    if (readiness >= bridge->config.integration_threshold) {
        bridge->stats.integrations_achieved++;
    }
    return 0;
}

int shadow_thalamic_set_attention(shadow_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int shadow_thalamic_get_attention(const shadow_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int shadow_thalamic_bridge_get_stats(const shadow_thalamic_bridge_t* bridge, shadow_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
