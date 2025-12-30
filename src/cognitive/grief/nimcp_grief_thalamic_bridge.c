/**
 * @file nimcp_grief_thalamic_bridge.c
 * @brief Grief-Thalamic Bridge Implementation
 */

#include "cognitive/grief/nimcp_grief_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct grief_thalamic_bridge {
    void* grief;
    thalamic_router_t* router;
    grief_thalamic_config_t config;
    grief_thalamic_stats_t stats;
    float attention_weight;
};

grief_thalamic_config_t grief_thalamic_default_config(void) {
    grief_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_gradual_processing = true,
        .min_intensity_threshold = 0.3f,
        .adaptation_threshold = 0.5f
    };
    return cfg;
}

grief_thalamic_bridge_t* grief_thalamic_bridge_create(void* grief, thalamic_router_t* router, const grief_thalamic_config_t* config) {
    grief_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(grief_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->grief = grief;
    bridge->router = router;
    bridge->config = config ? *config : grief_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void grief_thalamic_bridge_destroy(grief_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int grief_thalamic_bridge_reset(grief_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int grief_thalamic_route_loss(grief_thalamic_bridge_t* bridge, const grief_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->grief_intensity < bridge->config.min_intensity_threshold) {
        return 0;
    }
    bridge->stats.losses_processed++;
    bridge->stats.avg_grief_intensity = (bridge->stats.avg_grief_intensity * (bridge->stats.losses_processed - 1) +
                                         signal->grief_intensity) / bridge->stats.losses_processed;
    if (signal->signal_type == GRIEF_SIGNAL_ADAPTATION && signal->adaptation_progress >= bridge->config.adaptation_threshold) {
        bridge->stats.adaptations_achieved++;
    }
    return 0;
}

int grief_thalamic_route_processing(grief_thalamic_bridge_t* bridge, const void* stage, float progress) {
    if (!bridge) return -1;
    bridge->stats.processing_stages++;
    return 0;
}

int grief_thalamic_set_attention(grief_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int grief_thalamic_get_attention(const grief_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int grief_thalamic_bridge_get_stats(const grief_thalamic_bridge_t* bridge, grief_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
