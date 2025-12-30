/**
 * @file nimcp_mental_health_thalamic_bridge.c
 * @brief Mental Health-Thalamic Bridge Implementation
 */

#include "cognitive/mental_health/nimcp_mental_health_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct mental_health_thalamic_bridge {
    void* mental_health;
    thalamic_router_t* router;
    mental_health_thalamic_config_t config;
    mental_health_thalamic_stats_t stats;
    float attention_weight;
};

mental_health_thalamic_config_t mental_health_thalamic_default_config(void) {
    mental_health_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_warning_priority = true,
        .min_wellbeing_threshold = 0.3f,
        .stress_alert_threshold = 0.7f
    };
    return cfg;
}

mental_health_thalamic_bridge_t* mental_health_thalamic_bridge_create(void* mental_health, thalamic_router_t* router, const mental_health_thalamic_config_t* config) {
    mental_health_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(mental_health_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->mental_health = mental_health;
    bridge->router = router;
    bridge->config = config ? *config : mental_health_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void mental_health_thalamic_bridge_destroy(mental_health_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int mental_health_thalamic_bridge_reset(mental_health_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int mental_health_thalamic_route_wellbeing(mental_health_thalamic_bridge_t* bridge, const mental_health_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    bridge->stats.wellbeing_updates++;
    bridge->stats.avg_wellbeing_level = (bridge->stats.avg_wellbeing_level * (bridge->stats.wellbeing_updates - 1) +
                                         signal->wellbeing_level) / bridge->stats.wellbeing_updates;
    if (signal->stress_level >= bridge->config.stress_alert_threshold) {
        bridge->stats.stress_alerts++;
    }
    if (signal->signal_type == MENTAL_HEALTH_SIGNAL_WARNING) {
        bridge->stats.warnings_issued++;
    }
    return 0;
}

int mental_health_thalamic_route_warning(mental_health_thalamic_bridge_t* bridge, const void* concern, float severity) {
    if (!bridge) return -1;
    if (bridge->config.enable_warning_priority) {
        bridge->stats.warnings_issued++;
    }
    return 0;
}

int mental_health_thalamic_set_attention(mental_health_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int mental_health_thalamic_get_attention(const mental_health_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int mental_health_thalamic_bridge_get_stats(const mental_health_thalamic_bridge_t* bridge, mental_health_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
