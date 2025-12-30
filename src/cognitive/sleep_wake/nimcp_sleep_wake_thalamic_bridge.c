/**
 * @file nimcp_sleep_wake_thalamic_bridge.c
 * @brief Sleep-Wake-Thalamic Bridge Implementation
 */

#include "cognitive/sleep_wake/nimcp_sleep_wake_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct sleep_wake_thalamic_bridge {
    void* sleep_wake;
    thalamic_router_t* router;
    sleep_wake_thalamic_config_t config;
    sleep_wake_thalamic_stats_t stats;
    float attention_weight;
};

sleep_wake_thalamic_config_t sleep_wake_thalamic_default_config(void) {
    sleep_wake_thalamic_config_t cfg = {
        .enable_arousal_modulation = true,
        .enable_transition_gating = true,
        .min_arousal_threshold = 0.3f,
        .transition_threshold = 0.5f
    };
    return cfg;
}

sleep_wake_thalamic_bridge_t* sleep_wake_thalamic_bridge_create(void* sleep_wake, thalamic_router_t* router, const sleep_wake_thalamic_config_t* config) {
    sleep_wake_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(sleep_wake_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->sleep_wake = sleep_wake;
    bridge->router = router;
    bridge->config = config ? *config : sleep_wake_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void sleep_wake_thalamic_bridge_destroy(sleep_wake_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int sleep_wake_thalamic_bridge_reset(sleep_wake_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int sleep_wake_thalamic_route_arousal(sleep_wake_thalamic_bridge_t* bridge, const sleep_wake_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    bridge->stats.arousal_updates++;
    bridge->stats.avg_arousal_level = (bridge->stats.avg_arousal_level * (bridge->stats.arousal_updates - 1) +
                                       signal->arousal_level) / bridge->stats.arousal_updates;
    if (signal->signal_type == SLEEP_WAKE_SIGNAL_TRANSITION) {
        bridge->stats.state_transitions++;
    }
    if (signal->signal_type == SLEEP_WAKE_SIGNAL_CIRCADIAN) {
        bridge->stats.circadian_updates++;
    }
    return 0;
}

int sleep_wake_thalamic_modulate_gating(sleep_wake_thalamic_bridge_t* bridge, float arousal_level) {
    if (!bridge) return -1;
    /* Modulate attention based on arousal level */
    if (bridge->config.enable_arousal_modulation) {
        bridge->attention_weight = arousal_level < 0.0f ? 0.0f : (arousal_level > 1.0f ? 1.0f : arousal_level);
    }
    return 0;
}

int sleep_wake_thalamic_set_attention(sleep_wake_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int sleep_wake_thalamic_get_attention(const sleep_wake_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int sleep_wake_thalamic_bridge_get_stats(const sleep_wake_thalamic_bridge_t* bridge, sleep_wake_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
