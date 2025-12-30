/**
 * @file nimcp_working_memory_thalamic_bridge.c
 * @brief Working Memory-Thalamic Bridge Implementation
 */

#include "cognitive/working_memory/nimcp_working_memory_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <string.h>

struct working_memory_thalamic_bridge {
    void* working_memory;
    thalamic_router_t* router;
    working_memory_thalamic_config_t config;
    working_memory_thalamic_stats_t stats;
    float attention_weight;
};

working_memory_thalamic_config_t working_memory_thalamic_default_config(void) {
    return (working_memory_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_priority_routing = true,
        .enable_capacity_check = true,
        .min_urgency_threshold = 0.2f,
        .priority_boost = 1.3f
    };
}

working_memory_thalamic_bridge_t* working_memory_thalamic_bridge_create(
    void* working_memory,
    thalamic_router_t* router,
    const working_memory_thalamic_config_t* config
) {
    working_memory_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(working_memory_thalamic_bridge_t));
    if (!bridge) return NULL;

    bridge->working_memory = working_memory;
    bridge->router = router;
    bridge->config = config ? *config : working_memory_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void working_memory_thalamic_bridge_destroy(working_memory_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int working_memory_thalamic_bridge_reset(working_memory_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int working_memory_thalamic_route_signal(
    working_memory_thalamic_bridge_t* bridge,
    const working_memory_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;

    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->wm_urgency * bridge->attention_weight;

        /* High priority items get boost */
        if (bridge->config.enable_priority_routing && signal->item_priority > 0.7f) {
            effective_urgency *= bridge->config.priority_boost;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            return 0;
        }
    }

    switch (signal->signal_type) {
        case WM_SIGNAL_ENCODE:
            bridge->stats.encodings++;
            break;
        case WM_SIGNAL_MAINTAIN:
            bridge->stats.maintenances++;
            break;
        case WM_SIGNAL_UPDATE:
            bridge->stats.updates++;
            break;
        case WM_SIGNAL_RETRIEVE:
            bridge->stats.retrievals++;
            break;
        case WM_SIGNAL_CLEAR:
            bridge->stats.clears++;
            break;
        default:
            return -1;
    }

    uint64_t total = bridge->stats.encodings + bridge->stats.maintenances +
                     bridge->stats.updates + bridge->stats.retrievals +
                     bridge->stats.clears;
    if (total > 0) {
        bridge->stats.avg_capacity_used =
            (bridge->stats.avg_capacity_used * (total - 1) + signal->capacity_used) / total;
        bridge->stats.avg_item_priority =
            (bridge->stats.avg_item_priority * (total - 1) + signal->item_priority) / total;
    }

    return 0;
}

int working_memory_thalamic_route_encode(
    working_memory_thalamic_bridge_t* bridge,
    float priority,
    float urgency
) {
    if (!bridge) return -1;

    working_memory_thalamic_signal_t signal = {
        .signal_type = WM_SIGNAL_ENCODE,
        .wm_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .capacity_used = 0.5f,
        .item_priority = priority < 0.0f ? 0.0f : (priority > 1.0f ? 1.0f : priority),
        .decay_rate = 0.1f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return working_memory_thalamic_route_signal(bridge, &signal);
}

int working_memory_thalamic_route_update(
    working_memory_thalamic_bridge_t* bridge,
    float priority,
    float urgency
) {
    if (!bridge) return -1;

    working_memory_thalamic_signal_t signal = {
        .signal_type = WM_SIGNAL_UPDATE,
        .wm_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .capacity_used = 0.5f,
        .item_priority = priority < 0.0f ? 0.0f : (priority > 1.0f ? 1.0f : priority),
        .decay_rate = 0.1f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return working_memory_thalamic_route_signal(bridge, &signal);
}

int working_memory_thalamic_set_attention(working_memory_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int working_memory_thalamic_get_attention(const working_memory_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int working_memory_thalamic_bridge_get_stats(
    const working_memory_thalamic_bridge_t* bridge,
    working_memory_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
