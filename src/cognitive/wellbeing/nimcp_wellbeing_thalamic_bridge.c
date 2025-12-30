/**
 * @file nimcp_wellbeing_thalamic_bridge.c
 * @brief Wellbeing-Thalamic Bridge Implementation
 */

#include "cognitive/wellbeing/nimcp_wellbeing_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <string.h>

struct wellbeing_thalamic_bridge {
    bridge_base_t base;
    void* wellbeing;
    thalamic_router_t* router;
    wellbeing_thalamic_config_t config;
    wellbeing_thalamic_stats_t stats;
    float attention_weight;
};

wellbeing_thalamic_config_t wellbeing_thalamic_default_config(void) {
    return (wellbeing_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_threat_priority = true,
        .min_urgency_threshold = 0.15f,
        .threat_boost = 1.5f
    };
}

wellbeing_thalamic_bridge_t* wellbeing_thalamic_bridge_create(
    void* wellbeing,
    thalamic_router_t* router,
    const wellbeing_thalamic_config_t* config
) {
    wellbeing_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(wellbeing_thalamic_bridge_t));
    if (!bridge) return NULL;

    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->wellbeing = wellbeing;
    bridge->router = router;
    bridge->config = config ? *config : wellbeing_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void wellbeing_thalamic_bridge_destroy(wellbeing_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) {
        nimcp_mutex_destroy(bridge->base.mutex);
    }
    nimcp_free(bridge);
}

int wellbeing_thalamic_bridge_reset(wellbeing_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int wellbeing_thalamic_route_signal(
    wellbeing_thalamic_bridge_t* bridge,
    const wellbeing_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->wellbeing_urgency * bridge->attention_weight;

        /* Threats get priority boost */
        if (bridge->config.enable_threat_priority &&
            signal->signal_type == WELLBEING_SIGNAL_THREAT) {
            effective_urgency *= bridge->config.threat_boost;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            nimcp_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    switch (signal->signal_type) {
        case WELLBEING_SIGNAL_STATUS:
            bridge->stats.status_updates++;
            break;
        case WELLBEING_SIGNAL_CHANGE:
            bridge->stats.state_changes++;
            break;
        case WELLBEING_SIGNAL_THREAT:
            bridge->stats.threats_signaled++;
            break;
        case WELLBEING_SIGNAL_RECOVERY:
            bridge->stats.recoveries++;
            break;
        default:
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;
    }

    uint64_t total = bridge->stats.status_updates + bridge->stats.state_changes +
                     bridge->stats.threats_signaled + bridge->stats.recoveries;
    if (total > 0) {
        bridge->stats.avg_wellbeing_level =
            (bridge->stats.avg_wellbeing_level * (total - 1) + signal->current_level) / total;
        bridge->stats.avg_stability =
            (bridge->stats.avg_stability * (total - 1) + signal->stability) / total;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int wellbeing_thalamic_route_status(
    wellbeing_thalamic_bridge_t* bridge,
    float level,
    float stability
) {
    if (!bridge) return -1;

    wellbeing_thalamic_signal_t signal = {
        .signal_type = WELLBEING_SIGNAL_STATUS,
        .wellbeing_urgency = 0.3f,
        .current_level = level < 0.0f ? 0.0f : (level > 1.0f ? 1.0f : level),
        .change_rate = 0.0f,
        .stability = stability < 0.0f ? 0.0f : (stability > 1.0f ? 1.0f : stability),
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return wellbeing_thalamic_route_signal(bridge, &signal);
}

int wellbeing_thalamic_route_threat(
    wellbeing_thalamic_bridge_t* bridge,
    float severity,
    float urgency
) {
    if (!bridge) return -1;

    wellbeing_thalamic_signal_t signal = {
        .signal_type = WELLBEING_SIGNAL_THREAT,
        .wellbeing_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .current_level = 1.0f - severity,
        .change_rate = -severity,
        .stability = 0.2f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return wellbeing_thalamic_route_signal(bridge, &signal);
}

int wellbeing_thalamic_set_attention(wellbeing_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int wellbeing_thalamic_get_attention(const wellbeing_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int wellbeing_thalamic_bridge_get_stats(
    const wellbeing_thalamic_bridge_t* bridge,
    wellbeing_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
