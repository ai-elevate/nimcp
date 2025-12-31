/**
 * @file nimcp_memory_thalamic_bridge.c
 * @brief Memory-Thalamic Bridge Implementation
 */

#include "cognitive/memory/nimcp_memory_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <string.h>

struct memory_thalamic_bridge {
    bridge_base_t base;
    void* memory;
    thalamic_router_t* router;
    memory_thalamic_config_t config;
    memory_thalamic_stats_t stats;
    float attention_weight;
};

memory_thalamic_config_t memory_thalamic_default_config(void) {
    return (memory_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_emotional_boost = true,
        .enable_consolidation_routing = true,
        .min_urgency_threshold = 0.2f,
        .emotional_boost_factor = 1.4f
    };
}

memory_thalamic_bridge_t* memory_thalamic_bridge_create(
    void* memory,
    thalamic_router_t* router,
    const memory_thalamic_config_t* config
) {
    memory_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(memory_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->memory = memory;
    bridge->router = router;
    bridge->config = config ? *config : memory_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void memory_thalamic_bridge_destroy(memory_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) {
        nimcp_mutex_destroy(bridge->base.mutex);
    }
    nimcp_free(bridge);
}

int memory_thalamic_bridge_reset(memory_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int memory_thalamic_route_signal(
    memory_thalamic_bridge_t* bridge,
    const memory_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;
    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->memory_urgency * bridge->attention_weight;

        /* Emotional memories get boost */
        if (bridge->config.enable_emotional_boost && signal->emotional_weight > 0.5f) {
            effective_urgency *= bridge->config.emotional_boost_factor;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            nimcp_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    switch (signal->signal_type) {
        case MEMORY_SIGNAL_ENCODE:
            bridge->stats.encodings_routed++;
            break;
        case MEMORY_SIGNAL_RETRIEVE:
            bridge->stats.retrievals_routed++;
            break;
        case MEMORY_SIGNAL_CONSOLIDATE:
            bridge->stats.consolidations++;
            break;
        case MEMORY_SIGNAL_UPDATE:
            bridge->stats.updates++;
            break;
        default:
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;
    }

    uint64_t total = bridge->stats.encodings_routed + bridge->stats.retrievals_routed +
                     bridge->stats.consolidations + bridge->stats.updates;
    if (total > 0) {
        bridge->stats.avg_strength =
            (bridge->stats.avg_strength * (total - 1) + signal->strength) / total;
        bridge->stats.avg_emotional_weight =
            (bridge->stats.avg_emotional_weight * (total - 1) + signal->emotional_weight) / total;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int memory_thalamic_route_encode(
    memory_thalamic_bridge_t* bridge,
    float strength,
    float emotional_weight
) {
    if (!bridge) return -1;

    memory_thalamic_signal_t signal = {
        .signal_type = MEMORY_SIGNAL_ENCODE,
        .memory_urgency = 0.6f + (emotional_weight * 0.3f),
        .strength = strength < 0.0f ? 0.0f : (strength > 1.0f ? 1.0f : strength),
        .salience = 0.5f,
        .emotional_weight = emotional_weight < 0.0f ? 0.0f : (emotional_weight > 1.0f ? 1.0f : emotional_weight),
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return memory_thalamic_route_signal(bridge, &signal);
}

int memory_thalamic_route_retrieve(
    memory_thalamic_bridge_t* bridge,
    float salience,
    float urgency
) {
    if (!bridge) return -1;

    memory_thalamic_signal_t signal = {
        .signal_type = MEMORY_SIGNAL_RETRIEVE,
        .memory_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .strength = 0.5f,
        .salience = salience < 0.0f ? 0.0f : (salience > 1.0f ? 1.0f : salience),
        .emotional_weight = 0.0f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return memory_thalamic_route_signal(bridge, &signal);
}

int memory_thalamic_set_attention(memory_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int memory_thalamic_get_attention(const memory_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int memory_thalamic_bridge_get_stats(
    const memory_thalamic_bridge_t* bridge,
    memory_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}
