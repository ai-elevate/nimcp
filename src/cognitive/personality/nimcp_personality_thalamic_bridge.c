/**
 * @file nimcp_personality_thalamic_bridge.c
 * @brief Personality-Thalamic Bridge Implementation
 */

#include "cognitive/personality/nimcp_personality_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct personality_thalamic_bridge {
    bridge_base_t base;
    void* personality;
    thalamic_router_t* router;
    personality_thalamic_config_t config;
    personality_thalamic_stats_t stats;
    float attention_weight;
};

personality_thalamic_config_t personality_thalamic_default_config(void) {
    personality_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_state_modulation = true,
        .min_trait_activation = 0.3f,
        .regulation_threshold = 0.5f
    };
    return cfg;
}

personality_thalamic_bridge_t* personality_thalamic_bridge_create(void* personality, thalamic_router_t* router, const personality_thalamic_config_t* config) {
    personality_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(personality_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->personality = personality;
    bridge->router = router;
    bridge->config = config ? *config : personality_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void personality_thalamic_bridge_destroy(personality_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) {
        nimcp_mutex_destroy(bridge->base.mutex);
    }
    nimcp_free(bridge);
}

int personality_thalamic_bridge_reset(personality_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int personality_thalamic_route_trait(personality_thalamic_bridge_t* bridge, const personality_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->trait_activation < bridge->config.min_trait_activation) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.traits_expressed++;
    bridge->stats.avg_trait_activation = (bridge->stats.avg_trait_activation * (bridge->stats.traits_expressed - 1) +
                                          signal->trait_activation) / bridge->stats.traits_expressed;
    if (signal->signal_type == PERSONALITY_SIGNAL_STATE) {
        bridge->stats.state_changes++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int personality_thalamic_route_regulation(personality_thalamic_bridge_t* bridge, const void* regulation, float effort) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    if (effort >= bridge->config.regulation_threshold) {
        bridge->stats.regulations_applied++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int personality_thalamic_set_attention(personality_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int personality_thalamic_get_attention(const personality_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int personality_thalamic_bridge_get_stats(const personality_thalamic_bridge_t* bridge, personality_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
