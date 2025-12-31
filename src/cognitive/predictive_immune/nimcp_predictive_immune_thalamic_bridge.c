/**
 * @file nimcp_predictive_immune_thalamic_bridge.c
 * @brief Predictive Immune-Thalamic Bridge Implementation
 */

#include "cognitive/predictive_immune/nimcp_predictive_immune_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

struct predictive_immune_thalamic_bridge {
    bridge_base_t base;
    void* predictive_immune;
    thalamic_router_t* router;
    predictive_immune_thalamic_config_t config;
    predictive_immune_thalamic_stats_t stats;
    float attention_weight;
};

predictive_immune_thalamic_config_t predictive_immune_thalamic_default_config(void) {
    predictive_immune_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_urgency_boost = true,
        .min_salience_threshold = 0.2f,
        .urgency_boost = 0.35f
    };
    return cfg;
}

predictive_immune_thalamic_bridge_t* predictive_immune_thalamic_bridge_create(void* predictive_immune, thalamic_router_t* router, const predictive_immune_thalamic_config_t* config) {
    predictive_immune_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(predictive_immune_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->predictive_immune = predictive_immune;
    bridge->router = router;
    bridge->config = config ? *config : predictive_immune_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void predictive_immune_thalamic_bridge_destroy(predictive_immune_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) {
        nimcp_mutex_destroy(bridge->base.mutex);
    }
    nimcp_free(bridge);
}

int predictive_immune_thalamic_bridge_reset(predictive_immune_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_immune_thalamic_route_interoception(predictive_immune_thalamic_bridge_t* bridge, float urgency, float salience) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);

    /* Gate low-salience signals */
    if (bridge->config.enable_attention_gating && salience < bridge->config.min_salience_threshold) {
        bridge->stats.signals_gated++;
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }

    bridge->stats.interoceptions_routed++;
    bridge->stats.avg_immune_salience = (bridge->stats.avg_immune_salience * (bridge->stats.interoceptions_routed - 1) +
                                         salience) / bridge->stats.interoceptions_routed;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_immune_thalamic_route_cytokine(predictive_immune_thalamic_bridge_t* bridge, float level, float importance) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.cytokine_updates++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_immune_thalamic_route_signal(predictive_immune_thalamic_bridge_t* bridge, const predictive_immune_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    nimcp_mutex_lock(bridge->base.mutex);

    /* Apply attention gating */
    float effective_salience = signal->immune_salience * bridge->attention_weight;
    if (bridge->config.enable_attention_gating &&
        effective_salience < bridge->config.min_salience_threshold &&
        signal->interoceptive_urgency < 0.7f) {
        bridge->stats.signals_gated++;
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }

    switch (signal->signal_type) {
        case PRED_IMMUNE_SIGNAL_INTEROCEPTION:
            bridge->stats.interoceptions_routed++;
            break;
        case PRED_IMMUNE_SIGNAL_CYTOKINE:
            bridge->stats.cytokine_updates++;
            break;
        case PRED_IMMUNE_SIGNAL_SICKNESS:
            bridge->stats.sickness_signals++;
            break;
        case PRED_IMMUNE_SIGNAL_RECOVERY:
            /* Recovery signals always pass through */
            break;
        default:
            break;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_immune_thalamic_set_attention(predictive_immune_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_immune_thalamic_get_attention(const predictive_immune_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_immune_thalamic_bridge_get_stats(const predictive_immune_thalamic_bridge_t* bridge, predictive_immune_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}
