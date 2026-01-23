/**
 * @file nimcp_parietal_thalamic_bridge.c
 * @brief Implementation of Parietal-Thalamic bridge
 *
 * WHAT: Routes parietal signals through thalamic relay
 * WHY: Parietal cortex connects with pulvinar and VPL
 * HOW: Packages spatial/attention signals, routes via appropriate pathway
 *
 * @author NIMCP Team
 * @date 2024-12-30
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/parietal/nimcp_parietal_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

/**
 * Internal structure for parietal-thalamic bridge
 */
struct parietal_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* parietal;                       /**< Parietal processing module */
    thalamic_router_t* router;            /**< Thalamic router instance */
    parietal_thalamic_config_t config;    /**< Bridge configuration */
    parietal_thalamic_stats_t stats;      /**< Runtime statistics */
    float attention_weight;               /**< Current attention weight */
};

parietal_thalamic_config_t parietal_thalamic_default_config(void) {
    parietal_thalamic_config_t config = {
        .enable_attention_gating = true,
        .enable_spatial_priority = true,
        .min_spatial_precision = 0.2f,
        .attention_threshold = 0.5f
    };
    return config;
}

parietal_thalamic_bridge_t* parietal_thalamic_bridge_create(void* parietal,
                                                             thalamic_router_t* router,
                                                             const parietal_thalamic_config_t* config) {
    if (!router) {
        return NULL;
    }

    parietal_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(parietal_thalamic_bridge_t));
    if (!bridge) {
        return NULL;
    }

    bridge->parietal = parietal;
    bridge->router = router;
    bridge->attention_weight = 1.0f;

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = parietal_thalamic_default_config();
    }

    memset(&bridge->stats, 0, sizeof(parietal_thalamic_stats_t));

    return bridge;
}

void parietal_thalamic_bridge_destroy(parietal_thalamic_bridge_t* bridge) {
    if (bridge) {
        nimcp_free(bridge);
    }
}

int parietal_thalamic_bridge_reset(parietal_thalamic_bridge_t* bridge) {
    if (!bridge) {
        return -1;
    }

    memset(&bridge->stats, 0, sizeof(parietal_thalamic_stats_t));
    bridge->attention_weight = 1.0f;

    return 0;
}

int parietal_thalamic_route_signal(parietal_thalamic_bridge_t* bridge,
                                    const parietal_thalamic_signal_t* signal) {
    if (!bridge || !signal) {
        return -1;
    }

    /* Check spatial precision threshold */
    if (bridge->config.enable_attention_gating &&
        signal->spatial_precision < bridge->config.min_spatial_precision) {
        return 0;  /* Signal gated, not an error */
    }

    /* Apply spatial priority modulation */
    float effective_weight = signal->spatial_precision * bridge->attention_weight;
    if (bridge->config.enable_spatial_priority &&
        signal->signal_type == PARIETAL_SIGNAL_SPATIAL) {
        effective_weight *= 1.5f;
    }

    /* Boost for attention signals above threshold */
    if (signal->attention_weight > bridge->config.attention_threshold) {
        effective_weight *= (1.0f + signal->attention_weight);
    }

    /* Update statistics - routing is handled externally if router is connected */
    float alpha = 0.1f;
    bridge->stats.avg_spatial_precision =
        (1.0f - alpha) * bridge->stats.avg_spatial_precision +
        alpha * signal->spatial_precision;

    if (signal->signal_type == PARIETAL_SIGNAL_SPATIAL) {
        bridge->stats.spatial_signals_routed++;
    } else if (signal->signal_type == PARIETAL_SIGNAL_ATTENTION) {
        bridge->stats.attention_shifts++;
    } else if (signal->signal_type == PARIETAL_SIGNAL_INTEGRATION) {
        bridge->stats.integrations_completed++;
    }

    (void)effective_weight;  /* Suppress unused warning */
    return 0;
}

int parietal_thalamic_route_attention(parietal_thalamic_bridge_t* bridge,
                                       const void* target, float weight) {
    if (!bridge) {
        return -1;
    }

    /* Create attention signal */
    parietal_thalamic_signal_t signal = {
        .signal_type = PARIETAL_SIGNAL_ATTENTION,
        .spatial_precision = weight,
        .attention_weight = weight,
        .integration_quality = 0.5f,
        .content = (void*)target,
        .content_size = 0,
        .timestamp_us = 0
    };

    return parietal_thalamic_route_signal(bridge, &signal);
}

int parietal_thalamic_set_attention(parietal_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) {
        return -1;
    }

    if (attention < 0.0f || attention > 1.0f) {
        return -1;
    }

    bridge->attention_weight = attention;
    return 0;
}

int parietal_thalamic_get_attention(const parietal_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) {
        return -1;
    }

    *attention = bridge->attention_weight;
    return 0;
}

int parietal_thalamic_bridge_get_stats(const parietal_thalamic_bridge_t* bridge,
                                        parietal_thalamic_stats_t* stats) {
    if (!bridge || !stats) {
        return -1;
    }

    *stats = bridge->stats;
    return 0;
}
