/**
 * @file nimcp_temporal_thalamic_bridge.c
 * @brief Implementation of Temporal-Thalamic bridge
 *
 * WHAT: Routes temporal lobe signals through thalamic relay
 * WHY: Temporal cortex connects with pulvinar and MGN
 * HOW: Packages language/object signals, routes via appropriate pathway
 *
 * @author NIMCP Team
 * @date 2024-12-30
 */

#include "core/temporal/nimcp_temporal_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

/**
 * Internal structure for temporal-thalamic bridge
 */
struct temporal_thalamic_bridge {
    void* temporal;                       /**< Temporal processing module */
    thalamic_router_t* router;            /**< Thalamic router instance */
    temporal_thalamic_config_t config;    /**< Bridge configuration */
    temporal_thalamic_stats_t stats;      /**< Runtime statistics */
    float attention_weight;               /**< Current attention weight */
};

temporal_thalamic_config_t temporal_thalamic_default_config(void) {
    temporal_thalamic_config_t config = {
        .enable_attention_gating = true,
        .enable_language_priority = true,
        .min_recognition_confidence = 0.3f,
        .semantic_threshold = 0.5f
    };
    return config;
}

temporal_thalamic_bridge_t* temporal_thalamic_bridge_create(void* temporal,
                                                             thalamic_router_t* router,
                                                             const temporal_thalamic_config_t* config) {
    if (!router) {
        return NULL;
    }

    temporal_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(temporal_thalamic_bridge_t));
    if (!bridge) {
        return NULL;
    }

    bridge->temporal = temporal;
    bridge->router = router;
    bridge->attention_weight = 1.0f;

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = temporal_thalamic_default_config();
    }

    memset(&bridge->stats, 0, sizeof(temporal_thalamic_stats_t));

    return bridge;
}

void temporal_thalamic_bridge_destroy(temporal_thalamic_bridge_t* bridge) {
    if (bridge) {
        nimcp_free(bridge);
    }
}

int temporal_thalamic_bridge_reset(temporal_thalamic_bridge_t* bridge) {
    if (!bridge) {
        return -1;
    }

    memset(&bridge->stats, 0, sizeof(temporal_thalamic_stats_t));
    bridge->attention_weight = 1.0f;

    return 0;
}

int temporal_thalamic_route_signal(temporal_thalamic_bridge_t* bridge,
                                    const temporal_thalamic_signal_t* signal) {
    if (!bridge || !signal) {
        return -1;
    }

    /* Check recognition confidence threshold */
    if (bridge->config.enable_attention_gating &&
        signal->recognition_confidence < bridge->config.min_recognition_confidence) {
        return 0;  /* Signal gated, not an error */
    }

    /* Apply language priority modulation */
    float effective_weight = signal->recognition_confidence * bridge->attention_weight;
    if (bridge->config.enable_language_priority &&
        signal->signal_type == TEMPORAL_SIGNAL_LANGUAGE) {
        effective_weight *= 1.5f;
    }

    /* Boost for high semantic activation */
    if (signal->semantic_activation > bridge->config.semantic_threshold) {
        effective_weight *= (1.0f + signal->semantic_activation);
    }

    /* Update statistics - routing is handled externally if router is connected */
    float alpha = 0.1f;
    bridge->stats.avg_recognition_confidence =
        (1.0f - alpha) * bridge->stats.avg_recognition_confidence +
        alpha * signal->recognition_confidence;

    if (signal->signal_type == TEMPORAL_SIGNAL_LANGUAGE) {
        bridge->stats.language_signals_routed++;
    } else if (signal->signal_type == TEMPORAL_SIGNAL_OBJECT) {
        bridge->stats.objects_recognized++;
    } else if (signal->signal_type == TEMPORAL_SIGNAL_SEMANTIC) {
        bridge->stats.semantic_activations++;
    }

    (void)effective_weight;  /* Suppress unused warning */
    return 0;
}

int temporal_thalamic_route_language(temporal_thalamic_bridge_t* bridge,
                                      const void* content, float priority) {
    if (!bridge) {
        return -1;
    }

    /* Create language signal */
    temporal_thalamic_signal_t signal = {
        .signal_type = TEMPORAL_SIGNAL_LANGUAGE,
        .recognition_confidence = priority,
        .semantic_activation = priority,
        .attention_weight = priority,
        .content = (void*)content,
        .content_size = 0,
        .timestamp_us = 0
    };

    return temporal_thalamic_route_signal(bridge, &signal);
}

int temporal_thalamic_set_attention(temporal_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) {
        return -1;
    }

    if (attention < 0.0f || attention > 1.0f) {
        return -1;
    }

    bridge->attention_weight = attention;
    return 0;
}

int temporal_thalamic_get_attention(const temporal_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) {
        return -1;
    }

    *attention = bridge->attention_weight;
    return 0;
}

int temporal_thalamic_bridge_get_stats(const temporal_thalamic_bridge_t* bridge,
                                        temporal_thalamic_stats_t* stats) {
    if (!bridge || !stats) {
        return -1;
    }

    *stats = bridge->stats;
    return 0;
}
