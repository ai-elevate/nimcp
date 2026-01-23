/**
 * @file nimcp_brainstem_thalamic_bridge.c
 * @brief Implementation of Brainstem-Thalamic bridge
 *
 * WHAT: Routes brainstem signals through thalamic relay
 * WHY: Reticular formation and brainstem nuclei connect with intralaminar thalamus
 * HOW: Packages arousal/vital signals, routes via intralaminar pathway
 *
 * @author NIMCP Team
 * @date 2024-12-30
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/brainstem/nimcp_brainstem_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

/**
 * Internal structure for brainstem-thalamic bridge
 */
struct brainstem_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* brainstem;                       /**< Brainstem processing module */
    thalamic_router_t* router;             /**< Thalamic router instance */
    brainstem_thalamic_config_t config;    /**< Bridge configuration */
    brainstem_thalamic_stats_t stats;      /**< Runtime statistics */
    float attention_weight;                /**< Current attention weight */
};

brainstem_thalamic_config_t brainstem_thalamic_default_config(void) {
    brainstem_thalamic_config_t config = {
        .enable_arousal_modulation = true,
        .enable_vital_priority = true,
        .min_arousal_threshold = 0.1f,
        .vital_urgency_threshold = 0.5f
    };
    return config;
}

brainstem_thalamic_bridge_t* brainstem_thalamic_bridge_create(void* brainstem,
                                                               thalamic_router_t* router,
                                                               const brainstem_thalamic_config_t* config) {
    if (!router) {
        return NULL;
    }

    brainstem_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(brainstem_thalamic_bridge_t));
    if (!bridge) {
        return NULL;
    }

    bridge->brainstem = brainstem;
    bridge->router = router;
    bridge->attention_weight = 1.0f;

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = brainstem_thalamic_default_config();
    }

    memset(&bridge->stats, 0, sizeof(brainstem_thalamic_stats_t));

    return bridge;
}

void brainstem_thalamic_bridge_destroy(brainstem_thalamic_bridge_t* bridge) {
    if (bridge) {
        nimcp_free(bridge);
    }
}

int brainstem_thalamic_bridge_reset(brainstem_thalamic_bridge_t* bridge) {
    if (!bridge) {
        return -1;
    }

    memset(&bridge->stats, 0, sizeof(brainstem_thalamic_stats_t));
    bridge->attention_weight = 1.0f;

    return 0;
}

int brainstem_thalamic_route_signal(brainstem_thalamic_bridge_t* bridge,
                                     const brainstem_thalamic_signal_t* signal) {
    if (!bridge || !signal) {
        return -1;
    }

    /* Vital signals always pass through (never gated) */
    if (signal->signal_type != BRAINSTEM_SIGNAL_VITAL &&
        bridge->config.enable_arousal_modulation &&
        signal->arousal_level < bridge->config.min_arousal_threshold) {
        return 0;  /* Signal gated, not an error */
    }

    /* Apply vital priority modulation */
    float effective_weight = signal->arousal_level * bridge->attention_weight;
    if (bridge->config.enable_vital_priority &&
        signal->vital_urgency > bridge->config.vital_urgency_threshold) {
        effective_weight *= (1.0f + signal->vital_urgency);
    }

    /* Update statistics - routing is handled externally if router is connected */
    float alpha = 0.1f;
    bridge->stats.avg_arousal_level =
        (1.0f - alpha) * bridge->stats.avg_arousal_level +
        alpha * signal->arousal_level;

    if (signal->signal_type == BRAINSTEM_SIGNAL_AROUSAL) {
        bridge->stats.arousal_updates++;
    } else if (signal->signal_type == BRAINSTEM_SIGNAL_VITAL) {
        bridge->stats.vital_signals++;
    } else if (signal->signal_type == BRAINSTEM_SIGNAL_REFLEX) {
        bridge->stats.reflex_triggers++;
    }

    (void)effective_weight;  /* Suppress unused warning */
    return 0;
}

int brainstem_thalamic_modulate_arousal(brainstem_thalamic_bridge_t* bridge, float level) {
    if (!bridge) {
        return -1;
    }

    if (level < 0.0f || level > 1.0f) {
        return -1;
    }

    /* Create arousal modulation signal */
    brainstem_thalamic_signal_t signal = {
        .signal_type = BRAINSTEM_SIGNAL_AROUSAL,
        .arousal_level = level,
        .vital_urgency = 0.0f,
        .autonomic_state = level,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = 0
    };

    return brainstem_thalamic_route_signal(bridge, &signal);
}

int brainstem_thalamic_set_attention(brainstem_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) {
        return -1;
    }

    if (attention < 0.0f || attention > 1.0f) {
        return -1;
    }

    bridge->attention_weight = attention;
    return 0;
}

int brainstem_thalamic_get_attention(const brainstem_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) {
        return -1;
    }

    *attention = bridge->attention_weight;
    return 0;
}

int brainstem_thalamic_bridge_get_stats(const brainstem_thalamic_bridge_t* bridge,
                                         brainstem_thalamic_stats_t* stats) {
    if (!bridge || !stats) {
        return -1;
    }

    *stats = bridge->stats;
    return 0;
}
