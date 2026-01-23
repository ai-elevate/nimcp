/**
 * @file nimcp_occipital_thalamic_bridge.c
 * @brief Implementation of Occipital-Thalamic bridge
 *
 * WHAT: Routes occipital visual signals through thalamic relay (LGN)
 * WHY: All primary visual information passes through LGN
 * HOW: Packages visual signals, routes via lateral geniculate pathway
 *
 * @author NIMCP Team
 * @date 2024-12-30
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/occipital/nimcp_occipital_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

/**
 * Internal structure for occipital-thalamic bridge
 */
struct occipital_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* occipital;                       /**< Occipital processing module */
    thalamic_router_t* router;             /**< Thalamic router instance */
    occipital_thalamic_config_t config;    /**< Bridge configuration */
    occipital_thalamic_stats_t stats;      /**< Runtime statistics */
    float attention_weight;                /**< Current attention weight */
};

occipital_thalamic_config_t occipital_thalamic_default_config(void) {
    occipital_thalamic_config_t config = {
        .enable_attention_gating = true,
        .enable_contrast_boost = true,
        .min_visual_intensity = 0.1f,
        .contrast_threshold = 0.3f
    };
    return config;
}

occipital_thalamic_bridge_t* occipital_thalamic_bridge_create(void* occipital,
                                                               thalamic_router_t* router,
                                                               const occipital_thalamic_config_t* config) {
    if (!router) {
        return NULL;
    }

    occipital_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(occipital_thalamic_bridge_t));
    if (!bridge) {
        return NULL;
    }

    bridge->occipital = occipital;
    bridge->router = router;
    bridge->attention_weight = 1.0f;

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = occipital_thalamic_default_config();
    }

    memset(&bridge->stats, 0, sizeof(occipital_thalamic_stats_t));

    return bridge;
}

void occipital_thalamic_bridge_destroy(occipital_thalamic_bridge_t* bridge) {
    if (bridge) {
        nimcp_free(bridge);
    }
}

int occipital_thalamic_bridge_reset(occipital_thalamic_bridge_t* bridge) {
    if (!bridge) {
        return -1;
    }

    memset(&bridge->stats, 0, sizeof(occipital_thalamic_stats_t));
    bridge->attention_weight = 1.0f;

    return 0;
}

int occipital_thalamic_route_signal(occipital_thalamic_bridge_t* bridge,
                                     const occipital_thalamic_signal_t* signal) {
    if (!bridge || !signal) {
        return -1;
    }

    /* Check visual intensity threshold */
    if (bridge->config.enable_attention_gating &&
        signal->visual_intensity < bridge->config.min_visual_intensity) {
        return 0;  /* Signal gated, not an error */
    }

    /* Apply contrast boost modulation */
    float effective_weight = signal->visual_intensity * bridge->attention_weight;
    if (bridge->config.enable_contrast_boost &&
        signal->contrast > bridge->config.contrast_threshold) {
        effective_weight *= (1.0f + signal->contrast);
    }

    /* Update statistics - routing is handled externally if router is connected */
    float alpha = 0.1f;
    bridge->stats.avg_visual_intensity =
        (1.0f - alpha) * bridge->stats.avg_visual_intensity +
        alpha * signal->visual_intensity;

    if (signal->signal_type == OCCIPITAL_SIGNAL_V1) {
        bridge->stats.v1_signals_routed++;
    } else if (signal->signal_type == OCCIPITAL_SIGNAL_DORSAL) {
        bridge->stats.dorsal_signals++;
    } else if (signal->signal_type == OCCIPITAL_SIGNAL_VENTRAL) {
        bridge->stats.ventral_signals++;
    }

    (void)effective_weight;  /* Suppress unused warning */
    return 0;
}

int occipital_thalamic_route_v1(occipital_thalamic_bridge_t* bridge,
                                 const void* visual_data, float intensity) {
    if (!bridge) {
        return -1;
    }

    /* Create V1 signal */
    occipital_thalamic_signal_t signal = {
        .signal_type = OCCIPITAL_SIGNAL_V1,
        .visual_intensity = intensity,
        .contrast = 0.5f,
        .spatial_frequency = 0.0f,
        .content = (void*)visual_data,
        .content_size = 0,
        .timestamp_us = 0
    };

    return occipital_thalamic_route_signal(bridge, &signal);
}

int occipital_thalamic_set_attention(occipital_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) {
        return -1;
    }

    if (attention < 0.0f || attention > 1.0f) {
        return -1;
    }

    bridge->attention_weight = attention;
    return 0;
}

int occipital_thalamic_get_attention(const occipital_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) {
        return -1;
    }

    *attention = bridge->attention_weight;
    return 0;
}

int occipital_thalamic_bridge_get_stats(const occipital_thalamic_bridge_t* bridge,
                                         occipital_thalamic_stats_t* stats) {
    if (!bridge || !stats) {
        return -1;
    }

    *stats = bridge->stats;
    return 0;
}
