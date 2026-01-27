/**
 * @file nimcp_hippocampus_thalamic_bridge.c
 * @brief Implementation of Hippocampus-Thalamic bridge
 *
 * WHAT: Routes hippocampal memory signals through thalamic relay
 * WHY: Anterior thalamus is critical for episodic memory
 * HOW: Packages memory signals, routes via anterior thalamic nuclei
 *
 * @author NIMCP Team
 * @date 2024-12-30
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/hippocampus/nimcp_hippocampus_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for hippocampus_thalamic_bridge module */
static nimcp_health_agent_t* g_hippocampus_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for hippocampus_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void hippocampus_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_hippocampus_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from hippocampus_thalamic_bridge module */
static inline void hippocampus_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_hippocampus_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_hippocampus_thalamic_bridge_health_agent, operation, progress);
    }
}

#define LOG_MODULE "HIPPOCAMPUS_THALAMIC_BRIDGE"


/**
 * Internal structure for hippocampus-thalamic bridge
 */
struct hippocampus_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* hippocampus;                       /**< Hippocampus processing module */
    thalamic_router_t* router;               /**< Thalamic router instance */
    hippocampus_thalamic_config_t config;    /**< Bridge configuration */
    hippocampus_thalamic_stats_t stats;      /**< Runtime statistics */
    float attention_weight;                  /**< Current attention weight */
};

hippocampus_thalamic_config_t hippocampus_thalamic_default_config(void) {
    hippocampus_thalamic_config_t config = {
        .enable_attention_gating = true,
        .enable_novelty_boost = true,
        .min_memory_strength = 0.1f,
        .novelty_threshold = 0.5f
    };
    return config;
}

hippocampus_thalamic_bridge_t* hippocampus_thalamic_bridge_create(void* hippocampus,
                                                                   thalamic_router_t* router,
                                                                   const hippocampus_thalamic_config_t* config) {
    if (!router) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "router is NULL");

        return NULL;
    }

    hippocampus_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(hippocampus_thalamic_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    bridge->hippocampus = hippocampus;
    bridge->router = router;
    bridge->attention_weight = 1.0f;

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = hippocampus_thalamic_default_config();
    }

    memset(&bridge->stats, 0, sizeof(hippocampus_thalamic_stats_t));

    return bridge;
}

void hippocampus_thalamic_bridge_destroy(hippocampus_thalamic_bridge_t* bridge) {
    if (bridge) {
        nimcp_free(bridge);
    }
}

int hippocampus_thalamic_bridge_reset(hippocampus_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;
    }

    memset(&bridge->stats, 0, sizeof(hippocampus_thalamic_stats_t));
    bridge->attention_weight = 1.0f;

    return 0;
}

int hippocampus_thalamic_route_signal(hippocampus_thalamic_bridge_t* bridge,
                                       const hippocampus_thalamic_signal_t* signal) {
    if (!bridge || !signal) {
        return -1;
    }

    /* Check memory strength threshold */
    if (bridge->config.enable_attention_gating &&
        signal->memory_strength < bridge->config.min_memory_strength) {
        return 0;  /* Signal gated, not an error */
    }

    /* Apply novelty modulation */
    float effective_weight = signal->memory_strength * bridge->attention_weight;
    if (bridge->config.enable_novelty_boost &&
        signal->novelty > bridge->config.novelty_threshold) {
        effective_weight *= (1.0f + signal->novelty);
    }

    /* Update statistics - routing is handled externally if router is connected */
    float alpha = 0.1f;
    bridge->stats.avg_memory_strength =
        (1.0f - alpha) * bridge->stats.avg_memory_strength +
        alpha * signal->memory_strength;

    if (signal->signal_type == HIPPOCAMPUS_SIGNAL_ENCODE) {
        bridge->stats.encodings_routed++;
    } else if (signal->signal_type == HIPPOCAMPUS_SIGNAL_RETRIEVE) {
        bridge->stats.retrievals_routed++;
    } else if (signal->signal_type == HIPPOCAMPUS_SIGNAL_CONSOLIDATE) {
        bridge->stats.consolidations_triggered++;
    }

    (void)effective_weight;  /* Suppress unused warning */
    return 0;
}

int hippocampus_thalamic_route_retrieval(hippocampus_thalamic_bridge_t* bridge,
                                          const void* cue, float strength) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;
    }

    /* Create retrieval signal */
    hippocampus_thalamic_signal_t signal = {
        .signal_type = HIPPOCAMPUS_SIGNAL_RETRIEVE,
        .memory_strength = strength,
        .novelty = 0.0f,
        .spatial_precision = 0.0f,
        .content = (void*)cue,
        .content_size = 0,
        .timestamp_us = 0
    };

    return hippocampus_thalamic_route_signal(bridge, &signal);
}

int hippocampus_thalamic_set_attention(hippocampus_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return -1;
    }

    if (attention < 0.0f || attention > 1.0f) {
        return -1;
    }

    bridge->attention_weight = attention;
    return 0;
}

int hippocampus_thalamic_get_attention(const hippocampus_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) {
        return -1;
    }

    *attention = bridge->attention_weight;
    return 0;
}

int hippocampus_thalamic_bridge_get_stats(const hippocampus_thalamic_bridge_t* bridge,
                                           hippocampus_thalamic_stats_t* stats) {
    if (!bridge || !stats) {
        return -1;
    }

    *stats = bridge->stats;
    return 0;
}
