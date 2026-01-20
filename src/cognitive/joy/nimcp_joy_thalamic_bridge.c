/**
 * @file nimcp_joy_thalamic_bridge.c
 * @brief Joy-Thalamic Bridge Implementation
 *
 * WHAT: Routes joy/positive affect signals through the thalamic router
 * WHY: Joy experience requires conscious awareness via thalamic gating
 * HOW: Packages joy signals into routed_signal_t and calls thalamic_router_route_signal
 */

#include "cognitive/joy/nimcp_joy_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <string.h>

/* Source ID for joy signals in thalamic routing */
#define JOY_THALAMIC_SOURCE_ID 0x1200

/* Default destination IDs for joy signals */
#define JOY_DEST_VENTRAL_STRIATUM  0x2101
#define JOY_DEST_ORBITOFRONTAL     0x2102
#define JOY_DEST_PREFRONTAL        0x2103

struct joy_thalamic_bridge {
    bridge_base_t base;
    void* joy;
    thalamic_router_t* router;
    joy_thalamic_config_t config;
    joy_thalamic_stats_t stats;
    float attention_weight;
};

joy_thalamic_config_t joy_thalamic_default_config(void) {
    joy_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_savoring_boost = true,
        .min_hedonic_threshold = 0.3f,
        .anticipation_threshold = 0.5f
    };
    return cfg;
}

joy_thalamic_bridge_t* joy_thalamic_bridge_create(void* joy, thalamic_router_t* router, const joy_thalamic_config_t* config) {
    joy_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(joy_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->joy = joy;
    bridge->router = router;
    bridge->config = config ? *config : joy_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void joy_thalamic_bridge_destroy(joy_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) {
        nimcp_mutex_free(bridge->base.mutex);
    }
    nimcp_free(bridge);
}

int joy_thalamic_bridge_reset(joy_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/**
 * @brief Route joy/pleasure signal through thalamic router
 *
 * WHAT: Package joy signal and route through thalamic attention mechanism
 * WHY: Joy signals need attention-gated conscious experience
 * HOW: Create routed_signal_t, apply attention weight, call router
 */
int joy_thalamic_route_pleasure(joy_thalamic_bridge_t* bridge, const joy_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Attention gating: filter signals below threshold */
    if (bridge->config.enable_attention_gating &&
        signal->hedonic_value < bridge->config.min_hedonic_threshold) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;  /* Signal gated, not an error */
    }

    /* Route signal through thalamic router if available */
    if (bridge->router) {
        /* Define destinations for joy signals */
        uint32_t dest_ids[] = {
            JOY_DEST_VENTRAL_STRIATUM,
            JOY_DEST_ORBITOFRONTAL,
            JOY_DEST_PREFRONTAL
        };
        uint32_t num_dests = sizeof(dest_ids) / sizeof(dest_ids[0]);

        /* Package signal data: joy_intensity, hedonic_value, duration_expectation */
        float signal_data[3] = {
            signal->joy_intensity,
            signal->hedonic_value,
            signal->duration_expectation
        };

        /* Determine priority based on joy intensity */
        signal_priority_t priority = SIGNAL_PRIORITY_NORMAL;
        if (signal->joy_intensity > 0.8f) {
            priority = SIGNAL_PRIORITY_HIGH;
        } else if (signal->joy_intensity < 0.4f) {
            priority = SIGNAL_PRIORITY_LOW;
        }

        /* Create routed signal packet */
        routed_signal_t routed = {
            .source_id = JOY_THALAMIC_SOURCE_ID | signal->signal_type,
            .dest_ids = dest_ids,
            .num_dests = num_dests,
            .signal_data = signal_data,
            .signal_size = 3,
            .attention_weight = bridge->attention_weight,
            .priority = priority,
            .timestamp_ms = nimcp_time_get_ms(),
            .bypass_queue = (priority == SIGNAL_PRIORITY_HIGH)
        };

        /* Route through thalamic router */
        bool routed_ok = thalamic_router_route_signal(bridge->router, &routed);
        if (!routed_ok) {
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;  /* Routing failed */
        }
    }

    /* Update statistics AFTER successful routing */
    bridge->stats.pleasures_routed++;
    bridge->stats.avg_joy_intensity = (bridge->stats.avg_joy_intensity * (bridge->stats.pleasures_routed - 1) +
                                       signal->joy_intensity) / bridge->stats.pleasures_routed;

    if (signal->signal_type == JOY_SIGNAL_ANTICIPATION) {
        bridge->stats.anticipations_triggered++;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/**
 * @brief Route savoring experience through thalamic router
 *
 * WHAT: Route savoring experience through attention-gated pathway
 * WHY: Savoring requires sustained conscious attention for enhancement
 * HOW: Package duration data, route with boost if enabled
 */
int joy_thalamic_route_savoring(joy_thalamic_bridge_t* bridge, const void* experience, float duration) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Route through thalamic router if available */
    if (bridge->router) {
        uint32_t dest_ids[] = {
            JOY_DEST_ORBITOFRONTAL,
            JOY_DEST_PREFRONTAL
        };
        uint32_t num_dests = sizeof(dest_ids) / sizeof(dest_ids[0]);

        /* Package duration as signal data */
        float signal_data[1] = { duration };

        /* Savoring gets boosted attention if enabled */
        float attention = bridge->attention_weight;
        if (bridge->config.enable_savoring_boost) {
            attention = attention * 1.2f;
            if (attention > 1.0f) attention = 1.0f;
        }

        routed_signal_t routed = {
            .source_id = JOY_THALAMIC_SOURCE_ID | JOY_SIGNAL_SAVORING,
            .dest_ids = dest_ids,
            .num_dests = num_dests,
            .signal_data = signal_data,
            .signal_size = 1,
            .attention_weight = attention,
            .priority = SIGNAL_PRIORITY_NORMAL,
            .timestamp_ms = nimcp_time_get_ms(),
            .bypass_queue = false
        };

        bool routed_ok = thalamic_router_route_signal(bridge->router, &routed);
        if (!routed_ok) {
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;
        }
    }

    /* Update statistics after successful routing */
    if (bridge->config.enable_savoring_boost) {
        bridge->stats.savoring_episodes++;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int joy_thalamic_set_attention(joy_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int joy_thalamic_get_attention(joy_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int joy_thalamic_bridge_get_stats(joy_thalamic_bridge_t* bridge, joy_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int joy_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Joy_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Joy_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Joy_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
