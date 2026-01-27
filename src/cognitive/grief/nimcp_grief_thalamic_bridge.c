/**
 * @file nimcp_grief_thalamic_bridge.c
 * @brief Grief-Thalamic Bridge Implementation
 *
 * WHAT: Routes grief signals through the thalamic router for attention-gated processing
 * WHY: Grief processing requires conscious integration via thalamic gating
 * HOW: Packages grief signals into routed_signal_t and calls thalamic_router_route_signal
 */

#include "cognitive/grief/nimcp_grief_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for grief_thalamic_bridge module */
static nimcp_health_agent_t* g_grief_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for grief_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void grief_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_grief_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from grief_thalamic_bridge module */
static inline void grief_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_grief_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_grief_thalamic_bridge_health_agent, operation, progress);
    }
}

#define LOG_MODULE "GRIEF_THALAMIC_BRIDGE"


/* Source ID for grief signals in thalamic routing */
#define GRIEF_THALAMIC_SOURCE_ID 0x1100

/* Default destination IDs for grief signals */
#define GRIEF_DEST_ANTERIOR_CINGULATE 0x2001
#define GRIEF_DEST_LIMBIC            0x2002
#define GRIEF_DEST_PREFRONTAL        0x2003

struct grief_thalamic_bridge {
    bridge_base_t base;
    void* grief;
    thalamic_router_t* router;
    grief_thalamic_config_t config;
    grief_thalamic_stats_t stats;
    float attention_weight;
};

grief_thalamic_config_t grief_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    grief_thalamic_bridge_heartbeat("grief_thalam_grief_thalamic_defau", 0.0f);


    grief_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_gradual_processing = true,
        .min_intensity_threshold = 0.3f,
        .adaptation_threshold = 0.5f
    };
    return cfg;
}

grief_thalamic_bridge_t* grief_thalamic_bridge_create(void* grief, thalamic_router_t* router, const grief_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    grief_thalamic_bridge_heartbeat("grief_thalam_create", 0.0f);


    grief_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(grief_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    if (bridge_base_init(&bridge->base, 0, "grief_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->grief = grief;
    bridge->router = router;
    bridge->config = config ? *config : grief_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    NIMCP_LOGGING_INFO("Created %s bridge", "grief_thalamic");
    return bridge;
}

void grief_thalamic_bridge_destroy(grief_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "grief_thalamic");
    /* Phase 8: Heartbeat at operation start */
    grief_thalamic_bridge_heartbeat("grief_thalam_destroy", 0.0f);


    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int grief_thalamic_bridge_reset(grief_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    grief_thalamic_bridge_heartbeat("grief_thalam_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/**
 * @brief Route grief loss signal through thalamic router
 *
 * WHAT: Package grief signal and route through thalamic attention mechanism
 * WHY: Grief signals need attention-gated conscious processing
 * HOW: Create routed_signal_t, apply attention weight, call router
 */
int grief_thalamic_route_loss(grief_thalamic_bridge_t* bridge, const grief_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;

    /* Phase 8: Heartbeat at operation start */
    grief_thalamic_bridge_heartbeat("grief_thalam_grief_thalamic_route", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Attention gating: filter signals below threshold */
    if (bridge->config.enable_attention_gating &&
        signal->grief_intensity < bridge->config.min_intensity_threshold) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;  /* Signal gated, not an error */
    }

    /* Route signal through thalamic router if available */
    if (bridge->router) {
        /* Define destinations for grief signals */
        uint32_t dest_ids[] = {
            GRIEF_DEST_ANTERIOR_CINGULATE,
            GRIEF_DEST_LIMBIC,
            GRIEF_DEST_PREFRONTAL
        };
        uint32_t num_dests = sizeof(dest_ids) / sizeof(dest_ids[0]);

        /* Package signal data: grief_intensity, processing_stage, adaptation_progress */
        float signal_data[3] = {
            signal->grief_intensity,
            signal->processing_stage,
            signal->adaptation_progress
        };

        /* Determine priority based on grief intensity */
        signal_priority_t priority = SIGNAL_PRIORITY_NORMAL;
        if (signal->grief_intensity > 0.8f) {
            priority = SIGNAL_PRIORITY_HIGH;
        } else if (signal->grief_intensity < 0.4f) {
            priority = SIGNAL_PRIORITY_LOW;
        }

        /* Create routed signal packet */
        routed_signal_t routed = {
            .source_id = GRIEF_THALAMIC_SOURCE_ID | signal->signal_type,
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
    bridge->stats.losses_processed++;
    bridge->stats.avg_grief_intensity = (bridge->stats.avg_grief_intensity * (bridge->stats.losses_processed - 1) +
                                         signal->grief_intensity) / bridge->stats.losses_processed;

    if (signal->signal_type == GRIEF_SIGNAL_ADAPTATION &&
        signal->adaptation_progress >= bridge->config.adaptation_threshold) {
        bridge->stats.adaptations_achieved++;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/**
 * @brief Route grief processing stage through thalamic router
 *
 * WHAT: Route grief processing progress through attention-gated pathway
 * WHY: Processing stages need conscious awareness for integration
 * HOW: Package stage data, route with appropriate priority
 */
int grief_thalamic_route_processing(grief_thalamic_bridge_t* bridge, const void* stage, float progress) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    grief_thalamic_bridge_heartbeat("grief_thalam_grief_thalamic_route", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Route through thalamic router if available */
    if (bridge->router) {
        uint32_t dest_ids[] = {
            GRIEF_DEST_ANTERIOR_CINGULATE,
            GRIEF_DEST_PREFRONTAL
        };
        uint32_t num_dests = sizeof(dest_ids) / sizeof(dest_ids[0]);

        /* Package progress as signal data */
        float signal_data[1] = { progress };

        routed_signal_t routed = {
            .source_id = GRIEF_THALAMIC_SOURCE_ID | GRIEF_SIGNAL_PROCESSING,
            .dest_ids = dest_ids,
            .num_dests = num_dests,
            .signal_data = signal_data,
            .signal_size = 1,
            .attention_weight = bridge->attention_weight,
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
    bridge->stats.processing_stages++;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int grief_thalamic_set_attention(grief_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    grief_thalamic_bridge_heartbeat("grief_thalam_grief_thalamic_set_a", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int grief_thalamic_get_attention(grief_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    /* Phase 8: Heartbeat at operation start */
    grief_thalamic_bridge_heartbeat("grief_thalam_grief_thalamic_get_a", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int grief_thalamic_bridge_get_stats(grief_thalamic_bridge_t* bridge, grief_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    /* Phase 8: Heartbeat at operation start */
    grief_thalamic_bridge_heartbeat("grief_thalam_get_stats", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int grief_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    grief_thalamic_bridge_heartbeat("grief_thalam_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Grief_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                grief_thalamic_bridge_heartbeat("grief_thalam_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Grief_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Grief_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
