/**
 * @file nimcp_jepa_thalamic_bridge.c
 * @brief JEPA-Thalamic Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/jepa/nimcp_jepa_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "async/nimcp_bio_messages.h"
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for jepa_thalamic_bridge module */
static nimcp_health_agent_t* g_jepa_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for jepa_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void jepa_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_jepa_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from jepa_thalamic_bridge module */
static inline void jepa_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_jepa_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_jepa_thalamic_bridge_health_agent, operation, progress);
    }
}


/* Forward declarations for bio-async functions used before definition */
bool jepa_thalamic_bridge_register_bio_async(jepa_thalamic_bridge_t* bridge);
void jepa_thalamic_bridge_unregister_bio_async(jepa_thalamic_bridge_t* bridge);

struct jepa_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* jepa;
    thalamic_router_t* router;
    jepa_thalamic_config_t config;
    jepa_thalamic_stats_t stats;
    float attention_weight;
    bool bio_async_registered;
    uint32_t handler_id;
};

jepa_thalamic_config_t jepa_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    jepa_thalamic_bridge_heartbeat("jepa_thalami_jepa_thalamic_defaul", 0.0f);


    jepa_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_error_amplification = true,
        .min_prediction_confidence = 0.3f,
        .max_temporal_horizon = 100,
        .bio_async_enabled = true
    };
    return cfg;
}

jepa_thalamic_bridge_t* jepa_thalamic_bridge_create(void* jepa, thalamic_router_t* router, const jepa_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    jepa_thalamic_bridge_heartbeat("jepa_thalami_create", 0.0f);


    jepa_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(jepa_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    bridge->jepa = jepa;
    bridge->router = router;
    bridge->config = config ? *config : jepa_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    /* Register with bio-async if enabled */
    if (bridge->config.bio_async_enabled) {
        jepa_thalamic_bridge_register_bio_async(bridge);
    }

    return bridge;
}

void jepa_thalamic_bridge_destroy(jepa_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    jepa_thalamic_bridge_heartbeat("jepa_thalami_destroy", 0.0f);

    if (!bridge) return;

    /* Unregister from bio-async if registered */
    if (bridge->bio_async_registered) {
        jepa_thalamic_bridge_unregister_bio_async(bridge);
    }

    nimcp_free(bridge);
}

int jepa_thalamic_bridge_reset(jepa_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    jepa_thalamic_bridge_heartbeat("jepa_thalami_reset", 0.0f);


    NIMCP_CHECK_THROW(bridge, -1, "bridge is NULL");
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int jepa_thalamic_route_prediction(jepa_thalamic_bridge_t* bridge, const jepa_thalamic_signal_t* signal) {
    /* Phase 8: Heartbeat at operation start */
    jepa_thalamic_bridge_heartbeat("jepa_thalami_jepa_thalamic_route_", 0.0f);


    NIMCP_CHECK_THROW(bridge && signal, -1, "bridge or signal is NULL");
    if (bridge->config.enable_attention_gating && signal->prediction_confidence < bridge->config.min_prediction_confidence) {
        return 0;
    }
    if (signal->temporal_horizon > bridge->config.max_temporal_horizon) {
        return 0;
    }
    bridge->stats.predictions_routed++;
    bridge->stats.avg_prediction_confidence = (bridge->stats.avg_prediction_confidence * (bridge->stats.predictions_routed - 1) +
                                               signal->prediction_confidence) / bridge->stats.predictions_routed;
    if (signal->signal_type == JEPA_SIGNAL_EMBEDDING) {
        bridge->stats.embeddings_updated++;
    }
    return 0;
}

int jepa_thalamic_route_error(jepa_thalamic_bridge_t* bridge, const void* error, float magnitude) {
    /* Phase 8: Heartbeat at operation start */
    jepa_thalamic_bridge_heartbeat("jepa_thalami_jepa_thalamic_route_", 0.0f);


    NIMCP_CHECK_THROW(bridge, -1, "bridge is NULL");
    bridge->stats.errors_propagated++;
    return 0;
}

int jepa_thalamic_set_attention(jepa_thalamic_bridge_t* bridge, float attention) {
    /* Phase 8: Heartbeat at operation start */
    jepa_thalamic_bridge_heartbeat("jepa_thalami_jepa_thalamic_set_at", 0.0f);


    NIMCP_CHECK_THROW(bridge, -1, "bridge is NULL");
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int jepa_thalamic_get_attention(const jepa_thalamic_bridge_t* bridge, float* attention) {
    /* Phase 8: Heartbeat at operation start */
    jepa_thalamic_bridge_heartbeat("jepa_thalami_jepa_thalamic_get_at", 0.0f);


    NIMCP_CHECK_THROW(bridge && attention, -1, "bridge or attention is NULL");
    *attention = bridge->attention_weight;
    return 0;
}

int jepa_thalamic_bridge_get_stats(const jepa_thalamic_bridge_t* bridge, jepa_thalamic_stats_t* stats) {
    /* Phase 8: Heartbeat at operation start */
    jepa_thalamic_bridge_heartbeat("jepa_thalami_get_stats", 0.0f);


    NIMCP_CHECK_THROW(bridge && stats, -1, "bridge or stats is NULL");
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

bool jepa_thalamic_bridge_register_bio_async(jepa_thalamic_bridge_t* bridge) {
    if (!bridge || bridge->bio_async_registered) return false;

    bridge->bio_async_registered = true;
    bridge->handler_id = 0;
    return true;
}

void jepa_thalamic_bridge_unregister_bio_async(jepa_thalamic_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_registered) return;

    bridge->bio_async_registered = false;
    bridge->handler_id = 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int jepa_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    jepa_thalamic_bridge_heartbeat("jepa_thalami_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "JEPA_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                jepa_thalamic_bridge_heartbeat("jepa_thalami_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "JEPA_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "JEPA_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
