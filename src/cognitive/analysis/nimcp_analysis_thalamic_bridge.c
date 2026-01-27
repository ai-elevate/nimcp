/**
 * @file nimcp_analysis_thalamic_bridge.c
 * @brief Analysis-Thalamic Bridge Implementation
 *
 * Routes analytical processing signals through thalamic relay
 * for prefrontal coordination and attention gating.
 */

#include "cognitive/analysis/nimcp_analysis_thalamic_bridge.h"
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

/** Global health agent for analysis_thalamic_bridge module */
static nimcp_health_agent_t* g_analysis_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for analysis_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void analysis_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_analysis_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from analysis_thalamic_bridge module */
static inline void analysis_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_analysis_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_analysis_thalamic_bridge_health_agent, operation, progress);
    }
}

#define LOG_MODULE "ANALYSIS_THALAMIC_BRIDGE"


struct analysis_thalamic_bridge {
    bridge_base_t base;  /* MUST be first - provides mutex protection */
    void* analysis;
    thalamic_router_t* router;
    analysis_thalamic_config_t config;
    analysis_thalamic_stats_t stats;
    float attention_weight;
};

analysis_thalamic_config_t analysis_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    analysis_thalamic_bridge_heartbeat("analysis_tha_analysis_thalamic_de", 0.0f);


    return (analysis_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_complexity_routing = true,
        .min_urgency_threshold = 0.2f,
        .complexity_boost = 1.3f
    };
}

analysis_thalamic_bridge_t* analysis_thalamic_bridge_create(
    void* analysis,
    thalamic_router_t* router,
    const analysis_thalamic_config_t* config
) {
    /* Phase 8: Heartbeat at operation start */
    analysis_thalamic_bridge_heartbeat("analysis_tha_create", 0.0f);


    analysis_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(analysis_thalamic_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "analysis_thalamic_bridge_create: failed to allocate bridge");
        return NULL;
    }

    /* Initialize mutex for thread safety */
    if (bridge_base_init(&bridge->base, 0, "analysis_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "analysis_thalamic_bridge_create: failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->analysis = analysis;
    bridge->router = router;
    bridge->config = config ? *config : analysis_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void analysis_thalamic_bridge_destroy(analysis_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    analysis_thalamic_bridge_heartbeat("analysis_tha_destroy", 0.0f);


    if (bridge) {
        if (bridge->base.mutex) {
            bridge_base_cleanup(&bridge->base);
        }
        nimcp_free(bridge);
    }
}

int analysis_thalamic_bridge_reset(analysis_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "analysis_thalamic_bridge_reset: bridge is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    analysis_thalamic_bridge_heartbeat("analysis_tha_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int analysis_thalamic_route_signal(
    analysis_thalamic_bridge_t* bridge,
    const analysis_thalamic_signal_t* signal
) {
    if (!bridge || !signal) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "analysis_thalamic_route_signal: bridge or signal is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    analysis_thalamic_bridge_heartbeat("analysis_tha_analysis_thalamic_ro", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Apply attention gating */
    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->analysis_urgency * bridge->attention_weight;

        /* Complex analyses get priority boost */
        if (bridge->config.enable_complexity_routing &&
            signal->complexity > 0.7f) {
            effective_urgency *= bridge->config.complexity_boost;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            nimcp_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    /* Route based on signal type */
    switch (signal->signal_type) {
        case ANALYSIS_SIGNAL_DECOMPOSITION:
            bridge->stats.decompositions_routed++;
            break;

        case ANALYSIS_SIGNAL_DEPTH_REQUEST:
            bridge->stats.depth_requests++;
            break;

        case ANALYSIS_SIGNAL_PATTERN_FOUND:
            bridge->stats.patterns_found++;
            break;

        case ANALYSIS_SIGNAL_COMPLETION:
            bridge->stats.completions++;
            break;

        default:
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;
    }

    /* Update average urgency */
    uint64_t total = bridge->stats.decompositions_routed +
                     bridge->stats.depth_requests +
                     bridge->stats.patterns_found +
                     bridge->stats.completions;

    if (total > 0) {
        bridge->stats.avg_analysis_urgency =
            (bridge->stats.avg_analysis_urgency * (total - 1) +
             signal->analysis_urgency) / total;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int analysis_thalamic_route_decomposition(
    analysis_thalamic_bridge_t* bridge,
    const void* problem,
    uint32_t problem_size,
    float complexity
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "analysis_thalamic_route_decomposition: bridge is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    analysis_thalamic_bridge_heartbeat("analysis_tha_analysis_thalamic_ro", 0.0f);


    analysis_thalamic_signal_t signal = {
        .signal_type = ANALYSIS_SIGNAL_DECOMPOSITION,
        .analysis_urgency = 0.6f + (complexity * 0.3f),
        .depth_required = complexity,
        .complexity = complexity,
        .content = (void*)problem,
        .content_size = problem_size,
        .timestamp_us = nimcp_time_get_us()
    };

    return analysis_thalamic_route_signal(bridge, &signal);
}

int analysis_thalamic_request_depth(
    analysis_thalamic_bridge_t* bridge,
    float depth_required,
    float urgency
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "analysis_thalamic_request_depth: bridge is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    analysis_thalamic_bridge_heartbeat("analysis_tha_analysis_thalamic_re", 0.0f);


    analysis_thalamic_signal_t signal = {
        .signal_type = ANALYSIS_SIGNAL_DEPTH_REQUEST,
        .analysis_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .depth_required = depth_required,
        .complexity = 0.5f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return analysis_thalamic_route_signal(bridge, &signal);
}

int analysis_thalamic_set_attention(analysis_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "analysis_thalamic_set_attention: bridge is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    analysis_thalamic_bridge_heartbeat("analysis_tha_analysis_thalamic_se", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f :
                               (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int analysis_thalamic_get_attention(const analysis_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "analysis_thalamic_get_attention: bridge or attention is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    analysis_thalamic_bridge_heartbeat("analysis_tha_analysis_thalamic_ge", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int analysis_thalamic_bridge_get_stats(
    const analysis_thalamic_bridge_t* bridge,
    analysis_thalamic_stats_t* stats
) {
    if (!bridge || !stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "analysis_thalamic_bridge_get_stats: bridge or stats is NULL");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    analysis_thalamic_bridge_heartbeat("analysis_tha_get_stats", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

void analysis_thalamic_bridge_reset_stats(analysis_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    analysis_thalamic_bridge_heartbeat("analysis_tha_reset_stats", 0.0f);


    if (bridge) {
        nimcp_mutex_lock(bridge->base.mutex);
        memset(&bridge->stats, 0, sizeof(bridge->stats));
        nimcp_mutex_unlock(bridge->base.mutex);
    }
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Analysis Thalamic Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int analysis_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    analysis_thalamic_bridge_heartbeat("analysis_tha_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Analysis_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                analysis_thalamic_bridge_heartbeat("analysis_tha_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            /* Log observation */
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Analysis_Thalamic_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Analysis_Thalamic_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
