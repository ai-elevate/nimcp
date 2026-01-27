/**
 * @file nimcp_tom_thalamic_bridge.c
 * @brief Theory of Mind-Thalamic Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/tom/nimcp_tom_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
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

/** Global health agent for tom_thalamic_bridge module */
static nimcp_health_agent_t* g_tom_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for tom_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void tom_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_tom_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from tom_thalamic_bridge module */
static inline void tom_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_tom_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_tom_thalamic_bridge_health_agent, operation, progress);
    }
}

#define LOG_MODULE "TOM_THALAMIC_BRIDGE"


struct tom_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* tom;
    thalamic_router_t* router;
    tom_thalamic_config_t config;
    tom_thalamic_stats_t stats;
    float attention_weight;
};

tom_thalamic_config_t tom_thalamic_default_config(void) {
    return (tom_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_social_boost = true,
        .min_urgency_threshold = 0.2f,
        .social_boost_factor = 1.3f
    };
}

tom_thalamic_bridge_t* tom_thalamic_bridge_create(
    void* tom,
    thalamic_router_t* router,
    const tom_thalamic_config_t* config
) {
    tom_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(tom_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge->tom = tom;
    bridge->router = router;
    bridge->config = config ? *config : tom_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void tom_thalamic_bridge_destroy(tom_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int tom_thalamic_bridge_reset(tom_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int tom_thalamic_route_signal(
    tom_thalamic_bridge_t* bridge,
    const tom_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;

    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->tom_urgency * bridge->attention_weight;

        /* Social relevance gets boost */
        if (bridge->config.enable_social_boost && signal->social_relevance > 0.6f) {
            effective_urgency *= bridge->config.social_boost_factor;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            return 0;
        }
    }

    switch (signal->signal_type) {
        case TOM_SIGNAL_BELIEF_ATTR:
            bridge->stats.belief_attributions++;
            break;
        case TOM_SIGNAL_INTENT_INFER:
            bridge->stats.intent_inferences++;
            break;
        case TOM_SIGNAL_EMOTION_ATTR:
            bridge->stats.emotion_attributions++;
            break;
        case TOM_SIGNAL_PERSPECTIVE:
            bridge->stats.perspective_takes++;
            break;
        default:
            return -1;
    }

    uint64_t total = bridge->stats.belief_attributions + bridge->stats.intent_inferences +
                     bridge->stats.emotion_attributions + bridge->stats.perspective_takes;
    if (total > 0) {
        bridge->stats.avg_mentalizing_depth =
            (bridge->stats.avg_mentalizing_depth * (total - 1) + signal->mentalizing_depth) / total;
        bridge->stats.avg_confidence =
            (bridge->stats.avg_confidence * (total - 1) + signal->confidence) / total;
    }

    return 0;
}

int tom_thalamic_route_belief_attribution(
    tom_thalamic_bridge_t* bridge,
    float depth,
    float confidence
) {
    if (!bridge) return -1;

    tom_thalamic_signal_t signal = {
        .signal_type = TOM_SIGNAL_BELIEF_ATTR,
        .tom_urgency = 0.5f + (depth * 0.3f),
        .mentalizing_depth = depth < 0.0f ? 0.0f : (depth > 1.0f ? 1.0f : depth),
        .confidence = confidence < 0.0f ? 0.0f : (confidence > 1.0f ? 1.0f : confidence),
        .social_relevance = 0.7f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return tom_thalamic_route_signal(bridge, &signal);
}

int tom_thalamic_route_perspective(
    tom_thalamic_bridge_t* bridge,
    float social_relevance,
    float urgency
) {
    if (!bridge) return -1;

    tom_thalamic_signal_t signal = {
        .signal_type = TOM_SIGNAL_PERSPECTIVE,
        .tom_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .mentalizing_depth = 0.6f,
        .confidence = 0.7f,
        .social_relevance = social_relevance < 0.0f ? 0.0f : (social_relevance > 1.0f ? 1.0f : social_relevance),
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return tom_thalamic_route_signal(bridge, &signal);
}

int tom_thalamic_route_inference(tom_thalamic_bridge_t* bridge, const tom_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* Apply attention gating */
    if (bridge->config.enable_attention_gating &&
        signal->social_relevance < bridge->config.min_urgency_threshold) {
        bridge->stats.signals_gated++;
        return 0;
    }
    /* Count the inference based on signal type */
    switch (signal->signal_type) {
        case TOM_SIGNAL_BELIEF_ATTR:
            bridge->stats.belief_attributions++;
            break;
        case TOM_SIGNAL_INTENT_INFER:
            bridge->stats.intent_inferences++;
            break;
        case TOM_SIGNAL_EMOTION_ATTR:
            bridge->stats.emotion_attributions++;
            break;
        case TOM_SIGNAL_PERSPECTIVE:
            bridge->stats.perspective_takes++;
            break;
    }
    /* Update average confidence */
    uint64_t total = bridge->stats.belief_attributions + bridge->stats.intent_inferences +
                     bridge->stats.emotion_attributions + bridge->stats.perspective_takes;
    if (total > 0) {
        bridge->stats.avg_confidence = (bridge->stats.avg_confidence * (total - 1) +
                                        signal->confidence) / total;
    }
    return 0;
}

int tom_thalamic_set_attention(tom_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int tom_thalamic_get_attention(const tom_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int tom_thalamic_bridge_get_stats(
    const tom_thalamic_bridge_t* bridge,
    tom_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int tom_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "ToM_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "ToM_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "ToM_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
