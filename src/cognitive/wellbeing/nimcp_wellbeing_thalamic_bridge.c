/**
 * @file nimcp_wellbeing_thalamic_bridge.c
 * @brief Wellbeing-Thalamic Bridge Implementation
 */

#include "cognitive/wellbeing/nimcp_wellbeing_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"
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

/** Global health agent for wellbeing_thalamic_bridge module */
static nimcp_health_agent_t* g_wellbeing_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for wellbeing_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void wellbeing_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_wellbeing_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from wellbeing_thalamic_bridge module */
static inline void wellbeing_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_wellbeing_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_wellbeing_thalamic_bridge_health_agent, operation, progress);
    }
}


struct wellbeing_thalamic_bridge {
    bridge_base_t base;
    void* wellbeing;
    thalamic_router_t* router;
    wellbeing_thalamic_config_t config;
    wellbeing_thalamic_stats_t stats;
    float attention_weight;
};

wellbeing_thalamic_config_t wellbeing_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    wellbeing_thalamic_bridge_heartbeat("wellbeing_th_wellbeing_thalamic_d", 0.0f);


    return (wellbeing_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_threat_priority = true,
        .min_urgency_threshold = 0.15f,
        .threat_boost = 1.5f
    };
}

wellbeing_thalamic_bridge_t* wellbeing_thalamic_bridge_create(
    void* wellbeing,
    thalamic_router_t* router,
    const wellbeing_thalamic_config_t* config
) {
    /* Phase 8: Heartbeat at operation start */
    wellbeing_thalamic_bridge_heartbeat("wellbeing_th_create", 0.0f);


    wellbeing_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(wellbeing_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    if (bridge_base_init(&bridge->base, 0, "wellbeing_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->wellbeing = wellbeing;
    bridge->router = router;
    bridge->config = config ? *config : wellbeing_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void wellbeing_thalamic_bridge_destroy(wellbeing_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    /* Phase 8: Heartbeat at operation start */
    wellbeing_thalamic_bridge_heartbeat("wellbeing_th_destroy", 0.0f);


    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int wellbeing_thalamic_bridge_reset(wellbeing_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    wellbeing_thalamic_bridge_heartbeat("wellbeing_th_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int wellbeing_thalamic_route_signal(
    wellbeing_thalamic_bridge_t* bridge,
    const wellbeing_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;

    /* Phase 8: Heartbeat at operation start */
    wellbeing_thalamic_bridge_heartbeat("wellbeing_th_wellbeing_thalamic_r", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->wellbeing_urgency * bridge->attention_weight;

        /* Threats get priority boost */
        if (bridge->config.enable_threat_priority &&
            signal->signal_type == WELLBEING_SIGNAL_THREAT) {
            effective_urgency *= bridge->config.threat_boost;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            nimcp_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    switch (signal->signal_type) {
        case WELLBEING_SIGNAL_STATUS:
            bridge->stats.status_updates++;
            break;
        case WELLBEING_SIGNAL_CHANGE:
            bridge->stats.state_changes++;
            break;
        case WELLBEING_SIGNAL_THREAT:
            bridge->stats.threats_signaled++;
            break;
        case WELLBEING_SIGNAL_RECOVERY:
            bridge->stats.recoveries++;
            break;
        default:
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;
    }

    uint64_t total = bridge->stats.status_updates + bridge->stats.state_changes +
                     bridge->stats.threats_signaled + bridge->stats.recoveries;
    if (total > 0) {
        bridge->stats.avg_wellbeing_level =
            (bridge->stats.avg_wellbeing_level * (total - 1) + signal->current_level) / total;
        bridge->stats.avg_stability =
            (bridge->stats.avg_stability * (total - 1) + signal->stability) / total;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int wellbeing_thalamic_route_status(
    wellbeing_thalamic_bridge_t* bridge,
    float level,
    float stability
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    wellbeing_thalamic_bridge_heartbeat("wellbeing_th_wellbeing_thalamic_r", 0.0f);


    wellbeing_thalamic_signal_t signal = {
        .signal_type = WELLBEING_SIGNAL_STATUS,
        .wellbeing_urgency = 0.3f,
        .current_level = level < 0.0f ? 0.0f : (level > 1.0f ? 1.0f : level),
        .change_rate = 0.0f,
        .stability = stability < 0.0f ? 0.0f : (stability > 1.0f ? 1.0f : stability),
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return wellbeing_thalamic_route_signal(bridge, &signal);
}

int wellbeing_thalamic_route_threat(
    wellbeing_thalamic_bridge_t* bridge,
    float severity,
    float urgency
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    wellbeing_thalamic_bridge_heartbeat("wellbeing_th_wellbeing_thalamic_r", 0.0f);


    wellbeing_thalamic_signal_t signal = {
        .signal_type = WELLBEING_SIGNAL_THREAT,
        .wellbeing_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .current_level = 1.0f - severity,
        .change_rate = -severity,
        .stability = 0.2f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return wellbeing_thalamic_route_signal(bridge, &signal);
}

int wellbeing_thalamic_set_attention(wellbeing_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    wellbeing_thalamic_bridge_heartbeat("wellbeing_th_wellbeing_thalamic_s", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int wellbeing_thalamic_get_attention(const wellbeing_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    /* Phase 8: Heartbeat at operation start */
    wellbeing_thalamic_bridge_heartbeat("wellbeing_th_wellbeing_thalamic_g", 0.0f);


    return 0;
}

int wellbeing_thalamic_bridge_get_stats(
    const wellbeing_thalamic_bridge_t* bridge,
    wellbeing_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    /* Phase 8: Heartbeat at operation start */
    wellbeing_thalamic_bridge_heartbeat("wellbeing_th_get_stats", 0.0f);


    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Thalamic Wellbeing Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int thalamic_wellbeing_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    wellbeing_thalamic_bridge_heartbeat("wellbeing_th_thalamic_wellbeing_b", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Thalamic_Wellbeing_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                wellbeing_thalamic_bridge_heartbeat("wellbeing_th_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Thalamic Wellbeing Bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Thalamic_Wellbeing_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Thalamic_Wellbeing_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
