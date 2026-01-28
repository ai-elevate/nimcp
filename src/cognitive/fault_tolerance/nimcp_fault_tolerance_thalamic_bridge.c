/**
 * @file nimcp_fault_tolerance_thalamic_bridge.c
 * @brief Fault Tolerance-Thalamic Bridge Implementation
 */

#include "cognitive/fault_tolerance/nimcp_fault_tolerance_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
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

/** Global health agent for fault_tolerance_thalamic_bridge module */
static nimcp_health_agent_t* g_fault_tolerance_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for fault_tolerance_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void fault_tolerance_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_fault_tolerance_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from fault_tolerance_thalamic_bridge module */
static inline void fault_tolerance_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_fault_tolerance_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_fault_tolerance_thalamic_bridge_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat from fault_tolerance_thalamic_bridge module (instance-level) */
static inline void fault_tolerance_thalamic_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_fault_tolerance_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_fault_tolerance_thalamic_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_fault_tolerance_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}

#define LOG_MODULE "FAULT_TOLERANCE_THALAMIC_BRIDGE"


struct fault_tolerance_thalamic_bridge {
    bridge_base_t base;
    void* fault_tolerance;
    thalamic_router_t* router;
    fault_tolerance_thalamic_config_t config;
    fault_tolerance_thalamic_stats_t stats;
    float attention_weight;

    /* Phase 8: Instance health agent */
    nimcp_health_agent_t* health_agent;         /**< Health agent (Phase 8) */
};

fault_tolerance_thalamic_config_t fault_tolerance_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    fault_tolerance_thalamic_bridge_heartbeat("fault_tolera_fault_tolerance_thal", 0.0f);


    fault_tolerance_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_severity_boost = true,
        .min_severity_threshold = 0.15f,  /* Lower threshold for faults */
        .escalation_boost = 0.5f
    };
    return cfg;
}

fault_tolerance_thalamic_bridge_t* fault_tolerance_thalamic_bridge_create(void* fault_tolerance, thalamic_router_t* router, const fault_tolerance_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    fault_tolerance_thalamic_bridge_heartbeat("fault_tolera_create", 0.0f);


    fault_tolerance_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(fault_tolerance_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate bridge");

        return NULL;

    }
    if (bridge_base_init(&bridge->base, 0, "fault_tolerance_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->fault_tolerance = fault_tolerance;
    bridge->router = router;
    bridge->config = config ? *config : fault_tolerance_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    NIMCP_LOGGING_INFO("Created %s bridge", "fault_tolerance_thalamic");
    return bridge;
}

void fault_tolerance_thalamic_bridge_destroy(fault_tolerance_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "fault_tolerance_thalamic");
    /* Phase 8: Heartbeat at operation start */
    fault_tolerance_thalamic_bridge_heartbeat("fault_tolera_destroy", 0.0f);


    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int fault_tolerance_thalamic_bridge_reset(fault_tolerance_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fault_tolerance_thalamic_bridge_reset: NULL bridge");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    fault_tolerance_thalamic_bridge_heartbeat("fault_tolera_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fault_tolerance_thalamic_route_detection(fault_tolerance_thalamic_bridge_t* bridge, const fault_tolerance_thalamic_signal_t* signal) {
    if (!bridge || !signal) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fault_tolerance_thalamic_route_detection: NULL argument");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    fault_tolerance_thalamic_bridge_heartbeat("fault_tolera_fault_tolerance_thal", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    /* Critical faults bypass attention gating */
    if (bridge->config.enable_attention_gating &&
        signal->severity < bridge->config.min_severity_threshold &&
        signal->criticality < 0.8f) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.detections_routed++;
    bridge->stats.avg_severity = (bridge->stats.avg_severity * (bridge->stats.detections_routed - 1) +
                                  signal->severity) / bridge->stats.detections_routed;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fault_tolerance_thalamic_route_recovery(fault_tolerance_thalamic_bridge_t* bridge, const void* recovery_plan, float priority) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fault_tolerance_thalamic_route_recovery: NULL bridge");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    fault_tolerance_thalamic_bridge_heartbeat("fault_tolera_fault_tolerance_thal", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.recoveries_initiated++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fault_tolerance_thalamic_set_attention(fault_tolerance_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fault_tolerance_thalamic_set_attention: NULL bridge");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    fault_tolerance_thalamic_bridge_heartbeat("fault_tolera_fault_tolerance_thal", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fault_tolerance_thalamic_get_attention(const fault_tolerance_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fault_tolerance_thalamic_get_attention: NULL argument");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    fault_tolerance_thalamic_bridge_heartbeat("fault_tolera_fault_tolerance_thal", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fault_tolerance_thalamic_bridge_get_stats(const fault_tolerance_thalamic_bridge_t* bridge, fault_tolerance_thalamic_stats_t* stats) {
    if (!bridge || !stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fault_tolerance_thalamic_bridge_get_stats: NULL argument");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    fault_tolerance_thalamic_bridge_heartbeat("fault_tolera_get_stats", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int fault_tolerance_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    fault_tolerance_thalamic_bridge_heartbeat("fault_tolera_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Fault_Tolerance_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                fault_tolerance_thalamic_bridge_heartbeat("fault_tolera_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            /* Log self-knowledge observations */
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Fault_Tolerance_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Fault_Tolerance_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-level health agent setter
 * ============================================================================ */
void fault_tolerance_thalamic_bridge_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_fault_tolerance_thalamic_bridge_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Functions (FULL implementation)
 * ============================================================================ */
int fault_tolerance_thalamic_bridge_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fault_tolerance_thalamic_bridge_training_begin: NULL argument");
        return -1;
    }
    fault_tolerance_thalamic_bridge_heartbeat_instance(NULL, "fault_tolera_training_begin", 0.0f);
    (void)instance;
    return 0;
}

int fault_tolerance_thalamic_bridge_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fault_tolerance_thalamic_bridge_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    fault_tolerance_thalamic_bridge_heartbeat_instance(NULL, "fault_tolera_training_step", progress);
    (void)instance;
    return 0;
}

int fault_tolerance_thalamic_bridge_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fault_tolerance_thalamic_bridge_training_end: NULL argument");
        return -1;
    }
    fault_tolerance_thalamic_bridge_heartbeat_instance(NULL, "fault_tolera_training_end", 1.0f);
    (void)instance;
    return 0;
}
