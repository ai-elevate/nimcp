/**
 * @file nimcp_epistemic_thalamic_bridge.c
 * @brief Epistemic-Thalamic Bridge Implementation
 */

#include "cognitive/epistemic/nimcp_epistemic_thalamic_bridge.h"
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

/** Global health agent for epistemic_thalamic_bridge module */
static nimcp_health_agent_t* g_epistemic_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for epistemic_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void epistemic_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_epistemic_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from epistemic_thalamic_bridge module */
static inline void epistemic_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_epistemic_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_epistemic_thalamic_bridge_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat from epistemic_thalamic_bridge module (instance-level) */
static inline void epistemic_thalamic_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_epistemic_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_epistemic_thalamic_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_epistemic_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


#define LOG_MODULE "EPISTEMIC_THALAMIC_BRIDGE"


struct epistemic_thalamic_bridge {
    bridge_base_t base;  /* MUST be first - provides mutex protection */
    nimcp_health_agent_t* health_agent;  /**< Phase 8: instance-level health agent */
    void* epistemic;
    thalamic_router_t* router;
    epistemic_thalamic_config_t config;
    epistemic_thalamic_stats_t stats;
    float attention_weight;
};

epistemic_thalamic_config_t epistemic_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    epistemic_thalamic_bridge_heartbeat("epistemic_th_epistemic_thalamic_d", 0.0f);


    return (epistemic_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_uncertainty_boost = true,
        .min_urgency_threshold = 0.25f,
        .uncertainty_boost_factor = 1.3f
    };
}

epistemic_thalamic_bridge_t* epistemic_thalamic_bridge_create(
    void* epistemic,
    thalamic_router_t* router,
    const epistemic_thalamic_config_t* config
) {
    /* Phase 8: Heartbeat at operation start */
    epistemic_thalamic_bridge_heartbeat("epistemic_th_create", 0.0f);


    epistemic_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(epistemic_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate bridge");

        return NULL;

    }

    /* Initialize mutex for thread safety */
    if (bridge_base_init(&bridge->base, 0, "epistemic_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->epistemic = epistemic;
    bridge->router = router;
    bridge->config = config ? *config : epistemic_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void epistemic_thalamic_bridge_destroy(epistemic_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    epistemic_thalamic_bridge_heartbeat("epistemic_th_destroy", 0.0f);


    if (bridge) {
        if (bridge->base.mutex) {
            bridge_base_cleanup(&bridge->base);
        }
        nimcp_free(bridge);
    }
}

int epistemic_thalamic_bridge_reset(epistemic_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    epistemic_thalamic_bridge_heartbeat("epistemic_th_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int epistemic_thalamic_route_signal(
    epistemic_thalamic_bridge_t* bridge,
    const epistemic_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;

    /* Phase 8: Heartbeat at operation start */
    epistemic_thalamic_bridge_heartbeat("epistemic_th_epistemic_thalamic_r", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->epistemic_urgency * bridge->attention_weight;

        /* Uncertainty gets boost (drives exploration) */
        if (bridge->config.enable_uncertainty_boost &&
            signal->signal_type == EPISTEMIC_SIGNAL_UNCERTAINTY) {
            effective_urgency *= bridge->config.uncertainty_boost_factor;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            nimcp_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    switch (signal->signal_type) {
        case EPISTEMIC_SIGNAL_UNCERTAINTY:
            bridge->stats.uncertainties_routed++;
            break;
        case EPISTEMIC_SIGNAL_INQUIRY:
            bridge->stats.inquiries_routed++;
            break;
        case EPISTEMIC_SIGNAL_BELIEF_UPDATE:
            bridge->stats.belief_updates++;
            break;
        case EPISTEMIC_SIGNAL_CONFIDENCE:
            bridge->stats.confidence_assessments++;
            break;
        default:
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;
    }

    uint64_t total = bridge->stats.uncertainties_routed + bridge->stats.inquiries_routed +
                     bridge->stats.belief_updates + bridge->stats.confidence_assessments;
    if (total > 0) {
        bridge->stats.avg_uncertainty =
            (bridge->stats.avg_uncertainty * (total - 1) + signal->uncertainty_level) / total;
        bridge->stats.avg_information_gain =
            (bridge->stats.avg_information_gain * (total - 1) + signal->information_gain) / total;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int epistemic_thalamic_route_uncertainty(
    epistemic_thalamic_bridge_t* bridge,
    float uncertainty,
    float urgency
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    epistemic_thalamic_bridge_heartbeat("epistemic_th_epistemic_thalamic_r", 0.0f);


    epistemic_thalamic_signal_t signal = {
        .signal_type = EPISTEMIC_SIGNAL_UNCERTAINTY,
        .epistemic_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .uncertainty_level = uncertainty < 0.0f ? 0.0f : (uncertainty > 1.0f ? 1.0f : uncertainty),
        .confidence = 1.0f - uncertainty,
        .information_gain = 0.0f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return epistemic_thalamic_route_signal(bridge, &signal);
}

int epistemic_thalamic_route_inquiry(
    epistemic_thalamic_bridge_t* bridge,
    float expected_gain,
    float urgency
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    epistemic_thalamic_bridge_heartbeat("epistemic_th_epistemic_thalamic_r", 0.0f);


    epistemic_thalamic_signal_t signal = {
        .signal_type = EPISTEMIC_SIGNAL_INQUIRY,
        .epistemic_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .uncertainty_level = 0.5f,
        .confidence = 0.5f,
        .information_gain = expected_gain < 0.0f ? 0.0f : (expected_gain > 1.0f ? 1.0f : expected_gain),
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return epistemic_thalamic_route_signal(bridge, &signal);
}

int epistemic_thalamic_set_attention(epistemic_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    epistemic_thalamic_bridge_heartbeat("epistemic_th_epistemic_thalamic_s", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int epistemic_thalamic_get_attention(const epistemic_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    /* Phase 8: Heartbeat at operation start */
    epistemic_thalamic_bridge_heartbeat("epistemic_th_epistemic_thalamic_g", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int epistemic_thalamic_bridge_get_stats(
    const epistemic_thalamic_bridge_t* bridge,
    epistemic_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    /* Phase 8: Heartbeat at operation start */
    epistemic_thalamic_bridge_heartbeat("epistemic_th_get_stats", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int epistemic_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    epistemic_thalamic_bridge_heartbeat("epistemic_th_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Epistemic_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                epistemic_thalamic_bridge_heartbeat("epistemic_th_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Epistemic_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Epistemic_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void epistemic_thalamic_bridge_set_instance_health_agent(epistemic_thalamic_bridge_t* bridge, nimcp_health_agent_t* agent) {
    if (!bridge) {
        NIMCP_THROW(NIMCP_ERROR_NULL_POINTER,
                    "epistemic_thalamic_bridge_set_instance_health_agent: NULL bridge");
        return;
    }
    bridge->health_agent = agent;
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int epistemic_thalamic_bridge_training_begin(epistemic_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "epistemic_thalamic_bridge_training_begin: NULL argument");
        return -1;
    }
    epistemic_thalamic_bridge_heartbeat_instance(bridge->health_agent, "epistemic_thalamic_bridge_training_begin", 0.0f);
    return 0;
}

int epistemic_thalamic_bridge_training_end(epistemic_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "epistemic_thalamic_bridge_training_end: NULL argument");
        return -1;
    }
    epistemic_thalamic_bridge_heartbeat_instance(bridge->health_agent, "epistemic_thalamic_bridge_training_end", 1.0f);
    return 0;
}

int epistemic_thalamic_bridge_training_step(epistemic_thalamic_bridge_t* bridge, float progress) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "epistemic_thalamic_bridge_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    epistemic_thalamic_bridge_heartbeat_instance(bridge->health_agent, "epistemic_thalamic_bridge_training_step", progress);
    return 0;
}
