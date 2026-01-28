/**
 * @file nimcp_bias_thalamic_bridge.c
 * @brief Cognitive Bias-Thalamic Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/bias/nimcp_bias_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
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

/** Global health agent for bias_thalamic_bridge module */
static nimcp_health_agent_t* g_bias_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for bias_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void bias_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_bias_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from bias_thalamic_bridge module */
static inline void bias_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_bias_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_bias_thalamic_bridge_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat from bias_thalamic_bridge module (instance-level) */
static inline void bias_thalamic_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_bias_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_bias_thalamic_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_bias_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


#define LOG_MODULE "BIAS_THALAMIC_BRIDGE"


struct bias_thalamic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    nimcp_health_agent_t* health_agent;  /**< Phase 8: instance-level health agent */
    void* bias;
    thalamic_router_t* router;
    bias_thalamic_config_t config;
    bias_thalamic_stats_t stats;
    float attention_weight;
};

bias_thalamic_config_t bias_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    bias_thalamic_bridge_heartbeat("bias_thalami_bias_thalamic_defaul", 0.0f);


    bias_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_automatic_correction = true,
        .min_detection_confidence = 0.3f,
        .correction_threshold = 0.5f
    };
    return cfg;
}

bias_thalamic_bridge_t* bias_thalamic_bridge_create(void* bias, thalamic_router_t* router, const bias_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    bias_thalamic_bridge_heartbeat("bias_thalami_create", 0.0f);


    bias_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(bias_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate bridge");

        return NULL;

    }
    bridge->bias = bias;
    bridge->router = router;
    bridge->config = config ? *config : bias_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    NIMCP_LOGGING_INFO("Created %s bridge", "bias_thalamic");
    return bridge;
}

void bias_thalamic_bridge_destroy(bias_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    bias_thalamic_bridge_heartbeat("bias_thalami_destroy", 0.0f);


    if (bridge) nimcp_free(bridge);
}

int bias_thalamic_bridge_reset(bias_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    bias_thalamic_bridge_heartbeat("bias_thalami_reset", 0.0f);


    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int bias_thalamic_route_detection(bias_thalamic_bridge_t* bridge, const bias_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* Phase 8: Heartbeat at operation start */
    bias_thalamic_bridge_heartbeat("bias_thalami_bias_thalamic_route_", 0.0f);


    if (bridge->config.enable_attention_gating && signal->detection_confidence < bridge->config.min_detection_confidence) {
        return 0;
    }
    bridge->stats.biases_detected++;
    bridge->stats.avg_detection_confidence = (bridge->stats.avg_detection_confidence * (bridge->stats.biases_detected - 1) +
                                              signal->detection_confidence) / bridge->stats.biases_detected;
    if (signal->signal_type == BIAS_SIGNAL_OVERRIDE) {
        bridge->stats.overrides_successful++;
    }
    return 0;
}

int bias_thalamic_route_correction(bias_thalamic_bridge_t* bridge, const void* correction, float strength) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    bias_thalamic_bridge_heartbeat("bias_thalami_bias_thalamic_route_", 0.0f);


    if (bridge->config.enable_automatic_correction && strength >= bridge->config.correction_threshold) {
        bridge->stats.corrections_applied++;
    }
    return 0;
}

int bias_thalamic_set_attention(bias_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    bias_thalamic_bridge_heartbeat("bias_thalami_bias_thalamic_set_at", 0.0f);


    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int bias_thalamic_get_attention(const bias_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    /* Phase 8: Heartbeat at operation start */
    bias_thalamic_bridge_heartbeat("bias_thalami_bias_thalamic_get_at", 0.0f);


    return 0;
}

int bias_thalamic_bridge_get_stats(const bias_thalamic_bridge_t* bridge, bias_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    /* Phase 8: Heartbeat at operation start */
    bias_thalamic_bridge_heartbeat("bias_thalami_get_stats", 0.0f);


    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int bias_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    bias_thalamic_bridge_heartbeat("bias_thalami_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Bias_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                bias_thalamic_bridge_heartbeat("bias_thalami_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Bias_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Bias_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void bias_thalamic_bridge_set_instance_health_agent(bias_thalamic_bridge_t* bridge, nimcp_health_agent_t* agent) {
    if (!bridge) {
        NIMCP_THROW(NIMCP_ERROR_NULL_POINTER,
                    "bias_thalamic_bridge_set_instance_health_agent: NULL bridge");
        return;
    }
    bridge->health_agent = agent;
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int bias_thalamic_bridge_training_begin(bias_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "bias_thalamic_bridge_training_begin: NULL argument");
        return -1;
    }
    bias_thalamic_bridge_heartbeat_instance(bridge->health_agent, "bias_thalamic_bridge_training_begin", 0.0f);
    return 0;
}

int bias_thalamic_bridge_training_end(bias_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "bias_thalamic_bridge_training_end: NULL argument");
        return -1;
    }
    bias_thalamic_bridge_heartbeat_instance(bridge->health_agent, "bias_thalamic_bridge_training_end", 1.0f);
    return 0;
}

int bias_thalamic_bridge_training_step(bias_thalamic_bridge_t* bridge, float progress) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "bias_thalamic_bridge_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    bias_thalamic_bridge_heartbeat_instance(bridge->health_agent, "bias_thalamic_bridge_training_step", progress);
    return 0;
}
