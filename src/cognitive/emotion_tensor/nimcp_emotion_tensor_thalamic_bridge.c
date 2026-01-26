/**
 * @file nimcp_emotion_tensor_thalamic_bridge.c
 * @brief Emotion Tensor-Thalamic Bridge Implementation
 */

#include "cognitive/emotion_tensor/nimcp_emotion_tensor_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
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

/** Global health agent for emotion_tensor_thalamic_bridge module */
static nimcp_health_agent_t* g_emotion_tensor_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for emotion_tensor_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void emotion_tensor_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_emotion_tensor_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from emotion_tensor_thalamic_bridge module */
static inline void emotion_tensor_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_emotion_tensor_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_emotion_tensor_thalamic_bridge_health_agent, operation, progress);
    }
}


struct emotion_tensor_thalamic_bridge {
    bridge_base_t base;  /* MUST be first - provides mutex protection */
    void* emotion_tensor;
    thalamic_router_t* router;
    emotion_tensor_thalamic_config_t config;
    emotion_tensor_thalamic_stats_t stats;
    float attention_weight;
};

emotion_tensor_thalamic_config_t emotion_tensor_thalamic_default_config(void) {
    emotion_tensor_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_arousal_boost = true,
        .min_arousal_threshold = 0.25f,
        .contagion_boost = 0.35f
    };
    return cfg;
}

emotion_tensor_thalamic_bridge_t* emotion_tensor_thalamic_bridge_create(void* emotion_tensor, thalamic_router_t* router, const emotion_tensor_thalamic_config_t* config) {
    emotion_tensor_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(emotion_tensor_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    /* Initialize mutex for thread safety */
    if (bridge_base_init(&bridge->base, 0, "emotion_tensor_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->emotion_tensor = emotion_tensor;
    bridge->router = router;
    bridge->config = config ? *config : emotion_tensor_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void emotion_tensor_thalamic_bridge_destroy(emotion_tensor_thalamic_bridge_t* bridge) {
    if (bridge) {
        if (bridge->base.mutex) {
            bridge_base_cleanup(&bridge->base);
        }
        nimcp_free(bridge);
    }
}

int emotion_tensor_thalamic_bridge_reset(emotion_tensor_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int emotion_tensor_thalamic_route_update(emotion_tensor_thalamic_bridge_t* bridge, const emotion_tensor_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->arousal_level < bridge->config.min_arousal_threshold) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.updates_routed++;
    bridge->stats.avg_arousal_level = (bridge->stats.avg_arousal_level * (bridge->stats.updates_routed - 1) +
                                       signal->arousal_level) / bridge->stats.updates_routed;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int emotion_tensor_thalamic_route_blend(emotion_tensor_thalamic_bridge_t* bridge, const void* tensor, float blend_weight) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.blends_processed++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int emotion_tensor_thalamic_set_attention(emotion_tensor_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int emotion_tensor_thalamic_get_attention(const emotion_tensor_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int emotion_tensor_thalamic_bridge_get_stats(const emotion_tensor_thalamic_bridge_t* bridge, emotion_tensor_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int emotion_tensor_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Emotion_Tensor_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Emotion_Tensor_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Emotion_Tensor_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
