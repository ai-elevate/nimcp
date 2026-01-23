/**
 * @file nimcp_empathetic_response_thalamic_bridge.c
 * @brief Empathetic Response-Thalamic Bridge Implementation
 */

#include "cognitive/empathetic_response/nimcp_empathetic_response_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct empathetic_response_thalamic_bridge {
    bridge_base_t base;  /* MUST be first - provides mutex protection */
    void* empathetic_response;
    thalamic_router_t* router;
    empathetic_response_thalamic_config_t config;
    empathetic_response_thalamic_stats_t stats;
    float attention_weight;
};

empathetic_response_thalamic_config_t empathetic_response_thalamic_default_config(void) {
    empathetic_response_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_distress_boost = true,
        .min_distress_threshold = 0.2f,
        .mirror_boost = 0.4f
    };
    return cfg;
}

empathetic_response_thalamic_bridge_t* empathetic_response_thalamic_bridge_create(void* empathetic_response, thalamic_router_t* router, const empathetic_response_thalamic_config_t* config) {
    empathetic_response_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(empathetic_response_thalamic_bridge_t));
    if (!bridge) return NULL;

    /* Initialize mutex for thread safety */
    if (bridge_base_init(&bridge->base, 0, "empathetic_response_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->empathetic_response = empathetic_response;
    bridge->router = router;
    bridge->config = config ? *config : empathetic_response_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void empathetic_response_thalamic_bridge_destroy(empathetic_response_thalamic_bridge_t* bridge) {
    if (bridge) {
        if (bridge->base.mutex) {
            bridge_base_cleanup(&bridge->base);
        }
        nimcp_free(bridge);
    }
}

int empathetic_response_thalamic_bridge_reset(empathetic_response_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int empathetic_response_thalamic_route_recognition(empathetic_response_thalamic_bridge_t* bridge, const empathetic_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->distress_level < bridge->config.min_distress_threshold) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.recognitions_routed++;
    bridge->stats.avg_distress_level = (bridge->stats.avg_distress_level * (bridge->stats.recognitions_routed - 1) +
                                        signal->distress_level) / bridge->stats.recognitions_routed;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int empathetic_response_thalamic_route_response(empathetic_response_thalamic_bridge_t* bridge, const void* response, float intensity) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.responses_generated++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int empathetic_response_thalamic_set_attention(empathetic_response_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int empathetic_response_thalamic_get_attention(const empathetic_response_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int empathetic_response_thalamic_bridge_get_stats(const empathetic_response_thalamic_bridge_t* bridge, empathetic_response_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int empathetic_response_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Empathetic_Response_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Empathetic_Response_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Empathetic_Response_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
