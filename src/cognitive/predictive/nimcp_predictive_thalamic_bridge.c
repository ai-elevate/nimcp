/**
 * @file nimcp_predictive_thalamic_bridge.c
 * @brief Predictive Coding-Thalamic Bridge Implementation
 */

#include "cognitive/predictive/nimcp_predictive_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct predictive_thalamic_bridge {
    bridge_base_t base;
    void* predictive;
    thalamic_router_t* router;
    predictive_thalamic_config_t config;
    predictive_thalamic_stats_t stats;
    float attention_weight;
};

predictive_thalamic_config_t predictive_thalamic_default_config(void) {
    predictive_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_error_amplification = true,
        .min_error_threshold = 0.1f,
        .precision_threshold = 0.5f
    };
    return cfg;
}

predictive_thalamic_bridge_t* predictive_thalamic_bridge_create(void* predictive, thalamic_router_t* router, const predictive_thalamic_config_t* config) {
    predictive_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(predictive_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->predictive = predictive;
    bridge->router = router;
    bridge->config = config ? *config : predictive_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void predictive_thalamic_bridge_destroy(predictive_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) {
        nimcp_mutex_free(bridge->base.mutex);
    }
    nimcp_free(bridge);
}

int predictive_thalamic_bridge_reset(predictive_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_thalamic_route_error(predictive_thalamic_bridge_t* bridge, const predictive_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->error_magnitude < bridge->config.min_error_threshold) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.errors_routed++;
    bridge->stats.avg_error_magnitude = (bridge->stats.avg_error_magnitude * (bridge->stats.errors_routed - 1) +
                                         signal->error_magnitude) / bridge->stats.errors_routed;
    if (signal->signal_type == PREDICTIVE_SIGNAL_PREDICTION) {
        bridge->stats.predictions_routed++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_thalamic_route_update(predictive_thalamic_bridge_t* bridge, const void* update, uint32_t level) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.updates_triggered++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_thalamic_set_attention(predictive_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_thalamic_get_attention(const predictive_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int predictive_thalamic_bridge_get_stats(const predictive_thalamic_bridge_t* bridge, predictive_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int predictive_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Predictive_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Predictive_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Predictive_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
