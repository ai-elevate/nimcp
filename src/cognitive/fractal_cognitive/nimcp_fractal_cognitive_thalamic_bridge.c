/**
 * @file nimcp_fractal_cognitive_thalamic_bridge.c
 * @brief Fractal Cognitive-Thalamic Bridge Implementation
 */

#include "cognitive/fractal_cognitive/nimcp_fractal_cognitive_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct fractal_cognitive_thalamic_bridge {
    bridge_base_t base;
    void* fractal_cognitive;
    thalamic_router_t* router;
    fractal_cognitive_thalamic_config_t config;
    fractal_cognitive_thalamic_stats_t stats;
    float attention_weight;
};

fractal_cognitive_thalamic_config_t fractal_cognitive_thalamic_default_config(void) {
    fractal_cognitive_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_scale_boost = true,
        .min_complexity_threshold = 0.25f,
        .resonance_boost = 0.35f
    };
    return cfg;
}

fractal_cognitive_thalamic_bridge_t* fractal_cognitive_thalamic_bridge_create(void* fractal_cognitive, thalamic_router_t* router, const fractal_cognitive_thalamic_config_t* config) {
    fractal_cognitive_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(fractal_cognitive_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->base.mutex = nimcp_mutex_create(NULL);
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->fractal_cognitive = fractal_cognitive;
    bridge->router = router;
    bridge->config = config ? *config : fractal_cognitive_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void fractal_cognitive_thalamic_bridge_destroy(fractal_cognitive_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) {
        nimcp_mutex_destroy(bridge->base.mutex);
    }
    nimcp_free(bridge);
}

int fractal_cognitive_thalamic_bridge_reset(fractal_cognitive_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fractal_cognitive_thalamic_route_scale(fractal_cognitive_thalamic_bridge_t* bridge, const fractal_cognitive_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->complexity < bridge->config.min_complexity_threshold) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.scales_routed++;
    bridge->stats.avg_complexity = (bridge->stats.avg_complexity * (bridge->stats.scales_routed - 1) +
                                    signal->complexity) / bridge->stats.scales_routed;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fractal_cognitive_thalamic_route_integration(fractal_cognitive_thalamic_bridge_t* bridge, const void* scales, uint32_t num_scales) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.integrations_performed++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fractal_cognitive_thalamic_set_attention(fractal_cognitive_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fractal_cognitive_thalamic_get_attention(const fractal_cognitive_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fractal_cognitive_thalamic_bridge_get_stats(const fractal_cognitive_thalamic_bridge_t* bridge, fractal_cognitive_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int fractal_cognitive_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Fractal_Cognitive_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Fractal_Cognitive_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Fractal_Cognitive_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
