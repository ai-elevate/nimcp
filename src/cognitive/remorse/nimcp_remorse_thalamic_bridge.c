/**
 * @file nimcp_remorse_thalamic_bridge.c
 * @brief Remorse-Thalamic Bridge Implementation
 */

#include "cognitive/remorse/nimcp_remorse_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct remorse_thalamic_bridge {
    bridge_base_t base;
    void* remorse;
    thalamic_router_t* router;
    remorse_thalamic_config_t config;
    remorse_thalamic_stats_t stats;
    float attention_weight;
};

remorse_thalamic_config_t remorse_thalamic_default_config(void) {
    remorse_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_repair_routing = true,
        .min_guilt_threshold = 0.3f,
        .repair_threshold = 0.5f
    };
    return cfg;
}

remorse_thalamic_bridge_t* remorse_thalamic_bridge_create(void* remorse, thalamic_router_t* router, const remorse_thalamic_config_t* config) {
    remorse_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(remorse_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    if (bridge_base_init(&bridge->base, 0, "remorse_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->remorse = remorse;
    bridge->router = router;
    bridge->config = config ? *config : remorse_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void remorse_thalamic_bridge_destroy(remorse_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int remorse_thalamic_bridge_reset(remorse_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int remorse_thalamic_route_guilt(remorse_thalamic_bridge_t* bridge, const remorse_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->guilt_intensity < bridge->config.min_guilt_threshold) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.guilt_signals_routed++;
    bridge->stats.avg_guilt_intensity = (bridge->stats.avg_guilt_intensity * (bridge->stats.guilt_signals_routed - 1) +
                                         signal->guilt_intensity) / bridge->stats.guilt_signals_routed;
    if (signal->signal_type == REMORSE_SIGNAL_FORGIVENESS) {
        bridge->stats.forgiveness_achieved++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int remorse_thalamic_route_repair(remorse_thalamic_bridge_t* bridge, const void* action, float motivation) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_repair_routing && motivation >= bridge->config.repair_threshold) {
        bridge->stats.repair_actions++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int remorse_thalamic_set_attention(remorse_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int remorse_thalamic_get_attention(const remorse_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int remorse_thalamic_bridge_get_stats(const remorse_thalamic_bridge_t* bridge, remorse_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int remorse_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Remorse_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Remorse_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Remorse_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
