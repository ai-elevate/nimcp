/**
 * @file nimcp_mental_health_thalamic_bridge.c
 * @brief Mental Health-Thalamic Bridge Implementation
 */

#include "cognitive/mental_health/nimcp_mental_health_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct mental_health_thalamic_bridge {
    bridge_base_t base;
    void* mental_health;
    thalamic_router_t* router;
    mental_health_thalamic_config_t config;
    mental_health_thalamic_stats_t stats;
    float attention_weight;
};

mental_health_thalamic_config_t mental_health_thalamic_default_config(void) {
    mental_health_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_warning_priority = true,
        .min_wellbeing_threshold = 0.3f,
        .stress_alert_threshold = 0.7f
    };
    return cfg;
}

mental_health_thalamic_bridge_t* mental_health_thalamic_bridge_create(void* mental_health, thalamic_router_t* router, const mental_health_thalamic_config_t* config) {
    mental_health_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(mental_health_thalamic_bridge_t));
    if (!bridge) return NULL;
    if (bridge_base_init(&bridge->base, 0, "mental_health_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->mental_health = mental_health;
    bridge->router = router;
    bridge->config = config ? *config : mental_health_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void mental_health_thalamic_bridge_destroy(mental_health_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int mental_health_thalamic_bridge_reset(mental_health_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_thalamic_route_wellbeing(mental_health_thalamic_bridge_t* bridge, const mental_health_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.wellbeing_updates++;
    bridge->stats.avg_wellbeing_level = (bridge->stats.avg_wellbeing_level * (bridge->stats.wellbeing_updates - 1) +
                                         signal->wellbeing_level) / bridge->stats.wellbeing_updates;
    if (signal->stress_level >= bridge->config.stress_alert_threshold) {
        bridge->stats.stress_alerts++;
    }
    if (signal->signal_type == MENTAL_HEALTH_SIGNAL_WARNING) {
        bridge->stats.warnings_issued++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_thalamic_route_warning(mental_health_thalamic_bridge_t* bridge, const void* concern, float severity) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_warning_priority) {
        bridge->stats.warnings_issued++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_thalamic_set_attention(mental_health_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_thalamic_get_attention(const mental_health_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int mental_health_thalamic_bridge_get_stats(const mental_health_thalamic_bridge_t* bridge, mental_health_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int mental_health_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Mental_Health_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Mental_Health_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Mental_Health_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
