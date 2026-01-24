/**
 * @file nimcp_sleep_wake_thalamic_bridge.c
 * @brief Sleep-Wake-Thalamic Bridge Implementation
 */

#include "cognitive/sleep_wake/nimcp_sleep_wake_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct sleep_wake_thalamic_bridge {
    bridge_base_t base;
    void* sleep_wake;
    thalamic_router_t* router;
    sleep_wake_thalamic_config_t config;
    sleep_wake_thalamic_stats_t stats;
    float attention_weight;
};

sleep_wake_thalamic_config_t sleep_wake_thalamic_default_config(void) {
    sleep_wake_thalamic_config_t cfg = {
        .enable_arousal_modulation = true,
        .enable_transition_gating = true,
        .min_arousal_threshold = 0.3f,
        .transition_threshold = 0.5f
    };
    return cfg;
}

sleep_wake_thalamic_bridge_t* sleep_wake_thalamic_bridge_create(void* sleep_wake, thalamic_router_t* router, const sleep_wake_thalamic_config_t* config) {
    sleep_wake_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(sleep_wake_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    if (bridge_base_init(&bridge->base, 0, "sleep_wake_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->sleep_wake = sleep_wake;
    bridge->router = router;
    bridge->config = config ? *config : sleep_wake_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void sleep_wake_thalamic_bridge_destroy(sleep_wake_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int sleep_wake_thalamic_bridge_reset(sleep_wake_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int sleep_wake_thalamic_route_arousal(sleep_wake_thalamic_bridge_t* bridge, const sleep_wake_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.arousal_updates++;
    bridge->stats.avg_arousal_level = (bridge->stats.avg_arousal_level * (bridge->stats.arousal_updates - 1) +
                                       signal->arousal_level) / bridge->stats.arousal_updates;
    if (signal->signal_type == SLEEP_WAKE_SIGNAL_TRANSITION) {
        bridge->stats.state_transitions++;
    }
    if (signal->signal_type == SLEEP_WAKE_SIGNAL_CIRCADIAN) {
        bridge->stats.circadian_updates++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int sleep_wake_thalamic_modulate_gating(sleep_wake_thalamic_bridge_t* bridge, float arousal_level) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    /* Modulate attention based on arousal level */
    if (bridge->config.enable_arousal_modulation) {
        bridge->attention_weight = arousal_level < 0.0f ? 0.0f : (arousal_level > 1.0f ? 1.0f : arousal_level);
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int sleep_wake_thalamic_set_attention(sleep_wake_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int sleep_wake_thalamic_get_attention(const sleep_wake_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int sleep_wake_thalamic_bridge_get_stats(const sleep_wake_thalamic_bridge_t* bridge, sleep_wake_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int sleep_wake_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Sleep_Wake_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Sleep_Wake_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Sleep_Wake_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
