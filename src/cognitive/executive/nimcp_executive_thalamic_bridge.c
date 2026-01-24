/**
 * @file nimcp_executive_thalamic_bridge.c
 * @brief Executive-Thalamic Bridge Implementation
 */

#include "cognitive/executive/nimcp_executive_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct executive_thalamic_bridge {
    bridge_base_t base;  /* MUST be first - provides mutex protection */
    void* executive;
    thalamic_router_t* router;
    executive_thalamic_config_t config;
    executive_thalamic_stats_t stats;
    float attention_weight;
    float accumulated_switch_cost;
};

executive_thalamic_config_t executive_thalamic_default_config(void) {
    return (executive_thalamic_config_t){
        .enable_attention_gating = true,
        .enable_load_routing = true,
        .enable_inhibition_priority = true,
        .min_urgency_threshold = 0.2f,
        .inhibition_boost = 1.4f,
        .switch_penalty = 0.15f
    };
}

executive_thalamic_bridge_t* executive_thalamic_bridge_create(
    void* executive,
    thalamic_router_t* router,
    const executive_thalamic_config_t* config
) {
    executive_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(executive_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    /* Initialize mutex for thread safety */
    if (bridge_base_init(&bridge->base, 0, "executive_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->executive = executive;
    bridge->router = router;
    bridge->config = config ? *config : executive_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    bridge->accumulated_switch_cost = 0.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void executive_thalamic_bridge_destroy(executive_thalamic_bridge_t* bridge) {
    if (bridge) {
        if (bridge->base.mutex) {
            bridge_base_cleanup(&bridge->base);
        }
        nimcp_free(bridge);
    }
}

int executive_thalamic_bridge_reset(executive_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    bridge->accumulated_switch_cost = 0.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int executive_thalamic_route_signal(
    executive_thalamic_bridge_t* bridge,
    const executive_thalamic_signal_t* signal
) {
    if (!bridge || !signal) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->config.enable_attention_gating) {
        float effective_urgency = signal->control_urgency * bridge->attention_weight;

        /* Inhibition gets priority boost */
        if (bridge->config.enable_inhibition_priority &&
            signal->signal_type == EXECUTIVE_SIGNAL_INHIBITION) {
            effective_urgency *= bridge->config.inhibition_boost;
            if (effective_urgency > 1.0f) effective_urgency = 1.0f;
        }

        /* Switching penalized by accumulated cost */
        if (signal->signal_type == EXECUTIVE_SIGNAL_SWITCHING) {
            effective_urgency -= bridge->accumulated_switch_cost;
            if (effective_urgency < 0.0f) effective_urgency = 0.0f;
        }

        if (effective_urgency < bridge->config.min_urgency_threshold) {
            bridge->stats.signals_gated++;
            nimcp_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    switch (signal->signal_type) {
        case EXECUTIVE_SIGNAL_INHIBITION:
            bridge->stats.inhibitions_routed++;
            break;
        case EXECUTIVE_SIGNAL_SWITCHING:
            bridge->stats.switches_executed++;
            bridge->accumulated_switch_cost += signal->switch_cost * bridge->config.switch_penalty;
            if (bridge->accumulated_switch_cost > 0.5f) bridge->accumulated_switch_cost = 0.5f;
            /* Update average switch cost */
            bridge->stats.avg_switch_cost =
                (bridge->stats.avg_switch_cost * (bridge->stats.switches_executed - 1) +
                 signal->switch_cost) / bridge->stats.switches_executed;
            break;
        case EXECUTIVE_SIGNAL_PLANNING:
            bridge->stats.plans_routed++;
            break;
        case EXECUTIVE_SIGNAL_MONITORING:
            bridge->stats.monitors_updated++;
            break;
        case EXECUTIVE_SIGNAL_DECISION:
            bridge->stats.decisions_routed++;
            bridge->accumulated_switch_cost *= 0.8f; /* Decay on decision */
            break;
        default:
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;
    }

    uint64_t total = bridge->stats.inhibitions_routed + bridge->stats.switches_executed +
                     bridge->stats.plans_routed + bridge->stats.monitors_updated +
                     bridge->stats.decisions_routed;
    if (total > 0) {
        bridge->stats.avg_cognitive_load =
            (bridge->stats.avg_cognitive_load * (total - 1) + signal->cognitive_load) / total;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int executive_thalamic_route_inhibition(
    executive_thalamic_bridge_t* bridge,
    float strength,
    float urgency
) {
    if (!bridge) return -1;

    executive_thalamic_signal_t signal = {
        .signal_type = EXECUTIVE_SIGNAL_INHIBITION,
        .control_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .cognitive_load = 0.5f,
        .inhibition_strength = strength < 0.0f ? 0.0f : (strength > 1.0f ? 1.0f : strength),
        .switch_cost = 0.0f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return executive_thalamic_route_signal(bridge, &signal);
}

int executive_thalamic_route_switch(
    executive_thalamic_bridge_t* bridge,
    float cost,
    float urgency
) {
    if (!bridge) return -1;

    executive_thalamic_signal_t signal = {
        .signal_type = EXECUTIVE_SIGNAL_SWITCHING,
        .control_urgency = urgency < 0.0f ? 0.0f : (urgency > 1.0f ? 1.0f : urgency),
        .cognitive_load = 0.7f,
        .inhibition_strength = 0.0f,
        .switch_cost = cost < 0.0f ? 0.0f : (cost > 1.0f ? 1.0f : cost),
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return executive_thalamic_route_signal(bridge, &signal);
}

int executive_thalamic_set_attention(executive_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int executive_thalamic_get_attention(const executive_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int executive_thalamic_bridge_get_stats(
    const executive_thalamic_bridge_t* bridge,
    executive_thalamic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Executive Thalamic Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int executive_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Executive_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("Executive Thalamic Bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Executive_Thalamic_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Executive_Thalamic_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
