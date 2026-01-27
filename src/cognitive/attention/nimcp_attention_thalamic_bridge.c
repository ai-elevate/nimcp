/**
 * @file nimcp_attention_thalamic_bridge.c
 * @brief Attention-Thalamic Bridge Implementation
 *
 * Routes attention signals through thalamic reticular nucleus
 * for focus establishment, shifting, and filtering.
 */

#include "cognitive/attention/nimcp_attention_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "security/nimcp_bbb_helpers.h"
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

/** Global health agent for attention_thalamic_bridge module */
static nimcp_health_agent_t* g_attention_thalamic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for attention_thalamic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void attention_thalamic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_attention_thalamic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from attention_thalamic_bridge module */
static inline void attention_thalamic_bridge_heartbeat(const char* operation, float progress) {
    if (g_attention_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_attention_thalamic_bridge_health_agent, operation, progress);
    }
}


struct attention_thalamic_bridge {
    bridge_base_t base;  /* MUST be first - provides mutex protection */
    void* attention;
    thalamic_router_t* router;
    attention_thalamic_config_t config;
    attention_thalamic_stats_t stats;
    float attention_weight;
    float accumulated_shift_cost;
};

BRIDGE_DEFINE_SECURITY_SETTERS(attention_thalamic_bridge)

attention_thalamic_config_t attention_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_attention_thalamic_d", 0.0f);


    return (attention_thalamic_config_t){
        .enable_priority_gating = true,
        .enable_shift_cost = true,
        .enable_vigilance_boost = true,
        .min_priority_threshold = 0.25f,
        .shift_cost_penalty = 0.1f
    };
}

attention_thalamic_bridge_t* attention_thalamic_bridge_create(
    void* attention,
    thalamic_router_t* router,
    const attention_thalamic_config_t* config
) {
    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_create", 0.0f);


    attention_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(attention_thalamic_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "attention_thalamic_bridge_create: failed to allocate bridge");
        return NULL;
    }

    /* Initialize mutex for thread safety */
    if (bridge_base_init(&bridge->base, 0, "attention_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "attention_thalamic_bridge_create: failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->attention = attention;
    bridge->router = router;
    bridge->config = config ? *config : attention_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    bridge->accumulated_shift_cost = 0.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    return bridge;
}

void attention_thalamic_bridge_destroy(attention_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_destroy", 0.0f);


    if (bridge) {
        if (bridge->base.mutex) {
            bridge_base_cleanup(&bridge->base);
        }
        nimcp_free(bridge);
    }
}

int attention_thalamic_bridge_reset(attention_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_thalamic_bridge_reset: bridge is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    bridge->accumulated_shift_cost = 0.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int attention_thalamic_route_signal(
    attention_thalamic_bridge_t* bridge,
    const attention_thalamic_signal_t* signal
) {
    if (!bridge || !signal) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_thalamic_route_signal: bridge or signal is NULL");
        return -1;
    }

    if (signal->content && signal->content_size > 0) {
        BRIDGE_BBB_VALIDATE(bridge, signal->content, signal->content_size);
    }

    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_attention_thalamic_r", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Apply priority gating */
    if (bridge->config.enable_priority_gating) {
        float effective_priority = signal->attention_priority * bridge->attention_weight;

        /* Apply shift cost penalty if enabled */
        if (bridge->config.enable_shift_cost &&
            signal->signal_type == ATTENTION_SIGNAL_SHIFT) {
            effective_priority -= bridge->accumulated_shift_cost;
            if (effective_priority < 0.0f) effective_priority = 0.0f;
        }

        /* Boost vigilance signals */
        if (bridge->config.enable_vigilance_boost &&
            signal->signal_type == ATTENTION_SIGNAL_VIGILANCE) {
            effective_priority *= 1.2f;
            if (effective_priority > 1.0f) effective_priority = 1.0f;
        }

        if (effective_priority < bridge->config.min_priority_threshold) {
            bridge->stats.signals_gated++;
            nimcp_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    /* Route based on signal type */
    switch (signal->signal_type) {
        case ATTENTION_SIGNAL_FOCUS:
            bridge->stats.focus_requests++;
            bridge->accumulated_shift_cost = 0.0f; /* Reset on focus */
            break;

        case ATTENTION_SIGNAL_SHIFT:
            bridge->stats.shifts_executed++;
            /* Accumulate shift cost */
            bridge->accumulated_shift_cost += signal->shift_cost * bridge->config.shift_cost_penalty;
            if (bridge->accumulated_shift_cost > 0.5f) {
                bridge->accumulated_shift_cost = 0.5f;
            }
            /* Update average shift cost */
            bridge->stats.avg_shift_cost =
                (bridge->stats.avg_shift_cost * (bridge->stats.shifts_executed - 1) +
                 signal->shift_cost) / bridge->stats.shifts_executed;
            break;

        case ATTENTION_SIGNAL_FILTER:
            bridge->stats.filter_activations++;
            break;

        case ATTENTION_SIGNAL_RELEASE:
            bridge->stats.releases++;
            bridge->accumulated_shift_cost = 0.0f;
            break;

        case ATTENTION_SIGNAL_VIGILANCE:
            bridge->stats.vigilance_updates++;
            break;

        default:
            nimcp_mutex_unlock(bridge->base.mutex);
            return -1;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int attention_thalamic_request_focus(
    attention_thalamic_bridge_t* bridge,
    float priority,
    float target_salience
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_thalamic_request_focus: bridge is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_attention_thalamic_r", 0.0f);


    attention_thalamic_signal_t signal = {
        .signal_type = ATTENTION_SIGNAL_FOCUS,
        .attention_priority = priority < 0.0f ? 0.0f : (priority > 1.0f ? 1.0f : priority),
        .target_salience = target_salience,
        .shift_cost = 0.0f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return attention_thalamic_route_signal(bridge, &signal);
}

int attention_thalamic_request_shift(
    attention_thalamic_bridge_t* bridge,
    float new_priority,
    float shift_cost
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_thalamic_request_shift: bridge is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_attention_thalamic_r", 0.0f);


    attention_thalamic_signal_t signal = {
        .signal_type = ATTENTION_SIGNAL_SHIFT,
        .attention_priority = new_priority < 0.0f ? 0.0f : (new_priority > 1.0f ? 1.0f : new_priority),
        .target_salience = 0.5f,
        .shift_cost = shift_cost < 0.0f ? 0.0f : (shift_cost > 1.0f ? 1.0f : shift_cost),
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return attention_thalamic_route_signal(bridge, &signal);
}

int attention_thalamic_activate_filter(
    attention_thalamic_bridge_t* bridge,
    float filter_strength
) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_thalamic_activate_filter: bridge is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_attention_thalamic_a", 0.0f);


    attention_thalamic_signal_t signal = {
        .signal_type = ATTENTION_SIGNAL_FILTER,
        .attention_priority = filter_strength,
        .target_salience = 0.0f,
        .shift_cost = 0.0f,
        .content = NULL,
        .content_size = 0,
        .timestamp_us = nimcp_time_get_us()
    };

    return attention_thalamic_route_signal(bridge, &signal);
}

int attention_thalamic_set_attention(attention_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_thalamic_set_attention: bridge is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_attention_thalamic_s", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f :
                               (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int attention_thalamic_get_attention(const attention_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_thalamic_get_attention: bridge or attention is NULL");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_attention_thalamic_g", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int attention_thalamic_bridge_get_stats(
    const attention_thalamic_bridge_t* bridge,
    attention_thalamic_stats_t* stats
) {
    if (!bridge || !stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "attention_thalamic_bridge_get_stats: bridge or stats is NULL");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_get_stats", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

void attention_thalamic_bridge_reset_stats(attention_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_reset_stats", 0.0f);


    if (bridge) {
        nimcp_mutex_lock(bridge->base.mutex);
        memset(&bridge->stats, 0, sizeof(bridge->stats));
        nimcp_mutex_unlock(bridge->base.mutex);
    }
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Attention Thalamic Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int attention_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    attention_thalamic_bridge_heartbeat("attention_th_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Attention_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                attention_thalamic_bridge_heartbeat("attention_th_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Attention Thalamic Bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Attention_Thalamic_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Attention_Thalamic_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
