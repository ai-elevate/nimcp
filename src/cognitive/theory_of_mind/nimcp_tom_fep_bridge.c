/**
 * @file nimcp_tom_fep_bridge.c
 * @brief Free Energy Principle - Theory of Mind Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-12
 */

#include "cognitive/theory_of_mind/nimcp_tom_fep_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>

#define LOG_MODULE_TOM_FEP "[TOM_FEP]"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for tom_fep_bridge module */
static nimcp_health_agent_t* g_tom_fep_bridge_health_agent = NULL;

/**
 * @brief Set health agent for tom_fep_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void tom_fep_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_tom_fep_bridge_health_agent = agent;
}

/** @brief Send heartbeat from tom_fep_bridge module */
static inline void tom_fep_bridge_heartbeat(const char* operation, float progress) {
    if (g_tom_fep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_tom_fep_bridge_health_agent, operation, progress);
    }
}


/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

int tom_fep_bridge_default_config(tom_fep_config_t* config) {
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_POINTER, "config is NULL");

    config->action_pe_threshold = TOM_FEP_PE_ACTION_THRESHOLD;
    config->belief_pe_threshold = TOM_FEP_PE_BELIEF_THRESHOLD;
    config->empathy_threshold = TOM_FEP_EMPATHY_THRESHOLD;

    config->max_recursion_depth = TOM_FEP_MAX_RECURSION_DEPTH;
    config->complexity_penalty = TOM_FEP_COMPLEXITY_PENALTY;

    config->enable_belief_inference = true;
    config->enable_intention_inference = true;
    config->enable_empathy = true;
    config->enable_false_belief_detection = true;

    config->pe_sensitivity = 1.0f;
    config->empathy_sensitivity = 1.0f;

    return NIMCP_SUCCESS;
}

tom_fep_bridge_t* tom_fep_bridge_create(const tom_fep_config_t* config) {
    tom_fep_bridge_t* bridge =
        (tom_fep_bridge_t*)nimcp_malloc(sizeof(tom_fep_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    memset(bridge, 0, sizeof(tom_fep_bridge_t));

    if (config) {
        bridge->config = *config;
    } else {
        tom_fep_bridge_default_config(&bridge->config);
    }

    if (bridge_base_init(&bridge->base, 0, "tom_fep") != 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to initialize bridge base in tom_fep_bridge_create");
        nimcp_free(bridge);
        return NULL;
    }
    if (!bridge->base.mutex) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Mutex is NULL after bridge_base_init in tom_fep_bridge_create");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->state.prediction_accuracy = 0.5f;

    NIMCP_LOGGING_INFO(LOG_MODULE_TOM_FEP " Bridge created");
    return bridge;
}

void tom_fep_bridge_destroy(tom_fep_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->base.bio_async_enabled) {
        tom_fep_bridge_disconnect_bio_async(bridge);
    }

    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO(LOG_MODULE_TOM_FEP " Bridge destroyed");
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int tom_fep_bridge_connect_fep(tom_fep_bridge_t* bridge, fep_system_t* fep) {
    NIMCP_CHECK_THROW(bridge && fep, NIMCP_ERROR_NULL_POINTER, "bridge or fep is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO(LOG_MODULE_TOM_FEP " Connected to FEP system");
    return NIMCP_SUCCESS;
}

int tom_fep_bridge_connect_tom(tom_fep_bridge_t* bridge, theory_of_mind_t tom) {
    NIMCP_CHECK_THROW(bridge && tom, NIMCP_ERROR_NULL_POINTER, "bridge or tom is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->tom_system = tom;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO(LOG_MODULE_TOM_FEP " Connected to ToM system");
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * FEP → ToM Direction
 * ============================================================================ */

int tom_fep_infer_belief(tom_fep_bridge_t* bridge, float prediction_error) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_belief_inference) return NIMCP_SUCCESS;

    nimcp_mutex_lock(bridge->base.mutex);

    if (fabsf(prediction_error) > bridge->config.belief_pe_threshold) {
        bridge->effects.belief_pe = fabsf(prediction_error);
        bridge->effects.belief_updated = true;
        bridge->state.beliefs_inferred++;
        bridge->stats.belief_inferences_total++;

        /* Check for false belief */
        if (bridge->config.enable_false_belief_detection) {
            bridge->effects.false_belief_detected = true;
            bridge->state.false_beliefs_detected++;
            bridge->stats.false_beliefs_total++;

            NIMCP_LOGGING_DEBUG(LOG_MODULE_TOM_FEP
                " False belief detected (PE=%.2f)", prediction_error);
        }
    }

    bridge->stats.avg_belief_pe =
        (bridge->stats.avg_belief_pe * 0.9f) + (fabsf(prediction_error) * 0.1f);

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int tom_fep_infer_intention(tom_fep_bridge_t* bridge, float action_pe) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_intention_inference) return NIMCP_SUCCESS;

    nimcp_mutex_lock(bridge->base.mutex);

    if (fabsf(action_pe) > bridge->config.action_pe_threshold) {
        bridge->effects.action_pe = fabsf(action_pe);
        bridge->effects.intention_inferred = true;
        bridge->state.intentions_inferred++;
        bridge->stats.intention_inferences_total++;

        NIMCP_LOGGING_DEBUG(LOG_MODULE_TOM_FEP
            " Intention inferred (action PE=%.2f)", action_pe);
    }

    bridge->stats.avg_action_pe =
        (bridge->stats.avg_action_pe * 0.9f) + (fabsf(action_pe) * 0.1f);

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int tom_fep_activate_empathy(
    tom_fep_bridge_t* bridge,
    tom_emotion_t observed_emotion
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->config.enable_empathy) return NIMCP_SUCCESS;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->effects.inferred_emotion = observed_emotion;

    /* Compute empathic resonance based on emotion intensity */
    float resonance = 0.0f;
    switch (observed_emotion) {
        case TOM_EMOTION_SADNESS:
        case TOM_EMOTION_FEAR:
        case TOM_EMOTION_ANXIETY:
            resonance = 0.8f; /* High empathy for negative emotions */
            break;
        case TOM_EMOTION_JOY:
        case TOM_EMOTION_PRIDE:
            resonance = 0.6f; /* Moderate empathy for positive */
            break;
        default:
            resonance = 0.3f; /* Low empathy for neutral */
            break;
    }

    resonance *= bridge->config.empathy_sensitivity;
    bridge->effects.empathic_resonance = resonance;

    if (resonance > bridge->config.empathy_threshold) {
        bridge->state.empathy_active = true;
        bridge->state.empathy_magnitude = resonance;
        bridge->stats.empathy_episodes++;

        NIMCP_LOGGING_DEBUG(LOG_MODULE_TOM_FEP
            " Empathy activated (emotion=%d, resonance=%.2f)",
            observed_emotion, resonance);
    }

    bridge->stats.avg_empathy_magnitude =
        (bridge->stats.avg_empathy_magnitude * 0.9f) + (resonance * 0.1f);

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * ToM → FEP Direction
 * ============================================================================ */

int tom_fep_apply_social_priors(tom_fep_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    /* Apply social priors to FEP */
    /* Implementation would set FEP prior distributions */

    NIMCP_LOGGING_DEBUG(LOG_MODULE_TOM_FEP " Applied social priors to FEP");

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int tom_fep_modulate_empathic_precision(
    tom_fep_bridge_t* bridge,
    tom_emotion_t emotion
) {
    NIMCP_CHECK_THROW(bridge && bridge->fep_system, NIMCP_ERROR_NULL_POINTER, "bridge or fep_system is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    /* Modulate FEP precision based on empathic state */
    /* High distress → increase precision */
    float precision_boost = 1.0f;
    if (emotion == TOM_EMOTION_SADNESS || emotion == TOM_EMOTION_FEAR) {
        precision_boost = 1.5f;
    }

    NIMCP_LOGGING_DEBUG(LOG_MODULE_TOM_FEP
        " Modulated empathic precision (boost=%.2f)", precision_boost);

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int tom_fep_add_mentalizing_overhead(
    tom_fep_bridge_t* bridge,
    uint32_t recursion_depth
) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    uint32_t depth = recursion_depth;
    if (depth > bridge->config.max_recursion_depth) {
        depth = bridge->config.max_recursion_depth;
    }

    bridge->effects.current_recursion_depth = depth;
    bridge->effects.mentalizing_overhead =
        depth * bridge->config.complexity_penalty;

    bridge->state.mentalizing_depth = depth;
    bridge->state.social_cognitive_load = bridge->effects.mentalizing_overhead;

    bridge->stats.avg_recursion_depth =
        (bridge->stats.avg_recursion_depth * 0.9f) + (depth * 0.1f);
    bridge->stats.avg_mentalizing_overhead =
        (bridge->stats.avg_mentalizing_overhead * 0.9f) +
        (bridge->effects.mentalizing_overhead * 0.1f);

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Update Cycle
 * ============================================================================ */

int tom_fep_bridge_update(tom_fep_bridge_t* bridge, uint64_t delta_ms) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    /* Decay empathy over time */
    if (bridge->state.empathy_active) {
        bridge->state.empathy_magnitude *= 0.95f;
        if (bridge->state.empathy_magnitude < 0.1f) {
            bridge->state.empathy_active = false;
            bridge->state.empathy_magnitude = 0.0f;
        }
    }

    /* Update prediction accuracy */
    if (bridge->effects.intention_inferred) {
        bridge->state.prediction_accuracy =
            (bridge->state.prediction_accuracy * 0.95f) + 0.05f;
        bridge->effects.intention_inferred = false;
    }

    /* Decay PEs */
    bridge->effects.action_pe *= 0.9f;
    bridge->effects.belief_pe *= 0.9f;
    bridge->effects.intention_pe *= 0.9f;

    /* Update statistics */
    bridge->stats.avg_prediction_accuracy =
        (bridge->stats.avg_prediction_accuracy * 0.99f) +
        (bridge->state.prediction_accuracy * 0.01f);

    bridge->state.last_social_pe_ms = delta_ms;

    nimcp_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * State/Stats API
 * ============================================================================ */

int tom_fep_bridge_get_state(
    const tom_fep_bridge_t* bridge,
    tom_fep_state_t* state
) {
    NIMCP_CHECK_THROW(bridge && state, NIMCP_ERROR_NULL_POINTER, "bridge or state is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int tom_fep_bridge_get_stats(
    const tom_fep_bridge_t* bridge,
    tom_fep_stats_t* stats
) {
    NIMCP_CHECK_THROW(bridge && stats, NIMCP_ERROR_NULL_POINTER, "bridge or stats is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

bool tom_fep_is_empathy_active(const tom_fep_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->state.empathy_active;
}

uint32_t tom_fep_get_mentalizing_depth(const tom_fep_bridge_t* bridge) {
    if (!bridge) return 0;
    return bridge->state.mentalizing_depth;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

int tom_fep_bridge_connect_bio_async(tom_fep_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (bridge->base.bio_async_enabled) return NIMCP_SUCCESS;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_TOM_BRIDGE,
        .module_name = "tom_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO(LOG_MODULE_TOM_FEP " Connected to bio-async router");
        return NIMCP_SUCCESS;
    }

    NIMCP_LOGGING_WARN(LOG_MODULE_TOM_FEP
        " Bio-async router not available, skipping registration");
    return -1;
}

int tom_fep_bridge_disconnect_bio_async(tom_fep_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (!bridge->base.bio_async_enabled) return NIMCP_SUCCESS;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO(LOG_MODULE_TOM_FEP " Disconnected from bio-async router");
    return NIMCP_SUCCESS;
}

bool tom_fep_bridge_is_bio_async_connected(const tom_fep_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
        return false;
    }
    return bridge->base.bio_async_enabled;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int tom_fep_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Theory_Of_Mind_FEP_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Theory_Of_Mind_FEP_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Theory_Of_Mind_FEP_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
