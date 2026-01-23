/**
 * @file nimcp_rcog_collective_bridge.c
 * @brief Collective Consciousness/Swarm Integration Bridge Implementation for Recursive Cognition
 * @version 1.0.0
 * @date 2026-01-03
 */

#include "cognitive/recursive/nimcp_rcog_collective_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/platform/nimcp_platform_time.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>

/*=============================================================================
 * INTERNAL STRUCTURES
 *===========================================================================*/

/**
 * @brief Collective handle for tracking distributed subtasks
 */
struct rcog_collective_handle {
    uint64_t subtask_id;             /**< Subtask being tracked */
    uint16_t assigned_drone;         /**< Assigned drone */
    uint16_t volunteers[8];          /**< Volunteer drones */
    uint8_t num_volunteers;          /**< Number of volunteers */
    bool result_received;            /**< Result received flag */
    float result_confidence;         /**< Result confidence */
    uint64_t created_at_ms;          /**< Creation timestamp */
};

/**
 * @brief Collective bridge internal structure
 */
struct rcog_collective_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */

    /* Configuration */
    rcog_collective_bridge_config_t config;

    /* Connections */
    struct collective_workspace* workspace;
    struct swarm_consciousness* consciousness;
    struct rcog_engine* engine;
    bool connected;

    /* Effects state */
    rcog_to_collective_effects_t outgoing_effects;
    collective_to_rcog_effects_t incoming_effects;

    /* Swarm member tracking */
    rcog_swarm_member_info_t members[RCOG_COLLECTIVE_MAX_SWARM_MEMBERS];
    size_t num_members;

    /* Shared context tracking */
    rcog_shared_variable_info_t shared_vars[RCOG_COLLECTIVE_MAX_SHARED_VARS];
    size_t num_shared_vars;

    /* Collective subtask tracking */
    rcog_collective_subtask_t subtasks[RCOG_COLLECTIVE_MAX_CONCURRENT_SUBTASKS];
    size_t num_subtasks;

    /* Handles for active collective operations */
    struct rcog_collective_handle* active_handles[RCOG_COLLECTIVE_MAX_CONCURRENT_SUBTASKS];
    size_t num_active_handles;

    /* Consensus state */
    float consensus_confidence;
    uint32_t agreeing_drones;
    bool consensus_reached;

    /* Statistics */
    rcog_collective_bridge_stats_t stats;
};

/*=============================================================================
 * LIFECYCLE
 *===========================================================================*/

rcog_collective_bridge_config_t rcog_collective_bridge_default_config(void) {
    rcog_collective_bridge_config_t config = {0};

    config.local_drone_id = 0;  /* Should be set by caller */

    config.broadcast_threshold = RCOG_COLLECTIVE_DEFAULT_BROADCAST_THRESHOLD;
    config.auto_broadcast_failures = true;
    config.broadcast_timeout_ms = 5000;

    config.enable_volunteering = true;
    config.volunteer_threshold = 0.7f;
    config.max_volunteered_tasks = 4;

    config.enable_stigmergy = true;
    config.stigmergy_decay_rate = 0.01f;
    config.auto_import_high_salience = true;

    config.consensus_threshold = RCOG_COLLECTIVE_DEFAULT_CONSENSUS_THRESHOLD;
    config.min_consensus_drones = 2;

    return config;
}

rcog_collective_bridge_t* rcog_collective_bridge_create(
    const rcog_collective_bridge_config_t* config
) {
    rcog_collective_bridge_t* bridge = nimcp_calloc(1, sizeof(rcog_collective_bridge_t));
    if (!bridge) {
        return NULL;
    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = rcog_collective_bridge_default_config();
    }

    /* Initialize bridge base infrastructure (includes mutex) */
    if (bridge_base_init(&bridge->base, 0, "rcog_collective") != 0) {
        nimcp_free(bridge);
        return NULL;
    }

    return bridge;
}

rcog_collective_bridge_t* rcog_collective_bridge_create_default(void) {
    return rcog_collective_bridge_create(NULL);
}

void rcog_collective_bridge_destroy(rcog_collective_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    /* Free active handles */
    for (size_t i = 0; i < bridge->num_active_handles; i++) {
        if (bridge->active_handles[i]) {
            nimcp_free(bridge->active_handles[i]);
        }
    }

    /* Cleanup base bridge infrastructure */
    bridge_base_cleanup(&bridge->base);

    nimcp_free(bridge);
}

/*=============================================================================
 * CONNECTION
 *===========================================================================*/

int rcog_collective_bridge_connect_workspace(
    rcog_collective_bridge_t* bridge,
    struct collective_workspace* workspace
) {
    if (!bridge || !workspace) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->workspace = workspace;
    bridge->connected = (bridge->workspace != NULL &&
                         bridge->consciousness != NULL &&
                         bridge->engine != NULL);
    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

int rcog_collective_bridge_connect_consciousness(
    rcog_collective_bridge_t* bridge,
    struct swarm_consciousness* consciousness
) {
    if (!bridge || !consciousness) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->consciousness = consciousness;
    bridge->connected = (bridge->workspace != NULL &&
                         bridge->consciousness != NULL &&
                         bridge->engine != NULL);
    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

int rcog_collective_bridge_connect_engine(
    rcog_collective_bridge_t* bridge,
    struct rcog_engine* engine
) {
    if (!bridge || !engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->engine = engine;
    bridge->connected = (bridge->workspace != NULL &&
                         bridge->consciousness != NULL &&
                         bridge->engine != NULL);
    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

bool rcog_collective_bridge_is_connected(const rcog_collective_bridge_t* bridge) {
    return bridge && bridge->connected;
}

/*=============================================================================
 * UPDATE
 *===========================================================================*/

int rcog_collective_bridge_update(
    rcog_collective_bridge_t* bridge,
    float delta_time_ms
) {
    if (!bridge) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Decay stigmergy salience over time */
    if (bridge->config.enable_stigmergy) {
        float decay = delta_time_ms / 1000.0f * bridge->config.stigmergy_decay_rate;
        for (size_t i = 0; i < bridge->num_shared_vars; i++) {
            bridge->shared_vars[i].salience -= decay;
            if (bridge->shared_vars[i].salience < 0) {
                bridge->shared_vars[i].salience = 0;
            }
        }
    }

    /* Update incoming effects with current state */
    bridge->incoming_effects.swarm_size = (uint16_t)bridge->num_members;
    bridge->incoming_effects.available_members = 0;
    float total_coherence = 0.0f;
    float total_load = 0.0f;

    for (size_t i = 0; i < bridge->num_members; i++) {
        if (bridge->members[i].is_available) {
            bridge->incoming_effects.available_members++;
        }
        total_coherence += bridge->members[i].coherence;
        total_load += (1.0f - bridge->members[i].capacity);
    }

    if (bridge->num_members > 0) {
        bridge->incoming_effects.swarm_coherence = total_coherence / (float)bridge->num_members;
        bridge->incoming_effects.swarm_load = total_load / (float)bridge->num_members;
    }

    bridge->incoming_effects.consensus_reached = bridge->consensus_reached;
    bridge->incoming_effects.consensus_confidence = bridge->consensus_confidence;
    bridge->incoming_effects.agreeing_drones = bridge->agreeing_drones;

    /* Copy shared variables to incoming effects */
    bridge->incoming_effects.num_shared_variables = (uint32_t)bridge->num_shared_vars;
    size_t to_copy = bridge->num_shared_vars;
    if (to_copy > RCOG_COLLECTIVE_MAX_SHARED_VARS) {
        to_copy = RCOG_COLLECTIVE_MAX_SHARED_VARS;
    }
    memcpy(bridge->incoming_effects.shared_vars, bridge->shared_vars,
           to_copy * sizeof(rcog_shared_variable_info_t));

    /* Reset outgoing effect flags */
    bridge->outgoing_effects.broadcast_subtask = false;
    bridge->outgoing_effects.share_context_variable = false;
    bridge->outgoing_effects.sync_answer_state = false;
    bridge->outgoing_effects.request_handoff = false;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

/*=============================================================================
 * SUBTASK DISTRIBUTION
 *===========================================================================*/

int rcog_collective_bridge_broadcast_subtask(
    rcog_collective_bridge_t* bridge,
    const struct rcog_subtask* subtask,
    rcog_collective_handle_t** handle
) {
    if (!bridge || !subtask || !handle) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->connected) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_SWARM_DISCONNECTED;
    }

    if (bridge->num_active_handles >= RCOG_COLLECTIVE_MAX_CONCURRENT_SUBTASKS) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_CONTEXT_FULL;
    }

    /* Create handle */
    struct rcog_collective_handle* h = nimcp_calloc(1, sizeof(struct rcog_collective_handle));
    if (!h) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_OUT_OF_MEMORY;
    }

    h->subtask_id = (uint64_t)subtask;  /* Use pointer as ID in placeholder */
    h->created_at_ms = nimcp_platform_time_monotonic_ms();

    bridge->active_handles[bridge->num_active_handles++] = h;
    *handle = h;

    /* Mark outgoing broadcast */
    bridge->outgoing_effects.broadcast_subtask = true;
    bridge->outgoing_effects.subtask_to_broadcast = h->subtask_id;

    bridge->stats.subtasks_broadcast++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

int rcog_collective_bridge_collect_results(
    rcog_collective_bridge_t* bridge,
    rcog_collective_handle_t* handle,
    rcog_subtask_result_t* results,
    size_t max_results,
    size_t* num_results
) {
    if (!bridge || !handle || !results || !num_results) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* In full implementation, would collect results from swarm */
    *num_results = 0;

    if (handle->result_received) {
        /* Return placeholder result */
        if (max_results > 0) {
            results[0].success = true;
            results[0].confidence = handle->result_confidence;
            *num_results = 1;
        }
    }

    bridge->stats.results_shared++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

int rcog_collective_bridge_volunteer_for_subtask(
    rcog_collective_bridge_t* bridge,
    uint64_t subtask_id,
    uint16_t source_drone
) {
    if (!bridge) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->connected) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_SWARM_DISCONNECTED;
    }

    /* Check if we have capacity to volunteer */
    float local_load = bridge->outgoing_effects.local_load;
    if (local_load > bridge->config.volunteer_threshold) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_CONTEXT_FULL;
    }

    /* In full implementation, would send volunteer message */
    (void)subtask_id;
    (void)source_drone;

    bridge->stats.subtasks_volunteered++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

/*=============================================================================
 * CONTEXT SHARING (STIGMERGY)
 *===========================================================================*/

int rcog_collective_bridge_share_context(
    rcog_collective_bridge_t* bridge,
    const char* variable_name,
    float salience
) {
    if (!bridge || !variable_name) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->connected) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_SWARM_DISCONNECTED;
    }

    if (!bridge->config.enable_stigmergy) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_TOOL_ACCESS_DENIED;
    }

    /* Check if variable already shared */
    for (size_t i = 0; i < bridge->num_shared_vars; i++) {
        if (strcmp(bridge->shared_vars[i].name, variable_name) == 0) {
            /* Update salience */
            bridge->shared_vars[i].salience = salience;
            bridge->shared_vars[i].shared_at_ms = nimcp_platform_time_monotonic_ms();
            nimcp_mutex_unlock(bridge->base.mutex);
            return RCOG_OK;
        }
    }

    /* Add new shared variable */
    if (bridge->num_shared_vars >= RCOG_COLLECTIVE_MAX_SHARED_VARS) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_CONTEXT_FULL;
    }

    rcog_shared_variable_info_t* var = &bridge->shared_vars[bridge->num_shared_vars++];
    strncpy(var->name, variable_name, sizeof(var->name) - 1);
    var->name[sizeof(var->name) - 1] = '\0';
    var->source_drone = bridge->config.local_drone_id;
    var->salience = salience;
    var->shared_at_ms = nimcp_platform_time_monotonic_ms();
    var->is_local = true;

    /* Mark outgoing share */
    bridge->outgoing_effects.share_context_variable = true;
    bridge->outgoing_effects.variable_to_share = variable_name;
    bridge->outgoing_effects.variable_salience = salience;

    bridge->stats.context_vars_shared++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

int rcog_collective_bridge_import_context(
    rcog_collective_bridge_t* bridge,
    uint16_t source_drone,
    const char* variable_name
) {
    if (!bridge || !variable_name) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->connected) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_SWARM_DISCONNECTED;
    }

    /* Find the shared variable */
    for (size_t i = 0; i < bridge->num_shared_vars; i++) {
        if (bridge->shared_vars[i].source_drone == source_drone &&
            strcmp(bridge->shared_vars[i].name, variable_name) == 0) {
            bridge->shared_vars[i].is_local = true;
            bridge->stats.context_vars_imported++;
            nimcp_mutex_unlock(bridge->base.mutex);
            return RCOG_OK;
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return RCOG_ERROR_CONTEXT_NOT_FOUND;
}

int rcog_collective_bridge_list_shared_context(
    const rcog_collective_bridge_t* bridge,
    rcog_shared_variable_info_t* vars,
    size_t max_vars,
    size_t* num_vars
) {
    if (!bridge || !vars || !num_vars) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(((rcog_collective_bridge_t*)bridge)->base.mutex);

    size_t to_copy = bridge->num_shared_vars;
    if (to_copy > max_vars) {
        to_copy = max_vars;
    }

    memcpy(vars, bridge->shared_vars, to_copy * sizeof(rcog_shared_variable_info_t));
    *num_vars = to_copy;

    nimcp_mutex_unlock(((rcog_collective_bridge_t*)bridge)->base.mutex);

    return RCOG_OK;
}

/*=============================================================================
 * ANSWER CONSENSUS
 *===========================================================================*/

int rcog_collective_bridge_refine_answer(
    rcog_collective_bridge_t* bridge,
    struct rcog_answer_state* state
) {
    if (!bridge || !state) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->connected) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_SWARM_DISCONNECTED;
    }

    /* Mark sync request */
    bridge->outgoing_effects.sync_answer_state = true;

    /* In full implementation, would initiate distributed refinement */

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

bool rcog_collective_bridge_consensus_reached(
    const rcog_collective_bridge_t* bridge,
    const struct rcog_answer_state* state,
    float coherence_threshold
) {
    if (!bridge || !state) {
        return false;
    }

    (void)state;  /* Would compare in full implementation */

    return bridge->consensus_reached &&
           bridge->incoming_effects.swarm_coherence >= coherence_threshold;
}

float rcog_collective_bridge_get_consensus_confidence(
    const rcog_collective_bridge_t* bridge
) {
    if (!bridge) {
        return 0.0f;
    }
    return bridge->consensus_confidence;
}

/*=============================================================================
 * SWARM STATUS
 *===========================================================================*/

int rcog_collective_bridge_get_swarm_members(
    const rcog_collective_bridge_t* bridge,
    rcog_swarm_member_info_t* members,
    size_t max_members,
    size_t* num_members
) {
    if (!bridge || !members || !num_members) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(((rcog_collective_bridge_t*)bridge)->base.mutex);

    size_t to_copy = bridge->num_members;
    if (to_copy > max_members) {
        to_copy = max_members;
    }

    memcpy(members, bridge->members, to_copy * sizeof(rcog_swarm_member_info_t));
    *num_members = to_copy;

    nimcp_mutex_unlock(((rcog_collective_bridge_t*)bridge)->base.mutex);

    return RCOG_OK;
}

float rcog_collective_bridge_get_swarm_coherence(
    const rcog_collective_bridge_t* bridge
) {
    if (!bridge) {
        return 0.0f;
    }
    return bridge->incoming_effects.swarm_coherence;
}

uint32_t rcog_collective_bridge_get_active_subtasks(
    const rcog_collective_bridge_t* bridge
) {
    if (!bridge) {
        return 0;
    }
    return (uint32_t)bridge->num_subtasks;
}

/*=============================================================================
 * EFFECTS ACCESS
 *===========================================================================*/

int rcog_collective_bridge_get_outgoing_effects(
    const rcog_collective_bridge_t* bridge,
    rcog_to_collective_effects_t* effects
) {
    if (!bridge || !effects) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(((rcog_collective_bridge_t*)bridge)->base.mutex);
    *effects = bridge->outgoing_effects;
    nimcp_mutex_unlock(((rcog_collective_bridge_t*)bridge)->base.mutex);

    return RCOG_OK;
}

int rcog_collective_bridge_get_incoming_effects(
    const rcog_collective_bridge_t* bridge,
    collective_to_rcog_effects_t* effects
) {
    if (!bridge || !effects) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(((rcog_collective_bridge_t*)bridge)->base.mutex);
    *effects = bridge->incoming_effects;
    nimcp_mutex_unlock(((rcog_collective_bridge_t*)bridge)->base.mutex);

    return RCOG_OK;
}

/*=============================================================================
 * STATISTICS
 *===========================================================================*/

int rcog_collective_bridge_get_stats(
    const rcog_collective_bridge_t* bridge,
    rcog_collective_bridge_stats_t* stats
) {
    if (!bridge || !stats) {
        return RCOG_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(((rcog_collective_bridge_t*)bridge)->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(((rcog_collective_bridge_t*)bridge)->base.mutex);

    return RCOG_OK;
}

void rcog_collective_bridge_reset_stats(rcog_collective_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(rcog_collective_bridge_stats_t));
    nimcp_mutex_unlock(bridge->base.mutex);
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int rcog_collective_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Recursive_Cognition_Collective_Bridge_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Log self-knowledge observations */
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Recursive_Cognition_Collective_Bridge_Module");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Recursive_Cognition_Collective_Bridge_Module");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
