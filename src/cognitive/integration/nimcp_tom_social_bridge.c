/**
 * @file nimcp_tom_social_bridge.c
 * @brief Theory of Mind - Social Cognition Bridge Implementation
 *
 * WHAT: Bidirectional integration enabling ToM to inform social responses and
 *       social cues to trigger ToM inference processes.
 *
 * WHY: Theory of Mind and social cognition are deeply interdependent. Understanding
 *      others' mental states (ToM) is essential for appropriate social responses,
 *      while social cues provide input for mental state inference.
 *
 * HOW: Maintains shared agent models that both systems can query and update.
 *      ToM inferences guide social response selection; social cues trigger
 *      ToM updates about observed agents.
 *
 * @author NIMCP Development Team
 * @date 2025-01
 */

#include "cognitive/integration/nimcp_tom_social_bridge.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Internal agent model representation
 *
 * WHAT: Stores mental state model for a tracked agent
 * WHY: Shared representation for ToM and social systems
 * HOW: Contains belief/desire/intention states with confidence and timestamps
 */
typedef struct agent_model {
    uint32_t agent_id;
    float belief_state[8];      /**< Simplified belief representation */
    float desire_state[8];      /**< Simplified desire representation */
    float intention_state[8];   /**< Simplified intention representation */
    float confidence;
    uint64_t last_update;
    bool valid;
} agent_model_t;

/**
 * @brief ToM-Social bridge internal structure
 *
 * WHAT: Full bridge structure with agent tracking
 * WHY: Maintains shared agent models for ToM/social integration
 * HOW: Array of agent models with thread-safe access
 */
struct tom_social_bridge {
    tom_social_config_t config;
    agent_model_t* agents;
    size_t agent_capacity;
    size_t agent_count;
    tom_social_stats_t stats;
    nimcp_mutex_t* mutex;
    bool initialized;
};

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Find agent model by ID (unlocked)
 *
 * @param bridge Bridge instance
 * @param agent_id Agent ID to find
 * @return Pointer to agent model, NULL if not found
 */
static agent_model_t* find_agent_unlocked(tom_social_bridge_t* bridge, uint32_t agent_id) {
    for (size_t i = 0; i < bridge->agent_count; i++) {
        if (bridge->agents[i].valid && bridge->agents[i].agent_id == agent_id) {
            return &bridge->agents[i];
        }
    }
    return NULL;
}

/**
 * @brief Find or create agent model (unlocked)
 *
 * @param bridge Bridge instance
 * @param agent_id Agent ID to find or create
 * @return Pointer to agent model, NULL if capacity exceeded
 */
static agent_model_t* find_or_create_agent_unlocked(tom_social_bridge_t* bridge, uint32_t agent_id) {
    /* Try to find existing */
    agent_model_t* agent = find_agent_unlocked(bridge, agent_id);
    if (agent) return agent;

    /* Check capacity */
    if (bridge->agent_count >= bridge->agent_capacity) {
        return NULL;
    }

    /* Find first invalid slot or use next slot */
    for (size_t i = 0; i < bridge->agent_capacity; i++) {
        if (!bridge->agents[i].valid) {
            agent = &bridge->agents[i];
            break;
        }
    }

    if (!agent && bridge->agent_count < bridge->agent_capacity) {
        agent = &bridge->agents[bridge->agent_count];
    }

    if (!agent) return NULL;

    /* Initialize new agent */
    memset(agent, 0, sizeof(agent_model_t));
    agent->agent_id = agent_id;
    agent->confidence = 0.5f;  /* Start with moderate confidence */
    agent->valid = true;
    bridge->agent_count++;
    bridge->stats.active_agents = (uint32_t)bridge->agent_count;

    return agent;
}

/**
 * @brief Get current timestamp in milliseconds
 *
 * @return Current time in milliseconds
 */
static uint64_t get_current_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

int tom_social_default_config(tom_social_config_t* config) {
    if (!config) return -1;

    config->inference_depth = 2;
    config->social_weight = 0.7f;
    config->agent_capacity = 32;
    config->enable_auto_inference = true;
    config->enable_response_suggestions = true;
    config->inference_confidence_threshold = 0.5f;

    return 0;
}

/* ============================================================================
 * Lifecycle Management
 * ============================================================================ */

tom_social_bridge_t* tom_social_bridge_create(const tom_social_config_t* config) {
    tom_social_bridge_t* bridge = nimcp_malloc(sizeof(tom_social_bridge_t));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(tom_social_bridge_t));

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        tom_social_default_config(&bridge->config);
    }

    /* Allocate agents array */
    bridge->agent_capacity = bridge->config.agent_capacity;
    if (bridge->agent_capacity == 0) {
        bridge->agent_capacity = 32;  /* Default capacity */
    }
    if (bridge->agent_capacity > TOM_SOCIAL_MAX_AGENTS) {
        bridge->agent_capacity = TOM_SOCIAL_MAX_AGENTS;
    }

    bridge->agents = nimcp_calloc(bridge->agent_capacity, sizeof(agent_model_t));
    if (!bridge->agents) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Create mutex */
    bridge->mutex = nimcp_mutex_create(NULL);
    if (!bridge->mutex) {
        nimcp_free(bridge->agents);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->initialized = true;

    return bridge;
}

void tom_social_bridge_destroy(tom_social_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }

    if (bridge->agents) {
        nimcp_free(bridge->agents);
    }

    nimcp_free(bridge);
}

/* ============================================================================
 * Core Functions
 * ============================================================================ */

int tom_social_infer_for_response(
    tom_social_bridge_t* bridge,
    uint32_t agent_id,
    tom_social_mental_state_t* mental_state_out
) {
    if (!bridge || !mental_state_out) return -1;
    if (!bridge->initialized) return -1;

    nimcp_mutex_lock(bridge->mutex);

    agent_model_t* agent = find_agent_unlocked(bridge, agent_id);
    if (!agent) {
        bridge->stats.inference_failures++;
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Initialize output */
    memset(mental_state_out, 0, sizeof(tom_social_mental_state_t));
    mental_state_out->agent_id = agent_id;
    mental_state_out->confidence = agent->confidence;
    mental_state_out->last_update_ms = agent->last_update;

    /* Derive mental state from belief/desire/intention states */
    /* Compute emotional valence from belief state (average of first 4 beliefs) */
    float valence_sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        valence_sum += agent->belief_state[i];
    }
    mental_state_out->emotional_valence = (valence_sum / 4.0f) * bridge->config.social_weight;

    /* Compute arousal from intention state (average of first 4 intentions) */
    float arousal_sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        arousal_sum += agent->intention_state[i];
    }
    mental_state_out->emotional_arousal = (arousal_sum / 4.0f) * bridge->config.social_weight;

    /* Intent toward self from desire state */
    mental_state_out->intent_toward_self = agent->desire_state[0] * bridge->config.social_weight;

    /* Attention to self from intention state */
    mental_state_out->attention_to_self = agent->intention_state[0] * bridge->config.social_weight;

    /* Cognitive load from belief state complexity */
    float load_sum = 0.0f;
    for (int i = 4; i < 8; i++) {
        load_sum += agent->belief_state[i] * agent->belief_state[i];
    }
    mental_state_out->cognitive_load = (load_sum / 4.0f) * bridge->config.social_weight;

    /* Belief count based on non-zero beliefs */
    mental_state_out->belief_count = 0;
    for (int i = 0; i < 8; i++) {
        if (agent->belief_state[i] != 0.0f) {
            mental_state_out->belief_count++;
        }
    }

    /* Update statistics */
    bridge->stats.inferences_made++;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int tom_social_on_social_cue(
    tom_social_bridge_t* bridge,
    tom_social_cue_type_t cue_type,
    const void* cue_data
) {
    if (!bridge) return -1;
    if (!bridge->initialized) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Process social cue to update relevant agent models */
    /* The cue_data interpretation depends on cue_type */
    /* For now, we update a generic model based on cue type */

    /* Default agent (agent 0) if no specific agent provided */
    agent_model_t* agent = find_or_create_agent_unlocked(bridge, 0);
    if (!agent) {
        bridge->stats.cue_failures++;
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Update agent state based on cue type */
    switch (cue_type) {
        case TOM_SOCIAL_CUE_FACIAL_EXPRESSION:
            /* Update emotional belief state */
            agent->belief_state[0] += 0.1f;  /* Positive emotion indicator */
            if (agent->belief_state[0] > 1.0f) agent->belief_state[0] = 1.0f;
            break;

        case TOM_SOCIAL_CUE_BODY_LANGUAGE:
            /* Update intention state */
            agent->intention_state[0] += 0.1f;
            if (agent->intention_state[0] > 1.0f) agent->intention_state[0] = 1.0f;
            break;

        case TOM_SOCIAL_CUE_VOCAL_TONE:
            /* Update emotional arousal belief */
            agent->belief_state[1] += 0.1f;
            if (agent->belief_state[1] > 1.0f) agent->belief_state[1] = 1.0f;
            break;

        case TOM_SOCIAL_CUE_GAZE_DIRECTION:
            /* Update attention-related intention */
            agent->intention_state[1] += 0.1f;
            if (agent->intention_state[1] > 1.0f) agent->intention_state[1] = 1.0f;
            break;

        case TOM_SOCIAL_CUE_PROXIMITY:
            /* Update social desire state */
            agent->desire_state[0] += 0.1f;
            if (agent->desire_state[0] > 1.0f) agent->desire_state[0] = 1.0f;
            break;

        case TOM_SOCIAL_CUE_GESTURE:
            /* Update communication intention */
            agent->intention_state[2] += 0.1f;
            if (agent->intention_state[2] > 1.0f) agent->intention_state[2] = 1.0f;
            break;

        case TOM_SOCIAL_CUE_VERBAL_CONTENT:
            /* Update belief about agent's knowledge */
            agent->belief_state[4] += 0.1f;
            if (agent->belief_state[4] > 1.0f) agent->belief_state[4] = 1.0f;
            break;

        case TOM_SOCIAL_CUE_CONTEXT:
            /* Update contextual beliefs */
            agent->belief_state[5] += 0.1f;
            if (agent->belief_state[5] > 1.0f) agent->belief_state[5] = 1.0f;
            break;

        default:
            break;
    }

    agent->last_update = get_current_time_ms();
    bridge->stats.social_cues_processed++;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int tom_social_update_agent_model(
    tom_social_bridge_t* bridge,
    uint32_t agent_id,
    const tom_social_belief_update_t* belief_update
) {
    if (!bridge || !belief_update) return -1;
    if (!bridge->initialized) return -1;

    nimcp_mutex_lock(bridge->mutex);

    agent_model_t* agent = find_or_create_agent_unlocked(bridge, agent_id);
    if (!agent) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Compute learning rate based on inference depth */
    float learning_rate = 0.1f / (float)(bridge->config.inference_depth + 1);

    /* Apply belief update with learning rate and confidence weighting */
    uint32_t belief_index = belief_update->belief_type % 8;
    float update_magnitude = belief_update->belief_value * belief_update->confidence * learning_rate;

    /* Update appropriate state based on source */
    switch (belief_update->source) {
        case 0:  /* ToM source -> update belief state */
            agent->belief_state[belief_index] += update_magnitude;
            if (agent->belief_state[belief_index] > 1.0f)
                agent->belief_state[belief_index] = 1.0f;
            if (agent->belief_state[belief_index] < -1.0f)
                agent->belief_state[belief_index] = -1.0f;
            break;

        case 1:  /* Social source -> update desire state */
            agent->desire_state[belief_index] += update_magnitude;
            if (agent->desire_state[belief_index] > 1.0f)
                agent->desire_state[belief_index] = 1.0f;
            if (agent->desire_state[belief_index] < -1.0f)
                agent->desire_state[belief_index] = -1.0f;
            break;

        case 2:  /* External source -> update intention state */
        default:
            agent->intention_state[belief_index] += update_magnitude;
            if (agent->intention_state[belief_index] > 1.0f)
                agent->intention_state[belief_index] = 1.0f;
            if (agent->intention_state[belief_index] < -1.0f)
                agent->intention_state[belief_index] = -1.0f;
            break;
    }

    /* Update agent confidence (weighted average with new update confidence) */
    agent->confidence = agent->confidence * 0.9f + belief_update->confidence * 0.1f;

    agent->last_update = get_current_time_ms();
    bridge->stats.agent_models_updated++;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int tom_social_get_agent_state(
    tom_social_bridge_t* bridge,
    uint32_t agent_id,
    tom_social_agent_state_t* state_out
) {
    if (!bridge || !state_out) return -1;
    if (!bridge->initialized) return -1;

    nimcp_mutex_lock(bridge->mutex);

    agent_model_t* agent = find_agent_unlocked(bridge, agent_id);
    if (!agent) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Initialize output */
    memset(state_out, 0, sizeof(tom_social_agent_state_t));

    /* Copy mental state */
    state_out->mental_state.agent_id = agent->agent_id;
    state_out->mental_state.confidence = agent->confidence;
    state_out->mental_state.last_update_ms = agent->last_update;

    /* Derive mental state values from internal representation */
    float valence_sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        valence_sum += agent->belief_state[i];
    }
    state_out->mental_state.emotional_valence = valence_sum / 4.0f;

    float arousal_sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        arousal_sum += agent->intention_state[i];
    }
    state_out->mental_state.emotional_arousal = arousal_sum / 4.0f;

    state_out->mental_state.intent_toward_self = agent->desire_state[0];
    state_out->mental_state.attention_to_self = agent->intention_state[0];

    float load_sum = 0.0f;
    for (int i = 4; i < 8; i++) {
        load_sum += agent->belief_state[i] * agent->belief_state[i];
    }
    state_out->mental_state.cognitive_load = load_sum / 4.0f;

    /* Count non-zero beliefs */
    state_out->mental_state.belief_count = 0;
    for (int i = 0; i < 8; i++) {
        if (agent->belief_state[i] != 0.0f) {
            state_out->mental_state.belief_count++;
        }
    }

    /* Compute time since observation */
    uint64_t current_time = get_current_time_ms();
    state_out->time_since_observation = current_time - agent->last_update;

    /* Set observation metadata */
    state_out->observation_count = 1;  /* Simplified - would track in real impl */
    state_out->model_stability = agent->confidence;
    state_out->is_observed = (state_out->time_since_observation < 5000);  /* Observed if updated within 5s */
    state_out->is_valid = agent->valid;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/* ============================================================================
 * Statistics
 * ============================================================================ */

int tom_social_get_stats(
    const tom_social_bridge_t* bridge,
    tom_social_stats_t* stats_out
) {
    if (!bridge || !stats_out) return -1;
    if (!bridge->initialized) return -1;

    nimcp_mutex_lock(((tom_social_bridge_t*)bridge)->mutex);
    *stats_out = bridge->stats;
    nimcp_mutex_unlock(((tom_social_bridge_t*)bridge)->mutex);

    return 0;
}
