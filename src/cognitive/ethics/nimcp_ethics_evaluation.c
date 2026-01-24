//=============================================================================
// nimcp_ethics_evaluation.c - Ethics Evaluation Functions
//=============================================================================
// RESPONSIBILITY: Golden Rule evaluation, empathy network, perspective-taking
//
// This module implements the core ethical evaluation based on the Golden Rule
// using empathy networks to simulate others' perspectives.
//=============================================================================

#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/ethics/nimcp_ethics_internal.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "core/brain/nimcp_brain.h"
#include "cognitive/nimcp_working_memory.h"
#include "cognitive/nimcp_emotional_tagging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <math.h>
#include <string.h>

#define LOG_MODULE "ethics_evaluation"

//=============================================================================
// Helper Functions for Feature Copying
//=============================================================================

/**
 * @brief Copies action features to combined buffer
 *
 * WHY: Extracted from nested loop to improve readability and testability.
 * Single responsibility - only handles feature copying.
 *
 * COMPLEXITY: O(n) where n = min(max_features, num_features)
 */
void ethics_copy_action_features(float* dest, const action_context_t* action, uint32_t max_features)
{
    // Guard clause: Validate inputs
    if (!dest || !action)
        return;

    uint32_t count = (action->num_features < max_features) ? action->num_features : max_features;

    for (uint32_t i = 0; i < count; i++) {
        dest[i] = action->features[i];
    }
}

/**
 * @brief Copies agent emotional state to buffer
 *
 * WHY: Extracted from nested loop. Isolates emotional state handling.
 * Makes code more maintainable and testable.
 *
 * COMPLEXITY: O(n) where n = num_emotions
 */
void ethics_copy_emotional_state(float* dest, const float* emotional_states, agent_id_t agent,
                                 uint32_t num_emotions, uint32_t offset)
{
    // Guard clause: Validate inputs
    if (!dest || !emotional_states)
        return;

    for (uint32_t i = 0; i < num_emotions; i++) {
        dest[offset + i] = emotional_states[agent * num_emotions + i];
    }
}

/**
 * @brief Calculates perspective score from empathy state
 *
 * WHY: Extracted calculation logic for clarity. Single responsibility.
 * Mathematical formula isolated for easy testing and modification.
 *
 * COMPLEXITY: O(1)
 *
 * @return Perspective score [-1.0, 1.0] weighted by impact magnitude
 */
float ethics_calculate_perspective_score(const empathy_state_extended_t* state)
{
    // Guard clause: Validate input
    if (!state)
        return 0.0F;

    float emotional = state->emotional_valence;
    float material = state->material_impact;
    float autonomy = state->autonomy_impact;

    // Average the three impact dimensions
    float average_score = (emotional + material + autonomy) / 3.0F;

    // Weight by impact magnitude for importance
    return average_score * state->impact_magnitude;
}

/**
 * @brief Simulates single agent's perspective for an action
 *
 * WHY: Extracted from loop to eliminate nesting. Each agent's simulation
 * is independent and can be tested in isolation.
 *
 * COMPLEXITY: O(1) - Fixed neural network inference time
 *
 * @return Impact score for this agent's perspective
 */
float ethics_simulate_agent_perspective(empathy_network_t network, agent_id_t agent,
                                        const action_context_t* action)
{
    // Guard clause: Validate inputs
    if (!network || !action)
        return 0.0F;

    empathy_state_extended_t state = empathy_network_simulate_agent(network, agent, action);

    return ethics_calculate_perspective_score(&state);
}

/**
 * @brief Evaluates Golden Rule by simulating affected parties' perspectives
 *
 * WHY: Implements "Do unto others as you would have them done unto you"
 * with the ultimate goal of improving the human condition.
 * Refactored to eliminate nested loops - each agent processed independently.
 *
 * CORE DIRECTIVE: All actions must serve to improve the human condition
 * through empathy, fairness, and respect for human dignity.
 *
 * ALGORITHM:
 * 1. For each affected agent, simulate their experience (O(n))
 * 2. Calculate perspective score (would I want this?) (O(1))
 * 3. Accumulate weighted scores (O(1))
 * 4. Return normalized average (O(1))
 *
 * COMPLEXITY: O(n) where n = num_affected_agents (previously had nested loops)
 *
 * @return Golden Rule score [-1.0, 1.0] where positive = you'd want it
 */
float ethics_evaluate_golden_rule(ethics_engine_t engine, const action_context_t* action)
{
    // Guard clause: Validate inputs
    if (!engine || !action)
        return 0.0F;

    // Guard clause: Check for affected agents
    if (action->num_affected_agents == 0)
        return 0.0F;

    float total_impact = 0.0F;

    // Single pass through affected agents - O(n)
    for (uint32_t i = 0; i < action->num_affected_agents; i++) {
        agent_id_t agent = action->affected_agents[i];
        float perspective_score = ethics_simulate_agent_perspective(
            ethics_engine_get_empathy_net(engine), agent, action);
        total_impact += perspective_score;
    }

    // Normalize by number of affected parties
    return total_impact / action->num_affected_agents;
}

//=============================================================================
// Empathy Network Implementation
//=============================================================================

/**
 * @brief Creates empathy network for perspective-taking
 *
 * WHY: Enables "putting yourself in others' shoes" - core to Golden Rule.
 * Neural network learns to predict others' experiences.
 *
 * COMPLEXITY: O(1) - Fixed initialization
 *
 * @return Network instance or NULL on failure
 */
empathy_network_t empathy_network_create(const empathy_config_t* config)
{
    // Guard clause: Validate input
    if (!config) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config is NULL");

        return NULL;

    }

    empathy_network_t network = nimcp_calloc(1, sizeof(struct empathy_network_struct));
    // Guard clause: Check allocation
    if (!network) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "network is NULL");

        return NULL;

    }

    uint32_t max_agents = 1000;

    /**
     * WHAT: Create perspective-taking brain for empathy network
     * WHY: Model other agents' mental states for ethical evaluation
     * HOW: Use BRAIN_SIZE_MEDIUM (10K neurons) for sufficient capacity
     * PATTERN: Composition - brain encapsulates neural network
     * NOTE: Uses minimal brain to avoid infinite recursion (empathy -> ethics -> empathy)
     */
    network->perspective_network =
        brain_create_minimal("perspective_taking", BRAIN_SIZE_MEDIUM, BRAIN_TASK_REGRESSION,
                     20,  // Action + agent features (input)
                     5    // Perspective dimensions (output)
        );


    // Guard clause: Check network creation
    if (!network->perspective_network) {
        nimcp_free(network);
        return NULL;
    }


    network->num_agents = max_agents;
    network->states = nimcp_calloc(max_agents, sizeof(empathy_state_t));
    network->num_emotions = 10;
    network->emotional_states = nimcp_calloc(max_agents * network->num_emotions, sizeof(float));

    // Guard clause: Check allocations
    if (!network->states || !network->emotional_states) {
        brain_destroy(network->perspective_network);
        nimcp_free(network->states);
        nimcp_free(network->emotional_states);
        nimcp_free(network);
        return NULL;
    }

    return network;
}

/**
 * @brief Destroys empathy network and frees resources
 *
 * COMPLEXITY: O(1)
 */
void empathy_network_destroy(empathy_network_t network)
{
    // Guard clause: Validate input
    if (!network)
        return;

    brain_destroy(network->perspective_network);
    nimcp_free(network->states);
    nimcp_free(network->emotional_states);
    nimcp_free(network);
}

/**
 * @brief Simulates agent's experience of an action
 *
 * WHY: Core of perspective-taking. Predicts how action would feel from
 * the affected agent's viewpoint. Essential for Golden Rule evaluation.
 *
 * ALGORITHM:
 * 1. Combine action features with agent's emotional state (O(n))
 * 2. Run through perspective neural network (O(1) inference time)
 * 3. Extract predicted experience dimensions (O(1))
 *
 * COMPLEXITY: O(n) where n = feature count (typically small, ~20)
 *
 * @return Extended empathy state with predicted experience
 */
empathy_state_extended_t empathy_network_simulate_agent(empathy_network_t network, agent_id_t agent,
                                                        const action_context_t* action)
{
    empathy_state_extended_t state = {0};

    // Guard clause: Validate inputs
    if (!network || !action)
        return state;

    // Guard clause: Check agent bounds
    if (agent >= network->num_agents)
        return state;

    // Get buffer from pool for O(1) allocation
    // Simplified - would use engine's pool
    float* combined_features = NULL;

    // Fallback to stack allocation if pool exhausted
    float stack_buffer[MAX_FEATURE_SIZE] = {0};
    if (!combined_features) {
        combined_features = stack_buffer;
    }

    // Combine features (O(n))
    ethics_copy_action_features(combined_features, action, 10);
    ethics_copy_emotional_state(combined_features, network->emotional_states, agent, network->num_emotions,
                         10);

    // Neural network inference (O(1))
    brain_decision_t* decision = brain_decide(network->perspective_network, combined_features, 20);

    if (decision) {
        state.emotional_valence = decision->output_vector[0];
        state.material_impact = decision->output_vector[1];
        state.autonomy_impact = decision->output_vector[2];
        state.impact_magnitude = decision->output_vector[3];
        state.uncertainty = decision->output_vector[4];
        brain_free_decision(decision);
    }

    /* =========================================================================
     * PHASE 10.3: Emotional working memory modulation of empathy
     * =========================================================================
     * WHAT: Retrieve emotional context from working memory to modulate empathy
     * WHY:  Recent emotional experiences bias perspective-taking ability
     * HOW:  High arousal → impaired empathy, emotional congruence → enhanced empathy
     *
     * BIOLOGICAL BASIS:
     * - Emotional congruence: Same emotional state enhances empathy (Singer & Lamm, 2009)
     * - Emotional flooding: High arousal impairs perspective-taking accuracy
     * - Affective empathy vs cognitive empathy: Emotions modulate both pathways
     */
    if (network->main_brain) {
        working_memory_t* wm = brain_get_working_memory(network->main_brain);

        if (wm && working_memory_get_size(wm) > 0) {
            /* WHAT: Extract emotional context from working memory */
            float max_arousal = 0.0F;
            float avg_valence = 0.0F;
            uint32_t emotional_count = 0;
            uint32_t wm_size = working_memory_get_size(wm);

            /* WHAT: Aggregate recent emotional experiences */
            for (uint32_t i = 0; i < wm_size && i < 7; i++) {  // Miller's 7±2 limit
                emotional_tag_t emotion;
                if (working_memory_get_emotion(wm, i, &emotion) && emotion.intensity > 0.3F) {
                    if (emotion.arousal > max_arousal) {
                        max_arousal = emotion.arousal;
                    }
                    avg_valence += emotion.valence;
                    emotional_count++;
                }
            }

            if (emotional_count > 0) {
                avg_valence /= emotional_count;

                /* WHAT: High arousal impairs empathetic accuracy */
                /* WHY:  Emotional flooding reduces cognitive empathy (Decety & Cowell, 2014) */
                /* HOW:  Increase uncertainty and reduce magnitude for high arousal */
                if (max_arousal > 0.6F) {
                    state.uncertainty += (max_arousal - 0.6F) * 0.5F;  // Up to +0.2 uncertainty
                    if (state.uncertainty > 1.0F) state.uncertainty = 1.0F;

                    // Emotional flooding reduces empathetic accuracy
                    state.impact_magnitude *= (1.0F - (max_arousal - 0.6F) * 0.3F);  // Up to 12% reduction
                }

                /* WHAT: Emotional congruence enhances empathy */
                /* WHY:  Similar emotional states increase affective empathy */
                /* HOW:  If our emotional valence matches the predicted impact, boost confidence */
                float emotional_congruence = 1.0F - fabsf(avg_valence - state.emotional_valence);
                if (emotional_congruence > 0.7F) {
                    // High congruence: reduce uncertainty, boost magnitude
                    state.uncertainty *= 0.8F;  // 20% reduction in uncertainty
                    state.impact_magnitude *= 1.1F;  // 10% boost to empathetic accuracy
                    if (state.impact_magnitude > 1.0F) state.impact_magnitude = 1.0F;
                }

                /* WHAT: Emotional incongruence impairs empathy */
                /* WHY:  Opposite emotional states reduce perspective-taking accuracy */
                /* HOW:  If emotional states oppose, increase uncertainty */
                if (emotional_congruence < 0.3F) {
                    state.uncertainty += 0.1F;  // Increased uncertainty
                    if (state.uncertainty > 1.0F) state.uncertainty = 1.0F;
                }
            }
        }
    }

    state.agent_id = agent;
    state.active = true;

    // Release buffer back to pool if used
    if (combined_features != stack_buffer) {
        // Would release to pool here
    }

    return state;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Ethics Evaluation self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int ethics_evaluation_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Ethics_Evaluation_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            LOG_DEBUG("Ethics evaluation self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Ethics_Evaluation_Module");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Ethics_Evaluation_Module");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
