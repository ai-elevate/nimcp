//=============================================================================
// nimcp_ethics_learning.c - Learning and Adaptation Functions
//=============================================================================
// RESPONSIBILITY: Learning from outcomes and adapting behavior
//
// This module implements learning mechanisms that allow the ethics engine
// to improve its predictions based on actual outcomes.
//=============================================================================

#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/ethics/nimcp_ethics_internal.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "core/brain/nimcp_brain.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <math.h>

#define LOG_MODULE "ethics_learning"

//=============================================================================
// Learning Validation
//=============================================================================

/**
 * @brief Validates learning inputs
 *
 * WHY: Extracted validation. Guard clauses prevent nested ifs.
 *
 * COMPLEXITY: O(1)
 */
bool ethics_validate_learning_inputs(ethics_engine_t engine, const action_context_t* action,
                                     const action_outcome_t* outcome)
{
    // Guard clause: Check engine
    if (!engine)
        return false;

    // Guard clause: Check action
    if (!action)
        return false;

    // Guard clause: Check outcome
    if (!outcome)
        return false;

    // Guard clause: Check if learning enabled
    if (!ethics_engine_is_learning_enabled(engine))
        return false;

    return true;
}

//=============================================================================
// Learning Update Functions
//=============================================================================

/**
 * @brief Updates Golden Rule evaluator with outcome
 *
 * WHY: Extracted learning logic. Single responsibility.
 *
 * COMPLEXITY: O(1)
 */
void ethics_update_golden_rule_learning(ethics_engine_t engine, const action_context_t* action,
                                        float actual_impact)
{
    // Guard clause: Validate inputs
    if (!engine || !action)
        return;

    const char* label = (actual_impact < 0) ? "accept" : "reject";
    float confidence = fminf(fabsf(actual_impact), 1.0F);

    brain_t golden_rule_net = ethics_engine_get_golden_rule_net(engine);
    if (golden_rule_net) {
        brain_learn_example(golden_rule_net, action->features, action->num_features,
                            label, confidence);
    }
}

/**
 * @brief Updates empathy network with actual outcome
 *
 * WHY: Extracted empathy learning. Uses buffer pool to avoid allocation.
 *
 * COMPLEXITY: O(1)
 */
void ethics_update_empathy_learning(ethics_engine_t engine, const action_context_t* action,
                                    const action_outcome_t* outcome)
{
    // Guard clause: Validate inputs
    if (!engine || !action || !outcome)
        return;

    empathy_network_t empathy_net = ethics_engine_get_empathy_net(engine);
    if (!empathy_net)
        return;

    // Guard clause: Check agent bounds
    if (outcome->affected_agent >= empathy_net->num_agents)
        return;

    // Acquire buffer from pool (O(1))
    float* combined = ethics_engine_acquire_buffer(engine);
    float stack_buffer[MAX_FEATURE_SIZE] = {0};
    if (!combined) {
        combined = stack_buffer;
    }

    // Prepare features
    ethics_copy_action_features(combined, action, 10);

    char emotion_label[64];
    snprintf(emotion_label, sizeof(emotion_label), "impact_%.2f", outcome->emotional_impact);

    if (empathy_net->perspective_network) {
        brain_learn_example(empathy_net->perspective_network, combined, 20, emotion_label,
                            1.0F - outcome->uncertainty);
    }

    // Release buffer
    if (combined != stack_buffer) {
        ethics_engine_release_buffer(engine, combined);
    }
}

/**
 * @brief Learns from action outcome to improve future predictions
 *
 * WHY: Enables ethics engine to refine its Golden Rule predictions based
 * on actual outcomes. Critical for adapting to context-specific situations.
 *
 * ALGORITHM:
 * 1. Validate inputs and check if learning enabled (O(1))
 * 2. Calculate actual impact from outcome (O(1))
 * 3. Update Golden Rule evaluator (O(1))
 * 4. Update empathy network for perspective-taking (O(1))
 *
 * COMPLEXITY: O(1) - Constant time learning update
 *
 * @return true if learning performed, false otherwise
 */
bool ethics_learn_from_outcome(ethics_engine_t engine, const action_context_t* action,
                               const action_outcome_t* outcome)
{
    // Guard clause: Validate all inputs
    if (!ethics_validate_learning_inputs(engine, action, outcome)) {
        return false;
    }

    // Calculate actual impact
    float actual_impact = outcome->actual_harm - outcome->actual_benefit;

    // Update both networks
    ethics_update_golden_rule_learning(engine, action, actual_impact);
    ethics_update_empathy_learning(engine, action, outcome);

    return true;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Ethics Learning self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int ethics_learning_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Ethics_Learning_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            LOG_DEBUG("Ethics learning self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Ethics_Learning_Module");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Ethics_Learning_Module");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
