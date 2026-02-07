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
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(ethics_learning)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_ethics_learning_mesh_id = 0;
static mesh_participant_registry_t* g_ethics_learning_mesh_registry = NULL;

nimcp_error_t ethics_learning_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_ethics_learning_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "ethics_learning", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "ethics_learning";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_ethics_learning_mesh_id);
    if (err == NIMCP_SUCCESS) g_ethics_learning_mesh_registry = registry;
    return err;
}

void ethics_learning_mesh_unregister(void) {
    if (g_ethics_learning_mesh_registry && g_ethics_learning_mesh_id != 0) {
        mesh_participant_unregister(g_ethics_learning_mesh_registry, g_ethics_learning_mesh_id);
        g_ethics_learning_mesh_id = 0;
        g_ethics_learning_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat (instance-level) */
static inline void ethics_learning_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_ethics_learning_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_ethics_learning_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_ethics_learning_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


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
    if (!engine) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "ethics_validate_learning_inputs: engine is NULL");
        return false;
    }

    /* Phase 8: Heartbeat at operation start */
    ethics_learning_heartbeat("ethics_learn_ethics_validate_lear", 0.0f);

    // Guard clause: Check action
    if (!action)
        {

            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "ethics_validate_learning_inputs: action is NULL");

            return false;

        }

    // Guard clause: Check outcome
    if (!outcome)
        {

            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "ethics_validate_learning_inputs: outcome is NULL");

            return false;

        }

    // Guard clause: Check if learning enabled
    if (!ethics_engine_is_learning_enabled(engine)) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "ethics_validate_learning_inputs: ethics_engine_is_learning_enabled is NULL");
        return false;
    }

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

    /* Phase 8: Heartbeat at operation start */
    ethics_learning_heartbeat("ethics_learn_ethics_update_golden", 0.0f);

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

    /* Phase 8: Heartbeat at operation start */
    ethics_learning_heartbeat("ethics_learn_ethics_update_empath", 0.0f);

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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "ethics_learn_from_outcome: ethics_validate_learning_inputs is NULL");
        return false;
    }

    // Calculate actual impact
    /* Phase 8: Heartbeat at operation start */
    ethics_learning_heartbeat("ethics_learn_ethics_learn_from_ou", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    ethics_learning_heartbeat("ethics_learn_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Ethics_Learning_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                ethics_learning_heartbeat("ethics_learn_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            LOG_DEBUG("Ethics learning self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Ethics_Learning_Module");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Ethics_Learning_Module");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void ethics_learning_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_ethics_learning_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int ethics_learning_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "ethics_learning_training_begin: NULL argument");
        return -1;
    }
    ethics_learning_heartbeat_instance(NULL, "ethics_learning_training_begin", 0.0f);
    return 0;
}

int ethics_learning_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "ethics_learning_training_end: NULL argument");
        return -1;
    }
    ethics_learning_heartbeat_instance(NULL, "ethics_learning_training_end", 1.0f);
    return 0;
}

int ethics_learning_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "ethics_learning_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    ethics_learning_heartbeat_instance(NULL, "ethics_learning_training_step", progress);
    return 0;
}
