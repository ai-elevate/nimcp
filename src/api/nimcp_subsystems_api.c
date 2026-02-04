/**
 * @file nimcp_subsystems_api.c
 * @brief Subsystems API - neural network, ethics, knowledge
 *
 * This file contains neural network, ethics, and knowledge system APIs.
 * Extracted from nimcp.c (SRP refactoring - lines 1386-1697)
 */

#include "nimcp.h"
#include "api/nimcp_api_internal.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/knowledge/nimcp_knowledge.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdio.h>
#include <string.h>

#define LOG_MODULE "API.SUBSYSTEMS"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "security/nimcp_bbb_helpers.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(subsystems_api)

// External declarations from nimcp.c
extern void set_error(const char* fmt, ...);

//=============================================================================
// Neural Network API Implementation
//=============================================================================

NIMCP_EXPORT nimcp_network_t nimcp_network_create(
    uint32_t num_inputs,
    uint32_t num_outputs,
    uint32_t num_hidden,
    float learning_rate)
{
    // Allocate handle
    nimcp_network_t handle = (nimcp_network_t)nimcp_malloc(sizeof(struct nimcp_network_handle));
    NIMCP_API_CHECK_ALLOC_SIZE(handle, sizeof(struct nimcp_network_handle),
        "Failed to allocate network handle");

    // Create config for internal API
    network_config_t config = {0};
    config.input_size = num_inputs;
    config.output_size = num_outputs;
    // Calculate total neurons: inputs + hidden layers + outputs
    config.num_neurons = num_inputs + num_hidden + num_outputs;
    config.learning_rate = learning_rate;

    // Create internal neural network
    handle->internal_network = neural_network_create(&config);

    if (!handle->internal_network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NETWORK_CREATION, "Failed to create internal neural network");
        set_error("Failed to create internal neural network");
        nimcp_free(handle);
        return NULL;
    }

    set_error("No error");
    return handle;
}

NIMCP_EXPORT void nimcp_network_destroy(nimcp_network_t network) {
    if (!network) {
        return;
    }

    if (network->internal_network) {
        neural_network_destroy(network->internal_network);
    }

    nimcp_free(network);
}

NIMCP_EXPORT nimcp_status_t nimcp_network_forward(
    nimcp_network_t network,
    const float* inputs,
    uint32_t num_inputs,
    float* outputs,
    uint32_t num_outputs)
{
    NIMCP_API_CHECK_NULL(network, NIMCP_ERROR_NULL_ARG, "Network handle is NULL");
    NIMCP_API_CHECK_NULL(inputs, NIMCP_ERROR_NULL_ARG, "Inputs array is NULL");
    NIMCP_API_CHECK_NULL(outputs, NIMCP_ERROR_NULL_ARG, "Outputs array is NULL");

    // Call internal network API
    bool success = neural_network_forward(network->internal_network,
                                         inputs, num_inputs,
                                         outputs, num_outputs);

    if (!success) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INFERENCE_FAILED, "Network forward pass failed");
        set_error("Network forward pass failed");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_OK;
}

NIMCP_EXPORT nimcp_status_t nimcp_network_train(
    nimcp_network_t network,
    const float* inputs,
    uint32_t num_inputs,
    const float* targets,
    uint32_t num_targets)
{
    NIMCP_API_CHECK_NULL(network, NIMCP_ERROR_NULL_ARG, "Network handle is NULL");
    NIMCP_API_CHECK_NULL(inputs, NIMCP_ERROR_NULL_ARG, "Inputs array is NULL");
    NIMCP_API_CHECK_NULL(targets, NIMCP_ERROR_NULL_ARG, "Targets array is NULL");

    // Training not yet implemented in internal API
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_IMPLEMENTED, "Network training not yet implemented");
    set_error("Training not yet implemented");
    return NIMCP_ERROR;
}

//=============================================================================
// Ethics API Implementation
//=============================================================================

NIMCP_EXPORT nimcp_ethics_t nimcp_ethics_create(void) {
    nimcp_ethics_t handle = (nimcp_ethics_t)nimcp_malloc(sizeof(struct nimcp_ethics_handle));
    NIMCP_API_CHECK_ALLOC_SIZE(handle, sizeof(struct nimcp_ethics_handle),
        "Failed to allocate ethics handle");

    // Create default ethics configuration
    ethics_config_t config = {0};
    config.policies = NULL;
    config.num_policies = 0;
    config.callback = NULL;
    config.callback_context = NULL;
    config.default_severity = 0.5f;
    config.enable_learning = true;
    config.action_feature_size = 32;
    config.max_agents = 16;
    config.golden_rule_threshold = 0.0f;
    config.empathy_weight = 0.5f;

    // Create internal ethics engine
    handle->internal_ethics = ethics_engine_create(&config);

    if (!handle->internal_ethics) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_ETHICS_CREATION, "Failed to create internal ethics engine");
        set_error("Failed to create internal ethics engine");
        nimcp_free(handle);
        return NULL;
    }

    set_error("No error");
    return handle;
}

NIMCP_EXPORT void nimcp_ethics_destroy(nimcp_ethics_t ethics) {
    if (!ethics) {
        return;
    }

    if (ethics->internal_ethics) {
        ethics_engine_destroy(ethics->internal_ethics);
    }

    nimcp_free(ethics);
}

NIMCP_EXPORT nimcp_status_t nimcp_ethics_check(
    nimcp_ethics_t ethics,
    const float* situation,
    uint32_t num_features,
    float* out_score)
{
    NIMCP_API_CHECK_NULL(ethics, NIMCP_ERROR_NULL_ARG, "Ethics handle is NULL");
    NIMCP_API_CHECK_NULL(situation, NIMCP_ERROR_NULL_ARG, "Situation array is NULL");
    NIMCP_API_CHECK_NULL(out_score, NIMCP_ERROR_NULL_ARG, "Output score pointer is NULL");

    // Create action context from situation features
    action_context_t action = {0};
    action.features = (float*)situation;  // Cast away const for internal API
    action.num_features = num_features;
    action.affected_agents = NULL;
    action.num_affected_agents = 0;
    action.predicted_harm = 0.0f;

    // Evaluate using internal ethics engine
    ethics_evaluation_t eval = ethics_engine_evaluate_action(ethics->internal_ethics, &action);

    // Convert evaluation to simple score
    // Golden Rule score: -1 (harmful) to +1 (beneficial)
    *out_score = eval.golden_rule_score;

    set_error("No error");
    return NIMCP_OK;
}

//=============================================================================
// Knowledge API Implementation
//=============================================================================

NIMCP_EXPORT nimcp_knowledge_t nimcp_knowledge_create(void) {
    nimcp_knowledge_t handle = (nimcp_knowledge_t)nimcp_malloc(sizeof(struct nimcp_knowledge_handle));
    NIMCP_API_CHECK_ALLOC_SIZE(handle, sizeof(struct nimcp_knowledge_handle),
        "Failed to allocate knowledge handle");

    // Create internal knowledge system
    handle->internal_knowledge = knowledge_system_create("default");

    if (!handle->internal_knowledge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_KNOWLEDGE_CREATION, "Failed to create internal knowledge system");
        set_error("Failed to create internal knowledge system");
        nimcp_free(handle);
        return NULL;
    }

    set_error("No error");
    return handle;
}

NIMCP_EXPORT void nimcp_knowledge_destroy(nimcp_knowledge_t knowledge) {
    if (!knowledge) {
        return;
    }

    if (knowledge->internal_knowledge) {
        knowledge_system_destroy(knowledge->internal_knowledge);
    }

    nimcp_free(knowledge);
}

NIMCP_EXPORT nimcp_status_t nimcp_knowledge_add_fact(
    nimcp_knowledge_t knowledge,
    const char* subject,
    const char* predicate,
    const char* object)
{
    NIMCP_API_CHECK_NULL(knowledge, NIMCP_ERROR_NULL_ARG, "Knowledge handle is NULL");
    NIMCP_API_CHECK_NULL(subject, NIMCP_ERROR_NULL_ARG, "Subject is NULL");
    NIMCP_API_CHECK_NULL(predicate, NIMCP_ERROR_NULL_ARG, "Predicate is NULL");
    NIMCP_API_CHECK_NULL(object, NIMCP_ERROR_NULL_ARG, "Object is NULL");

    // Create a knowledge item from the fact
    knowledge_item_t item = {0};

    // Use subject as concept
    strncpy(item.concept_name, subject, sizeof(item.concept_name) - 1);

    // Create definition from predicate and object
    snprintf(item.definition, sizeof(item.definition), "%s %s", predicate, object);

    // Set defaults
    item.domain = KNOWLEDGE_DOMAIN_GENERAL;
    item.confidence = 1.0f;
    item.examples = NULL;
    item.num_examples = 0;
    item.related_concepts = NULL;
    item.num_related = 0;
    item.learned_timestamp = 0;
    item.reinforcement_count = 1;

    // Add to internal knowledge system
    bool success = knowledge_add_item(knowledge->internal_knowledge, &item);

    if (!success) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to add knowledge item '%s'", subject);
        set_error("Failed to add knowledge item");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_OK;
}

NIMCP_EXPORT nimcp_status_t nimcp_knowledge_query(
    nimcp_knowledge_t knowledge,
    const char* query,
    char* out_result,
    uint32_t max_result_len)
{
    NIMCP_API_CHECK_NULL(knowledge, NIMCP_ERROR_NULL_ARG, "Knowledge handle is NULL");
    NIMCP_API_CHECK_NULL(query, NIMCP_ERROR_NULL_ARG, "Query is NULL");
    NIMCP_API_CHECK_NULL(out_result, NIMCP_ERROR_NULL_ARG, "Output result buffer is NULL");
    NIMCP_API_CHECK(max_result_len > 0, NIMCP_ERROR_INVALID, "Invalid max_result_len (0)");

    // Try to retrieve knowledge about the query concept
    knowledge_item_t item;
    bool found = knowledge_retrieve(knowledge->internal_knowledge, query, &item);

    if (found) {
        // Return the definition
        snprintf(out_result, max_result_len, "%s", item.definition);
    } else {
        // Concept not found
        snprintf(out_result, max_result_len, "No knowledge found about '%s'", query);
    }

    set_error("No error");
    return NIMCP_OK;
}
