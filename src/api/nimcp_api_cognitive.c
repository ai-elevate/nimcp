/**
 * @file nimcp_api_cognitive.c
 * @brief Cognitive systems API implementation
 *
 * This module handles cognitive system operations including:
 * - Working memory management
 * - Global workspace theory implementation
 * - Ethics engine operations
 * - Knowledge system management
 *
 * Responsibilities:
 * - Working memory add/get/refresh/stats
 * - Global workspace compete/read/subscribe/stats
 * - Ethics evaluation
 * - Knowledge base operations
 */

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "API_COGNITIVE"

/* API Exception Integration (Phase 7) */
extern void set_error(const char* fmt, ...);
#define NIMCP_API_SET_ERROR(fmt, ...) set_error(fmt, ##__VA_ARGS__)
#include "api/nimcp_api_exception.h"

#include "nimcp.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "cognitive/nimcp_working_memory.h"
#include "cognitive/global_workspace/nimcp_global_workspace.h"
#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/knowledge/nimcp_knowledge.h"
#include "utils/memory/nimcp_memory.h"
#include <stdio.h>
#include <string.h>

//=============================================================================
// External References (from nimcp.c)
//=============================================================================

// These functions are defined in nimcp.c and shared across modules
extern void set_error(const char* fmt, ...);
extern const char* nimcp_get_error(void);

//=============================================================================
// Working Memory API Implementation
//=============================================================================

nimcp_status_t nimcp_brain_working_memory_add(
    nimcp_brain_t brain,
    const float* data,
    uint32_t size,
    float salience)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to working_memory_add");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL brain provided to working_memory_add");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain has NULL internal_brain in working_memory_add");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Validate parameters FIRST (before checking subsystem availability)
    if (!data) {
        set_error("NULL data provided to working_memory_add");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL data provided to working_memory_add");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (size == 0) {
        set_error("Invalid size (0) provided to working_memory_add");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Invalid size (0) provided to working_memory_add");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if working memory enabled (after parameter validation)
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    if (!wm) {
        set_error("Working memory not enabled in brain config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Working memory not enabled in brain config");
        return NIMCP_ERROR_INVALID;
    }

    // Add to working memory
    bool success = working_memory_add(wm, data, size, salience);
    if (!success) {
        set_error("Failed to add item to working memory");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to add item to working memory");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_OK;
}

const float* nimcp_brain_working_memory_get(
    nimcp_brain_t brain,
    uint32_t index,
    uint32_t* size_out)
{
    // Guard: Validate brain
    if (!brain || !brain->internal_brain) {
        set_error("Invalid brain handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Invalid brain handle in working_memory_get");
        return NULL;
    }

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    if (!wm) {
        set_error("Working memory not enabled");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Working memory not enabled in working_memory_get");
        return NULL;
    }

    // Get item
    const float* item = working_memory_get(wm, index, size_out);
    if (!item) {
        set_error("Invalid index or empty working memory");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Invalid index %u or empty working memory", index);
        return NULL;
    }

    set_error("No error");
    return item;
}

nimcp_status_t nimcp_brain_working_memory_stats(
    nimcp_brain_t brain,
    uint32_t* current_size_out,
    uint32_t* capacity_out)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL brain provided to working_memory_stats");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain has NULL internal_brain in working_memory_stats");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Validate output parameters
    if (!current_size_out || !capacity_out) {
        set_error("NULL output parameters");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL output parameters in working_memory_stats");
        return NIMCP_ERROR_NULL_ARG;
    }

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    if (!wm) {
        set_error("Working memory not enabled");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Working memory not enabled in working_memory_stats");
        return NIMCP_ERROR_INVALID;
    }

    // Get stats
    working_memory_stats_t stats;
    working_memory_get_stats(wm, &stats);

    *current_size_out = stats.current_size;
    *capacity_out = stats.capacity;

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_working_memory_refresh(
    nimcp_brain_t brain,
    uint32_t index)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL brain provided to working_memory_refresh");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain has NULL internal_brain in working_memory_refresh");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    if (!wm) {
        set_error("Working memory not enabled");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Working memory not enabled in working_memory_refresh");
        return NIMCP_ERROR_INVALID;
    }

    // Refresh item
    bool success = working_memory_refresh(wm, index);
    if (!success) {
        set_error("Invalid index for refresh");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Invalid index %u for working memory refresh", index);
        return NIMCP_ERROR_INVALID;
    }

    set_error("No error");
    return NIMCP_OK;
}

//=============================================================================
// Global Workspace API Implementation
//=============================================================================

/**
 * @brief Helper to convert public API module enum to internal enum
 */
static cognitive_module_t convert_module_enum(nimcp_cognitive_module_t public_module)
{
    // Direct mapping - enums have same values
    return (cognitive_module_t)public_module;
}

nimcp_status_t nimcp_brain_workspace_compete(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module,
    const float* content,
    uint32_t content_dim,
    float strength)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to workspace_compete");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL brain provided to workspace_compete");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain has NULL internal_brain in workspace_compete");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Validate parameters FIRST (before checking subsystem availability)
    if (!content) {
        set_error("NULL content provided to workspace_compete");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL content provided to workspace_compete");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (content_dim == 0) {
        set_error("Invalid content_dim (0)");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Invalid content_dim (0) in workspace_compete");
        return NIMCP_ERROR_INVALID;
    }

    if (strength < 0.0f || strength > 1.0f) {
        set_error("Strength must be in range [0.0, 1.0]");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Invalid strength %.2f in workspace_compete", strength);
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if global workspace enabled (after parameter validation)
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    if (!workspace) {
        set_error("Global workspace not enabled in brain config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Global workspace not enabled in brain config");
        return NIMCP_ERROR_INVALID;
    }

    // Convert module enum and compete
    cognitive_module_t internal_module = convert_module_enum(module);
    bool won = global_workspace_compete(workspace, internal_module, content, content_dim, strength);

    if (won) {
        set_error("No error");
        return NIMCP_OK;
    } else {
        set_error("Did not win workspace competition");
        return NIMCP_ERROR;
    }
}

nimcp_status_t nimcp_brain_workspace_read(
    nimcp_brain_t brain,
    float* content,
    uint32_t max_dim,
    uint32_t* actual_dim,
    nimcp_cognitive_module_t* source_module)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to workspace_read");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL brain provided to workspace_read");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain has NULL internal_brain in workspace_read");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Validate parameters FIRST (before checking subsystem availability)
    if (!content || !actual_dim || !source_module) {
        set_error("NULL output parameter provided to workspace_read");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL output parameter provided to workspace_read");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (max_dim == 0) {
        set_error("Invalid max_dim (0)");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Invalid max_dim (0) in workspace_read");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if global workspace enabled (after parameter validation)
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    if (!workspace) {
        set_error("Global workspace not enabled in brain config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Global workspace not enabled in brain config");
        return NIMCP_ERROR_INVALID;
    }

    // Read broadcast
    cognitive_module_t internal_source;
    bool success = global_workspace_read_broadcast(
        workspace, content, max_dim, actual_dim, &internal_source
    );

    if (success) {
        *source_module = (nimcp_cognitive_module_t)internal_source;
        set_error("No error");
        return NIMCP_OK;
    } else {
        set_error("No broadcast available or buffer too small");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "No broadcast available or buffer too small in workspace_read");
        return NIMCP_ERROR;
    }
}

nimcp_status_t nimcp_brain_workspace_subscribe(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to workspace_subscribe");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL brain provided to workspace_subscribe");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain has NULL internal_brain in workspace_subscribe");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    if (!workspace) {
        set_error("Global workspace not enabled in brain config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Global workspace not enabled in workspace_subscribe");
        return NIMCP_ERROR_INVALID;
    }

    // Subscribe module
    cognitive_module_t internal_module = convert_module_enum(module);
    bool success = global_workspace_subscribe(workspace, internal_module);

    if (success) {
        set_error("No error");
        return NIMCP_OK;
    } else {
        set_error("Failed to subscribe module");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to subscribe module to workspace");
        return NIMCP_ERROR;
    }
}

nimcp_status_t nimcp_brain_workspace_unsubscribe(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to workspace_unsubscribe");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL brain provided to workspace_unsubscribe");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain has NULL internal_brain in workspace_unsubscribe");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    if (!workspace) {
        set_error("Global workspace not enabled in brain config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Global workspace not enabled in workspace_unsubscribe");
        return NIMCP_ERROR_INVALID;
    }

    // Unsubscribe module
    cognitive_module_t internal_module = convert_module_enum(module);
    bool success = global_workspace_unsubscribe(workspace, internal_module);

    if (success) {
        set_error("No error");
        return NIMCP_OK;
    } else {
        set_error("Failed to unsubscribe module");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to unsubscribe module from workspace");
        return NIMCP_ERROR;
    }
}

nimcp_status_t nimcp_brain_workspace_has_broadcast(
    nimcp_brain_t brain,
    bool* has_broadcast)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to workspace_has_broadcast");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL brain provided to workspace_has_broadcast");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain has NULL internal_brain in workspace_has_broadcast");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    if (!workspace) {
        set_error("Global workspace not enabled in brain config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Global workspace not enabled in workspace_has_broadcast");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Validate output parameter
    if (!has_broadcast) {
        set_error("NULL has_broadcast parameter");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL has_broadcast parameter");
        return NIMCP_ERROR_NULL_ARG;
    }

    // Check for broadcast
    *has_broadcast = global_workspace_has_broadcast(workspace);
    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_workspace_stats(
    nimcp_brain_t brain,
    uint32_t* total_broadcasts,
    uint32_t* total_competitions,
    float* avg_strength)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to workspace_stats");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL brain provided to workspace_stats");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain has NULL internal_brain in workspace_stats");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    if (!workspace) {
        set_error("Global workspace not enabled in brain config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Global workspace not enabled in workspace_stats");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Validate output parameters
    if (!total_broadcasts || !total_competitions || !avg_strength) {
        set_error("NULL output parameter in workspace_stats");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "NULL output parameter in workspace_stats");
        return NIMCP_ERROR_NULL_ARG;
    }

    // Get statistics
    workspace_statistics_t stats;
    bool success = global_workspace_get_statistics(workspace, &stats);

    if (success) {
        *total_broadcasts = (uint32_t)stats.total_broadcasts;
        *total_competitions = (uint32_t)stats.total_competitions;
        // Note: avg_strength not tracked in statistics, return 0.0
        *avg_strength = 0.0f;
        set_error("No error");
        return NIMCP_OK;
    } else {
        set_error("Failed to get workspace statistics");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to get workspace statistics");
        return NIMCP_ERROR;
    }
}

//=============================================================================
// Ethics API Implementation
//=============================================================================

nimcp_ethics_t nimcp_ethics_create(void) {
    nimcp_ethics_t handle = (nimcp_ethics_t)nimcp_malloc(sizeof(struct nimcp_ethics_handle));
    if (!handle) {
        set_error("Failed to allocate ethics handle");
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(struct nimcp_ethics_handle),
            "Failed to allocate ethics handle");
        return NULL;
    }

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
        set_error("Failed to create internal ethics engine");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to create internal ethics engine");
        nimcp_free(handle);
        return NULL;
    }

    set_error("No error");
    return handle;
}

void nimcp_ethics_destroy(nimcp_ethics_t ethics) {
    if (!ethics) {
        return;
    }

    if (ethics->internal_ethics) {
        ethics_engine_destroy(ethics->internal_ethics);
    }

    nimcp_free(ethics);
}

nimcp_status_t nimcp_ethics_check(
    nimcp_ethics_t ethics,
    const float* situation,
    uint32_t num_features,
    float* out_score)
{
    if (!ethics) {
        set_error("Ethics handle is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Ethics handle is NULL in ethics_check");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!situation) {
        set_error("Situation array is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Situation array is NULL in ethics_check");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!out_score) {
        set_error("Output score pointer is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Output score pointer is NULL in ethics_check");
        return NIMCP_ERROR_NULL_ARG;
    }

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

nimcp_knowledge_t nimcp_knowledge_create(void) {
    nimcp_knowledge_t handle = (nimcp_knowledge_t)nimcp_malloc(sizeof(struct nimcp_knowledge_handle));
    if (!handle) {
        set_error("Failed to allocate knowledge handle");
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(struct nimcp_knowledge_handle),
            "Failed to allocate knowledge handle");
        return NULL;
    }

    // Create internal knowledge system
    handle->internal_knowledge = knowledge_system_create("default");

    if (!handle->internal_knowledge) {
        set_error("Failed to create internal knowledge system");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to create internal knowledge system");
        nimcp_free(handle);
        return NULL;
    }

    set_error("No error");
    return handle;
}

void nimcp_knowledge_destroy(nimcp_knowledge_t knowledge) {
    if (!knowledge) {
        return;
    }

    if (knowledge->internal_knowledge) {
        knowledge_system_destroy(knowledge->internal_knowledge);
    }

    nimcp_free(knowledge);
}

nimcp_status_t nimcp_knowledge_add_fact(
    nimcp_knowledge_t knowledge,
    const char* subject,
    const char* predicate,
    const char* object)
{
    if (!knowledge) {
        set_error("Knowledge handle is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Knowledge handle is NULL in knowledge_add_fact");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!subject || !predicate || !object) {
        set_error("Subject, predicate, or object is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Subject, predicate, or object is NULL in knowledge_add_fact");
        return NIMCP_ERROR_NULL_ARG;
    }

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
        set_error("Failed to add knowledge item");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to add knowledge item '%s'", subject);
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_knowledge_query(
    nimcp_knowledge_t knowledge,
    const char* query,
    char* out_result,
    uint32_t max_result_len)
{
    if (!knowledge) {
        set_error("Knowledge handle is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Knowledge handle is NULL in knowledge_query");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!query) {
        set_error("Query is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Query is NULL in knowledge_query");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!out_result) {
        set_error("Output result buffer is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Output result buffer is NULL in knowledge_query");
        return NIMCP_ERROR_NULL_ARG;
    }

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
