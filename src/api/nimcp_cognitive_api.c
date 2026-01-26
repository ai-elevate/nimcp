/**
 * @file nimcp_cognitive_api.c
 * @brief Cognitive systems API - working memory, global workspace, resize
 *
 * This file contains working memory, global workspace, and dynamic brain resizing APIs.
 * Extracted from nimcp.c (SRP refactoring - lines 936-2014)
 */

#include "nimcp.h"
#include "api/nimcp_api_internal.h"
#include "core/brain/nimcp_brain.h"
#include "cognitive/nimcp_working_memory.h"
#include "cognitive/global_workspace/nimcp_global_workspace.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

#define LOG_MODULE "API.COGNITIVE"

#include <stddef.h>  /* for NULL */
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for cognitive_api module */
static nimcp_health_agent_t* g_cognitive_api_health_agent = NULL;

/**
 * @brief Set health agent for cognitive_api heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void cognitive_api_set_health_agent(nimcp_health_agent_t* agent) {
    g_cognitive_api_health_agent = agent;
}

/** @brief Send heartbeat from cognitive_api module */
static inline void cognitive_api_heartbeat(const char* operation, float progress) {
    if (g_cognitive_api_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_cognitive_api_health_agent, operation, progress);
    }
}


// External declarations from nimcp.c
extern void set_error(const char* fmt, ...);

//=============================================================================
// Phase 10.2: Working Memory API Implementation
//=============================================================================

/**
 * @brief Add item to brain's working memory
 *
 * WHAT: Wrapper for working_memory_add() on brain's working memory
 * WHY:  Provide public API for adding items to working memory
 * HOW:  Validate brain → Get working memory → Call internal API
 */
NIMCP_EXPORT nimcp_status_t nimcp_brain_working_memory_add(
    nimcp_brain_t brain,
    const float* data,
    uint32_t size,
    float salience)
{
    // Guard: Validate brain
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to working_memory_add");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    NIMCP_API_CHECK(wm != NULL, NIMCP_ERROR_INVALID, "Working memory not enabled in brain config");

    // Guard: Validate parameters
    NIMCP_API_CHECK_NULL(data, NIMCP_ERROR_NULL_ARG, "NULL data provided to working_memory_add");
    NIMCP_API_CHECK(size > 0, NIMCP_ERROR_INVALID, "Invalid size (0) provided to working_memory_add");

    // Add to working memory
    bool success = working_memory_add(wm, data, size, salience);
    if (!success) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to add item to working memory");
        set_error("Failed to add item to working memory");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_OK;
}

/**
 * @brief Get item from brain's working memory
 *
 * WHAT: Wrapper for working_memory_get() on brain's working memory
 * WHY:  Provide public API for accessing working memory items
 * HOW:  Validate brain → Get working memory → Call internal API
 */
NIMCP_EXPORT const float* nimcp_brain_working_memory_get(
    nimcp_brain_t brain,
    uint32_t index,
    uint32_t* size_out)
{
    // Guard: Validate brain
    NIMCP_API_CHECK_NULL_RET_NULL(brain, "Invalid brain handle");
    NIMCP_API_CHECK_NULL_RET_NULL(brain->internal_brain, "Brain has NULL internal_brain");

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    if (!wm) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Working memory not enabled");
        set_error("Working memory not enabled");
        return NULL;
    }

    // Get item
    const float* item = working_memory_get(wm, index, size_out);
    if (!item) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Invalid index %u or empty working memory", index);
        set_error("Invalid index or empty working memory");
        return NULL;
    }

    set_error("No error");
    return item;
}

/**
 * @brief Get brain's working memory statistics
 *
 * WHAT: Wrapper for working_memory_get_stats() on brain's working memory
 * WHY:  Provide public API for monitoring working memory state
 * HOW:  Validate brain → Get working memory → Extract size/capacity
 */
NIMCP_EXPORT nimcp_status_t nimcp_brain_working_memory_stats(
    nimcp_brain_t brain,
    uint32_t* current_size_out,
    uint32_t* capacity_out)
{
    // Guard: Validate brain
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Validate output parameters
    NIMCP_API_CHECK_NULL(current_size_out, NIMCP_ERROR_NULL_ARG, "NULL current_size_out parameter");
    NIMCP_API_CHECK_NULL(capacity_out, NIMCP_ERROR_NULL_ARG, "NULL capacity_out parameter");

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    NIMCP_API_CHECK(wm != NULL, NIMCP_ERROR_INVALID, "Working memory not enabled");

    // Get stats
    working_memory_stats_t stats;
    working_memory_get_stats(wm, &stats);

    *current_size_out = stats.current_size;
    *capacity_out = stats.capacity;

    set_error("No error");
    return NIMCP_OK;
}

/**
 * @brief Refresh item in brain's working memory
 *
 * WHAT: Wrapper for working_memory_refresh() on brain's working memory
 * WHY:  Provide public API for preventing decay (rehearsal)
 * HOW:  Validate brain → Get working memory → Call internal API
 */
NIMCP_EXPORT nimcp_status_t nimcp_brain_working_memory_refresh(
    nimcp_brain_t brain,
    uint32_t index)
{
    // Guard: Validate brain
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    NIMCP_API_CHECK(wm != NULL, NIMCP_ERROR_INVALID, "Working memory not enabled");

    // Refresh item
    bool success = working_memory_refresh(wm, index);
    if (!success) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Invalid index %u for working memory refresh", index);
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

NIMCP_EXPORT nimcp_status_t nimcp_brain_workspace_compete(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module,
    const float* content,
    uint32_t content_dim,
    float strength)
{
    // Guard: Validate brain
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_compete");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_API_CHECK(workspace != NULL, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Guard: Validate parameters
    NIMCP_API_CHECK_NULL(content, NIMCP_ERROR_NULL_ARG, "NULL content provided to workspace_compete");
    NIMCP_API_CHECK(content_dim > 0, NIMCP_ERROR_INVALID, "Invalid content_dim (0)");
    NIMCP_API_CHECK(strength >= 0.0f && strength <= 1.0f, NIMCP_ERROR_INVALID,
        "Strength must be in range [0.0, 1.0]");

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

NIMCP_EXPORT nimcp_status_t nimcp_brain_workspace_read(
    nimcp_brain_t brain,
    float* content,
    uint32_t max_dim,
    uint32_t* actual_dim,
    nimcp_cognitive_module_t* source_module)
{
    // Guard: Validate brain
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_read");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_API_CHECK(workspace != NULL, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Guard: Validate parameters
    NIMCP_API_CHECK_NULL(content, NIMCP_ERROR_NULL_ARG, "NULL content buffer provided to workspace_read");
    NIMCP_API_CHECK_NULL(actual_dim, NIMCP_ERROR_NULL_ARG, "NULL actual_dim provided to workspace_read");
    NIMCP_API_CHECK_NULL(source_module, NIMCP_ERROR_NULL_ARG, "NULL source_module provided to workspace_read");
    NIMCP_API_CHECK(max_dim > 0, NIMCP_ERROR_INVALID, "Invalid max_dim (0)");

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
        return NIMCP_ERROR;
    }
}

NIMCP_EXPORT nimcp_status_t nimcp_brain_workspace_subscribe(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module)
{
    // Guard: Validate brain
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_subscribe");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_API_CHECK(workspace != NULL, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Subscribe module
    cognitive_module_t internal_module = convert_module_enum(module);
    bool success = global_workspace_subscribe(workspace, internal_module);

    if (success) {
        set_error("No error");
        return NIMCP_OK;
    } else {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to subscribe module to workspace");
        set_error("Failed to subscribe module");
        return NIMCP_ERROR;
    }
}

NIMCP_EXPORT nimcp_status_t nimcp_brain_workspace_unsubscribe(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module)
{
    // Guard: Validate brain
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_unsubscribe");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_API_CHECK(workspace != NULL, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Unsubscribe module
    cognitive_module_t internal_module = convert_module_enum(module);
    bool success = global_workspace_unsubscribe(workspace, internal_module);

    if (success) {
        set_error("No error");
        return NIMCP_OK;
    } else {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to unsubscribe module from workspace");
        set_error("Failed to unsubscribe module");
        return NIMCP_ERROR;
    }
}

NIMCP_EXPORT nimcp_status_t nimcp_brain_workspace_has_broadcast(
    nimcp_brain_t brain,
    bool* has_broadcast)
{
    // Guard: Validate brain
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_has_broadcast");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_API_CHECK(workspace != NULL, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Guard: Validate output parameter
    NIMCP_API_CHECK_NULL(has_broadcast, NIMCP_ERROR_NULL_ARG, "NULL has_broadcast parameter");

    // Check for broadcast
    *has_broadcast = global_workspace_has_broadcast(workspace);
    set_error("No error");
    return NIMCP_OK;
}

NIMCP_EXPORT nimcp_status_t nimcp_brain_workspace_stats(
    nimcp_brain_t brain,
    uint32_t* total_broadcasts,
    uint32_t* total_competitions,
    float* avg_strength)
{
    // Guard: Validate brain
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_stats");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_API_CHECK(workspace != NULL, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Guard: Validate output parameters
    NIMCP_API_CHECK_NULL(total_broadcasts, NIMCP_ERROR_NULL_ARG, "NULL total_broadcasts parameter");
    NIMCP_API_CHECK_NULL(total_competitions, NIMCP_ERROR_NULL_ARG, "NULL total_competitions parameter");
    NIMCP_API_CHECK_NULL(avg_strength, NIMCP_ERROR_NULL_ARG, "NULL avg_strength parameter");

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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to get workspace statistics");
        set_error("Failed to get workspace statistics");
        return NIMCP_ERROR;
    }
}

//=============================================================================
// Phase 2.8: Dynamic Brain Resizing API
//=============================================================================

NIMCP_EXPORT bool nimcp_brain_resize(nimcp_brain_t brain, uint32_t new_neuron_count) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain handle is NULL in brain_resize");
        set_error("Brain handle is NULL");
        return false;
    }

    bool success = brain_resize(brain->internal_brain, new_neuron_count);
    if (!success) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to resize brain to %u neurons", new_neuron_count);
    }
    return success;
}

NIMCP_EXPORT bool nimcp_brain_auto_resize(nimcp_brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain handle is NULL in brain_auto_resize");
        set_error("Brain handle is NULL");
        return false;
    }

    bool success = brain_auto_resize(brain->internal_brain);
    if (!success) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to auto-resize brain");
    }
    return success;
}

NIMCP_EXPORT uint32_t nimcp_brain_get_neuron_count(nimcp_brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain handle is NULL in get_neuron_count");
        set_error("Brain handle is NULL");
        return 0;
    }

    return brain_get_neuron_count(brain->internal_brain);
}

NIMCP_EXPORT bool nimcp_brain_get_utilization_metrics(nimcp_brain_t brain, float* utilization, float* saturation) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain handle is NULL in get_utilization_metrics");
        set_error("Brain handle is NULL");
        return false;
    }

    if (!utilization || !saturation) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Output parameters are NULL in get_utilization_metrics");
        set_error("Output parameters are NULL");
        return false;
    }

    return brain_get_utilization_metrics(brain->internal_brain, utilization, saturation);
}
