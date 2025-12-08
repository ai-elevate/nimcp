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
#include <string.h>

#define LOG_MODULE "API.COGNITIVE"

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
    if (!brain) {
        set_error("NULL brain provided to working_memory_add");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    if (!wm) {
        set_error("Working memory not enabled in brain config");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Validate parameters
    if (!data) {
        set_error("NULL data provided to working_memory_add");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (size == 0) {
        set_error("Invalid size (0) provided to working_memory_add");
        return NIMCP_ERROR_INVALID;
    }

    // Add to working memory
    bool success = working_memory_add(wm, data, size, salience);
    if (!success) {
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
    if (!brain || !brain->internal_brain) {
        set_error("Invalid brain handle");
        return NULL;
    }

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    if (!wm) {
        set_error("Working memory not enabled");
        return NULL;
    }

    // Get item
    const float* item = working_memory_get(wm, index, size_out);
    if (!item) {
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
    if (!brain) {
        set_error("NULL brain provided");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Validate output parameters
    if (!current_size_out || !capacity_out) {
        set_error("NULL output parameters");
        return NIMCP_ERROR_NULL_ARG;
    }

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    if (!wm) {
        set_error("Working memory not enabled");
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
    if (!brain) {
        set_error("NULL brain provided");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    if (!wm) {
        set_error("Working memory not enabled");
        return NIMCP_ERROR_INVALID;
    }

    // Refresh item
    bool success = working_memory_refresh(wm, index);
    if (!success) {
        set_error("Invalid index for refresh");
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

NIMCP_EXPORT nimcp_status_t nimcp_brain_workspace_compete(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module,
    const float* content,
    uint32_t content_dim,
    float strength)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to workspace_compete");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    if (!workspace) {
        set_error("Global workspace not enabled in brain config");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Validate parameters
    if (!content) {
        set_error("NULL content provided to workspace_compete");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (content_dim == 0) {
        set_error("Invalid content_dim (0)");
        return NIMCP_ERROR_INVALID;
    }

    if (strength < 0.0f || strength > 1.0f) {
        set_error("Strength must be in range [0.0, 1.0]");
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

NIMCP_EXPORT nimcp_status_t nimcp_brain_workspace_read(
    nimcp_brain_t brain,
    float* content,
    uint32_t max_dim,
    uint32_t* actual_dim,
    nimcp_cognitive_module_t* source_module)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to workspace_read");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    if (!workspace) {
        set_error("Global workspace not enabled in brain config");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Validate parameters
    if (!content || !actual_dim || !source_module) {
        set_error("NULL output parameter provided to workspace_read");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (max_dim == 0) {
        set_error("Invalid max_dim (0)");
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
        return NIMCP_ERROR;
    }
}

NIMCP_EXPORT nimcp_status_t nimcp_brain_workspace_subscribe(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to workspace_subscribe");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    if (!workspace) {
        set_error("Global workspace not enabled in brain config");
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
        return NIMCP_ERROR;
    }
}

NIMCP_EXPORT nimcp_status_t nimcp_brain_workspace_unsubscribe(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to workspace_unsubscribe");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    if (!workspace) {
        set_error("Global workspace not enabled in brain config");
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
        return NIMCP_ERROR;
    }
}

NIMCP_EXPORT nimcp_status_t nimcp_brain_workspace_has_broadcast(
    nimcp_brain_t brain,
    bool* has_broadcast)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain provided to workspace_has_broadcast");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    if (!workspace) {
        set_error("Global workspace not enabled in brain config");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Validate output parameter
    if (!has_broadcast) {
        set_error("NULL has_broadcast parameter");
        return NIMCP_ERROR_NULL_ARG;
    }

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
    if (!brain) {
        set_error("NULL brain provided to workspace_stats");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    if (!workspace) {
        set_error("Global workspace not enabled in brain config");
        return NIMCP_ERROR_INVALID;
    }

    // Guard: Validate output parameters
    if (!total_broadcasts || !total_competitions || !avg_strength) {
        set_error("NULL output parameter in workspace_stats");
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
        return NIMCP_ERROR;
    }
}

//=============================================================================
// Phase 2.8: Dynamic Brain Resizing API
//=============================================================================

NIMCP_EXPORT bool nimcp_brain_resize(nimcp_brain_t brain, uint32_t new_neuron_count) {
    if (!brain) {
        set_error("Brain handle is NULL");
        return false;
    }

    return brain_resize(brain->internal_brain, new_neuron_count);
}

NIMCP_EXPORT bool nimcp_brain_auto_resize(nimcp_brain_t brain) {
    if (!brain) {
        set_error("Brain handle is NULL");
        return false;
    }

    return brain_auto_resize(brain->internal_brain);
}

NIMCP_EXPORT uint32_t nimcp_brain_get_neuron_count(nimcp_brain_t brain) {
    if (!brain) {
        set_error("Brain handle is NULL");
        return 0;
    }

    return brain_get_neuron_count(brain->internal_brain);
}

NIMCP_EXPORT bool nimcp_brain_get_utilization_metrics(nimcp_brain_t brain, float* utilization, float* saturation) {
    if (!brain) {
        set_error("Brain handle is NULL");
        return false;
    }

    if (!utilization || !saturation) {
        set_error("Output parameters are NULL");
        return false;
    }

    return brain_get_utilization_metrics(brain->internal_brain, utilization, saturation);
}
