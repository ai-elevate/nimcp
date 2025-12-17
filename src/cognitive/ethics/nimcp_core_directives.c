//=============================================================================
// nimcp_core_directives.c - Core Ethical Directives Implementation
//=============================================================================
/**
 * @file nimcp_core_directives.c
 * @brief Core ethical foundation implementation
 *
 * RESPONSIBILITY: Enforcing ethical constraints on all brain actions
 *
 * FUNCTIONS:
 * - core_directives_create() - Create directives system
 * - core_directives_destroy() - Destroy directives system
 * - core_directives_evaluate() - Evaluate action against directives
 * - core_directives_record_action() - Record action in history
 * - core_directives_connect_bio_async() - Connect to bio-async
 * - core_directives_connect_immune() - Connect to brain immune
 * - core_directives_connect_fep() - Connect to FEP orchestrator
 */

#include "cognitive/ethics/nimcp_core_directives.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

#include <string.h>
#include <stdio.h>
#include <math.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Action history entry
 */
typedef struct {
    float* action_vector;       // Action representation
    uint32_t action_dim;        // Vector dimensionality
    char context_desc[128];     // Human-readable description
    uint64_t timestamp_us;      // When action was executed
    bool valid;                 // Whether entry is populated
} action_history_entry_t;

/**
 * @brief Core directives system implementation
 */
struct core_directives_system {
    // Configuration
    core_directives_config_t config;

    // Action history for combinatorial analysis
    action_history_entry_t* action_history;
    uint32_t history_head;          // Next write position
    uint32_t history_count;         // Number of valid entries

    // Statistics
    core_directives_stats_t stats;

    // Integration
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    brain_immune_system_t* immune_system;
    directive_immune_bridge_t* immune_bridge;

    fep_orchestrator_t* fep_orchestrator;
    directive_fep_bridge_t* fep_bridge;

    // Thread safety
    nimcp_platform_mutex_t mutex;
};

/**
 * @brief Directive-immune bridge (placeholder)
 */
struct directive_immune_bridge {
    core_directives_system_t* directives;
    brain_immune_system_t* immune_system;
    bool enabled;
};

/**
 * @brief Directive-FEP bridge (placeholder)
 */
struct directive_fep_bridge {
    core_directives_system_t* directives;
    fep_orchestrator_t* fep_orchestrator;
    bool enabled;
};

//=============================================================================
// Default Configuration
//=============================================================================

void core_directives_default_config(core_directives_config_t* config)
{
    if (!config) return;

    memset(config, 0, sizeof(core_directives_config_t));

    // Asimov's Laws
    config->enable_first_law = true;
    config->enable_second_law = true;
    config->enable_third_law = true;

    // Golden Rule
    config->enable_golden_rule = true;
    config->reciprocity_threshold = 0.7f;

    // Combinatorial Harm
    config->enable_combinatorial_harm = true;
    config->action_history_size = 100;
    config->max_combination_depth = 5;

    // Thresholds
    config->harm_threshold = 0.3f;
    config->severity_threshold = 0.5f;
    config->confidence_threshold = 0.6f;

    // Integration
    config->enable_bio_async = true;
    config->enable_immune_integration = true;
    config->enable_fep_integration = true;
}

//=============================================================================
// System Lifecycle
//=============================================================================

core_directives_system_t* core_directives_create(const core_directives_config_t* config)
{
    if (!config) {
        NIMCP_LOGGING_ERROR("NULL config in core_directives_create");
        return NULL;
    }

    core_directives_system_t* directives = nimcp_calloc(1, sizeof(core_directives_system_t));
    if (!directives) {
        NIMCP_LOGGING_ERROR("Failed to allocate core directives system");
        return NULL;
    }

    // Copy configuration
    memcpy(&directives->config, config, sizeof(core_directives_config_t));

    // Allocate action history
    if (config->action_history_size > 0) {
        directives->action_history = nimcp_calloc(
            config->action_history_size,
            sizeof(action_history_entry_t)
        );
        if (!directives->action_history) {
            NIMCP_LOGGING_ERROR("Failed to allocate action history");
            nimcp_free(directives);
            return NULL;
        }
    }

    // Initialize mutex
    if (nimcp_platform_mutex_init(&directives->mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize directives mutex");
        nimcp_free(directives->action_history);
        nimcp_free(directives);
        return NULL;
    }

    directives->history_head = 0;
    directives->history_count = 0;
    directives->bio_async_enabled = false;
    directives->immune_system = NULL;
    directives->immune_bridge = NULL;
    directives->fep_orchestrator = NULL;
    directives->fep_bridge = NULL;

    NIMCP_LOGGING_INFO("Created core directives system (history=%u)",
                       config->action_history_size);

    return directives;
}

void core_directives_destroy(core_directives_system_t* directives)
{
    if (!directives) return;

    // Disconnect integrations
    if (directives->bio_async_enabled) {
        core_directives_disconnect_bio_async(directives);
    }

    // Destroy bridges
    if (directives->immune_bridge) {
        nimcp_free(directives->immune_bridge);
    }
    if (directives->fep_bridge) {
        nimcp_free(directives->fep_bridge);
    }

    // Free action history
    if (directives->action_history) {
        for (uint32_t i = 0; i < directives->config.action_history_size; i++) {
            if (directives->action_history[i].action_vector) {
                nimcp_free(directives->action_history[i].action_vector);
            }
        }
        nimcp_free(directives->action_history);
    }

    // Destroy mutex
    nimcp_platform_mutex_destroy(&directives->mutex);

    NIMCP_LOGGING_INFO("Destroyed core directives system");
    nimcp_free(directives);
}

//=============================================================================
// Action Evaluation (Stub Implementation)
//=============================================================================

int core_directives_evaluate(
    core_directives_system_t* directives,
    const float* action_vector,
    uint32_t action_dim,
    const char* context_desc,
    directive_evaluation_t* result)
{
    if (!directives || !action_vector || !result) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (action_dim == 0) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    nimcp_platform_mutex_lock(&directives->mutex);

    // Stub implementation: always allow for now
    // TODO: Implement Asimov's Laws, Golden Rule, Combinatorial Harm
    result->action = DIRECTIVE_ALLOW;
    result->violation = DIRECTIVE_VIOLATION_NONE;
    result->severity = 0.0f;
    result->confidence = 1.0f;
    snprintf(result->reason, sizeof(result->reason), "No violations detected (stub)");

    directives->stats.total_evaluations++;

    nimcp_platform_mutex_unlock(&directives->mutex);

    return 0;
}

int core_directives_record_action(
    core_directives_system_t* directives,
    const float* action_vector,
    uint32_t action_dim,
    const char* context_desc)
{
    if (!directives || !action_vector) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (action_dim == 0 || !directives->action_history) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    nimcp_platform_mutex_lock(&directives->mutex);

    // Get next history slot
    uint32_t idx = directives->history_head;
    action_history_entry_t* entry = &directives->action_history[idx];

    // Free old action vector if exists
    if (entry->action_vector) {
        nimcp_free(entry->action_vector);
    }

    // Allocate and copy action vector
    entry->action_vector = nimcp_malloc(action_dim * sizeof(float));
    if (!entry->action_vector) {
        nimcp_platform_mutex_unlock(&directives->mutex);
        return NIMCP_ERROR_NO_MEMORY;
    }

    memcpy(entry->action_vector, action_vector, action_dim * sizeof(float));
    entry->action_dim = action_dim;

    if (context_desc) {
        strncpy(entry->context_desc, context_desc, sizeof(entry->context_desc) - 1);
        entry->context_desc[sizeof(entry->context_desc) - 1] = '\0';
    } else {
        entry->context_desc[0] = '\0';
    }

    entry->timestamp_us = 0; // TODO: Get actual timestamp
    entry->valid = true;

    // Update circular buffer
    directives->history_head = (idx + 1) % directives->config.action_history_size;
    if (directives->history_count < directives->config.action_history_size) {
        directives->history_count++;
    }

    nimcp_platform_mutex_unlock(&directives->mutex);

    return 0;
}

int core_directives_clear_history(core_directives_system_t* directives)
{
    if (!directives) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(&directives->mutex);

    if (directives->action_history) {
        for (uint32_t i = 0; i < directives->config.action_history_size; i++) {
            if (directives->action_history[i].action_vector) {
                nimcp_free(directives->action_history[i].action_vector);
                directives->action_history[i].action_vector = NULL;
            }
            directives->action_history[i].valid = false;
        }
    }

    directives->history_head = 0;
    directives->history_count = 0;

    nimcp_platform_mutex_unlock(&directives->mutex);

    NIMCP_LOGGING_INFO("Cleared core directives action history");
    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int core_directives_connect_bio_async(core_directives_system_t* directives)
{
    if (!directives) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (directives->bio_async_enabled) {
        return 0; // Already connected
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_CORE_DIRECTIVES, // 0x1000
        .module_name = "core_directives",
        .inbox_capacity = 32,
        .user_data = directives
    };

    directives->bio_ctx = bio_router_register_module(&info);
    if (directives->bio_ctx) {
        directives->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Core directives connected to bio-async router");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available for core directives");
    }

    return 0;
}

int core_directives_disconnect_bio_async(core_directives_system_t* directives)
{
    if (!directives) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!directives->bio_async_enabled) {
        return 0; // Not connected
    }

    if (directives->bio_ctx) {
        bio_router_unregister_module(directives->bio_ctx);
        directives->bio_ctx = NULL;
    }

    directives->bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Core directives disconnected from bio-async router");

    return 0;
}

bool core_directives_is_bio_async_connected(const core_directives_system_t* directives)
{
    return directives ? directives->bio_async_enabled : false;
}

//=============================================================================
// Immune Integration (Stub)
//=============================================================================

int core_directives_connect_immune(
    core_directives_system_t* directives,
    brain_immune_system_t* immune)
{
    if (!directives || !immune) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (directives->immune_bridge) {
        NIMCP_LOGGING_WARN("Core directives already connected to immune system");
        return 0;
    }

    // Create bridge
    directive_immune_bridge_t* bridge = nimcp_calloc(1, sizeof(directive_immune_bridge_t));
    if (!bridge) {
        return NIMCP_ERROR_NO_MEMORY;
    }

    bridge->directives = directives;
    bridge->immune_system = immune;
    bridge->enabled = true;

    directives->immune_bridge = bridge;
    directives->immune_system = immune;

    NIMCP_LOGGING_INFO("Core directives connected to brain immune system");
    return 0;
}

//=============================================================================
// FEP Integration (Stub)
//=============================================================================

int core_directives_connect_fep(
    core_directives_system_t* directives,
    fep_orchestrator_t* fep_orch)
{
    if (!directives || !fep_orch) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (directives->fep_bridge) {
        NIMCP_LOGGING_WARN("Core directives already connected to FEP orchestrator");
        return 0;
    }

    // Create bridge
    directive_fep_bridge_t* bridge = nimcp_calloc(1, sizeof(directive_fep_bridge_t));
    if (!bridge) {
        return NIMCP_ERROR_NO_MEMORY;
    }

    bridge->directives = directives;
    bridge->fep_orchestrator = fep_orch;
    bridge->enabled = true;

    directives->fep_bridge = bridge;
    directives->fep_orchestrator = fep_orch;

    NIMCP_LOGGING_INFO("Core directives connected to FEP orchestrator");
    return 0;
}

//=============================================================================
// Statistics
//=============================================================================

int core_directives_get_stats(
    const core_directives_system_t* directives,
    core_directives_stats_t* stats)
{
    if (!directives || !stats) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)&directives->mutex);
    memcpy(stats, &directives->stats, sizeof(core_directives_stats_t));
    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)&directives->mutex);

    return 0;
}

int core_directives_reset_stats(core_directives_system_t* directives)
{
    if (!directives) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_platform_mutex_lock(&directives->mutex);
    memset(&directives->stats, 0, sizeof(core_directives_stats_t));
    nimcp_platform_mutex_unlock(&directives->mutex);

    NIMCP_LOGGING_INFO("Reset core directives statistics");
    return 0;
}
