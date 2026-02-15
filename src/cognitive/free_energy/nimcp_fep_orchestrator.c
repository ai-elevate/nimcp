/**
 * @file nimcp_fep_orchestrator.c
 * @brief FEP Orchestrator Implementation
 * @version 1.0.0
 * @date 2025-12-15
 */

#include "cognitive/free_energy/nimcp_fep_orchestrator.h"
#include "core/brain/nimcp_brain_internal.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/platform/nimcp_platform_time.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(fep_orchestrator)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_fep_orchestrator_mesh_id = 0;
static mesh_participant_registry_t* g_fep_orchestrator_mesh_registry = NULL;

nimcp_error_t fep_orchestrator_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_fep_orchestrator_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "fep_orchestrator", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "fep_orchestrator";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_fep_orchestrator_mesh_id);
    if (err == NIMCP_SUCCESS) g_fep_orchestrator_mesh_registry = registry;
    return err;
}

void fep_orchestrator_mesh_unregister(void) {
    if (g_fep_orchestrator_mesh_registry && g_fep_orchestrator_mesh_id != 0) {
        mesh_participant_unregister(g_fep_orchestrator_mesh_registry, g_fep_orchestrator_mesh_id);
        g_fep_orchestrator_mesh_id = 0;
        g_fep_orchestrator_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat from fep_orchestrator module (instance-level) */
static inline void fep_orchestrator_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_fep_orchestrator_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_fep_orchestrator_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_fep_orchestrator_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


/* ============================================================================
 * String Constants
 * ============================================================================ */

static const char* CATEGORY_NAMES[] = {
    "cognitive",
    "swarm",
    "security",
    "plasticity",
    "middleware",
    "perception",
    "async",
    "glial",
    "core",
    "jepa"
};

static const char* STATE_NAMES[] = {
    "stopped",
    "starting",
    "running",
    "paused",
    "stopping",
    "error"
};

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * @brief Find bridge entry by ID
 */
static fep_bridge_entry_t* find_bridge_by_id(
    fep_orchestrator_t* orchestrator,
    uint32_t bridge_id
) {
    if (!orchestrator) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");

        return NULL;

    }
    for (uint32_t i = 0; i < orchestrator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && orchestrator->bridge_count > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)orchestrator->bridge_count);
        }

        if (orchestrator->bridges[i].bridge_id == bridge_id) {
            return &orchestrator->bridges[i];
        }
    }
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "find_bridge_by_id: validation failed");
    return NULL;
}

/**
 * @brief Check if category is due for update
 */
static bool category_needs_update(
    const fep_orchestrator_t* orchestrator,
    fep_bridge_category_t category,
    uint64_t current_time_ms
) {
    const fep_category_config_t* cat_cfg = &orchestrator->config.categories[category];
    if (!cat_cfg->enabled) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "category_needs_update: cat_cfg->enabled is NULL");
        return false;
    }
    
    uint64_t elapsed = current_time_ms - cat_cfg->last_update_time;
    return elapsed >= cat_cfg->update_interval_ms;
}

/**
 * @brief Update single bridge and track statistics
 */
static int update_single_bridge(
    fep_orchestrator_t* orchestrator,
    fep_bridge_entry_t* entry,
    uint64_t current_time_ms
) {
    if (!entry->enabled || !entry->update_fn) return 0;

    /* Check if bridge's category is enabled */
    if (!orchestrator->config.categories[entry->category].enabled) return 0;

    uint64_t start_us = nimcp_platform_time_monotonic_us();

    int result = entry->update_fn(entry->handle);

    uint64_t elapsed_us = nimcp_platform_time_monotonic_us() - start_us;

    entry->last_update_time = current_time_ms;
    entry->update_count++;
    entry->total_update_time_us += elapsed_us;

    if (result != 0) {
        orchestrator->stats.update_errors++;
        if (orchestrator->config.enable_logging) {
            NIMCP_LOGGING_WARN("FEP bridge update failed: %s (id=%u)",
                entry->bridge_name, entry->bridge_id);
        }
    }

    return result == 0 ? 1 : 0;
}

/* ============================================================================
 * Lifecycle API Implementation
 * ============================================================================ */

int fep_orchestrator_default_config(fep_orchestrator_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_default_config", 0.0f);


    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_POINTER, "config is NULL");
    
    memset(config, 0, sizeof(fep_orchestrator_config_t));
    
    /* Global settings */
    config->max_bridges = FEP_ORCHESTRATOR_MAX_BRIDGES;
    config->enable_auto_update = true;
    config->global_update_interval_ms = 0;  /* Use per-category intervals */
    
    /* Integration enables */
    config->enable_bio_async = true;
    config->enable_brain_immune = true;
    config->enable_statistics = true;
    config->enable_logging = true;
    
    /* Performance tuning */
    config->max_updates_per_cycle = 0;  /* All bridges */
    config->update_time_budget_ms = 0;  /* Unlimited */
    
    /* Category-specific intervals (biologically-inspired timescales) */
    config->categories[FEP_BRIDGE_CATEGORY_COGNITIVE].enabled = true;
    config->categories[FEP_BRIDGE_CATEGORY_COGNITIVE].update_interval_ms = FEP_UPDATE_INTERVAL_COGNITIVE;
    
    config->categories[FEP_BRIDGE_CATEGORY_SWARM].enabled = true;
    config->categories[FEP_BRIDGE_CATEGORY_SWARM].update_interval_ms = FEP_UPDATE_INTERVAL_SWARM;
    
    config->categories[FEP_BRIDGE_CATEGORY_SECURITY].enabled = true;
    config->categories[FEP_BRIDGE_CATEGORY_SECURITY].update_interval_ms = FEP_UPDATE_INTERVAL_SECURITY;
    
    config->categories[FEP_BRIDGE_CATEGORY_PLASTICITY].enabled = true;
    config->categories[FEP_BRIDGE_CATEGORY_PLASTICITY].update_interval_ms = FEP_UPDATE_INTERVAL_PLASTICITY;
    
    config->categories[FEP_BRIDGE_CATEGORY_MIDDLEWARE].enabled = true;
    config->categories[FEP_BRIDGE_CATEGORY_MIDDLEWARE].update_interval_ms = FEP_UPDATE_INTERVAL_MIDDLEWARE;
    
    config->categories[FEP_BRIDGE_CATEGORY_PERCEPTION].enabled = true;
    config->categories[FEP_BRIDGE_CATEGORY_PERCEPTION].update_interval_ms = FEP_UPDATE_INTERVAL_PERCEPTION;
    
    config->categories[FEP_BRIDGE_CATEGORY_ASYNC].enabled = true;
    config->categories[FEP_BRIDGE_CATEGORY_ASYNC].update_interval_ms = FEP_UPDATE_INTERVAL_ASYNC;
    
    config->categories[FEP_BRIDGE_CATEGORY_GLIAL].enabled = true;
    config->categories[FEP_BRIDGE_CATEGORY_GLIAL].update_interval_ms = FEP_UPDATE_INTERVAL_GLIAL;
    
    config->categories[FEP_BRIDGE_CATEGORY_CORE].enabled = true;
    config->categories[FEP_BRIDGE_CATEGORY_CORE].update_interval_ms = FEP_UPDATE_INTERVAL_CORE;

    config->categories[FEP_BRIDGE_CATEGORY_JEPA].enabled = true;
    config->categories[FEP_BRIDGE_CATEGORY_JEPA].update_interval_ms = FEP_UPDATE_INTERVAL_JEPA;

    return 0;
}

fep_orchestrator_t* fep_orchestrator_create(const fep_orchestrator_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_create", 0.0f);


    fep_orchestrator_t* orchestrator = (fep_orchestrator_t*)nimcp_calloc(1, sizeof(fep_orchestrator_t));
    NIMCP_API_CHECK_ALLOC(orchestrator, "Failed to allocate FEP orchestrator");
    
    /* Apply configuration */
    if (config) {
        orchestrator->config = *config;
    } else {
        fep_orchestrator_default_config(&orchestrator->config);
    }
    
    /* Allocate bridge registry */
    orchestrator->bridge_capacity = orchestrator->config.max_bridges;
    orchestrator->bridges = (fep_bridge_entry_t*)nimcp_calloc(
        orchestrator->bridge_capacity, sizeof(fep_bridge_entry_t));
    if (!orchestrator->bridges) {
        nimcp_free(orchestrator);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "fep_orchestrator_create: orchestrator->bridges is NULL");
        return NULL;
    }
    
    /* Initialize mutex */
    orchestrator->mutex = nimcp_platform_mutex_create();
    if (!orchestrator->mutex) {
        nimcp_free(orchestrator->bridges);
        nimcp_free(orchestrator);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "fep_orchestrator_create: orchestrator->mutex is NULL");
        return NULL;
    }
    
    /* Initialize state */
    orchestrator->state = FEP_ORCHESTRATOR_STOPPED;
    orchestrator->bridge_count = 0;
    orchestrator->next_bridge_id = 1;
    orchestrator->bio_async_connected = false;
    orchestrator->immune_connected = false;
    
    /* Initialize statistics */
    memset(&orchestrator->stats, 0, sizeof(fep_orchestrator_stats_t));
    
    if (orchestrator->config.enable_logging) {
        NIMCP_LOGGING_INFO("FEP orchestrator created: max_bridges=%u", 
            orchestrator->bridge_capacity);
    }
    
    return orchestrator;
}

void fep_orchestrator_destroy(fep_orchestrator_t* orchestrator) {
    if (!orchestrator) return;
    
    /* Stop if running */
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_destroy", 0.0f);


    if (orchestrator->state == FEP_ORCHESTRATOR_RUNNING) {
        fep_orchestrator_stop(orchestrator);
    }
    
    /* Disconnect integrations */
    if (orchestrator->bio_async_connected) {
        fep_orchestrator_disconnect_bio_async(orchestrator);
    }
    if (orchestrator->immune_connected) {
        fep_orchestrator_disconnect_brain_immune(orchestrator);
    }
    
    /* Destroy bridges if destroy_fn provided */
    for (uint32_t i = 0; i < orchestrator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && orchestrator->bridge_count > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)orchestrator->bridge_count);
        }

        fep_bridge_entry_t* entry = &orchestrator->bridges[i];
        if (entry->destroy_fn && entry->handle) {
            entry->destroy_fn(entry->handle);
            entry->handle = NULL;
        }
    }
    
    /* Free resources */
    if (orchestrator->mutex) {
        nimcp_platform_mutex_destroy(orchestrator->mutex);
        nimcp_free(orchestrator->mutex);
        orchestrator->mutex = NULL;
    }
    if (orchestrator->bridges) {
        nimcp_free(orchestrator->bridges);
    }
    
    if (orchestrator->config.enable_logging) {
        NIMCP_LOGGING_INFO("FEP orchestrator destroyed");
    }
    
    nimcp_free(orchestrator);
}

int fep_orchestrator_start(fep_orchestrator_t* orchestrator) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_start", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    if (orchestrator->state == FEP_ORCHESTRATOR_RUNNING) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return 0;  /* Already running */
    }
    
    orchestrator->state = FEP_ORCHESTRATOR_STARTING;
    orchestrator->start_time = nimcp_platform_time_monotonic_ms();
    orchestrator->last_update_time = orchestrator->start_time;
    
    /* Initialize category last update times */
    for (int i = 0; i < FEP_BRIDGE_CATEGORY_COUNT; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && FEP_BRIDGE_CATEGORY_COUNT > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)FEP_BRIDGE_CATEGORY_COUNT);
        }

        orchestrator->config.categories[i].last_update_time = orchestrator->start_time;
    }
    
    orchestrator->state = FEP_ORCHESTRATOR_RUNNING;
    
    if (orchestrator->config.enable_logging) {
        NIMCP_LOGGING_INFO("FEP orchestrator started: bridges=%u", orchestrator->bridge_count);
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

int fep_orchestrator_stop(fep_orchestrator_t* orchestrator) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_stop", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    if (orchestrator->state == FEP_ORCHESTRATOR_STOPPED) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return 0;  /* Already stopped */
    }
    
    orchestrator->state = FEP_ORCHESTRATOR_STOPPING;
    orchestrator->state = FEP_ORCHESTRATOR_STOPPED;
    
    if (orchestrator->config.enable_logging) {
        NIMCP_LOGGING_INFO("FEP orchestrator stopped");
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

int fep_orchestrator_pause(fep_orchestrator_t* orchestrator) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_pause", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    if (orchestrator->state != FEP_ORCHESTRATOR_RUNNING) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return NIMCP_ERROR_INVALID_STATE;
    }
    
    orchestrator->state = FEP_ORCHESTRATOR_PAUSED;
    
    if (orchestrator->config.enable_logging) {
        NIMCP_LOGGING_INFO("FEP orchestrator paused");
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

int fep_orchestrator_resume(fep_orchestrator_t* orchestrator) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_resume", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    if (orchestrator->state != FEP_ORCHESTRATOR_PAUSED) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return NIMCP_ERROR_INVALID_STATE;
    }
    
    /* Reset last update times to prevent immediate burst of updates */
    uint64_t now = nimcp_platform_time_monotonic_ms();
    for (int i = 0; i < FEP_BRIDGE_CATEGORY_COUNT; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && FEP_BRIDGE_CATEGORY_COUNT > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)FEP_BRIDGE_CATEGORY_COUNT);
        }

        orchestrator->config.categories[i].last_update_time = now;
    }
    
    orchestrator->state = FEP_ORCHESTRATOR_RUNNING;
    
    if (orchestrator->config.enable_logging) {
        NIMCP_LOGGING_INFO("FEP orchestrator resumed");
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

/* ============================================================================
 * Bridge Registration API Implementation
 * ============================================================================ */

int fep_orchestrator_register_bridge(
    fep_orchestrator_t* orchestrator,
    const char* name,
    fep_bridge_category_t category,
    fep_bridge_handle_t handle,
    fep_bridge_update_fn_t update_fn,
    fep_bridge_destroy_fn_t destroy_fn,
    uint32_t* bridge_id_out
) {
    if (!orchestrator || !name || !handle || !update_fn) {
        return NIMCP_ERROR_NULL_POINTER;
    }
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_register_bridge", 0.0f);


    if (category >= FEP_BRIDGE_CATEGORY_COUNT) {
        return NIMCP_ERROR_INVALID_PARAM;
    }
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    if (orchestrator->bridge_count >= orchestrator->bridge_capacity) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return NIMCP_ERROR_NO_MEMORY;
    }
    
    /* Create entry */
    fep_bridge_entry_t* entry = &orchestrator->bridges[orchestrator->bridge_count];
    entry->bridge_id = orchestrator->next_bridge_id++;
    entry->bridge_name = name;
    entry->category = category;
    entry->handle = handle;
    entry->update_fn = update_fn;
    entry->destroy_fn = destroy_fn;
    entry->last_update_time = 0;
    entry->update_count = 0;
    entry->total_update_time_us = 0;
    entry->enabled = true;
    
    orchestrator->bridge_count++;
    orchestrator->stats.total_bridges++;
    orchestrator->stats.active_bridges++;
    orchestrator->stats.categories[category].bridge_count++;
    
    if (bridge_id_out) {
        *bridge_id_out = entry->bridge_id;
    }
    
    if (orchestrator->config.enable_logging) {
        NIMCP_LOGGING_DEBUG("Registered FEP bridge: %s (id=%u, category=%s)", 
            name, entry->bridge_id, CATEGORY_NAMES[category]);
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

int fep_orchestrator_unregister_bridge(
    fep_orchestrator_t* orchestrator,
    uint32_t bridge_id
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_unregister_bridge", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    /* Find bridge */
    int found_idx = -1;
    for (uint32_t i = 0; i < orchestrator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && orchestrator->bridge_count > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)orchestrator->bridge_count);
        }

        if (orchestrator->bridges[i].bridge_id == bridge_id) {
            found_idx = (int)i;
            break;
        }
    }
    
    if (found_idx < 0) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }
    
    fep_bridge_entry_t* entry = &orchestrator->bridges[found_idx];
    fep_bridge_category_t category = entry->category;
    
    /* Update stats */
    orchestrator->stats.total_bridges--;
    if (entry->enabled) {
        orchestrator->stats.active_bridges--;
    }
    orchestrator->stats.categories[category].bridge_count--;
    
    /* Shift remaining entries */
    for (uint32_t i = (uint32_t)found_idx; i < orchestrator->bridge_count - 1; i++) {
        orchestrator->bridges[i] = orchestrator->bridges[i + 1];
    }
    orchestrator->bridge_count--;
    
    if (orchestrator->config.enable_logging) {
        NIMCP_LOGGING_DEBUG("Unregistered FEP bridge: id=%u", bridge_id);
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

int fep_orchestrator_set_bridge_enabled(
    fep_orchestrator_t* orchestrator,
    uint32_t bridge_id,
    bool enabled
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_set_bridge_enabled", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    fep_bridge_entry_t* entry = find_bridge_by_id(orchestrator, bridge_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }
    
    if (entry->enabled != enabled) {
        entry->enabled = enabled;
        if (enabled) {
            orchestrator->stats.active_bridges++;
        } else {
            orchestrator->stats.active_bridges--;
        }
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

const fep_bridge_entry_t* fep_orchestrator_get_bridge(
    const fep_orchestrator_t* orchestrator,
    uint32_t bridge_id
) {
    if (!orchestrator) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");

        return NULL;

    }
    
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_get_bridge", 0.0f);


    for (uint32_t i = 0; i < orchestrator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && orchestrator->bridge_count > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)orchestrator->bridge_count);
        }

        if (orchestrator->bridges[i].bridge_id == bridge_id) {
            return &orchestrator->bridges[i];
        }
    }
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "fep_orchestrator_get_bridge: validation failed");
    return NULL;
}

uint32_t fep_orchestrator_get_bridges_by_category(
    const fep_orchestrator_t* orchestrator,
    fep_bridge_category_t category,
    const fep_bridge_entry_t** bridges,
    uint32_t max_bridges
) {
    if (!orchestrator || !bridges || category >= FEP_BRIDGE_CATEGORY_COUNT) return 0;
    
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_get_bridges_by_categ", 0.0f);


    uint32_t count = 0;
    for (uint32_t i = 0; i < orchestrator->bridge_count && count < max_bridges; i++) {
        if (orchestrator->bridges[i].category == category) {
            bridges[count++] = &orchestrator->bridges[i];
        }
    }
    return count;
}

/* ============================================================================
 * Update API Implementation
 * ============================================================================ */

int fep_orchestrator_update(
    fep_orchestrator_t* orchestrator,
    uint64_t current_time_ms
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_update", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    if (orchestrator->state != FEP_ORCHESTRATOR_RUNNING) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return 0;
    }
    
    uint64_t cycle_start_us = nimcp_platform_time_monotonic_us();
    int total_updated = 0;
    
    /* Process bio-async inbox if connected */
    if (orchestrator->bio_async_connected && orchestrator->bio_context) {
        uint32_t processed = bio_router_process_inbox(orchestrator->bio_context, 10);
        orchestrator->stats.bio_async_messages_received += processed;
    }
    
    /* Update each category that is due */
    for (int cat = 0; cat < FEP_BRIDGE_CATEGORY_COUNT; cat++) {
        /* Phase 8: Loop progress heartbeat */
        if ((cat & 0xFF) == 0 && FEP_BRIDGE_CATEGORY_COUNT > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(cat + 1) / (float)FEP_BRIDGE_CATEGORY_COUNT);
        }

        if (!category_needs_update(orchestrator, (fep_bridge_category_t)cat, current_time_ms)) {
            continue;
        }
        
        /* Update all bridges in this category */
        for (uint32_t i = 0; i < orchestrator->bridge_count; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && orchestrator->bridge_count > 256) {
                fep_orchestrator_heartbeat("fep_orchestr_loop",
                                 (float)(i + 1) / (float)orchestrator->bridge_count);
            }

            fep_bridge_entry_t* entry = &orchestrator->bridges[i];
            if (entry->category != cat) continue;
            
            int result = update_single_bridge(orchestrator, entry, current_time_ms);
            if (result > 0) {
                total_updated++;
                orchestrator->stats.total_bridge_updates++;
                orchestrator->stats.categories[cat].total_updates++;
            }
        }
        
        /* Update category last update time */
        orchestrator->config.categories[cat].last_update_time = current_time_ms;
    }
    
    /* Update orchestrator statistics */
    uint64_t cycle_time_us = nimcp_platform_time_monotonic_us() - cycle_start_us;
    orchestrator->stats.total_update_cycles++;
    
    float cycle_time_ms = (float)cycle_time_us / 1000.0f;
    if (orchestrator->stats.max_cycle_time_us < (float)cycle_time_us) {
        orchestrator->stats.max_cycle_time_us = (float)cycle_time_us;
    }
    
    /* Rolling average */
    uint64_t n = orchestrator->stats.total_update_cycles;
    orchestrator->stats.avg_cycle_time_us = 
        (orchestrator->stats.avg_cycle_time_us * (n - 1) + (float)cycle_time_us) / n;
    
    /* Check for time budget overrun */
    if (orchestrator->config.update_time_budget_ms > 0 &&
        cycle_time_ms > orchestrator->config.update_time_budget_ms) {
        orchestrator->stats.overrun_count++;
    }
    
    orchestrator->last_update_time = current_time_ms;
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return total_updated;
}

int fep_orchestrator_update_category(
    fep_orchestrator_t* orchestrator,
    fep_bridge_category_t category,
    uint64_t current_time_ms
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_update_category", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    NIMCP_CHECK_THROW(category < FEP_BRIDGE_CATEGORY_COUNT, NIMCP_ERROR_INVALID_PARAM, "invalid bridge category");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    if (orchestrator->state != FEP_ORCHESTRATOR_RUNNING) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return 0;
    }
    
    int updated = 0;
    for (uint32_t i = 0; i < orchestrator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && orchestrator->bridge_count > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)orchestrator->bridge_count);
        }

        fep_bridge_entry_t* entry = &orchestrator->bridges[i];
        if (entry->category != category) continue;
        
        int result = update_single_bridge(orchestrator, entry, current_time_ms);
        if (result > 0) {
            updated++;
            orchestrator->stats.total_bridge_updates++;
            orchestrator->stats.categories[category].total_updates++;
        }
    }
    
    orchestrator->config.categories[category].last_update_time = current_time_ms;
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return updated;
}

int fep_orchestrator_update_bridge(
    fep_orchestrator_t* orchestrator,
    uint32_t bridge_id
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_update_bridge", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    fep_bridge_entry_t* entry = find_bridge_by_id(orchestrator, bridge_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }
    
    uint64_t current_time_ms = nimcp_platform_time_monotonic_ms();
    int result = update_single_bridge(orchestrator, entry, current_time_ms);
    
    if (result > 0) {
        orchestrator->stats.total_bridge_updates++;
        orchestrator->stats.categories[entry->category].total_updates++;
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return result > 0 ? 0 : NIMCP_ERROR_OPERATION_FAILED;
}

int fep_orchestrator_force_update_all(fep_orchestrator_t* orchestrator) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_force_update_all", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");

    nimcp_platform_mutex_lock(orchestrator->mutex);

    /* Check if orchestrator is in a state that allows updates */
    if (orchestrator->state != FEP_ORCHESTRATOR_RUNNING) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return 0;  /* No updates when not running */
    }

    uint64_t cycle_start_us = nimcp_platform_time_monotonic_us();
    uint64_t current_time_ms = nimcp_platform_time_monotonic_ms();
    int updated = 0;

    for (uint32_t i = 0; i < orchestrator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && orchestrator->bridge_count > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)orchestrator->bridge_count);
        }

        fep_bridge_entry_t* entry = &orchestrator->bridges[i];
        int result = update_single_bridge(orchestrator, entry, current_time_ms);
        if (result > 0) {
            updated++;
            orchestrator->stats.total_bridge_updates++;
            orchestrator->stats.categories[entry->category].total_updates++;
        }
    }

    /* Reset all category timers */
    for (int i = 0; i < FEP_BRIDGE_CATEGORY_COUNT; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && FEP_BRIDGE_CATEGORY_COUNT > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)FEP_BRIDGE_CATEGORY_COUNT);
        }

        orchestrator->config.categories[i].last_update_time = current_time_ms;
    }

    /* Track cycle time statistics */
    uint64_t cycle_time_us = nimcp_platform_time_monotonic_us() - cycle_start_us;
    uint32_t n = orchestrator->stats.total_update_cycles + 1;
    orchestrator->stats.total_update_cycles = n;
    orchestrator->stats.avg_cycle_time_us =
        (orchestrator->stats.avg_cycle_time_us * (n - 1) + (float)cycle_time_us) / n;
    if ((float)cycle_time_us > orchestrator->stats.max_cycle_time_us) {
        orchestrator->stats.max_cycle_time_us = (float)cycle_time_us;
    }

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return updated;
}

/* ============================================================================
 * Category Configuration API Implementation
 * ============================================================================ */

int fep_orchestrator_set_update_interval(
    fep_orchestrator_t* orchestrator,
    fep_bridge_category_t category,
    uint64_t interval_ms
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_set_update_interval", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    NIMCP_CHECK_THROW(category < FEP_BRIDGE_CATEGORY_COUNT, NIMCP_ERROR_INVALID_PARAM, "invalid bridge category");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    orchestrator->config.categories[category].update_interval_ms = interval_ms;
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    
    return 0;
}

int fep_orchestrator_set_category_enabled(
    fep_orchestrator_t* orchestrator,
    fep_bridge_category_t category,
    bool enabled
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_set_category_enabled", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    NIMCP_CHECK_THROW(category < FEP_BRIDGE_CATEGORY_COUNT, NIMCP_ERROR_INVALID_PARAM, "invalid bridge category");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    orchestrator->config.categories[category].enabled = enabled;
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    
    return 0;
}

int fep_orchestrator_get_category_config(
    const fep_orchestrator_t* orchestrator,
    fep_bridge_category_t category,
    fep_category_config_t* config
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_get_category_config", 0.0f);


    NIMCP_CHECK_THROW(orchestrator && config, NIMCP_ERROR_NULL_POINTER, "orchestrator or config is NULL");
    NIMCP_CHECK_THROW(category < FEP_BRIDGE_CATEGORY_COUNT, NIMCP_ERROR_INVALID_PARAM, "invalid bridge category");
    
    *config = orchestrator->config.categories[category];
    return 0;
}

/* ============================================================================
 * Integration API Implementation
 * ============================================================================ */

int fep_orchestrator_connect_brain_immune(
    fep_orchestrator_t* orchestrator,
    brain_immune_system_t* immune
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_connect_brain_immune", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    orchestrator->brain_immune = immune;
    orchestrator->immune_connected = (immune != NULL);
    
    if (orchestrator->config.enable_logging && immune) {
        NIMCP_LOGGING_INFO("FEP orchestrator connected to brain immune system");
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

int fep_orchestrator_disconnect_brain_immune(fep_orchestrator_t* orchestrator) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_disconnect_brain_imm", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    orchestrator->brain_immune = NULL;
    orchestrator->immune_connected = false;
    
    if (orchestrator->config.enable_logging) {
        NIMCP_LOGGING_INFO("FEP orchestrator disconnected from brain immune system");
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

int fep_orchestrator_connect_bio_async(fep_orchestrator_t* orchestrator) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_connect_bio_async", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    if (!bio_router_is_initialized()) {
        if (orchestrator->config.enable_logging) {
            NIMCP_LOGGING_WARN("Bio-async router not initialized, skipping FEP orchestrator registration");
        }
        return 0;  /* Non-fatal */
    }
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    if (orchestrator->bio_async_connected) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return 0;  /* Already connected */
    }
    
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_ORCHESTRATOR,
        .module_name = FEP_ORCHESTRATOR_MODULE_NAME,
        .inbox_capacity = 64,
        .user_data = orchestrator
    };
    
    orchestrator->bio_context = bio_router_register_module(&info);
    if (orchestrator->bio_context) {
        orchestrator->bio_async_connected = true;
        if (orchestrator->config.enable_logging) {
            NIMCP_LOGGING_INFO("FEP orchestrator connected to bio-async router");
        }
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

int fep_orchestrator_disconnect_bio_async(fep_orchestrator_t* orchestrator) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_disconnect_bio_async", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");

    nimcp_platform_mutex_lock(orchestrator->mutex);

    if (orchestrator->bio_async_connected && orchestrator->bio_context) {
        bio_router_unregister_module(orchestrator->bio_context);
        orchestrator->bio_context = NULL;
        orchestrator->bio_async_connected = false;

        if (orchestrator->config.enable_logging) {
            NIMCP_LOGGING_INFO("FEP orchestrator disconnected from bio-async router");
        }
    }

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

/* ============================================================================
 * Internal Knowledge Graph Integration
 * ============================================================================ */

int fep_orchestrator_connect_internal_kg(
    fep_orchestrator_t* orchestrator,
    brain_t brain)
{
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_connect_internal_kg", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");

    nimcp_platform_mutex_lock(orchestrator->mutex);

    /* Initialize KG context */
    if (kg_module_init(&orchestrator->kg_context, brain, "fep_orchestrator") != 0) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_INITIALIZED, "fep_orchestrator_connect_internal_kg: validation failed");
        return -1;
    }

    orchestrator->kg_connected = orchestrator->kg_context.kg_available;

    if (orchestrator->kg_connected) {
        /* Update node state to active */
        if (kg_has_node(&orchestrator->kg_context)) {
            uint64_t admin_token = brain ? brain->internal_kg_admin_token : 0;
            kg_module_update_state(&orchestrator->kg_context,
                                   BRAIN_KG_STATE_ACTIVE, admin_token);
            kg_module_set_ptr(&orchestrator->kg_context, orchestrator, admin_token);
        }

        if (orchestrator->config.enable_logging) {
            NIMCP_LOGGING_INFO("FEP orchestrator connected to internal KG");
        }
    }

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

int fep_orchestrator_disconnect_internal_kg(fep_orchestrator_t* orchestrator) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_disconnect_internal_", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");

    nimcp_platform_mutex_lock(orchestrator->mutex);

    if (orchestrator->kg_connected) {
        orchestrator->kg_context.kg = NULL;
        orchestrator->kg_context.kg_available = false;
        orchestrator->kg_connected = false;

        if (orchestrator->config.enable_logging) {
            NIMCP_LOGGING_INFO("FEP orchestrator disconnected from internal KG");
        }
    }

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return 0;
}

uint32_t fep_orchestrator_get_bridges_for_module(
    const fep_orchestrator_t* orchestrator,
    const char* module_name,
    const fep_bridge_entry_t** bridges,
    uint32_t max_bridges)
{
    if (!orchestrator || !module_name || !bridges || max_bridges == 0) {
        return 0;
    }

    /* If KG not connected, fall back to searching all bridges by name */
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_get_bridges_for_modu", 0.0f);


    fep_orchestrator_t* mutable_orch = (fep_orchestrator_t*)orchestrator;
    nimcp_platform_mutex_lock(mutable_orch->mutex);

    uint32_t count = 0;

    if (orchestrator->kg_connected && kg_is_available(&orchestrator->kg_context)) {
        /* Query KG for module node */
        brain_kg_node_id_t module_node = kg_find_node_safe(
            &orchestrator->kg_context, module_name);

        if (module_node != BRAIN_KG_INVALID_NODE) {
            /* Find bridges connected to this module */
            for (uint32_t i = 0; i < orchestrator->bridge_count && count < max_bridges; i++) {
                const fep_bridge_entry_t* entry = &orchestrator->bridges[i];
                if (!entry->enabled) continue;

                /* Check if bridge is connected to module via KG */
                if (entry->kg_node_id != BRAIN_KG_INVALID_NODE) {
                    if (brain_kg_are_connected(orchestrator->kg_context.kg,
                                               entry->kg_node_id, module_node)) {
                        bridges[count++] = entry;
                    }
                }
            }
        }
    } else {
        /* Fallback: search bridges by name substring */
        for (uint32_t i = 0; i < orchestrator->bridge_count && count < max_bridges; i++) {
            const fep_bridge_entry_t* entry = &orchestrator->bridges[i];
            if (!entry->enabled) continue;

            if (entry->bridge_name && strstr(entry->bridge_name, module_name)) {
                bridges[count++] = entry;
            }
        }
    }

    nimcp_platform_mutex_unlock(mutable_orch->mutex);
    return count;
}

int fep_orchestrator_get_topology_summary(
    const fep_orchestrator_t* orchestrator,
    char* summary,
    size_t summary_size)
{
    if (!orchestrator || !summary || summary_size == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "fep_orchestrator_get_topology_summary: required parameter is NULL (orchestrator, summary)");
        return -1;
    }

    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_get_topology_summary", 0.0f);


    fep_orchestrator_t* mutable_orch = (fep_orchestrator_t*)orchestrator;
    nimcp_platform_mutex_lock(mutable_orch->mutex);

    int written = 0;

    if (orchestrator->kg_connected && kg_is_available(&orchestrator->kg_context)) {
        /* Get neighbors via KG */
        brain_kg_node_list_t* neighbors = kg_get_neighbors_safe(&orchestrator->kg_context);
        if (neighbors) {
            written = snprintf(summary, summary_size,
                "FEP Orchestrator Topology:\n"
                "  Bridges: %u/%u\n"
                "  KG Neighbors: %u\n"
                "  State: %s\n",
                orchestrator->stats.active_bridges,
                orchestrator->bridge_count,
                neighbors->count,
                orchestrator->state == FEP_ORCHESTRATOR_RUNNING ? "Running" : "Stopped");

            brain_kg_node_list_destroy(neighbors);
        }
    } else {
        written = snprintf(summary, summary_size,
            "FEP Orchestrator Topology:\n"
            "  Bridges: %u/%u\n"
            "  KG: Not connected\n"
            "  State: %s\n",
            orchestrator->stats.active_bridges,
            orchestrator->bridge_count,
            orchestrator->state == FEP_ORCHESTRATOR_RUNNING ? "Running" : "Stopped");
    }

    nimcp_platform_mutex_unlock(mutable_orch->mutex);
    return written;
}

/* ============================================================================
 * Statistics and Monitoring API Implementation
 * ============================================================================ */

int fep_orchestrator_get_stats(
    const fep_orchestrator_t* orchestrator,
    fep_orchestrator_stats_t* stats
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_get_stats", 0.0f);


    NIMCP_CHECK_THROW(orchestrator && stats, NIMCP_ERROR_NULL_POINTER, "orchestrator or stats is NULL");

    /* Thread-safe copy under lock for consistency */
    fep_orchestrator_t* mutable_orch = (fep_orchestrator_t*)orchestrator;
    nimcp_platform_mutex_lock(mutable_orch->mutex);
    *stats = orchestrator->stats;
    nimcp_platform_mutex_unlock(mutable_orch->mutex);

    return 0;
}

void fep_orchestrator_reset_stats(fep_orchestrator_t* orchestrator) {
    if (!orchestrator) return;
    
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_reset_stats", 0.0f);


    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    /* Preserve bridge counts */
    uint32_t total = orchestrator->stats.total_bridges;
    uint32_t active = orchestrator->stats.active_bridges;
    uint32_t cat_counts[FEP_BRIDGE_CATEGORY_COUNT];
    for (int i = 0; i < FEP_BRIDGE_CATEGORY_COUNT; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && FEP_BRIDGE_CATEGORY_COUNT > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)FEP_BRIDGE_CATEGORY_COUNT);
        }

        cat_counts[i] = orchestrator->stats.categories[i].bridge_count;
    }
    
    memset(&orchestrator->stats, 0, sizeof(fep_orchestrator_stats_t));
    
    orchestrator->stats.total_bridges = total;
    orchestrator->stats.active_bridges = active;
    for (int i = 0; i < FEP_BRIDGE_CATEGORY_COUNT; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && FEP_BRIDGE_CATEGORY_COUNT > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)FEP_BRIDGE_CATEGORY_COUNT);
        }

        orchestrator->stats.categories[i].bridge_count = cat_counts[i];
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
}

float fep_orchestrator_get_load(const fep_orchestrator_t* orchestrator) {
    if (!orchestrator) return 0.0f;
    
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_get_load", 0.0f);


    if (orchestrator->config.update_time_budget_ms <= 0) {
        return 0.0f;  /* No budget = no load metric */
    }
    
    float avg_cycle_ms = orchestrator->stats.avg_cycle_time_us / 1000.0f;
    return avg_cycle_ms / orchestrator->config.update_time_budget_ms;
}

fep_orchestrator_state_t fep_orchestrator_get_state(
    const fep_orchestrator_t* orchestrator
) {
    if (!orchestrator) return FEP_ORCHESTRATOR_STOPPED;
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_get_state", 0.0f);


    return orchestrator->state;
}

/* ============================================================================
 * String Conversion API Implementation
 * ============================================================================ */

const char* fep_bridge_category_to_string(fep_bridge_category_t category) {
    if (category >= FEP_BRIDGE_CATEGORY_COUNT) return "unknown";
    return CATEGORY_NAMES[category];
}

const char* fep_orchestrator_state_to_string(fep_orchestrator_state_t state) {
    if (state > FEP_ORCHESTRATOR_ERROR) return "unknown";
    return STATE_NAMES[state];
}

/* ============================================================================
 * KG Self-Awareness Integration
 * ============================================================================ */

int fep_orchestrator_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "FEP_Orchestrator");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                fep_orchestrator_heartbeat("fep_orchestr_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("FEP Orchestrator self-knowledge: %s", self->observations[i]);
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "FEP_Orchestrator");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "FEP_Orchestrator");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void fep_orchestrator_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_fep_orchestrator_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int fep_orchestrator_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fep_orchestrator_training_begin: NULL argument");
        return -1;
    }
    fep_orchestrator_heartbeat_instance(NULL, "fep_orchestrator_training_begin", 0.0f);
    return 0;
}

int fep_orchestrator_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fep_orchestrator_training_end: NULL argument");
        return -1;
    }
    fep_orchestrator_heartbeat_instance(NULL, "fep_orchestrator_training_end", 1.0f);
    return 0;
}

int fep_orchestrator_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fep_orchestrator_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    fep_orchestrator_heartbeat_instance(NULL, "fep_orchestrator_training_step", progress);
    return 0;
}
