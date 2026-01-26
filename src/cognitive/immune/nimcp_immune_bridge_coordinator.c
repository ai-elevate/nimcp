/**
 * @file nimcp_immune_bridge_coordinator.c
 * @brief Immune Bridge Coordinator Implementation
 * @version 1.0.0
 * @date 2025-12-15
 */

#include "cognitive/immune/nimcp_immune_bridge_coordinator.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <time.h>
#include <math.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for immune_bridge_coordinator module */
static nimcp_health_agent_t* g_immune_bridge_coordinator_health_agent = NULL;

/**
 * @brief Set health agent for immune_bridge_coordinator heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void immune_bridge_coordinator_set_health_agent(nimcp_health_agent_t* agent) {
    g_immune_bridge_coordinator_health_agent = agent;
}

/** @brief Send heartbeat from immune_bridge_coordinator module */
static inline void immune_bridge_coordinator_heartbeat(const char* operation, float progress) {
    if (g_immune_bridge_coordinator_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_immune_bridge_coordinator_health_agent, operation, progress);
    }
}


/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

/**
 * @brief Get current time in milliseconds
 *
 * WHAT: Get system time for timing tracking
 * WHY:  Track update intervals and health checks
 * HOW:  Platform-specific time retrieval
 */
static uint64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

/**
 * @brief Get current time in microseconds
 */
static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + (uint64_t)ts.tv_nsec / 1000;
}

/**
 * @brief Find bridge by ID
 *
 * WHAT: Locate bridge entry in registry
 * WHY:  Internal helper for bridge lookup
 * HOW:  Linear search through bridge array
 */
static immune_bridge_entry_t* find_bridge(
    immune_bridge_coordinator_t* coordinator,
    uint32_t bridge_id
) {
    if (!coordinator) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

        return NULL;

    }

    for (uint32_t i = 0; i < coordinator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && coordinator->bridge_count > 256) {
            immune_bridge_coordinator_heartbeat("immune_bridg_loop",
                             (float)(i + 1) / (float)coordinator->bridge_count);
        }

        if (coordinator->bridges[i].bridge_id == bridge_id) {
            return &coordinator->bridges[i];
        }
    }

    return NULL;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

int immune_bridge_coordinator_default_config(
    immune_bridge_coordinator_config_t* config
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_default_config", 0.0f);


    NIMCP_CHECK_THROW(config != NULL, NIMCP_ERROR_NULL_POINTER, "config is NULL");

    memset(config, 0, sizeof(*config));

    /* Enable all categories by default */
    for (int i = 0; i < IMMUNE_BRIDGE_CATEGORY_COUNT; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && IMMUNE_BRIDGE_CATEGORY_COUNT > 256) {
            immune_bridge_coordinator_heartbeat("immune_bridg_loop",
                             (float)(i + 1) / (float)IMMUNE_BRIDGE_CATEGORY_COUNT);
        }

        config->categories[i].enabled = true;
        config->categories[i].update_priority = i;
    }

    /* Global settings */
    config->max_bridges = IMMUNE_COORDINATOR_MAX_BRIDGES;
    config->enable_auto_update = true;
    config->health_check_interval_ms = IMMUNE_COORDINATOR_HEALTH_CHECK_MS;

    /* Integration enables */
    config->enable_bio_async = true;
    config->enable_brain_immune = true;
    config->enable_statistics = true;
    config->enable_logging = true;

    /* Performance tuning */
    config->max_updates_per_cycle = 0;  /* 0 = all */
    config->update_time_budget_ms = 0.0f;  /* 0 = unlimited */
    config->max_consecutive_failures = 5;

    return NIMCP_SUCCESS;
}

immune_bridge_coordinator_t* immune_bridge_coordinator_create(
    const immune_bridge_coordinator_config_t* config
) {
    /* Allocate coordinator */
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_create", 0.0f);


    immune_bridge_coordinator_t* coordinator =
        (immune_bridge_coordinator_t*)nimcp_malloc(sizeof(immune_bridge_coordinator_t));
    if (!coordinator) {
        NIMCP_LOGGING_ERROR("Failed to allocate immune bridge coordinator");
        return NULL;
    }

    memset(coordinator, 0, sizeof(*coordinator));

    /* Set config */
    if (config) {
        coordinator->config = *config;
    } else {
        immune_bridge_coordinator_default_config(&coordinator->config);
    }

    /* Allocate bridge registry */
    coordinator->bridge_capacity = coordinator->config.max_bridges;
    coordinator->bridges = (immune_bridge_entry_t*)nimcp_malloc(
        coordinator->bridge_capacity * sizeof(immune_bridge_entry_t)
    );
    if (!coordinator->bridges) {
        NIMCP_LOGGING_ERROR("Failed to allocate bridge registry");
        nimcp_free(coordinator);
        return NULL;
    }

    memset(coordinator->bridges, 0,
           coordinator->bridge_capacity * sizeof(immune_bridge_entry_t));

    /* Create mutex */
    coordinator->mutex = nimcp_platform_mutex_create();
    if (!coordinator->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create coordinator mutex");
        nimcp_free(coordinator->bridges);
        nimcp_free(coordinator);
        return NULL;
    }

    /* Initialize state */
    coordinator->state = IMMUNE_COORDINATOR_STOPPED;
    coordinator->bridge_count = 0;
    coordinator->next_bridge_id = 1;
    coordinator->start_time = get_time_ms();
    coordinator->last_update_time = 0;
    coordinator->last_health_check_time = 0;
    coordinator->bio_async_connected = false;
    coordinator->immune_connected = false;

    /* Initialize statistics */
    memset(&coordinator->stats, 0, sizeof(coordinator->stats));
    coordinator->stats.system_health = 1.0f;  /* Perfect health when no bridges */

    if (coordinator->config.enable_logging) {
        NIMCP_LOGGING_INFO("Immune bridge coordinator created");
    }

    return coordinator;
}

void immune_bridge_coordinator_destroy(immune_bridge_coordinator_t* coordinator) {
    if (!coordinator) return;

    /* Stop if running */
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_destroy", 0.0f);


    if (coordinator->state == IMMUNE_COORDINATOR_RUNNING) {
        immune_bridge_coordinator_stop(coordinator);
    }

    /* Disconnect integrations */
    if (coordinator->bio_async_connected) {
        immune_bridge_coordinator_disconnect_bio_async(coordinator);
    }
    if (coordinator->immune_connected) {
        immune_bridge_coordinator_disconnect_brain_immune(coordinator);
    }

    /* Free registry */
    if (coordinator->bridges) {
        nimcp_free(coordinator->bridges);
    }

    /* Destroy mutex */
    if (coordinator->mutex) {
        nimcp_platform_mutex_destroy(coordinator->mutex);
    }

    if (coordinator->config.enable_logging) {
        NIMCP_LOGGING_INFO("Immune bridge coordinator destroyed");
    }

    nimcp_free(coordinator);
}

int immune_bridge_coordinator_start(immune_bridge_coordinator_t* coordinator) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_start", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    nimcp_platform_mutex_lock(coordinator->mutex);

    if (coordinator->state == IMMUNE_COORDINATOR_RUNNING) {
        nimcp_platform_mutex_unlock(coordinator->mutex);
        return NIMCP_SUCCESS;
    }

    coordinator->state = IMMUNE_COORDINATOR_STARTING;
    coordinator->start_time = get_time_ms();
    coordinator->last_update_time = coordinator->start_time;
    coordinator->last_health_check_time = coordinator->start_time;
    coordinator->state = IMMUNE_COORDINATOR_RUNNING;

    nimcp_platform_mutex_unlock(coordinator->mutex);

    if (coordinator->config.enable_logging) {
        NIMCP_LOGGING_INFO("Immune bridge coordinator started");
    }

    return NIMCP_SUCCESS;
}

int immune_bridge_coordinator_stop(immune_bridge_coordinator_t* coordinator) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_stop", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    nimcp_platform_mutex_lock(coordinator->mutex);

    if (coordinator->state == IMMUNE_COORDINATOR_STOPPED) {
        nimcp_platform_mutex_unlock(coordinator->mutex);
        return NIMCP_SUCCESS;
    }

    coordinator->state = IMMUNE_COORDINATOR_STOPPING;
    coordinator->state = IMMUNE_COORDINATOR_STOPPED;

    nimcp_platform_mutex_unlock(coordinator->mutex);

    if (coordinator->config.enable_logging) {
        NIMCP_LOGGING_INFO("Immune bridge coordinator stopped");
    }

    return NIMCP_SUCCESS;
}

int immune_bridge_coordinator_pause(immune_bridge_coordinator_t* coordinator) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_pause", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    nimcp_platform_mutex_lock(coordinator->mutex);

    if (coordinator->state != IMMUNE_COORDINATOR_RUNNING) {
        nimcp_platform_mutex_unlock(coordinator->mutex);
        return NIMCP_ERROR_INVALID_STATE;
    }

    coordinator->state = IMMUNE_COORDINATOR_PAUSED;
    nimcp_platform_mutex_unlock(coordinator->mutex);

    return NIMCP_SUCCESS;
}

int immune_bridge_coordinator_resume(immune_bridge_coordinator_t* coordinator) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_resume", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    nimcp_platform_mutex_lock(coordinator->mutex);

    if (coordinator->state != IMMUNE_COORDINATOR_PAUSED) {
        nimcp_platform_mutex_unlock(coordinator->mutex);
        return NIMCP_ERROR_INVALID_STATE;
    }

    coordinator->state = IMMUNE_COORDINATOR_RUNNING;
    nimcp_platform_mutex_unlock(coordinator->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bridge Registration API
 * ============================================================================ */

int immune_bridge_coordinator_register_bridge(
    immune_bridge_coordinator_t* coordinator,
    const char* name,
    immune_bridge_category_t category,
    immune_bridge_handle_t handle,
    immune_bridge_update_fn_t update_fn,
    immune_bridge_destroy_fn_t destroy_fn,
    immune_bridge_health_fn_t health_fn,
    uint32_t* bridge_id_out
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_register_bridge", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL && name != NULL && handle != NULL,
                      NIMCP_ERROR_NULL_POINTER, "NULL parameter in register_bridge");
    NIMCP_CHECK_THROW(category < IMMUNE_BRIDGE_CATEGORY_COUNT,
                      NIMCP_ERROR_INVALID_PARAM, "Invalid bridge category");

    nimcp_platform_mutex_lock(coordinator->mutex);

    /* Check capacity */
    if (coordinator->bridge_count >= coordinator->bridge_capacity) {
        nimcp_platform_mutex_unlock(coordinator->mutex);
        NIMCP_LOGGING_ERROR("Bridge registry full");
        return NIMCP_ERROR_NO_MEMORY;
    }

    /* Allocate bridge entry */
    immune_bridge_entry_t* entry = &coordinator->bridges[coordinator->bridge_count];
    memset(entry, 0, sizeof(*entry));

    /* Set fields */
    entry->bridge_id = coordinator->next_bridge_id++;
    entry->bridge_name = name;
    entry->category = category;
    entry->handle = handle;
    entry->update_fn = update_fn;
    entry->destroy_fn = destroy_fn;
    entry->health_fn = health_fn;
    entry->health_status = IMMUNE_BRIDGE_HEALTHY;
    entry->last_update_time = 0;
    entry->last_health_check_time = 0;
    entry->update_count = 0;
    entry->total_update_time_us = 0;
    entry->consecutive_failures = 0;
    entry->enabled = true;
    entry->bio_async_connected = false;

    coordinator->bridge_count++;
    coordinator->stats.total_bridges++;
    coordinator->stats.active_bridges++;
    coordinator->stats.healthy_bridges++;
    coordinator->stats.categories[category].bridge_count++;
    coordinator->stats.categories[category].healthy_bridges++;

    if (bridge_id_out) {
        *bridge_id_out = entry->bridge_id;
    }

    nimcp_platform_mutex_unlock(coordinator->mutex);

    if (coordinator->config.enable_logging) {
        NIMCP_LOGGING_INFO("Registered bridge: %s (ID=%u, category=%s)",
                          name, entry->bridge_id,
                          immune_bridge_category_to_string(category));
    }

    return NIMCP_SUCCESS;
}

int immune_bridge_coordinator_unregister_bridge(
    immune_bridge_coordinator_t* coordinator,
    uint32_t bridge_id
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_unregister_bridge", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    nimcp_platform_mutex_lock(coordinator->mutex);

    /* Find bridge */
    int32_t found_idx = -1;
    for (uint32_t i = 0; i < coordinator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && coordinator->bridge_count > 256) {
            immune_bridge_coordinator_heartbeat("immune_bridg_loop",
                             (float)(i + 1) / (float)coordinator->bridge_count);
        }

        if (coordinator->bridges[i].bridge_id == bridge_id) {
            found_idx = (int32_t)i;
            break;
        }
    }

    if (found_idx < 0) {
        nimcp_platform_mutex_unlock(coordinator->mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    immune_bridge_entry_t* entry = &coordinator->bridges[found_idx];
    immune_bridge_category_t category = entry->category;

    /* Update statistics */
    coordinator->stats.total_bridges--;
    if (entry->enabled) {
        coordinator->stats.active_bridges--;
    }
    if (entry->health_status == IMMUNE_BRIDGE_HEALTHY) {
        coordinator->stats.healthy_bridges--;
        coordinator->stats.categories[category].healthy_bridges--;
    }
    coordinator->stats.categories[category].bridge_count--;

    /* Remove from array (shift remaining bridges) */
    for (uint32_t i = (uint32_t)found_idx; i < coordinator->bridge_count - 1; i++) {
        coordinator->bridges[i] = coordinator->bridges[i + 1];
    }
    coordinator->bridge_count--;

    nimcp_platform_mutex_unlock(coordinator->mutex);

    if (coordinator->config.enable_logging) {
        NIMCP_LOGGING_INFO("Unregistered bridge ID=%u", bridge_id);
    }

    return NIMCP_SUCCESS;
}

int immune_bridge_coordinator_set_bridge_enabled(
    immune_bridge_coordinator_t* coordinator,
    uint32_t bridge_id,
    bool enabled
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_set_bridge_enabled", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    nimcp_platform_mutex_lock(coordinator->mutex);

    immune_bridge_entry_t* entry = find_bridge(coordinator, bridge_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(coordinator->mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Update state */
    bool was_enabled = entry->enabled;
    entry->enabled = enabled;

    /* Update statistics */
    if (was_enabled && !enabled) {
        coordinator->stats.active_bridges--;
    } else if (!was_enabled && enabled) {
        coordinator->stats.active_bridges++;
    }

    nimcp_platform_mutex_unlock(coordinator->mutex);

    return NIMCP_SUCCESS;
}

const immune_bridge_entry_t* immune_bridge_coordinator_get_bridge(
    const immune_bridge_coordinator_t* coordinator,
    uint32_t bridge_id
) {
    if (!coordinator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

        return NULL;
    }

    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_get_bridge", 0.0f);


    for (uint32_t i = 0; i < coordinator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && coordinator->bridge_count > 256) {
            immune_bridge_coordinator_heartbeat("immune_bridg_loop",
                             (float)(i + 1) / (float)coordinator->bridge_count);
        }

        if (coordinator->bridges[i].bridge_id == bridge_id) {
            return &coordinator->bridges[i];
        }
    }

    return NULL;
}

uint32_t immune_bridge_coordinator_get_bridges_by_category(
    const immune_bridge_coordinator_t* coordinator,
    immune_bridge_category_t category,
    const immune_bridge_entry_t** bridges,
    uint32_t max_bridges
) {
    if (!coordinator || !bridges || max_bridges == 0) {
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_get_bridges_by_categ", 0.0f);


    if (category >= IMMUNE_BRIDGE_CATEGORY_COUNT) {
        return 0;
    }

    uint32_t count = 0;
    for (uint32_t i = 0; i < coordinator->bridge_count && count < max_bridges; i++) {
        if (coordinator->bridges[i].category == category) {
            bridges[count++] = &coordinator->bridges[i];
        }
    }

    return count;
}

/* ============================================================================
 * Update API
 * ============================================================================ */

int immune_bridge_coordinator_update(
    immune_bridge_coordinator_t* coordinator,
    uint64_t current_time_ms
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_update", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    nimcp_platform_mutex_lock(coordinator->mutex);

    if (coordinator->state != IMMUNE_COORDINATOR_RUNNING) {
        nimcp_platform_mutex_unlock(coordinator->mutex);
        return 0;
    }

    uint64_t cycle_start = get_time_us();
    uint32_t updated_count = 0;
    uint32_t max_updates = coordinator->config.max_updates_per_cycle;
    if (max_updates == 0) {
        max_updates = coordinator->bridge_count;
    }

    /* Update enabled bridges */
    for (uint32_t i = 0; i < coordinator->bridge_count && updated_count < max_updates; i++) {
        immune_bridge_entry_t* entry = &coordinator->bridges[i];

        if (!entry->enabled || !entry->update_fn) {
            continue;
        }

        /* Call update function */
        uint64_t update_start = get_time_us();
        int result = entry->update_fn(entry->handle);
        uint64_t update_time = get_time_us() - update_start;

        /* Update statistics */
        entry->update_count++;
        entry->total_update_time_us += update_time;
        entry->last_update_time = current_time_ms;

        coordinator->stats.total_bridge_updates++;
        coordinator->stats.categories[entry->category].total_updates++;

        /* Track failures */
        if (result != 0) {
            entry->consecutive_failures++;
            coordinator->stats.update_errors++;

            if (entry->consecutive_failures >= coordinator->config.max_consecutive_failures) {
                entry->health_status = IMMUNE_BRIDGE_DEGRADED;
                coordinator->stats.healthy_bridges--;
                coordinator->stats.categories[entry->category].healthy_bridges--;
            }
        } else {
            if (entry->consecutive_failures > 0 &&
                entry->health_status == IMMUNE_BRIDGE_DEGRADED) {
                entry->health_status = IMMUNE_BRIDGE_HEALTHY;
                coordinator->stats.healthy_bridges++;
                coordinator->stats.categories[entry->category].healthy_bridges++;
            }
            entry->consecutive_failures = 0;
        }

        /* Update category stats */
        uint64_t cat_total = coordinator->stats.categories[entry->category].total_updates;
        if (cat_total > 0) {
            float old_avg = coordinator->stats.categories[entry->category].avg_update_time_us;
            coordinator->stats.categories[entry->category].avg_update_time_us =
                (old_avg * (cat_total - 1) + (float)update_time) / cat_total;

            if ((float)update_time > coordinator->stats.categories[entry->category].max_update_time_us) {
                coordinator->stats.categories[entry->category].max_update_time_us = (float)update_time;
            }
        }

        updated_count++;
    }

    /* Update global statistics */
    uint64_t cycle_time = get_time_us() - cycle_start;
    coordinator->stats.total_update_cycles++;
    coordinator->last_update_time = current_time_ms;

    if (coordinator->stats.total_update_cycles > 0) {
        float old_avg = coordinator->stats.avg_cycle_time_us;
        coordinator->stats.avg_cycle_time_us =
            (old_avg * (coordinator->stats.total_update_cycles - 1) + (float)cycle_time) /
            coordinator->stats.total_update_cycles;

        if ((float)cycle_time > coordinator->stats.max_cycle_time_us) {
            coordinator->stats.max_cycle_time_us = (float)cycle_time;
        }
    }

    nimcp_platform_mutex_unlock(coordinator->mutex);

    return (int)updated_count;
}

int immune_bridge_coordinator_update_category(
    immune_bridge_coordinator_t* coordinator,
    immune_bridge_category_t category
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_update_category", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");
    NIMCP_CHECK_THROW(category < IMMUNE_BRIDGE_CATEGORY_COUNT,
                      NIMCP_ERROR_INVALID_PARAM, "Invalid bridge category");

    nimcp_platform_mutex_lock(coordinator->mutex);

    if (coordinator->state != IMMUNE_COORDINATOR_RUNNING) {
        nimcp_platform_mutex_unlock(coordinator->mutex);
        return 0;
    }

    uint32_t updated_count = 0;
    uint64_t current_time = get_time_ms();

    for (uint32_t i = 0; i < coordinator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && coordinator->bridge_count > 256) {
            immune_bridge_coordinator_heartbeat("immune_bridg_loop",
                             (float)(i + 1) / (float)coordinator->bridge_count);
        }

        immune_bridge_entry_t* entry = &coordinator->bridges[i];

        if (entry->category != category || !entry->enabled || !entry->update_fn) {
            continue;
        }

        /* Call update */
        int result = entry->update_fn(entry->handle);
        entry->update_count++;
        entry->last_update_time = current_time;

        if (result == 0) {
            updated_count++;
        }
    }

    nimcp_platform_mutex_unlock(coordinator->mutex);

    return (int)updated_count;
}

int immune_bridge_coordinator_update_bridge(
    immune_bridge_coordinator_t* coordinator,
    uint32_t bridge_id
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_update_bridge", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    nimcp_platform_mutex_lock(coordinator->mutex);

    immune_bridge_entry_t* entry = find_bridge(coordinator, bridge_id);
    if (!entry || !entry->update_fn) {
        nimcp_platform_mutex_unlock(coordinator->mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Call update */
    int result = entry->update_fn(entry->handle);
    entry->update_count++;
    entry->last_update_time = get_time_ms();

    nimcp_platform_mutex_unlock(coordinator->mutex);

    return result;
}

/* ============================================================================
 * Health Monitoring API
 * ============================================================================ */

int immune_bridge_coordinator_health_check(
    immune_bridge_coordinator_t* coordinator
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_health_check", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    nimcp_platform_mutex_lock(coordinator->mutex);

    uint32_t healthy_count = 0;
    uint64_t current_time = get_time_ms();

    for (uint32_t i = 0; i < coordinator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && coordinator->bridge_count > 256) {
            immune_bridge_coordinator_heartbeat("immune_bridg_loop",
                             (float)(i + 1) / (float)coordinator->bridge_count);
        }

        immune_bridge_entry_t* entry = &coordinator->bridges[i];

        immune_bridge_health_t old_health = entry->health_status;
        immune_bridge_health_t new_health;

        /* Call health check if available */
        if (entry->health_fn) {
            new_health = entry->health_fn(entry->handle);
        } else {
            /* Infer health from update failures */
            if (entry->consecutive_failures >= coordinator->config.max_consecutive_failures) {
                new_health = IMMUNE_BRIDGE_DEGRADED;
            } else if (entry->consecutive_failures > 0) {
                new_health = IMMUNE_BRIDGE_DEGRADED;
            } else {
                new_health = IMMUNE_BRIDGE_HEALTHY;
            }
        }

        entry->health_status = new_health;
        entry->last_health_check_time = current_time;

        /* Update statistics if health changed */
        if (old_health != new_health) {
            if (old_health == IMMUNE_BRIDGE_HEALTHY) {
                coordinator->stats.healthy_bridges--;
                coordinator->stats.categories[entry->category].healthy_bridges--;
            }
            if (new_health == IMMUNE_BRIDGE_HEALTHY) {
                coordinator->stats.healthy_bridges++;
                coordinator->stats.categories[entry->category].healthy_bridges++;
            }

            /* Update category degraded/disconnected counts */
            if (new_health == IMMUNE_BRIDGE_DEGRADED) {
                coordinator->stats.categories[entry->category].degraded_bridges++;
            }
            if (old_health == IMMUNE_BRIDGE_DEGRADED) {
                coordinator->stats.categories[entry->category].degraded_bridges--;
            }
            if (new_health == IMMUNE_BRIDGE_DISCONNECTED) {
                coordinator->stats.categories[entry->category].disconnected_bridges++;
            }
            if (old_health == IMMUNE_BRIDGE_DISCONNECTED) {
                coordinator->stats.categories[entry->category].disconnected_bridges--;
            }
        }

        if (new_health == IMMUNE_BRIDGE_HEALTHY) {
            healthy_count++;
        }
    }

    coordinator->last_health_check_time = current_time;
    coordinator->stats.last_health_check_time = current_time;

    /* Compute system health */
    if (coordinator->bridge_count > 0) {
        coordinator->stats.system_health =
            (float)healthy_count / (float)coordinator->bridge_count;
    } else {
        coordinator->stats.system_health = 1.0f;
    }

    nimcp_platform_mutex_unlock(coordinator->mutex);

    return (int)healthy_count;
}

immune_bridge_health_t immune_bridge_coordinator_check_bridge_health(
    immune_bridge_coordinator_t* coordinator,
    uint32_t bridge_id
) {
    if (!coordinator) {
        return IMMUNE_BRIDGE_ERROR;
    }

    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_check_bridge_health", 0.0f);


    nimcp_platform_mutex_lock(coordinator->mutex);

    immune_bridge_entry_t* entry = find_bridge(coordinator, bridge_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(coordinator->mutex);
        return IMMUNE_BRIDGE_ERROR;
    }

    immune_bridge_health_t health;
    if (entry->health_fn) {
        health = entry->health_fn(entry->handle);
    } else {
        health = entry->health_status;
    }

    entry->health_status = health;
    entry->last_health_check_time = get_time_ms();

    nimcp_platform_mutex_unlock(coordinator->mutex);

    return health;
}

float immune_bridge_coordinator_get_system_health(
    const immune_bridge_coordinator_t* coordinator
) {
    if (!coordinator) {
        return 0.0f;
    }

    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_get_system_health", 0.0f);


    return coordinator->stats.system_health;
}

/* ============================================================================
 * Cross-Bridge Coordination API
 * ============================================================================ */

int immune_bridge_coordinator_broadcast_message(
    immune_bridge_coordinator_t* coordinator,
    bio_message_type_t message_type,
    const void* data,
    size_t data_len
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_broadcast_message", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    if (!coordinator->bio_async_connected) {
        return 0;  /* Not an error, just not connected */
    }

    /* This would use bio_router to send messages to all registered bridges */
    /* For now, return count of bridges that would receive message */
    coordinator->stats.bio_async_messages_sent += coordinator->bridge_count;
    return (int)coordinator->bridge_count;
}

int immune_bridge_coordinator_send_category_message(
    immune_bridge_coordinator_t* coordinator,
    immune_bridge_category_t category,
    bio_message_type_t message_type,
    const void* data,
    size_t data_len
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_send_category_messag", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");
    NIMCP_CHECK_THROW(category < IMMUNE_BRIDGE_CATEGORY_COUNT,
                      NIMCP_ERROR_INVALID_PARAM, "Invalid bridge category");

    if (!coordinator->bio_async_connected) {
        return 0;
    }

    /* Count bridges in category */
    uint32_t count = 0;
    for (uint32_t i = 0; i < coordinator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && coordinator->bridge_count > 256) {
            immune_bridge_coordinator_heartbeat("immune_bridg_loop",
                             (float)(i + 1) / (float)coordinator->bridge_count);
        }

        if (coordinator->bridges[i].category == category) {
            count++;
        }
    }

    coordinator->stats.bio_async_messages_sent += count;
    return (int)count;
}

/* ============================================================================
 * Integration API
 * ============================================================================ */

int immune_bridge_coordinator_connect_brain_immune(
    immune_bridge_coordinator_t* coordinator,
    brain_immune_system_t* immune
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_connect_brain_immune", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL && immune != NULL,
                      NIMCP_ERROR_NULL_POINTER, "NULL parameter in connect_brain_immune");

    nimcp_platform_mutex_lock(coordinator->mutex);
    coordinator->brain_immune = immune;
    coordinator->immune_connected = true;
    nimcp_platform_mutex_unlock(coordinator->mutex);

    if (coordinator->config.enable_logging) {
        NIMCP_LOGGING_INFO("Connected to brain immune system");
    }

    return NIMCP_SUCCESS;
}

int immune_bridge_coordinator_disconnect_brain_immune(
    immune_bridge_coordinator_t* coordinator
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_disconnect_brain_imm", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    nimcp_platform_mutex_lock(coordinator->mutex);
    coordinator->brain_immune = NULL;
    coordinator->immune_connected = false;
    nimcp_platform_mutex_unlock(coordinator->mutex);

    return NIMCP_SUCCESS;
}

int immune_bridge_coordinator_connect_bio_async(
    immune_bridge_coordinator_t* coordinator
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_connect_bio_async", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    if (coordinator->bio_async_connected) {
        return NIMCP_SUCCESS;
    }

    /* Register with bio-async router */
    bio_module_info_t info = {
        .module_id = BIO_MODULE_IMMUNE_COORDINATOR,
        .module_name = IMMUNE_COORDINATOR_MODULE_NAME,
        .inbox_capacity = 64,
        .user_data = coordinator
    };

    coordinator->bio_context = bio_router_register_module(&info);
    if (coordinator->bio_context) {
        coordinator->bio_async_connected = true;
        if (coordinator->config.enable_logging) {
            NIMCP_LOGGING_INFO("Connected to bio-async router");
        }
        return NIMCP_SUCCESS;
    }

    /* Not an error if bio-async not available */
    if (coordinator->config.enable_logging) {
        NIMCP_LOGGING_INFO("Bio-async router not available, skipping registration");
    }
    return NIMCP_SUCCESS;
}

int immune_bridge_coordinator_disconnect_bio_async(
    immune_bridge_coordinator_t* coordinator
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_disconnect_bio_async", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL, NIMCP_ERROR_NULL_POINTER, "coordinator is NULL");

    if (coordinator->bio_async_connected && coordinator->bio_context) {
        bio_router_unregister_module(coordinator->bio_context);
        coordinator->bio_context = NULL;
        coordinator->bio_async_connected = false;
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Statistics and Monitoring API
 * ============================================================================ */

int immune_bridge_coordinator_get_stats(
    const immune_bridge_coordinator_t* coordinator,
    immune_coordinator_stats_t* stats
) {
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_get_stats", 0.0f);


    NIMCP_CHECK_THROW(coordinator != NULL && stats != NULL,
                      NIMCP_ERROR_NULL_POINTER, "NULL parameter in get_stats");

    *stats = coordinator->stats;
    return NIMCP_SUCCESS;
}

void immune_bridge_coordinator_reset_stats(
    immune_bridge_coordinator_t* coordinator
) {
    if (!coordinator) {
        return;
    }

    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_reset_stats", 0.0f);


    nimcp_platform_mutex_lock(coordinator->mutex);

    /* Reset global stats but preserve current counts */
    uint32_t total_bridges = coordinator->stats.total_bridges;
    uint32_t active_bridges = coordinator->stats.active_bridges;
    uint32_t healthy_bridges = coordinator->stats.healthy_bridges;

    memset(&coordinator->stats, 0, sizeof(coordinator->stats));

    coordinator->stats.total_bridges = total_bridges;
    coordinator->stats.active_bridges = active_bridges;
    coordinator->stats.healthy_bridges = healthy_bridges;

    /* Reset per-bridge stats */
    for (uint32_t i = 0; i < coordinator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && coordinator->bridge_count > 256) {
            immune_bridge_coordinator_heartbeat("immune_bridg_loop",
                             (float)(i + 1) / (float)coordinator->bridge_count);
        }

        coordinator->bridges[i].update_count = 0;
        coordinator->bridges[i].total_update_time_us = 0;
        coordinator->bridges[i].consecutive_failures = 0;
    }

    nimcp_platform_mutex_unlock(coordinator->mutex);
}

immune_coordinator_state_t immune_bridge_coordinator_get_state(
    const immune_bridge_coordinator_t* coordinator
) {
    if (!coordinator) {
        return IMMUNE_COORDINATOR_ERROR;
    }

    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_get_state", 0.0f);


    return coordinator->state;
}

/* ============================================================================
 * String Conversion API
 * ============================================================================ */

const char* immune_bridge_category_to_string(immune_bridge_category_t category) {
    switch (category) {
        case IMMUNE_BRIDGE_CATEGORY_COGNITIVE:  return "cognitive";
        case IMMUNE_BRIDGE_CATEGORY_PLASTICITY: return "plasticity";
        case IMMUNE_BRIDGE_CATEGORY_MIDDLEWARE: return "middleware";
        case IMMUNE_BRIDGE_CATEGORY_PERCEPTION: return "perception";
        case IMMUNE_BRIDGE_CATEGORY_CORE:       return "core";
        case IMMUNE_BRIDGE_CATEGORY_GLIAL:      return "glial";
        case IMMUNE_BRIDGE_CATEGORY_SECURITY:   return "security";
        case IMMUNE_BRIDGE_CATEGORY_OTHER:      return "other";
        default:                                 return "unknown";
    }
}

const char* immune_bridge_health_to_string(immune_bridge_health_t health) {
    switch (health) {
        case IMMUNE_BRIDGE_HEALTHY:      return "healthy";
        case IMMUNE_BRIDGE_DEGRADED:     return "degraded";
        case IMMUNE_BRIDGE_DISCONNECTED: return "disconnected";
        case IMMUNE_BRIDGE_ERROR:        return "error";
        default:                          return "unknown";
    }
}

const char* immune_coordinator_state_to_string(immune_coordinator_state_t state) {
    switch (state) {
        case IMMUNE_COORDINATOR_STOPPED:  return "stopped";
        case IMMUNE_COORDINATOR_STARTING: return "starting";
        case IMMUNE_COORDINATOR_RUNNING:  return "running";
        case IMMUNE_COORDINATOR_PAUSED:   return "paused";
        case IMMUNE_COORDINATOR_STOPPING: return "stopping";
        case IMMUNE_COORDINATOR_ERROR:    return "error";
        default:                           return "unknown";
    }
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * @brief Query self-knowledge from knowledge graph
 *
 * WHAT: Query KG for module self-awareness information
 * WHY:  Enable introspective self-knowledge about immune bridge coordinator
 * HOW:  Look up entity and relations in KG
 *
 * @param kg Knowledge graph reader handle
 * @return 1 if self-knowledge found, 0 otherwise
 */
int immune_bridge_coordinator_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    immune_bridge_coordinator_heartbeat("immune_bridg_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Immune_Bridge_Coordinator");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                immune_bridge_coordinator_heartbeat("immune_bridg_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Immune bridge coordinator self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Immune_Bridge_Coordinator");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Immune_Bridge_Coordinator");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
