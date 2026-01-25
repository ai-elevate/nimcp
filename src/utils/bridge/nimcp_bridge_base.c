/**
 * @file nimcp_bridge_base.c
 * @brief Base Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-22
 *
 * WHAT: Implements common bridge lifecycle and management functions
 * WHY:  Centralizes boilerplate code for 350+ bridge modules
 * HOW:  Provides reusable functions for mutex, bio-async, and connection management
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

int bridge_base_init(bridge_base_t* base, uint32_t module_id, const char* module_name) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");

    /* Zero all fields */
    memset(base, 0, sizeof(bridge_base_t));

    /* Set module identification */
    base->module_id = module_id;
    base->module_name = module_name;

    /* Allocate and initialize mutex */
    base->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!base->mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex for %s", module_name ? module_name : "bridge");
        return NIMCP_ERROR_NO_MEMORY;
    }

    if (nimcp_mutex_init(base->mutex, NULL) != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex for %s", module_name ? module_name : "bridge");
        nimcp_free(base->mutex);
        base->mutex = NULL;
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    NIMCP_LOGGING_DEBUG("Initialized bridge base for %s (module_id=0x%04X)",
                        module_name ? module_name : "bridge", module_id);

    return 0;
}

void bridge_base_cleanup(bridge_base_t* base) {
    if (!base) {
        return;
    }

    /* Disconnect bio-async first (before destroying mutex) */
    if (base->bio_async_enabled) {
        bridge_base_disconnect_bio_async(base);
    }

    /* Destroy and free mutex (nimcp_mutex_free does both destroy + free) */
    if (base->mutex) {
        nimcp_mutex_free(base->mutex);
        base->mutex = NULL;
    }

    NIMCP_LOGGING_DEBUG("Cleaned up bridge base for %s",
                        base->module_name ? base->module_name : "bridge");
}

int bridge_base_reset(bridge_base_t* base) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");

    nimcp_mutex_lock(base->mutex);

    /* Reset statistics and timing, preserve connections */
    base->last_update_time_ms = 0;
    base->total_updates = 0;

    nimcp_mutex_unlock(base->mutex);

    NIMCP_LOGGING_DEBUG("Reset bridge base for %s",
                        base->module_name ? base->module_name : "bridge");

    return 0;
}

int bridge_base_reset_unlocked(bridge_base_t* base) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");

    /* Reset statistics and timing, preserve connections */
    /* NOTE: Caller must hold the mutex */
    base->last_update_time_ms = 0;
    base->total_updates = 0;

    NIMCP_LOGGING_DEBUG("Reset bridge base (unlocked) for %s",
                        base->module_name ? base->module_name : "bridge");

    return 0;
}

/* ============================================================================
 * Connection Functions
 * ============================================================================ */

int bridge_base_connect_a(bridge_base_t* base, void* system_a) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");
    NIMCP_CHECK_THROW(system_a, NIMCP_ERROR_NULL_POINTER, "system_a is NULL");

    nimcp_mutex_lock(base->mutex);

    base->system_a = system_a;
    base->system_a_connected = true;
    base->bridge_active = base->system_a_connected && base->system_b_connected;

    nimcp_mutex_unlock(base->mutex);

    NIMCP_LOGGING_DEBUG("Connected system_a to %s",
                        base->module_name ? base->module_name : "bridge");

    return 0;
}

int bridge_base_connect_b(bridge_base_t* base, void* system_b) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");
    NIMCP_CHECK_THROW(system_b, NIMCP_ERROR_NULL_POINTER, "system_b is NULL");

    nimcp_mutex_lock(base->mutex);

    base->system_b = system_b;
    base->system_b_connected = true;
    base->bridge_active = base->system_a_connected && base->system_b_connected;

    nimcp_mutex_unlock(base->mutex);

    NIMCP_LOGGING_DEBUG("Connected system_b to %s",
                        base->module_name ? base->module_name : "bridge");

    return 0;
}

int bridge_base_connect_a_unlocked(bridge_base_t* base, void* system_a) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");
    NIMCP_CHECK_THROW(system_a, NIMCP_ERROR_NULL_POINTER, "system_a is NULL");

    /* NOTE: Caller must hold the mutex */
    base->system_a = system_a;
    base->system_a_connected = true;
    base->bridge_active = base->system_a_connected && base->system_b_connected;

    NIMCP_LOGGING_DEBUG("Connected system_a (unlocked) to %s",
                        base->module_name ? base->module_name : "bridge");

    return 0;
}

int bridge_base_connect_b_unlocked(bridge_base_t* base, void* system_b) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");
    NIMCP_CHECK_THROW(system_b, NIMCP_ERROR_NULL_POINTER, "system_b is NULL");

    /* NOTE: Caller must hold the mutex */
    base->system_b = system_b;
    base->system_b_connected = true;
    base->bridge_active = base->system_a_connected && base->system_b_connected;

    NIMCP_LOGGING_DEBUG("Connected system_b (unlocked) to %s",
                        base->module_name ? base->module_name : "bridge");

    return 0;
}

int bridge_base_disconnect_a(bridge_base_t* base) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");

    nimcp_mutex_lock(base->mutex);

    base->system_a = NULL;
    base->system_a_connected = false;
    base->bridge_active = false;

    nimcp_mutex_unlock(base->mutex);

    NIMCP_LOGGING_DEBUG("Disconnected system_a from %s",
                        base->module_name ? base->module_name : "bridge");

    return 0;
}

int bridge_base_disconnect_b(bridge_base_t* base) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");

    nimcp_mutex_lock(base->mutex);

    base->system_b = NULL;
    base->system_b_connected = false;
    base->bridge_active = false;

    nimcp_mutex_unlock(base->mutex);

    NIMCP_LOGGING_DEBUG("Disconnected system_b from %s",
                        base->module_name ? base->module_name : "bridge");

    return 0;
}

bool bridge_base_is_connected(const bridge_base_t* base) {
    if (!base) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "bridge_base_is_connected: base is NULL");

            return false;
    }
    return base->bridge_active;
}

/* ============================================================================
 * Bio-Async Functions
 * ============================================================================ */

int bridge_base_connect_bio_async(bridge_base_t* base) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");

    /* Already connected */
    if (base->bio_async_enabled) {
        return 0;
    }

    /* Create module info */
    bio_module_info_t info = {
        .module_id = base->module_id,
        .module_name = base->module_name ? base->module_name : "bridge",
        .inbox_capacity = NIMCP_BIO_INBOX_CAPACITY,
        .user_data = base
    };

    /* Register with router */
    base->bio_ctx = bio_router_register_module(&info);
    if (!base->bio_ctx) {
        NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration for %s",
                          base->module_name ? base->module_name : "bridge");
        return 0;  /* Not fatal */
    }

    base->bio_async_enabled = true;
    NIMCP_LOGGING_INFO("Connected %s to bio-async router (module_id=0x%04X)",
                       base->module_name ? base->module_name : "bridge", base->module_id);

    return 0;
}

int bridge_base_disconnect_bio_async(bridge_base_t* base) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");

    if (!base->bio_async_enabled) {
        return 0;
    }

    if (base->bio_ctx) {
        bio_router_unregister_module(base->bio_ctx);
        base->bio_ctx = NULL;
    }

    base->bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected %s from bio-async router",
                       base->module_name ? base->module_name : "bridge");

    return 0;
}

bool bridge_base_is_bio_async_connected(const bridge_base_t* base) {
    if (!base) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "bridge_base_is_bio_async_connected: base is NULL");

            return false;
    }
    return base->bio_async_enabled;
}

/* ============================================================================
 * Update Functions
 * ============================================================================ */

int bridge_base_record_update(bridge_base_t* base) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");

    /* NOTE: Callers are expected to already hold the bridge lock.
     * This function does not acquire the lock to avoid deadlocks.
     */
    base->last_update_time_ms = nimcp_time_get_ms();
    base->total_updates++;

    return 0;
}

int bridge_base_get_stats(const bridge_base_t* base,
                          uint64_t* total_updates,
                          uint64_t* last_update) {
    NIMCP_CHECK_THROW(base, NIMCP_ERROR_NULL_POINTER, "bridge base is NULL");

    nimcp_mutex_lock((nimcp_mutex_t*)base->mutex);

    if (total_updates) {
        *total_updates = base->total_updates;
    }
    if (last_update) {
        *last_update = base->last_update_time_ms;
    }

    nimcp_mutex_unlock((nimcp_mutex_t*)base->mutex);

    return 0;
}
