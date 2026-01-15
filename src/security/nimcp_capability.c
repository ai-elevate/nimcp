/**
 * @file nimcp_capability.c
 * @brief Implementation of Capability-Based Access Control
 *
 * WHAT: Implements capability-based security for fine-grained access control.
 *
 * WHY:  Enforces principle of least privilege by requiring explicit
 *       capability tokens for all resource access.
 *
 * HOW:  Maintains table of capabilities with generation IDs to prevent
 *       forgery. Supports delegation with attenuation and revocation.
 *
 * Part of Phase SC-1: Security Coverage Framework (Tier 0.7)
 */

#include "security/nimcp_capability.h"
#include "security/nimcp_security.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"

#define LOG_MODULE "security_capability"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Internal capability system context
 */
struct nimcp_capability_system {
    // Capability table
    nimcp_cap_entry_t entries[NIMCP_CAP_MAX_CAPABILITIES];
    uint32_t num_capabilities;
    uint32_t next_free;

    // Holders
    nimcp_cap_holder_t holders[NIMCP_CAP_MAX_HOLDERS];
    uint32_t num_holders;
    uint32_t next_holder_id;

    // Statistics
    nimcp_cap_stats_t stats;

    // State
    bool initialized;
    bool root_created;

    // Thread safety
    nimcp_mutex_t* mutex;
};

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Get current timestamp
 */
static uint64_t get_timestamp(void)
{
    return (uint64_t)time(NULL);
}

/**
 * @brief Find free capability slot
 */
static int32_t find_free_slot(nimcp_capability_system_t* caps)
{
    for (uint32_t i = 0; i < NIMCP_CAP_MAX_CAPABILITIES; i++) {
        if (!caps->entries[i].valid && !caps->entries[i].revoked) {
            return (int32_t)i;
        }
    }
    return -1;
}

/**
 * @brief Validate capability token
 */
static nimcp_cap_entry_t* validate_capability(
    nimcp_capability_system_t* caps,
    nimcp_capability_t capability)
{
    if (!caps || capability.index >= NIMCP_CAP_MAX_CAPABILITIES)
        return NULL;

    nimcp_cap_entry_t* entry = &caps->entries[capability.index];

    // Check validity
    if (!entry->valid || entry->revoked)
        return NULL;

    // Check generation (prevents use-after-revoke attacks)
    if (entry->generation != capability.generation)
        return NULL;

    return entry;
}

/**
 * @brief Recursively revoke delegated capabilities
 */
static void revoke_children(nimcp_capability_system_t* caps, uint32_t parent_index)
{
    for (uint32_t i = 0; i < NIMCP_CAP_MAX_CAPABILITIES; i++) {
        if (caps->entries[i].valid &&
            caps->entries[i].parent_cap == parent_index) {
            caps->entries[i].revoked = true;
            caps->entries[i].valid = false;
            caps->stats.revoked_capabilities++;
            caps->stats.active_capabilities--;
            caps->stats.revocations++;

            // Recursively revoke children
            revoke_children(caps, i);
        }
    }
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

nimcp_capability_system_t* nimcp_capability_system_create(void)
{
    nimcp_capability_system_t* caps =
        (nimcp_capability_system_t*)nimcp_calloc(1, sizeof(nimcp_capability_system_t));

    if (!caps)
        return NULL;

    // Create mutex for thread safety
    caps->mutex = nimcp_mutex_create(NULL);
    if (!caps->mutex) {
        nimcp_free(caps);
        return NULL;
    }

    caps->initialized = false;
    caps->root_created = false;
    caps->next_holder_id = 1;

    return caps;
}

nimcp_result_t nimcp_capability_system_init(nimcp_capability_system_t* caps)
{
    if (!caps)
        return NIMCP_INVALID_PARAM;

    if (caps->initialized)
        return NIMCP_INVALID_STATE;

    caps->initialized = true;

    nimcp_security_log_event(
        NIMCP_SECURITY_EVENT_DIRECTIVE_VERIFIED,
        NIMCP_THREAT_NONE,
        "Capability system initialized"
    );

    return NIMCP_SUCCESS;
}

void nimcp_capability_system_destroy(nimcp_capability_system_t* caps)
{
    if (!caps)
        return;

    // Destroy mutex
    if (caps->mutex) {
        nimcp_mutex_destroy(caps->mutex);
    }

    // Free holder capability arrays
    for (uint32_t i = 0; i < caps->num_holders; i++) {
        if (caps->holders[i].capabilities) {
            nimcp_free(caps->holders[i].capabilities);
        }
    }

    memset(caps, 0, sizeof(nimcp_capability_system_t));
    nimcp_free(caps);
}

//=============================================================================
// Capability Creation
//=============================================================================

/**
 * @brief Internal unlocked version of capability create
 * @note Caller must hold caps->mutex
 */
static nimcp_result_t capability_create_unlocked(
    nimcp_capability_system_t* caps,
    nimcp_resource_type_t resource_type,
    void* resource_ptr,
    uint32_t permissions,
    nimcp_capability_t* capability)
{
    if (!caps->initialized)
        return NIMCP_INVALID_STATE;

    int32_t slot = find_free_slot(caps);
    if (slot < 0)
        return NIMCP_BUFFER_TOO_SMALL;

    nimcp_cap_entry_t* entry = &caps->entries[slot];

    entry->generation++;
    entry->permissions = permissions;
    entry->resource_type = resource_type;
    entry->resource_ptr = resource_ptr;
    entry->parent_cap = UINT32_MAX;  // No parent
    entry->holder_id = 0;
    entry->valid = true;
    entry->revoked = false;
    entry->created_time = get_timestamp();
    entry->last_used = 0;
    entry->use_count = 0;

    capability->index = (uint32_t)slot;
    capability->generation = entry->generation;
    capability->permissions = permissions;

    caps->num_capabilities++;
    caps->stats.total_capabilities++;
    caps->stats.active_capabilities++;

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_capability_create(
    nimcp_capability_system_t* caps,
    nimcp_resource_type_t resource_type,
    void* resource_ptr,
    uint32_t permissions,
    nimcp_capability_t* capability)
{
    if (!caps || !capability)
        return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(caps->mutex);
    nimcp_result_t result = capability_create_unlocked(
        caps, resource_type, resource_ptr, permissions, capability);
    nimcp_mutex_unlock(caps->mutex);

    return result;
}

nimcp_result_t nimcp_capability_create_root(
    nimcp_capability_system_t* caps,
    nimcp_capability_t* capability)
{
    if (!caps || !capability)
        return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(caps->mutex);

    if (caps->root_created) {
        nimcp_mutex_unlock(caps->mutex);
        return NIMCP_INVALID_STATE;
    }

    nimcp_result_t result = capability_create_unlocked(
        caps,
        NIMCP_RES_GENERIC,
        NULL,
        NIMCP_PERM_ALL,
        capability
    );

    if (result == NIMCP_SUCCESS) {
        caps->root_created = true;
        nimcp_security_log_event(
            NIMCP_SECURITY_EVENT_DIRECTIVE_VERIFIED,
            NIMCP_THREAT_NONE,
            "Root capability created"
        );
    }

    nimcp_mutex_unlock(caps->mutex);

    return result;
}

nimcp_result_t nimcp_capability_delegate(
    nimcp_capability_system_t* caps,
    nimcp_capability_t parent,
    uint32_t permissions,
    nimcp_capability_t* child)
{
    if (!caps || !child)
        return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(caps->mutex);

    nimcp_cap_entry_t* parent_entry = validate_capability(caps, parent);
    if (!parent_entry) {
        nimcp_mutex_unlock(caps->mutex);
        return NIMCP_INVALID_STATE;
    }

    // Check delegation permission
    if (!(parent_entry->permissions & NIMCP_PERM_DELEGATE)) {
        caps->stats.checks_failed++;
        nimcp_mutex_unlock(caps->mutex);
        return NIMCP_INVALID_STATE;
    }

    // Permissions must be subset of parent (attenuation)
    uint32_t allowed = parent_entry->permissions & ~NIMCP_PERM_DELEGATE;
    if ((permissions & ~allowed) != 0) {
        caps->stats.checks_failed++;
        nimcp_mutex_unlock(caps->mutex);
        return NIMCP_INVALID_PARAM;
    }

    int32_t slot = find_free_slot(caps);
    if (slot < 0) {
        nimcp_mutex_unlock(caps->mutex);
        return NIMCP_BUFFER_TOO_SMALL;
    }

    nimcp_cap_entry_t* entry = &caps->entries[slot];

    entry->generation++;
    entry->permissions = permissions;
    entry->resource_type = parent_entry->resource_type;
    entry->resource_ptr = parent_entry->resource_ptr;
    entry->parent_cap = parent.index;
    entry->holder_id = 0;
    entry->valid = true;
    entry->revoked = false;
    entry->created_time = get_timestamp();
    entry->last_used = 0;
    entry->use_count = 0;

    child->index = (uint32_t)slot;
    child->generation = entry->generation;
    child->permissions = permissions;

    caps->num_capabilities++;
    caps->stats.total_capabilities++;
    caps->stats.active_capabilities++;
    caps->stats.delegations++;

    nimcp_mutex_unlock(caps->mutex);

    return NIMCP_SUCCESS;
}

//=============================================================================
// Capability Verification
//=============================================================================

bool nimcp_capability_is_valid(
    nimcp_capability_system_t* caps,
    nimcp_capability_t capability)
{
    if (!caps)
        return false;

    nimcp_mutex_lock(caps->mutex);
    bool valid = validate_capability(caps, capability) != NULL;
    nimcp_mutex_unlock(caps->mutex);

    return valid;
}

/**
 * @brief Internal unlocked version of capability check
 * @note Caller must hold caps->mutex
 */
static bool capability_check_unlocked(
    nimcp_capability_system_t* caps,
    nimcp_capability_t capability,
    uint32_t permission)
{
    caps->stats.checks_performed++;

    nimcp_cap_entry_t* entry = validate_capability(caps, capability);
    if (!entry) {
        caps->stats.checks_failed++;
        return false;
    }

    entry->last_used = get_timestamp();
    entry->use_count++;

    bool has_permission = (entry->permissions & permission) == permission;

    if (has_permission) {
        caps->stats.checks_passed++;
    } else {
        caps->stats.checks_failed++;
        nimcp_security_log_event(
            NIMCP_SECURITY_EVENT_INPUT_REJECTED,
            NIMCP_THREAT_MEDIUM,
            "Capability permission check failed"
        );
    }

    return has_permission;
}

bool nimcp_capability_check(
    nimcp_capability_system_t* caps,
    nimcp_capability_t capability,
    uint32_t permission)
{
    if (!caps)
        return false;

    nimcp_mutex_lock(caps->mutex);
    bool result = capability_check_unlocked(caps, capability, permission);
    nimcp_mutex_unlock(caps->mutex);

    return result;
}

bool nimcp_capability_check_access(
    nimcp_capability_system_t* caps,
    nimcp_capability_t capability,
    void* resource_ptr,
    uint32_t permission)
{
    if (!caps)
        return false;

    nimcp_mutex_lock(caps->mutex);

    nimcp_cap_entry_t* entry = validate_capability(caps, capability);
    if (!entry) {
        caps->stats.checks_failed++;
        nimcp_mutex_unlock(caps->mutex);
        return false;
    }

    // Check resource matches (NULL means any resource)
    if (entry->resource_ptr != NULL && entry->resource_ptr != resource_ptr) {
        caps->stats.checks_failed++;
        nimcp_mutex_unlock(caps->mutex);
        return false;
    }

    bool result = capability_check_unlocked(caps, capability, permission);
    nimcp_mutex_unlock(caps->mutex);

    return result;
}

uint32_t nimcp_capability_get_permissions(
    nimcp_capability_system_t* caps,
    nimcp_capability_t capability)
{
    if (!caps)
        return 0;

    nimcp_mutex_lock(caps->mutex);

    nimcp_cap_entry_t* entry = validate_capability(caps, capability);
    if (!entry) {
        nimcp_mutex_unlock(caps->mutex);
        return 0;
    }

    uint32_t permissions = entry->permissions;
    nimcp_mutex_unlock(caps->mutex);

    return permissions;
}

//=============================================================================
// Capability Revocation
//=============================================================================

nimcp_result_t nimcp_capability_revoke(
    nimcp_capability_system_t* caps,
    nimcp_capability_t capability)
{
    if (!caps)
        return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(caps->mutex);

    nimcp_cap_entry_t* entry = validate_capability(caps, capability);
    if (!entry) {
        nimcp_mutex_unlock(caps->mutex);
        return NIMCP_INVALID_STATE;
    }

    entry->revoked = true;
    entry->valid = false;
    caps->stats.revoked_capabilities++;
    caps->stats.active_capabilities--;
    caps->stats.revocations++;

    // Revoke all children
    revoke_children(caps, capability.index);

    nimcp_mutex_unlock(caps->mutex);

    nimcp_security_log_event(
        NIMCP_SECURITY_EVENT_DIRECTIVE_VERIFIED,
        NIMCP_THREAT_NONE,
        "Capability revoked"
    );

    return NIMCP_SUCCESS;
}

uint32_t nimcp_capability_revoke_for_resource(
    nimcp_capability_system_t* caps,
    void* resource_ptr)
{
    if (!caps)
        return 0;

    nimcp_mutex_lock(caps->mutex);

    uint32_t count = 0;

    for (uint32_t i = 0; i < NIMCP_CAP_MAX_CAPABILITIES; i++) {
        if (caps->entries[i].valid &&
            caps->entries[i].resource_ptr == resource_ptr) {
            caps->entries[i].revoked = true;
            caps->entries[i].valid = false;
            caps->stats.revoked_capabilities++;
            caps->stats.active_capabilities--;
            caps->stats.revocations++;
            count++;
        }
    }

    nimcp_mutex_unlock(caps->mutex);

    return count;
}

/**
 * @brief Internal unlocked version of revoke holder
 * @note Caller must hold caps->mutex
 */
static uint32_t capability_revoke_holder_unlocked(
    nimcp_capability_system_t* caps,
    uint32_t holder_id)
{
    uint32_t count = 0;

    for (uint32_t i = 0; i < NIMCP_CAP_MAX_CAPABILITIES; i++) {
        if (caps->entries[i].valid &&
            caps->entries[i].holder_id == holder_id) {
            caps->entries[i].revoked = true;
            caps->entries[i].valid = false;
            caps->stats.revoked_capabilities++;
            caps->stats.active_capabilities--;
            caps->stats.revocations++;
            count++;
        }
    }

    return count;
}

uint32_t nimcp_capability_revoke_holder(
    nimcp_capability_system_t* caps,
    uint32_t holder_id)
{
    if (!caps)
        return 0;

    nimcp_mutex_lock(caps->mutex);
    uint32_t count = capability_revoke_holder_unlocked(caps, holder_id);
    nimcp_mutex_unlock(caps->mutex);

    return count;
}

//=============================================================================
// Holder Management
//=============================================================================

nimcp_result_t nimcp_capability_register_holder(
    nimcp_capability_system_t* caps,
    const char* name,
    uint32_t* holder_id)
{
    if (!caps || !holder_id)
        return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(caps->mutex);

    if (caps->num_holders >= NIMCP_CAP_MAX_HOLDERS) {
        nimcp_mutex_unlock(caps->mutex);
        return NIMCP_BUFFER_TOO_SMALL;
    }

    nimcp_cap_holder_t* holder = &caps->holders[caps->num_holders];

    holder->holder_id = caps->next_holder_id++;
    holder->name = name;
    holder->capabilities = nimcp_calloc(32, sizeof(uint32_t));
    holder->num_capabilities = 0;
    holder->max_capabilities = 32;
    holder->active = true;

    *holder_id = holder->holder_id;
    caps->num_holders++;

    nimcp_mutex_unlock(caps->mutex);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_capability_assign(
    nimcp_capability_system_t* caps,
    uint32_t holder_id,
    nimcp_capability_t capability)
{
    if (!caps)
        return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(caps->mutex);

    // Find holder
    nimcp_cap_holder_t* holder = NULL;
    for (uint32_t i = 0; i < caps->num_holders; i++) {
        if (caps->holders[i].holder_id == holder_id && caps->holders[i].active) {
            holder = &caps->holders[i];
            break;
        }
    }

    if (!holder) {
        nimcp_mutex_unlock(caps->mutex);
        return NIMCP_INVALID_STATE;
    }

    // Validate capability
    nimcp_cap_entry_t* entry = validate_capability(caps, capability);
    if (!entry) {
        nimcp_mutex_unlock(caps->mutex);
        return NIMCP_INVALID_STATE;
    }

    // Add to holder's list
    if (holder->num_capabilities >= holder->max_capabilities) {
        nimcp_mutex_unlock(caps->mutex);
        return NIMCP_BUFFER_TOO_SMALL;
    }

    holder->capabilities[holder->num_capabilities++] = capability.index;
    entry->holder_id = holder_id;

    nimcp_mutex_unlock(caps->mutex);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_capability_remove_holder(
    nimcp_capability_system_t* caps,
    uint32_t holder_id)
{
    if (!caps)
        return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(caps->mutex);

    // Revoke all capabilities (using unlocked version to avoid deadlock)
    capability_revoke_holder_unlocked(caps, holder_id);

    // Mark holder inactive
    for (uint32_t i = 0; i < caps->num_holders; i++) {
        if (caps->holders[i].holder_id == holder_id) {
            caps->holders[i].active = false;
            break;
        }
    }

    nimcp_mutex_unlock(caps->mutex);

    return NIMCP_SUCCESS;
}

//=============================================================================
// Statistics
//=============================================================================

nimcp_result_t nimcp_capability_get_stats(
    nimcp_capability_system_t* caps,
    nimcp_cap_stats_t* stats)
{
    if (!caps || !stats)
        return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(caps->mutex);
    memcpy(stats, &caps->stats, sizeof(nimcp_cap_stats_t));
    nimcp_mutex_unlock(caps->mutex);

    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_capability_reset_stats(nimcp_capability_system_t* caps)
{
    if (!caps)
        return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(caps->mutex);

    caps->stats.checks_performed = 0;
    caps->stats.checks_passed = 0;
    caps->stats.checks_failed = 0;
    caps->stats.delegations = 0;
    caps->stats.revocations = 0;

    nimcp_mutex_unlock(caps->mutex);

    return NIMCP_SUCCESS;
}

//=============================================================================
// Utility Functions
//=============================================================================

bool nimcp_capability_is_null(nimcp_capability_t capability)
{
    return capability.index == 0 &&
           capability.generation == 0 &&
           capability.permissions == 0;
}

bool nimcp_capability_equals(nimcp_capability_t a, nimcp_capability_t b)
{
    return a.index == b.index &&
           a.generation == b.generation &&
           a.permissions == b.permissions;
}

const char* nimcp_resource_type_name(nimcp_resource_type_t type)
{
    static const char* names[] = {
        "Generic",
        "Memory",
        "File",
        "Neural Network",
        "Neural Layer",
        "Neural Synapse",
        "Security",
        "Directive",
        "IPC Channel",
        "Thread",
        "Module"
    };

    if (type > NIMCP_RES_MODULE)
        return "Unknown";

    return names[type];
}

const char* nimcp_permission_name(uint32_t permission)
{
    if (permission == NIMCP_PERM_NONE) return "None";
    if (permission == NIMCP_PERM_ALL) return "All";
    if (permission & NIMCP_PERM_READ) return "Read";
    if (permission & NIMCP_PERM_WRITE) return "Write";
    if (permission & NIMCP_PERM_EXECUTE) return "Execute";
    if (permission & NIMCP_PERM_DELETE) return "Delete";
    if (permission & NIMCP_PERM_CREATE) return "Create";
    if (permission & NIMCP_PERM_DELEGATE) return "Delegate";
    if (permission & NIMCP_PERM_REVOKE) return "Revoke";
    if (permission & NIMCP_PERM_SECURITY) return "Security";
    return "Unknown";
}

int nimcp_permissions_to_string(
    uint32_t permissions,
    char* buffer,
    size_t size)
{
    if (!buffer || size == 0)
        return -1;

    int offset = 0;
    int remaining = (int)size;

    buffer[0] = '\0';

    if (permissions == NIMCP_PERM_NONE) {
        return snprintf(buffer, size, "None");
    }

    if (permissions == NIMCP_PERM_ALL) {
        return snprintf(buffer, size, "All");
    }

    const struct {
        uint32_t flag;
        const char* name;
    } flags[] = {
        {NIMCP_PERM_READ, "R"},
        {NIMCP_PERM_WRITE, "W"},
        {NIMCP_PERM_EXECUTE, "X"},
        {NIMCP_PERM_DELETE, "D"},
        {NIMCP_PERM_CREATE, "C"},
        {NIMCP_PERM_DELEGATE, "G"},
        {NIMCP_PERM_REVOKE, "V"},
        {0, NULL}
    };

    for (int i = 0; flags[i].name && remaining > 0; i++) {
        if (permissions & flags[i].flag) {
            int written = snprintf(buffer + offset, remaining, "%s", flags[i].name);
            if (written < 0) break;
            offset += written;
            remaining -= written;
        }
    }

    return offset;
}
