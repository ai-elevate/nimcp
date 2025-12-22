//=============================================================================
// nimcp_mutex_pool.c - Shared Mutex Pool Implementation
//=============================================================================
/**
 * @file nimcp_mutex_pool.c
 * @brief Implementation of shared mutex pool for memory optimization
 *
 * WHAT: Pool of shared mutexes for bridge synchronization
 * WHY:  Reduce memory footprint from 5.6KB to ~640 bytes (70+ bridges)
 * HOW:  Hash-based assignment of bridges to fixed pool of mutexes
 *
 * @author NIMCP Development Team
 * @date 2024-12-22
 */

#include "utils/thread/nimcp_mutex_pool.h"
#include "utils/memory/nimcp_memory_guards.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

//=============================================================================
// Internal State
//=============================================================================

/**
 * @brief Internal mutex pool structure
 */
typedef struct {
    nimcp_mutex_t mutexes[NIMCP_MUTEX_POOL_SIZE];   /**< The actual mutexes */
    uint32_t ref_counts[NIMCP_MUTEX_POOL_SIZE];     /**< Reference count per slot */
    nimcp_mutex_pool_stats_t stats;                  /**< Usage statistics */
    nimcp_mutex_t stats_lock;                        /**< Protects stats */
    bool initialized;                                /**< Init flag */
} mutex_pool_t;

/**
 * @brief Global mutex pool instance
 */
static mutex_pool_t g_pool = {0};

//=============================================================================
// Hash Function
//=============================================================================

/**
 * @brief FNV-1a hash for strings
 *
 * WHAT: Fast non-cryptographic hash
 * WHY:  Deterministic slot assignment
 * HOW:  FNV-1a algorithm
 */
static uint32_t fnv1a_hash(const char* str) {
    if (!str) return 0;

    uint32_t hash = 2166136261u;  // FNV offset basis
    while (*str) {
        hash ^= (uint8_t)*str++;
        hash *= 16777619u;  // FNV prime
    }
    return hash;
}

//=============================================================================
// Lifecycle API
//=============================================================================

int nimcp_mutex_pool_init(void) {
    if (g_pool.initialized) {
        return 0;  // Already initialized
    }

    // Initialize all pool mutexes
    for (int i = 0; i < NIMCP_MUTEX_POOL_SIZE; i++) {
        int rc = nimcp_platform_mutex_init(&g_pool.mutexes[i], false);
        if (rc != 0) {
            // Rollback on failure
            for (int j = 0; j < i; j++) {
                nimcp_platform_mutex_destroy(&g_pool.mutexes[j]);
            }
            return -1;
        }
        g_pool.ref_counts[i] = 0;
    }

    // Initialize stats lock
    int rc = nimcp_platform_mutex_init(&g_pool.stats_lock, false);
    if (rc != 0) {
        for (int i = 0; i < NIMCP_MUTEX_POOL_SIZE; i++) {
            nimcp_platform_mutex_destroy(&g_pool.mutexes[i]);
        }
        return -1;
    }

    // Clear statistics
    memset(&g_pool.stats, 0, sizeof(g_pool.stats));

    g_pool.initialized = true;
    return 0;
}

int nimcp_mutex_pool_destroy(void) {
    if (!g_pool.initialized) {
        return 0;
    }

    // Destroy all pool mutexes
    for (int i = 0; i < NIMCP_MUTEX_POOL_SIZE; i++) {
        nimcp_platform_mutex_destroy(&g_pool.mutexes[i]);
    }

    nimcp_platform_mutex_destroy(&g_pool.stats_lock);

    g_pool.initialized = false;
    return 0;
}

bool nimcp_mutex_pool_is_initialized(void) {
    return g_pool.initialized;
}

//=============================================================================
// Slot Management API
//=============================================================================

nimcp_mutex_slot_t nimcp_mutex_pool_acquire(const char* bridge_name) {
    if (!g_pool.initialized) {
        // Auto-initialize on first use
        if (nimcp_mutex_pool_init() != 0) {
            return NIMCP_MUTEX_SLOT_INVALID;
        }
    }

    if (!bridge_name) {
        return NIMCP_MUTEX_SLOT_INVALID;
    }

    // Hash name to get slot
    uint32_t hash = fnv1a_hash(bridge_name);
    uint32_t slot = hash % NIMCP_MUTEX_POOL_SIZE;

    // Update stats (thread-safe)
    nimcp_platform_mutex_lock(&g_pool.stats_lock);
    g_pool.ref_counts[slot]++;
    g_pool.stats.total_acquires++;
    g_pool.stats.slot_usage[slot]++;
    nimcp_platform_mutex_unlock(&g_pool.stats_lock);

    return slot;
}

nimcp_mutex_slot_t nimcp_mutex_pool_acquire_by_id(uint32_t bridge_id) {
    if (!g_pool.initialized) {
        if (nimcp_mutex_pool_init() != 0) {
            return NIMCP_MUTEX_SLOT_INVALID;
        }
    }

    uint32_t slot = bridge_id % NIMCP_MUTEX_POOL_SIZE;

    nimcp_platform_mutex_lock(&g_pool.stats_lock);
    g_pool.ref_counts[slot]++;
    g_pool.stats.total_acquires++;
    g_pool.stats.slot_usage[slot]++;
    nimcp_platform_mutex_unlock(&g_pool.stats_lock);

    return slot;
}

void nimcp_mutex_pool_release(nimcp_mutex_slot_t slot) {
    if (!g_pool.initialized || slot >= NIMCP_MUTEX_POOL_SIZE) {
        return;
    }

    nimcp_platform_mutex_lock(&g_pool.stats_lock);
    if (g_pool.ref_counts[slot] > 0) {
        g_pool.ref_counts[slot]--;
    }
    g_pool.stats.total_releases++;
    nimcp_platform_mutex_unlock(&g_pool.stats_lock);
}

//=============================================================================
// Lock/Unlock API
//=============================================================================

int nimcp_mutex_pool_lock(nimcp_mutex_slot_t slot) {
    if (!g_pool.initialized || slot >= NIMCP_MUTEX_POOL_SIZE) {
        return -1;
    }

    int rc = nimcp_platform_mutex_lock(&g_pool.mutexes[slot]);

    // Update stats (best effort, don't lock stats during mutex operation)
    if (rc == 0) {
        // Note: This is a race but acceptable for statistics
        g_pool.stats.total_locks++;
    }

    return rc;
}

int nimcp_mutex_pool_trylock(nimcp_mutex_slot_t slot) {
    if (!g_pool.initialized || slot >= NIMCP_MUTEX_POOL_SIZE) {
        return -1;
    }

    int rc = nimcp_platform_mutex_trylock(&g_pool.mutexes[slot]);

    if (rc == 0) {
        g_pool.stats.total_locks++;
    } else if (rc == NIMCP_BUSY || rc > 0) {
        g_pool.stats.contention_events++;
    }

    return rc;
}

int nimcp_mutex_pool_unlock(nimcp_mutex_slot_t slot) {
    if (!g_pool.initialized || slot >= NIMCP_MUTEX_POOL_SIZE) {
        return -1;
    }

    int rc = nimcp_platform_mutex_unlock(&g_pool.mutexes[slot]);

    if (rc == 0) {
        g_pool.stats.total_unlocks++;
    }

    return rc;
}

//=============================================================================
// Statistics API
//=============================================================================

int nimcp_mutex_pool_get_stats(nimcp_mutex_pool_stats_t* stats) {
    if (!stats) {
        return -1;
    }

    if (!g_pool.initialized) {
        memset(stats, 0, sizeof(*stats));
        return 0;
    }

    nimcp_platform_mutex_lock(&g_pool.stats_lock);
    *stats = g_pool.stats;
    nimcp_platform_mutex_unlock(&g_pool.stats_lock);

    return 0;
}

void nimcp_mutex_pool_reset_stats(void) {
    if (!g_pool.initialized) {
        return;
    }

    nimcp_platform_mutex_lock(&g_pool.stats_lock);
    memset(&g_pool.stats, 0, sizeof(g_pool.stats));
    nimcp_platform_mutex_unlock(&g_pool.stats_lock);
}
