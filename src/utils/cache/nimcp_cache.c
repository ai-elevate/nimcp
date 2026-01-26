/**
 * @file nimcp_cache.c
 * @brief Copy-on-Write caching implementation
 *
 * IMPLEMENTATION NOTES:
 * - Uses atomic operations for thread-safe reference counting
 * - Hash table for O(1) cache metadata lookup
 * - Mutex for protecting hash table operations
 * - Canary values for detecting corruption
 */

#include "utils/cache/nimcp_cache.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/containers/nimcp_hash_table.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <stddef.h>  /* for NULL */
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for cache module */
static nimcp_health_agent_t* g_cache_health_agent = NULL;

/**
 * @brief Set health agent for cache heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void cache_set_health_agent(nimcp_health_agent_t* agent) {
    g_cache_health_agent = agent;
}

/** @brief Send heartbeat from cache module */
static inline void cache_heartbeat(const char* operation, float progress) {
    if (g_cache_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_cache_health_agent, operation, progress);
    }
}


//=============================================================================
// Internal Structures
//=============================================================================

/** WHAT: Magic number for cache header validation */
#define NIMCP_CACHE_MAGIC 0xCAC4EFED

/** WHAT: Canary value for corruption detection */
#define NIMCP_CACHE_CANARY 0xDEADBEEF

/**
 * WHAT: Internal cache allocation header
 * WHY: Track metadata for each cached allocation
 * HOW: Prepended to user data
 */
typedef struct nimcp_cache_header {
    uint32_t magic;                /**< Magic number for validation */
    uint32_t ref_count;            /**< Reference count (protected by mutex) */
    size_t size;                   /**< Size of user data */
    uint32_t canary_start;         /**< Start canary */

#ifdef NIMCP_CACHE_DEBUG
    const char* alloc_file;        /**< Allocation source file */
    uint32_t alloc_line;           /**< Allocation source line */
    uint64_t alloc_timestamp;      /**< Allocation timestamp */
#endif

    // User data follows here
    // uint32_t canary_end follows user data
} nimcp_cache_header_t;

/**
 * WHAT: Global cache state
 * WHY: Track system-wide statistics and configuration
 */
typedef struct {
    bool initialized;              /**< Is system initialized? */
    nimcp_cache_config_t config;   /**< Configuration */
    nimcp_cache_stats_t stats;     /**< Statistics */
    hash_table_t* tracking_table;  /**< Allocation tracking (ptr -> header) */
    nimcp_platform_mutex_t mutex;  /**< Protects tracking table and stats */
} nimcp_cache_state_t;

/** Global cache state (initialized by nimcp_cache_init) */
static nimcp_cache_state_t g_cache_state = {
    .initialized = false
};

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * WHAT: Get cache header from user pointer
 * WHY: Access metadata from user-visible pointer
 * HOW: Pointer arithmetic
 */
static inline nimcp_cache_header_t* get_cache_header(const void* ptr) {
    if (!ptr) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ptr is NULL");

        return NULL;

    }
    return (nimcp_cache_header_t*)((char*)ptr - sizeof(nimcp_cache_header_t));
}

/**
 * WHAT: Get user pointer from cache header
 * WHY: Return user-visible pointer
 * HOW: Pointer arithmetic
 */
static inline void* get_user_ptr(nimcp_cache_header_t* header) {
    if (!header) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "header is NULL");

        return NULL;

    }
    return (char*)header + sizeof(nimcp_cache_header_t);
}

/**
 * WHAT: Get end canary location
 * WHY: Detect buffer overflows
 * HOW: Pointer arithmetic to end of user data
 */
__attribute__((no_sanitize("address")))
static inline uint32_t* get_end_canary(nimcp_cache_header_t* header) {
    void* user_ptr = get_user_ptr(header);
    return (uint32_t*)((char*)user_ptr + header->size);
}

/**
 * WHAT: Validate cache header
 * WHY: Detect corruption and invalid pointers
 * HOW: Check magic number and canaries
 *
 * @return true if valid, false if corrupted/invalid
 *
 * Note: This function may intentionally access invalid memory when
 * checking if arbitrary pointers are cached. ASAN is disabled to allow
 * this behavior in test cases.
 */
__attribute__((no_sanitize("address")))
static bool validate_cache_header(nimcp_cache_header_t* header) {
    if (!header) return false;

    // Check magic number
    if (header->magic != NIMCP_CACHE_MAGIC) {
        if (g_cache_state.config.enable_debug_output) {
            fprintf(stderr, "[CACHE] Invalid magic number: 0x%X (expected 0x%X)\n",
                    header->magic, NIMCP_CACHE_MAGIC);
        }
        return false;
    }

    // Check start canary
    if (header->canary_start != NIMCP_CACHE_CANARY) {
        if (g_cache_state.config.enable_debug_output) {
            fprintf(stderr, "[CACHE] Start canary corrupted: 0x%X\n",
                    header->canary_start);
        }
        return false;
    }

    // Check end canary (read via memcpy to avoid alignment issues)
    uint32_t* end_canary = get_end_canary(header);
    uint32_t end_canary_value;
    memcpy(&end_canary_value, end_canary, sizeof(uint32_t));
    if (end_canary_value != NIMCP_CACHE_CANARY) {
        if (g_cache_state.config.enable_debug_output) {
            fprintf(stderr, "[CACHE] End canary corrupted: 0x%X\n", end_canary_value);
        }
        return false;
    }

    return true;
}

/**
 * WHAT: Check if allocation would exceed limits
 * WHY: Prevent unbounded memory growth
 * HOW: Compare against configured limits
 */
static bool check_allocation_limit(size_t size) {
    nimcp_cache_config_t* cfg = &g_cache_state.config;

    // Check single allocation limit
    if (cfg->max_single_allocation > 0 && size > cfg->max_single_allocation) {
        if (cfg->enable_debug_output) {
            fprintf(stderr, "[CACHE] Allocation size %zu exceeds limit %zu\n",
                    size, cfg->max_single_allocation);
        }
        return false;
    }

    // Check total memory limit
    if (cfg->max_total_memory > 0) {
        nimcp_platform_mutex_lock(&g_cache_state.mutex);
        size_t total = g_cache_state.stats.memory_allocated + size;
        nimcp_platform_mutex_unlock(&g_cache_state.mutex);

        if (total > cfg->max_total_memory) {
            if (cfg->enable_debug_output) {
                fprintf(stderr, "[CACHE] Total memory %zu would exceed limit %zu\n",
                        total, cfg->max_total_memory);
            }
            return false;
        }
    }

    return true;
}

/**
 * WHAT: Update statistics (thread-safe)
 * WHY: Track cache system performance
 * HOW: Mutex-protected updates
 */
static void update_stats_alloc(size_t size) {
    nimcp_platform_mutex_lock(&g_cache_state.mutex);
    g_cache_state.stats.allocations_created++;
    g_cache_state.stats.memory_allocated += size;
    g_cache_state.stats.active_allocations++;
    if (g_cache_state.stats.active_allocations > g_cache_state.stats.peak_allocations) {
        g_cache_state.stats.peak_allocations = g_cache_state.stats.active_allocations;
    }
    nimcp_platform_mutex_unlock(&g_cache_state.mutex);
}

static void update_stats_reference(size_t size) {
    nimcp_platform_mutex_lock(&g_cache_state.mutex);
    g_cache_state.stats.references_created++;
    g_cache_state.stats.memory_shared += size;
    g_cache_state.stats.memory_saved += size;
    nimcp_platform_mutex_unlock(&g_cache_state.mutex);
}

static void update_stats_copy(size_t size) {
    nimcp_platform_mutex_lock(&g_cache_state.mutex);
    g_cache_state.stats.copies_triggered++;
    g_cache_state.stats.memory_shared -= size;
    nimcp_platform_mutex_unlock(&g_cache_state.mutex);
}

static void update_stats_release(size_t size) {
    nimcp_platform_mutex_lock(&g_cache_state.mutex);
    g_cache_state.stats.memory_allocated -= size;
    g_cache_state.stats.active_allocations--;
    nimcp_platform_mutex_unlock(&g_cache_state.mutex);
}

//=============================================================================
// Initialization and Configuration
//=============================================================================

void nimcp_cache_init(void) {
    if (g_cache_state.initialized) {
        return;  // Already initialized
    }

    // Initialize mutex
    nimcp_platform_mutex_init(&g_cache_state.mutex, false);

    // Set default configuration
    g_cache_state.config = nimcp_cache_get_default_config();

    // Create tracking table (if tracking enabled)
    // TODO: Implement proper pointer-key hash table for tracking
    // For Phase 1, tracking table is disabled
    g_cache_state.tracking_table = NULL;

    // Clear statistics
    memset(&g_cache_state.stats, 0, sizeof(nimcp_cache_stats_t));

    g_cache_state.initialized = true;

    if (g_cache_state.config.enable_debug_output) {
        printf("[CACHE] System initialized\n");
    }
}

void nimcp_cache_cleanup(void) {
    if (!g_cache_state.initialized) {
        return;
    }

    // Check for leaks
    if (g_cache_state.stats.active_allocations > 0) {
        fprintf(stderr, "[CACHE] WARNING: %u allocations still active at cleanup\n",
                g_cache_state.stats.active_allocations);
        nimcp_cache_check_leaks();
    }

    // Destroy tracking table
    if (g_cache_state.tracking_table) {
        hash_table_destroy(g_cache_state.tracking_table);
        g_cache_state.tracking_table = NULL;
    }

    // Destroy mutex
    nimcp_platform_mutex_destroy(&g_cache_state.mutex);

    g_cache_state.initialized = false;

    if (g_cache_state.config.enable_debug_output) {
        printf("[CACHE] System cleaned up\n");
    }
}

void nimcp_cache_configure(const nimcp_cache_config_t* config) {
    if (!config) return;

    nimcp_platform_mutex_lock(&g_cache_state.mutex);
    g_cache_state.config = *config;
    nimcp_platform_mutex_unlock(&g_cache_state.mutex);
}

nimcp_cache_config_t nimcp_cache_get_default_config(void) {
    nimcp_cache_config_t config = {
        .max_total_memory = 0,           // Unlimited
        .max_single_allocation = 1024 * 1024 * 1024,  // 1GB
        .enable_tracking = true,
        .enable_debug_output = false
    };
    return config;
}

//=============================================================================
// Core Cache Functions
//=============================================================================

void* nimcp_cache_alloc(size_t size) {
    if (!g_cache_state.initialized) {
        nimcp_cache_init();
    }

    if (size == 0) {
        return NULL;
    }

    // Check limits
    if (!check_allocation_limit(size)) {
        return NULL;
    }

    // Calculate total allocation size
    size_t total_size = sizeof(nimcp_cache_header_t) + size + sizeof(uint32_t);

    // Allocate memory
    nimcp_cache_header_t* header = nimcp_malloc(total_size);
    if (!header) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "header is NULL");

        return NULL;
    }

    // Initialize header
    header->magic = NIMCP_CACHE_MAGIC;
    header->ref_count = 1;
    header->size = size;
    header->canary_start = NIMCP_CACHE_CANARY;

#ifdef NIMCP_CACHE_DEBUG
    header->alloc_file = NULL;
    header->alloc_line = 0;
    header->alloc_timestamp = nimcp_time_get_ms();
#endif

    // Set end canary (use memcpy to avoid alignment issues)
    uint32_t* end_canary = get_end_canary(header);
    uint32_t canary_value = NIMCP_CACHE_CANARY;
    memcpy(end_canary, &canary_value, sizeof(uint32_t));

    // Get user pointer
    void* user_ptr = get_user_ptr(header);

    // Track allocation
    // TODO: Implement proper pointer-key hash table for tracking
    // For Phase 1, tracking is disabled
    (void)g_cache_state;  // Suppress unused variable warning

    // Update statistics
    update_stats_alloc(size);

    if (g_cache_state.config.enable_debug_output) {
        printf("[CACHE] Allocated %zu bytes at %p\n", size, user_ptr);
    }

    return user_ptr;
}

void* nimcp_cache_calloc(size_t count, size_t size) {
    size_t total = count * size;
    void* ptr = nimcp_cache_alloc(total);
    if (ptr) {
        memset(ptr, 0, total);
    }
    return ptr;
}

void* nimcp_cache_reference(void* ptr) {
    if (!ptr) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ptr is NULL");

        return NULL;

    }

    nimcp_cache_header_t* header = get_cache_header(ptr);
    if (!validate_cache_header(header)) {
        return NULL;
    }

    // Atomic increment reference count
    uint32_t old_count = __atomic_fetch_add(&header->ref_count, 1, __ATOMIC_ACQ_REL);

    // Update statistics (only on transition from 1 to 2)
    if (old_count == 1) {
        update_stats_reference(header->size);
    }

    if (g_cache_state.config.enable_debug_output) {
        printf("[CACHE] Referenced %p (ref_count: %u -> %u)\n",
               ptr, old_count, old_count + 1);
    }

    return ptr;
}

void* nimcp_cache_make_writable(void* ptr) {
    if (!ptr) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ptr is NULL");

        return NULL;

    }

    nimcp_cache_header_t* header = get_cache_header(ptr);
    if (!validate_cache_header(header)) {
        return NULL;
    }

    // Check if already private (ref_count == 1)
    uint32_t ref_count = __atomic_load_n(&header->ref_count, __ATOMIC_ACQUIRE);
    if (ref_count == 1) {
        // Already private, no copy needed
        return ptr;
    }

    // Need to copy - allocate new cached memory
    void* new_ptr = nimcp_cache_alloc(header->size);
    if (!new_ptr) {
        return NULL;  // Allocation failed, return NULL
    }

    // Copy data
    memcpy(new_ptr, ptr, header->size);

    // Release reference to original
    __atomic_fetch_sub(&header->ref_count, 1, __ATOMIC_ACQ_REL);

    // Update statistics
    update_stats_copy(header->size);

    if (g_cache_state.config.enable_debug_output) {
        printf("[CACHE] Made writable %p -> %p (copied %zu bytes)\n",
               ptr, new_ptr, header->size);
    }

    return new_ptr;
}

void nimcp_cache_release(void* ptr) {
    if (!ptr) return;

    nimcp_cache_header_t* header = get_cache_header(ptr);
    if (!validate_cache_header(header)) {
        fprintf(stderr, "[CACHE] Attempted to release invalid cache pointer %p\n", ptr);
        return;
    }

    // Atomic decrement reference count
    uint32_t old_count = __atomic_fetch_sub(&header->ref_count, 1, __ATOMIC_ACQ_REL);

    if (g_cache_state.config.enable_debug_output) {
        printf("[CACHE] Released %p (ref_count: %u -> %u)\n",
               ptr, old_count, old_count - 1);
    }

    // If last reference, free memory
    if (old_count == 1) {
        // Remove from tracking table
        // TODO: Implement proper pointer-key hash table for tracking
        // For Phase 1, tracking is disabled

        // Update statistics
        update_stats_release(header->size);

        // Free memory
        nimcp_free(header);

        if (g_cache_state.config.enable_debug_output) {
            printf("[CACHE] Freed %p\n", ptr);
        }
    }
}

//=============================================================================
// Query Functions
//=============================================================================

__attribute__((no_sanitize("address")))
bool nimcp_cache_is_cached(const void* ptr) {
    if (!ptr) return false;

    nimcp_cache_header_t* header = get_cache_header(ptr);
    return validate_cache_header(header);
}

bool nimcp_cache_is_shared(const void* ptr) {
    if (!ptr) return false;

    nimcp_cache_header_t* header = get_cache_header(ptr);
    if (!validate_cache_header(header)) {
        return false;
    }

    uint32_t ref_count = __atomic_load_n(&header->ref_count, __ATOMIC_ACQUIRE);
    return ref_count > 1;
}

uint32_t nimcp_cache_get_refcount(const void* ptr) {
    if (!ptr) return 0;

    nimcp_cache_header_t* header = get_cache_header(ptr);
    if (!validate_cache_header(header)) {
        return 0;
    }

    return __atomic_load_n(&header->ref_count, __ATOMIC_ACQUIRE);
}

size_t nimcp_cache_get_size(const void* ptr) {
    if (!ptr) return 0;

    nimcp_cache_header_t* header = get_cache_header(ptr);
    if (!validate_cache_header(header)) {
        return 0;
    }

    return header->size;
}

//=============================================================================
// Statistics and Debugging
//=============================================================================

bool nimcp_cache_get_stats(nimcp_cache_stats_t* stats) {
    if (!stats) return false;

    nimcp_platform_mutex_lock(&g_cache_state.mutex);
    *stats = g_cache_state.stats;
    nimcp_platform_mutex_unlock(&g_cache_state.mutex);

    return true;
}

void nimcp_cache_clear_stats(void) {
    nimcp_platform_mutex_lock(&g_cache_state.mutex);

    // Preserve memory_allocated (reflects current state)
    size_t allocated = g_cache_state.stats.memory_allocated;
    uint32_t active = g_cache_state.stats.active_allocations;

    memset(&g_cache_state.stats, 0, sizeof(nimcp_cache_stats_t));

    g_cache_state.stats.memory_allocated = allocated;
    g_cache_state.stats.active_allocations = active;

    nimcp_platform_mutex_unlock(&g_cache_state.mutex);
}

void nimcp_cache_record_reference(size_t size) {
    nimcp_platform_mutex_lock(&g_cache_state.mutex);
    g_cache_state.stats.references_created++;
    g_cache_state.stats.memory_saved += size;
    g_cache_state.stats.memory_shared += size;
    nimcp_platform_mutex_unlock(&g_cache_state.mutex);
}

void nimcp_cache_dump_allocations(void) {
    if (!g_cache_state.tracking_table) {
        printf("[CACHE] Tracking not enabled\n");
        return;
    }

    printf("\n=== Cache Allocations ===\n");

    nimcp_platform_mutex_lock(&g_cache_state.mutex);

    // Iterate through tracking table
    // (This requires hash_table_iterate function - placeholder for now)
    printf("Total active allocations: %u\n", g_cache_state.stats.active_allocations);

    nimcp_platform_mutex_unlock(&g_cache_state.mutex);

    printf("=========================\n\n");
}

void nimcp_cache_check_leaks(void) {
    if (g_cache_state.stats.active_allocations == 0) {
        printf("[CACHE] No leaks detected\n");
        return;
    }

    printf("\n=== Cache Memory Leaks ===\n");
    printf("Active allocations: %u\n", g_cache_state.stats.active_allocations);
    printf("Memory leaked: %.2f MB\n",
           g_cache_state.stats.memory_allocated / (1024.0 * 1024.0));
    printf("==========================\n\n");
}

void nimcp_cache_analyze_efficiency(void) {
    nimcp_cache_stats_t stats;
    nimcp_cache_get_stats(&stats);

    printf("\n=== Cache Efficiency Analysis ===\n");
    printf("Allocations created: %lu\n", stats.allocations_created);
    printf("References created: %lu\n", stats.references_created);
    printf("Copies triggered: %lu\n", stats.copies_triggered);
    printf("Memory allocated: %.2f MB\n",
           stats.memory_allocated / (1024.0 * 1024.0));
    printf("Memory saved (est): %.2f MB\n",
           stats.memory_saved / (1024.0 * 1024.0));

    if (stats.references_created > 0) {
        float copy_rate = (float)stats.copies_triggered / stats.references_created * 100.0F;
        printf("Copy rate: %.2f%% (lower is better)\n", copy_rate);
    }

    if (stats.memory_saved > 0) {
        float efficiency = (float)stats.memory_saved /
                          (stats.memory_saved + stats.memory_allocated) * 100.0F;
        printf("Memory efficiency: %.2f%%\n", efficiency);
    }

    printf("Active allocations: %u (peak: %u)\n",
           stats.active_allocations, stats.peak_allocations);
    printf("=================================\n\n");
}

//=============================================================================
// Advanced Functions
//=============================================================================

void* nimcp_cache_force_copy(void* ptr) {
    if (!ptr) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ptr is NULL");

        return NULL;

    }

    nimcp_cache_header_t* header = get_cache_header(ptr);
    if (!validate_cache_header(header)) {
        return NULL;
    }

    // Allocate new memory
    void* new_ptr = nimcp_cache_alloc(header->size);
    if (!new_ptr) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "new_ptr is NULL");

        return NULL;
    }

    // Copy data
    memcpy(new_ptr, ptr, header->size);

    // Release reference to original
    nimcp_cache_release(ptr);

    return new_ptr;
}

bool nimcp_cache_are_shared(const void* ptr1, const void* ptr2) {
    if (!ptr1 || !ptr2) return false;
    if (ptr1 == ptr2) return true;

    nimcp_cache_header_t* header1 = get_cache_header(ptr1);
    nimcp_cache_header_t* header2 = get_cache_header(ptr2);

    if (!validate_cache_header(header1) || !validate_cache_header(header2)) {
        return false;
    }

    // Same underlying cache allocation
    return header1 == header2;
}

bool nimcp_cache_get_info(const void* ptr, char* buffer, size_t buffer_size) {
    if (!ptr || !buffer || buffer_size == 0) return false;

    nimcp_cache_header_t* header = get_cache_header(ptr);
    if (!validate_cache_header(header)) {
        return false;
    }

    uint32_t ref_count = __atomic_load_n(&header->ref_count, __ATOMIC_ACQUIRE);

    snprintf(buffer, buffer_size,
             "Cached allocation at %p: size=%zu bytes, ref_count=%u, %s",
             ptr, header->size, ref_count,
             ref_count > 1 ? "SHARED" : "PRIVATE");

    return true;
}
