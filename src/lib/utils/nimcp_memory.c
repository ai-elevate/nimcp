/**
 * @file nimcp_memory.c
 * @brief Memory tracking implementation (standalone version)
 *
 * WHAT: Tracked memory allocator with leak detection
 * WHY: Find memory bugs during development
 * HOW: Canary guards + tracking list + pattern analysis
 */

#include "utils/nimcp_memory.h"
#include "../include/utils/nimcp_thread.h"

/**
 * WHAT: Undefine macro redirections
 * WHY: Implementation needs to call real malloc/calloc/free, not wrapped versions
 * HOW: Undefine the macros that redirect to nimcp_* functions
 */
#undef malloc
#undef calloc
#undef realloc
#undef free

#include <stdlib.h>
#include "../include/utils/nimcp_thread.h"
#include <string.h>
#include "../include/utils/nimcp_thread.h"
#include <stdio.h>
#include "../include/utils/nimcp_thread.h"
#include <time.h>
#include "../include/utils/nimcp_thread.h"
#include "../include/utils/nimcp_thread.h"
#include <stdarg.h>
#include "../include/utils/nimcp_thread.h"

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * WHAT: Time tracking structure
 * WHY: Calculate allocation lifetimes
 */
typedef struct {
    time_t sec;
    long nsec;
} timespec_internal_t;

/**
 * WHAT: Allocation size pattern tracking
 * WHY: Understand allocation behavior
 */
typedef struct size_pattern {
    size_t allocation_size;
    uint64_t allocation_count;
    uint64_t free_count;
    uint64_t total_lifetime_ms;
    struct size_pattern* next;
} size_pattern_t;

/**
 * WHAT: Individual memory block tracking
 * WHY: Track each allocation for leak detection
 */
typedef struct memory_block {
    void* ptr;
    size_t size;
    const char* file;
    int line;
    const char* function;
    timespec_internal_t allocation_time;
    uint32_t head_canary;
    uint32_t tail_canary;
    struct memory_block* next;
} memory_block_t;

/**
 * WHAT: Global memory tracking state
 * WHY: Single source of truth for all tracking
 */
typedef struct {
    uint32_t CANARY_VALUE;
    bool initialized;
    bool tracking_enabled;
    bool debug_output;
    nimcp_mutex_t lock;
    memory_block_t* blocks;
    size_pattern_t* patterns;
    nimcp_memory_stats_t stats;
} memory_state_t;

//=============================================================================
// Global State
//=============================================================================

static memory_state_t g_memory_state = {
    .CANARY_VALUE = 0xDEADBEEF,
    .initialized = false,
    .tracking_enabled = false,
    .debug_output = false
};

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * WHAT: Get current time
 * WHY: Track allocation timestamps
 */
static void get_current_time(timespec_internal_t* ts) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    ts->sec = now.tv_sec;
    ts->nsec = now.tv_nsec;
}

/**
 * WHAT: Calculate time difference in milliseconds
 * WHY: Measure allocation lifetimes
 */
static uint64_t timespec_diff_ms(const timespec_internal_t* end,
                                 const timespec_internal_t* start) {
    uint64_t sec_diff = end->sec - start->sec;
    int64_t nsec_diff = end->nsec - start->nsec;
    return (sec_diff * 1000) + (nsec_diff / 1000000);
}

/**
 * WHAT: Initialize memory system if needed
 * WHY: Lazy initialization (idempotent)
 */
static void init_if_needed(void) {
    if (!g_memory_state.initialized) {
        nimcp_mutex_init(&g_memory_state.lock, NULL);
        memset(&g_memory_state.stats, 0, sizeof(nimcp_memory_stats_t));
        g_memory_state.initialized = true;
        g_memory_state.blocks = NULL;
        g_memory_state.patterns = NULL;
    }
}

/**
 * WHAT: Add canary guards around allocation
 * WHY: Detect buffer overflows
 * HOW: Place CANARY_VALUE at head and tail
 */
static void* add_memory_guards(void* ptr, size_t size) {
    if (!ptr) return NULL;

    uint32_t* guard_ptr = (uint32_t*)ptr;
    *guard_ptr = g_memory_state.CANARY_VALUE;
    guard_ptr = (uint32_t*)((char*)ptr + size - sizeof(uint32_t));
    *guard_ptr = g_memory_state.CANARY_VALUE;

    // Return pointer after head guard
    return (char*)ptr + sizeof(uint32_t);
}

/**
 * WHAT: Check canary guards for corruption
 * WHY: Detect buffer overflows before they cause crashes
 */
static bool check_memory_guards(void* ptr, size_t size) {
    if (!ptr) return false;

    uint32_t* head_guard = (uint32_t*)((char*)ptr - sizeof(uint32_t));
    uint32_t* tail_guard = (uint32_t*)((char*)ptr + size - sizeof(uint32_t));

    if (*head_guard != g_memory_state.CANARY_VALUE ||
        *tail_guard != g_memory_state.CANARY_VALUE) {
        fprintf(stderr, "[MEMORY] Buffer overflow detected at %p\n", ptr);
        return false;
    }
    return true;
}

/**
 * WHAT: Get allocation size from tracking list
 * WHY: Needed for realloc and guard checking
 */
static size_t get_allocation_size(void* ptr) {
    if (!ptr || !g_memory_state.tracking_enabled) return 0;

    nimcp_mutex_lock(&g_memory_state.lock);
    memory_block_t* current = g_memory_state.blocks;
    size_t size = 0;

    while (current) {
        if (current->ptr == ptr) {
            size = current->size;
            break;
        }
        current = current->next;
    }

    nimcp_mutex_unlock(&g_memory_state.lock);
    return size;
}

/**
 * WHAT: Get allocation lifetime in milliseconds
 * WHY: Pattern analysis
 */
static uint64_t get_allocation_lifetime_ms(void* ptr) {
    if (!ptr || !g_memory_state.tracking_enabled) return 0;

    nimcp_mutex_lock(&g_memory_state.lock);
    memory_block_t* current = g_memory_state.blocks;
    uint64_t lifetime = 0;

    while (current) {
        if (current->ptr == ptr) {
            timespec_internal_t now;
            get_current_time(&now);
            lifetime = timespec_diff_ms(&now, &current->allocation_time);
            break;
        }
        current = current->next;
    }

    nimcp_mutex_unlock(&g_memory_state.lock);
    return lifetime;
}

/**
 * WHAT: Update allocation pattern statistics
 * WHY: Understand allocation behavior
 */
static void update_memory_patterns(void* ptr, size_t size, bool is_alloc) {
    if (!g_memory_state.tracking_enabled) return;

    size_pattern_t* pattern = g_memory_state.patterns;
    while (pattern) {
        if (pattern->allocation_size == size) {
            if (is_alloc) {
                pattern->allocation_count++;
            } else {
                pattern->free_count++;
                pattern->total_lifetime_ms += get_allocation_lifetime_ms(ptr);
            }
            return;
        }
        pattern = pattern->next;
    }

    // New pattern
    if (is_alloc) {
        pattern = (size_pattern_t*)malloc(sizeof(size_pattern_t));
        if (pattern) {
            pattern->allocation_size = size;
            pattern->allocation_count = 1;
            pattern->free_count = 0;
            pattern->total_lifetime_ms = 0;
            pattern->next = g_memory_state.patterns;
            g_memory_state.patterns = pattern;
        }
    }
}

/**
 * WHAT: Track allocation in global list
 * WHY: Enable leak detection and analysis
 */
static void track_allocation(void* ptr, size_t size,
                           const char* file, int line, const char* function) {
    if (!g_memory_state.tracking_enabled || !ptr) return;

    memory_block_t* block = (memory_block_t*)malloc(sizeof(memory_block_t));
    if (!block) return;

    block->ptr = ptr;
    block->size = size;
    block->file = file;
    block->line = line;
    block->function = function;
    get_current_time(&block->allocation_time);
    block->head_canary = g_memory_state.CANARY_VALUE;
    block->tail_canary = g_memory_state.CANARY_VALUE;

    nimcp_mutex_lock(&g_memory_state.lock);
    block->next = g_memory_state.blocks;
    g_memory_state.blocks = block;
    g_memory_state.stats.total_allocated += size;
    g_memory_state.stats.current_allocated += size;
    g_memory_state.stats.allocation_count++;

    if (g_memory_state.stats.current_allocated > g_memory_state.stats.peak_allocated) {
        g_memory_state.stats.peak_allocated = g_memory_state.stats.current_allocated;
    }

    update_memory_patterns(ptr, size, true);
    nimcp_mutex_unlock(&g_memory_state.lock);
}

/**
 * WHAT: Untrack allocation from global list
 * WHY: Remove freed blocks from tracking
 */
static void untrack_allocation(void* ptr) {
    if (!g_memory_state.tracking_enabled || !ptr) return;

    nimcp_mutex_lock(&g_memory_state.lock);

    memory_block_t* current = g_memory_state.blocks;
    memory_block_t* prev = NULL;

    while (current) {
        if (current->ptr == ptr) {
            if (prev) {
                prev->next = current->next;
            } else {
                g_memory_state.blocks = current->next;
            }
            g_memory_state.stats.current_allocated -= current->size;
            g_memory_state.stats.free_count++;
            update_memory_patterns(ptr, current->size, false);
            free(current);
            break;
        }
        prev = current;
        current = current->next;
    }

    nimcp_mutex_unlock(&g_memory_state.lock);
}

/**
 * WHAT: Check for double-free
 * WHY: Prevent double-free crashes
 */
static bool check_double_free(void* ptr) {
    if (!ptr) return false;

    nimcp_mutex_lock(&g_memory_state.lock);
    memory_block_t* current = g_memory_state.blocks;
    bool found = false;

    while (current) {
        if (current->ptr == ptr) {
            found = true;
            break;
        }
        current = current->next;
    }

    nimcp_mutex_unlock(&g_memory_state.lock);

    if (!found) {
        fprintf(stderr, "[MEMORY] Double-free detected at %p\n", ptr);
        return true;
    }

    return false;
}

//=============================================================================
// Public API Implementation
//=============================================================================

void* nimcp_malloc(size_t size) {
    init_if_needed();

    size_t total_size = size + (2 * sizeof(uint32_t));
    void* ptr = malloc(total_size);

    if (ptr) {
        ptr = add_memory_guards(ptr, total_size);
        track_allocation(ptr, size, __FILE__, __LINE__, __func__);

        if (g_memory_state.debug_output) {
            printf("[MEMORY] Allocated: %zu bytes at %p\n", size, ptr);
        }
    } else {
        nimcp_mutex_lock(&g_memory_state.lock);
        g_memory_state.stats.failed_allocations++;
        nimcp_mutex_unlock(&g_memory_state.lock);

        if (g_memory_state.debug_output) {
            fprintf(stderr, "[MEMORY] Allocation failed: %zu bytes\n", size);
        }
    }

    return ptr;
}

void* nimcp_calloc(size_t count, size_t size) {
    init_if_needed();

    size_t total_size = (count * size) + (2 * sizeof(uint32_t));
    void* ptr = calloc(1, total_size);

    if (ptr) {
        ptr = add_memory_guards(ptr, total_size);
        track_allocation(ptr, count * size, __FILE__, __LINE__, __func__);

        if (g_memory_state.debug_output) {
            printf("[MEMORY] Allocated (calloc): %zu bytes at %p\n",
                   count * size, ptr);
        }
    } else {
        nimcp_mutex_lock(&g_memory_state.lock);
        g_memory_state.stats.failed_allocations++;
        nimcp_mutex_unlock(&g_memory_state.lock);

        if (g_memory_state.debug_output) {
            fprintf(stderr, "[MEMORY] Allocation failed (calloc): %zu bytes\n",
                   count * size);
        }
    }

    return ptr;
}

void* nimcp_realloc(void* ptr, size_t new_size) {
    init_if_needed();

    if (!ptr) return nimcp_malloc(new_size);

    size_t old_size = get_allocation_size(ptr);
    if (new_size < old_size && g_memory_state.debug_output) {
        printf("[MEMORY] Realloc reducing size from %zu to %zu at %p\n",
               old_size, new_size, ptr);
    }

    untrack_allocation(ptr);
    size_t total_size = new_size + (2 * sizeof(uint32_t));
    void* real_ptr = (char*)ptr - sizeof(uint32_t);
    void* new_ptr = realloc(real_ptr, total_size);

    if (new_ptr) {
        new_ptr = add_memory_guards(new_ptr, total_size);
        track_allocation(new_ptr, new_size, __FILE__, __LINE__, __func__);

        if (g_memory_state.debug_output) {
            printf("[MEMORY] Reallocated: %zu bytes at %p (old: %p)\n",
                   new_size, new_ptr, ptr);
        }
    } else {
        nimcp_mutex_lock(&g_memory_state.lock);
        g_memory_state.stats.failed_allocations++;
        nimcp_mutex_unlock(&g_memory_state.lock);

        if (g_memory_state.debug_output) {
            fprintf(stderr, "[MEMORY] Reallocation failed: %zu bytes\n", new_size);
        }
    }

    return new_ptr;
}

char* nimcp_strdup(const char* str) {
    if (!str) return NULL;

    size_t len = strlen(str) + 1;
    char* new_str = (char*)nimcp_malloc(len);
    if (!new_str) return NULL;

    memcpy(new_str, str, len);
    return new_str;
}

void nimcp_free(void* ptr) {
    if (!ptr) return;
    init_if_needed();

    if (check_double_free(ptr)) {
        return;
    }

    if (g_memory_state.debug_output) {
        printf("[MEMORY] Freed at %p\n", ptr);
    }

    if (g_memory_state.tracking_enabled) {
        if (!check_memory_guards(ptr, get_allocation_size(ptr))) {
            fprintf(stderr, "[MEMORY] Memory corruption detected during free at %p\n", ptr);
        }
    }

    void* real_ptr = (char*)ptr - sizeof(uint32_t);
    untrack_allocation(ptr);
    free(real_ptr);
}

void* nimcp_aligned_malloc(size_t size, size_t alignment) {
    init_if_needed();

    void* ptr;
    int result = posix_memalign(&ptr, alignment, size);
    if (result != 0) {
        nimcp_mutex_lock(&g_memory_state.lock);
        g_memory_state.stats.failed_allocations++;
        nimcp_mutex_unlock(&g_memory_state.lock);
        return NULL;
    }
    track_allocation(ptr, size, __FILE__, __LINE__, __func__);
    return ptr;
}

void nimcp_aligned_free(void* ptr) {
    if (!ptr) return;
    untrack_allocation(ptr);
    free(ptr);
}

void nimcp_memory_init(void) {
    init_if_needed();
}

void nimcp_memory_cleanup(void) {
    if (!g_memory_state.initialized) return;

    nimcp_mutex_lock(&g_memory_state.lock);

    // Free all leaked blocks
    memory_block_t* current = g_memory_state.blocks;
    while (current) {
        memory_block_t* next = current->next;
        if (g_memory_state.debug_output) {
            fprintf(stderr, "[MEMORY] Leak detected: %zu bytes at %p\n",
                   current->size, current->ptr);
        }
        free((char*)current->ptr - sizeof(uint32_t));
        free(current);
        current = next;
    }

    // Free patterns
    size_pattern_t* pattern = g_memory_state.patterns;
    while (pattern) {
        size_pattern_t* next = pattern->next;
        free(pattern);
        pattern = next;
    }

    g_memory_state.blocks = NULL;
    g_memory_state.patterns = NULL;
    memset(&g_memory_state.stats, 0, sizeof(nimcp_memory_stats_t));

    nimcp_mutex_unlock(&g_memory_state.lock);
    nimcp_mutex_destroy(&g_memory_state.lock);
    g_memory_state.initialized = false;
}

bool nimcp_memory_get_stats(nimcp_memory_stats_t* stats) {
    if (!stats) return false;

    nimcp_mutex_lock(&g_memory_state.lock);
    memcpy(stats, &g_memory_state.stats, sizeof(nimcp_memory_stats_t));
    nimcp_mutex_unlock(&g_memory_state.lock);

    return true;
}

void nimcp_memory_clear_stats(void) {
    nimcp_mutex_lock(&g_memory_state.lock);
    memset(&g_memory_state.stats, 0, sizeof(nimcp_memory_stats_t));
    nimcp_mutex_unlock(&g_memory_state.lock);
}

void nimcp_memory_enable_tracking(bool enable) {
    init_if_needed();
    g_memory_state.tracking_enabled = enable;
}

void nimcp_memory_enable_debug_output(bool enable) {
    init_if_needed();
    g_memory_state.debug_output = enable;
}

void nimcp_memory_dump_allocations(void) {
    if (!g_memory_state.tracking_enabled) {
        fprintf(stderr, "[MEMORY] Tracking is not enabled\n");
        return;
    }

    nimcp_mutex_lock(&g_memory_state.lock);

    printf("\n========== Memory Allocation Dump ==========\n");
    printf("Total allocated: %zu bytes\n", g_memory_state.stats.total_allocated);
    printf("Current allocated: %zu bytes\n", g_memory_state.stats.current_allocated);
    printf("Peak allocated: %zu bytes\n", g_memory_state.stats.peak_allocated);
    printf("Allocation count: %zu\n", g_memory_state.stats.allocation_count);
    printf("Free count: %zu\n", g_memory_state.stats.free_count);
    printf("Failed allocations: %zu\n", g_memory_state.stats.failed_allocations);
    printf("\nCurrent allocations:\n");

    memory_block_t* current = g_memory_state.blocks;
    uint32_t count = 0;
    while (current) {
        printf("  [%u] %p: %zu bytes (from %s:%d in %s())\n",
               count++, current->ptr, current->size,
               current->file, current->line, current->function);
        current = current->next;
    }
    printf("============================================\n\n");

    nimcp_mutex_unlock(&g_memory_state.lock);
}

void nimcp_memory_check_leaks(void) {
    if (!g_memory_state.tracking_enabled) {
        fprintf(stderr, "[MEMORY] Tracking is not enabled\n");
        return;
    }

    nimcp_mutex_lock(&g_memory_state.lock);

    memory_block_t* current = g_memory_state.blocks;
    size_t total_leaked = 0;
    uint32_t leak_count = 0;

    if (!current) {
        printf("\n[MEMORY] ✓ No memory leaks detected!\n\n");
        nimcp_mutex_unlock(&g_memory_state.lock);
        return;
    }

    printf("\n========== Memory Leak Report ==========\n");
    while (current) {
        printf("[LEAK %u]\n", leak_count + 1);
        printf("  Address: %p\n", current->ptr);
        printf("  Size: %zu bytes\n", current->size);
        printf("  Function: %s()\n", current->function);
        printf("  Location: %s:%d\n", current->file, current->line);
        printf("  Lifetime: %lu ms\n", get_allocation_lifetime_ms(current->ptr));
        printf("\n");

        total_leaked += current->size;
        leak_count++;
        current = current->next;
    }

    printf("Total leaks: %u blocks, %zu bytes\n", leak_count, total_leaked);
    printf("========================================\n\n");

    nimcp_mutex_unlock(&g_memory_state.lock);
}

void nimcp_memory_analyze_patterns(void) {
    if (!g_memory_state.tracking_enabled) {
        fprintf(stderr, "[MEMORY] Tracking is not enabled\n");
        return;
    }

    nimcp_mutex_lock(&g_memory_state.lock);

    printf("\n========== Allocation Pattern Analysis ==========\n");
    size_pattern_t* pattern = g_memory_state.patterns;

    if (!pattern) {
        printf("No allocation patterns recorded\n");
        nimcp_mutex_unlock(&g_memory_state.lock);
        return;
    }

    while (pattern) {
        double avg_lifetime = pattern->free_count > 0 ?
            (double)pattern->total_lifetime_ms / pattern->free_count : 0;

        printf("Size: %zu bytes\n", pattern->allocation_size);
        printf("  Allocations: %lu\n", pattern->allocation_count);
        printf("  Frees: %lu\n", pattern->free_count);
        printf("  Avg lifetime: %.2f ms\n", avg_lifetime);

        if (pattern->allocation_count > pattern->free_count) {
            printf("  ⚠ Potential leak: %lu blocks not freed\n",
                   pattern->allocation_count - pattern->free_count);
        }
        printf("\n");

        pattern = pattern->next;
    }
    printf("=================================================\n\n");

    nimcp_mutex_unlock(&g_memory_state.lock);
}
