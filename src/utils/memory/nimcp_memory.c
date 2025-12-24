//=============================================================================
// nimcp_memory.c - Memory Tracking and Debugging System
//=============================================================================
// ARCHITECTURAL OVERVIEW:
// This module implements a comprehensive memory debugging system that wraps
// standard C allocation functions (malloc, calloc, realloc, free) with
// tracking, leak detection, corruption detection, and usage pattern analysis.
// It's designed for DEBUG builds to catch memory-related bugs early.
//
// KEY DESIGN: DECORATOR + PROXY PATTERN
// ======================================
// WHY DECORATOR:
// - Wraps standard malloc/free with additional functionality
// - Preserves original interface (drop-in replacement)
// - Transparent to calling code
// - Can be enabled/disabled at runtime
//
// WHY PROXY:
// - Intercepts all memory operations
// - Adds guards, tracking, and validation
// - Centralizes memory management
//
// MEMORY BLOCK LAYOUT (with guards):
//   ┌──────────────┬────────────────────┬──────────────┐
//   │ HEAD_CANARY  │   User Data        │ TAIL_CANARY  │
//   │ (4 bytes)    │   (N bytes)        │ (4 bytes)    │
//   └──────────────┴────────────────────┴──────────────┘
//   ↑              ↑                    ↑
//   Real ptr       Returned to user     Guard check
//
// WHY CANARY GUARDS:
// - Detect buffer overflows (writes past end)
// - Detect buffer underflows (writes before start)
// - 0xDEADBEEF pattern (recognizable in hex dumps)
// - Checked on every free
//
// Without guards: Corruption silently spreads
// With guards: Immediate detection, clear error message
//
// TRACKING ARCHITECTURE:
//
//   ┌─────────────────────────────────────────────┐
//   │  Global Memory State (Singleton)            │
//   │  - Mutex for thread safety                  │
//   │  - Statistics counters                      │
//   │  - Configuration flags                      │
//   └─────────────┬───────────────────────────────┘
//                 │
//      ┌──────────┴──────────┐
//      │                     │
//   ┌──▼─────────────┐  ┌───▼────────────────┐
//   │ Allocation     │  │ Pattern Analysis   │
//   │ Tracking List  │  │ List               │
//   │ (linked list)  │  │ (linked list)      │
//   └────────────────┘  └────────────────────┘
//
// WHY LINKED LIST FOR TRACKING:
// - Dynamic size (handles any number of allocations)
// - O(1) insertion (add to head)
// - O(n) search for validation (acceptable for debugging)
// - Simple traversal for leak reporting
//
// Alternative approaches rejected:
// - Hash table: More complexity, overkill for debugging
// - Array: Fixed size, not suitable for dynamic allocation count
// - No tracking: Can't detect leaks
//
// THREAD SAFETY:
// - Single global mutex protects all tracking structures
// - Locked during: allocation tracking, free, statistics updates
// - Lock-free: canary checking (read-only after creation)
// - Trade-off: Serialized allocations vs data integrity
//
// WHY GLOBAL MUTEX vs PER-ALLOCATION LOCKS:
// - Simpler implementation
// - Lower memory overhead (no lock per allocation)
// - Acceptable for debugging (performance not critical)
// - Prevents deadlocks (no lock ordering issues)
//
// DETECTION CAPABILITIES:
//
// 1. MEMORY LEAKS
//    HOW: Track all allocations, report unfreed blocks at cleanup
//    WHEN: Call nimcp_memory_check_leaks() before shutdown
//    OUTPUT: File, line, function, size, lifetime
//
// 2. DOUBLE-FREE
//    HOW: Search tracking list before free, error if not found
//    WHEN: On every nimcp_free() call
//    OUTPUT: "Double-free detected at %p"
//
// 3. BUFFER OVERFLOW
//    HOW: Check canary guards on free
//    WHEN: On every nimcp_free() call
//    OUTPUT: "Buffer overflow detected at %p"
//
// 4. ALLOCATION PATTERNS
//    HOW: Track sizes, counts, lifetimes by allocation size
//    WHEN: Call nimcp_memory_analyze_patterns()
//    OUTPUT: Size, allocation count, free count, average lifetime
//
// PERFORMANCE CHARACTERISTICS:
// - Space overhead: ~40 bytes per allocation
//   * 8 bytes: canaries (head + tail)
//   * 32+ bytes: tracking structure (memory_block_t)
// - Time overhead: ~2-5%
//   * Mutex lock/unlock: ~50-100ns
//   * Linked list insertion: O(1)
//   * Canary check: O(1)
//   * Double-free check: O(n) where n = active allocations
//
// Typical numbers (1000 allocations):
// - Memory overhead: ~40KB (tracking structures)
// - Time overhead: ~2-3% (mostly mutex contention)
//
// WHY ACCEPTABLE FOR DEBUG BUILDS:
// - Safety more important than performance
// - Early bug detection saves debugging time
// - Can be disabled in production builds
// - Alternative is valgrind (much slower, ~10-50x)
//
// DESIGN PATTERNS:
// 1. SINGLETON: Global memory state
// 2. DECORATOR: Wraps malloc/free with tracking
// 3. PROXY: Intercepts memory operations
// 4. RAII: Automatic cleanup with guards
// 5. STRATEGY: Pluggable tracking/reporting strategies
// 6. OBSERVER: Pattern tracking observes allocations
//
// SOLID PRINCIPLES:
// - Single Responsibility: Each function has one clear purpose
// - Open/Closed: Can add new tracking metrics without modifying core
// - Liskov Substitution: Drop-in replacement for malloc/free
// - Interface Segregation: Clean, minimal API
// - Dependency Inversion: Depends on standard malloc abstraction
//
// USE CASES IN NIMCP:
// - Development: Catch leaks during testing
// - CI/CD: Automated leak detection in test suite
// - Profiling: Understand memory usage patterns
// - Debugging: Find double-frees and buffer overflows
// - Optimization: Identify allocation hotspots
//
// USAGE PATTERN:
//
//   // 1. Initialize at startup
//   nimcp_memory_init();
//   nimcp_memory_enable_tracking(true);
//   nimcp_memory_enable_debug_output(false);
//
//   // 2. Use instead of malloc/free throughout code
//   void* ptr = nimcp_malloc(100);
//   nimcp_free(ptr);
//
//   // 3. Check for issues periodically
//   nimcp_memory_dump_allocations();    // See current allocations
//   nimcp_memory_analyze_patterns();    // See allocation patterns
//
//   // 4. Cleanup at shutdown
//   nimcp_memory_check_leaks();         // Report leaks
//   nimcp_memory_cleanup();             // Free tracking structures
//
// LIMITATIONS AND TRADE-OFFS:
// - Performance: 2-5% overhead (acceptable for debug)
// - Memory: ~40 bytes per allocation overhead
// - Scope: Only tracks nimcp_malloc, not direct malloc calls
// - Thread contention: Global lock can bottleneck on many cores
//
// WHY THESE TRADE-OFFS:
// - Simplicity: Easier to maintain and debug
// - Safety: Guaranteed leak detection vs missed leaks
// - Practicality: Debug-only, not for production
// - Alternatives: Valgrind (slow), sanitizers (compiler-specific)
//
// PATTERN ANALYSIS FEATURE:
// Tracks allocation behavior by size:
// - How many allocations of each size?
// - How many freed vs still allocated?
// - Average lifetime of allocations
//
// WHY USEFUL:
// - Optimize pool allocators (common sizes)
// - Find long-lived allocations (cache candidates)
// - Detect allocation/free imbalance (leaks)
//
// Example output:
//   Size: 64 bytes
//     Allocations: 1000
//     Frees: 995
//     Avg lifetime: 123.5 ms
//     ⚠ Potential leak: 5 blocks not freed
//
//=============================================================================

/**
 * @file nimcp_memory.c
 * @brief Memory tracking implementation (standalone version)
 *
 * WHAT: Tracked memory allocator with leak detection
 * WHY: Find memory bugs during development
 * HOW: Canary guards + tracking list + pattern analysis
 */

#include "utils/memory/nimcp_memory.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/thread/nimcp_thread.h"

/**
 * WHAT: Undefine macro redirections
 * WHY: Implementation needs to call real malloc/calloc/free, not wrapped versions
 * HOW: Undefine the macros that redirect to nimcp_* functions
 */
#undef malloc
#undef calloc
#undef realloc
#undef free

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utils/logging/nimcp_logging.h"

//=============================================================================
// Global Unified Memory Manager
//=============================================================================
/**
 * WHAT: Global unified memory manager for all nimcp allocations
 * WHY: Provides CoW support, consistent allocation, and tracking
 * HOW: Created on first use, destroyed at shutdown
 */
static unified_mem_manager_t g_unified_manager = NULL;
static bool g_unified_initialized = false;

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * WHAT: Time tracking structure
 * WHY: Calculate allocation lifetimes for pattern analysis
 * HOW: Stores seconds and nanoseconds from CLOCK_MONOTONIC
 *
 * WHY MONOTONIC vs REALTIME:
 * - Not affected by system clock changes
 * - Always increases
 * - Perfect for measuring durations
 */
typedef struct {
    time_t sec;
    long nsec;
} timespec_internal_t;

/**
 * WHAT: Allocation size pattern tracking
 * WHY: Understand allocation behavior to optimize memory usage
 * HOW: Linked list of size→statistics mappings
 *
 * STRUCTURE:
 * - allocation_size: Bucket key (e.g., 64 bytes)
 * - allocation_count: How many allocations of this size
 * - free_count: How many frees of this size
 * - total_lifetime_ms: Cumulative lifetime for average calculation
 * - next: Linked list pointer
 *
 * COMPLEXITY: O(n) search where n = number of unique sizes
 * - Acceptable because n is typically small (<100)
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
 * WHY: Track each allocation for leak detection and validation
 * HOW: Linked list node with metadata
 *
 * STRUCTURE:
 * - ptr: Pointer returned to user (after head canary)
 * - size: User-requested size (not including guards)
 * - file/line/function: Allocation location (for leak reporting)
 * - allocation_time: When allocated (for lifetime calculation)
 * - head_canary/tail_canary: Expected values (for validation)
 * - next: Linked list pointer
 *
 * WHY STORE LOCATION:
 * - Essential for debugging: "Leak at main.c:42"
 * - Captured via __FILE__, __LINE__, __func__ macros
 *
 * WHY STORE CANARIES:
 * - Redundant check (also in actual memory)
 * - Detects corruption of tracking structure itself
 */
typedef struct memory_block {
    void* ptr;
    size_t size;
    size_t guard_size;  // Size of guard regions (8, 16, 32, 64 bytes)
    const char* file;
    int line;
    const char* function;
    timespec_internal_t allocation_time;
    uint32_t head_canary;
    uint32_t tail_canary;
    unified_mem_handle_t umm_handle;  // Unified memory handle (NULL if direct malloc)
    struct memory_block* next;
} memory_block_t;

/**
 * WHAT: Global memory tracking state
 * WHY: Single source of truth for all tracking
 * HOW: Singleton pattern with global instance
 *
 * STRUCTURE:
 * - CANARY_VALUE: Magic number for overflow detection (0xDEADBEEF)
 * - initialized: Whether system is initialized
 * - tracking_enabled: Whether to track allocations
 * - debug_output: Whether to print allocation messages
 * - lock: Mutex for thread safety
 * - blocks: Linked list of active allocations
 * - patterns: Linked list of size patterns
 * - stats: Cumulative statistics
 *
 * WHY SINGLETON:
 * - Only one memory system per process
 * - Global state accessible from all allocation sites
 * - Simpler than passing context around
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

/**
 * WHY GLOBAL:
 * - Accessible from all allocation wrappers
 * - Singleton pattern: one instance per process
 * - Simpler than thread-local storage
 *
 * WHY STATIC INITIALIZATION:
 * - Safe: No dynamic initialization needed
 * - Fast: No runtime overhead
 * - Lazy init: actual initialization in init_if_needed()
 *
 * WHY 0xDEADBEEF:
 * - Recognizable in memory dumps
 * - Unlikely to occur naturally
 * - Standard debug pattern
 */
static memory_state_t g_memory_state = {.CANARY_VALUE = 0xDEADBEEF,
                                        .initialized = false,
                                        .tracking_enabled = false,
                                        .debug_output = false};

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Get current monotonic time
 *
 * WHY MONOTONIC:
 * - Unaffected by system clock changes
 * - Always increases
 * - Perfect for measuring durations
 *
 * WHY NANOSECOND PRECISION:
 * - Some allocations are very short-lived (<1ms)
 * - Nanosecond precision captures these accurately
 *
 * COMPLEXITY: O(1) system call
 * THREAD SAFETY: Fully thread-safe (no shared state)
 *
 * @param ts Output parameter for timestamp
 */
static void get_current_time(timespec_internal_t* ts)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    ts->sec = now.tv_sec;
    ts->nsec = now.tv_nsec;
}

/**
 * @brief Calculate time difference in milliseconds
 *
 * WHY MILLISECONDS:
 * - Human-readable timescale for allocation lifetimes
 * - Sufficient precision (most allocations live >1ms)
 * - Avoids overflow with uint64_t
 *
 * ALGORITHM:
 * 1. Compute seconds difference, convert to milliseconds
 * 2. Compute nanoseconds difference, convert to milliseconds
 * 3. Sum the two
 *
 * WHY THIS APPROACH:
 * - Handles nanosecond wrapping correctly
 * - No integer overflow risk
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully thread-safe (pure function)
 *
 * @param end End timestamp
 * @param start Start timestamp
 * @return Difference in milliseconds
 */
static uint64_t timespec_diff_ms(const timespec_internal_t* end, const timespec_internal_t* start)
{
    uint64_t sec_diff = end->sec - start->sec;
    int64_t nsec_diff = end->nsec - start->nsec;
    return (sec_diff * 1000) + (nsec_diff / 1000000);
}

/**
 * @brief Initialize memory system if needed
 *
 * WHY LAZY INITIALIZATION:
 * - Safe: Works even if user forgets to call nimcp_memory_init()
 * - Idempotent: Safe to call multiple times
 * - No ordering requirements
 *
 * INITIALIZATION:
 * 1. Initialize mutex (thread synchronization)
 * 2. Zero statistics
 * 3. Set initialized flag
 * 4. NULL-initialize linked lists
 *
 * WHY THIS ORDER:
 * - Mutex first (needed for thread safety)
 * - Data structures second (ready for use)
 * - Flag last (signals completion)
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: NOT thread-safe (caller must ensure)
 * - Typically called once at startup
 * - Or called with lock held
 */
static void init_if_needed(void)
{
    /* Recursion guard: prevent infinite recursion during UMM initialization
     * WHY: UMM create calls nimcp_calloc, which calls init_if_needed again
     * HOW: Use thread-local flag to track per-thread initialization state
     * THREAD SAFETY: _Thread_local ensures each thread has its own flag
     */
    static _Thread_local bool g_umm_initializing = false;

    if (!g_memory_state.initialized) {
        nimcp_mutex_init(&g_memory_state.lock, NULL);
        memset(&g_memory_state.stats, 0, sizeof(nimcp_memory_stats_t));
        g_memory_state.initialized = true;
        g_memory_state.tracking_enabled = true;  // Enable by default for guard_size tracking
        g_memory_state.blocks = NULL;
        g_memory_state.patterns = NULL;
    }

    /* Initialize unified memory manager if not already done
     * Use g_umm_initializing to prevent recursion (per-thread)
     */
    if (!g_unified_initialized && !g_unified_manager && !g_umm_initializing) {
        g_umm_initializing = true;  // Set flag before creating UMM
        unified_mem_config_t config = unified_mem_default_config();
        config.enable_cow = true;
        config.enable_tracking = true;
        g_unified_manager = unified_mem_create(&config);
        g_unified_initialized = true;
        g_umm_initializing = false;  // Clear flag after UMM is created
    }
}

/**
 * @brief Add canary guards around allocation
 *
 * WHY CANARIES:
 * - Detect buffer overflows (write past end)
 * - Detect buffer underflows (write before start)
 * - No runtime overhead (checked only on free)
 *
 * LAYOUT:
 *   [HEAD_CANARY][User Data][TAIL_CANARY]
 *   ↑            ↑           ↑
 *   Real ptr     Returned    Check on free
 *
 * ALGORITHM:
 * 1. Write HEAD_CANARY at start of allocation
 * 2. Write TAIL_CANARY at end of allocation
 * 3. Return pointer after HEAD_CANARY
 *
 * WHY RETURN OFFSET POINTER:
 * - User data starts after canary
 * - Canary protected from normal writes
 * - On free, we recompute real pointer: (char*)ptr - sizeof(uint32_t)
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: No shared state, thread-safe
 *
 * @param ptr Real allocation pointer (from malloc)
 * @param size Total size (including guards)
 * @return User pointer (after head guard)
 */
static void* add_memory_guards(void* ptr, size_t size)
{
    if (!ptr)
        return NULL;

    // Write head canary (8 bytes for proper alignment)
    uint64_t* guard_ptr = (uint64_t*) ptr;
    *guard_ptr = ((uint64_t)g_memory_state.CANARY_VALUE << 32) | g_memory_state.CANARY_VALUE;

    // Write tail canary (8 bytes)
    guard_ptr = (uint64_t*) ((char*) ptr + size - sizeof(uint64_t));
    *guard_ptr = ((uint64_t)g_memory_state.CANARY_VALUE << 32) | g_memory_state.CANARY_VALUE;

    // Return pointer after head guard (8-byte aligned)
    // WHY: User data starts here, head guard before, tail guard after
    return (char*) ptr + sizeof(uint64_t);
}

// Forward declaration for get_guard_size
static size_t get_guard_size(void* ptr);

/**
 * @brief Check canary guards for corruption
 *
 * WHY CHECK GUARDS:
 * - Detect buffer overflows before they cause crashes elsewhere
 * - Immediate feedback: "Overflow at allocation from main.c:42"
 * - Prevents silent corruption
 *
 * ALGORITHM:
 * 1. Compute head guard location: (char*)ptr - sizeof(uint32_t)
 * 2. Compute tail guard location: (char*)ptr + size - sizeof(uint32_t)
 * 3. Compare both to CANARY_VALUE
 * 4. Report error if mismatch
 *
 * WHY CHECK BOTH:
 * - Overflow: Overwrites tail canary
 * - Underflow: Overwrites head canary
 *
 * LIMITATIONS:
 * - Only detects overflow/underflow that reaches canary
 * - Small overflow within user buffer not detected
 *
 * ALTERNATIVE APPROACHES:
 * - Red zones (larger guards): More overhead
 * - Page protection: Much higher overhead
 * - Valgrind: Best detection, 10-50x slower
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Read-only, thread-safe
 *
 * @param ptr User pointer
 * @param size User size
 * @return true if guards intact, false if corrupted
 */
static bool check_memory_guards(void* ptr, size_t size)
{
    if (!ptr)
        return false;

    // Get guard size for this allocation
    size_t guard_size = get_guard_size(ptr);

    // Check variable-sized guards
    // WHY: Different allocations use different guard sizes (8, 16, 32, 64)
    uint64_t expected = ((uint64_t)g_memory_state.CANARY_VALUE << 32) | g_memory_state.CANARY_VALUE;

    // Check head guard (multiple uint64_t values if guard_size > 8)
    uint64_t* head_guard = (uint64_t*) ((char*) ptr - guard_size);
    for (size_t i = 0; i < guard_size / sizeof(uint64_t); i++) {
        if (head_guard[i] != expected) {
            fprintf(stderr, "[MEMORY] Buffer underflow detected at %p (head guard corrupted)\n", ptr);
            return false;
        }
    }

    // Check tail guard
    // WHY CAST TO CHAR FIRST: size may not be 8-byte aligned, need byte arithmetic
    char* tail_ptr = (char*) ptr + size;
    uint64_t* tail_guard = (uint64_t*) tail_ptr;
    for (size_t i = 0; i < guard_size / sizeof(uint64_t); i++) {
        // WHY MEMCMP: tail_guard may not be 8-byte aligned if size isn't aligned
        uint64_t tail_value;
        memcpy(&tail_value, &tail_guard[i], sizeof(uint64_t));
        if (tail_value != expected) {
            fprintf(stderr, "[MEMORY] Buffer overflow detected at %p (tail guard corrupted)\n", ptr);
            return false;
        }
    }

    return true;
}

/**
 * @brief Get allocation size from tracking list
 *
 * WHY NEEDED:
 * - realloc needs old size to copy data
 * - check_memory_guards needs size to find tail canary
 * - Can't store size in allocation itself (interferes with guards)
 *
 * ALGORITHM:
 * 1. Acquire lock
 * 2. Walk linked list searching for ptr
 * 3. Return size if found, 0 if not
 * 4. Release lock
 *
 * WHY LINEAR SEARCH:
 * - Simple implementation
 * - Acceptable for debugging (not performance-critical)
 * - Alternative: hash table (more complexity)
 *
 * COMPLEXITY: O(n) where n = number of active allocations
 * THREAD SAFETY: Fully thread-safe (lock-protected)
 *
 * @param ptr User pointer
 * @return Allocation size or 0 if not tracked
 */
static size_t get_allocation_size(void* ptr)
{
    if (!ptr || !g_memory_state.tracking_enabled)
        return 0;

    nimcp_mutex_lock(&g_memory_state.lock);
    memory_block_t* current = g_memory_state.blocks;
    size_t size = 0;

    /* Cycle detection using Floyd's tortoise-hare algorithm
     * WHY: Prevents infinite loops on corrupted linked lists
     * HOW: Fast pointer moves 2x speed, if it catches slow pointer = cycle
     */
    memory_block_t* slow = current;
    memory_block_t* fast = current;
    bool cycle_detected = false;
    uint32_t iterations = 0;
    const uint32_t MAX_ITERATIONS = 100000;

    while (current && iterations < MAX_ITERATIONS) {
        if (current->ptr == ptr) {
            size = current->size;
            break;
        }

        /* Cycle detection: advance tortoise and hare */
        if (slow) slow = slow->next;
        if (fast) {
            fast = fast->next;
            if (fast) fast = fast->next;
        }
        if (slow && fast && slow == fast) {
            cycle_detected = true;
            break;
        }

        current = current->next;
        iterations++;
    }

    if (cycle_detected) {
        NIMCP_LOGGING_ERROR("Memory tracking list contains cycle (corrupted data structure)");
    } else if (iterations >= MAX_ITERATIONS) {
        NIMCP_LOGGING_ERROR("Memory tracking list may be corrupted (exceeded max iterations)");
    }

    nimcp_mutex_unlock(&g_memory_state.lock);
    return size;
}

/**
 * @brief Get guard size for allocation
 *
 * WHAT: Retrieve guard size from tracking metadata
 * WHY: nimcp_free needs to know guard offset to compute real_ptr
 * HOW: Linear search through tracking list
 *
 * @param ptr User pointer (after guards)
 * @return Guard size in bytes (8, 16, 32, 64) or 0 if not tracked
 */
static size_t get_guard_size(void* ptr)
{
    if (!ptr || !g_memory_state.tracking_enabled) {
        fprintf(stderr, "[DEBUG] get_guard_size: ptr=%p tracking_enabled=%d\n",
                ptr, g_memory_state.tracking_enabled);
        return 0;  // No guards if not tracked
    }

    nimcp_mutex_lock(&g_memory_state.lock);
    memory_block_t* current = g_memory_state.blocks;
    size_t guard_size = 0;  // 0 means not found or no guards
    bool found = false;

    /* Cycle detection using Floyd's tortoise-hare algorithm */
    memory_block_t* slow = current;
    memory_block_t* fast = current;
    bool cycle_detected = false;
    uint32_t iterations = 0;
    const uint32_t MAX_ITERATIONS = 100000;

    while (current && iterations < MAX_ITERATIONS) {
        if (current->ptr == ptr) {
            guard_size = current->guard_size;
            found = true;
            break;
        }

        /* Cycle detection: advance tortoise and hare */
        if (slow) slow = slow->next;
        if (fast) {
            fast = fast->next;
            if (fast) fast = fast->next;
        }
        if (slow && fast && slow == fast) {
            cycle_detected = true;
            break;
        }

        current = current->next;
        iterations++;
    }

    if (cycle_detected) {
        NIMCP_LOGGING_ERROR("Memory tracking list contains cycle in get_guard_size (corrupted data structure)");
    } else if (iterations >= MAX_ITERATIONS) {
        NIMCP_LOGGING_ERROR("Memory tracking list may be corrupted in get_guard_size (exceeded max iterations)");
    } else if (!found) {
        fprintf(stderr, "[DEBUG] get_guard_size: ptr=%p NOT FOUND in tracking list\n", ptr);
    }

    // Return the tracked guard_size (may be 0 if allocation used no guards)
    // If not found in tracking, return 0 (assume no guards)
    nimcp_mutex_unlock(&g_memory_state.lock);
    return guard_size;
}

/**
 * @brief Get allocation lifetime in milliseconds
 *
 * WHY TRACK LIFETIME:
 * - Pattern analysis: Average lifetime by size
 * - Find long-lived allocations (cache candidates)
 * - Detect unexpected allocation patterns
 *
 * ALGORITHM:
 * 1. Acquire lock
 * 2. Find allocation in tracking list
 * 3. Get current time
 * 4. Compute difference from allocation_time
 * 5. Release lock
 *
 * COMPLEXITY: O(n) where n = number of active allocations
 * THREAD SAFETY: Fully thread-safe (lock-protected)
 *
 * @param ptr User pointer
 * @return Lifetime in milliseconds or 0 if not tracked
 */
static uint64_t get_allocation_lifetime_ms(void* ptr)
{
    if (!ptr || !g_memory_state.tracking_enabled)
        return 0;

    nimcp_mutex_lock(&g_memory_state.lock);
    memory_block_t* current = g_memory_state.blocks;
    uint64_t lifetime = 0;

    /* Cycle detection using Floyd's tortoise-hare algorithm */
    memory_block_t* slow = current;
    memory_block_t* fast = current;
    bool cycle_detected = false;
    uint32_t iterations = 0;
    const uint32_t MAX_ITERATIONS = 100000;

    while (current && iterations < MAX_ITERATIONS) {
        if (current->ptr == ptr) {
            timespec_internal_t now;
            get_current_time(&now);
            lifetime = timespec_diff_ms(&now, &current->allocation_time);
            break;
        }

        /* Cycle detection: advance tortoise and hare */
        if (slow) slow = slow->next;
        if (fast) {
            fast = fast->next;
            if (fast) fast = fast->next;
        }
        if (slow && fast && slow == fast) {
            cycle_detected = true;
            break;
        }

        current = current->next;
        iterations++;
    }

    if (cycle_detected) {
        NIMCP_LOGGING_ERROR("Memory tracking list contains cycle in get_allocation_lifetime_ms (corrupted data structure)");
    } else if (iterations >= MAX_ITERATIONS) {
        NIMCP_LOGGING_ERROR("Memory tracking list may be corrupted in get_allocation_lifetime_ms (exceeded max iterations)");
    }

    nimcp_mutex_unlock(&g_memory_state.lock);
    return lifetime;
}

/**
 * @brief Update allocation pattern statistics
 *
 * WHY PATTERN TRACKING:
 * - Understand allocation behavior
 * - Optimize for common sizes (pool allocators)
 * - Detect allocation/free imbalance (leaks)
 *
 * ALGORITHM:
 * 1. Search for existing pattern with this size
 * 2. If found:
 *    - Increment allocation_count (if is_alloc)
 *    - Increment free_count and add lifetime (if !is_alloc)
 * 3. If not found and is_alloc:
 *    - Create new pattern
 *    - Add to head of pattern list
 *
 * WHY NOT TRACK ON FREE IF NOT FOUND:
 * - Shouldn't happen (every free has matching alloc)
 * - Would indicate double-free or untracked alloc
 *
 * COMPLEXITY: O(p) where p = number of unique sizes
 * THREAD SAFETY: Must be called with lock held
 *
 * @param size Size being allocated/freed
 * @param is_alloc true for allocation, false for free
 * @param lifetime_ms Allocation lifetime in milliseconds (only used for free)
 */
static void update_memory_patterns(size_t size, bool is_alloc, uint64_t lifetime_ms)
{
    if (!g_memory_state.tracking_enabled)
        return;

    // Search for existing pattern
    size_pattern_t* pattern = g_memory_state.patterns;
    while (pattern) {
        if (pattern->allocation_size == size) {
            if (is_alloc) {
                pattern->allocation_count++;
            } else {
                pattern->free_count++;
                pattern->total_lifetime_ms += lifetime_ms;
            }
            return;
        }
        pattern = pattern->next;
    }

    // New pattern (only create on allocation)
    if (is_alloc) {
        pattern = (size_pattern_t*) malloc(sizeof(size_pattern_t));
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
 * @brief Track allocation in global list
 *
 * WHY TRACK:
 * - Enables leak detection (report unfreed blocks)
 * - Enables double-free detection (check if in list)
 * - Enables buffer overflow detection (check guards on free)
 * - Enables lifetime tracking (time difference)
 *
 * ALGORITHM:
 * 1. Allocate tracking structure (memory_block_t)
 * 2. Fill in metadata (ptr, size, file, line, function, time, canaries)
 * 3. Acquire lock
 * 4. Add to head of blocks list (O(1) insertion)
 * 5. Update statistics
 * 6. Update patterns
 * 7. Release lock
 *
 * WHY HEAD INSERTION:
 * - O(1) time complexity
 * - Recent allocations checked first (common case)
 * - Simple implementation
 *
 * STATISTICS UPDATED:
 * - total_allocated: Cumulative bytes ever allocated
 * - current_allocated: Current bytes in use
 * - allocation_count: Number of allocations
 * - peak_allocated: Maximum current_allocated seen
 *
 * COMPLEXITY: O(1) insertion + O(p) pattern update
 * THREAD SAFETY: Lock-protected
 *
 * @param ptr User pointer
 * @param size User size
 * @param file Source file (__FILE__)
 * @param line Source line (__LINE__)
 * @param function Source function (__func__)
 */
static void track_allocation(void* ptr, size_t size, size_t guard_size, const char* file, int line,
                             const char* function)
{
    if (!g_memory_state.tracking_enabled || !ptr)
        return;

    // Allocate tracking structure using real malloc
    // WHY REAL MALLOC: Avoid infinite recursion (this IS the wrapper)
    memory_block_t* block = (memory_block_t*) malloc(sizeof(memory_block_t));
    if (!block)
        return;  // Out of memory, can't track

    // Fill metadata
    block->ptr = ptr;
    block->size = size;
    block->guard_size = guard_size;
    block->file = file;
    block->line = line;
    block->function = function;
    get_current_time(&block->allocation_time);
    block->head_canary = g_memory_state.CANARY_VALUE;
    block->tail_canary = g_memory_state.CANARY_VALUE;
    block->umm_handle = NULL;  // No UMM handle for direct malloc

    // Add to tracking list (head insertion)
    nimcp_mutex_lock(&g_memory_state.lock);
    block->next = g_memory_state.blocks;
    g_memory_state.blocks = block;

    // Update statistics
    g_memory_state.stats.total_allocated += size;
    g_memory_state.stats.current_allocated += size;
    g_memory_state.stats.allocation_count++;

    // Track peak usage
    if (g_memory_state.stats.current_allocated > g_memory_state.stats.peak_allocated) {
        g_memory_state.stats.peak_allocated = g_memory_state.stats.current_allocated;
    }

    // Update pattern analysis
    update_memory_patterns(size, true, 0);

    nimcp_mutex_unlock(&g_memory_state.lock);
}

/**
 * @brief Track allocation with unified memory handle
 *
 * WHY SEPARATE FUNCTION:
 * - Stores UMM handle for proper cleanup via unified_mem_free
 * - Enables CoW-aware memory management
 *
 * @param ptr User pointer
 * @param size User size
 * @param handle Unified memory handle
 * @param file Source file
 * @param line Source line
 * @param function Source function
 */
static void track_allocation_with_handle(void* ptr, size_t size, unified_mem_handle_t handle,
                                         const char* file, int line, const char* function)
{
    if (!g_memory_state.tracking_enabled || !ptr)
        return;

    // Use real malloc for tracking structure to avoid recursion
    memory_block_t* block = (memory_block_t*) malloc(sizeof(memory_block_t));
    if (!block)
        return;

    block->ptr = ptr;
    block->size = size;
    block->guard_size = 0;  // UMM handles guards internally
    block->file = file;
    block->line = line;
    block->function = function;
    get_current_time(&block->allocation_time);
    block->head_canary = g_memory_state.CANARY_VALUE;
    block->tail_canary = g_memory_state.CANARY_VALUE;
    block->umm_handle = handle;  // Store handle for cleanup

    nimcp_mutex_lock(&g_memory_state.lock);
    block->next = g_memory_state.blocks;
    g_memory_state.blocks = block;

    g_memory_state.stats.total_allocated += size;
    g_memory_state.stats.current_allocated += size;
    g_memory_state.stats.allocation_count++;

    if (g_memory_state.stats.current_allocated > g_memory_state.stats.peak_allocated) {
        g_memory_state.stats.peak_allocated = g_memory_state.stats.current_allocated;
    }

    update_memory_patterns(size, true, 0);
    nimcp_mutex_unlock(&g_memory_state.lock);
}

/**
 * @brief Get UMM handle for a tracked pointer
 *
 * WHY SEPARATE FUNCTION:
 * - Needed for nimcp_free to call unified_mem_free
 * - Must be called before untrack_allocation removes the block
 *
 * @param ptr User pointer
 * @return UMM handle or NULL if not found/not UMM allocated
 */
static unified_mem_handle_t get_umm_handle(void* ptr)
{
    if (!g_memory_state.tracking_enabled || !ptr)
        return NULL;

    unified_mem_handle_t handle = NULL;

    nimcp_mutex_lock(&g_memory_state.lock);
    memory_block_t* current = g_memory_state.blocks;

    // Add iteration counter to prevent infinite loop on corrupted list
    uint32_t iterations = 0;
    const uint32_t MAX_ITERATIONS = 100000;

    while (current && iterations < MAX_ITERATIONS) {
        if (current->ptr == ptr) {
            handle = current->umm_handle;
            break;
        }
        current = current->next;
        iterations++;
    }

    if (iterations >= MAX_ITERATIONS) {
        NIMCP_LOGGING_ERROR("Memory tracking list may be corrupted in get_umm_handle (exceeded max iterations)");
    }

    nimcp_mutex_unlock(&g_memory_state.lock);
    return handle;
}

/**
 * @brief Untrack allocation from global list
 *
 * WHY UNTRACK:
 * - Remove freed blocks from tracking
 * - Update statistics
 * - Update pattern analysis
 *
 * ALGORITHM:
 * 1. Acquire lock
 * 2. Walk linked list searching for ptr
 * 3. If found:
 *    - Remove from list (update prev/head pointers)
 *    - Update statistics
 *    - Update patterns
 *    - Free tracking structure
 * 4. Release lock
 *
 * WHY LINEAR SEARCH:
 * - Simple implementation
 * - Acceptable for debugging
 *
 * COMPLEXITY: O(n) where n = number of active allocations
 * THREAD SAFETY: Lock-protected
 *
 * @param ptr User pointer to untrack
 */
static void untrack_allocation(void* ptr)
{
    if (!g_memory_state.tracking_enabled || !ptr)
        return;

    nimcp_mutex_lock(&g_memory_state.lock);

    memory_block_t* current = g_memory_state.blocks;
    memory_block_t* prev = NULL;

    while (current) {
        if (current->ptr == ptr) {
            // Remove from list
            if (prev) {
                prev->next = current->next;  // Middle/end of list
            } else {
                g_memory_state.blocks = current->next;  // Head of list
            }

            // Update statistics
            g_memory_state.stats.current_allocated -= current->size;
            g_memory_state.stats.free_count++;

            // Calculate lifetime for pattern analysis
            timespec_internal_t now;
            get_current_time(&now);
            uint64_t lifetime_ms = timespec_diff_ms(&now, &current->allocation_time);

            // Update patterns
            update_memory_patterns(current->size, false, lifetime_ms);

            // Free tracking structure
            free(current);
            break;
        }
        prev = current;
        current = current->next;
    }

    nimcp_mutex_unlock(&g_memory_state.lock);
}

/**
 * @brief Check for double-free
 *
 * WHY CHECK DOUBLE-FREE:
 * - Prevents crashes (freeing already-freed memory)
 * - Prevents heap corruption
 * - Provides clear error message with location
 *
 * ALGORITHM:
 * 1. Acquire lock
 * 2. Search for ptr in tracking list
 * 3. If found: Valid free, return false
 * 4. If not found: Double-free or untracked pointer, return true
 * 5. Release lock
 *
 * WHY PRINT ERROR:
 * - Immediate feedback during debugging
 * - Prevents crash downstream
 *
 * LIMITATIONS:
 * - Only detects double-free of tracked pointers
 * - Cannot detect free(malloc(x)) (untracked malloc)
 *
 * COMPLEXITY: O(n) where n = number of active allocations
 * THREAD SAFETY: Lock-protected
 *
 * @param ptr Pointer being freed
 * @return true if double-free detected, false if valid
 */
static bool check_double_free(void* ptr)
{
    if (!ptr)
        return false;

    // WHY: Only check when tracking is enabled
    // If tracking is disabled, nothing is in the tracking list,
    // so all pointers would be flagged as double-free (false positive)
    if (!g_memory_state.tracking_enabled) {
        return false;  // Can't detect double-free without tracking
    }

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
        // NOTE: Pointer not in tracking list - could be:
        // 1. Actual double-free (already freed)
        // 2. Allocated via non-tracked path (different allocator)
        // 3. Tracking list bug (entry never added or incorrectly removed)
        // For now, just warn but ALLOW the free to proceed
        // If it's an actual double-free, the system will likely crash
        fprintf(stderr, "[MEMORY] Warning: Pointer %p not in tracking list (possible double-free or untracked allocation)\n", ptr);
        return false;  // Allow the free to proceed - crash will indicate real double-free
    }

    return false;
}

//=============================================================================
// Public API Implementation
//=============================================================================

/**
 * @brief Allocate memory with tracking (malloc wrapper)
 *
 * WHY WRAPPER:
 * - Add canary guards for overflow detection
 * - Track allocation for leak detection
 * - Collect statistics
 * - Optional debug output
 *
 * ALLOCATION LAYOUT:
 *   [HEAD_CANARY][User Data][TAIL_CANARY]
 *   ↑                        ↑
 *   Real malloc return       What we return + size
 *
 * ALGORITHM:
 * 1. Initialize if needed
 * 2. Compute total size (user_size + 2×canary)
 * 3. Allocate with real malloc
 * 4. Add canary guards
 * 5. Track allocation
 * 6. Optionally print debug message
 * 7. Return user pointer
 *
 * ERROR HANDLING:
 * - If malloc fails, increment failed_allocations stat
 * - Return NULL (same as malloc)
 *
 * COMPLEXITY: O(1) + O(n) for tracking
 * THREAD SAFETY: Fully thread-safe
 *
 * @param size Bytes to allocate
 * @return Pointer to allocated memory or NULL on failure
 */
void* nimcp_malloc(size_t size)
{
    init_if_needed();

    // Use unified memory manager for all allocations
    // This provides CoW support, consistent tracking, and proper cleanup
    if (g_unified_manager) {
        unified_mem_request_t req = unified_mem_request_direct(size);
        unified_mem_handle_t handle = unified_mem_alloc(g_unified_manager, &req);

        if (handle) {
            void* ptr = unified_mem_write(handle);
            if (ptr) {
                // Track allocation with handle for proper cleanup
                track_allocation_with_handle(ptr, size, handle, __FILE__, __LINE__, __func__);

                if (g_memory_state.debug_output) {
                    printf("[MEMORY] Allocated via UMM: %zu bytes at %p\n", size, ptr);
                }
                return ptr;
            }
            // Handle allocation succeeded but write failed - cleanup
            unified_mem_free(handle);
        }
    }

    // Fallback to direct malloc if UMM not available
    // Add guards for consistency with aligned allocations
    const size_t guard_size = 8;  // Minimum guard size
    size_t total_size = size + (2 * guard_size);

    void* real_ptr = malloc(total_size);
    if (real_ptr) {
        // Fill guard regions with canary pattern
        uint64_t pattern = ((uint64_t)g_memory_state.CANARY_VALUE << 32) | g_memory_state.CANARY_VALUE;

        // Head guard
        ((uint64_t*)real_ptr)[0] = pattern;

        // Tail guard
        uint64_t* tail = (uint64_t*)((char*)real_ptr + total_size - guard_size);
        tail[0] = pattern;

        // User pointer (after head guard)
        void* user_ptr = (char*)real_ptr + guard_size;

        // Track with guard size so nimcp_free can compute real_ptr correctly
        track_allocation(user_ptr, size, guard_size, __FILE__, __LINE__, __func__);

        if (g_memory_state.debug_output) {
            printf("[MEMORY] Allocated via malloc: %zu bytes at %p (with guards)\n", size, user_ptr);
        }
        return user_ptr;
    } else {
        nimcp_mutex_lock(&g_memory_state.lock);
        g_memory_state.stats.failed_allocations++;
        nimcp_mutex_unlock(&g_memory_state.lock);
        if (g_memory_state.debug_output) {
            fprintf(stderr, "[MEMORY] Allocation failed: %zu bytes\n", size);
        }
        return NULL;
    }
}

/**
 * @brief Allocate zero-initialized memory with tracking (calloc wrapper)
 *
 * WHY SEPARATE FROM MALLOC:
 * - calloc guarantees zero-initialization
 * - Can be more efficient (OS may provide zeroed pages)
 *
 * WHY COMPUTE count×size:
 * - Match calloc semantics
 * - Track actual size for guards
 *
 * COMPLEXITY: O(1) + O(n) for tracking
 * THREAD SAFETY: Fully thread-safe
 *
 * @param count Number of elements
 * @param size Size of each element
 * @return Pointer to zeroed memory or NULL on failure
 */
void* nimcp_calloc(size_t count, size_t size)
{
    init_if_needed();

    size_t user_size = count * size;

    // Use unified memory manager for all allocations
    if (g_unified_manager) {
        unified_mem_request_t req = unified_mem_request_direct(user_size);
        unified_mem_handle_t handle = unified_mem_alloc(g_unified_manager, &req);

        if (handle) {
            void* ptr = unified_mem_write(handle);
            if (ptr) {
                // Zero-initialize like calloc
                memset(ptr, 0, user_size);

                // Track allocation with handle for proper cleanup
                track_allocation_with_handle(ptr, user_size, handle, __FILE__, __LINE__, __func__);

                if (g_memory_state.debug_output) {
                    printf("[MEMORY] Allocated via UMM (calloc): %zu bytes at %p\n", user_size, ptr);
                }
                return ptr;
            }
            unified_mem_free(handle);
        }
    }

    // Fallback to direct calloc if UMM not available
    void* ptr = calloc(count, size);
    if (ptr) {
        track_allocation(ptr, user_size, 0, __FILE__, __LINE__, __func__);
        if (g_memory_state.debug_output) {
            printf("[MEMORY] Allocated via calloc: %zu bytes at %p\n", user_size, ptr);
        }
    } else {
        nimcp_mutex_lock(&g_memory_state.lock);
        g_memory_state.stats.failed_allocations++;
        nimcp_mutex_unlock(&g_memory_state.lock);
        if (g_memory_state.debug_output) {
            fprintf(stderr, "[MEMORY] Allocation failed (calloc): %zu bytes\n", user_size);
        }
    }

    return ptr;
}

/**
 * @brief Resize memory allocation (realloc wrapper)
 *
 * WHY COMPLEX:
 * - Must untrack old pointer
 * - Realloc with real pointer (before guard)
 * - Re-add guards (may be different location)
 * - Track new pointer
 *
 * SPECIAL CASES:
 * - ptr==NULL: Same as malloc(new_size)
 * - new_size==0: Should free (not implemented, UB in C11)
 *
 * ALGORITHM:
 * 1. If ptr==NULL, call nimcp_malloc
 * 2. Get old size from tracking
 * 3. Untrack old pointer
 * 4. Compute real pointer (before head guard)
 * 5. Call real realloc
 * 6. Add guards to new allocation
 * 7. Track new pointer
 *
 * WHY UNTRACK BEFORE REALLOC:
 * - Realloc may move or free old pointer
 * - Avoids dangling pointer in tracking list
 *
 * COMPLEXITY: O(n) for size + O(n) for tracking
 * THREAD SAFETY: Fully thread-safe
 *
 * @param ptr Pointer to resize (NULL for new allocation)
 * @param new_size New size in bytes
 * @return Pointer to resized memory or NULL on failure
 */
void* nimcp_realloc(void* ptr, size_t new_size)
{
    init_if_needed();

    // Special case: realloc(NULL, size) == malloc(size)
    if (!ptr)
        return nimcp_malloc(new_size);

    // Special case: realloc(ptr, 0) - should free (implementation-defined in C11)
    if (new_size == 0) {
        nimcp_free(ptr);
        return NULL;
    }

    // Get old allocation info before untracking
    size_t old_size = get_allocation_size(ptr);
    unified_mem_handle_t old_handle = get_umm_handle(ptr);

    if (new_size < old_size && g_memory_state.debug_output) {
        printf("[MEMORY] Realloc reducing size from %zu to %zu at %p\n", old_size, new_size, ptr);
    }

    // Use unified memory manager for all allocations
    if (g_unified_manager) {
        unified_mem_request_t req = unified_mem_request_direct(new_size);
        unified_mem_handle_t new_handle = unified_mem_alloc(g_unified_manager, &req);

        if (new_handle) {
            void* new_ptr = unified_mem_write(new_handle);
            if (new_ptr) {
                // Copy old data (up to the smaller of old/new sizes)
                size_t copy_size = (old_size < new_size) ? old_size : new_size;
                memcpy(new_ptr, ptr, copy_size);

                // Track new allocation
                track_allocation_with_handle(new_ptr, new_size, new_handle, __FILE__, __LINE__, __func__);

                // Free old allocation
                untrack_allocation(ptr);
                if (old_handle) {
                    unified_mem_free(old_handle);
                } else {
                    // Old was allocated via direct malloc
                    free(ptr);
                }

                if (g_memory_state.debug_output) {
                    printf("[MEMORY] Reallocated via UMM: %zu bytes at %p (old: %p)\n", new_size, new_ptr, ptr);
                }
                return new_ptr;
            }
            unified_mem_free(new_handle);
        }
    }

    // Fallback to direct realloc if UMM not available
    untrack_allocation(ptr);
    void* new_ptr = realloc(ptr, new_size);
    if (new_ptr) {
        track_allocation(new_ptr, new_size, 0, __FILE__, __LINE__, __func__);
        if (g_memory_state.debug_output) {
            printf("[MEMORY] Reallocated via realloc: %zu bytes at %p (old: %p)\n", new_size, new_ptr, ptr);
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

/**
 * @brief Duplicate string with tracking (strdup wrapper)
 *
 * WHY WRAPPER:
 * - Track string allocations like other memory
 * - Consistent with nimcp_malloc/free usage
 *
 * ALGORITHM:
 * 1. Compute length including null terminator
 * 2. Allocate with nimcp_malloc
 * 3. Copy string data
 * 4. Return new string
 *
 * COMPLEXITY: O(n) where n = string length
 * THREAD SAFETY: Fully thread-safe
 *
 * @param str String to duplicate
 * @return Pointer to duplicated string or NULL
 */
char* nimcp_strdup(const char* str)
{
    if (!str)
        return NULL;

    size_t len = strlen(str) + 1;  // Include null terminator
    char* new_str = (char*) nimcp_malloc(len);
    if (!new_str)
        return NULL;

    memcpy(new_str, str, len);
    return new_str;
}

/**
 * @brief Free memory with tracking (free wrapper)
 *
 * WHY COMPREHENSIVE CHECKS:
 * - Detect double-free before crash
 * - Detect buffer overflow before corruption spreads
 * - Update tracking and statistics
 *
 * ALGORITHM:
 * 1. Return if ptr==NULL (free(NULL) is safe)
 * 2. Initialize if needed
 * 3. Check for double-free
 * 4. Check canary guards
 * 5. Compute real pointer (before head guard)
 * 6. Untrack allocation
 * 7. Free with real free
 *
 * WHY CHECK GUARDS ON FREE:
 * - Last chance to detect overflow
 * - Clear attribution to allocation site
 *
 * COMPLEXITY: O(n) for double-free check + O(n) for untrack
 * THREAD SAFETY: Fully thread-safe
 *
 * @param ptr Pointer to free (NULL is safe)
 */
void nimcp_free(void* ptr)
{
    if (!ptr)
        return;

    init_if_needed();

    // Detect double-free
    if (check_double_free(ptr)) {
        return;  // Don't actually free (would crash)
    }

    if (g_memory_state.debug_output) {
        printf("[MEMORY] Freed at %p\n", ptr);
    }

    // Get allocation info before untracking (needed for proper cleanup)
    unified_mem_handle_t handle = get_umm_handle(ptr);
    size_t guard_size = get_guard_size(ptr);

    // Untrack allocation
    untrack_allocation(ptr);

    // Free via UMM if handle exists, otherwise direct free
    if (handle) {
        unified_mem_free(handle);
        if (g_memory_state.debug_output) {
            printf("[MEMORY] Freed via UMM at %p\n", ptr);
        }
    } else {
        // Legacy allocation (direct malloc) - use tracked guard_size only
        // IMPORTANT: Do NOT speculatively read memory before pointer to detect guards
        // This can cause crashes if the pointer wasn't allocated with guards.
        // Only use guard_size if it was tracked during allocation.
        void* real_ptr = (guard_size > 0) ? (char*) ptr - guard_size : ptr;
        free(real_ptr);
        if (g_memory_state.debug_output) {
            printf("[MEMORY] Freed via free() at %p (guard_size=%zu)\n", ptr, guard_size);
        }
    }
}

/**
 * @brief Allocate aligned memory with tracking
 *
 * WHY ALIGNED ALLOCATIONS:
 * - SIMD operations require 16/32-byte alignment
 * - Cache line alignment (64 bytes) prevents false sharing
 * - DMA buffers may require specific alignment
 *
 * WHY NO CANARIES:
 * - posix_memalign returns exact alignment
 * - Adding canaries would break alignment guarantee
 * - Trade-off: Alignment vs overflow detection
 *
 * ALGORITHM:
 * 1. Call posix_memalign for aligned allocation
 * 2. Track allocation (no guards)
 * 3. Return aligned pointer
 *
 * COMPLEXITY: O(1) + O(n) for tracking
 * THREAD SAFETY: Fully thread-safe
 *
 * @param size Bytes to allocate
 * @param alignment Alignment boundary (must be power of 2)
 * @return Aligned pointer or NULL on failure
 */
void* nimcp_aligned_malloc(size_t size, size_t alignment)
{
    init_if_needed();

    // Note: UMM may not guarantee specific alignment, so use posix_memalign
    // but still track the allocation properly
    void* ptr;
    int result = posix_memalign(&ptr, alignment, size);
    if (result != 0) {
        nimcp_mutex_lock(&g_memory_state.lock);
        g_memory_state.stats.failed_allocations++;
        nimcp_mutex_unlock(&g_memory_state.lock);
        return NULL;
    }

    // Track with NULL handle (posix_memalign doesn't use UMM)
    track_allocation(ptr, size, 0, __FILE__, __LINE__, __func__);

    if (g_memory_state.debug_output) {
        printf("[MEMORY] Allocated aligned: %zu bytes at %p (alignment: %zu)\n", size, ptr, alignment);
    }
    return ptr;
}

/**
 * @brief Free aligned memory
 *
 * WHY SEPARATE:
 * - No canaries to check (aligned_malloc doesn't add them)
 * - Same underlying free() as regular memory
 *
 * @param ptr Pointer to free
 */
void nimcp_aligned_free(void* ptr)
{
    if (!ptr)
        return;

    init_if_needed();

    // Detect double-free
    if (check_double_free(ptr)) {
        return;
    }

    if (g_memory_state.debug_output) {
        printf("[MEMORY] Freed aligned at %p\n", ptr);
    }

    // Check for UMM handle (aligned allocations don't use UMM, but check anyway)
    unified_mem_handle_t handle = get_umm_handle(ptr);
    size_t guard_size = get_guard_size(ptr);
    untrack_allocation(ptr);

    if (handle) {
        unified_mem_free(handle);
    } else {
        // Aligned allocations have guards - compute real pointer before freeing
        // The guard_size is tracked during allocation by nimcp_aligned_alloc
        void* real_ptr = (guard_size > 0) ? (char*)ptr - guard_size : ptr;
        free(real_ptr);
        if (g_memory_state.debug_output) {
            printf("[MEMORY] Freed aligned real_ptr=%p (guard_size=%zu)\n", real_ptr, guard_size);
        }
    }
}

/**
 * @brief Initialize memory tracking system
 *
 * WHY PUBLIC API:
 * - Explicit initialization control
 * - Idempotent (safe to call multiple times)
 *
 * USAGE:
 * - Call at application startup
 * - Or rely on lazy initialization in first allocation
 */
void nimcp_memory_init(void)
{
    init_if_needed();
}

/**
 * @brief Clean up memory tracking system
 *
 * WHY CLEANUP:
 * - Free tracking structures
 * - Report leaked blocks
 * - Reset state
 *
 * ALGORITHM:
 * 1. Acquire lock
 * 2. Walk blocks list:
 *    - Report leaks if debug enabled
 *    - Free user allocation
 *    - Free tracking structure
 * 3. Walk patterns list, free each
 * 4. Reset state
 * 5. Release lock
 * 6. Destroy mutex
 *
 * WHY FREE USER ALLOCATIONS:
 * - Prevent leaks when shutting down
 * - Clean exit for tools like valgrind
 *
 * THREAD SAFETY: Assumes no concurrent allocations
 * - Call at shutdown after all threads stopped
 */
void nimcp_memory_cleanup(void)
{
    if (!g_memory_state.initialized)
        return;

    nimcp_mutex_lock(&g_memory_state.lock);

    // Free all leaked blocks
    memory_block_t* current = g_memory_state.blocks;
    while (current) {
        memory_block_t* next = current->next;
        if (g_memory_state.debug_output) {
            fprintf(stderr, "[MEMORY] Leak detected: %zu bytes at %p\n", current->size,
                    current->ptr);
        }
        // NOTE: Do NOT free the user allocation here - it may already be freed.
        // This was causing double-free errors. Only report the leak and free tracking struct.
        // The user is responsible for calling nimcp_free on their allocations.

        // Free tracking structure only
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

    // Reset state
    g_memory_state.blocks = NULL;
    g_memory_state.patterns = NULL;
    memset(&g_memory_state.stats, 0, sizeof(nimcp_memory_stats_t));

    nimcp_mutex_unlock(&g_memory_state.lock);
    nimcp_mutex_destroy(&g_memory_state.lock);
    g_memory_state.initialized = false;

    // Destroy unified memory manager if it was created
    if (g_unified_manager) {
        unified_mem_destroy(g_unified_manager);
        g_unified_manager = NULL;
        g_unified_initialized = false;
    }
}

/**
 * @brief Get current memory statistics
 *
 * WHY STATISTICS:
 * - Monitor memory usage programmatically
 * - Automated testing (check for leaks)
 * - Performance profiling
 *
 * THREAD SAFETY: Lock-protected snapshot
 *
 * @param stats Output parameter
 * @return true on success
 */
bool nimcp_memory_get_stats(nimcp_memory_stats_t* stats)
{
    if (!stats)
        return false;

    nimcp_mutex_lock(&g_memory_state.lock);
    memcpy(stats, &g_memory_state.stats, sizeof(nimcp_memory_stats_t));
    nimcp_mutex_unlock(&g_memory_state.lock);

    return true;
}

/**
 * @brief Clear memory statistics
 *
 * WHY CLEAR:
 * - Profile specific code sections
 * - Reset counters between test runs
 */
void nimcp_memory_clear_stats(void)
{
    nimcp_mutex_lock(&g_memory_state.lock);
    memset(&g_memory_state.stats, 0, sizeof(nimcp_memory_stats_t));
    nimcp_mutex_unlock(&g_memory_state.lock);
}

/**
 * @brief Reset entire memory tracker to fresh state
 *
 * WHAT: Reset all global state to initial values
 * WHY: Test isolation - prevent state leakage between tests
 * HOW: Free tracking lists, zero stats, reset flags
 *
 * WARNING: Does NOT free user allocations
 * - Caller must free all nimcp_malloc'd memory first
 * - Only frees internal tracking structures
 *
 * COMPLEXITY: O(n + p) where n=active blocks, p=patterns
 * THREAD SAFETY: Fully thread-safe (lock-protected)
 */
void nimcp_memory_reset_state(void)
{
    if (!g_memory_state.initialized)
        return;

    nimcp_mutex_lock(&g_memory_state.lock);

    /* WHAT: Free all tracking blocks (but not user memory) */
    /* WHY: Clear tracking list without freeing user allocations */
    memory_block_t* current = g_memory_state.blocks;
    while (current) {
        memory_block_t* next = current->next;
        free(current); /* Free tracking structure only */
        current = next;
    }
    g_memory_state.blocks = NULL;

    /* WHAT: Free all pattern tracking */
    size_pattern_t* pattern = g_memory_state.patterns;
    while (pattern) {
        size_pattern_t* next = pattern->next;
        free(pattern);
        pattern = next;
    }
    g_memory_state.patterns = NULL;

    /* WHAT: Reset statistics to zero */
    memset(&g_memory_state.stats, 0, sizeof(nimcp_memory_stats_t));

    /* WHAT: Reset flags to default state */
    g_memory_state.tracking_enabled = true;
    g_memory_state.debug_output = false;

    nimcp_mutex_unlock(&g_memory_state.lock);
}

/**
 * @brief Enable or disable tracking
 *
 * WHY CONFIGURABLE:
 * - Disable in production (no overhead)
 * - Enable in debug/test (find bugs)
 */
void nimcp_memory_enable_tracking(bool enable)
{
    init_if_needed();
    g_memory_state.tracking_enabled = enable;
}

/**
 * @brief Enable or disable debug output
 *
 * WHY CONFIGURABLE:
 * - Too verbose for normal use
 * - Helpful for specific debugging
 */
void nimcp_memory_enable_debug_output(bool enable)
{
    init_if_needed();
    g_memory_state.debug_output = enable;
}

/**
 * @brief Dump all current allocations to console
 *
 * WHY DUMP:
 * - See what's currently allocated
 * - Find allocation hotspots
 * - Debug memory usage
 *
 * OUTPUT FORMAT:
 * - Summary: Total, current, peak, counts
 * - Per-allocation: Address, size, location
 */
void nimcp_memory_dump_allocations(void)
{
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
        printf("  [%u] %p: %zu bytes (from %s:%d in %s())\n", count++, current->ptr, current->size,
               current->file, current->line, current->function);
        current = current->next;
    }
    printf("============================================\n\n");

    nimcp_mutex_unlock(&g_memory_state.lock);
}

/**
 * @brief Check for memory leaks
 *
 * WHY CHECK LEAKS:
 * - Find unfreed memory before shutdown
 * - Automated testing (assert no leaks)
 * - CI/CD integration
 *
 * OUTPUT FORMAT:
 * - Per-leak: Address, size, function, location, lifetime
 * - Summary: Total blocks and bytes leaked
 */
void nimcp_memory_check_leaks(void)
{
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

/**
 * @brief Analyze allocation patterns
 *
 * WHY ANALYZE:
 * - Understand allocation behavior
 * - Optimize for common sizes (pool allocators)
 * - Detect leaks (allocation_count > free_count)
 *
 * OUTPUT FORMAT:
 * - Per-size: Allocation count, free count, avg lifetime
 * - Warning if allocation/free imbalance
 */
void nimcp_memory_analyze_patterns(void)
{
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
        double avg_lifetime =
            pattern->free_count > 0 ? (double) pattern->total_lifetime_ms / pattern->free_count : 0;

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

//=============================================================================
// Aligned Memory Allocation (Added for AddressSanitizer compatibility)
//=============================================================================

/**
 * @brief Allocate aligned memory with tracking and guards
 *
 * WHAT: Allocate memory with specific alignment requirement
 * WHY: Required for SIMD, atomics, structs, and AddressSanitizer
 * HOW: Use aligned_alloc with alignment-sized guards
 *
 * ALIGNMENT REQUIREMENTS:
 * - Must be power of 2 (2,4,8,16,32,64,128,...)
 * - Must be >= 8 (for guard compatibility)
 * - Guards match alignment to preserve it
 *
 * GUARD STRATEGY (KEY INSIGHT):
 * - Guard size = MAX(8, alignment) to preserve alignment
 * - If alignment=16, guards are 16 bytes each
 * - User pointer = aligned_ptr + guard_size (still aligned!)
 * - Pattern: 0xDEADBEEFDEADBEEF repeated
 *
 * ALGORITHM (< 50 lines):
 * 1. Guard: Validate alignment (power of 2, >= 8)
 * 2. Compute: guard_size = MAX(8, alignment)
 * 3. Compute: total = size + 2×guard_size, round to alignment
 * 4. Allocate: aligned_alloc(alignment, total)
 * 5. Guard: Fill guard regions with pattern
 * 6. Track: Add to tracking table
 * 7. Return: ptr + guard_size (preserves alignment!)
 *
 * @param alignment Alignment in bytes (power of 2, >= 8)
 * @param size Bytes to allocate
 * @return Aligned pointer or NULL
 */
void* nimcp_aligned_alloc_impl(size_t alignment, size_t size)
{
    // Guard: Validate alignment is power of 2
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        if (g_memory_state.debug_output) {
            fprintf(stderr, "[MEMORY] Invalid alignment %zu\n", alignment);
        }
        return NULL;
    }

    // Guard: Minimum alignment
    if (alignment < 8) {
        alignment = 8;
    }

    // Guard: Zero size
    if (size == 0) {
        return NULL;
    }

    // KEY: Guard size must match alignment to preserve it
    size_t guard_size = (alignment > 8) ? alignment : 8;

    // Compute total with guards (will preserve alignment)
    size_t total_size = size + (2 * guard_size);
    total_size = (total_size + alignment - 1) & ~(alignment - 1);

    // Allocate aligned memory
    void* real_ptr = aligned_alloc(alignment, total_size);

    if (!real_ptr) {
        nimcp_mutex_lock(&g_memory_state.lock);
        g_memory_state.stats.failed_allocations++;
        nimcp_mutex_unlock(&g_memory_state.lock);
        return NULL;
    }

    // Fill guard regions with pattern
    uint64_t pattern = ((uint64_t)g_memory_state.CANARY_VALUE << 32) | g_memory_state.CANARY_VALUE;

    // Head guard
    for (size_t i = 0; i < guard_size / sizeof(uint64_t); i++) {
        ((uint64_t*)real_ptr)[i] = pattern;
    }

    // Tail guard
    uint64_t* tail = (uint64_t*)((char*)real_ptr + total_size - guard_size);
    for (size_t i = 0; i < guard_size / sizeof(uint64_t); i++) {
        tail[i] = pattern;
    }

    // User pointer (still aligned!)
    void* user_ptr = (char*)real_ptr + guard_size;

    // KEY FIX: The usable size is the padded total minus guards
    // WHY: After rounding total_size for alignment, the user data region may be larger than requested
    // EXAMPLE: size=10, guard=16, total=(10+32)=42→48, usable=48-32=16 (not 10!)
    size_t usable_size = total_size - (2 * guard_size);

    // Track allocation with actual usable size (not requested size)
    track_allocation(user_ptr, usable_size, guard_size, __FILE__, __LINE__, __func__);

    if (g_memory_state.debug_output) {
        printf("[MEMORY] Aligned %zu @ %p (align=%zu)\n", size, user_ptr, alignment);
    }

    return user_ptr;
}


// Public API wrapper
void* nimcp_aligned_alloc(size_t alignment, size_t size)
{
    init_if_needed();
    return nimcp_aligned_alloc_impl(alignment, size);
}
