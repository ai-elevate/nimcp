//=============================================================================
// nimcp_thread.c - Thread Abstraction Layer for NIMCP
//=============================================================================
// ARCHITECTURAL OVERVIEW:
// This module implements a comprehensive POSIX thread abstraction layer that
// provides platform-independent threading primitives with robust error handling,
// resource tracking, and named lock management. It serves as the foundation
// for all concurrent operations in NIMCP.
//
// KEY DESIGN: ADAPTER + FACADE + REGISTRY PATTERN
// ================================================
// WHY ADAPTER:
// - Wraps POSIX pthread API with NIMCP-specific error handling
// - Translates pthread error codes to nimcp_result_t status codes
// - Provides consistent interface across platforms (future Windows support)
// - Decouples NIMCP code from platform-specific threading details
//
// WHY FACADE:
// - Simplifies complex pthread API into focused operations
// - Hides pthread attribute management complexity
// - Provides sensible defaults (stack size, priority, etc.)
// - Single point of control for threading policy
//
// WHY REGISTRY:
// - Named resource locks enable string-based synchronization
// - Reference counting ensures lock lifecycle management
// - Hash table (bucketed) for O(1) average lock lookup
// - Automatic cleanup when last reference released
//
// THREADING MODEL ARCHITECTURE:
//
//   ┌────────────────────────────────────────────────────┐
//   │         Application Code (NIMCP Modules)           │
//   └────────────────┬───────────────────────────────────┘
//                    │
//        ┌───────────┴──────────┐
//        │                      │
//   ┌────▼─────────────┐  ┌────▼──────────────────┐
//   │ Thread API       │  │ Named Resource Locks  │
//   │ - create         │  │ - get_resource_lock   │
//   │ - join           │  │ - release_lock        │
//   │ - detach         │  │ (String → Mutex)      │
//   └──────┬───────────┘  └──────┬────────────────┘
//          │                     │
//   ┌──────▼─────────────────────▼────────┐
//   │      POSIX pthread Library           │
//   │  - pthread_create, pthread_mutex_*   │
//   │  - pthread_cond_*, pthread_once      │
//   └──────────────────────────────────────┘
//
// WHY THIS LAYERED DESIGN:
// - SEPARATION OF CONCERNS: Thread management vs synchronization vs naming
// - TESTABILITY: Can mock pthread layer for unit testing
// - PORTABILITY: Single point to add Windows threading support
// - MAINTAINABILITY: Changes to error handling don't affect caller
//
// NAMED RESOURCE LOCK SYSTEM:
// ============================
// WHAT: Map string identifiers to mutexes with automatic lifecycle
// WHY: Enable intuitive synchronization without explicit mutex management
//
// EXAMPLE USE CASE:
//   // Thread A needs exclusive access to "brain_network"
//   nimcp_mutex_t* lock;
//   nimcp_get_resource_lock("brain_network", &lock);
//   nimcp_mutex_lock(lock);
//   // ... critical section ...
//   nimcp_mutex_unlock(lock);
//   nimcp_release_resource_lock("brain_network");
//
// BENEFITS:
// - No global mutex variables needed
// - Automatically shared across modules (same name = same lock)
// - Reference counting prevents premature destruction
// - Hash table for fast lookup (O(1) average case)
//
// HASH TABLE STRUCTURE:
//
//   resource_lock_table_t
//   ┌──────────────────────────────────┐
//   │ global_mutex (protects table)    │
//   │ initialized flag                 │
//   │ buckets[256]                     │
//   └───┬──────────────────────────────┘
//       │
//   ┌───▼───────────┬─────────┬─────────┬─────────┐
//   │ Bucket 0      │ Bucket 1│ Bucket 2│  ...    │
//   │ bucket_mutex  │         │         │         │
//   │ entries →     │         │         │         │
//   └───┬───────────┴─────────┴─────────┴─────────┘
//       │
//   ┌───▼────────────────┐     ┌─────────────────┐
//   │ resource_entry_t   │────→│ resource_entry  │
//   │ - resource_id      │     │ - resource_id   │
//   │ - lock (mutex*)    │     │ - lock          │
//   │ - ref_count        │     │ - ref_count     │
//   │ - next             │     │ - next → NULL   │
//   └────────────────────┘     └─────────────────┘
//
// WHY BUCKETED HASH TABLE:
// - Reduces lock contention (256 bucket mutexes, not 1 global)
// - O(1) average lookup (hash function distributes keys)
// - Simple collision handling (linked list per bucket)
// - Fixed memory overhead (256 buckets × small bucket structure)
//
// WHY 256 BUCKETS:
// - Power of 2 (fast modulo: hash % 256 = hash & 0xFF)
// - Balance: More buckets = less contention but more memory
// - Expected use: <100 named locks, so short chains (<1 entry/bucket)
//
// ALTERNATIVE APPROACHES REJECTED:
// - Single global hash table: Lock contention bottleneck
// - Binary search tree: O(log n) lookup, more complex
// - Fixed array: Can't handle dynamic names, wastes memory
//
// REFERENCE COUNTING LIFECYCLE:
//
//   get_resource_lock("X")              get_resource_lock("X")
//         ↓                                     ↓
//   ┌─────────────┐                       ┌─────────────┐
//   │ Create lock │                       │ Find lock   │
//   │ ref_count=1 │                       │ ref_count++ │
//   └─────────────┘                       └─────────────┘
//         ↓                                     ↓
//   Use lock (ref=1)                      Use lock (ref=2)
//         ↓                                     ↓
//   release_resource_lock("X")            release_resource_lock("X")
//         ↓                                     ↓
//   ┌─────────────┐                       ┌─────────────┐
//   │ ref_count-- │                       │ ref_count-- │
//   │ (now = 0)   │                       │ (now = 1)   │
//   │ DESTROY     │                       │ Keep alive  │
//   └─────────────┘                       └─────────────┘
//
// WHY REFERENCE COUNTING:
// - Automatic cleanup: No manual destroy calls needed
// - Safe sharing: Lock lives while any holder exists
// - Memory efficient: Lock freed when not needed
//
// THREAD SAFETY GUARANTEES:
// =========================
// 1. THREAD-LOCAL ERROR STORAGE
//    - Each thread has its own error buffer
//    - No cross-contamination of error messages
//    - __thread storage class (TLS)
//
// 2. GLOBAL MUTEX PROTECTION
//    - resource_table.global_mutex protects table initialization
//    - Each bucket has its own mutex (fine-grained locking)
//
// 3. PTHREAD_ONCE INITIALIZATION
//    - Ensures thread_init_routine runs exactly once
//    - Safe across all threads
//    - No race conditions in initialization
//
// 4. LOCK ORDERING
//    - Global mutex → Bucket mutex → Resource lock
//    - Prevents deadlock by consistent ordering
//
// MUTEX vs SPINLOCK TRADE-OFFS:
// ==============================
// WHY MUTEXES (not spinlocks):
// - BLOCKING: Thread yields CPU while waiting (better for long waits)
// - FAIRNESS: POSIX mutexes typically provide fairness guarantees
// - NO BUSY-WAIT: Doesn't waste CPU cycles spinning
// - KERNEL SUPPORT: OS-level scheduling, priority inheritance
//
// WHEN SPINLOCKS ARE BETTER:
// - Very short critical sections (<100 cycles)
// - High lock acquisition rate
// - Real-time requirements
// - Non-preemptive environment
//
// WHY NIMCP USES MUTEXES:
// - Critical sections are non-trivial (hash lookup, list traversal)
// - I/O operations possible (logging, file access)
// - Multi-core systems (yield is better than spin)
// - Simpler programming model (no need to disable preemption)
//
// MUTEX TYPES SUPPORTED:
// ======================
// 1. NORMAL: Default, undefined behavior on recursive lock
// 2. RECURSIVE: Can be locked multiple times by same thread
// 3. ERRORCHECK: Returns error on recursive lock (debugging)
//
// WHY SUPPORT MULTIPLE TYPES:
// - NORMAL: Fastest, no tracking overhead
// - RECURSIVE: Simplifies complex call chains (lock held across layers)
// - ERRORCHECK: Catches programming errors during development
//
// CONDITION VARIABLE PATTERNS:
// ============================
// CLASSIC WAIT PATTERN:
//   nimcp_mutex_lock(&mutex);
//   while (!condition) {
//       nimcp_cond_wait(&cond, &mutex);  // Atomically unlocks and waits
//   }
//   // condition is true, mutex is locked
//   nimcp_mutex_unlock(&mutex);
//
// WHY WHILE LOOP (not if):
// - SPURIOUS WAKEUPS: pthread_cond_wait can wake without signal
// - MULTIPLE WAITERS: Another thread may consume condition
// - CORRECTNESS: Always re-check condition after wake
//
// SIGNAL vs BROADCAST:
// - nimcp_cond_signal: Wakes ONE waiter (efficient for single consumer)
// - nimcp_cond_broadcast: Wakes ALL waiters (needed for multiple consumers)
//
// TIMED WAIT IMPLEMENTATION:
// - Uses CLOCK_REALTIME for absolute timeout
// - Handles nanosecond overflow (>1 second)
// - Returns NIMCP_BUSY on timeout (consistent with trylock)
//
// ERROR HANDLING STRATEGY:
// ========================
// THREE-TIER ERROR HANDLING:
//
// 1. RETURN CODE (nimcp_result_t)
//    - Immediate error status for caller
//    - Can be checked without additional calls
//    - Consistent across all functions
//
// 2. THREAD-LOCAL ERROR MESSAGE
//    - Detailed error explanation (strerror, context)
//    - Retrieved via nimcp_thread_get_error()
//    - Preserved until next error or clear
//
// 3. STDERR LOGGING (optional)
//    - Critical errors printed immediately
//    - Helps debugging during development
//
// EXAMPLE:
//   nimcp_result_t result = nimcp_mutex_lock(&mutex);
//   if (result != NIMCP_SUCCESS) {
//       fprintf(stderr, "Lock failed: %s\n", nimcp_thread_get_error());
//       return result;
//   }
//
// WHY THREAD-LOCAL STORAGE:
// - Safe in multi-threaded environment
// - No global error variable contention
// - Each thread sees only its errors
//
// PLATFORM ABSTRACTION LAYER:
// ===========================
// CURRENT: Linux/POSIX (pthread)
// FUTURE: Windows (CreateThread, CRITICAL_SECTION, etc.)
//
// ABSTRACTION STRATEGY:
// 1. Define nimcp_thread_t, nimcp_mutex_t as typedefs
// 2. Implement functions using platform-specific APIs
// 3. Conditional compilation (#ifdef _WIN32)
// 4. Keep interface identical across platforms
//
// EXAMPLE (future Windows support):
//   #ifdef _WIN32
//       typedef HANDLE nimcp_thread_t;
//       typedef CRITICAL_SECTION nimcp_mutex_t;
//   #else
//       typedef pthread_t nimcp_thread_t;
//       typedef pthread_mutex_t nimcp_mutex_t;
//   #endif
//
// THREAD CREATION ATTRIBUTES:
// ===========================
// SUPPORTED ATTRIBUTES:
// - stack_size: Thread stack size in bytes
// - priority: Scheduling priority (POSIX scheduling)
// - detached: Automatically cleanup on exit
//
// WHY CONFIGURABLE STACK SIZE:
// - Default (2MB on Linux) may be too large for many threads
// - Deep recursion needs larger stack
// - Embedded systems have limited memory
//
// WHY PRIORITY:
// - Real-time response for critical threads
// - Background tasks (lower priority)
// - Note: Requires POSIX priority scheduling support
//
// WHY DETACHED:
// - No need to join (fire-and-forget)
// - Automatic resource cleanup
// - Simpler lifecycle management
//
// PERFORMANCE CHARACTERISTICS:
// ============================
// MUTEX OPERATIONS:
// - Lock/unlock: ~50-100ns (uncontended), ~1-10µs (contended)
// - Trylock: ~20-50ns (always non-blocking)
// - Init/destroy: ~500ns-1µs (rare operations)
//
// CONDITION VARIABLES:
// - Wait: Blocks until signal (no CPU cost)
// - Signal: ~500ns (wake one thread)
// - Broadcast: ~500ns + N×context_switch (wake all threads)
//
// NAMED RESOURCE LOCKS:
// - get_resource_lock: O(1) average, ~2-5µs
//   * Hash computation: ~100ns
//   * Bucket lock: ~50-100ns
//   * List search: O(n) per bucket, usually n=0-2
//   * Malloc/init (first time): ~5-10µs
// - release_resource_lock: O(1) average, ~1-2µs
//
// MEMORY OVERHEAD:
// - Per thread: ~2MB default stack + pthread control structures
// - Per mutex: ~40 bytes (pthread_mutex_t)
// - Per condition variable: ~48 bytes (pthread_cond_t)
// - Per named lock: ~80 bytes (entry + mutex + string)
// - Resource table: 256 buckets × ~48 bytes = ~12KB
//
// SCALABILITY:
// - Thread creation: Linear with thread count (context switch overhead)
// - Named locks: Sub-linear with bucketing (256-way parallelism)
// - Condition variables: Linear with waiter count
//
// DESIGN PATTERNS:
// ================
// 1. ADAPTER: Wraps pthread API with NIMCP interface
// 2. FACADE: Simplifies complex pthread attribute management
// 3. REGISTRY: Named resource lock management
// 4. SINGLETON: Global resource_lock_table
// 5. FACTORY: nimcp_mutex_init with attribute-based creation
// 6. RAII: Reference counting for automatic cleanup
// 7. TEMPLATE METHOD: Error handling pattern across all functions
//
// SOLID PRINCIPLES:
// =================
// - SINGLE RESPONSIBILITY: Each function has one clear purpose
//   * nimcp_mutex_lock: Only locks mutex
//   * set_thread_error: Only stores error message
//   * hash_string: Only computes hash
//
// - OPEN/CLOSED: Can extend with new mutex types without modifying core
//   * Add new mutex_type_t enum value
//   * Add case in nimcp_mutex_init
//   * No changes to lock/unlock/destroy
//
// - LISKOV SUBSTITUTION: Drop-in replacement for pthread
//   * Same semantics (lock/unlock/wait)
//   * Same thread safety guarantees
//   * Same error conditions
//
// - INTERFACE SEGREGATION: Clean, focused APIs
//   * Thread API: create/join/detach/exit
//   * Mutex API: init/destroy/lock/unlock/trylock
//   * Condition API: init/destroy/wait/signal/broadcast
//   * Named lock API: get/release
//
// - DEPENDENCY INVERSION: Depends on pthread abstraction, not concrete types
//   * nimcp_thread_t could be pthread_t or HANDLE
//   * Implementation detail hidden behind typedef
//
// USE CASES IN NIMCP:
// ===================
// 1. PARALLEL MESSAGE PROCESSING
//    - Worker threads process queue items concurrently
//    - Mutex protects queue access
//    - Condition variable signals new work
//
// 2. BRAIN NETWORK SYNCHRONIZATION
//    - Named lock: "brain_network" ensures exclusive access
//    - Multiple modules need same lock without sharing pointer
//
// 3. MEMORY ALLOCATOR THREAD SAFETY
//    - Global mutex protects allocation tracking structures
//    - Condition variable for memory pressure notification
//
// 4. ASYNC I/O OPERATIONS
//    - Detached threads for non-blocking file operations
//    - Condition variable for I/O completion
//
// 5. INITIALIZATION GUARANTEES
//    - pthread_once ensures subsystems initialize exactly once
//    - Thread-safe even during concurrent module initialization
//
// LIMITATIONS AND TRADE-OFFS:
// ===========================
// 1. POSIX-ONLY (currently)
//    TRADE-OFF: Simplicity vs portability
//    MITIGATION: Clean abstraction for future Windows support
//
// 2. GLOBAL RESOURCE TABLE
//    TRADE-OFF: Easy access vs potential contention
//    MITIGATION: Bucketed locks (256-way parallelism)
//
// 3. NO PRIORITY INHERITANCE (depends on pthread implementation)
//    TRADE-OFF: Simplicity vs priority inversion protection
//    MITIGATION: Use PTHREAD_PRIO_INHERIT where available
//
// 4. FIXED 256 BUCKETS
//    TRADE-OFF: Fixed memory vs dynamic scaling
//    MITIGATION: 256 is sufficient for expected workloads (<100 named locks)
//
// 5. STRING-BASED NAMING (hash overhead)
//    TRADE-OFF: Convenience vs performance
//    MITIGATION: Hash computed once, cached in entry
//
// WHY THESE TRADE-OFFS:
// - POSIX-only: 90% of deployment targets are Linux
// - Global table: Simpler than per-module registration
// - Fixed buckets: Predictable memory, sufficient for use case
// - String naming: Dramatically improves code readability
//
// DEBUGGING AND DIAGNOSTICS:
// ==========================
// THREAD-LOCAL ERROR MESSAGES:
//   - Format includes operation context and errno translation
//   - Example: "Mutex lock failed: Resource deadlock avoided"
//
// ERROR CODES:
//   - NIMCP_SUCCESS: Operation succeeded
//   - NIMCP_ERROR_INVALID_PARAM: NULL pointer or invalid argument
//   - NIMCP_ERROR_SYSTEM: Underlying pthread call failed
//   - NIMCP_BUSY: Trylock failed (locked) or timeout occurred
//   - NIMCP_ERROR_MEMORY: Allocation failed
//   - NIMCP_ERROR_NOT_FOUND: Named resource doesn't exist
//
// COMMON ERRORS AND CAUSES:
//   - "Mutex lock failed: Invalid argument" → Uninitialized mutex
//   - "Mutex lock failed: Resource deadlock avoided" → Recursive lock attempt
//   - "Thread creation failed: Resource temporarily unavailable" → Too many threads
//   - "Cond timedwait failed: Invalid argument" → Negative timeout
//
// HASH FUNCTION ANALYSIS:
// =======================
// DJBX33A (DJB2) HASH:
//   hash = 5381
//   for each char c:
//       hash = hash * 33 + c
//   return hash % 256
//
// WHY DJBX33A:
// - Fast: One multiply, one add per character
// - Good distribution: Prime multiplier (33) spreads bits
// - Low collision rate: ~1-2% for typical string sets
// - Simple: No complex bit operations
//
// ALTERNATIVES CONSIDERED:
// - FNV-1a: Similar performance, slightly more complex
// - CRC32: Hardware support, overkill for our use case
// - Simple sum: Terrible distribution (all anagrams collide)
//
// COLLISION HANDLING:
// - Chaining: Linked list per bucket
// - Average chain length: N / 256 (where N = number of locks)
// - Expected: <1 entry per bucket (<100 locks total)
//
// INITIALIZATION PATTERN:
// =======================
// PTHREAD_ONCE USAGE:
//   static nimcp_once_t init_once = NIMCP_ONCE_INIT;
//   nimcp_once(&init_once, thread_init_routine);
//
// WHY PTHREAD_ONCE (not manual flag):
// - THREAD-SAFE: Built-in synchronization
// - RACE-FREE: Guarantees single execution
// - SIMPLER: No manual mutex needed for init flag
// - PORTABLE: Standard POSIX mechanism
//
// INIT ROUTINE:
// - Initialize global_mutex (protects table)
// - Initialize 256 bucket mutexes
// - NULL-initialize entry lists
// - Set initialized flag
//
// WHY THIS ORDER:
// - Mutexes first (needed for thread safety)
// - Pointers second (ready for use)
// - Flag last (signals completion)
//
//=============================================================================

/**
 * @file nimcp_thread.c
 * @brief Thread abstraction layer implementation
 *
 * WHAT: POSIX thread wrapper with error handling and resource management
 * WHY: Simplify threading, enable portability, provide named locks
 * HOW: Adapter pattern + bucketed hash table for named resources
 */

#include "utils/nimcp_thread.h"
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "utils/nimcp_memory.h"  // CRITICAL: Declares nimcp_calloc/nimcp_free return types

//=============================================================================
// Thread-Local Error Storage
//=============================================================================

/**
 * WHAT: Thread-local error state
 * WHY: Each thread needs its own error buffer (no cross-contamination)
 * HOW: __thread storage class (TLS)
 *
 * WHY THREAD-LOCAL:
 * - Multi-threaded safety: Thread A's error doesn't overwrite Thread B's
 * - No mutex needed: Each thread has private copy
 * - Zero-init: New threads start with empty error
 *
 * STRUCTURE:
 * - error_code: nimcp_result_t status code
 * - error_message: Human-readable description (256 bytes)
 *
 * TRADE-OFF:
 * - Memory: 264 bytes per thread (acceptable)
 * - Speed: Fast access (no lock, no syscall)
 */
static __thread nimcp_thread_error_t thread_error = {0};

//=============================================================================
// Resource Lock Table
//=============================================================================

/**
 * WHAT: Global hash table mapping string names to mutexes
 * WHY: Enable intuitive named synchronization without manual mutex management
 * HOW: 256 buckets, each with linked list of entries
 *
 * STRUCTURE:
 * - resource_table: Global singleton instance
 * - init_once: pthread_once control for initialization
 *
 * WHY STATIC:
 * - Global visibility (accessible from all functions)
 * - Single instance (singleton pattern)
 * - Zero-initialized (safe initial state)
 */
static resource_lock_table_t resource_table = {0};
static nimcp_once_t init_once = NIMCP_ONCE_INIT;

//=============================================================================
// Error Handling
//=============================================================================

/**
 * @brief Set thread-local error message
 *
 * WHY PRINTF-STYLE:
 * - Flexible: Can include errno, pointers, context
 * - Readable: "Mutex lock failed: %s" is clear
 *
 * ALGORITHM:
 * 1. Initialize va_list for variable arguments
 * 2. Store error_code in thread-local storage
 * 3. Format message using vsnprintf (safe, bounded)
 * 4. Clean up va_list
 *
 * WHY VSNPRINTF (not sprintf):
 * - Buffer overflow protection (truncates at 256 bytes)
 * - Null-termination guaranteed
 * - Standard C function
 *
 * COMPLEXITY: O(n) where n = formatted string length
 * THREAD SAFETY: Fully safe (thread-local storage, no shared state)
 *
 * @param error_code Error status code
 * @param format Printf-style format string
 * @param ... Variable arguments for format
 */
static void set_thread_error(int error_code, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    thread_error.error_code = error_code;
    vsnprintf(thread_error.error_message, sizeof(thread_error.error_message), format, args);
    va_end(args);
}

/**
 * @brief Get last error message for current thread
 *
 * WHY NEEDED:
 * - Return codes tell WHAT failed, this tells WHY
 * - Essential for debugging: "Mutex lock failed: Invalid argument"
 * - Preserved until next error or clear
 *
 * USAGE:
 *   if (nimcp_mutex_lock(&m) != NIMCP_SUCCESS) {
 *       fprintf(stderr, "Error: %s\n", nimcp_thread_get_error());
 *   }
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (thread-local storage)
 *
 * @return Pointer to error message string (thread-local, valid until next error)
 */
const char* nimcp_thread_get_error(void)
{
    return thread_error.error_message;
}

/**
 * @brief Clear error state for current thread
 *
 * WHY NEEDED:
 * - Reset error state before operation sequence
 * - Check if new error occurred (was cleared, now set)
 * - Clean slate for testing
 *
 * ALGORITHM:
 * 1. Zero error_code
 * 2. Zero first byte of message (empty string)
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (thread-local storage)
 */
void nimcp_thread_clear_error(void)
{
    thread_error.error_code = 0;
    thread_error.error_message[0] = '\0';
}

//=============================================================================
// Initialization
//=============================================================================

/**
 * @brief Initialize threading subsystem (pthread_once callback)
 *
 * WHY NEEDED:
 * - Set up resource_lock_table before first use
 * - Initialize all bucket mutexes
 * - Thread-safe initialization (called via pthread_once)
 *
 * ALGORITHM:
 * 1. Initialize global_mutex (protects table-level operations)
 * 2. For each of 256 buckets:
 *    a. Initialize bucket_mutex (protects bucket's entry list)
 *    b. NULL-initialize entries pointer (empty list)
 * 3. Set initialized flag (ready for use)
 *
 * WHY INITIALIZE BUCKET MUTEXES:
 * - Fine-grained locking: 256-way parallelism
 * - Reduces contention vs single global lock
 * - Each bucket independently accessible
 *
 * COMPLEXITY: O(256) = O(1) constant time
 * THREAD SAFETY: Called via pthread_once (guaranteed single execution)
 * CALLED BY: pthread_once in nimcp_thread_init
 */
static void thread_init_routine(void)
{
    // Global mutex protects table-level operations (future use)
    pthread_mutex_init(&resource_table.global_mutex, NULL);

    // Initialize all buckets
    // WHY LOOP: Each bucket needs its own mutex and empty entry list
    for (int i = 0; i < RESOURCE_LOCK_BUCKETS; i++) {
        pthread_mutex_init(&resource_table.buckets[i].bucket_mutex, NULL);
        resource_table.buckets[i].entries = NULL;  // Empty list
    }

    // Signal initialization complete
    // WHY LAST: Ensures all structures ready before marking initialized
    resource_table.initialized = true;
}

/**
 * @brief Initialize threading subsystem
 *
 * WHY PUBLIC API:
 * - Explicit control over initialization timing
 * - Can be called early (before first thread/lock)
 * - Idempotent: safe to call multiple times
 *
 * ALGORITHM:
 * 1. Call pthread_once with init_once control variable
 * 2. pthread_once ensures thread_init_routine runs exactly once
 * 3. All subsequent calls to pthread_once are no-ops
 *
 * WHY PTHREAD_ONCE:
 * - Thread-safe: Built-in synchronization
 * - Race-free: Only one thread executes init routine
 * - Standard: POSIX portable mechanism
 * - Simple: No manual flag + mutex management
 *
 * ALTERNATIVE APPROACHES:
 * - Manual flag + mutex: More complex, error-prone
 * - Constructor attribute: No control over timing
 * - Lazy init on first use: Works but less explicit
 *
 * COMPLEXITY: O(1) after first call (fast path in pthread_once)
 * THREAD SAFETY: Fully safe (pthread_once guarantees)
 *
 * @return NIMCP_SUCCESS on success, NIMCP_ERROR_SYSTEM on failure
 */
nimcp_result_t nimcp_thread_init(void)
{
    // pthread_once returns 0 on success
    // WHY CHECK: pthread_once can fail (rare: invalid once_control)
    return pthread_once(&init_once, thread_init_routine) == 0 ? NIMCP_SUCCESS : NIMCP_ERROR_SYSTEM;
}

//=============================================================================
// Thread Management
//=============================================================================

/**
 * @brief Create a new thread (Adapter for pthread_create)
 *
 * WHY WRAPPER:
 * - Translate pthread_create's complex interface to simple NIMCP API
 * - Consistent error handling (nimcp_result_t + error messages)
 * - Attribute management (stack size, priority, detached)
 * - Single point for future platform abstraction (Windows)
 *
 * ALGORITHM:
 * 1. Validate parameters (thread and start_routine required)
 * 2. Initialize pthread attributes
 * 3. If attr provided:
 *    a. Set stack size (if non-zero)
 *    b. Set detached state (if requested)
 *    c. Set priority (if >0 and scheduling available)
 * 4. Call pthread_create with configured attributes
 * 5. Destroy attribute object (no longer needed)
 * 6. Return success or error with message
 *
 * WHY CONDITIONAL ATTRIBUTES:
 * - Defaults work for most cases (NULL attr → default attributes)
 * - Custom attributes when needed (large stack, real-time priority)
 * - Avoids forcing callers to understand pthread_attr complexity
 *
 * ATTRIBUTES SUPPORTED:
 * - stack_size: Bytes for thread stack (0 = default ~2MB)
 * - detached: Automatic cleanup (no join needed)
 * - priority: Scheduling priority (POSIX priority scheduling)
 *
 * WHY PRIORITY SCHEDULING CONDITIONAL:
 * - Not all systems support POSIX priority scheduling
 * - _POSIX_PRIORITY_SCHEDULING compile-time feature test
 * - Gracefully degrades if unavailable
 *
 * ERROR CASES:
 * - EINVAL: Invalid attribute value
 * - EAGAIN: Insufficient resources (too many threads)
 * - EPERM: No permission for requested priority
 *
 * COMPLEXITY: O(1) (pthread_create is system call)
 * THREAD SAFETY: Fully safe (creates independent thread)
 *
 * @param thread Output parameter for thread handle
 * @param start_routine Thread entry point function
 * @param arg Argument passed to start_routine
 * @param attr Thread attributes (NULL for defaults)
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_thread_create(nimcp_thread_t* thread, void* (*start_routine)(void*), void* arg,
                                   const thread_attr_t* attr)
{
    // Validate required parameters
    // WHY CHECK: pthread_create with NULL thread or start_routine is UB
    if (!thread || !start_routine) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid thread parameters");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Initialize pthread attributes
    // WHY LOCAL: Attribute object only needed during pthread_create
    pthread_attr_t pthread_attr;
    pthread_attr_init(&pthread_attr);

    // Configure attributes if provided
    if (attr) {
        // Stack size: Custom size for deep recursion or memory constraints
        if (attr->stack_size > 0) {
            pthread_attr_setstacksize(&pthread_attr, attr->stack_size);
        }

        // Detached: Thread cleans up automatically on exit (no join needed)
        if (attr->detached) {
            pthread_attr_setdetachstate(&pthread_attr, PTHREAD_CREATE_DETACHED);
        }

// Priority: Real-time or background priority (if supported)
#ifdef _POSIX_PRIORITY_SCHEDULING
        if (attr->priority > 0) {
            struct sched_param param;
            param.sched_priority = attr->priority;
            pthread_attr_setschedparam(&pthread_attr, &param);
        }
#endif
    }

    // Create thread with configured attributes
    int result = pthread_create(thread, &pthread_attr, start_routine, arg);

    // Clean up attribute object (no longer needed after pthread_create)
    // WHY DESTROY: Free any internal resources allocated by pthread_attr_*
    pthread_attr_destroy(&pthread_attr);

    // Check result and set error if failed
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Thread creation failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Wait for thread to finish (Adapter for pthread_join)
 *
 * WHY NEEDED:
 * - Synchronize with thread termination
 * - Retrieve thread return value
 * - Reclaim thread resources
 *
 * SEMANTICS:
 * - Blocks until target thread exits
 * - Can only join non-detached threads
 * - Thread must not have been joined already
 *
 * ERROR CASES:
 * - EINVAL: Thread is not joinable (detached or already joined)
 * - ESRCH: No thread with given ID
 * - EDEADLK: Deadlock detected (thread joining itself)
 *
 * COMPLEXITY: O(1) system call, blocks until thread exits
 * THREAD SAFETY: Fully safe (pthread_join is thread-safe)
 *
 * @param thread Thread handle to join
 * @param retval Output parameter for thread return value (NULL if not needed)
 * @return NIMCP_SUCCESS or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_thread_join(nimcp_thread_t thread, void** retval)
{
    int result = pthread_join(thread, retval);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Thread join failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }
    return NIMCP_SUCCESS;
}

/**
 * @brief Detach thread for automatic cleanup (Adapter for pthread_detach)
 *
 * WHY DETACH:
 * - No need to join (fire-and-forget thread)
 * - Automatic resource reclamation on exit
 * - Simpler lifecycle management
 *
 * SEMANTICS:
 * - Thread resources freed immediately on exit
 * - Cannot join after detach
 * - Useful for background/daemon threads
 *
 * ERROR CASES:
 * - EINVAL: Thread already detached
 * - ESRCH: No thread with given ID
 *
 * COMPLEXITY: O(1) system call
 * THREAD SAFETY: Fully safe
 *
 * @param thread Thread handle to detach
 * @return NIMCP_SUCCESS or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_thread_detach(nimcp_thread_t thread)
{
    int result = pthread_detach(thread);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Thread detach failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }
    return NIMCP_SUCCESS;
}

/**
 * @brief Exit current thread (Adapter for pthread_exit)
 *
 * WHY WRAPPER:
 * - Consistent naming with nimcp_thread_create
 * - Explicit return value passing
 * - Future platform abstraction point
 *
 * SEMANTICS:
 * - Terminates calling thread immediately
 * - Return value available to pthread_join
 * - Does NOT return (marked noreturn in header)
 *
 * WHY PASS RETVAL:
 * - Communicate status to joining thread
 * - Can pass pointer to result structure
 * - NULL is valid (no return value)
 *
 * THREAD SAFETY: Fully safe (only affects calling thread)
 *
 * @param retval Return value for pthread_join
 */
void nimcp_thread_exit(void* retval)
{
    pthread_exit(retval);
}

/**
 * @brief Get current thread ID (Adapter for pthread_self)
 *
 * WHY NEEDED:
 * - Thread can identify itself
 * - Compare with other thread IDs
 * - Use as key in thread-specific data structures
 *
 * COMPLEXITY: O(1) (typically reads TLS)
 * THREAD SAFETY: Fully safe
 *
 * @return Current thread handle
 */
nimcp_thread_t nimcp_thread_self(void)
{
    return pthread_self();
}

/**
 * @brief Compare two thread IDs (Adapter for pthread_equal)
 *
 * WHY NOT ==:
 * - pthread_t is opaque type (may not be integer)
 * - Portable comparison requires pthread_equal
 * - Some systems use struct for pthread_t
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (read-only comparison)
 *
 * @param t1 First thread ID
 * @param t2 Second thread ID
 * @return true if equal, false otherwise
 */
bool nimcp_thread_equal(nimcp_thread_t t1, nimcp_thread_t t2)
{
    return pthread_equal(t1, t2);
}

/**
 * @brief Execute function exactly once (Adapter for pthread_once)
 *
 * WHY NEEDED:
 * - Thread-safe initialization (singleton pattern)
 * - Lazy initialization of global state
 * - No race conditions even with multiple threads
 *
 * ALGORITHM:
 * 1. Check if once_control indicates already initialized
 * 2. If not, acquire internal lock
 * 3. Double-check (another thread may have initialized)
 * 4. If still not initialized, call init_routine
 * 5. Mark once_control as initialized
 * 6. Release lock
 *
 * WHY PTHREAD_ONCE (not manual flag):
 * - Built-in synchronization (no manual mutex needed)
 * - Efficient fast path (atomic check)
 * - Portable across platforms
 * - Correct in all edge cases
 *
 * TYPICAL USAGE:
 *   static nimcp_once_t once = NIMCP_ONCE_INIT;
 *   nimcp_once(&once, initialize_module);
 *   // initialize_module called exactly once across all threads
 *
 * COMPLEXITY: O(1) after first call (fast atomic check)
 * THREAD SAFETY: Guaranteed single execution
 *
 * @param once_control Control variable (must be static)
 * @param init_routine Function to call exactly once
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_once(nimcp_once_t* once_control, void (*init_routine)(void))
{
    // Validate parameters
    if (!once_control || !init_routine) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid once parameters");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_once(once_control, init_routine);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "pthread_once failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Mutex Operations
//=============================================================================

/**
 * @brief Initialize mutex with attributes (Factory + Adapter pattern)
 *
 * WHY FACTORY PATTERN:
 * - Creates different mutex types based on attributes
 * - Encapsulates pthread attribute complexity
 * - Single creation point for all mutex types
 *
 * MUTEX TYPES:
 * - NORMAL (default): Fast, no recursion checking, UB on relock
 * - RECURSIVE: Can be locked multiple times by same thread
 * - ERRORCHECK: Returns error on recursive lock (debugging)
 *
 * ALGORITHM:
 * 1. Validate mutex pointer
 * 2. Initialize pthread_mutexattr_t
 * 3. If attr provided, translate type:
 *    - MUTEX_TYPE_RECURSIVE → PTHREAD_MUTEX_RECURSIVE
 *    - MUTEX_TYPE_ERRORCHECK → PTHREAD_MUTEX_ERRORCHECK
 *    - MUTEX_TYPE_NORMAL → PTHREAD_MUTEX_NORMAL
 * 4. Set attribute type
 * 5. Initialize mutex with attributes
 * 6. Destroy attribute object (no longer needed)
 *
 * WHY SUPPORT MULTIPLE TYPES:
 * - NORMAL: Fastest, for simple critical sections
 * - RECURSIVE: Simplifies complex call chains (lock held across layers)
 *   Example: Function A locks, calls Function B which locks same mutex
 * - ERRORCHECK: Catches programming errors during development
 *   Example: Detects if function forgets to unlock before re-locking
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (creates new independent mutex)
 *
 * @param mutex Mutex to initialize
 * @param attr Mutex attributes (NULL for default NORMAL type)
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_mutex_init(nimcp_mutex_t* mutex, const mutex_attr_t* attr)
{
    if (!mutex) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid mutex pointer");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    pthread_mutexattr_t mutex_attr;
    pthread_mutexattr_init(&mutex_attr);

    // Translate NIMCP mutex types to pthread types
    if (attr) {
        int type;
        switch (attr->type) {
            case MUTEX_TYPE_RECURSIVE:
                type = PTHREAD_MUTEX_RECURSIVE;
                break;
            case MUTEX_TYPE_ERRORCHECK:
                type = PTHREAD_MUTEX_ERRORCHECK;
                break;
            default:
                type = PTHREAD_MUTEX_NORMAL;
        }
        pthread_mutexattr_settype(&mutex_attr, type);
    }

    // Initialize mutex with attributes
    int result = pthread_mutex_init(mutex, &mutex_attr);

    // Clean up attribute object
    pthread_mutexattr_destroy(&mutex_attr);

    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex initialization failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Destroy mutex (Adapter for pthread_mutex_destroy)
 *
 * WHY DESTROY:
 * - Free kernel resources (futex on Linux)
 * - Clean shutdown
 * - Detect programming errors (destroying locked mutex fails)
 *
 * PRECONDITIONS:
 * - Mutex must not be locked
 * - No threads waiting on mutex
 *
 * ERROR CASES:
 * - EBUSY: Mutex is locked or has waiters
 * - EINVAL: Invalid mutex
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Caller must ensure no concurrent use
 *
 * @param mutex Mutex to destroy
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_mutex_destroy(nimcp_mutex_t* mutex)
{
    if (!mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_mutex_destroy(mutex);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex destruction failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Lock mutex (Adapter for pthread_mutex_lock)
 *
 * WHY LOCK:
 * - Protect critical section (shared data access)
 * - Ensure mutual exclusion (only one thread in section)
 * - Serialize access to non-thread-safe operations
 *
 * SEMANTICS:
 * - Blocks if mutex already locked
 * - Acquires lock when available
 * - Must be unlocked by same thread
 *
 * CRITICAL SECTION PATTERN:
 *   nimcp_mutex_lock(&mutex);
 *   // ... critical section ...
 *   nimcp_mutex_unlock(&mutex);
 *
 * ERROR CASES:
 * - EINVAL: Invalid mutex or uninitialized
 * - EDEADLK: Deadlock detected (ERRORCHECK or RECURSIVE type)
 *
 * COMPLEXITY: O(1) uncontended, O(n) if queued (n = number of waiters)
 * THREAD SAFETY: Fully safe (that's the point!)
 *
 * @param mutex Mutex to lock
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_mutex_lock(nimcp_mutex_t* mutex)
{
    if (!mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_mutex_lock(mutex);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex lock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Try to lock mutex without blocking (Adapter for pthread_mutex_trylock)
 *
 * WHY TRYLOCK:
 * - Non-blocking alternative to lock
 * - Avoid deadlock (can back off and retry)
 * - Polling pattern (try lock, do other work, retry)
 *
 * SEMANTICS:
 * - Returns immediately (never blocks)
 * - NIMCP_SUCCESS if lock acquired
 * - NIMCP_BUSY if already locked (not an error!)
 *
 * TYPICAL USAGE (opportunistic locking):
 *   if (nimcp_mutex_trylock(&mutex) == NIMCP_SUCCESS) {
 *       // Got lock, do critical section
 *       work_on_shared_data();
 *       nimcp_mutex_unlock(&mutex);
 *   } else {
 *       // Couldn't get lock, do something else
 *       work_on_local_data();
 *   }
 *
 * WHY NIMCP_BUSY (not error):
 * - EBUSY is expected outcome (mutex already locked)
 * - Not a failure, just status information
 * - Consistent with timeout behavior
 *
 * COMPLEXITY: O(1) (always non-blocking)
 * THREAD SAFETY: Fully safe
 *
 * @param mutex Mutex to try locking
 * @return NIMCP_SUCCESS if locked, NIMCP_BUSY if already locked, NIMCP_ERROR_* on error
 */
nimcp_result_t nimcp_mutex_trylock(nimcp_mutex_t* mutex)
{
    if (!mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_mutex_trylock(mutex);

    // EBUSY means mutex is locked (expected, not error)
    if (result == EBUSY) {
        return NIMCP_BUSY;
    } else if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex trylock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Unlock mutex (Adapter for pthread_mutex_unlock)
 *
 * WHY UNLOCK:
 * - Release critical section
 * - Allow other threads to acquire lock
 * - Required pairing with lock (must unlock what you locked)
 *
 * SEMANTICS:
 * - Must be called by thread that locked mutex
 * - Wakes one waiting thread (if any)
 * - Undefined behavior if unlocking non-locked mutex
 *
 * ERROR CASES:
 * - EINVAL: Invalid mutex
 * - EPERM: Mutex not locked by current thread (ERRORCHECK type)
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (must be called by locking thread)
 *
 * @param mutex Mutex to unlock
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_mutex_unlock(nimcp_mutex_t* mutex)
{
    if (!mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_mutex_unlock(mutex);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Mutex unlock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Spinlock Operations (implemented as mutexes for portability)
//=============================================================================

/**
 * @brief Initialize spinlock (Adapter for pthread_mutex_init)
 *
 * WHY MUTEX NOT SPINLOCK:
 * - pthread_spin_lock not universally available (optional POSIX feature)
 * - Mutex provides better portability across platforms
 * - For short critical sections, mutex performance is comparable
 * - Futex-based mutexes (Linux) already spin briefly before blocking
 *
 * WHY SPINLOCK INTERFACE:
 * - Semantic clarity: indicates intent for short critical sections
 * - Future optimization: could use real spinlocks on platforms that support them
 * - Matches expected API in glial cell modules
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (creates new independent lock)
 *
 * @param lock Spinlock to initialize
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_spinlock_init(nimcp_spinlock_t* lock)
{
    if (!lock) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid spinlock pointer");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_mutex_init(lock, NULL);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Spinlock initialization failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Destroy spinlock (Adapter for pthread_mutex_destroy)
 *
 * WHY DESTROY:
 * - Free kernel resources (futex on Linux)
 * - Clean shutdown
 * - Detect programming errors (destroying locked spinlock fails)
 *
 * PRECONDITIONS:
 * - Spinlock must not be locked
 * - No threads waiting on spinlock
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Caller must ensure no concurrent use
 *
 * @param lock Spinlock to destroy
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_spinlock_destroy(nimcp_spinlock_t* lock)
{
    if (!lock) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_mutex_destroy(lock);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Spinlock destruction failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Lock spinlock (Adapter for pthread_mutex_lock)
 *
 * WHY LOCK:
 * - Protect critical section in glial cells (short duration)
 * - Ensure mutual exclusion for activity score updates
 * - Serialize access to monitored synapse arrays
 *
 * SEMANTICS:
 * - Blocks if spinlock already locked
 * - Acquires lock when available
 * - Must be unlocked by same thread
 *
 * TYPICAL USAGE (microglia activity tracking):
 *   nimcp_spinlock_lock(&microglia->lock);
 *   microglia->synapse_activity_scores[idx] = new_score;
 *   nimcp_spinlock_unlock(&microglia->lock);
 *
 * COMPLEXITY: O(1) uncontended
 * THREAD SAFETY: Fully safe
 *
 * @param lock Spinlock to lock
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_spinlock_lock(nimcp_spinlock_t* lock)
{
    if (!lock) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_mutex_lock(lock);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Spinlock lock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Unlock spinlock (Adapter for pthread_mutex_unlock)
 *
 * WHY UNLOCK:
 * - Release critical section
 * - Allow other threads to acquire lock
 * - Required pairing with lock
 *
 * SEMANTICS:
 * - Must be called by thread that locked spinlock
 * - Wakes one waiting thread (if any)
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (must be called by locking thread)
 *
 * @param lock Spinlock to unlock
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_spinlock_unlock(nimcp_spinlock_t* lock)
{
    if (!lock) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_mutex_unlock(lock);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Spinlock unlock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Read-Write Lock Operations
//=============================================================================

/**
 * @brief Initialize read-write lock (Adapter for pthread_rwlock_init)
 *
 * WHY READ-WRITE LOCKS:
 * - Allow multiple concurrent readers (shared access)
 * - Exclusive write access (only one writer, no readers)
 * - Better performance for read-heavy workloads
 * - Used in neuromodulator system for concurrent reads
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (creates new independent lock)
 *
 * @param lock Read-write lock to initialize
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_rwlock_init(nimcp_rwlock_t* lock)
{
    if (!lock) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid rwlock pointer");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_rwlock_init(lock, NULL);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "RWlock initialization failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Destroy read-write lock (Adapter for pthread_rwlock_destroy)
 *
 * PRECONDITIONS:
 * - Lock must not be held by any thread
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Caller must ensure no concurrent use
 *
 * @param lock Read-write lock to destroy
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_rwlock_destroy(nimcp_rwlock_t* lock)
{
    if (!lock) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_rwlock_destroy(lock);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "RWlock destruction failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Acquire read lock (Adapter for pthread_rwlock_rdlock)
 *
 * WHY READ LOCK:
 * - Multiple readers can hold lock simultaneously
 * - Blocks if writer holds lock
 * - Allows concurrent read access to shared data
 *
 * TYPICAL USAGE (neuromodulator levels):
 *   nimcp_rwlock_rdlock(&neuromod->lock);
 *   float level = neuromod->concentration;
 *   nimcp_rwlock_unlock(&neuromod->lock);
 *
 * COMPLEXITY: O(1) if no writers
 * THREAD SAFETY: Fully safe
 *
 * @param lock Read-write lock
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_rwlock_rdlock(nimcp_rwlock_t* lock)
{
    if (!lock) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_rwlock_rdlock(lock);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "RWlock rdlock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Acquire write lock (Adapter for pthread_rwlock_wrlock)
 *
 * WHY WRITE LOCK:
 * - Exclusive access (no readers, no other writers)
 * - Blocks until all readers and writers release lock
 * - Required for modifying shared data
 *
 * TYPICAL USAGE (neuromodulator update):
 *   nimcp_rwlock_wrlock(&neuromod->lock);
 *   neuromod->concentration += delta;
 *   nimcp_rwlock_unlock(&neuromod->lock);
 *
 * COMPLEXITY: O(1) once acquired (may wait for readers to finish)
 * THREAD SAFETY: Fully safe
 *
 * @param lock Read-write lock
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_rwlock_wrlock(nimcp_rwlock_t* lock)
{
    if (!lock) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_rwlock_wrlock(lock);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "RWlock wrlock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Unlock read-write lock (Adapter for pthread_rwlock_unlock)
 *
 * WHY UNLOCK:
 * - Release read or write lock
 * - Allow other threads to acquire lock
 * - Works for both read and write locks
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Must be called by thread that acquired lock
 *
 * @param lock Read-write lock to unlock
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_rwlock_unlock(nimcp_rwlock_t* lock)
{
    if (!lock) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_rwlock_unlock(lock);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "RWlock unlock failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Condition Variables
//=============================================================================

/**
 * @brief Initialize condition variable (Adapter for pthread_cond_init)
 *
 * WHY CONDITION VARIABLES:
 * - Efficient thread notification (no polling)
 * - Wait for condition to become true
 * - Producer-consumer pattern
 * - Work queue pattern
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Fully safe (creates new independent condvar)
 *
 * @param cond Condition variable to initialize
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_cond_init(nimcp_cond_t* cond)
{
    if (!cond) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid cond pointer");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_cond_init(cond, NULL);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Cond initialization failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Destroy condition variable (Adapter for pthread_cond_destroy)
 *
 * WHY DESTROY:
 * - Free kernel resources
 * - Clean shutdown
 *
 * PRECONDITIONS:
 * - No threads waiting on condition variable
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Caller must ensure no concurrent waiters
 *
 * @param cond Condition variable to destroy
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_cond_destroy(nimcp_cond_t* cond)
{
    if (!cond) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_cond_destroy(cond);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Cond destruction failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Wait on condition variable (Adapter for pthread_cond_wait)
 *
 * WHY WAIT:
 * - Efficient blocking (no busy-wait)
 * - Atomically releases mutex and waits
 * - Re-acquires mutex when signaled
 *
 * ALGORITHM (atomic):
 * 1. Add thread to condition variable wait queue
 * 2. Release mutex (allow signaler to modify condition)
 * 3. Block thread (sleep until signaled)
 * 4. Wake when signaled (by cond_signal or cond_broadcast)
 * 5. Re-acquire mutex (lock held when returning)
 *
 * WHY ATOMIC UNLOCK+WAIT:
 * - Prevents race: Signaler can't signal between unlock and wait
 * - Without atomicity: Signal could be lost (wake before wait)
 *
 * STANDARD PATTERN (always use while loop!):
 *   nimcp_mutex_lock(&mutex);
 *   while (!condition_is_true) {
 *       nimcp_cond_wait(&cond, &mutex);  // Releases mutex, waits, re-acquires
 *   }
 *   // condition is true, mutex is locked
 *   process_condition();
 *   nimcp_mutex_unlock(&mutex);
 *
 * WHY WHILE LOOP (not if):
 * - SPURIOUS WAKEUPS: pthread_cond_wait can wake without signal (OS behavior)
 * - MULTIPLE WAITERS: Another thread may consume condition before this thread runs
 * - CORRECTNESS: Always re-check condition after wake
 *
 * COMPLEXITY: O(1) to enter wait, blocks until signaled
 * THREAD SAFETY: Fully safe (atomic unlock+wait)
 *
 * @param cond Condition variable to wait on
 * @param mutex Mutex that must be locked by caller (will be released during wait)
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_cond_wait(nimcp_cond_t* cond, nimcp_mutex_t* mutex)
{
    if (!cond || !mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_cond_wait(cond, mutex);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Cond wait failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Wait on condition variable with timeout (Adapter for pthread_cond_timedwait)
 *
 * WHY TIMEOUT:
 * - Prevent indefinite blocking (watchdog pattern)
 * - Implement polling with backoff (check periodically)
 * - Handle slow/unresponsive operations
 *
 * ALGORITHM:
 * 1. Get current absolute time (CLOCK_REALTIME)
 * 2. Add timeout_ms to current time:
 *    a. Add milliseconds to seconds (ms / 1000)
 *    b. Add remainder to nanoseconds ((ms % 1000) * 1000000)
 * 3. Handle nanosecond overflow (≥1 billion → carry to seconds)
 * 4. Call pthread_cond_timedwait with absolute deadline
 * 5. If ETIMEDOUT, return NIMCP_BUSY (consistent with trylock)
 *
 * WHY ABSOLUTE TIME (not relative):
 * - pthread_cond_timedwait requires absolute time
 * - Prevents drift with multiple timed operations
 * - Standard POSIX interface
 *
 * WHY CLOCK_REALTIME (not MONOTONIC):
 * - pthread_cond_timedwait uses REALTIME by default
 * - Some systems support MONOTONIC (better but optional)
 * - Trade-off: Simplicity vs robustness to clock changes
 *
 * NANOSECOND OVERFLOW HANDLING:
 * - 1000 ms = 1,000,000,000 ns = 1 second
 * - Adding ms can cause ns ≥ 1,000,000,000
 * - Must carry overflow to seconds field
 *
 * TYPICAL USAGE (timeout pattern):
 *   nimcp_mutex_lock(&mutex);
 *   while (!condition && !timeout) {
 *       nimcp_result_t r = nimcp_cond_timedwait(&cond, &mutex, 5000);  // 5s
 *       if (r == NIMCP_BUSY) {
 *           timeout = true;  // Timed out waiting
 *       }
 *   }
 *   nimcp_mutex_unlock(&mutex);
 *
 * COMPLEXITY: O(1) to enter wait, blocks up to timeout_ms
 * THREAD SAFETY: Fully safe (atomic unlock+wait)
 *
 * @param cond Condition variable to wait on
 * @param mutex Mutex that must be locked by caller
 * @param timeout_ms Timeout in milliseconds
 * @return NIMCP_SUCCESS if signaled, NIMCP_BUSY if timeout, NIMCP_ERROR_* on error
 */
nimcp_result_t nimcp_cond_timedwait(nimcp_cond_t* cond, nimcp_mutex_t* mutex, uint32_t timeout_ms)
{
    if (!cond || !mutex) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Get current absolute time
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    // Add timeout (convert milliseconds to seconds and nanoseconds)
    ts.tv_sec += timeout_ms / 1000;               // Add seconds
    ts.tv_nsec += (timeout_ms % 1000) * 1000000;  // Add nanoseconds

    // Handle nanosecond overflow
    // WHY NEEDED: tv_nsec must be in [0, 999999999]
    if (ts.tv_nsec >= 1000000000) {
        ts.tv_sec += 1;            // Carry to seconds
        ts.tv_nsec -= 1000000000;  // Normalize nanoseconds
    }

    int result = pthread_cond_timedwait(cond, mutex, &ts);

    // ETIMEDOUT is expected outcome (not error)
    if (result == ETIMEDOUT) {
        return NIMCP_BUSY;  // Consistent with trylock semantics
    } else if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Cond timedwait failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Signal one waiting thread (Adapter for pthread_cond_signal)
 *
 * WHY SIGNAL:
 * - Wake one waiter (efficient for single consumer)
 * - Notify condition change
 * - Producer-consumer pattern
 *
 * SEMANTICS:
 * - Wakes ONE waiting thread (scheduler chooses which)
 * - If no waiters, signal is lost (no queuing)
 * - Waiter must re-acquire mutex before returning from wait
 *
 * TYPICAL USAGE (producer):
 *   nimcp_mutex_lock(&mutex);
 *   queue_push(item);        // Modify condition
 *   condition = true;        // Set flag
 *   nimcp_cond_signal(&cond);  // Wake one waiter
 *   nimcp_mutex_unlock(&mutex);
 *
 * WHY SIGNAL UNDER LOCK:
 * - Not required but recommended
 * - Prevents "wake-up waiting" race
 * - Waiter can immediately check condition
 *
 * SIGNAL vs BROADCAST:
 * - Use signal for single consumer (work queue)
 * - Use broadcast for multiple consumers (barrier)
 *
 * COMPLEXITY: O(1) (wakes one thread)
 * THREAD SAFETY: Fully safe
 *
 * @param cond Condition variable to signal
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_cond_signal(nimcp_cond_t* cond)
{
    if (!cond) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_cond_signal(cond);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Cond signal failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Signal all waiting threads (Adapter for pthread_cond_broadcast)
 *
 * WHY BROADCAST:
 * - Wake ALL waiters (multiple consumers)
 * - Barrier pattern (release all threads at once)
 * - Shutdown signal (wake all to exit)
 *
 * SEMANTICS:
 * - Wakes ALL waiting threads
 * - If no waiters, broadcast is lost (no queuing)
 * - All woken threads must acquire mutex (serialized)
 *
 * TYPICAL USAGE (shutdown):
 *   nimcp_mutex_lock(&mutex);
 *   shutdown_flag = true;        // Set condition
 *   nimcp_cond_broadcast(&cond);   // Wake all waiters
 *   nimcp_mutex_unlock(&mutex);
 *
 * THUNDERING HERD CONCERN:
 * - All threads wake simultaneously
 * - Serialize on mutex acquisition
 * - Most threads may find condition already consumed
 * - Trade-off: Simplicity vs efficiency
 *
 * WHEN TO USE BROADCAST (not signal):
 * - Multiple threads need to check same condition
 * - Barrier synchronization
 * - Shutdown/cleanup operations
 *
 * COMPLEXITY: O(n) where n = number of waiters (wake all + n context switches)
 * THREAD SAFETY: Fully safe
 *
 * @param cond Condition variable to broadcast
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_SYSTEM
 */
nimcp_result_t nimcp_cond_broadcast(nimcp_cond_t* cond)
{
    if (!cond) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    int result = pthread_cond_broadcast(cond);
    if (result != 0) {
        set_thread_error(NIMCP_ERROR_SYSTEM, "Cond broadcast failed: %s", strerror(result));
        return NIMCP_ERROR_SYSTEM;
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Named Resource Locks
//=============================================================================

/**
 * @brief Hash string to bucket index (DJBX33A hash)
 *
 * WHY HASH FUNCTION:
 * - Map arbitrary strings to fixed bucket range [0, 255]
 * - Distribute keys evenly across buckets
 * - Fast computation (one multiply + add per character)
 *
 * ALGORITHM (DJBX33A):
 * 1. Initialize hash = 5381 (prime seed)
 * 2. For each character c:
 *    hash = hash * 33 + c
 * 3. Modulo 256 (bucket count)
 *
 * WHY DJBX33A (DJB2):
 * - Fast: One multiply, one add per character (no division)
 * - Good distribution: Prime multiplier (33) spreads bits
 * - Low collision rate: ~1-2% for typical string sets
 * - Widely used: Proven in many hash table implementations
 *
 * WHY 5381 SEED:
 * - Prime number (good mathematical properties)
 * - Empirically tested by DJB (Dan Bernstein)
 * - Prevents zero hash for short strings
 *
 * WHY 33 MULTIPLIER:
 * - Prime number (good distribution)
 * - Power of 2 + 1 (fast: hash * 33 = (hash << 5) + hash)
 * - Empirically optimal for ASCII strings
 *
 * ALTERNATIVES CONSIDERED:
 * - FNV-1a: Similar speed, slightly more complex (XOR + multiply)
 * - CRC32: Hardware support but overkill for our use case
 * - Simple sum: Terrible distribution (all anagrams collide!)
 * - One-at-a-time (Jenkins): Slower, no advantage for our strings
 *
 * COLLISION HANDLING:
 * - Chaining: Each bucket has linked list
 * - Expected chain length: N / 256 (where N = total locks)
 * - Typical: <1 entry per bucket (<100 locks total)
 *
 * COMPLEXITY: O(n) where n = string length (typically 10-30 characters)
 * THREAD SAFETY: Fully safe (read-only, no shared state)
 *
 * @param str String to hash
 * @return Bucket index in [0, 255]
 */
static unsigned int hash_string(const char* str)
{
    unsigned int hash = 5381;  // Prime seed
    int c;

    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }

    return hash % RESOURCE_LOCK_BUCKETS;  // Map to [0, 255]
}

/**
 * @brief Get or create named resource lock (Registry pattern)
 *
 * WHY NAMED LOCKS:
 * - Intuitive: "brain_network" more readable than &global_mutex_42
 * - Automatic sharing: Same name = same lock across all modules
 * - Lifecycle management: Reference counting + automatic cleanup
 * - No global variables: Locks created on demand
 *
 * ALGORITHM:
 * 1. Initialize resource table if needed (lazy init)
 * 2. Compute hash bucket for resource_id
 * 3. Acquire bucket mutex (fine-grained locking)
 * 4. Search bucket's entry list for resource_id
 * 5. If found:
 *    a. Increment ref_count (another holder)
 *    b. Return existing mutex
 * 6. If not found (first request):
 *    a. Allocate resource_entry_t
 *    b. Duplicate resource_id string
 *    c. Allocate and initialize mutex
 *    d. Set ref_count = 1
 *    e. Add to head of bucket's entry list
 *    f. Return new mutex
 * 7. Release bucket mutex
 *
 * WHY REFERENCE COUNTING:
 * - Automatic cleanup: Last release destroys lock
 * - Safe sharing: Lock lives while any holder exists
 * - Memory efficient: Lock freed when not needed
 *
 * REFERENCE COUNT LIFECYCLE:
 *   get_resource_lock("X"):   Create, ref=1
 *   get_resource_lock("X"):   Found,  ref=2
 *   release_resource_lock("X"): ref=1 (keep alive)
 *   release_resource_lock("X"): ref=0 (destroy)
 *
 * WHY BUCKETED LOCKING:
 * - Concurrency: 256 buckets = 256-way parallelism
 * - Reduced contention: Only threads accessing same bucket block each other
 * - Scalability: Sub-linear scaling with thread count
 *
 * TYPICAL USAGE:
 *   nimcp_mutex_t* lock;
 *   nimcp_get_resource_lock("brain_network", &lock);
 *   nimcp_mutex_lock(lock);
 *   // ... critical section ...
 *   nimcp_mutex_unlock(lock);
 *   nimcp_release_resource_lock("brain_network");
 *
 * ERROR CASES:
 * - NIMCP_ERROR_INVALID_PARAM: NULL resource_id or mutex
 * - NIMCP_ERROR_MEMORY: Allocation failed (out of memory)
 * - NIMCP_ERROR_SYSTEM: Mutex initialization failed
 *
 * COMPLEXITY:
 * - Average: O(1) (hash + short list search)
 * - Worst: O(n) where n = entries in bucket (rare collision)
 * - Typical: O(1) with ~0-1 entries per bucket
 *
 * THREAD SAFETY: Fully safe (bucket mutex protects entry list)
 *
 * @param resource_id String identifier for resource
 * @param mutex Output parameter for mutex pointer
 * @return NIMCP_SUCCESS or NIMCP_ERROR_*
 */
nimcp_result_t nimcp_get_resource_lock(const char* resource_id, nimcp_mutex_t** mutex)
{
    // Validate parameters
    if (!resource_id || !mutex) {
        set_thread_error(NIMCP_ERROR_INVALID_PARAM, "Invalid resource lock parameters");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Lazy initialization: Ensure table is initialized
    if (!resource_table.initialized) {
        nimcp_thread_init();
    }

    // Compute hash bucket (DJBX33A hash % 256)
    unsigned int bucket = hash_string(resource_id);

    // Acquire bucket mutex (fine-grained lock)
    // WHY BUCKET LOCK: Only blocks threads accessing this specific bucket
    pthread_mutex_lock(&resource_table.buckets[bucket].bucket_mutex);

    // Search bucket's entry list for resource_id
    resource_entry_t* entry = resource_table.buckets[bucket].entries;
    while (entry) {
        if (strcmp(entry->resource_id, resource_id) == 0) {
            // FOUND: Resource already exists, increment reference count
            entry->ref_count++;
            *mutex = entry->lock;
            pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
            return NIMCP_SUCCESS;
        }
        entry = entry->next;
    }

    // NOT FOUND: Create new resource entry (first request)

    // Allocate entry structure
    // WHY MALLOC: Dynamic number of locks (not known at compile time)
    entry = nimcp_malloc(sizeof(resource_entry_t));
    if (!entry) {
        pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
        set_thread_error(NIMCP_ERROR_MEMORY, "Failed to allocate resource entry");
        return NIMCP_ERROR_MEMORY;
    }

    // Duplicate resource_id string (use nimcp_malloc to match nimcp_free below)
    size_t id_len = strlen(resource_id);
    entry->resource_id = nimcp_malloc(id_len + 1);
    if (entry->resource_id) {
        strcpy(entry->resource_id, resource_id);
    }

    // Allocate mutex structure
    entry->lock = nimcp_malloc(sizeof(nimcp_mutex_t));

    // Check allocations
    if (!entry->resource_id || !entry->lock) {
        // Cleanup partial allocation
        nimcp_free(entry->resource_id);
        nimcp_free(entry->lock);
        nimcp_free(entry);
        pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
        set_thread_error(NIMCP_ERROR_MEMORY, "Failed to allocate resource lock");
        return NIMCP_ERROR_MEMORY;
    }

    // Initialize mutex
    if (pthread_mutex_init(entry->lock, NULL) != 0) {
        // Cleanup on init failure
        nimcp_free(entry->resource_id);
        nimcp_free(entry->lock);
        nimcp_free(entry);
        pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
        set_thread_error(NIMCP_ERROR_SYSTEM, "Failed to initialize resource mutex");
        return NIMCP_ERROR_SYSTEM;
    }

    // Set initial reference count
    entry->ref_count = 1;

    // Add to head of bucket's entry list (O(1) insertion)
    entry->next = resource_table.buckets[bucket].entries;
    resource_table.buckets[bucket].entries = entry;

    // Return mutex to caller
    *mutex = entry->lock;

    pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
    return NIMCP_SUCCESS;
}

/**
 * @brief Release named resource lock (decrement reference count)
 *
 * WHY RELEASE:
 * - Decrement reference count (holder no longer needs lock)
 * - Automatic cleanup: Destroy lock when ref_count reaches 0
 * - Memory efficiency: Free locks that are no longer needed
 *
 * ALGORITHM:
 * 1. Validate resource_id
 * 2. Compute hash bucket
 * 3. Acquire bucket mutex
 * 4. Search bucket's entry list for resource_id
 * 5. If found:
 *    a. Decrement ref_count
 *    b. If ref_count == 0 (last holder):
 *       - Remove from entry list
 *       - Destroy mutex
 *       - Free resource_id string
 *       - Free entry structure
 * 6. If not found: Return NIMCP_ERROR_NOT_FOUND
 * 7. Release bucket mutex
 *
 * WHY DESTROY ON ref_count==0:
 * - No holders remain (safe to destroy)
 * - Free memory (no memory leak)
 * - Free kernel resources (futex on Linux)
 *
 * REMOVAL FROM LIST:
 * - Two cases:
 *   1. Head of list: Update bucket->entries
 *   2. Middle/end: Update prev->next
 * - WHY TRACK PREV: Singly-linked list requires previous pointer to remove
 *
 * ERROR CASES:
 * - NIMCP_ERROR_INVALID_PARAM: NULL resource_id
 * - NIMCP_ERROR_NOT_FOUND: Resource doesn't exist (double-release?)
 *
 * TYPICAL USAGE:
 *   nimcp_mutex_t* lock;
 *   nimcp_get_resource_lock("brain_network", &lock);  // ref=1
 *   nimcp_mutex_lock(lock);
 *   // ... work ...
 *   nimcp_mutex_unlock(lock);
 *   nimcp_release_resource_lock("brain_network");  // ref=0, destroyed
 *
 * COMPLEXITY: O(n) where n = entries in bucket (typically 0-2)
 * THREAD SAFETY: Fully safe (bucket mutex protects entry list)
 *
 * @param resource_id String identifier for resource
 * @return NIMCP_SUCCESS or NIMCP_ERROR_INVALID_PARAM or NIMCP_ERROR_NOT_FOUND
 */
nimcp_result_t nimcp_release_resource_lock(const char* resource_id)
{
    if (!resource_id) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Compute hash bucket
    unsigned int bucket = hash_string(resource_id);

    // Acquire bucket mutex
    pthread_mutex_lock(&resource_table.buckets[bucket].bucket_mutex);

    // Search bucket's entry list for resource_id
    resource_entry_t* entry = resource_table.buckets[bucket].entries;
    resource_entry_t* prev = NULL;

    while (entry) {
        if (strcmp(entry->resource_id, resource_id) == 0) {
            // FOUND: Decrement reference count
            entry->ref_count--;

            if (entry->ref_count == 0) {
                // Last reference: Remove from list and destroy

                // Remove from linked list
                if (prev) {
                    // Middle or end of list
                    prev->next = entry->next;
                } else {
                    // Head of list
                    resource_table.buckets[bucket].entries = entry->next;
                }

                // Destroy mutex (free kernel resources)
                pthread_mutex_destroy(entry->lock);

                // Free allocated memory
                nimcp_free(entry->lock);
                nimcp_free(entry->resource_id);  // Free duplicated string
                nimcp_free(entry);               // Free entry structure
            }

            pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
            return NIMCP_SUCCESS;
        }

        // Move to next entry
        prev = entry;
        entry = entry->next;
    }

    // NOT FOUND: Resource doesn't exist
    pthread_mutex_unlock(&resource_table.buckets[bucket].bucket_mutex);
    return NIMCP_ERROR_NOT_FOUND;
}

//=============================================================================
// Cleanup
//=============================================================================

/**
 * @brief Clean up threading subsystem
 *
 * WHY CLEANUP:
 * - Free all remaining named resource locks
 * - Destroy all bucket mutexes
 * - Clean shutdown (no memory leaks)
 *
 * ALGORITHM:
 * 1. Check if initialized (skip if not)
 * 2. Acquire global mutex (table-level lock)
 * 3. For each of 256 buckets:
 *    a. Acquire bucket mutex
 *    b. Walk entry list:
 *       - Destroy mutex
 *       - Free resource_id string
 *       - Free entry structure
 *    c. Release bucket mutex
 *    d. Destroy bucket mutex
 * 4. Release global mutex
 * 5. Destroy global mutex
 * 6. Set initialized = false
 *
 * WHY DESTROY REMAINING LOCKS:
 * - Application may not have called release for all locks
 * - Clean exit (valgrind clean)
 * - Free all resources
 *
 * WARNING:
 * - Should be called at shutdown AFTER all threads stopped
 * - Not safe if threads still using locks
 * - Caller must ensure synchronization
 *
 * TYPICAL USAGE:
 *   // Shutdown sequence
 *   stop_all_threads();      // Join or cancel threads
 *   nimcp_thread_cleanup();  // Free threading resources
 *   exit(0);
 *
 * COMPLEXITY: O(n) where n = total number of named locks
 * THREAD SAFETY: NOT safe if threads still active (caller must ensure)
 */
void nimcp_thread_cleanup(void)
{
    if (!resource_table.initialized) {
        return;  // Nothing to clean up
    }

    // Acquire global mutex (table-level protection)
    pthread_mutex_lock(&resource_table.global_mutex);

    // Clean up all buckets
    for (int i = 0; i < RESOURCE_LOCK_BUCKETS; i++) {
        // Acquire bucket mutex
        pthread_mutex_lock(&resource_table.buckets[i].bucket_mutex);

        // Free all entries in this bucket
        resource_entry_t* entry = resource_table.buckets[i].entries;
        while (entry) {
            resource_entry_t* next = entry->next;

            // Destroy mutex and free memory
            pthread_mutex_destroy(entry->lock);
            nimcp_free(entry->lock);
            nimcp_free(entry->resource_id);
            nimcp_free(entry);

            entry = next;
        }

        // Release and destroy bucket mutex
        pthread_mutex_unlock(&resource_table.buckets[i].bucket_mutex);
        pthread_mutex_destroy(&resource_table.buckets[i].bucket_mutex);
    }

    // Release and destroy global mutex
    pthread_mutex_unlock(&resource_table.global_mutex);
    pthread_mutex_destroy(&resource_table.global_mutex);

    // Mark as uninitialized
    resource_table.initialized = false;
}
