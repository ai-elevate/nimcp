//=============================================================================
// nimcp_queue.c - Thread-Safe Circular Buffer Queue Implementation
//=============================================================================
// ARCHITECTURAL OVERVIEW:
// This module implements a thread-safe, bounded FIFO queue using a circular
// buffer. It's a fundamental building block used throughout NIMCP for
// producer-consumer patterns and asynchronous communication.
//
// KEY DESIGN: CIRCULAR BUFFER
// ============================
// WHY CIRCULAR: O(1) enqueue/dequeue vs O(n) for array shift
// - Linear buffer would require shifting all elements on dequeue
// - Circular buffer: just increment head/tail pointers (modulo)
// - Memory-efficient: Fixed allocation, no dynamic resizing
//
// CIRCULAR BUFFER MECHANICS:
//   [2][3][4][_][_][0][1]  <- Physical layout
//        ^           ^
//       tail       head
//
//   Logical order: 0,1,2,3,4
//   head points to next item to dequeue
//   tail points to next empty slot
//   Full when: (tail + 1) % max_size == head
//   Empty when: tail == head
//
// WHY RESERVE ONE SLOT:
// - Distinguishes full from empty without extra flag
// - Empty: head == tail
// - Full:  (tail + 1) % max_size == head
// - Without reserved slot, both states would have head == tail
//
// DESIGN PATTERNS USED:
// - Monitor Pattern: Mutex + condition variables for thread coordination
// - Bounded Buffer: Fixed-size producer-consumer queue
// - Strategy Pattern: Blocking vs non-blocking configurable
// - Template Method: Generic queue works with any item type
//
// THREAD SAFETY MECHANISM:
// ========================
// INVARIANTS (protected by mutex):
// - head, tail always valid indices (0 to max_size-1)
// - head == tail means empty
// - (tail + 1) % max_size == head means full
// - current_size always accurate
// - Statistics monotonically increasing
//
// CONDITION VARIABLES:
// - not_empty: Signaled when item enqueued, waited by dequeue
// - not_full:  Signaled when item dequeued, waited by enqueue
// - Prevents busy-waiting, enables efficient blocking
//
// BLOCKING VS NON-BLOCKING MODES:
// ================================
// NON-BLOCKING (is_blocking = false):
// - Enqueue on full: Return NIMCP_QUEUE_FULL immediately
// - Dequeue on empty: Return NIMCP_QUEUE_EMPTY immediately
// - Use case: Real-time systems, want fast failure
//
// BLOCKING (is_blocking = true):
// - Enqueue on full: Wait on not_full condition
// - Dequeue on empty: Wait on not_empty condition
// - Timeout support: timeout_ms = 0 means infinite
// - Use case: Producer-consumer coordination
//
// SOLID PRINCIPLES APPLIED:
// - Single Responsibility: Queue only manages FIFO semantics
// - Open/Closed: Extensible via configuration, closed for modification
// - Liskov Substitution: Any queue handle behaves identically
// - Interface Segregation: Clean separation of query vs mutation operations
// - Dependency Inversion: Depends on abstractions (handles, not implementations)
//
// PERFORMANCE CHARACTERISTICS:
// - Enqueue: O(1) - memcpy + pointer arithmetic
// - Dequeue: O(1) - memcpy + pointer arithmetic
// - Peek: O(1) - memcpy only
// - Clear: O(1) - pointer reset
// - Is Empty/Full: O(1) - arithmetic check
// - Memory: O(max_size * item_size) - pre-allocated
//
// SPACE COMPLEXITY:
// - Queue structure: ~64 bytes (struct overhead)
// - Buffer: max_size * item_size bytes
// - Total: O(max_size * item_size)
//
// CACHE EFFICIENCY:
// - Sequential access pattern (good cache locality)
// - Prefetching friendly (next item likely in cache)
// - Circular wrap-around: minor cache miss, negligible impact
//
// USE CASES IN NIMCP:
// - Event queuing (event system priority queues)
// - Thread pool task queues
// - Message buffering (protocol layer)
// - Stream input buffering
// - Logging message queues
//
// LIMITATIONS & TRADE-OFFS:
// - Fixed size: Must size appropriately (no dynamic resize)
// - One slot wasted: (max_size - 1) effective capacity
// - Generic type: Uses memcpy (no type safety, caller responsibility)
// - Mutex overhead: ~100ns per operation (necessary for thread safety)
//
//=============================================================================

#include "utils/nimcp_queue.h"
#include "utils/nimcp_thread.h"
#include "utils/nimcp_memory.h"
#include <string.h>

//=============================================================================
// Internal Structure Definition
//=============================================================================

/**
 * @brief Internal queue structure using circular buffer
 *
 * WHY THIS LAYOUT:
 * - Circular buffer for O(1) operations
 * - Mutex + condition variables for thread safety
 * - Statistics for monitoring and debugging
 *
 * MEMORY LAYOUT:
 * - Structure: ~64 bytes (on 64-bit system)
 * - Buffer: Separate allocation for flexibility
 *
 * INVARIANTS:
 * - 0 <= head < max_size
 * - 0 <= tail < max_size
 * - head == tail implies empty
 * - (tail + 1) % max_size == head implies full
 * - current_size == actual number of items
 * - total_enqueued >= total_dequeued always
 * - peak_size >= current_size always
 */
struct nimcp_queue {
    uint8_t* buffer;              // Circular buffer storage (byte array)
    size_t head;                  // Read position (next dequeue)
    size_t tail;                  // Write position (next enqueue)
    nimcp_queue_config_t config;  // Configuration (immutable after create)
    nimcp_queue_status_t status;  // Runtime statistics (mutable)
    nimcp_mutex_t mutex;          // Protects all mutable state
    nimcp_cond_t not_empty;       // Signaled when enqueue occurs
    nimcp_cond_t not_full;        // Signaled when dequeue occurs
};

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Checks if circular buffer is full
 *
 * WHY THIS FORMULA: (tail + 1) % max_size == head
 * - Tail points to next empty slot
 * - If incrementing tail would equal head, buffer is full
 * - This is why we reserve one slot (to distinguish from empty)
 *
 * ALTERNATIVE CONSIDERED: Use count field
 * - Rejected: Extra field to maintain, more state to synchronize
 * - Current approach: Zero extra state, just arithmetic
 *
 * COMPLEXITY: O(1) - Single modulo and comparison
 * THREAD SAFETY: Must be called with mutex held
 *
 * @param q Queue instance (must be non-null)
 * @return true if full (cannot enqueue), false otherwise
 */
static bool is_queue_full(struct nimcp_queue* q) {
    return ((q->tail + 1) % q->config.max_size) == q->head;
}

/**
 * @brief Checks if circular buffer is empty
 *
 * WHY THIS FORMULA: head == tail
 * - When all items consumed, read and write positions meet
 * - Simpler than tracking count
 *
 * COMPLEXITY: O(1) - Single comparison
 * THREAD SAFETY: Must be called with mutex held
 *
 * @param q Queue instance (must be non-null)
 * @return true if empty (cannot dequeue), false otherwise
 */
static bool is_queue_empty(struct nimcp_queue* q) {
    return q->head == q->tail;
}

//=============================================================================
// Public API Implementation
//=============================================================================

/**
 * @brief Creates a new thread-safe queue with specified configuration
 *
 * WHY SEPARATE ALLOCATION: Buffer and structure separate
 * - Allows buffer to be cache-aligned if needed
 * - Simplifies cleanup (two separate frees)
 * - Buffer size can be large, keep structure small
 *
 * ALGORITHM:
 * 1. Validate configuration (O(1))
 * 2. Allocate queue structure (O(1))
 * 3. Allocate circular buffer (O(n) where n = max_size * item_size)
 * 4. Initialize synchronization primitives (O(1))
 * 5. Initialize statistics to zero (O(1))
 *
 * ERROR HANDLING:
 * - Validates all inputs before allocation
 * - Cleans up partial allocations on failure
 * - Returns specific error codes (not just NULL)
 *
 * COMPLEXITY: O(n) where n = max_size * item_size (for buffer allocation)
 * MEMORY: Allocates max_size * item_size + sizeof(struct nimcp_queue)
 *
 * DESIGN PATTERN: Factory Method - Creates fully-initialized object
 *
 * @param config Configuration (max_size, item_size, blocking mode, timeout)
 * @param queue Output handle (receives created queue on success)
 * @return NIMCP_SUCCESS, NIMCP_INVALID_PARAM, NIMCP_NO_MEMORY, NIMCP_INIT_FAILED
 */
nimcp_result_t nimcp_queue_create(const nimcp_queue_config_t* config,
                                 nimcp_queue_handle_t* queue) {
    // Guard clause: Validate pointers
    if (!config || !queue) return NIMCP_INVALID_PARAM;

    // Guard clause: Validate configuration values
    // WHY: Zero size would cause division by zero in modulo operations
    if (config->max_size == 0 || config->item_size == 0) {
        return NIMCP_INVALID_PARAM;
    }

    // Allocate queue structure
    // WHY calloc: Zero-initializes statistics
    struct nimcp_queue* q = nimcp_calloc(1, sizeof(struct nimcp_queue));
    if (!q) return NIMCP_NO_MEMORY;

    // Allocate circular buffer
    // WHY malloc: Buffer content doesn't need initialization (will be overwritten)
    q->buffer = nimcp_malloc(config->max_size * config->item_size);
    if (!q->buffer) {
        nimcp_free(q);
        return NIMCP_NO_MEMORY;
    }

    // Copy configuration (immutable after creation)
    q->config = *config;

    // Initialize circular buffer indices
    // WHY both zero: Empty state is head == tail
    q->head = q->tail = 0;

    // Initialize statistics to zero
    // WHY memset: Ensures all counters start at zero
    memset(&q->status, 0, sizeof(nimcp_queue_status_t));

    // Initialize synchronization primitives
    // WHY separate if: Each can fail independently, need precise error handling
    if (nimcp_mutex_init(&q->mutex, NULL) != NIMCP_SUCCESS ||
        nimcp_cond_init(&q->not_empty) != NIMCP_SUCCESS ||
        nimcp_cond_init(&q->not_full) != NIMCP_SUCCESS) {
        // Cleanup on failure
        nimcp_free(q->buffer);
        nimcp_free(q);
        return NIMCP_INIT_FAILED;
    }

    *queue = q;
    return NIMCP_SUCCESS;
}

/**
 * @brief Destroys queue and frees all resources
 *
 * WHY THIS ORDER:
 * 1. Destroy synchronization primitives first (no more waiting threads)
 * 2. Free buffer (large allocation)
 * 3. Free structure (small allocation)
 *
 * THREAD SAFETY CONSIDERATION:
 * - Caller must ensure no other threads using queue
 * - We don't check for waiting threads (would require tracking)
 * - Undefined behavior if other threads still accessing
 *
 * DESIGN PATTERN: RAII (Resource Acquisition Is Initialization)
 * - All resources acquired in create
 * - All resources released in destroy
 * - No partial states
 *
 * COMPLEXITY: O(1) - Fixed number of operations
 *
 * @param queue Queue to destroy
 * @return NIMCP_SUCCESS or NIMCP_INVALID_PARAM
 */
nimcp_result_t nimcp_queue_destroy(nimcp_queue_handle_t queue) {
    // Guard clause: Handle NULL gracefully
    if (!queue) return NIMCP_INVALID_PARAM;

    // Destroy synchronization primitives
    // WHY first: Prevents new lock/wait operations
    nimcp_mutex_destroy(&queue->mutex);
    nimcp_cond_destroy(&queue->not_empty);
    nimcp_cond_destroy(&queue->not_full);

    // Free buffer memory
    nimcp_free(queue->buffer);

    // Free queue structure
    nimcp_free(queue);

    return NIMCP_SUCCESS;
}

/**
 * @brief Enqueues an item into the queue
 *
 * WHY COPY SEMANTICS: memcpy instead of pointer
 * - Generic: Works with any data type
 * - Safe: Caller can free/reuse source memory immediately
 * - Simple: No ownership tracking needed
 *
 * BLOCKING BEHAVIOR:
 * - Non-blocking: Return immediately with NIMCP_QUEUE_FULL
 * - Blocking (timeout=0): Wait indefinitely until space available
 * - Blocking (timeout>0): Wait up to timeout_ms milliseconds
 *
 * ALGORITHM:
 * 1. Acquire mutex
 * 2. If full:
 *    - Non-blocking: Increment dropped counter, return error
 *    - Blocking: Wait on not_full condition
 * 3. Copy item to tail position
 * 4. Advance tail (with wrap-around)
 * 5. Update statistics
 * 6. Signal not_empty
 * 7. Release mutex
 *
 * WHY SIGNAL not_empty:
 * - Wakes up any thread waiting in dequeue
 * - Only one thread woken (signal, not broadcast)
 * - Efficient: Avoids thundering herd
 *
 * COMPLEXITY: O(k) where k = item_size (for memcpy)
 * THREAD SAFETY: Fully thread-safe via mutex
 *
 * @param queue Queue handle
 * @param item Pointer to item to enqueue (copied, not stored)
 * @param timeout_ms Timeout in milliseconds (0 = infinite, ignored if non-blocking)
 * @return NIMCP_SUCCESS, NIMCP_INVALID_PARAM, NIMCP_QUEUE_FULL, NIMCP_TIMEOUT
 */
nimcp_result_t nimcp_queue_enqueue(nimcp_queue_handle_t queue,
                                  const void* item,
                                  uint32_t timeout_ms) {
    // Guard clause: Validate inputs
    if (!queue || !item) return NIMCP_INVALID_PARAM;

    // Acquire exclusive access
    nimcp_mutex_lock(&queue->mutex);

    // Check if full
    if (is_queue_full(queue)) {
        // NON-BLOCKING MODE: Fail fast
        if (!queue->config.is_blocking) {
            queue->status.dropped_items++;
            nimcp_mutex_unlock(&queue->mutex);
            return NIMCP_QUEUE_FULL;
        }

        // BLOCKING MODE: Wait for space
        if (timeout_ms == 0) {
            // Infinite wait
            // WHY loop: Spurious wakeups possible, re-check condition
            while (is_queue_full(queue)) {
                nimcp_cond_wait(&queue->not_full, &queue->mutex);
            }
        } else {
            // Timed wait
            nimcp_result_t result = nimcp_cond_timedwait(&queue->not_full,
                                                        &queue->mutex,
                                                        timeout_ms);
            if (result != NIMCP_SUCCESS) {
                nimcp_mutex_unlock(&queue->mutex);
                return result;  // NIMCP_TIMEOUT or other error
            }
        }
    }

    // Copy item into circular buffer at tail position
    // WHY memcpy: Generic, works for any type
    memcpy(queue->buffer + (queue->tail * queue->config.item_size),
           item,
           queue->config.item_size);

    // Advance tail with wrap-around
    // WHY modulo: Implements circular behavior
    queue->tail = (queue->tail + 1) % queue->config.max_size;

    // Update statistics
    queue->status.current_size++;
    queue->status.total_enqueued++;

    // Track peak size for capacity planning
    if (queue->status.current_size > queue->status.peak_size) {
        queue->status.peak_size = queue->status.current_size;
    }

    // Signal waiting dequeue operations
    // WHY signal vs broadcast: Only one thread needs to wake
    nimcp_cond_signal(&queue->not_empty);

    nimcp_mutex_unlock(&queue->mutex);
    return NIMCP_SUCCESS;
}

/**
 * @brief Dequeues an item from the queue
 *
 * WHY COPY OUT: memcpy to caller's buffer
 * - Caller provides storage, we copy into it
 * - Queue owns buffer, doesn't return pointers
 * - Safe: No dangling pointer issues
 *
 * BLOCKING BEHAVIOR: Same as enqueue but waits on not_empty
 *
 * ALGORITHM:
 * 1. Acquire mutex
 * 2. If empty:
 *    - Non-blocking: Return error immediately
 *    - Blocking: Wait on not_empty condition
 * 3. Copy item from head position
 * 4. Advance head (with wrap-around)
 * 5. Update statistics
 * 6. Signal not_full
 * 7. Release mutex
 *
 * WHY SIGNAL not_full:
 * - Wakes up any thread waiting in enqueue
 * - Enables producer-consumer coordination
 *
 * COMPLEXITY: O(k) where k = item_size (for memcpy)
 * THREAD SAFETY: Fully thread-safe via mutex
 *
 * @param queue Queue handle
 * @param item Buffer to receive dequeued item (must be item_size bytes)
 * @param timeout_ms Timeout in milliseconds (0 = infinite, ignored if non-blocking)
 * @return NIMCP_SUCCESS, NIMCP_INVALID_PARAM, NIMCP_QUEUE_EMPTY, NIMCP_TIMEOUT
 */
nimcp_result_t nimcp_queue_dequeue(nimcp_queue_handle_t queue,
                                  void* item,
                                  uint32_t timeout_ms) {
    // Guard clause: Validate inputs
    if (!queue || !item) return NIMCP_INVALID_PARAM;

    // Acquire exclusive access
    nimcp_mutex_lock(&queue->mutex);

    // Check if empty
    if (is_queue_empty(queue)) {
        // NON-BLOCKING MODE: Fail fast
        if (!queue->config.is_blocking) {
            nimcp_mutex_unlock(&queue->mutex);
            return NIMCP_QUEUE_EMPTY;
        }

        // BLOCKING MODE: Wait for item
        if (timeout_ms == 0) {
            // Infinite wait
            // WHY loop: Spurious wakeups possible, re-check condition
            while (is_queue_empty(queue)) {
                nimcp_cond_wait(&queue->not_empty, &queue->mutex);
            }
        } else {
            // Timed wait
            nimcp_result_t result = nimcp_cond_timedwait(&queue->not_empty,
                                                        &queue->mutex,
                                                        timeout_ms);
            if (result != NIMCP_SUCCESS) {
                nimcp_mutex_unlock(&queue->mutex);
                return result;  // NIMCP_TIMEOUT or other error
            }
        }
    }

    // Copy item from circular buffer at head position
    memcpy(item,
           queue->buffer + (queue->head * queue->config.item_size),
           queue->config.item_size);

    // Advance head with wrap-around
    queue->head = (queue->head + 1) % queue->config.max_size;

    // Update statistics
    queue->status.current_size--;
    queue->status.total_dequeued++;

    // Signal waiting enqueue operations
    // WHY signal vs broadcast: Only one thread needs to wake
    nimcp_cond_signal(&queue->not_full);

    nimcp_mutex_unlock(&queue->mutex);
    return NIMCP_SUCCESS;
}

/**
 * @brief Peeks at front item without removing it
 *
 * WHY PEEK: Non-destructive read
 * - Allows inspection without commitment
 * - Useful for decision-making before dequeue
 * - Pattern: Peek-then-dequeue based on content
 *
 * DIFFERENCE FROM DEQUEUE:
 * - Doesn't advance head
 * - Doesn't signal not_full
 * - Always non-blocking (no wait option)
 *
 * COMPLEXITY: O(k) where k = item_size (for memcpy)
 * THREAD SAFETY: Fully thread-safe via mutex
 *
 * @param queue Queue handle
 * @param item Buffer to receive copy of front item
 * @return NIMCP_SUCCESS, NIMCP_INVALID_PARAM, NIMCP_QUEUE_EMPTY
 */
nimcp_result_t nimcp_queue_peek(nimcp_queue_handle_t queue, void* item) {
    // Guard clause: Validate inputs
    if (!queue || !item) return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(&queue->mutex);

    // Check if empty (always non-blocking)
    if (is_queue_empty(queue)) {
        nimcp_mutex_unlock(&queue->mutex);
        return NIMCP_QUEUE_EMPTY;
    }

    // Copy front item without removing
    // WHY: Head position unchanged, size unchanged
    memcpy(item,
           queue->buffer + (queue->head * queue->config.item_size),
           queue->config.item_size);

    nimcp_mutex_unlock(&queue->mutex);
    return NIMCP_SUCCESS;
}

/**
 * @brief Clears all items from queue
 *
 * WHY RESET POINTERS: Simpler than zeroing buffer
 * - Don't need to overwrite buffer (will be overwritten on next use)
 * - Just reset head = tail = 0 (empty state)
 * - Current_size = 0, but keep historical statistics
 *
 * WHY BROADCAST not_full:
 * - Multiple threads may be waiting to enqueue
 * - All can now proceed (queue is empty)
 * - Broadcast instead of signal wakes all waiters
 *
 * USE CASES:
 * - Error recovery (flush bad data)
 * - Mode changes (switch protocols)
 * - Shutdown preparation
 *
 * COMPLEXITY: O(1) - Just pointer resets
 * THREAD SAFETY: Fully thread-safe via mutex
 *
 * @param queue Queue handle
 * @return NIMCP_SUCCESS or NIMCP_INVALID_PARAM
 */
nimcp_result_t nimcp_queue_clear(nimcp_queue_handle_t queue) {
    // Guard clause: Validate input
    if (!queue) return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(&queue->mutex);

    // Reset to empty state
    // WHY: head == tail means empty
    queue->head = queue->tail = 0;
    queue->status.current_size = 0;

    // Wake all waiting enqueue operations
    // WHY broadcast: All can now proceed (queue has space)
    nimcp_cond_broadcast(&queue->not_full);

    nimcp_mutex_unlock(&queue->mutex);
    return NIMCP_SUCCESS;
}

/**
 * @brief Retrieves queue statistics
 *
 * WHY SEPARATE FUNCTION: Atomic snapshot
 * - All statistics copied under single lock
 * - Consistent view of queue state
 * - Avoids TOCTOU (time-of-check-time-of-use) issues
 *
 * STATISTICS PROVIDED:
 * - current_size: Items currently in queue
 * - peak_size: Maximum items ever in queue (capacity planning)
 * - total_enqueued: Total items ever added (throughput metric)
 * - total_dequeued: Total items ever removed (throughput metric)
 * - dropped_items: Items rejected when full (backpressure metric)
 *
 * USE CASES:
 * - Monitoring and alerting
 * - Performance tuning
 * - Capacity planning
 * - Debugging
 *
 * COMPLEXITY: O(1) - Structure copy
 * THREAD SAFETY: Fully thread-safe via mutex
 *
 * @param queue Queue handle
 * @param status Output structure for statistics
 * @return NIMCP_SUCCESS or NIMCP_INVALID_PARAM
 */
nimcp_result_t nimcp_queue_get_status(nimcp_queue_handle_t queue,
                                     nimcp_queue_status_t* status) {
    // Guard clause: Validate inputs
    if (!queue || !status) return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(&queue->mutex);

    // Atomic copy of all statistics
    // WHY: Ensures consistent snapshot
    *status = queue->status;

    nimcp_mutex_unlock(&queue->mutex);
    return NIMCP_SUCCESS;
}

/**
 * @brief Checks if queue is empty
 *
 * WHY SEPARATE FUNCTION: Convenience + thread safety
 * - Caller doesn't need to acquire lock
 * - Simple boolean result
 * - Commonly needed check
 *
 * THREAD SAFETY NOTE:
 * - Result is snapshot in time
 * - May be stale by time caller acts on it
 * - Fine for monitoring, not for synchronization
 *
 * COMPLEXITY: O(1) - Simple check
 * THREAD SAFETY: Fully thread-safe via mutex
 *
 * @param queue Queue handle
 * @return true if empty (or NULL), false otherwise
 */
bool nimcp_queue_is_empty(nimcp_queue_handle_t queue) {
    // Guard clause: NULL is "empty"
    if (!queue) return true;

    nimcp_mutex_lock(&queue->mutex);
    bool empty = is_queue_empty(queue);
    nimcp_mutex_unlock(&queue->mutex);

    return empty;
}

/**
 * @brief Checks if queue is full
 *
 * WHY SEPARATE FUNCTION: Same as is_empty rationale
 *
 * THREAD SAFETY NOTE: Same as is_empty
 *
 * COMPLEXITY: O(1) - Simple arithmetic
 * THREAD SAFETY: Fully thread-safe via mutex
 *
 * @param queue Queue handle
 * @return true if full (or NULL), false otherwise
 */
bool nimcp_queue_is_full(nimcp_queue_handle_t queue) {
    // Guard clause: NULL is "full" (can't add)
    if (!queue) return true;

    nimcp_mutex_lock(&queue->mutex);
    bool full = is_queue_full(queue);
    nimcp_mutex_unlock(&queue->mutex);

    return full;
}

/**
 * @brief Gets current number of items in queue
 *
 * WHY SEPARATE FUNCTION: Most common statistic query
 * - Lighter than full get_status call
 * - Simple size_t return
 *
 * THREAD SAFETY NOTE:
 * - Result is snapshot
 * - May change before caller uses value
 * - OK for monitoring/logging
 *
 * COMPLEXITY: O(1) - Counter read
 * THREAD SAFETY: Fully thread-safe via mutex
 *
 * @param queue Queue handle
 * @return Current size (0 if NULL)
 */
size_t nimcp_queue_get_size(nimcp_queue_handle_t queue) {
    // Guard clause: NULL has size 0
    if (!queue) return 0;

    nimcp_mutex_lock(&queue->mutex);
    size_t size = queue->status.current_size;
    nimcp_mutex_unlock(&queue->mutex);

    return size;
}
