//=============================================================================
// nimcp_event_queue.c - Priority Event Queue Implementation
//=============================================================================

#include "middleware/events/nimcp_event_queue.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/time/nimcp_time.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_memory_pool.h"
#include <string.h>
#include <stdio.h>

//=============================================================================
// Internal Structure
//=============================================================================

/**
 * @brief Heap entry with timestamp for FIFO ordering
 */
typedef struct {
    event_t event;              /**< The event */
    uint64_t enqueue_time_us;   /**< When enqueued (for FIFO within priority) */
    bool used_pool;             /**< Whether payload came from pool (Phase 1.5) */
} heap_entry_t;

/**
 * @brief Event queue internal structure
 */
struct event_queue_struct {
    // Heap storage (min-heap by priority)
    heap_entry_t* heap;         /**< Heap array */
    uint32_t capacity;          /**< Maximum capacity */
    uint32_t size;              /**< Current size */

    // Configuration
    overflow_policy_t overflow_policy;
    bool enable_coalescing;
    uint64_t block_timeout_us;
    uint32_t max_payload_size;  /**< Max size for pooled payloads */

    // Memory pool for event payloads (Phase 1.5)
    memory_pool_t payload_pool;

    // Statistics
    uint64_t total_enqueued;
    uint64_t total_dequeued;
    uint64_t total_dropped;
    uint64_t total_coalesced;
    uint32_t peak_size;
    uint64_t total_wait_time_us;

    // Thread safety
    nimcp_platform_mutex_t mutex;
};

//=============================================================================
// Error Handling
//=============================================================================

static __thread char last_error[256] = {0};

static void set_error(const char* msg) {
    snprintf(last_error, sizeof(last_error), "%s", msg);
}

const char* event_queue_get_last_error(void) {
    return last_error;
}

//=============================================================================
// Pool-Aware Event Helpers (Phase 1.5)
//=============================================================================

/**
 * WHAT: Copy event with pool-aware payload allocation
 * WHY:  1.13x faster allocation for small payloads
 * HOW:  Try pool first (if size fits), fallback to malloc
 *
 * @param dest Destination event
 * @param src Source event
 * @param pool Memory pool for payloads
 * @param max_pool_size Maximum size for pool allocation
 * @param used_pool Output: true if pool was used
 * @return true on success
 */
static bool event_copy_pooled(event_t* dest, const event_t* src,
                                memory_pool_t pool, uint32_t max_pool_size,
                                bool* used_pool) {
    if (!dest || !src || !used_pool) return false;

    // Copy basic structure
    *dest = *src;
    *used_pool = false;

    // For types without dynamic allocations, just return
    if (src->type != EVENT_TYPE_SPIKE_BURST &&
        src->type != EVENT_TYPE_MEMORY_FORMED &&
        src->type != EVENT_TYPE_DECISION_MADE &&
        src->type != EVENT_TYPE_CUSTOM) {
        return true;
    }

    // Calculate payload size
    size_t payload_size = 0;
    void** dest_ptr = NULL;
    const void* src_ptr = NULL;

    switch (src->type) {
        case EVENT_TYPE_SPIKE_BURST:
            if (src->data.spike_burst.neuron_ids && src->data.spike_burst.num_neurons > 0) {
                payload_size = src->data.spike_burst.num_neurons * sizeof(uint32_t);
                dest_ptr = (void**)&dest->data.spike_burst.neuron_ids;
                src_ptr = src->data.spike_burst.neuron_ids;
            }
            break;

        case EVENT_TYPE_MEMORY_FORMED:
            if (src->data.memory_formed.memory_trace && src->data.memory_formed.trace_size > 0) {
                payload_size = src->data.memory_formed.trace_size * sizeof(float);
                dest_ptr = (void**)&dest->data.memory_formed.memory_trace;
                src_ptr = src->data.memory_formed.memory_trace;
            }
            break;

        case EVENT_TYPE_DECISION_MADE:
            if (src->data.decision_made.decision_vector && src->data.decision_made.vector_size > 0) {
                payload_size = src->data.decision_made.vector_size * sizeof(float);
                dest_ptr = (void**)&dest->data.decision_made.decision_vector;
                src_ptr = src->data.decision_made.decision_vector;
            }
            break;

        case EVENT_TYPE_CUSTOM:
            if (src->data.custom.data && src->data.custom.data_size > 0) {
                payload_size = src->data.custom.data_size;
                dest_ptr = (void**)&dest->data.custom.data;
                src_ptr = src->data.custom.data;
            }
            break;

        default:
            break;
    }

    // No payload to allocate
    if (payload_size == 0 || !dest_ptr || !src_ptr) {
        return true;
    }

    // Try pool allocation first if size fits
    void* allocated = NULL;
    if (payload_size <= max_pool_size && pool) {
        allocated = memory_pool_acquire(pool);
        if (allocated) {
            *used_pool = true;
        }
    }

    // Fallback to malloc if pool failed or size too large
    if (!allocated) {
        allocated = nimcp_malloc(payload_size);
        if (!allocated) {
            return false;
        }
        *used_pool = false;
    }

    // Copy payload data
    memcpy(allocated, src_ptr, payload_size);
    *dest_ptr = allocated;

    return true;
}

/**
 * WHAT: Free event with pool-aware payload deallocation
 * WHY:  Return memory to pool instead of system heap
 * HOW:  Release to pool if used_pool, otherwise nimcp_free()
 *
 * @param event Event to free
 * @param pool Memory pool
 * @param used_pool Whether payload came from pool
 */
static void event_free_pooled(event_t* event, memory_pool_t pool, bool used_pool) {
    if (!event) return;

    // Get payload pointer based on type
    void* payload = NULL;
    switch (event->type) {
        case EVENT_TYPE_SPIKE_BURST:
            payload = event->data.spike_burst.neuron_ids;
            event->data.spike_burst.neuron_ids = NULL;
            break;

        case EVENT_TYPE_MEMORY_FORMED:
            payload = event->data.memory_formed.memory_trace;
            event->data.memory_formed.memory_trace = NULL;
            break;

        case EVENT_TYPE_DECISION_MADE:
            payload = event->data.decision_made.decision_vector;
            event->data.decision_made.decision_vector = NULL;
            break;

        case EVENT_TYPE_CUSTOM:
            payload = event->data.custom.data;
            event->data.custom.data = NULL;
            break;

        default:
            break;
    }

    // Free payload
    if (payload) {
        if (used_pool && pool) {
            memory_pool_release(pool, payload);
        } else {
            nimcp_free(payload);
        }
    }
}

//=============================================================================
// Min-Heap Operations
//=============================================================================

/**
 * WHAT: Compare two heap entries (priority first, then FIFO)
 * WHY:  Maintain min-heap invariant with FIFO within same priority
 * HOW:  Compare priority, break ties with timestamp
 * RETURNS: true if a should be higher in heap than b
 */
static bool heap_less_than(const heap_entry_t* a, const heap_entry_t* b) {
    if (a->event.priority != b->event.priority) {
        return a->event.priority < b->event.priority; // Lower priority number = higher priority
    }
    return a->enqueue_time_us < b->enqueue_time_us; // Earlier = higher priority (FIFO)
}

/**
 * WHAT: Swap two heap entries
 */
static void heap_swap(heap_entry_t* a, heap_entry_t* b) {
    heap_entry_t temp = *a;
    *a = *b;
    *b = temp;
}

/**
 * WHAT: Bubble element up to restore heap property
 * COMPLEXITY: O(log n)
 */
static void heap_bubble_up(heap_entry_t* heap, uint32_t index) {
    while (index > 0) {
        uint32_t parent = (index - 1) / 2;
        if (heap_less_than(&heap[index], &heap[parent])) {
            heap_swap(&heap[index], &heap[parent]);
            index = parent;
        } else {
            break;
        }
    }
}

/**
 * WHAT: Bubble element down to restore heap property
 * COMPLEXITY: O(log n)
 */
static void heap_bubble_down(heap_entry_t* heap, uint32_t size, uint32_t index) {
    while (true) {
        uint32_t left = 2 * index + 1;
        uint32_t right = 2 * index + 2;
        uint32_t smallest = index;

        if (left < size && heap_less_than(&heap[left], &heap[smallest])) {
            smallest = left;
        }
        if (right < size && heap_less_than(&heap[right], &heap[smallest])) {
            smallest = right;
        }

        if (smallest != index) {
            heap_swap(&heap[index], &heap[smallest]);
            index = smallest;
        } else {
            break;
        }
    }
}

//=============================================================================
// Queue Lifecycle
//=============================================================================

event_queue_config_t event_queue_default_config(void) {
    event_queue_config_t config = {0};
    config.capacity = 1024;
    config.overflow_policy = OVERFLOW_POLICY_DROP_OLDEST;
    config.enable_coalescing = false;
    config.block_timeout_us = 0; // No timeout
    config.max_payload_size = 256; // 256 bytes (Phase 1.5)
    return config;
}

event_queue_t event_queue_create(const event_queue_config_t* config) {
    // Use defaults if config is NULL
    event_queue_config_t cfg = config ? *config : event_queue_default_config();

    // Validate configuration
    if (cfg.capacity == 0 || cfg.capacity > 1000000) {
        set_error("Invalid capacity");
        return NULL;
    }

    // Allocate queue structure
    event_queue_t queue = nimcp_calloc(1, sizeof(struct event_queue_struct));
    if (!queue) {
        set_error("Failed to allocate queue");
        return NULL;
    }

    // Allocate heap
    queue->heap = nimcp_calloc(cfg.capacity, sizeof(heap_entry_t));
    if (!queue->heap) {
        nimcp_free(queue);
        set_error("Failed to allocate heap");
        return NULL;
    }

    // Initialize
    queue->capacity = cfg.capacity;
    queue->size = 0;
    queue->overflow_policy = cfg.overflow_policy;
    queue->enable_coalescing = cfg.enable_coalescing;
    queue->block_timeout_us = cfg.block_timeout_us;
    queue->max_payload_size = cfg.max_payload_size;

    // Create payload memory pool (Phase 1.5)
    memory_pool_config_t pool_config = {
        .block_size = cfg.max_payload_size,
        .num_blocks = cfg.capacity * 2,  // Double-buffer for enqueue/dequeue
        .alignment = 16,
        .enable_tracking = false,
        .enable_guard_pages = false
    };
    queue->payload_pool = memory_pool_create(&pool_config);
    if (!queue->payload_pool) {
        nimcp_free(queue->heap);
        nimcp_free(queue);
        set_error("Failed to create payload pool");
        return NULL;
    }

    // Create mutex
    if (nimcp_platform_mutex_init(&queue->mutex, false) != 0) {
        memory_pool_destroy(queue->payload_pool);
        nimcp_free(queue->heap);
        nimcp_free(queue);
        set_error("Failed to create mutex");
        return NULL;
    }

    return queue;
}

void event_queue_destroy(event_queue_t queue) {
    if (!queue) return;

    nimcp_platform_mutex_lock(&queue->mutex);

    // Free all events in queue (Phase 1.5: use pool-aware free)
    for (uint32_t i = 0; i < queue->size; i++) {
        event_free_pooled(&queue->heap[i].event, queue->payload_pool, queue->heap[i].used_pool);
    }

    nimcp_free(queue->heap);

    // Destroy payload memory pool (Phase 1.5)
    memory_pool_destroy(queue->payload_pool);

    nimcp_platform_mutex_unlock(&queue->mutex);
    nimcp_platform_mutex_destroy(&queue->mutex);

    nimcp_free(queue);
}

//=============================================================================
// Queue Operations
//=============================================================================

bool event_queue_enqueue(event_queue_t queue, const event_t* event) {
    if (!queue || !event) {
        set_error("Invalid parameters");
        return false;
    }

    nimcp_platform_mutex_lock(&queue->mutex);

    uint64_t enqueue_time = nimcp_time_get_us();
    bool success = true;

    // Handle full queue
    if (queue->size >= queue->capacity) {
        switch (queue->overflow_policy) {
            case OVERFLOW_POLICY_DROP_NEWEST:
                // Drop incoming event
                queue->total_dropped++;
                success = false;
                goto unlock;

            case OVERFLOW_POLICY_DROP_OLDEST:
                // Remove root (oldest high-priority event) - Phase 1.5: use pool-aware free
                event_free_pooled(&queue->heap[0].event, queue->payload_pool, queue->heap[0].used_pool);
                queue->heap[0] = queue->heap[queue->size - 1];
                queue->size--;
                heap_bubble_down(queue->heap, queue->size, 0);
                queue->total_dropped++;
                break;

            case OVERFLOW_POLICY_DROP_LOWEST:
                // Find and remove lowest priority event
                {
                    uint32_t lowest_idx = 0;
                    for (uint32_t i = 1; i < queue->size; i++) {
                        if (queue->heap[i].event.priority > queue->heap[lowest_idx].event.priority) {
                            lowest_idx = i;
                        }
                    }
                    // Phase 1.5: use pool-aware free
                    event_free_pooled(&queue->heap[lowest_idx].event, queue->payload_pool,
                                      queue->heap[lowest_idx].used_pool);
                    queue->heap[lowest_idx] = queue->heap[queue->size - 1];
                    queue->size--;
                    heap_bubble_up(queue->heap, lowest_idx);
                    heap_bubble_down(queue->heap, queue->size, lowest_idx);
                    queue->total_dropped++;
                }
                break;

            case OVERFLOW_POLICY_BLOCK:
                // For now, just drop (full blocking would need condition variable)
                queue->total_dropped++;
                success = false;
                goto unlock;
        }
    }

    // Add event to heap (Phase 1.5: use pool-aware copy)
    heap_entry_t entry;
    if (!event_copy_pooled(&entry.event, event, queue->payload_pool,
                           queue->max_payload_size, &entry.used_pool)) {
        set_error("Failed to copy event");
        success = false;
        goto unlock;
    }
    entry.enqueue_time_us = enqueue_time;

    queue->heap[queue->size] = entry;
    heap_bubble_up(queue->heap, queue->size);
    queue->size++;

    // Update statistics
    queue->total_enqueued++;
    if (queue->size > queue->peak_size) {
        queue->peak_size = queue->size;
    }

unlock:
    nimcp_platform_mutex_unlock(&queue->mutex);
    return success;
}

bool event_queue_dequeue(event_queue_t queue, event_t* event) {
    if (!queue || !event) {
        set_error("Invalid parameters");
        return false;
    }

    nimcp_platform_mutex_lock(&queue->mutex);

    if (queue->size == 0) {
        nimcp_platform_mutex_unlock(&queue->mutex);
        return false;
    }

    // Copy root to output
    *event = queue->heap[0].event;

    // Calculate wait time
    uint64_t now = nimcp_time_get_us();
    uint64_t wait_time = now - queue->heap[0].enqueue_time_us;
    queue->total_wait_time_us += wait_time;

    // Move last element to root and restore heap
    queue->heap[0] = queue->heap[queue->size - 1];
    queue->size--;
    if (queue->size > 0) {
        heap_bubble_down(queue->heap, queue->size, 0);
    }

    queue->total_dequeued++;

    nimcp_platform_mutex_unlock(&queue->mutex);
    return true;
}

bool event_queue_peek(event_queue_t queue, event_t* event) {
    if (!queue || !event) return false;

    nimcp_platform_mutex_lock(&queue->mutex);

    bool result = false;
    if (queue->size > 0) {
        *event = queue->heap[0].event;
        result = true;
    }

    nimcp_platform_mutex_unlock(&queue->mutex);
    return result;
}

uint32_t event_queue_dequeue_batch(event_queue_t queue, event_t* events,
                                    uint32_t max_events) {
    if (!queue || !events || max_events == 0) return 0;

    nimcp_platform_mutex_lock(&queue->mutex);

    uint32_t dequeued = 0;
    while (dequeued < max_events && queue->size > 0) {
        events[dequeued] = queue->heap[0].event;

        uint64_t now = nimcp_time_get_us();
        uint64_t wait_time = now - queue->heap[0].enqueue_time_us;
        queue->total_wait_time_us += wait_time;

        queue->heap[0] = queue->heap[queue->size - 1];
        queue->size--;
        if (queue->size > 0) {
            heap_bubble_down(queue->heap, queue->size, 0);
        }

        queue->total_dequeued++;
        dequeued++;
    }

    nimcp_platform_mutex_unlock(&queue->mutex);
    return dequeued;
}

uint32_t event_queue_size(event_queue_t queue) {
    if (!queue) return 0;

    nimcp_platform_mutex_lock(&queue->mutex);
    uint32_t size = queue->size;
    nimcp_platform_mutex_unlock(&queue->mutex);

    return size;
}

bool event_queue_is_empty(event_queue_t queue) {
    return event_queue_size(queue) == 0;
}

bool event_queue_is_full(event_queue_t queue) {
    if (!queue) return true;

    nimcp_platform_mutex_lock(&queue->mutex);
    bool full = (queue->size >= queue->capacity);
    nimcp_platform_mutex_unlock(&queue->mutex);

    return full;
}

void event_queue_clear(event_queue_t queue) {
    if (!queue) return;

    nimcp_platform_mutex_lock(&queue->mutex);

    // Phase 1.5: use pool-aware free
    for (uint32_t i = 0; i < queue->size; i++) {
        event_free_pooled(&queue->heap[i].event, queue->payload_pool, queue->heap[i].used_pool);
    }
    queue->size = 0;

    nimcp_platform_mutex_unlock(&queue->mutex);
}

//=============================================================================
// Advanced Operations
//=============================================================================

uint32_t event_queue_remove_if(event_queue_t queue, event_filter_fn filter, void* context) {
    if (!queue || !filter) return 0;

    nimcp_platform_mutex_lock(&queue->mutex);

    uint32_t removed = 0;
    uint32_t i = 0;

    while (i < queue->size) {
        if (filter(&queue->heap[i].event, context)) {
            // Remove this event (Phase 1.5: use pool-aware free)
            event_free_pooled(&queue->heap[i].event, queue->payload_pool, queue->heap[i].used_pool);
            queue->heap[i] = queue->heap[queue->size - 1];
            queue->size--;
            removed++;

            // Restore heap property
            if (i < queue->size) {
                heap_bubble_up(queue->heap, i);
                heap_bubble_down(queue->heap, queue->size, i);
            }
        } else {
            i++;
        }
    }

    nimcp_platform_mutex_unlock(&queue->mutex);
    return removed;
}

uint32_t event_queue_count_if(event_queue_t queue, event_filter_fn filter, void* context) {
    if (!queue || !filter) return 0;

    nimcp_platform_mutex_lock(&queue->mutex);

    uint32_t count = 0;
    for (uint32_t i = 0; i < queue->size; i++) {
        if (filter(&queue->heap[i].event, context)) {
            count++;
        }
    }

    nimcp_platform_mutex_unlock(&queue->mutex);
    return count;
}

//=============================================================================
// Statistics
//=============================================================================

bool event_queue_get_stats(event_queue_t queue, event_queue_stats_t* stats) {
    if (!queue || !stats) return false;

    nimcp_platform_mutex_lock(&queue->mutex);

    stats->total_enqueued = queue->total_enqueued;
    stats->total_dequeued = queue->total_dequeued;
    stats->total_dropped = queue->total_dropped;
    stats->total_coalesced = queue->total_coalesced;
    stats->current_size = queue->size;
    stats->peak_size = queue->peak_size;

    if (queue->total_dequeued > 0) {
        stats->avg_wait_time_us = (float)queue->total_wait_time_us / queue->total_dequeued;
    } else {
        stats->avg_wait_time_us = 0.0f;
    }

    nimcp_platform_mutex_unlock(&queue->mutex);
    return true;
}

void event_queue_reset_stats(event_queue_t queue) {
    if (!queue) return;

    nimcp_platform_mutex_lock(&queue->mutex);

    queue->total_enqueued = 0;
    queue->total_dequeued = 0;
    queue->total_dropped = 0;
    queue->total_coalesced = 0;
    queue->peak_size = queue->size;
    queue->total_wait_time_us = 0;

    nimcp_platform_mutex_unlock(&queue->mutex);
}
