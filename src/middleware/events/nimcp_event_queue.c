//=============================================================================
// nimcp_event_queue.c - Priority Event Queue Implementation
//=============================================================================

#include "middleware/events/nimcp_event_queue.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/time/nimcp_time.h"
#include "utils/memory/nimcp_memory.h"
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

    // Create mutex
    if (nimcp_platform_mutex_init(&queue->mutex, false) != 0) {
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

    // Free all events in queue
    for (uint32_t i = 0; i < queue->size; i++) {
        event_free(&queue->heap[i].event);
    }

    nimcp_free(queue->heap);

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
                // Remove root (oldest high-priority event)
                event_free(&queue->heap[0].event);
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
                    event_free(&queue->heap[lowest_idx].event);
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

    // Add event to heap
    heap_entry_t entry;
    if (!event_copy(&entry.event, event)) {
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

    for (uint32_t i = 0; i < queue->size; i++) {
        event_free(&queue->heap[i].event);
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
            // Remove this event
            event_free(&queue->heap[i].event);
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
