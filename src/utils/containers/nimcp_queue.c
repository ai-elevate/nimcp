//=============================================================================
// nimcp_queue.c - Unified Queue API with Factory Pattern
//=============================================================================
// This module provides a unified interface to multiple queue implementations:
// - BLOCKING: Mutex-based (nimcp_queue_blocking.c)
// - SPSC: Lock-free single-producer single-consumer (nimcp_queue_spsc.c)
// - MPMC: Lock-free multi-producer multi-consumer (nimcp_queue_mpmc.c)
//
// The factory pattern dispatches to the appropriate implementation based on
// the queue type specified in the configuration.
//=============================================================================

#include "utils/containers/nimcp_queue.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "nimcp_queue_internal.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include "utils/memory/nimcp_memory_guards.h"  // For nimcp_calloc/nimcp_free
#include "utils/memory/nimcp_unified_memory.h"

#define LOG_MODULE "QUEUE"

//=============================================================================
// Factory Functions
//=============================================================================

nimcp_queue_config_t nimcp_queue_default_config(nimcp_queue_type_t type) {
    nimcp_queue_config_t config = {0};

    config.max_size = 1024;
    config.item_size = sizeof(void*);
    config.type = type;
    config.is_blocking = true;
    config.timeout_ms = 0;  // Infinite
    config.spin_count = NIMCP_QUEUE_DEFAULT_SPIN_COUNT;
    config.mem_manager = NULL;

    return config;
}

nimcp_result_t nimcp_queue_create(
    const nimcp_queue_config_t* config,
    nimcp_queue_handle_t* queue
) {
    if (!config || !queue) {
        LOG_ERROR(LOG_MODULE, "nimcp_queue_create: NULL config or queue pointer");
        return NIMCP_INVALID_PARAM;
    }

    if (config->max_size == 0 || config->item_size == 0) {
        LOG_ERROR(LOG_MODULE, "nimcp_queue_create: Invalid max_size=%zu or item_size=%zu",
                  config->max_size, config->item_size);
        return NIMCP_INVALID_PARAM;
    }

    LOG_DEBUG(LOG_MODULE, "Creating queue type=%s, max_size=%zu, item_size=%zu",
              nimcp_queue_type_name(config->type), config->max_size, config->item_size);

    // Allocate base queue structure
    struct nimcp_queue* q = nimcp_calloc(1, sizeof(struct nimcp_queue));
    if (!q) {
        LOG_ERROR(LOG_MODULE, "Failed to allocate queue structure");
        return NIMCP_NO_MEMORY;
    }

    // Store configuration
    q->type = config->type;
    q->config = *config;
    memset(&q->status, 0, sizeof(nimcp_queue_status_t));
    q->status.type = config->type;
    q->status.capacity = config->max_size;

    // Dispatch to type-specific creation
    nimcp_result_t result;
    switch (config->type) {
        case NIMCP_QUEUE_TYPE_BLOCKING:
            result = nimcp_queue_blocking_create(q, config);
            break;
        case NIMCP_QUEUE_TYPE_SPSC:
            result = nimcp_queue_spsc_create(q, config);
            break;
        case NIMCP_QUEUE_TYPE_MPMC:
            result = nimcp_queue_mpmc_create(q, config);
            break;
        default:
            LOG_ERROR(LOG_MODULE, "Unknown queue type: %d", config->type);
            nimcp_free(q);
            return NIMCP_INVALID_PARAM;
    }

    if (result != NIMCP_SUCCESS) {
        LOG_ERROR(LOG_MODULE, "Failed to create %s queue: %d",
                  nimcp_queue_type_name(config->type), result);
        nimcp_free(q);
        return result;
    }

    *queue = q;
    LOG_INFO(LOG_MODULE, "Queue created successfully (type=%s, capacity=%zu)",
             nimcp_queue_type_name(config->type), q->capacity);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_queue_destroy(nimcp_queue_handle_t queue) {
    if (!queue) {
        return NIMCP_SUCCESS;  // NULL is OK
    }

    LOG_DEBUG(LOG_MODULE, "Destroying queue type=%s", nimcp_queue_type_name(queue->type));

    // Dispatch to type-specific destruction
    switch (queue->type) {
        case NIMCP_QUEUE_TYPE_BLOCKING:
            nimcp_queue_blocking_destroy(queue);
            break;
        case NIMCP_QUEUE_TYPE_SPSC:
            nimcp_queue_spsc_destroy(queue);
            break;
        case NIMCP_QUEUE_TYPE_MPMC:
            nimcp_queue_mpmc_destroy(queue);
            break;
        default:
            LOG_WARN(LOG_MODULE, "Unknown queue type during destroy: %d", queue->type);
            break;
    }

    nimcp_free(queue);
    return NIMCP_SUCCESS;
}

//=============================================================================
// Single Item Operations (Dispatch)
//=============================================================================

nimcp_result_t nimcp_queue_enqueue(
    nimcp_queue_handle_t queue,
    const void* item,
    uint32_t timeout_ms
) {
    if (!queue || !item) {
        return NIMCP_INVALID_PARAM;
    }

    switch (queue->type) {
        case NIMCP_QUEUE_TYPE_BLOCKING:
            return nimcp_queue_blocking_enqueue(queue, item, timeout_ms);
        case NIMCP_QUEUE_TYPE_SPSC:
            return nimcp_queue_spsc_enqueue(queue, item, timeout_ms);
        case NIMCP_QUEUE_TYPE_MPMC:
            return nimcp_queue_mpmc_enqueue(queue, item, timeout_ms);
        default:
            return NIMCP_INVALID_PARAM;
    }
}

nimcp_result_t nimcp_queue_dequeue(
    nimcp_queue_handle_t queue,
    void* item,
    uint32_t timeout_ms
) {
    if (!queue || !item) {
        return NIMCP_INVALID_PARAM;
    }

    switch (queue->type) {
        case NIMCP_QUEUE_TYPE_BLOCKING:
            return nimcp_queue_blocking_dequeue(queue, item, timeout_ms);
        case NIMCP_QUEUE_TYPE_SPSC:
            return nimcp_queue_spsc_dequeue(queue, item, timeout_ms);
        case NIMCP_QUEUE_TYPE_MPMC:
            return nimcp_queue_mpmc_dequeue(queue, item, timeout_ms);
        default:
            return NIMCP_INVALID_PARAM;
    }
}

bool nimcp_queue_try_enqueue(nimcp_queue_handle_t queue, const void* item) {
    return nimcp_queue_enqueue(queue, item, 0) == NIMCP_SUCCESS;
}

bool nimcp_queue_try_dequeue(nimcp_queue_handle_t queue, void* item) {
    return nimcp_queue_dequeue(queue, item, 0) == NIMCP_SUCCESS;
}

//=============================================================================
// Batch Operations
//=============================================================================

nimcp_result_t nimcp_queue_enqueue_batch(
    nimcp_queue_handle_t queue,
    const void* items,
    size_t count,
    size_t* enqueued_count,
    uint32_t timeout_ms
) {
    if (!queue || !items || count == 0) {
        if (enqueued_count) *enqueued_count = 0;
        return NIMCP_INVALID_PARAM;
    }

    size_t enqueued = 0;
    const uint8_t* data = (const uint8_t*)items;

    for (size_t i = 0; i < count; i++) {
        // Use timeout only for first item, non-blocking for rest
        uint32_t item_timeout = (i == 0) ? timeout_ms : 0;
        nimcp_result_t r = nimcp_queue_enqueue(queue, data + (i * queue->config.item_size), item_timeout);
        if (r != NIMCP_SUCCESS) {
            break;
        }
        enqueued++;
    }

    if (enqueued_count) *enqueued_count = enqueued;
    queue->status.batch_enqueue_ops++;
    queue->status.batch_items_enqueued += enqueued;

    return (enqueued == count) ? NIMCP_SUCCESS :
           (enqueued > 0) ? NIMCP_SUCCESS_PARTIAL : NIMCP_QUEUE_FULL;
}

nimcp_result_t nimcp_queue_dequeue_batch(
    nimcp_queue_handle_t queue,
    void* items,
    size_t max_count,
    size_t* dequeued_count,
    uint32_t timeout_ms
) {
    if (!queue || !items || max_count == 0) {
        if (dequeued_count) *dequeued_count = 0;
        return NIMCP_INVALID_PARAM;
    }

    size_t dequeued = 0;
    uint8_t* data = (uint8_t*)items;

    for (size_t i = 0; i < max_count; i++) {
        uint32_t item_timeout = (i == 0) ? timeout_ms : 0;
        nimcp_result_t r = nimcp_queue_dequeue(queue, data + (i * queue->config.item_size), item_timeout);
        if (r != NIMCP_SUCCESS) {
            break;
        }
        dequeued++;
    }

    if (dequeued_count) *dequeued_count = dequeued;
    queue->status.batch_dequeue_ops++;
    queue->status.batch_items_dequeued += dequeued;

    return (dequeued > 0) ? NIMCP_SUCCESS : NIMCP_QUEUE_EMPTY;
}

//=============================================================================
// State Queries (Dispatch)
//=============================================================================

nimcp_result_t nimcp_queue_peek(nimcp_queue_handle_t queue, void* item) {
    if (!queue || !item) {
        return NIMCP_INVALID_PARAM;
    }

    switch (queue->type) {
        case NIMCP_QUEUE_TYPE_BLOCKING:
            return nimcp_queue_blocking_peek(queue, item);
        case NIMCP_QUEUE_TYPE_SPSC:
            return nimcp_queue_spsc_peek(queue, item);
        case NIMCP_QUEUE_TYPE_MPMC:
            return nimcp_queue_mpmc_peek(queue, item);
        default:
            return NIMCP_INVALID_PARAM;
    }
}

nimcp_result_t nimcp_queue_clear(nimcp_queue_handle_t queue) {
    if (!queue) {
        return NIMCP_INVALID_PARAM;
    }

    switch (queue->type) {
        case NIMCP_QUEUE_TYPE_BLOCKING:
            return nimcp_queue_blocking_clear(queue);
        case NIMCP_QUEUE_TYPE_SPSC:
            return nimcp_queue_spsc_clear(queue);
        case NIMCP_QUEUE_TYPE_MPMC:
            return nimcp_queue_mpmc_clear(queue);
        default:
            return NIMCP_INVALID_PARAM;
    }
}

bool nimcp_queue_is_empty(nimcp_queue_handle_t queue) {
    if (!queue) return true;

    switch (queue->type) {
        case NIMCP_QUEUE_TYPE_BLOCKING:
            return nimcp_queue_blocking_is_empty(queue);
        case NIMCP_QUEUE_TYPE_SPSC:
            return nimcp_queue_spsc_is_empty(queue);
        case NIMCP_QUEUE_TYPE_MPMC:
            return nimcp_queue_mpmc_is_empty(queue);
        default:
            return true;
    }
}

bool nimcp_queue_is_full(nimcp_queue_handle_t queue) {
    if (!queue) return true;

    switch (queue->type) {
        case NIMCP_QUEUE_TYPE_BLOCKING:
            return nimcp_queue_blocking_is_full(queue);
        case NIMCP_QUEUE_TYPE_SPSC:
            return nimcp_queue_spsc_is_full(queue);
        case NIMCP_QUEUE_TYPE_MPMC:
            return nimcp_queue_mpmc_is_full(queue);
        default:
            return true;
    }
}

size_t nimcp_queue_get_size(nimcp_queue_handle_t queue) {
    if (!queue) return 0;

    switch (queue->type) {
        case NIMCP_QUEUE_TYPE_BLOCKING:
            return nimcp_queue_blocking_get_size(queue);
        case NIMCP_QUEUE_TYPE_SPSC:
            return nimcp_queue_spsc_get_size(queue);
        case NIMCP_QUEUE_TYPE_MPMC:
            return nimcp_queue_mpmc_get_size(queue);
        default:
            return 0;
    }
}

size_t nimcp_queue_get_capacity(nimcp_queue_handle_t queue) {
    if (!queue) return 0;
    // MPMC/SPSC use power of 2 internally, return full capacity
    // BLOCKING uses requested size directly
    return queue->capacity;
}

size_t nimcp_queue_get_available(nimcp_queue_handle_t queue) {
    if (!queue) return 0;
    size_t cap = nimcp_queue_get_capacity(queue);
    size_t size = nimcp_queue_get_size(queue);
    return (cap > size) ? (cap - size) : 0;
}

nimcp_queue_type_t nimcp_queue_get_type(nimcp_queue_handle_t queue) {
    if (!queue) return NIMCP_QUEUE_TYPE_BLOCKING;
    return queue->type;
}

//=============================================================================
// Statistics and Monitoring
//=============================================================================

nimcp_result_t nimcp_queue_get_status(
    nimcp_queue_handle_t queue,
    nimcp_queue_status_t* status
) {
    if (!queue || !status) {
        return NIMCP_INVALID_PARAM;
    }

    *status = queue->status;
    status->current_size = nimcp_queue_get_size(queue);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_queue_reset_stats(nimcp_queue_handle_t queue) {
    if (!queue) {
        return NIMCP_INVALID_PARAM;
    }

    // Keep current_size and capacity, reset everything else
    size_t current = queue->status.current_size;
    size_t capacity = queue->status.capacity;
    nimcp_queue_type_t type = queue->status.type;

    memset(&queue->status, 0, sizeof(nimcp_queue_status_t));
    queue->status.current_size = current;
    queue->status.capacity = capacity;
    queue->status.type = type;

    return NIMCP_SUCCESS;
}

float nimcp_queue_get_utilization(nimcp_queue_handle_t queue) {
    if (!queue) return 0.0f;

    size_t cap = nimcp_queue_get_capacity(queue);
    if (cap == 0) return 0.0f;

    size_t size = nimcp_queue_get_size(queue);
    return (float)size / (float)cap * 100.0f;
}

//=============================================================================
// Runtime Configuration
//=============================================================================

nimcp_result_t nimcp_queue_set_spin_count(
    nimcp_queue_handle_t queue,
    uint32_t spin_count
) {
    if (!queue) {
        return NIMCP_INVALID_PARAM;
    }

    // Only MPMC uses spin count currently
    // Implementation would update the impl_data
    (void)spin_count;
    return NIMCP_SUCCESS;
}

uint32_t nimcp_queue_get_spin_count(nimcp_queue_handle_t queue) {
    if (!queue) return 0;
    return queue->config.spin_count;
}
