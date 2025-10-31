#include "utils/nimcp_queue.h"
#include "utils/nimcp_thread.h"
#include "utils/nimcp_memory.h"
#include <string.h>

struct nimcp_queue {
    uint8_t* buffer;              // Circular buffer
    size_t head;                  // Index for dequeue
    size_t tail;                  // Index for enqueue
    nimcp_queue_config_t config;  // Queue configuration
    nimcp_queue_status_t status;  // Queue status
    nimcp_mutex_t mutex;          // Queue mutex
    nimcp_cond_t not_empty;       // Condition for items available
    nimcp_cond_t not_full;        // Condition for space available
};

nimcp_result_t nimcp_queue_create(const nimcp_queue_config_t* config,
                                 nimcp_queue_handle_t* queue) {
    if (!config || !queue) return NIMCP_INVALID_PARAM;
    if (config->max_size == 0 || config->item_size == 0) return NIMCP_INVALID_PARAM;

    struct nimcp_queue* q = nimcp_calloc(1, sizeof(struct nimcp_queue));
    if (!q) return NIMCP_NO_MEMORY;

    q->buffer = nimcp_malloc(config->max_size * config->item_size);
    if (!q->buffer) {
        nimcp_free(q);
        return NIMCP_NO_MEMORY;
    }

    q->config = *config;
    q->head = q->tail = 0;
    memset(&q->status, 0, sizeof(nimcp_queue_status_t));

    if (nimcp_mutex_init(&q->mutex) != NIMCP_SUCCESS ||
        nimcp_cond_init(&q->not_empty) != NIMCP_SUCCESS ||
        nimcp_cond_init(&q->not_full) != NIMCP_SUCCESS) {
        nimcp_free(q->buffer);
        nimcp_free(q);
        return NIMCP_INIT_FAILED;
    }

    *queue = q;
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_queue_destroy(nimcp_queue_handle_t queue) {
    if (!queue) return NIMCP_INVALID_PARAM;

    nimcp_mutex_destroy(&queue->mutex);
    nimcp_cond_destroy(&queue->not_empty);
    nimcp_cond_destroy(&queue->not_full);
    nimcp_free(queue->buffer);
    nimcp_free(queue);
    return NIMCP_SUCCESS;
}

static bool is_queue_full(struct nimcp_queue* q) {
    return ((q->tail + 1) % q->config.max_size) == q->head;
}

static bool is_queue_empty(struct nimcp_queue* q) {
    return q->head == q->tail;
}

nimcp_result_t nimcp_queue_enqueue(nimcp_queue_handle_t queue,
                                  const void* item,
                                  uint32_t timeout_ms) {
    if (!queue || !item) return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(&queue->mutex);

    if (is_queue_full(queue)) {
        if (!queue->config.is_blocking) {
            queue->status.dropped_items++;
            nimcp_mutex_unlock(&queue->mutex);
            return NIMCP_QUEUE_FULL;
        }

        if (timeout_ms == 0) {
            while (is_queue_full(queue)) {
                nimcp_cond_wait(&queue->not_full, &queue->mutex);
            }
        } else {
            nimcp_result_t result = nimcp_cond_timedwait(&queue->not_full,
                                                        &queue->mutex,
                                                        timeout_ms);
            if (result != NIMCP_SUCCESS) {
                nimcp_mutex_unlock(&queue->mutex);
                return result;
            }
        }
    }

    memcpy(queue->buffer + (queue->tail * queue->config.item_size),
           item,
           queue->config.item_size);
    
    queue->tail = (queue->tail + 1) % queue->config.max_size;
    queue->status.current_size++;
    queue->status.total_enqueued++;
    
    if (queue->status.current_size > queue->status.peak_size) {
        queue->status.peak_size = queue->status.current_size;
    }

    nimcp_cond_signal(&queue->not_empty);
    nimcp_mutex_unlock(&queue->mutex);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_queue_dequeue(nimcp_queue_handle_t queue,
                                  void* item,
                                  uint32_t timeout_ms) {
    if (!queue || !item) return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(&queue->mutex);

    if (is_queue_empty(queue)) {
        if (!queue->config.is_blocking) {
            nimcp_mutex_unlock(&queue->mutex);
            return NIMCP_QUEUE_EMPTY;
        }

        if (timeout_ms == 0) {
            while (is_queue_empty(queue)) {
                nimcp_cond_wait(&queue->not_empty, &queue->mutex);
            }
        } else {
            nimcp_result_t result = nimcp_cond_timedwait(&queue->not_empty,
                                                        &queue->mutex,
                                                        timeout_ms);
            if (result != NIMCP_SUCCESS) {
                nimcp_mutex_unlock(&queue->mutex);
                return result;
            }
        }
    }

    memcpy(item,
           queue->buffer + (queue->head * queue->config.item_size),
           queue->config.item_size);
    
    queue->head = (queue->head + 1) % queue->config.max_size;
    queue->status.current_size--;
    queue->status.total_dequeued++;

    nimcp_cond_signal(&queue->not_full);
    nimcp_mutex_unlock(&queue->mutex);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_queue_peek(nimcp_queue_handle_t queue, void* item) {
    if (!queue || !item) return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(&queue->mutex);
    
    if (is_queue_empty(queue)) {
        nimcp_mutex_unlock(&queue->mutex);
        return NIMCP_QUEUE_EMPTY;
    }

    memcpy(item,
           queue->buffer + (queue->head * queue->config.item_size),
           queue->config.item_size);

    nimcp_mutex_unlock(&queue->mutex);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_queue_clear(nimcp_queue_handle_t queue) {
    if (!queue) return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(&queue->mutex);
    queue->head = queue->tail = 0;
    queue->status.current_size = 0;
    nimcp_cond_broadcast(&queue->not_full);
    nimcp_mutex_unlock(&queue->mutex);
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_queue_get_status(nimcp_queue_handle_t queue,
                                     nimcp_queue_status_t* status) {
    if (!queue || !status) return NIMCP_INVALID_PARAM;

    nimcp_mutex_lock(&queue->mutex);
    *status = queue->status;
    nimcp_mutex_unlock(&queue->mutex);
    return NIMCP_SUCCESS;
}

bool nimcp_queue_is_empty(nimcp_queue_handle_t queue) {
    if (!queue) return true;
    nimcp_mutex_lock(&queue->mutex);
    bool empty = is_queue_empty(queue);
    nimcp_mutex_unlock(&queue->mutex);
    return empty;
}

bool nimcp_queue_is_full(nimcp_queue_handle_t queue) {
    if (!queue) return true;
    nimcp_mutex_lock(&queue->mutex);
    bool full = is_queue_full(queue);
    nimcp_mutex_unlock(&queue->mutex);
    return full;
}

size_t nimcp_queue_get_size(nimcp_queue_handle_t queue) {
    if (!queue) return 0;
    nimcp_mutex_lock(&queue->mutex);
    size_t size = queue->status.current_size;
    nimcp_mutex_unlock(&queue->mutex);
    return size;
}
