#ifndef NIMCP_QUEUE_H
#define NIMCP_QUEUE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "utils/validation/nimcp_common.h"

// Queue configuration
typedef struct {
    size_t max_size;      // Maximum number of items
    size_t item_size;     // Size of each item in bytes
    bool is_blocking;     // Whether queue operations block when full/empty
    uint32_t timeout_ms;  // Timeout for blocking operations (0 = infinite)
} nimcp_queue_config_t;

// Queue status
typedef struct {
    size_t current_size;      // Current number of items
    size_t peak_size;         // Maximum size reached
    uint64_t total_enqueued;  // Total items enqueued
    uint64_t total_dequeued;  // Total items dequeued
    uint64_t dropped_items;   // Number of items dropped due to overflow
} nimcp_queue_status_t;

// Queue handle
typedef struct nimcp_queue* nimcp_queue_handle_t;

// Queue operations
nimcp_result_t nimcp_queue_create(const nimcp_queue_config_t* config, nimcp_queue_handle_t* queue);
nimcp_result_t nimcp_queue_destroy(nimcp_queue_handle_t queue);

nimcp_result_t nimcp_queue_enqueue(nimcp_queue_handle_t queue, const void* item,
                                   uint32_t timeout_ms);
nimcp_result_t nimcp_queue_dequeue(nimcp_queue_handle_t queue, void* item, uint32_t timeout_ms);

nimcp_result_t nimcp_queue_peek(nimcp_queue_handle_t queue, void* item);
nimcp_result_t nimcp_queue_clear(nimcp_queue_handle_t queue);

// Status and properties
nimcp_result_t nimcp_queue_get_status(nimcp_queue_handle_t queue, nimcp_queue_status_t* status);
bool nimcp_queue_is_empty(nimcp_queue_handle_t queue);
bool nimcp_queue_is_full(nimcp_queue_handle_t queue);
size_t nimcp_queue_get_size(nimcp_queue_handle_t queue);

#endif  // NIMCP_QUEUE_H
