/**
 * @file nimcp_queue_manager.c
 * @brief Implementation of thread pool-based queue management system
 */

#define NIMCP_INTERNAL
#include "utils/nimcp_queue_manager.h"
#include "utils/nimcp_memory.h"
#include "utils/nimcp_thread_pool.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Default configuration values
#define DEFAULT_WORKER_THREADS 4

// Forward declarations of helper functions
static uint64_t nimcp_get_timestamp_ms(void);
static void nimcp_yield_thread(void);

// Message helper functions (simple implementations since messages module was removed)
static nimcp_result_t nimcp_msg_clone(const nimcp_message_t* src, nimcp_message_t** dst) {
    if (!src || !dst) return NIMCP_INVALID_PARAM;

    nimcp_message_t* msg = nimcp_malloc(sizeof(nimcp_message_t));
    if (!msg) return NIMCP_NO_MEMORY;

    msg->type = src->type;
    msg->flags = src->flags;
    msg->size = src->size;

    if (src->data && src->size > 0) {
        msg->data = nimcp_malloc(src->size);
        if (!msg->data) {
            nimcp_free(msg);
            return NIMCP_NO_MEMORY;
        }
        memcpy(msg->data, src->data, src->size);
    } else {
        msg->data = NULL;
    }

    *dst = msg;
    return NIMCP_SUCCESS;
}

static void nimcp_msg_destroy(nimcp_message_t* msg) {
    if (msg) {
        if (msg->data) {
            nimcp_free(msg->data);
        }
        nimcp_free(msg);
    }
}

static nimcp_result_t validate_config(const nimcp_queue_manager_config_t* config) {
    if (!config) return NIMCP_INVALID_PARAM;
    
    if (config->max_channels == 0 || config->max_channels > NIMCP_QUEUE_MAX_CHANNELS ||
        config->queue_sizes.high > NIMCP_QUEUE_MAX_SIZE ||
        config->queue_sizes.normal > NIMCP_QUEUE_MAX_SIZE ||
        config->queue_sizes.low > NIMCP_QUEUE_MAX_SIZE ||
        config->queue_sizes.high < NIMCP_QUEUE_MIN_SIZE ||
        config->queue_sizes.normal < NIMCP_QUEUE_MIN_SIZE ||
        config->queue_sizes.low < NIMCP_QUEUE_MIN_SIZE) {
        return NIMCP_INVALID_PARAM;
    }
    
    return NIMCP_SUCCESS;
}

static size_t get_queue_size_for_priority(const nimcp_queue_manager_config_t* config, 
                                        nimcp_queue_priority_t priority) {
    switch (priority) {
        case NIMCP_QUEUE_PRIORITY_HIGH:
            return config->queue_sizes.high;
        case NIMCP_QUEUE_PRIORITY_NORMAL:
            return config->queue_sizes.normal;
        case NIMCP_QUEUE_PRIORITY_LOW:
            return config->queue_sizes.low;
        default:
            return NIMCP_QUEUE_DEFAULT_SIZE;
    }
}

static bool is_valid_channel(nimcp_queue_manager_handle_t manager, uint32_t channel_id) {
    return (manager && manager->initialized && 
            !atomic_load(&manager->shutting_down) &&
            channel_id < manager->config.max_channels);
}

static nimcp_result_t init_channel(nimcp_queue_channel_t* channel,
                                 const nimcp_queue_manager_config_t* config) {
    memset(channel, 0, sizeof(nimcp_queue_channel_t));
    
    for (int pri = 0; pri < NIMCP_QUEUE_PRIORITY_COUNT; pri++) {
        nimcp_queue_config_t queue_config = {
            .max_size = get_queue_size_for_priority(config, pri),
            .item_size = sizeof(nimcp_message_t*),
            .is_blocking = config->blocking_mode,
            .timeout_ms = config->default_timeout
        };

        nimcp_result_t result = nimcp_queue_create(&queue_config, &channel->queues[pri]);
        if (result != NIMCP_SUCCESS) {
            // Cleanup previously created queues
            for (int j = 0; j < pri; j++) {
                nimcp_queue_destroy(channel->queues[j]);
            }
            return result;
        }
    }
    
    return NIMCP_SUCCESS;
}

static void destroy_channel(nimcp_queue_channel_t* channel) {
    if (!channel) return;
    
    for (int pri = 0; pri < NIMCP_QUEUE_PRIORITY_COUNT; pri++) {
        if (channel->queues[pri]) {
            nimcp_queue_destroy(channel->queues[pri]);
        }
    }
}

static void queue_operation_handler(void* arg) {
    nimcp_queue_operation_ctx_t* ctx = (nimcp_queue_operation_ctx_t*)arg;
    if (!ctx) return;

    uint64_t start_time = nimcp_get_timestamp_ms();
    nimcp_queue_manager_handle_t manager = (nimcp_queue_manager_handle_t)ctx->result;
    nimcp_queue_channel_t* channel = &manager->channels[ctx->channel_id];

    switch (ctx->op_type) {
        case NIMCP_QUEUE_OP_ENQUEUE: {
            nimcp_message_t* msg_copy;
            ctx->status = nimcp_msg_clone(ctx->message, &msg_copy);
            if (ctx->status == NIMCP_SUCCESS) {
                ctx->status = nimcp_queue_enqueue(
                    channel->queues[ctx->priority],
                    &msg_copy,
                    ctx->timeout_ms
                );
                
                if (ctx->status == NIMCP_SUCCESS) {
                    atomic_fetch_add(&channel->stats.priorities[ctx->priority].enqueued, 1);
                    atomic_fetch_add(&channel->stats.priorities[ctx->priority].current_size, 1);
                    
                    size_t current = atomic_load(&channel->stats.priorities[ctx->priority].current_size);
                    size_t peak = atomic_load(&channel->stats.priorities[ctx->priority].peak_size);
                    
                    if (current > peak) {
                        atomic_store(&channel->stats.priorities[ctx->priority].peak_size, current);
                    }
                } else {
                    atomic_fetch_add(&channel->stats.priorities[ctx->priority].dropped, 1);
                    nimcp_msg_destroy(msg_copy);
                }
            }
            break;
        }
        
        case NIMCP_QUEUE_OP_DEQUEUE: {
            nimcp_message_t** msg_ptr = (nimcp_message_t**)ctx->result;
            ctx->status = nimcp_queue_dequeue(
                channel->queues[ctx->priority],
                msg_ptr,
                ctx->timeout_ms
            );
            
            if (ctx->status == NIMCP_SUCCESS) {
                atomic_fetch_add(&channel->stats.priorities[ctx->priority].dequeued, 1);
                atomic_fetch_sub(&channel->stats.priorities[ctx->priority].current_size, 1);
            }
            break;
        }
        
        case NIMCP_QUEUE_OP_CLEAR: {
            nimcp_queue_clear(channel->queues[ctx->priority]);
            atomic_store(&channel->stats.priorities[ctx->priority].current_size, 0);
            ctx->status = NIMCP_SUCCESS;
            break;
        }
        
        case NIMCP_QUEUE_OP_GET_STATS: {
            nimcp_queue_manager_stats_t* stats = (nimcp_queue_manager_stats_t*)ctx->result;
            memcpy(stats, &channel->stats, sizeof(nimcp_queue_manager_stats_t));
            ctx->status = NIMCP_SUCCESS;
            break;
        }
    }

    // Update operation latency statistics
    uint64_t latency = nimcp_get_timestamp_ms() - start_time;
    atomic_fetch_add(&channel->stats.priorities[ctx->priority].op_latency_sum, latency);
    atomic_fetch_add(&channel->stats.priorities[ctx->priority].op_count, 1);
    
    atomic_store(&ctx->completed, true);
}

static nimcp_result_t submit_queue_operation(
    nimcp_queue_manager_handle_t manager,
    nimcp_queue_operation_ctx_t* op_ctx
) {
    if (!manager || !op_ctx) return NIMCP_INVALID_PARAM;
    
    op_ctx->result = manager; // Store manager reference for handler
    atomic_store(&op_ctx->completed, false);
    
    nimcp_result_t result = nimcp_pool_submit(
        manager->thread_pool,
        queue_operation_handler,
        op_ctx
    );
    
    if (result != NIMCP_SUCCESS) {
        return result;
    }

    // Wait for operation completion
    uint64_t start_time = nimcp_get_timestamp_ms();
    while (!atomic_load(&op_ctx->completed)) {
        if (nimcp_get_timestamp_ms() - start_time > op_ctx->timeout_ms) {
            return NIMCP_TIMEOUT;
        }
        nimcp_yield_thread();
    }

    return op_ctx->status;
}

nimcp_result_t nimcp_queue_manager_create(
    const nimcp_queue_manager_config_t* config,
    nimcp_queue_manager_handle_t* manager
) {
    if (!config || !manager) return NIMCP_INVALID_PARAM;
    
    nimcp_result_t result = validate_config(config);
    if (result != NIMCP_SUCCESS) return result;

    nimcp_queue_manager_t* mgr = nimcp_calloc(1, sizeof(nimcp_queue_manager_t));
    if (!mgr) return NIMCP_NO_MEMORY;

    mgr->channels = nimcp_calloc(config->max_channels, sizeof(nimcp_queue_channel_t));
    if (!mgr->channels) {
        nimcp_free(mgr);
        return NIMCP_NO_MEMORY;
    }

    // Initialize channels
    for (size_t i = 0; i < config->max_channels; i++) {
        result = init_channel(&mgr->channels[i], config);
        if (result != NIMCP_SUCCESS) {
            for (size_t j = 0; j < i; j++) {
                destroy_channel(&mgr->channels[j]);
            }
            nimcp_free(mgr->channels);
            nimcp_free(mgr);
            return result;
        }
    }

    // Create thread pool
    mgr->thread_pool = nimcp_pool_create(config->worker_threads > 0 ? config->worker_threads : DEFAULT_WORKER_THREADS);
    if (!mgr->thread_pool) {
        for (size_t i = 0; i < config->max_channels; i++) {
            destroy_channel(&mgr->channels[i]);
        }
        nimcp_free(mgr->channels);
        nimcp_free(mgr);
        return result;
    }

    mgr->config = *config;
    atomic_store(&mgr->shutting_down, false);
    mgr->initialized = true;
    *manager = mgr;
    
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_queue_manager_destroy(nimcp_queue_manager_handle_t manager) {
    if (!manager) return NIMCP_INVALID_PARAM;

    atomic_store(&manager->shutting_down, true);
    
    // Destroy thread pool
    nimcp_pool_destroy(manager->thread_pool);

    // Cleanup channels
    for (size_t i = 0; i < manager->config.max_channels; i++) {
        destroy_channel(&manager->channels[i]);
    }

    nimcp_free(manager->channels);
    nimcp_free(manager);
    
    return NIMCP_SUCCESS;
}

nimcp_result_t nimcp_queue_manager_enqueue(
    nimcp_queue_manager_handle_t manager,
    uint32_t channel_id,
    const nimcp_message_t* message,
    uint32_t timeout_ms
) {
    if (!is_valid_channel(manager, channel_id) || !message) {
        return NIMCP_INVALID_PARAM;
    }

    // Map message flags to queue priority
    // Lower 2 bits of flags can indicate priority (0=low, 1=normal, 2=high)
    nimcp_queue_priority_t priority = NIMCP_QUEUE_PRIORITY_NORMAL;
    uint32_t priority_bits = message->flags & 0x3;
    if (priority_bits == 2) {
        priority = NIMCP_QUEUE_PRIORITY_HIGH;
    } else if (priority_bits == 0) {
        priority = NIMCP_QUEUE_PRIORITY_LOW;
    }

    nimcp_queue_operation_ctx_t op_ctx = {
        .op_type = NIMCP_QUEUE_OP_ENQUEUE,
        .channel_id = channel_id,
        .priority = priority,
        .message = (nimcp_message_t*)message,
        .timeout_ms = timeout_ms ? timeout_ms : manager->config.default_timeout
    };

    return submit_queue_operation(manager, &op_ctx);
}

nimcp_result_t nimcp_queue_manager_dequeue(
    nimcp_queue_manager_handle_t manager,
    uint32_t channel_id,
    nimcp_message_t** message,
    uint32_t timeout_ms
) {
    if (!is_valid_channel(manager, channel_id) || !message) {
        return NIMCP_INVALID_PARAM;
    }

    // Try dequeuing from each priority level, starting with highest
    for (int pri = NIMCP_QUEUE_PRIORITY_HIGH; pri < NIMCP_QUEUE_PRIORITY_COUNT; pri++) {
        nimcp_queue_operation_ctx_t op_ctx = {
            .op_type = NIMCP_QUEUE_OP_DEQUEUE,
            .channel_id = channel_id,
            .priority = pri,
            .result = message,
            .timeout_ms = (pri == NIMCP_QUEUE_PRIORITY_LOW) ? 
                         (timeout_ms ? timeout_ms : manager->config.default_timeout) : 0
        };

        nimcp_result_t result = submit_queue_operation(manager, &op_ctx);
        if (result == NIMCP_SUCCESS) {
            return NIMCP_SUCCESS;
        }
        
        // Only continue to lower priority if queue was empty
        if (result != NIMCP_QUEUE_EMPTY) {
            return result;
        }
    }

    return NIMCP_QUEUE_EMPTY;
}

nimcp_result_t nimcp_queue_manager_get_stats(
    nimcp_queue_manager_handle_t manager,
    uint32_t channel_id,
    nimcp_queue_manager_stats_t* stats
) {
    if (!is_valid_channel(manager, channel_id) || !stats) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_queue_operation_ctx_t op_ctx = {
        .op_type = NIMCP_QUEUE_OP_GET_STATS,
        .channel_id = channel_id,
        .result = stats,
        .timeout_ms = manager->config.default_timeout
    };

    return submit_queue_operation(manager, &op_ctx);
}

nimcp_result_t nimcp_queue_manager_clear(
    nimcp_queue_manager_handle_t manager,
    uint32_t channel_id
) {
    if (!is_valid_channel(manager, channel_id)) {
        return NIMCP_INVALID_PARAM;
    }

    nimcp_result_t final_result = NIMCP_SUCCESS;

    // Clear all priority queues
    for (int pri = 0; pri < NIMCP_QUEUE_PRIORITY_COUNT; pri++) {
        nimcp_queue_operation_ctx_t op_ctx = {
            .op_type = NIMCP_QUEUE_OP_CLEAR,
            .channel_id = channel_id,
            .priority = pri,
            .timeout_ms = manager->config.default_timeout
        };

        nimcp_result_t result = submit_queue_operation(manager, &op_ctx);
        if (result != NIMCP_SUCCESS) {
            final_result = result;
        }
    }

    return final_result;
}

nimcp_result_t nimcp_queue_manager_set_timeout(
    nimcp_queue_manager_handle_t manager,
    uint32_t timeout_ms
) {
    if (!manager || !manager->initialized) {
        return NIMCP_INVALID_PARAM;
    }

    manager->config.default_timeout = timeout_ms;
    return NIMCP_SUCCESS;
}

bool nimcp_queue_manager_is_empty(
    nimcp_queue_manager_handle_t manager,
    uint32_t channel_id,
    nimcp_queue_priority_t priority
) {
    if (!is_valid_channel(manager, channel_id) || 
        priority >= NIMCP_QUEUE_PRIORITY_COUNT) {
        return true;
    }

    nimcp_queue_channel_t* channel = &manager->channels[channel_id];
    return atomic_load(&channel->stats.priorities[priority].current_size) == 0;
}

bool nimcp_queue_manager_is_full(
    nimcp_queue_manager_handle_t manager,
    uint32_t channel_id,
    nimcp_queue_priority_t priority
) {
    if (!is_valid_channel(manager, channel_id) || 
        priority >= NIMCP_QUEUE_PRIORITY_COUNT) {
        return true;
    }

    nimcp_queue_channel_t* channel = &manager->channels[channel_id];
    size_t max_size = get_queue_size_for_priority(&manager->config, priority);
    return atomic_load(&channel->stats.priorities[priority].current_size) >= max_size;
}

size_t nimcp_queue_manager_get_size(
    nimcp_queue_manager_handle_t manager,
    uint32_t channel_id,
    nimcp_queue_priority_t priority
) {
    if (!is_valid_channel(manager, channel_id) || 
        priority >= NIMCP_QUEUE_PRIORITY_COUNT) {
        return 0;
    }

    nimcp_queue_channel_t* channel = &manager->channels[channel_id];
    return atomic_load(&channel->stats.priorities[priority].current_size);
}

// Helper function to get current timestamp in milliseconds
static uint64_t nimcp_get_timestamp_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

// Helper function to yield thread
static void nimcp_yield_thread(void) {
    struct timespec ts = {0, 100000}; // 100 microseconds
    nanosleep(&ts, NULL);
}
