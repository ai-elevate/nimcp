/**
 * @file nimcp_bio_router.c
 * @brief Bio-async message router implementation
 *
 * WHAT: Central message routing for inter-module communication
 * WHY:  Decouples modules via message passing with biological semantics
 * HOW:  Per-module inboxes, handler registry, async dispatch, statistics
 *
 * ARCHITECTURE:
 * - Global router singleton with module registry
 * - Per-module context with inbox queue and handler table
 * - Lock-protected module list, lock-free message dispatch where possible
 * - Worker thread pool for async message delivery
 *
 * @author NIMCP Development Team
 * @date 2025-11-28
 */

#define LOG_MODULE "bio_router"

#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"

#include "async/nimcp_bio_messages.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/platform/nimcp_platform_cond.h"
#include "utils/platform/nimcp_platform_time.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Global BBB system accessor (defined in nimcp_brain_init.c)
extern bbb_system_t nimcp_bbb_get_global_system(void);

/*=============================================================================
 * CONSTANTS
 *============================================================================*/

#define BIO_ROUTER_MAGIC 0x42494F52  // 'BIOR'
#define BIO_MODULE_MAGIC 0x424D4F44  // 'BMOD'
#define MAX_HANDLERS_PER_MODULE 256
#define MAX_INBOX_MESSAGES 1024
#define MAX_WORKER_THREADS 8
#define DEFAULT_TIMEOUT_MS 5000
#define MAX_MODULE_NAME 64

/*=============================================================================
 * INTERNAL STRUCTURES
 *============================================================================*/

/**
 * @brief Message queue entry (inbox/outbox)
 */
typedef struct {
    void* msg_data;                 /**< Message buffer (header + payload) */
    size_t msg_size;                /**< Total message size */
    nimcp_bio_promise_t response_promise; /**< Response promise (may be NULL) */
    uint64_t enqueue_time_us;       /**< Timestamp when queued */
} bio_msg_queue_entry_t;

/**
 * @brief Simple ring buffer for messages
 */
typedef struct {
    bio_msg_queue_entry_t* entries;
    uint32_t capacity;
    uint32_t read_idx;
    uint32_t write_idx;
    uint32_t count;
    nimcp_platform_mutex_t mutex;
    nimcp_platform_cond_t not_empty;
    nimcp_platform_cond_t not_full;
} bio_msg_queue_t;

/**
 * @brief Handler registration entry
 */
typedef struct {
    bio_message_type_t msg_type;    /**< Message type handled */
    uint32_t category_mask;         /**< Category mask (0 = specific type) */
    bio_message_handler_t handler;  /**< Handler callback */
    bool is_category_handler;       /**< true if category, false if specific */
} bio_handler_entry_t;

/**
 * @brief Module entry in router registry
 */
typedef struct {
    uint32_t magic;
    bio_module_id_t module_id;
    char module_name[MAX_MODULE_NAME];
    void* user_data;

    bio_msg_queue_t inbox;
    bio_handler_entry_t handlers[MAX_HANDLERS_PER_MODULE];
    uint32_t handler_count;
    nimcp_platform_mutex_t handler_mutex;

    // Statistics
    uint64_t messages_received;
    uint64_t messages_sent;
    uint64_t handler_invocations;
    uint64_t handler_errors;
} bio_module_entry_t;

/**
 * @brief Module context (opaque handle)
 */
struct bio_module_context_struct {
    uint32_t magic;
    bio_module_entry_t* entry;  /**< Back-reference to registry entry */
};

/**
 * @brief Router global state
 */
struct bio_router_struct {
    uint32_t magic;
    bio_router_config_t config;

    bio_module_entry_t* modules;
    uint32_t module_count;
    uint32_t module_capacity;
    nimcp_platform_mutex_t modules_mutex;

    // Statistics
    bio_router_stats_t stats;
    nimcp_platform_mutex_t stats_mutex;

    unified_mem_manager_t mem_mgr;
    bool initialized;
    bool shutdown_requested;
};

/*=============================================================================
 * GLOBAL STATE
 *============================================================================*/

static struct bio_router_struct* g_router = NULL;
static nimcp_platform_mutex_t g_router_init_mutex;
static bool g_router_init_mutex_initialized = false;

/*=============================================================================
 * MESSAGE QUEUE OPERATIONS
 *============================================================================*/

/**
 * WHAT: Initialize message queue
 * WHY:  Setup inbox/outbox for module
 * HOW:  Allocate ring buffer, initialize mutex and condvars
 */
static nimcp_error_t bio_msg_queue_init(bio_msg_queue_t* queue, uint32_t capacity) {
    if (!queue || capacity == 0) return -1;

    queue->entries = nimcp_calloc(capacity, sizeof(bio_msg_queue_entry_t));
    if (!queue->entries) return -1;

    queue->capacity = capacity;
    queue->read_idx = 0;
    queue->write_idx = 0;
    queue->count = 0;

    if (nimcp_platform_mutex_init(&queue->mutex, false) != 0) {
        nimcp_free(queue->entries);
        return -1;
    }

    if (nimcp_platform_cond_init(&queue->not_empty) != 0) {
        nimcp_platform_mutex_destroy(&queue->mutex);
        nimcp_free(queue->entries);
        return -1;
    }

    if (nimcp_platform_cond_init(&queue->not_full) != 0) {
        nimcp_platform_cond_destroy(&queue->not_empty);
        nimcp_platform_mutex_destroy(&queue->mutex);
        nimcp_free(queue->entries);
        return -1;
    }

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Destroy message queue
 * WHY:  Free resources on module unregister
 * HOW:  Free pending messages, destroy synchronization primitives
 * NOTE: Promises are NOT destroyed here - they are owned by the caller of
 *       bio_router_send_async and must be destroyed by that caller even if
 *       the message was never processed.
 */
static void bio_msg_queue_destroy(bio_msg_queue_t* queue) {
    if (!queue || !queue->entries) return;

    // Count pending messages for warning
    uint32_t pending_with_promises = 0;

    // Free all pending messages (but NOT the promises - caller owns those)
    for (uint32_t i = 0; i < queue->count; i++) {
        uint32_t idx = (queue->read_idx + i) % queue->capacity;
        if (queue->entries[idx].msg_data) {
            nimcp_free(queue->entries[idx].msg_data);
        }
        if (queue->entries[idx].response_promise) {
            // DO NOT destroy the promise - caller of bio_router_send_async owns it
            // Just log a warning that there are unprocessed async messages
            pending_with_promises++;
        }
    }

    if (pending_with_promises > 0) {
        LOG_WARNING("bio_msg_queue_destroy: %u pending messages with response promises "
                    "will not have their handlers invoked. Caller must still destroy "
                    "the promises.", pending_with_promises);
    }

    nimcp_platform_cond_destroy(&queue->not_full);
    nimcp_platform_cond_destroy(&queue->not_empty);
    nimcp_platform_mutex_destroy(&queue->mutex);
    nimcp_free(queue->entries);

    memset(queue, 0, sizeof(*queue));
}

/**
 * WHAT: Enqueue message to queue
 * WHY:  Add message to module inbox
 * HOW:  Blocking if full, copy message data, signal waiters
 */
static nimcp_error_t bio_msg_queue_enqueue(bio_msg_queue_t* queue,
                                            const void* msg,
                                            size_t msg_size,
                                            nimcp_bio_promise_t response_promise,
                                            uint32_t timeout_ms) {
    if (!queue || !msg || msg_size == 0) return -1;

    nimcp_platform_mutex_lock(&queue->mutex);

    // Wait for space if full
    if (queue->count >= queue->capacity) {
        if (timeout_ms == 0) {
            nimcp_platform_mutex_unlock(&queue->mutex);
            return -1;  // Would block
        }

        int wait_result = nimcp_platform_cond_timedwait(&queue->not_full,
                                                          &queue->mutex,
                                                          timeout_ms);
        if (wait_result != 0) {
            nimcp_platform_mutex_unlock(&queue->mutex);
            return -1;  // Timeout or error
        }

        if (queue->count >= queue->capacity) {
            nimcp_platform_mutex_unlock(&queue->mutex);
            return -1;  // Still full after wait
        }
    }

    // Copy message data
    void* msg_copy = nimcp_malloc(msg_size);
    if (!msg_copy) {
        nimcp_platform_mutex_unlock(&queue->mutex);
        return -1;
    }
    memcpy(msg_copy, msg, msg_size);

    // Add to queue
    bio_msg_queue_entry_t* entry = &queue->entries[queue->write_idx];
    entry->msg_data = msg_copy;
    entry->msg_size = msg_size;
    entry->response_promise = response_promise;
    entry->enqueue_time_us = nimcp_platform_time_monotonic_us();

    queue->write_idx = (queue->write_idx + 1) % queue->capacity;
    queue->count++;

    // Signal waiting consumers
    nimcp_platform_cond_signal(&queue->not_empty);

    nimcp_platform_mutex_unlock(&queue->mutex);
    return NIMCP_SUCCESS;
}

/**
 * WHAT: Dequeue message from queue
 * WHY:  Process next message from inbox
 * HOW:  Blocking if empty, return message data
 */
static nimcp_error_t bio_msg_queue_dequeue(bio_msg_queue_t* queue,
                                            void** out_msg,
                                            size_t* out_size,
                                            nimcp_bio_promise_t* out_promise,
                                            uint32_t timeout_ms) {
    if (!queue || !out_msg || !out_size) return -1;

    nimcp_platform_mutex_lock(&queue->mutex);

    // Wait for message if empty
    if (queue->count == 0) {
        if (timeout_ms == 0) {
            nimcp_platform_mutex_unlock(&queue->mutex);
            return -1;  // Would block
        }

        int wait_result = nimcp_platform_cond_timedwait(&queue->not_empty,
                                                          &queue->mutex,
                                                          timeout_ms);
        if (wait_result != 0) {
            nimcp_platform_mutex_unlock(&queue->mutex);
            return -1;  // Timeout or error
        }

        if (queue->count == 0) {
            nimcp_platform_mutex_unlock(&queue->mutex);
            return -1;  // Still empty after wait
        }
    }

    // Remove from queue
    bio_msg_queue_entry_t* entry = &queue->entries[queue->read_idx];
    *out_msg = entry->msg_data;
    *out_size = entry->msg_size;
    if (out_promise) {
        *out_promise = entry->response_promise;
    }

    entry->msg_data = NULL;
    entry->response_promise = NULL;

    queue->read_idx = (queue->read_idx + 1) % queue->capacity;
    queue->count--;

    // Signal waiting producers
    nimcp_platform_cond_signal(&queue->not_full);

    nimcp_platform_mutex_unlock(&queue->mutex);
    return NIMCP_SUCCESS;
}

/**
 * WHAT: Get queue count (non-blocking)
 * WHY:  Check inbox size
 * HOW:  Lock, read count, unlock
 */
static uint32_t bio_msg_queue_count(bio_msg_queue_t* queue) {
    if (!queue) return 0;

    nimcp_platform_mutex_lock(&queue->mutex);
    uint32_t count = queue->count;
    nimcp_platform_mutex_unlock(&queue->mutex);

    return count;
}

/*=============================================================================
 * ROUTER LIFECYCLE
 *============================================================================*/

bio_router_config_t bio_router_default_config(void) {
    bio_router_config_t config = {
        .max_modules = 64,
        .inbox_capacity = 256,
        .outbox_capacity = 256,
        .max_message_size = 64 * 1024,  // 64KB
        .worker_threads = 4,
        .enable_logging = true,
        .enable_statistics = true,
        .routing_timeout_ms = DEFAULT_TIMEOUT_MS
    };
    return config;
}

nimcp_error_t bio_router_init(const bio_router_config_t* config) {
    // Initialize global init mutex once
    if (!g_router_init_mutex_initialized) {
        nimcp_platform_mutex_init(&g_router_init_mutex, false);
        g_router_init_mutex_initialized = true;
    }

    nimcp_platform_mutex_lock(&g_router_init_mutex);

    // Check if already initialized
    if (g_router != NULL) {
        nimcp_platform_mutex_unlock(&g_router_init_mutex);
        LOG_WARN("Bio-router already initialized");
        return NIMCP_ERROR_ALREADY_EXISTS;
    }

    // Use defaults if no config provided
    bio_router_config_t cfg = config ? *config : bio_router_default_config();

    // Allocate router
    g_router = nimcp_calloc(1, sizeof(struct bio_router_struct));
    if (!g_router) {
        nimcp_platform_mutex_unlock(&g_router_init_mutex);
        LOG_ERROR("Failed to allocate bio-router");
        return -1;
    }

    g_router->magic = BIO_ROUTER_MAGIC;
    g_router->config = cfg;
    g_router->module_capacity = cfg.max_modules;

    // Allocate module registry
    g_router->modules = nimcp_calloc(cfg.max_modules, sizeof(bio_module_entry_t));
    if (!g_router->modules) {
        nimcp_free(g_router);
        g_router = NULL;
        nimcp_platform_mutex_unlock(&g_router_init_mutex);
        LOG_ERROR("Failed to allocate module registry");
        return -1;
    }

    // Initialize mutexes
    if (nimcp_platform_mutex_init(&g_router->modules_mutex, false) != 0) {
        nimcp_free(g_router->modules);
        nimcp_free(g_router);
        g_router = NULL;
        nimcp_platform_mutex_unlock(&g_router_init_mutex);
        LOG_ERROR("Failed to initialize modules mutex");
        return -1;
    }

    if (nimcp_platform_mutex_init(&g_router->stats_mutex, false) != 0) {
        nimcp_platform_mutex_destroy(&g_router->modules_mutex);
        nimcp_free(g_router->modules);
        nimcp_free(g_router);
        g_router = NULL;
        nimcp_platform_mutex_unlock(&g_router_init_mutex);
        LOG_ERROR("Failed to initialize stats mutex");
        return -1;
    }

    // Initialize unified memory if requested
    if (cfg.max_message_size > 0) {
        unified_mem_config_t mem_cfg = unified_mem_default_config();
        mem_cfg.enable_cow = false;  // Direct allocation for messages
        mem_cfg.object_pool_num_blocks = cfg.max_modules * cfg.inbox_capacity;
        g_router->mem_mgr = unified_mem_create(&mem_cfg);
    }

    g_router->initialized = true;
    g_router->shutdown_requested = false;

    nimcp_platform_mutex_unlock(&g_router_init_mutex);

    LOG_INFO("Bio-router initialized (max_modules=%u, inbox_capacity=%u)",
             cfg.max_modules, cfg.inbox_capacity);

    return NIMCP_SUCCESS;
}

bio_router_t bio_router_get_global(void) {
    return g_router;
}

void bio_router_shutdown(void) {
    if (!g_router) return;

    LOG_INFO("Shutting down bio-router");

    g_router->shutdown_requested = true;

    // Unregister all modules
    nimcp_platform_mutex_lock(&g_router->modules_mutex);
    for (uint32_t i = 0; i < g_router->module_count; i++) {
        bio_module_entry_t* entry = &g_router->modules[i];
        if (entry->magic == BIO_MODULE_MAGIC) {
            bio_msg_queue_destroy(&entry->inbox);
            nimcp_platform_mutex_destroy(&entry->handler_mutex);
            entry->magic = 0;
        }
    }
    nimcp_platform_mutex_unlock(&g_router->modules_mutex);

    // Destroy memory manager
    if (g_router->mem_mgr) {
        unified_mem_destroy(g_router->mem_mgr);
    }

    // Destroy mutexes
    nimcp_platform_mutex_destroy(&g_router->stats_mutex);
    nimcp_platform_mutex_destroy(&g_router->modules_mutex);

    // Free module registry
    nimcp_free(g_router->modules);

    // Free router
    nimcp_free(g_router);
    g_router = NULL;

    LOG_INFO("Bio-router shutdown complete");
}

bool bio_router_is_initialized(void) {
    return g_router != NULL && g_router->initialized;
}

nimcp_error_t bio_router_get_stats(bio_router_stats_t* stats) {
    if (!g_router || !stats) return -1;

    nimcp_platform_mutex_lock(&g_router->stats_mutex);
    *stats = g_router->stats;
    nimcp_platform_mutex_unlock(&g_router->stats_mutex);

    return NIMCP_SUCCESS;
}

void bio_router_reset_stats(void) {
    if (!g_router) return;

    // Count active modules before reset (need to preserve this)
    nimcp_platform_mutex_lock(&g_router->modules_mutex);
    uint32_t active_count = 0;
    for (uint32_t i = 0; i < g_router->module_count; i++) {
        if (g_router->modules[i].magic == BIO_MODULE_MAGIC) {
            active_count++;
        }
    }
    nimcp_platform_mutex_unlock(&g_router->modules_mutex);

    // Reset stats but preserve active module count
    nimcp_platform_mutex_lock(&g_router->stats_mutex);
    memset(&g_router->stats, 0, sizeof(g_router->stats));
    g_router->stats.active_modules = active_count;
    nimcp_platform_mutex_unlock(&g_router->stats_mutex);

    LOG_DEBUG("Bio-router statistics reset (active_modules=%u preserved)", active_count);
}

/*=============================================================================
 * MODULE REGISTRATION
 *============================================================================*/

bio_module_context_t bio_router_register_module(const bio_module_info_t* info) {
    if (!g_router || !info) {
        LOG_ERROR("Cannot register module: router not initialized or invalid info");
        return NULL;
    }

    nimcp_platform_mutex_lock(&g_router->modules_mutex);

    // Check for duplicate module ID - if already registered, create a NEW context
    // wrapping the existing entry (allows multiple consumers of same module ID)
    for (uint32_t i = 0; i < g_router->module_count; i++) {
        if (g_router->modules[i].magic == BIO_MODULE_MAGIC &&
            g_router->modules[i].module_id == info->module_id) {
            bio_module_entry_t* existing = &g_router->modules[i];
            nimcp_platform_mutex_unlock(&g_router->modules_mutex);
            LOG_DEBUG("Module ID %u (%s) already registered, creating new context wrapper",
                     info->module_id, existing->module_name);
            // CRITICAL FIX: Must allocate a proper context struct, NOT cast entry!
            // The old code returned (bio_module_context_t)existing which was wrong because
            // bio_module_entry_t and bio_module_context_struct are different structs.
            bio_module_context_t ctx = nimcp_calloc(1, sizeof(struct bio_module_context_struct));
            if (!ctx) {
                LOG_ERROR("Failed to allocate context for existing module");
                return NULL;
            }
            ctx->magic = BIO_MODULE_MAGIC;
            ctx->entry = existing;
            return ctx;
        }
    }

    // Find a free slot (either a reusable invalid slot, or a new slot at the end)
    bio_module_entry_t* entry = NULL;

    // First try to reuse an invalid slot
    for (uint32_t i = 0; i < g_router->module_count; i++) {
        if (g_router->modules[i].magic != BIO_MODULE_MAGIC) {
            entry = &g_router->modules[i];
            LOG_DEBUG("Reusing module slot %u for new module %u", i, info->module_id);
            break;
        }
    }

    // If no reusable slot, allocate at the end
    if (!entry) {
        if (g_router->module_count >= g_router->module_capacity) {
            nimcp_platform_mutex_unlock(&g_router->modules_mutex);
            LOG_ERROR("Module registry full (max=%u)", g_router->module_capacity);
            return NULL;
        }
        entry = &g_router->modules[g_router->module_count];
        g_router->module_count++;
    }

    memset(entry, 0, sizeof(*entry));

    entry->magic = BIO_MODULE_MAGIC;
    entry->module_id = info->module_id;
    strncpy(entry->module_name, info->module_name ? info->module_name : "unknown",
            MAX_MODULE_NAME - 1);
    entry->user_data = info->user_data;

    // Initialize inbox
    uint32_t inbox_cap = info->inbox_capacity > 0 ?
                         info->inbox_capacity : g_router->config.inbox_capacity;
    if (bio_msg_queue_init(&entry->inbox, inbox_cap) != NIMCP_SUCCESS) {
        nimcp_platform_mutex_unlock(&g_router->modules_mutex);
        LOG_ERROR("Failed to initialize inbox for module %s", entry->module_name);
        return NULL;
    }

    // Initialize handler mutex
    if (nimcp_platform_mutex_init(&entry->handler_mutex, false) != 0) {
        bio_msg_queue_destroy(&entry->inbox);
        nimcp_platform_mutex_unlock(&g_router->modules_mutex);
        LOG_ERROR("Failed to initialize handler mutex for module %s", entry->module_name);
        return NULL;
    }

    // Count active modules for statistics
    uint32_t active_count = 0;
    for (uint32_t i = 0; i < g_router->module_count; i++) {
        if (g_router->modules[i].magic == BIO_MODULE_MAGIC) {
            active_count++;
        }
    }

    // Update statistics
    nimcp_platform_mutex_lock(&g_router->stats_mutex);
    g_router->stats.active_modules = active_count;
    nimcp_platform_mutex_unlock(&g_router->stats_mutex);

    nimcp_platform_mutex_unlock(&g_router->modules_mutex);

    // Create context handle
    bio_module_context_t ctx = nimcp_calloc(1, sizeof(struct bio_module_context_struct));
    if (!ctx) {
        LOG_ERROR("Failed to allocate module context");
        return NULL;
    }

    ctx->magic = BIO_MODULE_MAGIC;
    ctx->entry = entry;

    LOG_INFO("Registered module: id=%u, name=%s, inbox_capacity=%u",
             info->module_id, entry->module_name, inbox_cap);

    return ctx;
}

void bio_router_unregister_module(bio_module_context_t ctx) {
    if (!g_router || !ctx || ctx->magic != BIO_MODULE_MAGIC) return;

    bio_module_entry_t* entry = ctx->entry;
    if (!entry || entry->magic != BIO_MODULE_MAGIC) return;

    LOG_INFO("Unregistering module: id=%u, name=%s",
             entry->module_id, entry->module_name);

    nimcp_platform_mutex_lock(&g_router->modules_mutex);

    // Destroy inbox
    bio_msg_queue_destroy(&entry->inbox);

    // Destroy handler mutex
    nimcp_platform_mutex_destroy(&entry->handler_mutex);

    // Mark as invalid (DO NOT compact array - other contexts have pointers to entries)
    entry->magic = 0;

    // Note: We don't decrement module_count because it's a high-water mark.
    // Invalid entries (magic=0) are skipped by find_module and shutdown.

    // Count active modules for statistics
    uint32_t active_count = 0;
    for (uint32_t i = 0; i < g_router->module_count; i++) {
        if (g_router->modules[i].magic == BIO_MODULE_MAGIC) {
            active_count++;
        }
    }

    // Update statistics
    nimcp_platform_mutex_lock(&g_router->stats_mutex);
    g_router->stats.active_modules = active_count;
    nimcp_platform_mutex_unlock(&g_router->stats_mutex);

    nimcp_platform_mutex_unlock(&g_router->modules_mutex);

    // Free context
    ctx->magic = 0;
    nimcp_free(ctx);
}

nimcp_error_t bio_router_register_handler(bio_module_context_t ctx,
                                           bio_message_type_t msg_type,
                                           bio_message_handler_t handler) {
    if (!ctx || ctx->magic != BIO_MODULE_MAGIC || !handler) return -1;

    bio_module_entry_t* entry = ctx->entry;
    if (!entry || entry->magic != BIO_MODULE_MAGIC) return -1;

    nimcp_platform_mutex_lock(&entry->handler_mutex);

    if (entry->handler_count >= MAX_HANDLERS_PER_MODULE) {
        nimcp_platform_mutex_unlock(&entry->handler_mutex);
        LOG_ERROR("Handler table full for module %s", entry->module_name);
        return -1;
    }

    // Check for duplicate
    for (uint32_t i = 0; i < entry->handler_count; i++) {
        if (entry->handlers[i].msg_type == msg_type &&
            !entry->handlers[i].is_category_handler) {
            nimcp_platform_mutex_unlock(&entry->handler_mutex);
            LOG_WARN("Handler for message type 0x%04X already registered in module %s",
                     msg_type, entry->module_name);
            return -1;
        }
    }

    // Add handler
    bio_handler_entry_t* h = &entry->handlers[entry->handler_count];
    h->msg_type = msg_type;
    h->category_mask = 0;
    h->handler = handler;
    h->is_category_handler = false;

    entry->handler_count++;

    nimcp_platform_mutex_unlock(&entry->handler_mutex);

    LOG_DEBUG("Registered handler for message type 0x%04X in module %s",
              msg_type, entry->module_name);

    return NIMCP_SUCCESS;
}

nimcp_error_t bio_router_register_category_handler(bio_module_context_t ctx,
                                                     uint32_t category_base,
                                                     bio_message_handler_t handler) {
    if (!ctx || ctx->magic != BIO_MODULE_MAGIC || !handler) return -1;

    bio_module_entry_t* entry = ctx->entry;
    if (!entry || entry->magic != BIO_MODULE_MAGIC) return -1;

    nimcp_platform_mutex_lock(&entry->handler_mutex);

    if (entry->handler_count >= MAX_HANDLERS_PER_MODULE) {
        nimcp_platform_mutex_unlock(&entry->handler_mutex);
        LOG_ERROR("Handler table full for module %s", entry->module_name);
        return -1;
    }

    // Add category handler
    bio_handler_entry_t* h = &entry->handlers[entry->handler_count];
    h->msg_type = category_base;
    h->category_mask = 0xFF00;  // Match top byte
    h->handler = handler;
    h->is_category_handler = true;

    entry->handler_count++;

    nimcp_platform_mutex_unlock(&entry->handler_mutex);

    LOG_DEBUG("Registered category handler for 0x%04X in module %s",
              category_base, entry->module_name);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * MESSAGE SENDING
 *============================================================================*/

/**
 * WHAT: Find module entry by ID
 * WHY:  Lookup target module for routing
 * HOW:  Linear search through registry (modules_mutex must be held)
 */
static bio_module_entry_t* bio_router_find_module(bio_module_id_t module_id) {
    for (uint32_t i = 0; i < g_router->module_count; i++) {
        if (g_router->modules[i].module_id == module_id &&
            g_router->modules[i].magic == BIO_MODULE_MAGIC) {
            return &g_router->modules[i];
        }
    }
    return NULL;
}

nimcp_error_t bio_router_send(bio_module_context_t ctx,
                               const void* msg,
                               size_t msg_size,
                               uint32_t timeout_ms) {
    if (!g_router || !ctx || ctx->magic != BIO_MODULE_MAGIC || !msg || msg_size == 0) {
        return -1;
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;
    uint32_t target_id = header->target_module;

    uint64_t start_time = nimcp_platform_time_monotonic_us();

    LOG_TRACE("Routing message type=0x%04X from=%u to=%u size=%zu",
              header->type, header->source_module, target_id, msg_size);

    // BBB Security Gate: Validate message content before dispatch
    // WHAT: Apply Blood-Brain Barrier validation to all messages
    // WHY:  Prevent malicious or malformed messages from affecting modules
    // HOW:  Use global BBB system to validate raw message bytes
    bbb_system_t bbb = nimcp_bbb_get_global_system();
    if (bbb && bbb_system_is_enabled(bbb)) {
        bbb_validation_result_t bbb_result;
        if (!bbb_validate_input(bbb, msg, msg_size, &bbb_result)) {
            LOG_WARN("BBB validation failed for message type=0x%04X from=%u: threat=%d severity=%d",
                     header->type, header->source_module, bbb_result.threat, bbb_result.severity);

            // Update dropped message stats
            nimcp_platform_mutex_lock(&g_router->stats_mutex);
            g_router->stats.messages_dropped++;
            nimcp_platform_mutex_unlock(&g_router->stats_mutex);

            return NIMCP_ERROR_PERMISSION_DENIED;
        }
        LOG_TRACE("BBB validation passed for message type=0x%04X", header->type);
    }

    nimcp_platform_mutex_lock(&g_router->modules_mutex);

    nimcp_error_t result = NIMCP_SUCCESS;

    if (target_id == 0) {
        // Broadcast to all modules except sender
        for (uint32_t i = 0; i < g_router->module_count; i++) {
            bio_module_entry_t* entry = &g_router->modules[i];
            if (entry->magic == BIO_MODULE_MAGIC &&
                entry->module_id != header->source_module) {

                nimcp_error_t enq_result = bio_msg_queue_enqueue(&entry->inbox,
                                                                  msg, msg_size,
                                                                  NULL, timeout_ms);
                if (enq_result == NIMCP_SUCCESS) {
                    entry->messages_received++;
                } else {
                    result = -1;
                    LOG_WARN("Failed to enqueue broadcast to module %u", entry->module_id);
                }
            }
        }

        // Update broadcast stats
        nimcp_platform_mutex_lock(&g_router->stats_mutex);
        g_router->stats.broadcasts_sent++;
        nimcp_platform_mutex_unlock(&g_router->stats_mutex);

    } else {
        // Send to specific module
        bio_module_entry_t* target = bio_router_find_module(target_id);
        if (!target) {
            nimcp_platform_mutex_unlock(&g_router->modules_mutex);
            LOG_ERROR("Target module %u not found", target_id);

            nimcp_platform_mutex_lock(&g_router->stats_mutex);
            g_router->stats.messages_dropped++;
            nimcp_platform_mutex_unlock(&g_router->stats_mutex);

            return -1;
        }

        result = bio_msg_queue_enqueue(&target->inbox, msg, msg_size,
                                        NULL, timeout_ms);
        if (result == NIMCP_SUCCESS) {
            target->messages_received++;
        } else {
            LOG_WARN("Failed to enqueue to module %u inbox", target_id);

            nimcp_platform_mutex_lock(&g_router->stats_mutex);
            g_router->stats.messages_dropped++;
            nimcp_platform_mutex_unlock(&g_router->stats_mutex);
        }
    }

    nimcp_platform_mutex_unlock(&g_router->modules_mutex);

    // Update routing statistics
    if (result == NIMCP_SUCCESS) {
        uint64_t latency_us = nimcp_platform_time_monotonic_us() - start_time;
        float latency = (float)latency_us;

        nimcp_platform_mutex_lock(&g_router->stats_mutex);
        g_router->stats.messages_routed++;

        // Update latency stats
        if (g_router->stats.messages_routed == 1) {
            g_router->stats.avg_routing_latency_us = latency;
            g_router->stats.max_routing_latency_us = latency;
        } else {
            g_router->stats.avg_routing_latency_us =
                (g_router->stats.avg_routing_latency_us * 0.95f) + (latency * 0.05f);
            if (latency > g_router->stats.max_routing_latency_us) {
                g_router->stats.max_routing_latency_us = latency;
            }
        }

        nimcp_platform_mutex_unlock(&g_router->stats_mutex);
    }

    return result;
}

/**
 * @brief Internal send with optional response promise
 */
static nimcp_error_t bio_router_send_with_promise(bio_module_context_t ctx,
                                                   const void* msg,
                                                   size_t msg_size,
                                                   nimcp_bio_promise_t response_promise,
                                                   uint32_t timeout_ms) {
    if (!g_router || !ctx || ctx->magic != BIO_MODULE_MAGIC || !msg || msg_size == 0) {
        return -1;
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;
    uint32_t target_id = header->target_module;

    LOG_TRACE("Routing message (with promise) type=0x%04X from=%u to=%u size=%zu",
              header->type, header->source_module, target_id, msg_size);

    // BBB Security Gate: Validate message content before dispatch
    // WHAT: Apply Blood-Brain Barrier validation to all messages
    // WHY:  Prevent malicious or malformed messages from affecting modules
    // HOW:  Use global BBB system to validate raw message bytes
    bbb_system_t bbb = nimcp_bbb_get_global_system();
    if (bbb && bbb_system_is_enabled(bbb)) {
        bbb_validation_result_t bbb_result;
        if (!bbb_validate_input(bbb, msg, msg_size, &bbb_result)) {
            LOG_WARN("BBB validation failed for async message type=0x%04X from=%u: threat=%d severity=%d",
                     header->type, header->source_module, bbb_result.threat, bbb_result.severity);

            // Update dropped message stats
            nimcp_platform_mutex_lock(&g_router->stats_mutex);
            g_router->stats.messages_dropped++;
            nimcp_platform_mutex_unlock(&g_router->stats_mutex);

            return NIMCP_ERROR_PERMISSION_DENIED;
        }
        LOG_TRACE("BBB validation passed for async message type=0x%04X", header->type);
    }

    nimcp_platform_mutex_lock(&g_router->modules_mutex);

    nimcp_error_t result = NIMCP_SUCCESS;

    if (target_id == 0) {
        // Broadcast - can't have response promise for broadcasts
        nimcp_platform_mutex_unlock(&g_router->modules_mutex);
        LOG_ERROR("Cannot use async send with response for broadcasts");
        return -1;
    } else {
        // Send to specific module with promise
        bio_module_entry_t* target = bio_router_find_module(target_id);
        if (!target) {
            nimcp_platform_mutex_unlock(&g_router->modules_mutex);
            LOG_ERROR("Target module %u not found", target_id);
            return -1;
        }

        result = bio_msg_queue_enqueue(&target->inbox, msg, msg_size,
                                        response_promise, timeout_ms);
        if (result == NIMCP_SUCCESS) {
            target->messages_received++;
        } else {
            LOG_WARN("Failed to enqueue to module %u inbox", target_id);
        }
    }

    nimcp_platform_mutex_unlock(&g_router->modules_mutex);

    if (result == NIMCP_SUCCESS) {
        nimcp_platform_mutex_lock(&g_router->stats_mutex);
        g_router->stats.messages_routed++;
        nimcp_platform_mutex_unlock(&g_router->stats_mutex);
    }

    return result;
}

nimcp_bio_promise_t bio_router_send_async(bio_module_context_t ctx,
                                           const void* msg,
                                           size_t msg_size,
                                           nimcp_bio_channel_type_t channel) {
    if (!g_router || !ctx || !msg || msg_size == 0) return NULL;

    // Create promise for response
    // Note: result_size is used as capacity - actual copy size determined by
    // nimcp_bio_promise_complete_sized(). Use 0 to indicate handler will
    // provide the actual size.
    nimcp_bio_promise_t promise = nimcp_bio_promise_create(channel, 0);
    if (!promise) {
        LOG_ERROR("Failed to create promise for async send");
        return NULL;
    }

    // Send message with promise attached so handler can complete it
    nimcp_error_t result = bio_router_send_with_promise(ctx, msg, msg_size, promise, 0);
    if (result != NIMCP_SUCCESS) {
        nimcp_bio_promise_fail(promise, result);
    }

    return promise;
}

nimcp_error_t bio_router_request(bio_module_context_t ctx,
                                  const void* request,
                                  size_t request_size,
                                  void* response,
                                  size_t response_size,
                                  uint32_t timeout_ms) {
    if (!ctx || !request || !response) return -1;

    bio_module_entry_t* entry = ctx->entry;
    if (!entry || entry->magic != BIO_MODULE_MAGIC) return -1;

    // Create a promise for the response
    nimcp_bio_promise_t promise = nimcp_bio_promise_create(BIO_CHANNEL_DOPAMINE, response_size);
    if (!promise) {
        LOG_ERROR("Failed to create promise for request");
        return -1;
    }

    // Send request with promise
    nimcp_error_t send_result = bio_router_send_with_promise(ctx, request, request_size,
                                                               promise, timeout_ms);
    if (send_result != NIMCP_SUCCESS) {
        nimcp_bio_promise_destroy(promise);
        LOG_ERROR("Failed to send request");
        return send_result;
    }

    // Get future from promise
    nimcp_bio_future_t future = nimcp_bio_promise_get_future(promise);
    if (!future) {
        nimcp_bio_promise_destroy(promise);
        LOG_ERROR("Failed to get future from promise");
        return -1;
    }

    // Wait for response with timeout
    nimcp_error_t wait_result = nimcp_bio_future_wait(future, response,
                                                        timeout_ms > 0 ? timeout_ms : DEFAULT_TIMEOUT_MS);

    // Cleanup
    nimcp_bio_promise_destroy(promise);

    if (wait_result != NIMCP_SUCCESS) {
        LOG_DEBUG("Request timed out or failed");

        // Update timeout statistics
        nimcp_platform_mutex_lock(&g_router->stats_mutex);
        g_router->stats.timeouts++;
        nimcp_platform_mutex_unlock(&g_router->stats_mutex);

        return wait_result;
    }

    LOG_TRACE("Request completed successfully");
    return NIMCP_SUCCESS;
}

nimcp_error_t bio_router_broadcast(bio_module_context_t ctx,
                                    const void* msg,
                                    size_t msg_size) {
    if (!ctx || !msg || msg_size == 0) return -1;

    // Create a copy with target set to 0 (broadcast)
    bio_message_header_t* header = (bio_message_header_t*)msg;
    uint32_t original_target = header->target_module;
    header->target_module = 0;
    header->flags |= BIO_MSG_FLAG_BROADCAST;

    nimcp_error_t result = bio_router_send(ctx, msg, msg_size, 0);

    // Restore original target
    header->target_module = original_target;
    header->flags &= ~BIO_MSG_FLAG_BROADCAST;

    return result;
}

/*=============================================================================
 * MESSAGE RECEIVING
 *============================================================================*/

/**
 * WHAT: Find handler for message type
 * WHY:  Dispatch message to appropriate handler
 * HOW:  Search handlers for exact match or category match
 */
static bio_message_handler_t bio_router_find_handler(bio_module_entry_t* entry,
                                                       bio_message_type_t msg_type) {
    nimcp_platform_mutex_lock(&entry->handler_mutex);

    // First try exact match
    for (uint32_t i = 0; i < entry->handler_count; i++) {
        if (!entry->handlers[i].is_category_handler &&
            entry->handlers[i].msg_type == msg_type) {
            bio_message_handler_t handler = entry->handlers[i].handler;
            nimcp_platform_mutex_unlock(&entry->handler_mutex);
            return handler;
        }
    }

    // Try category match
    for (uint32_t i = 0; i < entry->handler_count; i++) {
        if (entry->handlers[i].is_category_handler) {
            uint32_t masked_type = msg_type & entry->handlers[i].category_mask;
            uint32_t masked_handler = entry->handlers[i].msg_type & entry->handlers[i].category_mask;
            if (masked_type == masked_handler) {
                bio_message_handler_t handler = entry->handlers[i].handler;
                nimcp_platform_mutex_unlock(&entry->handler_mutex);
                return handler;
            }
        }
    }

    nimcp_platform_mutex_unlock(&entry->handler_mutex);
    return NULL;
}

uint32_t bio_router_inbox_count(bio_module_context_t ctx) {
    if (!ctx || ctx->magic != BIO_MODULE_MAGIC) return 0;

    bio_module_entry_t* entry = ctx->entry;
    if (!entry || entry->magic != BIO_MODULE_MAGIC) return 0;

    nimcp_platform_mutex_lock(&entry->inbox.mutex);
    uint32_t count = entry->inbox.count;
    nimcp_platform_mutex_unlock(&entry->inbox.mutex);

    return count;
}

uint32_t bio_router_process_inbox(bio_module_context_t ctx, uint32_t max_messages) {
    if (!ctx || ctx->magic != BIO_MODULE_MAGIC) return 0;

    bio_module_entry_t* entry = ctx->entry;
    if (!entry || entry->magic != BIO_MODULE_MAGIC) return 0;

    uint32_t processed = 0;
    uint32_t limit = max_messages > 0 ? max_messages : UINT32_MAX;

    while (processed < limit) {
        void* msg_data = NULL;
        size_t msg_size = 0;
        nimcp_bio_promise_t response_promise = NULL;

        // Dequeue next message (non-blocking)
        nimcp_error_t deq_result = bio_msg_queue_dequeue(&entry->inbox,
                                                          &msg_data,
                                                          &msg_size,
                                                          &response_promise,
                                                          0);
        if (deq_result != NIMCP_SUCCESS || !msg_data) {
            break;  // No more messages
        }

        const bio_message_header_t* header = (const bio_message_header_t*)msg_data;

        // Find handler
        bio_message_handler_t handler = bio_router_find_handler(entry, header->type);

        if (handler) {
            // Invoke handler
            entry->handler_invocations++;

            nimcp_error_t handler_result = handler(msg_data, msg_size,
                                                    response_promise,
                                                    entry->user_data);

            if (handler_result != NIMCP_SUCCESS) {
                entry->handler_errors++;

                nimcp_platform_mutex_lock(&g_router->stats_mutex);
                g_router->stats.handler_errors++;
                nimcp_platform_mutex_unlock(&g_router->stats_mutex);

                LOG_WARN("Handler error for message type 0x%04X in module %s",
                         header->type, entry->module_name);
            }
        } else {
            LOG_DEBUG("No handler for message type 0x%04X in module %s",
                      header->type, entry->module_name);
        }

        // Free message data
        free(msg_data);

        processed++;
    }

    return processed;
}

/*=============================================================================
 * PREDICTIVE CODING INTEGRATION
 *============================================================================*/

/**
 * @brief Signal observer entry for predictive coding
 */
typedef struct {
    char signal_name[64];
    float prediction;
    float precision;
    bio_prediction_observer_t callback;
    void* user_data;
    bool active;
} signal_observer_t;

/** Global signal observer registry (simplified) */
static signal_observer_t g_signal_observers[256];
static uint32_t g_signal_observer_count = 0;
static nimcp_platform_mutex_t g_signal_mutex;
static bool g_signal_mutex_initialized = false;

nimcp_error_t bio_router_observe_signal(bio_module_context_t ctx,
                                         const char* signal_name,
                                         float initial_prediction,
                                         float precision,
                                         bio_prediction_observer_t callback) {
    if (!ctx || !signal_name || !callback) return -1;

    // Initialize mutex on first use
    if (!g_signal_mutex_initialized) {
        nimcp_platform_mutex_init(&g_signal_mutex, false);
        g_signal_mutex_initialized = true;
    }

    nimcp_platform_mutex_lock(&g_signal_mutex);

    if (g_signal_observer_count >= 256) {
        nimcp_platform_mutex_unlock(&g_signal_mutex);
        LOG_ERROR("Signal observer registry full");
        return -1;
    }

    // Register observer
    signal_observer_t* obs = &g_signal_observers[g_signal_observer_count];
    strncpy(obs->signal_name, signal_name, sizeof(obs->signal_name) - 1);
    obs->signal_name[sizeof(obs->signal_name) - 1] = '\0';
    obs->prediction = initial_prediction;
    obs->precision = precision;
    obs->callback = callback;
    obs->user_data = bio_module_context_get_user_data(ctx);
    obs->active = true;

    g_signal_observer_count++;

    nimcp_platform_mutex_unlock(&g_signal_mutex);

    LOG_DEBUG("Registered observer for signal '%s' (prediction=%.3f, precision=%.3f)",
              signal_name, initial_prediction, precision);

    return NIMCP_SUCCESS;
}

nimcp_error_t bio_router_publish_signal(bio_module_context_t ctx,
                                         const char* signal_name,
                                         float value) {
    if (!ctx || !signal_name) return -1;

    if (!g_signal_mutex_initialized) {
        // No observers registered yet
        return NIMCP_SUCCESS;
    }

    nimcp_platform_mutex_lock(&g_signal_mutex);

    // Notify all observers of this signal
    uint32_t notified_count = 0;
    for (uint32_t i = 0; i < g_signal_observer_count; i++) {
        signal_observer_t* obs = &g_signal_observers[i];

        if (obs->active && strcmp(obs->signal_name, signal_name) == 0) {
            // Compute prediction error
            float error = value - obs->prediction;

            // Update prediction with learning (simple exponential moving average)
            obs->prediction = obs->prediction + 0.1f * error;

            // Call observer callback
            obs->callback(signal_name, value, obs->user_data);

            notified_count++;
        }
    }

    nimcp_platform_mutex_unlock(&g_signal_mutex);

    if (notified_count > 0) {
        LOG_TRACE("Published signal '%s' value=%.3f (notified %u observers)",
                  signal_name, value, notified_count);
    }

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PHASE SYNCHRONIZATION
 *============================================================================*/

/**
 * @brief Phase sync context (simplified implementation)
 */
typedef struct {
    nimcp_oscillation_band_t band;
    nimcp_bio_promise_t* promises;
    uint32_t promise_count;
    uint32_t completed_count;
    nimcp_platform_mutex_t mutex;
    nimcp_platform_cond_t cond;
    bool all_ready;
} phase_sync_context_t;

nimcp_phase_sync_t bio_router_sync_request(bio_module_context_t ctx,
                                            nimcp_oscillation_band_t band,
                                            const bio_module_id_t* targets,
                                            size_t target_count,
                                            const void* request,
                                            size_t request_size) {
    if (!ctx || !targets || target_count == 0 || !request || request_size == 0) {
        LOG_ERROR("Invalid parameters for sync request");
        return NULL;
    }

    if (!g_router) {
        LOG_ERROR("Router not initialized");
        return NULL;
    }

    // Allocate sync context
    phase_sync_context_t* sync_ctx = nimcp_calloc(1, sizeof(phase_sync_context_t));
    if (!sync_ctx) {
        LOG_ERROR("Failed to allocate phase sync context");
        return NULL;
    }

    sync_ctx->band = band;
    sync_ctx->promise_count = (uint32_t)target_count;
    sync_ctx->completed_count = 0;
    sync_ctx->all_ready = false;

    // Allocate promise array
    sync_ctx->promises = nimcp_calloc(target_count, sizeof(nimcp_bio_promise_t));
    if (!sync_ctx->promises) {
        nimcp_free(sync_ctx);
        LOG_ERROR("Failed to allocate promise array");
        return NULL;
    }

    // Initialize synchronization primitives
    nimcp_platform_mutex_init(&sync_ctx->mutex, false);
    nimcp_platform_cond_init(&sync_ctx->cond);

    // Send request to all targets and collect promises
    nimcp_bio_channel_type_t channel = BIO_CHANNEL_DOPAMINE;
    for (size_t i = 0; i < target_count; i++) {
        // Create a promise for this target's response
        sync_ctx->promises[i] = nimcp_bio_promise_create(channel, 0);

        if (!sync_ctx->promises[i]) {
            LOG_WARN("Failed to create promise for target %u", targets[i]);
            continue;
        }

        // Send request with promise
        nimcp_error_t send_result = bio_router_send_with_promise(
            ctx, request, request_size, sync_ctx->promises[i], 0);

        if (send_result != NIMCP_SUCCESS) {
            LOG_WARN("Failed to send sync request to target %u", targets[i]);
        }
    }

    LOG_DEBUG("Created phase sync request for %zu targets on %s band",
              target_count,
              band == BIO_OSC_GAMMA ? "GAMMA" :
              band == BIO_OSC_BETA ? "BETA" :
              band == BIO_OSC_ALPHA ? "ALPHA" : "OTHER");

    return (nimcp_phase_sync_t)sync_ctx;
}

/*=============================================================================
 * GLIAL WAVE INTEGRATION
 *============================================================================*/

/**
 * @brief Glial wave context
 */
typedef struct {
    uint32_t wave_id;
    bio_module_id_t source_module;
    float intensity;
    float current_intensity;
    uint64_t start_time_us;
    uint8_t metadata[256];
    size_t metadata_size;
    bool active;
} glial_wave_context_t;

/**
 * @brief Wave arrival callback registration
 */
typedef struct {
    bio_module_id_t module_id;
    nimcp_wave_callback_t callback;
    void* user_data;
    bool active;
} wave_callback_entry_t;

/** Global wave state (simplified) */
static glial_wave_context_t g_waves[64];
static uint32_t g_wave_count = 0;
static uint32_t g_next_wave_id = 1;
static wave_callback_entry_t g_wave_callbacks[128];
static uint32_t g_wave_callback_count = 0;
static nimcp_platform_mutex_t g_wave_mutex;
static bool g_wave_mutex_initialized = false;

nimcp_glial_wave_t bio_router_initiate_wave(bio_module_context_t ctx,
                                             float intensity,
                                             const void* metadata) {
    if (!ctx || intensity <= 0.0f) return NULL;

    // Initialize mutex on first use
    if (!g_wave_mutex_initialized) {
        nimcp_platform_mutex_init(&g_wave_mutex, false);
        g_wave_mutex_initialized = true;
    }

    nimcp_platform_mutex_lock(&g_wave_mutex);

    if (g_wave_count >= 64) {
        nimcp_platform_mutex_unlock(&g_wave_mutex);
        LOG_ERROR("Glial wave registry full");
        return NULL;
    }

    // Create wave context
    glial_wave_context_t* wave = &g_waves[g_wave_count];
    wave->wave_id = g_next_wave_id++;
    wave->source_module = bio_module_context_get_id(ctx);
    wave->intensity = intensity;
    wave->current_intensity = intensity;
    wave->start_time_us = nimcp_platform_time_monotonic_us();
    wave->active = true;

    // Copy metadata if provided
    if (metadata) {
        wave->metadata_size = 256;  // Simplified: fixed size
        memcpy(wave->metadata, metadata, 256);
    } else {
        wave->metadata_size = 0;
    }

    g_wave_count++;

    nimcp_platform_mutex_unlock(&g_wave_mutex);

    LOG_DEBUG("Initiated glial wave %u from module %u with intensity %.3f",
              wave->wave_id, wave->source_module, intensity);

    // Notify all registered callbacks
    nimcp_platform_mutex_lock(&g_wave_mutex);

    for (uint32_t i = 0; i < g_wave_callback_count; i++) {
        wave_callback_entry_t* entry = &g_wave_callbacks[i];
        if (entry->active && entry->module_id != wave->source_module) {
            // Call the callback with current intensity
            entry->callback((nimcp_glial_wave_t)wave,
                          entry->module_id,
                          wave->current_intensity,
                          entry->user_data);
        }
    }

    nimcp_platform_mutex_unlock(&g_wave_mutex);

    return (nimcp_glial_wave_t)wave;
}

nimcp_error_t bio_router_on_wave_arrival(bio_module_context_t ctx,
                                          nimcp_wave_callback_t callback,
                                          void* user_data) {
    if (!ctx || !callback) return -1;

    // Initialize mutex on first use
    if (!g_wave_mutex_initialized) {
        nimcp_platform_mutex_init(&g_wave_mutex, false);
        g_wave_mutex_initialized = true;
    }

    nimcp_platform_mutex_lock(&g_wave_mutex);

    if (g_wave_callback_count >= 128) {
        nimcp_platform_mutex_unlock(&g_wave_mutex);
        LOG_ERROR("Wave callback registry full");
        return -1;
    }

    // Register callback
    wave_callback_entry_t* entry = &g_wave_callbacks[g_wave_callback_count];
    entry->module_id = bio_module_context_get_id(ctx);
    entry->callback = callback;
    entry->user_data = user_data;
    entry->active = true;

    g_wave_callback_count++;

    nimcp_platform_mutex_unlock(&g_wave_mutex);

    LOG_DEBUG("Registered wave arrival callback for module %u", entry->module_id);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * MODULE CONTEXT ACCESSORS
 *============================================================================*/

bio_module_id_t bio_module_context_get_id(bio_module_context_t ctx) {
    if (!ctx || ctx->magic != BIO_MODULE_MAGIC) return BIO_MODULE_UNKNOWN;

    bio_module_entry_t* entry = ctx->entry;
    if (!entry || entry->magic != BIO_MODULE_MAGIC) return BIO_MODULE_UNKNOWN;

    return entry->module_id;
}

const char* bio_module_context_get_name(bio_module_context_t ctx) {
    if (!ctx || ctx->magic != BIO_MODULE_MAGIC) return "unknown";

    bio_module_entry_t* entry = ctx->entry;
    if (!entry || entry->magic != BIO_MODULE_MAGIC) return "unknown";

    return entry->module_name;
}

void* bio_module_context_get_user_data(bio_module_context_t ctx) {
    if (!ctx || ctx->magic != BIO_MODULE_MAGIC) return NULL;

    bio_module_entry_t* entry = ctx->entry;
    if (!entry || entry->magic != BIO_MODULE_MAGIC) return NULL;

    return entry->user_data;
}

/*=============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

const char* bio_msg_type_name(bio_message_type_t type) {
    switch (type) {
        // Brain messages
        case BIO_MSG_BRAIN_STATE_QUERY: return "BRAIN_STATE_QUERY";
        case BIO_MSG_BRAIN_STATE_RESPONSE: return "BRAIN_STATE_RESPONSE";
        case BIO_MSG_NEURON_ACTIVATION_REQUEST: return "NEURON_ACTIVATION_REQUEST";
        case BIO_MSG_NEURON_ACTIVATION_RESPONSE: return "NEURON_ACTIVATION_RESPONSE";

        // Plasticity messages
        case BIO_MSG_WEIGHT_UPDATE_REQUEST: return "WEIGHT_UPDATE_REQUEST";
        case BIO_MSG_WEIGHT_UPDATE_RESPONSE: return "WEIGHT_UPDATE_RESPONSE";
        case BIO_MSG_STDP_EVENT: return "STDP_EVENT";
        case BIO_MSG_NEUROMODULATOR_RELEASE: return "NEUROMODULATOR_RELEASE";

        // Cognitive messages
        case BIO_MSG_INTROSPECTION_QUERY: return "INTROSPECTION_QUERY";
        case BIO_MSG_ETHICS_EVALUATION_REQUEST: return "ETHICS_EVALUATION_REQUEST";
        case BIO_MSG_SALIENCE_QUERY: return "SALIENCE_QUERY";
        case BIO_MSG_ATTENTION_SHIFT: return "ATTENTION_SHIFT";

        // System messages
        case BIO_MSG_HEALTH_CHECK: return "HEALTH_CHECK";
        case BIO_MSG_ERROR_REPORT: return "ERROR_REPORT";

        default:
            if (type >= 0x0100 && type < 0x0200) return "BRAIN_MSG";
            if (type >= 0x0200 && type < 0x0300) return "PLASTICITY_MSG";
            if (type >= 0x0300 && type < 0x0400) return "COGNITIVE_MSG";
            if (type >= 0x0400 && type < 0x0500) return "GLIAL_MSG";
            if (type >= 0x0500 && type < 0x0600) return "MIDDLEWARE_MSG";
            if (type >= 0x0600 && type < 0x0700) return "TRAINING_MSG";
            return "UNKNOWN";
    }
}

const char* bio_module_name(bio_module_id_t module) {
    switch (module) {
        case BIO_MODULE_BRAIN: return "BRAIN";
        case BIO_MODULE_INTROSPECTION: return "INTROSPECTION";
        case BIO_MODULE_ETHICS: return "ETHICS";
        case BIO_MODULE_SALIENCE: return "SALIENCE";
        case BIO_MODULE_ATTENTION: return "ATTENTION";
        case BIO_MODULE_STDP: return "STDP";
        case BIO_MODULE_ASTROCYTE: return "ASTROCYTE";
        case BIO_MODULE_PIPELINE: return "PIPELINE";
        case BIO_MODULE_TRAINING: return "TRAINING";
        case BIO_MODULE_SYSTEM: return "SYSTEM";
        default: return "UNKNOWN";
    }
}
