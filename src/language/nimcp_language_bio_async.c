//=============================================================================
// nimcp_language_bio_async.c - Language Layer Bio-Async Integration
//=============================================================================
/**
 * @file nimcp_language_bio_async.c
 * @brief Bio-async messaging implementation for Language Layer
 *
 * Implements message handlers, registration, and message sending for
 * event-driven language processing through the bio-async system.
 *
 * @version 1.0.0 - Phase L5: Orchestrator Implementation
 * @author NIMCP Development Team
 * @date 2026-01-05
 */

/* Include bio_router.h FIRST to get correct pointer types */
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

#include "language/nimcp_language_bio_async.h"

/* Define error codes if not available (NIMCP_SUCCESS is in bio_async.h) */
#ifndef NIMCP_ERROR_INVALID_ARGUMENT
#define NIMCP_ERROR_INVALID_ARGUMENT -1
#endif
#include "language/nimcp_language_orchestrator.h"
#include <stdlib.h>
#include <string.h>

//=============================================================================
// Internal Constants
//=============================================================================

#define LANGUAGE_BIO_ASYNC_MAX_HANDLERS     16
#define LANGUAGE_BIO_ASYNC_INBOX_CAPACITY   64

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Message handler entry
 */
typedef struct {
    uint32_t message_type;
    language_message_handler_t handler;
    bool active;
} handler_entry_t;

/**
 * @brief Bio-async context for language layer
 */
typedef struct {
    language_orchestrator_t* orchestrator;
    bio_router_t router;
    bio_module_context_t module_ctx;  /**< Module context for bio_router */
    bool registered;

    /* Message handlers */
    handler_entry_t handlers[LANGUAGE_BIO_ASYNC_MAX_HANDLERS];
    uint32_t num_handlers;

    /* Statistics */
    uint64_t messages_sent;
    uint64_t messages_received;
} language_bio_async_ctx_t;

/* Global context (one per orchestrator would require internal storage) */
static language_bio_async_ctx_t* g_bio_async_ctx = NULL;

//=============================================================================
// Forward Declarations
//=============================================================================

static nimcp_error_t language_bio_async_message_callback(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data
);

//=============================================================================
// Registration API
//=============================================================================

int language_bio_async_register(
    language_orchestrator_t* orchestrator,
    bio_router_t router)
{
    if (!orchestrator) {
        return -1;
    }

    /* Allocate context if needed */
    if (!g_bio_async_ctx) {
        g_bio_async_ctx = calloc(1, sizeof(language_bio_async_ctx_t));
        if (!g_bio_async_ctx) {
            return -1;
        }
    }

    g_bio_async_ctx->orchestrator = orchestrator;
    g_bio_async_ctx->router = router;
    g_bio_async_ctx->messages_sent = 0;
    g_bio_async_ctx->messages_received = 0;
    g_bio_async_ctx->module_ctx = NULL;

    /* Register with bio router if available */
    if (router && bio_router_is_initialized()) {
        bio_module_info_t module_info = {
            .module_id = BIO_MODULE_LANGUAGE_LAYER,
            .module_name = "language_layer",
            .inbox_capacity = LANGUAGE_BIO_ASYNC_INBOX_CAPACITY,
            .user_data = orchestrator
        };

        g_bio_async_ctx->module_ctx = bio_router_register_module(&module_info);
        if (g_bio_async_ctx->module_ctx) {
            /* Register handlers for all language message types */
            for (uint32_t msg_type = BIO_MSG_LANG_UTTERANCE_START;
                 msg_type < BIO_MSG_LANG_COUNT; msg_type++) {
                bio_router_register_handler(
                    g_bio_async_ctx->module_ctx,
                    msg_type,
                    language_bio_async_message_callback
                );
            }
        }
    }

    g_bio_async_ctx->registered = true;
    return 0;
}

int language_bio_async_unregister(language_orchestrator_t* orchestrator)
{
    if (!orchestrator || !g_bio_async_ctx) {
        return -1;
    }

    if (g_bio_async_ctx->orchestrator != orchestrator) {
        return -1;
    }

    /* Unregister from router */
    if (g_bio_async_ctx->module_ctx) {
        bio_router_unregister_module(g_bio_async_ctx->module_ctx);
        g_bio_async_ctx->module_ctx = NULL;
    }

    g_bio_async_ctx->registered = false;
    g_bio_async_ctx->orchestrator = NULL;
    g_bio_async_ctx->router = NULL;

    return 0;
}

bool language_bio_async_is_registered(const language_orchestrator_t* orchestrator)
{
    if (!orchestrator || !g_bio_async_ctx) {
        return false;
    }
    return g_bio_async_ctx->registered &&
           g_bio_async_ctx->orchestrator == orchestrator;
}

bio_module_context_t language_bio_async_get_context(
    const language_orchestrator_t* orchestrator)
{
    if (!orchestrator || !g_bio_async_ctx) {
        return NULL;
    }
    if (g_bio_async_ctx->orchestrator != orchestrator) {
        return NULL;
    }
    return g_bio_async_ctx->module_ctx;
}

//=============================================================================
// Message Sending API
//=============================================================================

int language_bio_async_send_phoneme(
    language_orchestrator_t* orchestrator,
    const language_msg_phoneme_t* phoneme)
{
    if (!orchestrator || !phoneme) {
        return -1;
    }

    if (!g_bio_async_ctx || !g_bio_async_ctx->registered) {
        return -1;
    }

    /* Send through bio_router if module context is available */
    if (g_bio_async_ctx->module_ctx) {
        /* Create message header + payload */
        struct {
            bio_message_header_t header;
            language_msg_phoneme_t payload;
        } msg;

        memset(&msg.header, 0, sizeof(msg.header));
        msg.header.type = BIO_MSG_LANG_PHONEME_RECOGNIZED;
        msg.header.source_module = BIO_MODULE_LANGUAGE_LAYER;
        msg.header.target_module = BIO_MODULE_UNKNOWN;  /* Broadcast */
        msg.header.payload_size = sizeof(language_msg_phoneme_t);
        msg.header.timestamp_us = 0;  /* Router will fill */
        msg.payload = *phoneme;

        nimcp_error_t err = bio_router_send(
            g_bio_async_ctx->module_ctx,
            &msg,
            sizeof(msg),
            0  /* Default timeout */
        );
        if (err != NIMCP_SUCCESS) {
            return -1;
        }
    }

    g_bio_async_ctx->messages_sent++;
    return 0;
}

int language_bio_async_send_word(
    language_orchestrator_t* orchestrator,
    const language_msg_word_t* word)
{
    if (!orchestrator || !word) {
        return -1;
    }

    if (!g_bio_async_ctx || !g_bio_async_ctx->registered) {
        return -1;
    }

    if (g_bio_async_ctx->module_ctx) {
        struct {
            bio_message_header_t header;
            language_msg_word_t payload;
        } msg;

        memset(&msg.header, 0, sizeof(msg.header));
        msg.header.type = BIO_MSG_LANG_WORD_RECOGNIZED;
        msg.header.source_module = BIO_MODULE_LANGUAGE_LAYER;
        msg.header.target_module = BIO_MODULE_UNKNOWN;
        msg.header.payload_size = sizeof(language_msg_word_t);
        msg.header.timestamp_us = 0;
        msg.payload = *word;

        if (bio_router_send(g_bio_async_ctx->module_ctx, &msg, sizeof(msg), 0) != NIMCP_SUCCESS) {
            return -1;
        }
    }

    g_bio_async_ctx->messages_sent++;
    return 0;
}

int language_bio_async_send_concept(
    language_orchestrator_t* orchestrator,
    const language_msg_concept_t* concept_msg)
{
    if (!orchestrator || !concept_msg) {
        return -1;
    }

    if (!g_bio_async_ctx || !g_bio_async_ctx->registered) {
        return -1;
    }

    if (g_bio_async_ctx->module_ctx) {
        struct {
            bio_message_header_t header;
            language_msg_concept_t payload;
        } msg;

        memset(&msg.header, 0, sizeof(msg.header));
        msg.header.type = BIO_MSG_LANG_CONCEPT_ACTIVATED;
        msg.header.source_module = BIO_MODULE_LANGUAGE_LAYER;
        msg.header.target_module = BIO_MODULE_UNKNOWN;
        msg.header.payload_size = sizeof(language_msg_concept_t);
        msg.header.timestamp_us = 0;
        msg.payload = *concept_msg;

        if (bio_router_send(g_bio_async_ctx->module_ctx, &msg, sizeof(msg), 0) != NIMCP_SUCCESS) {
            return -1;
        }
    }

    g_bio_async_ctx->messages_sent++;
    return 0;
}

int language_bio_async_send_comprehension(
    language_orchestrator_t* orchestrator,
    const language_msg_comprehension_t* result)
{
    if (!orchestrator || !result) {
        return -1;
    }

    if (!g_bio_async_ctx || !g_bio_async_ctx->registered) {
        return -1;
    }

    if (g_bio_async_ctx->module_ctx) {
        struct {
            bio_message_header_t header;
            language_msg_comprehension_t payload;
        } msg;

        memset(&msg.header, 0, sizeof(msg.header));
        msg.header.type = BIO_MSG_LANG_COMPREHENSION_COMPLETE;
        msg.header.source_module = BIO_MODULE_LANGUAGE_LAYER;
        msg.header.target_module = BIO_MODULE_UNKNOWN;
        msg.header.payload_size = sizeof(language_msg_comprehension_t);
        msg.header.timestamp_us = 0;
        msg.payload = *result;

        if (bio_router_send(g_bio_async_ctx->module_ctx, &msg, sizeof(msg), 0) != NIMCP_SUCCESS) {
            return -1;
        }
    }

    g_bio_async_ctx->messages_sent++;
    return 0;
}

int language_bio_async_send_production(
    language_orchestrator_t* orchestrator,
    const language_msg_production_t* result)
{
    if (!orchestrator || !result) {
        return -1;
    }

    if (!g_bio_async_ctx || !g_bio_async_ctx->registered) {
        return -1;
    }

    if (g_bio_async_ctx->module_ctx) {
        struct {
            bio_message_header_t header;
            language_msg_production_t payload;
        } msg;

        memset(&msg.header, 0, sizeof(msg.header));
        msg.header.type = BIO_MSG_LANG_PRODUCTION_COMPLETE;
        msg.header.source_module = BIO_MODULE_LANGUAGE_LAYER;
        msg.header.target_module = BIO_MODULE_UNKNOWN;
        msg.header.payload_size = sizeof(language_msg_production_t);
        msg.header.timestamp_us = 0;
        msg.payload = *result;

        if (bio_router_send(g_bio_async_ctx->module_ctx, &msg, sizeof(msg), 0) != NIMCP_SUCCESS) {
            return -1;
        }
    }

    g_bio_async_ctx->messages_sent++;
    return 0;
}

int language_bio_async_send_anomaly(
    language_orchestrator_t* orchestrator,
    const language_msg_anomaly_t* anomaly)
{
    if (!orchestrator || !anomaly) {
        return -1;
    }

    if (!g_bio_async_ctx || !g_bio_async_ctx->registered) {
        return -1;
    }

    if (g_bio_async_ctx->module_ctx) {
        struct {
            bio_message_header_t header;
            language_msg_anomaly_t payload;
        } msg;

        memset(&msg.header, 0, sizeof(msg.header));
        msg.header.type = BIO_MSG_LANG_SEMANTIC_ANOMALY;
        msg.header.source_module = BIO_MODULE_LANGUAGE_LAYER;
        msg.header.target_module = BIO_MODULE_UNKNOWN;
        msg.header.payload_size = sizeof(language_msg_anomaly_t);
        msg.header.timestamp_us = 0;
        msg.payload = *anomaly;

        if (bio_router_send(g_bio_async_ctx->module_ctx, &msg, sizeof(msg), 0) != NIMCP_SUCCESS) {
            return -1;
        }
    }

    g_bio_async_ctx->messages_sent++;
    return 0;
}

int language_bio_async_send_state_change(
    language_orchestrator_t* orchestrator,
    const language_msg_state_change_t* state_change)
{
    if (!orchestrator || !state_change) {
        return -1;
    }

    if (!g_bio_async_ctx || !g_bio_async_ctx->registered) {
        return -1;
    }

    if (g_bio_async_ctx->module_ctx) {
        struct {
            bio_message_header_t header;
            language_msg_state_change_t payload;
        } msg;

        memset(&msg.header, 0, sizeof(msg.header));
        msg.header.type = BIO_MSG_LANG_STATE_CHANGE;
        msg.header.source_module = BIO_MODULE_LANGUAGE_LAYER;
        msg.header.target_module = BIO_MODULE_UNKNOWN;
        msg.header.payload_size = sizeof(language_msg_state_change_t);
        msg.header.timestamp_us = 0;
        msg.payload = *state_change;

        if (bio_router_send(g_bio_async_ctx->module_ctx, &msg, sizeof(msg), 0) != NIMCP_SUCCESS) {
            return -1;
        }
    }

    g_bio_async_ctx->messages_sent++;
    return 0;
}

int language_bio_async_send_error(
    language_orchestrator_t* orchestrator,
    const language_msg_error_t* error)
{
    if (!orchestrator || !error) {
        return -1;
    }

    if (!g_bio_async_ctx || !g_bio_async_ctx->registered) {
        return -1;
    }

    if (g_bio_async_ctx->module_ctx) {
        struct {
            bio_message_header_t header;
            language_msg_error_t payload;
        } msg;

        memset(&msg.header, 0, sizeof(msg.header));
        msg.header.type = BIO_MSG_LANG_ERROR;
        msg.header.source_module = BIO_MODULE_LANGUAGE_LAYER;
        msg.header.target_module = BIO_MODULE_UNKNOWN;
        msg.header.payload_size = sizeof(language_msg_error_t);
        msg.header.timestamp_us = 0;
        msg.payload = *error;

        if (bio_router_send(g_bio_async_ctx->module_ctx, &msg, sizeof(msg), 0) != NIMCP_SUCCESS) {
            return -1;
        }
    }

    g_bio_async_ctx->messages_sent++;
    return 0;
}

//=============================================================================
// Message Handler Registration
//=============================================================================

int language_bio_async_register_handler(
    language_orchestrator_t* orchestrator,
    uint32_t message_type,
    language_message_handler_t handler)
{
    if (!orchestrator || !handler) {
        return -1;
    }

    if (!g_bio_async_ctx) {
        return -1;
    }

    /* Find empty slot */
    for (uint32_t i = 0; i < LANGUAGE_BIO_ASYNC_MAX_HANDLERS; i++) {
        if (!g_bio_async_ctx->handlers[i].active) {
            g_bio_async_ctx->handlers[i].message_type = message_type;
            g_bio_async_ctx->handlers[i].handler = handler;
            g_bio_async_ctx->handlers[i].active = true;
            g_bio_async_ctx->num_handlers++;
            return 0;
        }
    }

    return -1;  /* No space */
}

//=============================================================================
// Message Processing
//=============================================================================

int language_bio_async_process_messages(language_orchestrator_t* orchestrator)
{
    if (!orchestrator) {
        return -1;
    }

    if (!g_bio_async_ctx || !g_bio_async_ctx->registered) {
        return 0;
    }

    int processed = 0;

    /* Process pending messages from bio_router if module context available */
    if (g_bio_async_ctx->module_ctx) {
        /* Process all pending messages - handlers invoked via callback */
        processed = (int)bio_router_process_inbox(g_bio_async_ctx->module_ctx, 0);
    }

    return processed;
}

uint32_t language_bio_async_pending_count(const language_orchestrator_t* orchestrator)
{
    if (!orchestrator || !g_bio_async_ctx) {
        return 0;
    }

    /* Query pending message count from bio_router if available */
    if (g_bio_async_ctx->module_ctx) {
        return bio_router_inbox_count(g_bio_async_ctx->module_ctx);
    }
    return 0;
}

//=============================================================================
// Internal Functions
//=============================================================================

/**
 * @brief Internal callback invoked by bio_router for incoming messages
 *
 * This matches the bio_message_handler_t signature from bio_router.h
 */
static nimcp_error_t language_bio_async_message_callback(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    (void)msg_size;
    (void)response_promise;

    language_orchestrator_t* orchestrator = (language_orchestrator_t*)user_data;

    if (!orchestrator || !g_bio_async_ctx || !msg) {
        return NIMCP_ERROR_INVALID_ARGUMENT;
    }

    /* Extract message type from header */
    const bio_message_header_t* header = (const bio_message_header_t*)msg;
    uint32_t message_type = header->type;
    const void* payload = (const uint8_t*)msg + sizeof(bio_message_header_t);
    uint32_t payload_size = header->payload_size;

    g_bio_async_ctx->messages_received++;

    /* Find and invoke registered handler */
    for (uint32_t i = 0; i < LANGUAGE_BIO_ASYNC_MAX_HANDLERS; i++) {
        if (g_bio_async_ctx->handlers[i].active &&
            g_bio_async_ctx->handlers[i].message_type == message_type) {
            g_bio_async_ctx->handlers[i].handler(
                orchestrator,
                message_type,
                payload,
                payload_size
            );
            return NIMCP_SUCCESS;
        }
    }

    /* No handler found - message ignored */
    return NIMCP_SUCCESS;
}
