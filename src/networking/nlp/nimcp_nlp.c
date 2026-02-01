#include <stddef.h>  /* for NULL */
//=============================================================================
// nimcp_nlp.c - Neural Link Protocol Core Implementation
//=============================================================================
/**
 * @file nimcp_nlp.c
 * @brief Core implementation of the Neural Link Protocol
 *
 * WHAT: Main NLP node implementation with lifecycle, messaging, and mode control
 * WHY:  Enable secure, resilient brain-to-brain communication across devices
 * HOW:  Unified interface with mode-specific behavior (Standard/Tactical/Stealth)
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 */

#include "networking/nlp/nimcp_neural_link_protocol.h"
#include "networking/nlp/nimcp_nlp_internal.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/platform/nimcp_platform_time.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "utils/validation/nimcp_common.h"
#include "utils/rng/nimcp_rand.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_wiring_helpers.h"
#include "security/nimcp_bbb_helpers.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for nlp module */
static nimcp_health_agent_t* g_nlp_health_agent = NULL;

/**
 * @brief Set health agent for nlp heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void nlp_set_health_agent(nimcp_health_agent_t* agent) {
    g_nlp_health_agent = agent;
}

/** @brief Send heartbeat from nlp module */
static inline void nlp_heartbeat(const char* operation, float progress) {
    if (g_nlp_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_nlp_health_agent, operation, progress);
    }
}


//=============================================================================
// Module Registration
//=============================================================================

#define NLP_MODULE_NAME "neural_link_protocol"

static bool g_nlp_initialized = false;
static nimcp_mutex_t g_nlp_global_mutex;

// Node structure is defined in nimcp_nlp_internal.h

//=============================================================================
// Forward Declarations
//=============================================================================

static void* nlp_recv_thread(void* arg);
static void* nlp_heartbeat_thread(void* arg);
static void* nlp_stealth_thread(void* arg);
static int nlp_process_message(nlp_node_t node, const uint8_t* data, size_t len,
                               const struct sockaddr_in* from);
static int nlp_send_raw(nlp_node_t node, uint32_t peer_id, const uint8_t* data, size_t len);
static void nlp_auto_mode_check(nlp_node_t node);

/**
 * @brief Thread-safe check if threads should continue running
 * @param node NLP node
 * @return true if threads should continue, false if shutdown requested
 */
static inline bool nlp_should_continue(nlp_node_t node) {
    nimcp_mutex_lock(&node->state_mutex);
    bool running = node->threads_running;
    nimcp_mutex_unlock(&node->state_mutex);
    return running;
}

/**
 * @brief Thread-safe check if node is in stealth mode and running
 * @param node NLP node
 * @return true if in stealth mode and running
 */
static inline bool nlp_stealth_active(nlp_node_t node) {
    nimcp_mutex_lock(&node->state_mutex);
    bool active = node->threads_running && node->current_mode == NLP_MODE_STEALTH;
    nimcp_mutex_unlock(&node->state_mutex);
    return active;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

static bio_module_context_t g_nlp_bio_ctx = NULL;

//=============================================================================
// KG-Driven Wiring Callback (Phase 2: KG-Based Runtime Module Assembly)
//=============================================================================

/**
 * @brief KG-driven wiring handler callback for NLP module
 *
 * WHAT: Register message handlers based on discovered wiring from KG
 * WHY:  Enables runtime assembly - module discovers its handlers from KG
 * HOW:  Orchestrator invokes this with message types from HANDLES_MESSAGE relations
 *
 * @param ctx Bio-async module context
 * @param message_types Array of message types to handle (from KG)
 * @param message_count Number of message types
 * @param user_data NLP node pointer
 * @return 0 on success, -1 on error
 */
static int nlp_wiring_handler_callback(
    bio_module_context_t ctx,
    const bio_message_type_t* message_types,
    uint32_t message_count,
    void* user_data
);

/* Forward declaration for bio handler */
static nimcp_error_t nlp_bio_handler(const void* msg, size_t msg_size,
                                      nimcp_bio_promise_t response_promise,
                                      void* user_data);

/**
 * @brief Broadcast NLP session event to cognitive modules
 */
void nlp_broadcast_session_event(nlp_node_t node, uint64_t peer_id,
                                  bio_message_type_t event_type,
                                  uint8_t old_state, uint8_t new_state) {
    if (!g_nlp_bio_ctx) return;

    bio_msg_nlp_session_state_change_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, event_type, BIO_MODULE_NLP,
                        BIO_MODULE_ALL, sizeof(msg));
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.channel = BIO_CHANNEL_NOREPINEPHRINE;  // Alerting channel
    msg.peer_id = peer_id;
    msg.session_id = (node ? node->brain_id : 0);
    msg.old_state = old_state;
    msg.new_state = new_state;
    msg.state_change_time_us = nimcp_time_get_us();

    bio_router_broadcast(g_nlp_bio_ctx, &msg, sizeof(msg));

    NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
        "Bio-async: broadcast session event peer=%lu state=%u->%u",
        (unsigned long)peer_id, old_state, new_state);
}

/**
 * @brief Broadcast NLP message received event
 */
void nlp_broadcast_message_received(nlp_node_t node, uint64_t peer_id,
                                     uint32_t msg_type, uint32_t msg_size,
                                     bool encrypted, bool compressed) {
    if (!g_nlp_bio_ctx) return;

    bio_msg_nlp_message_received_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_NLP_MESSAGE_RECEIVED,
                        BIO_MODULE_NLP, BIO_MODULE_ALL, sizeof(msg));
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.channel = BIO_CHANNEL_ACETYLCHOLINE;  // Fast signaling
    msg.peer_id = peer_id;
    msg.message_type = msg_type;
    msg.message_size = msg_size;
    msg.sequence_num = 0;  // Sequence from NLP message header, not tracked in node
    msg.encrypted = encrypted;
    msg.compressed = compressed;

    bio_router_broadcast(g_nlp_bio_ctx, &msg, sizeof(msg));
}

/**
 * @brief Broadcast NLP mode change event
 */
void nlp_broadcast_mode_change(nlp_node_t node, uint8_t old_mode,
                                uint8_t new_mode, uint32_t reason) {
    if (!g_nlp_bio_ctx) return;

    bio_msg_nlp_mode_change_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_NLP_PROTOCOL_MODE_CHANGE,
                        BIO_MODULE_NLP, BIO_MODULE_ALL, sizeof(msg));
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.channel = BIO_CHANNEL_SEROTONIN;  // State change
    msg.old_mode = old_mode;
    msg.new_mode = new_mode;
    msg.peer_id = 0;  // Applies to all
    msg.reason = reason;

    bio_router_broadcast(g_nlp_bio_ctx, &msg, sizeof(msg));

    bbb_audit_log(BBB_AUDIT_INFO, NLP_MODULE_NAME, "mode_change_broadcast",
                  "old=%u new=%u reason=%u", old_mode, new_mode, reason);
}

/**
 * @brief Broadcast NLP error event
 */
void nlp_broadcast_error(uint64_t peer_id, uint32_t error_code,
                          uint8_t severity, const char* message) {
    if (!g_nlp_bio_ctx) return;

    bio_msg_nlp_error_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_NLP_ERROR,
                        BIO_MODULE_NLP, BIO_MODULE_ALL, sizeof(msg));
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.channel = BIO_CHANNEL_NOREPINEPHRINE;  // Alert
    msg.peer_id = peer_id;
    msg.error_code = error_code;
    msg.severity = severity;
    strncpy(msg.module_name, NLP_MODULE_NAME, sizeof(msg.module_name) - 1);
    if (message) {
        strncpy(msg.error_message, message, sizeof(msg.error_message) - 1);
    }

    bio_router_broadcast(g_nlp_bio_ctx, &msg, sizeof(msg));

    if (severity >= 2) {  // error or critical
        bbb_audit_log(BBB_AUDIT_WARNING, NLP_MODULE_NAME, "error_broadcast",
                      "peer=%lu code=%u severity=%u: %s",
                      (unsigned long)peer_id, error_code, severity,
                      message ? message : "");
    }
}

/**
 * @brief Handle incoming bio-async messages for NLP
 */
static nimcp_error_t nlp_bio_handler(const void* msg, size_t msg_size,
                                      nimcp_bio_promise_t response_promise,
                                      void* user_data) {
    if (!msg || msg_size < sizeof(bio_message_header_t)) {
        return NIMCP_BIO_ERROR_INVALID_CHANNEL;  // Generic error for invalid input
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;
    nlp_node_t node = (nlp_node_t)user_data;

    NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
        "Bio-async: received msg type=0x%04X from module=%u",
        header->type, header->source_module);

    switch (header->type) {
        case BIO_MSG_HEALTH_CHECK: {
            // Respond to health check
            if (response_promise && node) {
                bio_msg_health_response_t resp;
                memset(&resp, 0, sizeof(resp));
                bio_msg_init_header(&resp.header, BIO_MSG_HEALTH_RESPONSE,
                                    BIO_MODULE_NLP, header->source_module, sizeof(resp));
                resp.healthy = node->running;
                resp.active_threads = 3;  // recv, heartbeat, stealth
                resp.pending_messages = 0;  // Pending messages tracked separately
                // Complete promise with response
            }
            break;
        }

        case BIO_MSG_ATTENTION_SHIFT: {
            // Attention system wants NLP to prioritize certain traffic
            const bio_msg_attention_shift_t* shift =
                (const bio_msg_attention_shift_t*)msg;
            if (node && shift->preemptive) {
                // Could adjust message processing priority
                NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
                    "Attention shift: target=%u weight=%.2f",
                    shift->target_id, shift->attention_weight);
            }
            break;
        }

        case BIO_MSG_SECURITY_ALERT: {
            // Security system alerting about potential threat
            const bio_msg_nlp_error_t* alert =
                (const bio_msg_nlp_error_t*)msg;
            if (alert->severity >= 2) {
                NIMCP_LOGGING_WARN(NLP_MODULE_NAME,
                    "Security alert received: code=%u %s",
                    alert->error_code, alert->error_message);
                // Could trigger mode switch to Tactical
                if (node && node->config.auto_mode_switch) {
                    nlp_auto_mode_check(node);
                }
            }
            break;
        }

        case BIO_MSG_SHUTDOWN_REQUEST: {
            // Graceful shutdown request
            NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "Shutdown requested via bio-async");
            if (node) {
                nlp_node_stop(node);
            }
            break;
        }

        default:
            // Unknown message type - just log
            NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
                "Unhandled bio-async message type: 0x%04X", header->type);
            break;
    }

    (void)response_promise;  // May be unused in some cases
    return NIMCP_SUCCESS;
}

/**
 * @brief KG-driven wiring handler callback implementation
 *
 * WHAT: Register message handlers based on discovered wiring from KG
 * WHY:  Enables runtime assembly - module discovers its handlers from KG
 * HOW:  Orchestrator invokes this with message types from HANDLES_MESSAGE relations
 */
static int nlp_wiring_handler_callback(
    bio_module_context_t ctx,
    const bio_message_type_t* message_types,
    uint32_t message_count,
    void* user_data
) {
    if (!ctx || !message_types || message_count == 0) {
        return 0;  /* No handlers to register */
    }

    (void)user_data;  /* NLP node, if needed for context */

    NIMCP_LOGGING_INFO(NLP_MODULE_NAME,
        "nlp_wiring_handler_callback: registering %u handlers from KG",
        message_count);

    for (uint32_t i = 0; i < message_count; i++) {
        switch (message_types[i]) {
            case BIO_MSG_HEALTH_CHECK:
                bio_router_register_handler(ctx, message_types[i], nlp_bio_handler);
                NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
                    "  Registered handler for BIO_MSG_HEALTH_CHECK");
                break;

            case BIO_MSG_ATTENTION_SHIFT:
                bio_router_register_handler(ctx, message_types[i], nlp_bio_handler);
                NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
                    "  Registered handler for BIO_MSG_ATTENTION_SHIFT");
                break;

            case BIO_MSG_SECURITY_ALERT:
                bio_router_register_handler(ctx, message_types[i], nlp_bio_handler);
                NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
                    "  Registered handler for BIO_MSG_SECURITY_ALERT");
                break;

            case BIO_MSG_SHUTDOWN_REQUEST:
                bio_router_register_handler(ctx, message_types[i], nlp_bio_handler);
                NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
                    "  Registered handler for BIO_MSG_SHUTDOWN_REQUEST");
                break;

            default:
                NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
                    "  Unknown message type 0x%04X - skipping", message_types[i]);
                break;
        }
    }

    return 0;
}

/**
 * @brief Register NLP with bio-router
 */
static void nlp_register_bio_async(nlp_node_t node) {
    if (!bio_router_is_initialized()) {
        NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
            "Bio-router not initialized, skipping bio-async registration");
        return;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_NLP,
        .module_name = NLP_MODULE_NAME,
        .inbox_capacity = 64,
        .user_data = node
    };

    g_nlp_bio_ctx = bio_router_register_module(&info);
    if (g_nlp_bio_ctx) {
        if (node) {
            node->bio_module_ctx = g_nlp_bio_ctx;
        }

        /* KG-Driven Wiring: Register callback for orchestrator to invoke
         * When orchestrator starts, it discovers HANDLES_MESSAGE relations
         * from the KG and invokes this callback with the message types */
        nimcp_error_t cb_result = bio_router_register_wiring_callback(
            BIO_MODULE_NLP,
            (void*)nlp_wiring_handler_callback,
            node
        );

        if (cb_result != NIMCP_SUCCESS) {
            /* Fallback: Direct registration if orchestrator not available
             * This ensures backward compatibility with non-KG systems */
            LEGACY_HANDLER_REGISTRATION(
                bio_router_register_handler(g_nlp_bio_ctx, BIO_MSG_HEALTH_CHECK, nlp_bio_handler)
            );
            LEGACY_HANDLER_REGISTRATION(
                bio_router_register_handler(g_nlp_bio_ctx, BIO_MSG_ATTENTION_SHIFT, nlp_bio_handler)
            );
            LEGACY_HANDLER_REGISTRATION(
                bio_router_register_handler(g_nlp_bio_ctx, BIO_MSG_SECURITY_ALERT, nlp_bio_handler)
            );
            LEGACY_HANDLER_REGISTRATION(
                bio_router_register_handler(g_nlp_bio_ctx, BIO_MSG_SHUTDOWN_REQUEST, nlp_bio_handler)
            );

            // Also register for NLP-specific messages
            bio_router_register_category_handler(g_nlp_bio_ctx, 0x0A00, nlp_bio_handler);

            NIMCP_LOGGING_INFO(NLP_MODULE_NAME,
                "Registered with bio-router as module 0x%04X (legacy direct registration)",
                BIO_MODULE_NLP);
        } else {
            NIMCP_LOGGING_INFO(NLP_MODULE_NAME,
                "Registered with bio-router as module 0x%04X (KG-driven wiring callback)",
                BIO_MODULE_NLP);
        }

        bbb_audit_log(BBB_AUDIT_INFO, NLP_MODULE_NAME, "bio_async_registered",
                      "module_id=0x%04X", BIO_MODULE_NLP);
    }
}

/**
 * @brief Process incoming bio-async messages (call periodically)
 */
void nlp_process_bio_inbox(nlp_node_t node) {
    if (!g_nlp_bio_ctx) return;

    // Process up to 16 messages per call to avoid blocking
    uint32_t processed = bio_router_process_inbox(g_nlp_bio_ctx, 16);
    if (processed > 0) {
        NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
            "Processed %u bio-async messages", processed);
    }
    (void)node;
}

//=============================================================================
// Initialization
//=============================================================================

static bool nlp_global_init(void) {
    if (g_nlp_initialized) return true;

    if (nimcp_mutex_init(&g_nlp_global_mutex, NULL) != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Failed to init global mutex");
        return false;
    }

    // Register with BBB
    bbb_register_module(NLP_MODULE_NAME, BBB_MODULE_TYPE_NETWORK);

    // Bio-router registration (will set up handlers)
    nlp_register_bio_async(NULL);

    g_nlp_initialized = true;
    NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "Neural Link Protocol initialized");
    return true;
}

//=============================================================================
// Configuration
//=============================================================================

nlp_config_t nlp_config_default(void) {
    nlp_config_t config;
    memset(&config, 0, sizeof(config));

    config.brain_id = nlp_generate_brain_id();
    config.is_master = false;
    config.default_mode = NLP_MODE_STANDARD;
    config.auto_mode_switch = true;

    strncpy(config.bind_address, "0.0.0.0", sizeof(config.bind_address) - 1);
    config.port = 9999;
    config.max_peers = NLP_MAX_PEERS;

    config.heartbeat_interval_ms = NLP_HEARTBEAT_INTERVAL;
    config.session_timeout_ms = NLP_SESSION_TIMEOUT;
    config.handshake_timeout_ms = 5000;

    config.burst_interval_s = NLP_BURST_INTERVAL_DEFAULT;
    config.initial_emcon = NLP_EMCON_NORMAL;

    config.require_encryption = true;
    config.key_rotation_interval_s = 3600;  // 1 hour

    config.user_data = NULL;

    return config;
}

//=============================================================================
// Node Lifecycle
//=============================================================================

nlp_node_t nlp_node_create(const nlp_config_t* config) {
    if (!nlp_global_init()) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Global init failed");
        return NULL;
    }

    nlp_node_t node = (nlp_node_t)nimcp_calloc(1, sizeof(struct nlp_node_struct));
    if (!node) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Failed to allocate node");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "node is NULL");

        return NULL;
    }

    // Apply configuration
    if (config) {
        memcpy(&node->config, config, sizeof(nlp_config_t));
    } else {
        node->config = nlp_config_default();
    }

    node->magic = NLP_NODE_MAGIC;  // Set validation magic number
    node->brain_id = node->config.brain_id;
    node->is_master = node->config.is_master;
    node->current_mode = node->config.default_mode;
    node->emcon_level = node->config.initial_emcon;
    node->running = false;
    node->socket_fd = -1;

    // Copy PSK slots from config
    memcpy(node->psk_slots, node->config.psk, sizeof(node->psk_slots));

    // Initialize mutexes
    if (nimcp_mutex_init(&node->state_mutex, NULL) != NIMCP_SUCCESS ||
        nimcp_mutex_init(&node->peer_mutex, NULL) != NIMCP_SUCCESS ||
        nimcp_mutex_init(&node->key_mutex, NULL) != NIMCP_SUCCESS ||
        nimcp_mutex_init(&node->env_mutex, NULL) != NIMCP_SUCCESS ||
        nimcp_mutex_init(&node->stats_mutex, NULL) != NIMCP_SUCCESS ||
        nimcp_mutex_init(&node->seq_mutex, NULL) != NIMCP_SUCCESS ||
        nimcp_mutex_init(&node->burst_mutex, NULL) != NIMCP_SUCCESS ||
        nimcp_mutex_init(&node->queue_mutex, NULL) != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Failed to init mutexes");
        nlp_node_destroy(node);
        return NULL;
    }

    // Allocate stealth burst buffer
    node->burst_buffer = (uint8_t*)nimcp_malloc(
        NLP_STEALTH_PACKET_SIZE * 64);  // Buffer for 64 messages
    if (!node->burst_buffer) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Failed to allocate burst buffer");
        nlp_node_destroy(node);
        return NULL;
    }
    node->burst_buffer_size = NLP_STEALTH_PACKET_SIZE * 64;

    // Initialize cryptographic subsystem
    if (nlp_crypto_init(node) != 0) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Failed to initialize crypto subsystem");
        nlp_node_destroy(node);
        return NULL;
    }

    // Bio-async integration - use existing global context or register new
    if (g_nlp_bio_ctx) {
        node->bio_module_ctx = g_nlp_bio_ctx;
    } else if (bio_router_is_initialized()) {
        // Register now that we have a node context
        nlp_register_bio_async(node);
    }

    node->user_data = node->config.user_data;

    NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "Node created: brain_id=0x%08X master=%d mode=%s",
                   node->brain_id, node->is_master, nlp_mode_name(node->current_mode));

    bbb_audit_log(BBB_AUDIT_INFO, NLP_MODULE_NAME, "node_created",
                  "brain_id=0x%08X", node->brain_id);

    return node;
}

void nlp_node_destroy(nlp_node_t node) {
    if (!node) return;

    // Stop if running
    if (node->running) {
        nlp_node_stop(node);
    }

    // Clean up crypto subsystem
    nlp_crypto_shutdown(node);

    // Clean up bio-async context if any
    node->bio_module_ctx = NULL;

    // Free burst buffer
    if (node->burst_buffer) {
        nimcp_free(node->burst_buffer);
    }

    // Free pending messages
    for (uint32_t i = 0; i < 256; i++) {
        if (node->pending_messages[i]) {
            if (node->pending_messages[i]->payload) {
                nimcp_free(node->pending_messages[i]->payload);
            }
            nimcp_free(node->pending_messages[i]);
        }
    }

    // Destroy mutexes
    nimcp_mutex_destroy(&node->peer_mutex);
    nimcp_mutex_destroy(&node->key_mutex);
    nimcp_mutex_destroy(&node->env_mutex);
    nimcp_mutex_destroy(&node->stats_mutex);
    nimcp_mutex_destroy(&node->seq_mutex);
    nimcp_mutex_destroy(&node->burst_mutex);
    nimcp_mutex_destroy(&node->queue_mutex);

    bbb_audit_log(BBB_AUDIT_INFO, NLP_MODULE_NAME, "node_destroyed",
                  "brain_id=0x%08X", node->brain_id);

    nimcp_free(node);
}

int nlp_node_start(nlp_node_t node) {
    if (!bbb_check_pointer(node, "nlp_node_start")) {
        return -EINVAL;
    }

    if (node->running) {
        NIMCP_LOGGING_WARN(NLP_MODULE_NAME, "Node already running");
        return 0;
    }

    // Create UDP socket
    node->socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (node->socket_fd < 0) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Failed to create socket: %s", strerror(errno));
        return -errno;
    }

    // Set non-blocking
    int flags = fcntl(node->socket_fd, F_GETFL, 0);
    fcntl(node->socket_fd, F_SETFL, flags | O_NONBLOCK);

    // Allow address reuse
    int optval = 1;
    setsockopt(node->socket_fd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    // Bind socket
    memset(&node->bind_addr, 0, sizeof(node->bind_addr));
    node->bind_addr.sin_family = AF_INET;
    node->bind_addr.sin_port = htons(node->config.port);
    inet_pton(AF_INET, node->config.bind_address, &node->bind_addr.sin_addr);

    if (bind(node->socket_fd, (struct sockaddr*)&node->bind_addr,
             sizeof(node->bind_addr)) < 0) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Failed to bind socket: %s", strerror(errno));
        close(node->socket_fd);
        node->socket_fd = -1;
        return -errno;
    }

    nimcp_mutex_lock(&node->state_mutex);
    node->running = true;
    node->threads_running = true;
    nimcp_mutex_unlock(&node->state_mutex);

    // Start receive thread
    if (nimcp_thread_create(&node->recv_thread, nlp_recv_thread, node, NULL) != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Failed to create recv thread");
        nimcp_mutex_lock(&node->state_mutex);
        node->running = false;
        node->threads_running = false;
        nimcp_mutex_unlock(&node->state_mutex);
        close(node->socket_fd);
        return -ENOMEM;
    }

    // Start heartbeat thread
    if (nimcp_thread_create(&node->heartbeat_thread, nlp_heartbeat_thread, node, NULL) != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Failed to create heartbeat thread");
        nimcp_mutex_lock(&node->state_mutex);
        node->running = false;
        node->threads_running = false;
        nimcp_mutex_unlock(&node->state_mutex);
        nimcp_thread_join(node->recv_thread, NULL);
        close(node->socket_fd);
        return -ENOMEM;
    }

    // Start stealth thread if in stealth mode
    nimcp_mutex_lock(&node->state_mutex);
    bool start_stealth = (node->current_mode == NLP_MODE_STEALTH);
    nimcp_mutex_unlock(&node->state_mutex);
    if (start_stealth) {
        if (nimcp_thread_create(&node->stealth_thread, nlp_stealth_thread, node, NULL) != NIMCP_SUCCESS) {
            NIMCP_LOGGING_WARN(NLP_MODULE_NAME, "Failed to create stealth thread");
        }
    }

    NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "Node started on %s:%d",
                   node->config.bind_address, node->config.port);

    bbb_audit_log(BBB_AUDIT_INFO, NLP_MODULE_NAME, "node_started",
                  "port=%d mode=%s", node->config.port, nlp_mode_name(node->current_mode));

    return 0;
}

int nlp_node_stop(nlp_node_t node) {
    if (!bbb_check_pointer(node, "nlp_node_stop")) {
        return -EINVAL;
    }

    nimcp_mutex_lock(&node->state_mutex);
    if (!node->running) {
        nimcp_mutex_unlock(&node->state_mutex);
        return 0;
    }

    node->running = false;
    node->threads_running = false;
    bool was_stealth = (node->current_mode == NLP_MODE_STEALTH);
    nimcp_mutex_unlock(&node->state_mutex);

    // Wake up threads
    if (node->socket_fd >= 0) {
        shutdown(node->socket_fd, SHUT_RDWR);
    }

    // Wait for threads
    nimcp_thread_join(node->recv_thread, NULL);
    nimcp_thread_join(node->heartbeat_thread, NULL);
    if (was_stealth) {
        nimcp_thread_join(node->stealth_thread, NULL);
    }

    // Close socket
    if (node->socket_fd >= 0) {
        close(node->socket_fd);
        node->socket_fd = -1;
    }

    // Disconnect all peers
    nimcp_mutex_lock(&node->peer_mutex);
    for (uint32_t i = 0; i < node->peer_count; i++) {
        node->peers[i].session_state = NLP_SESSION_DISCONNECTED;
    }
    node->peer_count = 0;
    nimcp_mutex_unlock(&node->peer_mutex);

    NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "Node stopped");

    bbb_audit_log(BBB_AUDIT_INFO, NLP_MODULE_NAME, "node_stopped",
                  "brain_id=0x%08X", node->brain_id);

    return 0;
}

//=============================================================================
// Peer Management
//=============================================================================

uint32_t nlp_connect_peer(nlp_node_t node, const char* address, uint16_t port) {
    if (!bbb_check_pointer(node, "nlp_connect_peer") ||
        !bbb_check_string(address, 64, "nlp_connect_peer")) {
        return 0;
    }

    nimcp_mutex_lock(&node->peer_mutex);

    // Check if already connected
    for (uint32_t i = 0; i < node->peer_count; i++) {
        if (strcmp(node->peers[i].address, address) == 0 &&
            node->peers[i].port == port) {
            nimcp_mutex_unlock(&node->peer_mutex);
            return node->peers[i].peer_id;
        }
    }

    // Check capacity
    if (node->peer_count >= node->config.max_peers) {
        NIMCP_LOGGING_WARN(NLP_MODULE_NAME, "Max peers reached");
        nimcp_mutex_unlock(&node->peer_mutex);
        return 0;
    }

    // Add new peer
    nlp_peer_t* peer = &node->peers[node->peer_count];
    memset(peer, 0, sizeof(nlp_peer_t));

    // Generate peer ID from address hash
    uint32_t hash = 5381;
    for (const char* p = address; *p; p++) {
        hash = ((hash << 5) + hash) + *p;
    }
    hash ^= port;
    peer->peer_id = hash;

    strncpy(peer->address, address, sizeof(peer->address) - 1);
    peer->port = port;
    peer->session_state = NLP_SESSION_DISCONNECTED;
    peer->healthy = false;

    uint64_t now = nimcp_platform_time_monotonic_ms();
    peer->last_seen_ms = now;
    peer->last_sent_ms = now;

    node->peer_count++;

    nimcp_mutex_unlock(&node->peer_mutex);

    // Initiate handshake based on mode
    if (node->current_mode == NLP_MODE_STANDARD) {
        // Start 3-way handshake - transition state first
        nlp_session_start_handshake(node, peer);

        // Actually send the handshake request message over the network
        // This is necessary because nlp_session_start_handshake only manages state,
        // it doesn't send the actual network message
        int send_rc = nlp_send(node, peer->peer_id, NLP_MSG_HANDSHAKE_REQ,
                               NULL, 0, NLP_PRIORITY_HIGH);
        if (send_rc < 0) {
            NIMCP_LOGGING_WARN(NLP_MODULE_NAME,
                "Failed to send handshake request to peer 0x%08X: %d",
                peer->peer_id, send_rc);
            // Don't fail peer creation - peer will retry or timeout
        } else {
            NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
                "Sent handshake request to peer 0x%08X", peer->peer_id);
        }
    } else {
        // Tactical/Stealth: use PSK, mark as established immediately
        peer->session_state = NLP_SESSION_ESTABLISHED;

        // Select active PSK
        nimcp_mutex_lock(&node->key_mutex);
        for (int i = 0; i < NLP_KEY_SLOTS; i++) {
            if (node->psk_slots[i].active) {
                memcpy(peer->session_key, node->psk_slots[i].key, NLP_KEY_SIZE);
                break;
            }
        }
        nimcp_mutex_unlock(&node->key_mutex);

        peer->healthy = true;
    }

    NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "Peer added: %s:%d id=0x%08X",
                   address, port, peer->peer_id);

    // Notify callback
    if (node->peer_callback) {
        node->peer_callback(node, peer, NLP_SESSION_DISCONNECTED,
                           peer->session_state, node->user_data);
    }

    return peer->peer_id;
}

int nlp_disconnect_peer(nlp_node_t node, uint32_t peer_id) {
    if (!bbb_check_pointer(node, "nlp_disconnect_peer")) {
        return -EINVAL;
    }

    nimcp_mutex_lock(&node->peer_mutex);

    for (uint32_t i = 0; i < node->peer_count; i++) {
        if (node->peers[i].peer_id == peer_id) {
            nlp_peer_t* peer = &node->peers[i];
            nlp_session_state_t old_state = peer->session_state;

            // Send disconnect message if in standard mode
            if (node->current_mode == NLP_MODE_STANDARD &&
                peer->session_state == NLP_SESSION_ESTABLISHED) {
                nlp_send(node, peer_id, NLP_MSG_DISCONNECT, NULL, 0, NLP_PRIORITY_HIGH);
            }

            // Notify callback
            if (node->peer_callback) {
                node->peer_callback(node, peer, old_state,
                                   NLP_SESSION_DISCONNECTED, node->user_data);
            }

            // Remove peer by shifting array
            if (i < node->peer_count - 1) {
                memmove(&node->peers[i], &node->peers[i + 1],
                       (node->peer_count - i - 1) * sizeof(nlp_peer_t));
            }
            node->peer_count--;

            nimcp_mutex_unlock(&node->peer_mutex);

            NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "Peer disconnected: 0x%08X", peer_id);
            return 0;
        }
    }

    nimcp_mutex_unlock(&node->peer_mutex);
    return -ENOENT;
}

int nlp_get_peer(nlp_node_t node, uint32_t peer_id, nlp_peer_t* peer) {
    if (!bbb_check_pointer(node, "nlp_get_peer") ||
        !bbb_check_pointer(peer, "nlp_get_peer")) {
        return -EINVAL;
    }

    nimcp_mutex_lock(&node->peer_mutex);

    for (uint32_t i = 0; i < node->peer_count; i++) {
        if (node->peers[i].peer_id == peer_id) {
            memcpy(peer, &node->peers[i], sizeof(nlp_peer_t));
            nimcp_mutex_unlock(&node->peer_mutex);
            return 0;
        }
    }

    nimcp_mutex_unlock(&node->peer_mutex);
    return -ENOENT;
}

uint32_t nlp_get_peers(nlp_node_t node, nlp_peer_t* peers, uint32_t max_peers) {
    if (!bbb_check_pointer(node, "nlp_get_peers")) {
        return 0;
    }

    nimcp_mutex_lock(&node->peer_mutex);

    uint32_t count = node->peer_count;
    if (count > max_peers) count = max_peers;

    if (peers && count > 0) {
        memcpy(peers, node->peers, count * sizeof(nlp_peer_t));
    }

    nimcp_mutex_unlock(&node->peer_mutex);
    return count;
}

//=============================================================================
// Messaging
//=============================================================================

int nlp_send(nlp_node_t node, uint32_t peer_id, nlp_msg_type_t msg_type,
             const void* payload, size_t payload_len, nlp_priority_t priority) {
    if (!bbb_check_pointer(node, "nlp_send")) {
        return -EINVAL;
    }

    if (payload_len > 0 && !bbb_check_pointer(payload, "nlp_send")) {
        return -EINVAL;
    }

    if (payload_len > NLP_MAX_PAYLOAD) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Payload too large: %zu", payload_len);
        return -EMSGSIZE;
    }

    // Check EMCON restrictions in stealth mode
    if (node->current_mode == NLP_MODE_STEALTH) {
        if (node->emcon_level == NLP_EMCON_RECEIVE ||
            node->emcon_level == NLP_EMCON_SILENT) {
            // Cannot transmit unless emergency
            if (node->emcon_level != NLP_EMCON_EMERGENCY &&
                priority != NLP_PRIORITY_CRITICAL) {
                NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME, "TX blocked by EMCON level %d",
                               node->emcon_level);
                return -EAGAIN;
            }
        }
    }

    // For broadcasts (peer_id == 0) in STANDARD mode, directly call nlp_broadcast
    // which will iterate over all established peers and send to each with their
    // individual session keys. We can't encrypt a single broadcast message in
    // STANDARD mode because each peer has a different session key.
    if (peer_id == 0 && node->current_mode == NLP_MODE_STANDARD) {
        return nlp_broadcast(node, msg_type, payload, payload_len, priority);
    }

    // Build message (pass NULL payload - we handle encryption separately)
    nlp_message_t* msg = nlp_message_create(msg_type, NULL, 0);
    if (!msg) {
        return -ENOMEM;
    }

    // Initialize header
    nlp_header_init(&msg->header);
    NLP_SET_VERSION(&msg->header, NLP_VERSION);
    NLP_SET_MODE(&msg->header, node->current_mode);
    NLP_SET_PRIORITY(&msg->header, priority);

    uint8_t flags = NLP_FLAG_ENCRYPTED;
    if (priority >= NLP_PRIORITY_HIGH) {
        flags |= NLP_FLAG_ACK_REQUIRED;
    }
    NLP_SET_FLAGS(&msg->header, flags);

    // Set header fields in HOST byte order
    // nlp_header_serialize will convert to network byte order
    msg->header.msg_type = msg_type;
    msg->header.sender_id = node->brain_id;
    msg->header.timestamp = (uint32_t)time(NULL);

    // Always set dest_id = 0 for now.
    // For initiated connections, peer_id is hash(address:port), which doesn't
    // match the remote brain_id, causing dest_id check failures on the receiver.
    // Unicast is determined by destination address:port, not dest_id.
    // TODO: Track remote brain_id separately and use it for dest_id in mesh mode.
    msg->header.dest_id = 0;

    // Generate nonce using node's crypto state
    nlp_crypto_generate_nonce(node, msg->header.nonce);

    // Get sequence number
    nimcp_mutex_lock(&node->seq_mutex);
    msg->header.sequence = node->tx_sequence++;
    nimcp_mutex_unlock(&node->seq_mutex);

    // Select key slot based on mode
    uint8_t key_slot = 0;
    if (node->current_mode != NLP_MODE_STANDARD) {
        nlp_session_select_psk(node, &key_slot);
    }
    NLP_SET_KEY_SLOT(&msg->header, key_slot);

    // Check if this is a handshake message (sent before session key is established)
    // Handshake messages in standard mode cannot be encrypted because
    // session keys don't exist yet - they're negotiated during handshake
    bool is_handshake_msg = (msg_type == NLP_MSG_HANDSHAKE_REQ ||
                             msg_type == NLP_MSG_HANDSHAKE_RESP ||
                             msg_type == NLP_MSG_HANDSHAKE_FINAL);

    // For standard mode handshake messages, skip encryption
    if (is_handshake_msg && node->current_mode == NLP_MODE_STANDARD) {
        // Clear encrypted flag for handshake messages
        uint8_t flags = NLP_GET_FLAGS(&msg->header);
        flags &= ~NLP_FLAG_ENCRYPTED;
        NLP_SET_FLAGS(&msg->header, flags);

        // Copy unencrypted payload if present
        if (payload && payload_len > 0) {
            msg->payload = (uint8_t*)nimcp_malloc(payload_len);
            if (!msg->payload) {
                nlp_message_destroy(msg);
                return -ENOMEM;
            }
            memcpy(msg->payload, payload, payload_len);
            msg->header.payload_len = (uint16_t)payload_len;
        } else {
            msg->header.payload_len = 0;
        }

        // Clear auth tag (no encryption = no auth tag)
        memset(msg->auth_tag, 0, NLP_TAG_SIZE);

        NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
            "Sending unencrypted handshake message type=0x%04X to peer 0x%08X",
            msg_type, peer_id);

        goto serialize_and_send;
    }

    // Encrypt payload
    if (payload && payload_len > 0) {
        msg->payload = (uint8_t*)nimcp_malloc(payload_len + NLP_TAG_SIZE);
        if (!msg->payload) {
            nlp_message_destroy(msg);
            return -ENOMEM;
        }

        // Get encryption key
        uint8_t* key = NULL;
        if (peer_id == 0) {
            // Broadcast: use PSK
            nimcp_mutex_lock(&node->key_mutex);
            if (node->psk_slots[key_slot].active) {
                key = node->psk_slots[key_slot].key;
            }
            nimcp_mutex_unlock(&node->key_mutex);
        } else {
            // Unicast: use session key
            nimcp_mutex_lock(&node->peer_mutex);
            for (uint32_t i = 0; i < node->peer_count; i++) {
                if (node->peers[i].peer_id == peer_id) {
                    key = node->peers[i].session_key;
                    break;
                }
            }
            nimcp_mutex_unlock(&node->peer_mutex);
        }

        if (!key) {
            NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "No encryption key available");
            nlp_message_destroy(msg);
            return -ENOKEY;
        }

        // Set payload length BEFORE serializing header (needed for AAD)
        msg->header.payload_len = (uint16_t)payload_len;

        // Serialize header to get wire format for AAD
        // We need the exact bytes that will be sent on the wire
        nlp_header_t wire_header;
        memcpy(&wire_header, &msg->header, sizeof(nlp_header_t));
        nlp_header_serialize(&wire_header);

        // Encrypt with node's crypto state, using SERIALIZED header as AAD
        // IMPORTANT: Must use wire format to match what receiver sees
        int rc = nlp_crypto_encrypt(node, key, msg->header.nonce,
                                    payload, payload_len,
                                    (const uint8_t*)&wire_header, sizeof(nlp_header_t),
                                    msg->payload, payload_len + NLP_TAG_SIZE,
                                    msg->auth_tag);
        if (rc < 0) {
            NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Encryption failed: %d", rc);
            nlp_message_destroy(msg);
            return rc;
        }
    } else {
        msg->header.payload_len = 0;
    }

serialize_and_send:
    // Note: Header CRC is calculated in nlp_header_serialize, called from nlp_message_serialize

    // Serialize message
    uint8_t wire_buffer[NLP_MAX_PAYLOAD + NLP_HEADER_SIZE + NLP_TAG_SIZE];
    size_t wire_len = 0;

    int rc = nlp_message_serialize(msg, wire_buffer, sizeof(wire_buffer), &wire_len);
    if (rc < 0) {
        nlp_message_destroy(msg);
        return rc;
    }

    // In stealth mode, pad to fixed size or queue for burst
    if (node->current_mode == NLP_MODE_STEALTH) {
        // Queue for burst UNLESS emergency override or critical priority
        if (node->emcon_level >= NLP_EMCON_REDUCED &&
            node->emcon_level != NLP_EMCON_EMERGENCY &&
            priority != NLP_PRIORITY_CRITICAL) {
            // Queue for burst transmission
            nimcp_mutex_lock(&node->burst_mutex);
            if (node->burst_buffer_used + wire_len <= node->burst_buffer_size) {
                memcpy(node->burst_buffer + node->burst_buffer_used,
                       wire_buffer, wire_len);
                node->burst_buffer_used += wire_len;
            }
            nimcp_mutex_unlock(&node->burst_mutex);
            nlp_message_destroy(msg);
            return 0;  // Queued for later
        }

        // Pad to fixed size for traffic analysis resistance
        // nlp_message_pad_to_fixed_size expects message + buffer, not wire_buffer + len
        // Create temporary padded buffer
        uint8_t padded_buffer[NLP_STEALTH_PACKET_SIZE];
        if (nlp_message_pad_to_fixed_size(msg, padded_buffer) == 0) {
            memcpy(wire_buffer, padded_buffer, NLP_STEALTH_PACKET_SIZE);
            wire_len = NLP_STEALTH_PACKET_SIZE;
        }
    }

    // Send to peer(s)
    if (peer_id == 0) {
        // Broadcast
        rc = nlp_broadcast(node, msg_type, payload, payload_len, priority);
    } else {
        rc = nlp_send_raw(node, peer_id, wire_buffer, wire_len);
    }

    nlp_message_destroy(msg);

    // Update statistics
    if (rc >= 0) {
        nimcp_mutex_lock(&node->stats_mutex);
        node->stats.messages_sent++;
        node->stats.bytes_sent += wire_len;
        nimcp_mutex_unlock(&node->stats_mutex);
    }

    return rc;
}

int nlp_broadcast(nlp_node_t node, nlp_msg_type_t msg_type,
                  const void* payload, size_t payload_len, nlp_priority_t priority) {
    if (!bbb_check_pointer(node, "nlp_broadcast")) {
        return -EINVAL;
    }

    int sent = 0;

    nimcp_mutex_lock(&node->peer_mutex);
    for (uint32_t i = 0; i < node->peer_count; i++) {
        if (node->peers[i].session_state == NLP_SESSION_ESTABLISHED) {
            nimcp_mutex_unlock(&node->peer_mutex);

            int rc = nlp_send(node, node->peers[i].peer_id, msg_type,
                             payload, payload_len, priority);
            if (rc >= 0) sent++;

            nimcp_mutex_lock(&node->peer_mutex);
        }
    }
    nimcp_mutex_unlock(&node->peer_mutex);

    return sent;
}

int nlp_relay(nlp_node_t node, uint32_t dest_id, nlp_msg_type_t msg_type,
              const void* payload, size_t payload_len) {
    if (!bbb_check_pointer(node, "nlp_relay")) {
        return -EINVAL;
    }

    // Build relay header
    struct {
        uint32_t final_dest;
        uint8_t ttl;
        uint8_t hop_count;
        uint16_t reserved;
    } relay_header;

    relay_header.final_dest = htonl(dest_id);
    relay_header.ttl = 16;  // Max 16 hops
    relay_header.hop_count = 0;
    relay_header.reserved = 0;

    // Combine relay header with original payload
    size_t total_len = sizeof(relay_header) + payload_len;
    uint8_t* relay_payload = (uint8_t*)nimcp_malloc(total_len);
    if (!relay_payload) {
        return -ENOMEM;
    }

    memcpy(relay_payload, &relay_header, sizeof(relay_header));
    if (payload && payload_len > 0) {
        memcpy(relay_payload + sizeof(relay_header), payload, payload_len);
    }

    // Broadcast relay message
    int rc = nlp_broadcast(node, NLP_MSG_RELAY, relay_payload, total_len,
                          NLP_PRIORITY_NORMAL);

    nimcp_free(relay_payload);

    nimcp_mutex_lock(&node->stats_mutex);
    node->stats.messages_relayed++;
    nimcp_mutex_unlock(&node->stats_mutex);

    return rc;
}

//=============================================================================
// Neural Sync
//=============================================================================

int nlp_send_spikes(nlp_node_t node, uint32_t peer_id,
                    const uint32_t* neuron_ids, const uint16_t* spike_times,
                    uint32_t count) {
    if (!bbb_check_pointer(node, "nlp_send_spikes")) {
        return -EINVAL;
    }

    if (count == 0) return 0;

    if (!bbb_check_pointer(neuron_ids, "nlp_send_spikes") ||
        !bbb_check_pointer(spike_times, "nlp_send_spikes")) {
        return -EINVAL;
    }

    // Pack spike batch
    size_t payload_len = sizeof(nlp_spike_batch_t) +
                         count * sizeof(uint32_t) +
                         count * sizeof(uint16_t);

    uint8_t* payload = (uint8_t*)nimcp_malloc(payload_len);
    if (!payload) {
        return -ENOMEM;
    }

    nlp_spike_batch_t* batch = (nlp_spike_batch_t*)payload;
    batch->batch_id = htonl((uint32_t)time(NULL));
    batch->timestamp_us = htonl(0);
    batch->spike_count = htons((uint16_t)count);
    batch->reserved = 0;

    uint32_t* ids_out = (uint32_t*)(payload + sizeof(nlp_spike_batch_t));
    uint16_t* times_out = (uint16_t*)(ids_out + count);

    for (uint32_t i = 0; i < count; i++) {
        ids_out[i] = htonl(neuron_ids[i]);
        times_out[i] = htons(spike_times[i]);
    }

    int rc = nlp_send(node, peer_id, NLP_MSG_SPIKE_BATCH, payload, payload_len,
                     NLP_PRIORITY_HIGH);

    nimcp_free(payload);
    return rc;
}

int nlp_send_weight_deltas(nlp_node_t node, uint32_t peer_id,
                           const uint32_t* synapse_ids,
                           const float* old_weights, const float* new_weights,
                           uint32_t count) {
    if (!bbb_check_pointer(node, "nlp_send_weight_deltas")) {
        return -EINVAL;
    }

    if (count == 0) return 0;

    if (!bbb_check_pointer(synapse_ids, "nlp_send_weight_deltas") ||
        !bbb_check_pointer(old_weights, "nlp_send_weight_deltas") ||
        !bbb_check_pointer(new_weights, "nlp_send_weight_deltas")) {
        return -EINVAL;
    }

    // Pack weight deltas
    size_t payload_len = sizeof(nlp_weight_delta_header_t) +
                         count * sizeof(nlp_weight_delta_entry_t);

    uint8_t* payload = (uint8_t*)nimcp_malloc(payload_len);
    if (!payload) {
        return -ENOMEM;
    }

    nlp_weight_delta_header_t* header = (nlp_weight_delta_header_t*)payload;
    header->base_version = htonl(0);  // TODO: track versions
    header->new_version = htonl(1);
    header->delta_count = htons((uint16_t)count);
    header->reserved = 0;

    nlp_weight_delta_entry_t* entries =
        (nlp_weight_delta_entry_t*)(payload + sizeof(nlp_weight_delta_header_t));

    for (uint32_t i = 0; i < count; i++) {
        entries[i].synapse_id = htonl(synapse_ids[i]);
        // Convert float to network byte order
        uint32_t old_w, new_w;
        memcpy(&old_w, &old_weights[i], sizeof(float));
        memcpy(&new_w, &new_weights[i], sizeof(float));
        entries[i].old_weight = *(float*)&old_w;  // Keep native float representation
        entries[i].new_weight = *(float*)&new_w;
    }

    int rc = nlp_send(node, peer_id, NLP_MSG_WEIGHT_DELTA, payload, payload_len,
                     NLP_PRIORITY_NORMAL);

    nimcp_free(payload);
    return rc;
}

int nlp_send_state(nlp_node_t node, uint32_t peer_id,
                   const void* state_data, size_t state_len) {
    if (!bbb_check_pointer(node, "nlp_send_state")) {
        return -EINVAL;
    }

    if (state_len > 0 && !bbb_check_pointer(state_data, "nlp_send_state")) {
        return -EINVAL;
    }

    return nlp_send(node, peer_id, NLP_MSG_STATE_SYNC, state_data, state_len,
                   NLP_PRIORITY_LOW);
}

//=============================================================================
// Mode Control
//=============================================================================

int nlp_set_mode(nlp_node_t node, nlp_mode_t mode) {
    if (!bbb_check_pointer(node, "nlp_set_mode")) {
        return -EINVAL;
    }

    if (mode > NLP_MODE_STEALTH) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Invalid mode: %d", mode);
        return -EINVAL;
    }

    nlp_mode_t old_mode = node->current_mode;
    if (old_mode == mode) {
        return 0;  // No change
    }

    node->current_mode = mode;

    // Mode-specific initialization
    if (mode == NLP_MODE_STEALTH) {
        // Start stealth thread if not running
        if (node->running && !node->threads_running) {
            nimcp_thread_create(&node->stealth_thread, nlp_stealth_thread, node, NULL);
        }
        node->next_burst_time = nimcp_platform_time_monotonic_ms() +
                                node->config.burst_interval_s * NIMCP_MS_PER_SEC;
    }

    // Update statistics
    nimcp_mutex_lock(&node->stats_mutex);
    node->stats.mode_switches++;
    node->stats.current_mode = mode;
    nimcp_mutex_unlock(&node->stats_mutex);

    NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "Mode changed: %s -> %s",
                   nlp_mode_name(old_mode), nlp_mode_name(mode));

    // Notify callback
    if (node->mode_callback) {
        node->mode_callback(node, old_mode, mode, "manual", node->user_data);
    }

    // Broadcast mode change to peers
    uint8_t mode_payload[4];
    mode_payload[0] = (uint8_t)old_mode;
    mode_payload[1] = (uint8_t)mode;
    mode_payload[2] = 0;
    mode_payload[3] = 0;
    nlp_broadcast(node, NLP_MSG_EMCON_CHANGE, mode_payload, 4, NLP_PRIORITY_HIGH);

    bbb_audit_log(BBB_AUDIT_INFO, NLP_MODULE_NAME, "mode_changed",
                  "old=%s new=%s", nlp_mode_name(old_mode), nlp_mode_name(mode));

    return 0;
}

nlp_mode_t nlp_get_mode(nlp_node_t node) {
    if (!node) return NLP_MODE_STANDARD;
    return node->current_mode;
}

int nlp_set_emcon(nlp_node_t node, nlp_emcon_level_t level) {
    if (!bbb_check_pointer(node, "nlp_set_emcon")) {
        return -EINVAL;
    }

    if (level > NLP_EMCON_EMERGENCY) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Invalid EMCON level: %d", level);
        return -EINVAL;
    }

    nlp_emcon_level_t old_level = node->emcon_level;
    node->emcon_level = level;

    nimcp_mutex_lock(&node->stats_mutex);
    node->stats.current_emcon = level;
    nimcp_mutex_unlock(&node->stats_mutex);

    NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "EMCON level changed: %s -> %s",
                   nlp_emcon_name(old_level), nlp_emcon_name(level));

    bbb_audit_log(BBB_AUDIT_INFO, NLP_MODULE_NAME, "emcon_changed",
                  "old=%s new=%s", nlp_emcon_name(old_level), nlp_emcon_name(level));

    return 0;
}

nlp_emcon_level_t nlp_get_emcon(nlp_node_t node) {
    if (!node) return NLP_EMCON_NORMAL;
    return node->emcon_level;
}

void nlp_update_environment(nlp_node_t node, const nlp_environment_t* env) {
    if (!bbb_check_pointer(node, "nlp_update_environment") ||
        !bbb_check_pointer(env, "nlp_update_environment")) {
        return;
    }

    nimcp_mutex_lock(&node->env_mutex);
    memcpy(&node->environment, env, sizeof(nlp_environment_t));
    nimcp_mutex_unlock(&node->env_mutex);

    // Check for automatic mode switching
    if (node->config.auto_mode_switch) {
        nlp_auto_mode_check(node);
    }
}

static void nlp_auto_mode_check(nlp_node_t node) {
    nimcp_mutex_lock(&node->env_mutex);
    nlp_environment_t env = node->environment;
    nimcp_mutex_unlock(&node->env_mutex);

    nlp_mode_t suggested_mode = node->current_mode;
    const char* reason = NULL;

    // Check for degraded conditions requiring tactical mode
    if (env.packet_loss_rate > 0.3F ||
        env.jamming_events > 0 ||
        !env.master_reachable ||
        env.master_timeout_ms > 30000) {
        suggested_mode = NLP_MODE_TACTICAL;
        reason = "degraded network conditions";
    }

    // Check for stealth requirements
    if (env.rf_anomaly_detected ||
        env.unknown_peer_contact ||
        env.replay_attempt_detected) {
        suggested_mode = NLP_MODE_STEALTH;
        reason = "security threat detected";
    }

    // Check for power conservation
    if (env.low_power_mode || env.battery_percent < 20.0F) {
        if (node->current_mode == NLP_MODE_STANDARD) {
            suggested_mode = NLP_MODE_TACTICAL;  // Less overhead
            reason = "low power mode";
        }
    }

    // Check if we can return to standard mode
    if (node->current_mode != NLP_MODE_STANDARD &&
        env.packet_loss_rate < 0.1F &&
        env.master_reachable &&
        !env.rf_anomaly_detected &&
        env.battery_percent > 50.0F) {
        suggested_mode = NLP_MODE_STANDARD;
        reason = "conditions improved";
    }

    // Apply mode change if different
    if (suggested_mode != node->current_mode) {
        nlp_mode_t old_mode = node->current_mode;
        node->current_mode = suggested_mode;

        NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "Auto mode switch: %s -> %s (%s)",
                       nlp_mode_name(old_mode), nlp_mode_name(suggested_mode), reason);

        if (node->mode_callback) {
            node->mode_callback(node, old_mode, suggested_mode, reason, node->user_data);
        }
    }
}

//=============================================================================
// Key Management
//=============================================================================

int nlp_set_psk(nlp_node_t node, uint8_t slot, const uint8_t* key,
                uint32_t key_id, uint64_t valid_from, uint64_t valid_until) {
    if (!bbb_check_pointer(node, "nlp_set_psk") ||
        !bbb_check_pointer(key, "nlp_set_psk")) {
        return -EINVAL;
    }

    if (slot >= NLP_KEY_SLOTS) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Invalid key slot: %d", slot);
        return -EINVAL;
    }

    nimcp_mutex_lock(&node->key_mutex);

    nlp_key_slot_t* psk = &node->psk_slots[slot];
    memcpy(psk->key, key, NLP_KEY_SIZE);
    psk->key_id = key_id;
    psk->valid_from = valid_from;
    psk->valid_until = valid_until;
    psk->active = true;

    nimcp_mutex_unlock(&node->key_mutex);

    NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "PSK set in slot %d, key_id=0x%08X", slot, key_id);

    bbb_audit_log(BBB_AUDIT_INFO, NLP_MODULE_NAME, "psk_set",
                  "slot=%d key_id=0x%08X", slot, key_id);

    return 0;
}

int nlp_rotate_session_key(nlp_node_t node, uint32_t peer_id) {
    if (!bbb_check_pointer(node, "nlp_rotate_session_key")) {
        return -EINVAL;
    }

    // Only applicable in standard mode
    if (node->current_mode != NLP_MODE_STANDARD) {
        NIMCP_LOGGING_WARN(NLP_MODULE_NAME, "Key rotation only in standard mode");
        return -EINVAL;
    }

    // Generate new session key (using nonce generator as source of randomness)
    uint8_t new_key[NLP_KEY_SIZE];
    nlp_crypto_generate_nonce(node, new_key);       // First 12 bytes
    nlp_crypto_generate_nonce(node, new_key + 12);  // Next 12 bytes
    // Last 8 bytes are initialized to zero (calloc-style init from stack)

    // Find peer and update key
    nimcp_mutex_lock(&node->peer_mutex);
    for (uint32_t i = 0; i < node->peer_count; i++) {
        if (node->peers[i].peer_id == peer_id) {
            memcpy(node->peers[i].session_key, new_key, NLP_KEY_SIZE);
            nimcp_mutex_unlock(&node->peer_mutex);

            // Send key rotation message
            nlp_send(node, peer_id, NLP_MSG_KEY_ROTATE, new_key, NLP_KEY_SIZE,
                    NLP_PRIORITY_HIGH);

            NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "Session key rotated for peer 0x%08X", peer_id);
            return 0;
        }
    }
    nimcp_mutex_unlock(&node->peer_mutex);

    return -ENOENT;
}

//=============================================================================
// Callbacks
//=============================================================================

void nlp_set_message_callback(nlp_node_t node, nlp_message_callback_t callback) {
    if (node) {
        node->message_callback = callback;
    }
}

void nlp_set_peer_callback(nlp_node_t node, nlp_peer_callback_t callback) {
    if (node) {
        node->peer_callback = callback;
    }
}

void nlp_set_mode_callback(nlp_node_t node, nlp_mode_callback_t callback) {
    if (node) {
        node->mode_callback = callback;
    }
}

//=============================================================================
// SAR/Disaster API
//=============================================================================

int nlp_send_location(nlp_node_t node, const nlp_location_t* location) {
    if (!bbb_check_pointer(node, "nlp_send_location") ||
        !bbb_check_pointer(location, "nlp_send_location")) {
        return -EINVAL;
    }

    return nlp_broadcast(node, NLP_MSG_LOCATION_UPDATE, location,
                        sizeof(nlp_location_t), NLP_PRIORITY_NORMAL);
}

int nlp_send_victim_report(nlp_node_t node, const nlp_victim_report_t* report,
                           const char* notes) {
    if (!bbb_check_pointer(node, "nlp_send_victim_report") ||
        !bbb_check_pointer(report, "nlp_send_victim_report")) {
        return -EINVAL;
    }

    size_t notes_len = notes ? strlen(notes) : 0;
    size_t payload_len = sizeof(nlp_victim_report_t) + notes_len;

    uint8_t* payload = (uint8_t*)nimcp_malloc(payload_len);
    if (!payload) {
        return -ENOMEM;
    }

    memcpy(payload, report, sizeof(nlp_victim_report_t));
    nlp_victim_report_t* r = (nlp_victim_report_t*)payload;
    r->notes_len = (uint16_t)notes_len;

    if (notes && notes_len > 0) {
        memcpy(payload + sizeof(nlp_victim_report_t), notes, notes_len);
    }

    // Victim reports are high priority
    int rc = nlp_broadcast(node, NLP_MSG_VICTIM_REPORT, payload, payload_len,
                          NLP_PRIORITY_HIGH);

    nimcp_free(payload);

    NIMCP_LOGGING_INFO(NLP_MODULE_NAME, "Victim report sent: id=%u triage=%d",
                   report->victim_id, report->triage);

    return rc;
}

int nlp_send_sensors(nlp_node_t node, const nlp_sensor_data_t* sensors) {
    if (!bbb_check_pointer(node, "nlp_send_sensors") ||
        !bbb_check_pointer(sensors, "nlp_send_sensors")) {
        return -EINVAL;
    }

    return nlp_broadcast(node, NLP_MSG_SENSOR_DATA, sensors,
                        sizeof(nlp_sensor_data_t), NLP_PRIORITY_LOW);
}

//=============================================================================
// Statistics
//=============================================================================

int nlp_get_stats(nlp_node_t node, nlp_stats_t* stats) {
    if (!bbb_check_pointer(node, "nlp_get_stats") ||
        !bbb_check_pointer(stats, "nlp_get_stats")) {
        return -EINVAL;
    }

    nimcp_mutex_lock(&node->stats_mutex);
    memcpy(stats, &node->stats, sizeof(nlp_stats_t));
    nimcp_mutex_unlock(&node->stats_mutex);

    // Add peer count
    nimcp_mutex_lock(&node->peer_mutex);
    stats->connected_peers = 0;
    for (uint32_t i = 0; i < node->peer_count; i++) {
        if (node->peers[i].session_state == NLP_SESSION_ESTABLISHED) {
            stats->connected_peers++;
        }
    }
    stats->active_sessions = stats->connected_peers;
    nimcp_mutex_unlock(&node->peer_mutex);

    return 0;
}

void nlp_reset_stats(nlp_node_t node) {
    if (!node) return;

    nimcp_mutex_lock(&node->stats_mutex);
    memset(&node->stats, 0, sizeof(nlp_stats_t));
    node->stats.current_mode = node->current_mode;
    node->stats.current_emcon = node->emcon_level;
    nimcp_mutex_unlock(&node->stats_mutex);
}

//=============================================================================
// Utility Functions
//=============================================================================

const char* nlp_msg_type_name(nlp_msg_type_t type) {
    switch (type) {
        case NLP_MSG_HANDSHAKE_REQ:    return "HANDSHAKE_REQ";
        case NLP_MSG_HANDSHAKE_RESP:   return "HANDSHAKE_RESP";
        case NLP_MSG_HANDSHAKE_FINAL:  return "HANDSHAKE_FINAL";
        case NLP_MSG_KEY_ROTATE:       return "KEY_ROTATE";
        case NLP_MSG_DISCONNECT:       return "DISCONNECT";
        case NLP_MSG_SESSION_RESUME:   return "SESSION_RESUME";
        case NLP_MSG_SPIKE_BATCH:      return "SPIKE_BATCH";
        case NLP_MSG_WEIGHT_DELTA:     return "WEIGHT_DELTA";
        case NLP_MSG_WEIGHT_FULL:      return "WEIGHT_FULL";
        case NLP_MSG_STATE_SYNC:       return "STATE_SYNC";
        case NLP_MSG_GRADIENT_PUSH:    return "GRADIENT_PUSH";
        case NLP_MSG_ACTIVATION_SYNC:  return "ACTIVATION_SYNC";
        case NLP_MSG_HEARTBEAT:        return "HEARTBEAT";
        case NLP_MSG_PEER_ANNOUNCE:    return "PEER_ANNOUNCE";
        case NLP_MSG_PEER_LIST:        return "PEER_LIST";
        case NLP_MSG_MASTER_ELECTION:  return "MASTER_ELECTION";
        case NLP_MSG_CONSENSUS_VOTE:   return "CONSENSUS_VOTE";
        case NLP_MSG_CONSENSUS_COMMIT: return "CONSENSUS_COMMIT";
        case NLP_MSG_ROLE_ASSIGN:      return "ROLE_ASSIGN";
        case NLP_MSG_PRIORITY_CMD:     return "PRIORITY_CMD";
        case NLP_MSG_EMERGENCY:        return "EMERGENCY";
        case NLP_MSG_RELAY:            return "RELAY";
        case NLP_MSG_ACK:              return "ACK";
        case NLP_MSG_NACK:             return "NACK";
        case NLP_MSG_RESEND_REQ:       return "RESEND_REQ";
        case NLP_MSG_BURST_SYNC:       return "BURST_SYNC";
        case NLP_MSG_CHAFF:            return "CHAFF";
        case NLP_MSG_LISTEN_WINDOW:    return "LISTEN_WINDOW";
        case NLP_MSG_EMCON_CHANGE:     return "EMCON_CHANGE";
        case NLP_MSG_LOCATION_UPDATE:  return "LOCATION_UPDATE";
        case NLP_MSG_VICTIM_REPORT:    return "VICTIM_REPORT";
        case NLP_MSG_SENSOR_DATA:      return "SENSOR_DATA";
        case NLP_MSG_HAZARD_ALERT:     return "HAZARD_ALERT";
        case NLP_MSG_RESOURCE_STATUS:  return "RESOURCE_STATUS";
        case NLP_MSG_PING:             return "PING";
        case NLP_MSG_PONG:             return "PONG";
        case NLP_MSG_STATS_REQ:        return "STATS_REQ";
        case NLP_MSG_STATS_RESP:       return "STATS_RESP";
        case NLP_MSG_DEBUG:            return "DEBUG";
        default:                       return "UNKNOWN";
    }
}

const char* nlp_mode_name(nlp_mode_t mode) {
    switch (mode) {
        case NLP_MODE_STANDARD: return "STANDARD";
        case NLP_MODE_TACTICAL: return "TACTICAL";
        case NLP_MODE_STEALTH:  return "STEALTH";
        default:                return "UNKNOWN";
    }
}

const char* nlp_emcon_name(nlp_emcon_level_t level) {
    switch (level) {
        case NLP_EMCON_NORMAL:    return "NORMAL";
        case NLP_EMCON_REDUCED:   return "REDUCED";
        case NLP_EMCON_RECEIVE:   return "RECEIVE_ONLY";
        case NLP_EMCON_SILENT:    return "SILENT";
        case NLP_EMCON_EMERGENCY: return "EMERGENCY";
        default:                  return "UNKNOWN";
    }
}

uint32_t nlp_generate_brain_id(void) {
    uint32_t id = 0;

    // Use time and random for uniqueness
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    id = (uint32_t)ts.tv_sec ^ (uint32_t)ts.tv_nsec;
    id ^= (uint32_t)getpid() << 16;
    id ^= nimcp_rand_uint(UINT32_MAX);

    // Ensure non-zero
    if (id == 0) id = 1;

    return id;
}

//=============================================================================
// Thread Functions
//=============================================================================

static void* nlp_recv_thread(void* arg) {
    nlp_node_t node = (nlp_node_t)arg;
    uint8_t buffer[65536];
    struct sockaddr_in from;
    socklen_t from_len;

    NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME, "Receive thread started");

    while (nlp_should_continue(node)) {
        from_len = sizeof(from);

        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(node->socket_fd, &read_fds);

        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000;  // 100ms timeout

        int ready = select(node->socket_fd + 1, &read_fds, NULL, NULL, &tv);
        if (ready <= 0) continue;

        ssize_t len = recvfrom(node->socket_fd, buffer, sizeof(buffer), 0,
                               (struct sockaddr*)&from, &from_len);

        if (len < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            if (!nlp_should_continue(node)) break;
            NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "recvfrom error: %s", strerror(errno));
            continue;
        }

        // Minimum size is header only for unencrypted handshake messages
        // Encrypted messages need header + auth tag (NLP_MIN_MESSAGE_SIZE)
        if (len < NLP_HEADER_SIZE) {
            NIMCP_LOGGING_WARN(NLP_MODULE_NAME, "Message too short: %zd", len);
            continue;
        }

        // Process message
        nlp_process_message(node, buffer, len, &from);
    }

    NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME, "Receive thread stopped");
    return NULL;
}

static void* nlp_heartbeat_thread(void* arg) {
    nlp_node_t node = (nlp_node_t)arg;

    NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME, "Heartbeat thread started");

    while (nlp_should_continue(node)) {
        usleep(node->config.heartbeat_interval_ms * NIMCP_US_PER_MS);

        if (!nlp_should_continue(node)) break;

        uint64_t now = nimcp_platform_time_monotonic_ms();

        nimcp_mutex_lock(&node->peer_mutex);
        for (uint32_t i = 0; i < node->peer_count; i++) {
            nlp_peer_t* peer = &node->peers[i];

            if (peer->session_state != NLP_SESSION_ESTABLISHED) continue;

            // Check for timeout
            if (now - peer->last_seen_ms > node->config.session_timeout_ms) {
                peer->missed_heartbeats++;
                if (peer->missed_heartbeats >= 3) {
                    peer->healthy = false;
                    NIMCP_LOGGING_WARN(NLP_MODULE_NAME, "Peer 0x%08X unhealthy", peer->peer_id);
                }
            }

            // Send heartbeat
            if (now - peer->last_sent_ms > node->config.heartbeat_interval_ms) {
                nimcp_mutex_unlock(&node->peer_mutex);
                nlp_send(node, peer->peer_id, NLP_MSG_HEARTBEAT, NULL, 0,
                        NLP_PRIORITY_LOW);
                nimcp_mutex_lock(&node->peer_mutex);
                peer->last_sent_ms = now;
            }
        }
        nimcp_mutex_unlock(&node->peer_mutex);

        // Bio-async processing done via external router if registered
        // (node->bio_module_ctx managed by bio-router)
    }

    NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME, "Heartbeat thread stopped");
    return NULL;
}

static void* nlp_stealth_thread(void* arg) {
    nlp_node_t node = (nlp_node_t)arg;

    NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME, "Stealth thread started");

    while (nlp_stealth_active(node)) {
        uint64_t now = nimcp_platform_time_monotonic_ms();

        // Check if it's time for a burst
        if (now >= node->next_burst_time) {
            nimcp_mutex_lock(&node->burst_mutex);

            if (node->burst_buffer_used > 0 &&
                node->emcon_level != NLP_EMCON_RECEIVE &&
                node->emcon_level != NLP_EMCON_SILENT) {

                // Send buffered messages as a burst
                NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME, "Burst TX: %zu bytes",
                               node->burst_buffer_used);

                // Broadcast burst to all peers
                nimcp_mutex_lock(&node->peer_mutex);
                for (uint32_t i = 0; i < node->peer_count; i++) {
                    if (node->peers[i].session_state == NLP_SESSION_ESTABLISHED) {
                        struct sockaddr_in addr;
                        addr.sin_family = AF_INET;
                        addr.sin_port = htons(node->peers[i].port);
                        inet_pton(AF_INET, node->peers[i].address, &addr.sin_addr);

                        sendto(node->socket_fd, node->burst_buffer,
                               node->burst_buffer_used, 0,
                               (struct sockaddr*)&addr, sizeof(addr));
                    }
                }
                nimcp_mutex_unlock(&node->peer_mutex);

                node->burst_buffer_used = 0;
            }

            // Generate chaff if needed
            if (node->emcon_level == NLP_EMCON_NORMAL) {
                // Random chaff to mask traffic patterns
                nlp_message_t* chaff = nlp_message_create_chaff(node->brain_id, 0);
                if (chaff) {
                    // Optionally send chaff - don't actually send in reduced EMCON
                    nlp_message_destroy(chaff);
                }
            }

            nimcp_mutex_unlock(&node->burst_mutex);

            node->next_burst_time = now + node->config.burst_interval_s * NIMCP_MS_PER_SEC;
        }

        usleep(100000);  // 100ms
    }

    NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME, "Stealth thread stopped");
    return NULL;
}

//=============================================================================
// Internal Functions
//=============================================================================

static int nlp_send_raw(nlp_node_t node, uint32_t peer_id,
                        const uint8_t* data, size_t len) {
    // Find peer address
    struct sockaddr_in addr;
    bool found = false;

    nimcp_mutex_lock(&node->peer_mutex);
    for (uint32_t i = 0; i < node->peer_count; i++) {
        if (node->peers[i].peer_id == peer_id) {
            addr.sin_family = AF_INET;
            addr.sin_port = htons(node->peers[i].port);
            inet_pton(AF_INET, node->peers[i].address, &addr.sin_addr);
            found = true;
            break;
        }
    }
    nimcp_mutex_unlock(&node->peer_mutex);

    if (!found) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "Peer not found: 0x%08X", peer_id);
        return -ENOENT;
    }

    ssize_t sent = sendto(node->socket_fd, data, len, 0,
                          (struct sockaddr*)&addr, sizeof(addr));

    if (sent < 0) {
        NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "sendto failed: %s", strerror(errno));
        return -errno;
    }

    return 0;
}

static int nlp_process_message(nlp_node_t node, const uint8_t* data, size_t len,
                               const struct sockaddr_in* from) {
    // Parse header
    if (len < NLP_HEADER_SIZE) {
        NIMCP_LOGGING_WARN(NLP_MODULE_NAME, "Message too short for header");
        return -EINVAL;
    }

    nlp_header_t header;
    memcpy(&header, data, sizeof(header));

    // Verify CRC
    uint16_t received_crc = ntohs(header.header_crc);
    header.header_crc = 0;
    uint16_t computed_crc = nlp_header_crc(&header);
    header.header_crc = htons(received_crc);

    if (received_crc != computed_crc) {
        NIMCP_LOGGING_WARN(NLP_MODULE_NAME, "Header CRC mismatch: 0x%04X vs 0x%04X",
                       received_crc, computed_crc);
        return -EINVAL;
    }

    // Extract fields
    uint8_t version = NLP_GET_VERSION(&header);
    nlp_mode_t mode = (nlp_mode_t)NLP_GET_MODE(&header);
    nlp_priority_t priority = (nlp_priority_t)NLP_GET_PRIORITY(&header);
    uint8_t key_slot = NLP_GET_KEY_SLOT(&header);
    uint8_t flags = NLP_GET_FLAGS(&header);

    uint32_t sender_id = ntohl(header.sender_id);
    uint32_t timestamp = ntohl(header.timestamp);
    nlp_msg_type_t msg_type = (nlp_msg_type_t)ntohs(header.msg_type);
    uint16_t payload_len = ntohs(header.payload_len);
    uint32_t dest_id = ntohl(header.dest_id);

    // Validate timestamp (replay protection)
    uint32_t now = (uint32_t)time(NULL);
    if (abs((int)(now - timestamp)) > NLP_TIMESTAMP_WINDOW) {
        NIMCP_LOGGING_WARN(NLP_MODULE_NAME, "Message timestamp outside window: %u vs %u",
                       timestamp, now);
        nimcp_mutex_lock(&node->stats_mutex);
        node->stats.replay_attacks_blocked++;
        nimcp_mutex_unlock(&node->stats_mutex);
        return -EINVAL;
    }

    // Check destination
    if (dest_id != 0 && dest_id != node->brain_id) {
        // Not for us - relay if in mesh mode
        if (node->current_mode == NLP_MODE_TACTICAL) {
            NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME, "Relaying message to 0x%08X", dest_id);
            nlp_relay(node, dest_id, msg_type, data + NLP_HEADER_SIZE,
                     len - NLP_HEADER_SIZE - NLP_TAG_SIZE);
        }
        return 0;
    }

    // Find or create peer
    uint32_t peer_id = sender_id;
    nlp_peer_t* peer = NULL;

    // Extract source address:port from incoming packet
    char from_addr_str[INET_ADDRSTRLEN] = {0};
    uint16_t from_port = 0;
    if (from) {
        inet_ntop(AF_INET, &from->sin_addr, from_addr_str, sizeof(from_addr_str));
        from_port = ntohs(from->sin_port);
    }

    nimcp_mutex_lock(&node->peer_mutex);

    // Always look up peer by address:port first for consistency
    // This works for both initiated connections (hash-based peer_id) and
    // received connections (brain_id-based peer_id)
    if (from) {
        for (uint32_t i = 0; i < node->peer_count; i++) {
            if (strcmp(node->peers[i].address, from_addr_str) == 0 &&
                node->peers[i].port == from_port) {
                peer = &node->peers[i];
                break;
            }
        }
    }

    // Fallback to peer_id lookup if address:port lookup failed
    // This handles cases where address might differ (NAT, etc.)
    if (!peer) {
        for (uint32_t i = 0; i < node->peer_count; i++) {
            if (node->peers[i].peer_id == sender_id) {
                peer = &node->peers[i];
                break;
            }
        }
    }

    // Auto-create peer for handshake requests from unknown senders
    if (!peer && msg_type == NLP_MSG_HANDSHAKE_REQ && from != NULL) {
        if (node->peer_count < node->config.max_peers) {
            // Extract address from sockaddr
            char addr_str[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &from->sin_addr, addr_str, sizeof(addr_str));
            uint16_t port = ntohs(from->sin_port);

            // Create new peer
            peer = &node->peers[node->peer_count];
            memset(peer, 0, sizeof(nlp_peer_t));

            peer->peer_id = sender_id;
            strncpy(peer->address, addr_str, sizeof(peer->address) - 1);
            peer->port = port;
            peer->session_state = NLP_SESSION_DISCONNECTED;
            peer->healthy = false;

            uint64_t now_ms = nimcp_platform_time_monotonic_ms();
            peer->last_seen_ms = now_ms;
            peer->last_sent_ms = now_ms;

            node->peer_count++;

            NIMCP_LOGGING_INFO(NLP_MODULE_NAME,
                "Auto-created peer 0x%08X from handshake request at %s:%u",
                sender_id, addr_str, port);
        } else {
            NIMCP_LOGGING_WARN(NLP_MODULE_NAME,
                "Cannot auto-create peer 0x%08X: max peers reached", sender_id);
        }
    }

    nimcp_mutex_unlock(&node->peer_mutex);

    // Decrypt payload if present
    uint8_t* decrypted = NULL;
    if (payload_len > 0 && (flags & NLP_FLAG_ENCRYPTED)) {
        decrypted = (uint8_t*)nimcp_malloc(payload_len);
        if (!decrypted) return -ENOMEM;

        // Get decryption key
        uint8_t* key = NULL;
        if (peer && peer->session_state == NLP_SESSION_ESTABLISHED) {
            key = peer->session_key;
        } else if (key_slot < NLP_KEY_SLOTS) {
            nimcp_mutex_lock(&node->key_mutex);
            if (node->psk_slots[key_slot].active) {
                key = node->psk_slots[key_slot].key;
            }
            nimcp_mutex_unlock(&node->key_mutex);
        }

        if (!key) {
            NIMCP_LOGGING_ERROR(NLP_MODULE_NAME, "No decryption key available");
            nimcp_free(decrypted);
            return -ENOKEY;
        }

        const uint8_t* encrypted = data + NLP_HEADER_SIZE;
        // Auth tag position is AFTER the encrypted payload, NOT at end of buffer
        // In STEALTH mode, messages are padded to fixed size so len > actual message size
        // The actual auth_tag is at NLP_HEADER_SIZE + payload_len
        const uint8_t* auth_tag = data + NLP_HEADER_SIZE + payload_len;

        // Decrypt with node's crypto state, using raw wire header as AAD
        // IMPORTANT: Use the original wire bytes (data), not the local header struct,
        // because the sender used the serialized header bytes for AAD
        int rc = nlp_crypto_decrypt(node, key, header.nonce,
                                    encrypted, payload_len,
                                    data, NLP_HEADER_SIZE,
                                    auth_tag,
                                    decrypted, payload_len);
        if (rc < 0) {
            NIMCP_LOGGING_WARN(NLP_MODULE_NAME, "Decryption failed");
            nimcp_mutex_lock(&node->stats_mutex);
            node->stats.decryption_errors++;
            nimcp_mutex_unlock(&node->stats_mutex);
            nimcp_free(decrypted);
            return rc;
        }
    }

    // Update peer stats
    if (peer) {
        peer->last_seen_ms = nimcp_platform_time_monotonic_ms();
        peer->messages_received++;
        peer->bytes_received += len;
        peer->missed_heartbeats = 0;
        peer->healthy = true;
    }

    // Update global stats
    nimcp_mutex_lock(&node->stats_mutex);
    node->stats.messages_received++;
    node->stats.bytes_received += len;
    nimcp_mutex_unlock(&node->stats_mutex);

    // Handle message by type
    switch (msg_type) {
        case NLP_MSG_HANDSHAKE_REQ:
            if (peer) {
                int req_rc = nlp_session_handle_handshake_req(node, peer, &header);
                if (req_rc == NIMCP_SUCCESS && node->current_mode == NLP_MODE_STANDARD) {
                    // Send handshake response
                    nlp_send(node, peer->peer_id, NLP_MSG_HANDSHAKE_RESP,
                             NULL, 0, NLP_PRIORITY_HIGH);
                    NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
                        "Sent handshake response to peer 0x%08X", peer->peer_id);
                }
            }
            break;

        case NLP_MSG_HANDSHAKE_RESP:
            if (peer) {
                int resp_rc = nlp_session_handle_handshake_resp(node, peer, &header);
                if (resp_rc == NIMCP_SUCCESS && node->current_mode == NLP_MODE_STANDARD) {
                    // Send final handshake to complete 3-way handshake
                    nlp_send(node, peer->peer_id, NLP_MSG_HANDSHAKE_FINAL,
                             NULL, 0, NLP_PRIORITY_HIGH);
                    NIMCP_LOGGING_DEBUG(NLP_MODULE_NAME,
                        "Sent handshake final to peer 0x%08X", peer->peer_id);

                    // Establish session on our side after sending final
                    nlp_session_establish(node, peer);
                }
            }
            break;

        case NLP_MSG_HANDSHAKE_FINAL:
            if (peer) {
                nlp_session_handle_handshake_final(node, peer, &header);
                // nlp_session_handle_handshake_final calls nlp_session_establish internally
            }
            break;

        case NLP_MSG_HEARTBEAT:
            // Just update peer stats (already done above)
            break;

        case NLP_MSG_ACK:
        case NLP_MSG_NACK:
            // Handle acknowledgments
            break;

        default:
            // Pass to user callback
            if (node->message_callback && peer) {
                nlp_message_t msg;
                memcpy(&msg.header, &header, sizeof(header));
                // Use decrypted payload if available, otherwise raw payload
                if (decrypted) {
                    msg.payload = decrypted;
                } else if (payload_len > 0) {
                    msg.payload = (uint8_t*)(data + NLP_HEADER_SIZE);
                } else {
                    msg.payload = NULL;
                }
                memcpy(msg.auth_tag, data + len - NLP_TAG_SIZE, NLP_TAG_SIZE);

                node->message_callback(node, peer, &msg, node->user_data);
            }
            break;
    }

    if (decrypted) {
        nimcp_free(decrypted);
    }

    return 0;
}
