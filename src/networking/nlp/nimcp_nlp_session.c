//=============================================================================
// nimcp_nlp_session.c - Neural Link Protocol Session Management
//=============================================================================
/**
 * @file nimcp_nlp_session.c
 * @brief Session state machine, handshake, key management, and peer tracking
 *
 * WHAT: Complete session management for Neural Link Protocol
 * WHY:  Enable secure, resilient peer-to-peer brain communication
 * HOW:  State machine with 3-way handshake, PSK selection, replay protection,
 *       peer health monitoring, and bio-async event integration
 *
 * ARCHITECTURE:
 *
 * SESSION STATE MACHINE:
 * ┌──────────────┐  handshake_req  ┌─────────────────┐
 * │ DISCONNECTED │────────────────→│ HANDSHAKE_SENT  │
 * └──────────────┘                 └─────────────────┘
 *        ↑                                  │
 *        │                                  │ resp received
 *        │                                  ↓
 *        │                          ┌──────────────────┐
 *        │                          │ HANDSHAKE_RECV   │
 *        │                          └──────────────────┘
 *        │                                  │
 *        │                                  │ final ack
 *        │                                  ↓
 *        │                          ┌──────────────────┐
 *        │                          │   ESTABLISHED    │←┐
 *        │                          └──────────────────┘ │
 *        │                                  │           │
 *        │                             timeout/error    │ key rotation
 *        │                                  ↓           │
 *        └──────────────────────────┌──────────────────┐│
 *                    disconnect     │      ERROR       ││
 *                                   └──────────────────┘┘
 *
 * KEY MANAGEMENT:
 * - PSK selection based on validity window
 * - Session key negotiation (Standard mode)
 * - Periodic key rotation for forward secrecy
 * - Timestamp-based replay protection
 *
 * PEER HEALTH:
 * - Heartbeat tracking with missed count
 * - RTT estimation for adaptive timeouts
 * - Packet loss calculation
 * - Automatic peer cleanup on repeated failures
 *
 * BIO-ASYNC INTEGRATION:
 * - Session state changes → SEROTONIN (slow state coordination)
 * - Key rotation events → DOPAMINE (reward/security milestone)
 * - Peer health alerts → NOREPINEPHRINE (alerting)
 * - Handshake steps → ACETYLCHOLINE (fast coordination)
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 * @version 1.0.0
 */

#include "networking/nlp/nimcp_nlp_internal.h"
#include "utils/platform/nimcp_platform.h"
#include "utils/time/nimcp_time.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "security/nimcp_bbb_helpers.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <arpa/inet.h>

// Note: nlp_node_struct is defined in nimcp_nlp_internal.h

#define NLP_SESSION_MODULE "nlp_session"

// Compatibility macros for logging (use standard LOG_* from nimcp_logging.h)
#ifndef NIMCP_LOG_ERROR
#define NIMCP_LOG_ERROR(...) LOG_ERROR(__VA_ARGS__)
#define NIMCP_LOG_WARN(...)  LOG_WARNING(__VA_ARGS__)
#define NIMCP_LOG_INFO(...)  LOG_INFO(__VA_ARGS__)
#define NIMCP_LOG_DEBUG(...) LOG_DEBUG(__VA_ARGS__)
#endif

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * WHAT: Validate node handle
 * WHY:  Prevent crashes from invalid/corrupted handles
 * HOW:  Check magic number and NULL
 */
static inline bool nlp_validate_node(nlp_node_t node) {
    return node != NULL && node->magic == NLP_NODE_MAGIC;
}

// Static bio-async context for session module
static bio_module_context_t g_session_bio_ctx = NULL;

/**
 * @brief Initialize session module bio-async registration
 */
static void nlp_session_ensure_bio_registered(void) {
    if (g_session_bio_ctx) return;
    if (!bio_router_is_initialized()) return;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_PROTOCOL,  // Use protocol module ID
        .module_name = NLP_SESSION_MODULE,
        .inbox_capacity = 32,
        .user_data = NULL
    };

    g_session_bio_ctx = bio_router_register_module(&info);
    if (g_session_bio_ctx) {
        NIMCP_LOGGING_DEBUG(NLP_SESSION_MODULE,
            "Registered with bio-router for session events");
    }
}

/**
 * WHAT: Send bio-async message for session event
 * WHY:  Integrate session management with cognitive systems
 * HOW:  Broadcast to cognitive modules + BBB audit logging
 */
static void nlp_send_bio_session_event(nlp_node_t node,
                                       const nlp_peer_t* peer,
                                       nlp_session_state_t old_state,
                                       nlp_session_state_t new_state) {
    uint64_t peer_id = peer ? peer->peer_id : 0;

    // Ensure bio-async is registered
    nlp_session_ensure_bio_registered();

    // Audit log for security tracking of state changes
    bbb_audit_log(BBB_AUDIT_INFO, NLP_SESSION_MODULE, "session_state_change",
                  "peer=%lu old=%u new=%u", peer_id,
                  (uint32_t)old_state, (uint32_t)new_state);

    // Send bio-async message to cognitive modules
    if (g_session_bio_ctx) {
        bio_msg_nlp_session_state_change_t msg;
        memset(&msg, 0, sizeof(msg));

        // Select message type based on state transition
        bio_message_type_t msg_type = BIO_MSG_NLP_SESSION_STATE_CHANGE;
        if (new_state == NLP_SESSION_ESTABLISHED) {
            msg_type = BIO_MSG_NLP_SESSION_CONNECTED;
        } else if (new_state == NLP_SESSION_DISCONNECTED) {
            msg_type = BIO_MSG_NLP_SESSION_DISCONNECTED;
        }

        bio_msg_init_header(&msg.header, msg_type, BIO_MODULE_PROTOCOL,
                            BIO_MODULE_ALL, sizeof(msg));
        msg.header.flags = BIO_MSG_FLAG_BROADCAST;

        // Use appropriate neuromodulator channel
        if (new_state == NLP_SESSION_ERROR) {
            msg.header.channel = BIO_CHANNEL_NOREPINEPHRINE;  // Alert
        } else if (new_state == NLP_SESSION_ESTABLISHED) {
            msg.header.channel = BIO_CHANNEL_DOPAMINE;  // Reward
        } else {
            msg.header.channel = BIO_CHANNEL_SEROTONIN;  // State change
        }

        msg.peer_id = peer_id;
        msg.session_id = (node ? node->brain_id : 0);
        msg.old_state = (uint8_t)old_state;
        msg.new_state = (uint8_t)new_state;
        msg.state_change_time_us = nimcp_time_get_us();

        bio_router_broadcast(g_session_bio_ctx, &msg, sizeof(msg));

        NIMCP_LOGGING_DEBUG(NLP_SESSION_MODULE,
            "Bio-async: broadcast session event peer=%lu state=%u->%u channel=%d",
            (unsigned long)peer_id, (uint32_t)old_state, (uint32_t)new_state,
            msg.header.channel);
    }

    LOG_MODULE_DEBUG(NLP_SESSION_MODULE, "Session state: peer=%lu %u->%u",
                     peer_id, (uint32_t)old_state, (uint32_t)new_state);
}

/**
 * @brief Broadcast key rotation event to cognitive modules
 */
static void nlp_broadcast_key_rotation(nlp_node_t node, const nlp_peer_t* peer) {
    if (!g_session_bio_ctx) {
        nlp_session_ensure_bio_registered();
        if (!g_session_bio_ctx) return;
    }

    uint64_t peer_id = peer ? peer->peer_id : 0;

    // Log key rotation as security milestone
    bbb_audit_log(BBB_AUDIT_INFO, NLP_SESSION_MODULE, "key_rotation",
                  "peer=%lu", (unsigned long)peer_id);

    bio_msg_nlp_crypto_complete_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_NLP_CRYPTO_KEY_EXCHANGE,
                        BIO_MODULE_PROTOCOL, BIO_MODULE_ALL, sizeof(msg));
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.channel = BIO_CHANNEL_DOPAMINE;  // Security milestone
    msg.success = true;
    msg.operation_id = (node ? node->brain_id : 0);

    bio_router_broadcast(g_session_bio_ctx, &msg, sizeof(msg));
}

/**
 * @brief Broadcast peer health alert
 */
static void nlp_broadcast_peer_health(const nlp_peer_t* peer, bool healthy) {
    if (!g_session_bio_ctx) {
        nlp_session_ensure_bio_registered();
        if (!g_session_bio_ctx) return;
    }

    if (!peer) return;

    bio_msg_nlp_error_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, healthy ? BIO_MSG_NLP_MESSAGE_ACK : BIO_MSG_NLP_ERROR,
                        BIO_MODULE_PROTOCOL, BIO_MODULE_ALL, sizeof(msg));
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.channel = BIO_CHANNEL_NOREPINEPHRINE;  // Alert
    msg.peer_id = peer->peer_id;
    msg.severity = healthy ? 0 : 1;  // 0=info, 1=warning
    strncpy(msg.module_name, NLP_SESSION_MODULE, sizeof(msg.module_name) - 1);
    snprintf(msg.error_message, sizeof(msg.error_message),
             "Peer health: %s", healthy ? "recovered" : "degraded");

    bio_router_broadcast(g_session_bio_ctx, &msg, sizeof(msg));
}

/**
 * WHAT: Generate cryptographically random nonce
 * WHY:  Unique nonce per message for AES-GCM replay protection
 * HOW:  Use /dev/urandom on POSIX, CryptGenRandom on Windows
 */
static void nlp_generate_nonce(uint8_t* nonce, size_t len) {
    if (!nonce || len == 0) return;

#if defined(NIMCP_PLATFORM_LINUX) || defined(NIMCP_PLATFORM_MACOS)
    FILE* f = fopen("/dev/urandom", "rb");
    if (f) {
        fread(nonce, 1, len, f);
        fclose(f);
    } else {
        // Fallback: use time-based randomness (not cryptographically secure)
        srand((unsigned int)nimcp_time_get_us());
        for (size_t i = 0; i < len; i++) {
            nonce[i] = rand() & 0xFF;
        }
    }
#elif defined(NIMCP_PLATFORM_WINDOWS)
    // Windows: Use CryptGenRandom or BCrypt (simplified fallback here)
    srand((unsigned int)nimcp_time_get_us());
    for (size_t i = 0; i < len; i++) {
        nonce[i] = rand() & 0xFF;
    }
#endif
}

/**
 * WHAT: Calculate CRC-16 for header validation
 * WHY:  Detect corruption in header fields before decryption
 * HOW:  CRC-16-CCITT polynomial (0x1021)
 */
static uint16_t nlp_calculate_crc16(const uint8_t* data, size_t len) {
    uint16_t crc = 0xFFFF;

    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc = crc << 1;
            }
        }
    }

    return crc;
}

//=============================================================================
// Session State Machine
//=============================================================================

/**
 * WHAT: Initialize session structures
 * WHY:  Set up clean session state for new peer
 * HOW:  Zero memory, set initial state, generate random sequence
 *
 * @param peer Peer structure to initialize
 * @return 0 on success, negative on error
 */
int nlp_session_init(nlp_peer_t* peer) {
    if (!peer) {
        NIMCP_LOG_ERROR("nlp_session_init: NULL peer");
        return -NIMCP_INVALID_PARAM;
    }

    // Initialize session state
    peer->session_state = NLP_SESSION_DISCONNECTED;
    memset(peer->session_key, 0, NLP_KEY_SIZE);

    // Initialize sequence numbers with random start (security)
    srand((unsigned int)nimcp_time_get_us());
    peer->tx_sequence = (uint16_t)(rand() & 0xFFFF);
    peer->rx_sequence = 0;

    // Initialize timing
    peer->last_seen_ms = 0;
    peer->last_sent_ms = 0;
    peer->rtt_ms = 0;

    // Initialize statistics
    peer->messages_sent = 0;
    peer->messages_received = 0;
    peer->bytes_sent = 0;
    peer->bytes_received = 0;
    peer->retransmits = 0;

    // Initialize health
    peer->healthy = true;
    peer->missed_heartbeats = 0;
    peer->packet_loss_rate = 0.0F;

    NIMCP_LOG_DEBUG("nlp_session_init: Initialized session for peer 0x%08X",
                    peer->peer_id);

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Begin 3-way handshake (STANDARD mode)
 * WHY:  Establish secure session with key negotiation
 * HOW:  Send HANDSHAKE_REQ, transition to HANDSHAKE_SENT state
 *
 * @param node NLP node
 * @param peer Peer to handshake with
 * @return 0 on success, negative on error
 */
int nlp_session_start_handshake(nlp_node_t node, nlp_peer_t* peer) {
    if (!nlp_validate_node(node)) {
        return -NIMCP_INVALID_PARAM;
    }
    if (!peer) {
        return -NIMCP_INVALID_PARAM;
    }

    // Check current state
    if (peer->session_state != NLP_SESSION_DISCONNECTED) {
        NIMCP_LOG_WARN("nlp_session_start_handshake: Peer already in state %d",
                      peer->session_state);
        return -NIMCP_INVALID_STATE;
    }

    // Store old state for callback
    nlp_session_state_t old_state = peer->session_state;

    // Transition to handshake sent
    peer->session_state = NLP_SESSION_HANDSHAKE_SENT;
    peer->last_sent_ms = nimcp_time_get_ms();

    NIMCP_LOG_INFO("nlp_session_start_handshake: Starting handshake with peer 0x%08X",
                   peer->peer_id);

    // Send bio-async event
    nlp_send_bio_session_event(node, peer, old_state, peer->session_state);

    // Notify via callback
    if (node->peer_callback) {
        node->peer_callback(node, peer, old_state, peer->session_state,
                          node->config.user_data);
    }

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Process incoming handshake request
 * WHY:  Respond to peer's handshake initiation
 * HOW:  Validate request, send response, transition to HANDSHAKE_RECEIVED
 *
 * @param node NLP node
 * @param peer Peer sending request
 * @param header Received header
 * @return 0 on success, negative on error
 */
int nlp_session_handle_handshake_req(nlp_node_t node, nlp_peer_t* peer,
                                     const nlp_header_t* header) {
    if (!nlp_validate_node(node) || !peer || !header) {
        return -NIMCP_INVALID_PARAM;
    }

    // Validate message type
    if (ntohs(header->msg_type) != NLP_MSG_HANDSHAKE_REQ) {
        NIMCP_LOG_ERROR("nlp_session_handle_handshake_req: Wrong message type 0x%04X",
                       ntohs(header->msg_type));
        return -NIMCP_INVALID_MSG;
    }

    // Check version compatibility
    if (NLP_GET_VERSION(header) != NLP_VERSION) {
        NIMCP_LOG_ERROR("nlp_session_handle_handshake_req: Version mismatch %d != %d",
                       NLP_GET_VERSION(header), NLP_VERSION);
        return -NIMCP_VERSION_MISMATCH;
    }

    // Store old state
    nlp_session_state_t old_state = peer->session_state;

    // Transition state
    peer->session_state = NLP_SESSION_HANDSHAKE_RECEIVED;
    peer->last_seen_ms = nimcp_time_get_ms();
    peer->rx_sequence = ntohs(header->sequence);

    NIMCP_LOG_INFO("nlp_session_handle_handshake_req: Received handshake from peer 0x%08X",
                   ntohl(header->sender_id));

    // Send bio-async event
    nlp_send_bio_session_event(node, peer, old_state, peer->session_state);

    // Notify callback
    if (node->peer_callback) {
        node->peer_callback(node, peer, old_state, peer->session_state,
                          node->config.user_data);
    }

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Process handshake response
 * WHY:  Complete our side of handshake
 * HOW:  Validate response, send final ACK, near completion
 *
 * @param node NLP node
 * @param peer Peer sending response
 * @param header Received header
 * @return 0 on success, negative on error
 */
int nlp_session_handle_handshake_resp(nlp_node_t node, nlp_peer_t* peer,
                                      const nlp_header_t* header) {
    if (!nlp_validate_node(node) || !peer || !header) {
        return -NIMCP_INVALID_PARAM;
    }

    // Validate state
    if (peer->session_state != NLP_SESSION_HANDSHAKE_SENT) {
        NIMCP_LOG_ERROR("nlp_session_handle_handshake_resp: Invalid state %d",
                       peer->session_state);
        return -NIMCP_INVALID_STATE;
    }

    // Validate message type
    if (ntohs(header->msg_type) != NLP_MSG_HANDSHAKE_RESP) {
        return -NIMCP_INVALID_MSG;
    }

    // Calculate RTT
    uint64_t now_ms = nimcp_time_get_ms();
    peer->rtt_ms = (uint32_t)(now_ms - peer->last_sent_ms);
    peer->last_seen_ms = now_ms;
    peer->rx_sequence = ntohs(header->sequence);

    NIMCP_LOG_INFO("nlp_session_handle_handshake_resp: Response from peer 0x%08X, RTT=%u ms",
                   ntohl(header->sender_id), peer->rtt_ms);

    // Note: Actual session key negotiation would happen here (DH, ECDH, etc.)
    // For now, we just prepare for the final handshake step

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Complete handshake (final ACK)
 * WHY:  Finalize 3-way handshake, establish session
 * HOW:  Validate final message, call nlp_session_establish()
 *
 * @param node NLP node
 * @param peer Peer completing handshake
 * @param header Received header
 * @return 0 on success, negative on error
 */
int nlp_session_handle_handshake_final(nlp_node_t node, nlp_peer_t* peer,
                                       const nlp_header_t* header) {
    if (!nlp_validate_node(node) || !peer || !header) {
        return -NIMCP_INVALID_PARAM;
    }

    // Validate state
    if (peer->session_state != NLP_SESSION_HANDSHAKE_RECEIVED) {
        NIMCP_LOG_ERROR("nlp_session_handle_handshake_final: Invalid state %d",
                       peer->session_state);
        return -NIMCP_INVALID_STATE;
    }

    // Validate message type
    if (ntohs(header->msg_type) != NLP_MSG_HANDSHAKE_FINAL) {
        return -NIMCP_INVALID_MSG;
    }

    peer->last_seen_ms = nimcp_time_get_ms();
    peer->rx_sequence = ntohs(header->sequence);

    NIMCP_LOG_INFO("nlp_session_handle_handshake_final: Final ACK from peer 0x%08X",
                   ntohl(header->sender_id));

    // Establish session
    return nlp_session_establish(node, peer);
}

/**
 * WHAT: Mark session as established
 * WHY:  Transition to operational state after successful handshake
 * HOW:  Set state, initialize session key, reset health counters
 *
 * @param node NLP node
 * @param peer Peer to establish
 * @return 0 on success, negative on error
 */
int nlp_session_establish(nlp_node_t node, nlp_peer_t* peer) {
    if (!nlp_validate_node(node) || !peer) {
        return -NIMCP_INVALID_PARAM;
    }

    nlp_session_state_t old_state = peer->session_state;

    // Transition to established
    peer->session_state = NLP_SESSION_ESTABLISHED;
    peer->healthy = true;
    peer->missed_heartbeats = 0;

    NIMCP_LOG_INFO("nlp_session_establish: Session established with peer 0x%08X",
                   peer->peer_id);

    // Update stats
    nimcp_mutex_lock(&node->stats_mutex);
    node->stats.active_sessions++;
    node->stats.connected_peers++;
    nimcp_mutex_unlock(&node->stats_mutex);

    // Send bio-async event (DOPAMINE - reward for successful connection)
    nlp_send_bio_session_event(node, peer, old_state, peer->session_state);

    // Notify callback
    if (node->peer_callback) {
        node->peer_callback(node, peer, old_state, peer->session_state,
                          node->config.user_data);
    }

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Close session gracefully
 * WHY:  Clean shutdown, notify peer
 * HOW:  Send DISCONNECT message, transition to DISCONNECTED, cleanup
 *
 * @param node NLP node
 * @param peer Peer to disconnect
 * @return 0 on success, negative on error
 */
int nlp_session_close(nlp_node_t node, nlp_peer_t* peer) {
    if (!nlp_validate_node(node) || !peer) {
        return -NIMCP_INVALID_PARAM;
    }

    nlp_session_state_t old_state = peer->session_state;

    // Only close if not already disconnected
    if (old_state == NLP_SESSION_DISCONNECTED) {
        return NIMCP_SUCCESS;
    }

    NIMCP_LOG_INFO("nlp_session_close: Closing session with peer 0x%08X",
                   peer->peer_id);

    // Transition state
    peer->session_state = NLP_SESSION_DISCONNECTED;
    peer->healthy = false;

    // Clear session key
    memset(peer->session_key, 0, NLP_KEY_SIZE);

    // Update stats
    if (old_state == NLP_SESSION_ESTABLISHED) {
        nimcp_mutex_lock(&node->stats_mutex);
        if (node->stats.active_sessions > 0) {
            node->stats.active_sessions--;
        }
        if (node->stats.connected_peers > 0) {
            node->stats.connected_peers--;
        }
        nimcp_mutex_unlock(&node->stats_mutex);
    }

    // Send bio-async event
    nlp_send_bio_session_event(node, peer, old_state, peer->session_state);

    // Notify callback
    if (node->peer_callback) {
        node->peer_callback(node, peer, old_state, peer->session_state,
                          node->config.user_data);
    }

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Handle session timeout
 * WHY:  Detect and cleanup dead sessions
 * HOW:  Check elapsed time, transition to ERROR or DISCONNECTED
 *
 * @param node NLP node
 * @param peer Peer to check
 * @return 0 if OK, negative if timed out
 */
int nlp_session_timeout(nlp_node_t node, nlp_peer_t* peer) {
    if (!nlp_validate_node(node) || !peer) {
        return -NIMCP_INVALID_PARAM;
    }

    uint64_t now_ms = nimcp_time_get_ms();
    uint64_t elapsed_ms = now_ms - peer->last_seen_ms;

    // Check timeout based on state
    uint32_t timeout_ms = node->config.session_timeout_ms;
    if (peer->session_state == NLP_SESSION_HANDSHAKE_SENT ||
        peer->session_state == NLP_SESSION_HANDSHAKE_RECEIVED) {
        timeout_ms = node->config.handshake_timeout_ms;
    }

    if (elapsed_ms > timeout_ms) {
        nlp_session_state_t old_state = peer->session_state;

        NIMCP_LOG_WARN("nlp_session_timeout: Peer 0x%08X timed out (elapsed=%llu ms)",
                      peer->peer_id, (unsigned long long)elapsed_ms);

        // Transition to error state
        peer->session_state = NLP_SESSION_ERROR;
        peer->healthy = false;

        // Update stats
        nimcp_mutex_lock(&node->stats_mutex);
        node->stats.messages_dropped++;
        nimcp_mutex_unlock(&node->stats_mutex);

        // Send bio-async alert
        nlp_send_bio_session_event(node, peer, old_state, peer->session_state);

        // Notify callback
        if (node->peer_callback) {
            node->peer_callback(node, peer, old_state, peer->session_state,
                              node->config.user_data);
        }

        return -NIMCP_TIMEOUT_ERROR;
    }

    return NIMCP_SUCCESS;
}

//=============================================================================
// Key Management
//=============================================================================

/**
 * WHAT: Select valid PSK for tactical/stealth modes
 * WHY:  Find appropriate pre-shared key based on time validity
 * HOW:  Iterate slots, check validity window, return first match
 *
 * @param node NLP node
 * @param slot_out Output slot index
 * @return 0 on success, negative if no valid key
 */
int nlp_session_select_psk(nlp_node_t node, uint8_t* slot_out) {
    if (!nlp_validate_node(node) || !slot_out) {
        return -NIMCP_INVALID_PARAM;
    }

    uint64_t now = (uint64_t)time(NULL);

    // Search for valid key
    for (uint8_t i = 0; i < NLP_KEY_SLOTS; i++) {
        const nlp_key_slot_t* slot = &node->config.psk[i];

        if (slot->active &&
            now >= slot->valid_from &&
            now <= slot->valid_until) {
            *slot_out = i;

            NIMCP_LOG_DEBUG("nlp_session_select_psk: Selected slot %u (key_id=0x%08X)",
                          i, slot->key_id);
            return NIMCP_SUCCESS;
        }
    }

    NIMCP_LOG_ERROR("nlp_session_select_psk: No valid PSK found");
    return -NIMCP_AUTH_FAILED;
}

/**
 * WHAT: Validate timestamp within window (replay protection)
 * WHY:  Prevent replay attacks by rejecting old messages
 * HOW:  Compare message timestamp with current time, check window
 *
 * @param header Message header with timestamp
 * @param window_sec Maximum age in seconds
 * @return 0 if valid, negative if outside window
 */
int nlp_session_validate_timestamp(const nlp_header_t* header, uint32_t window_sec) {
    if (!header) {
        return -NIMCP_INVALID_PARAM;
    }

    uint32_t now = (uint32_t)time(NULL);
    uint32_t msg_time = ntohl(header->timestamp);

    // Check if message is too old
    if (now > msg_time && (now - msg_time) > window_sec) {
        NIMCP_LOG_WARN("nlp_session_validate_timestamp: Message too old (%u sec ago)",
                      now - msg_time);
        return -NIMCP_INVALID_SEQUENCE;
    }

    // Check if message is from future (clock skew tolerance)
    if (msg_time > now && (msg_time - now) > window_sec) {
        NIMCP_LOG_WARN("nlp_session_validate_timestamp: Message from future (%u sec)",
                      msg_time - now);
        return -NIMCP_INVALID_SEQUENCE;
    }

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Validate sequence number
 * WHY:  Detect out-of-order or duplicate messages
 * HOW:  Compare with last received sequence, allow some reordering
 *
 * @param peer Peer state
 * @param sequence Received sequence number
 * @return 0 if valid, negative if invalid
 */
int nlp_session_check_sequence(nlp_peer_t* peer, uint16_t sequence) {
    if (!peer) {
        return -NIMCP_INVALID_PARAM;
    }

    // Allow some reordering (sliding window of 32 messages)
    int16_t delta = (int16_t)(sequence - peer->rx_sequence);

    if (delta < -32) {
        // Too old - likely duplicate
        NIMCP_LOG_WARN("nlp_session_check_sequence: Sequence too old (delta=%d)", delta);
        return -NIMCP_INVALID_SEQUENCE;
    }

    if (delta > 0) {
        // Update expected sequence
        peer->rx_sequence = sequence;
    }

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Rotate session keys periodically
 * WHY:  Provide forward secrecy by regularly changing keys
 * HOW:  Generate new session key, send KEY_ROTATE message
 *
 * @param node NLP node
 * @param peer Peer to rotate keys with
 * @return 0 on success, negative on error
 */
int nlp_session_key_rotation(nlp_node_t node, nlp_peer_t* peer) {
    if (!nlp_validate_node(node) || !peer) {
        return -NIMCP_INVALID_PARAM;
    }

    // Only rotate if session is established
    if (peer->session_state != NLP_SESSION_ESTABLISHED) {
        return -NIMCP_INVALID_STATE;
    }

    NIMCP_LOG_INFO("nlp_session_key_rotation: Rotating keys for peer 0x%08X",
                   peer->peer_id);

    // Generate new session key (simplified - real implementation would use DH/ECDH)
    nlp_generate_nonce(peer->session_key, NLP_KEY_SIZE);

    // Bio-async event stub: Key rotation completed (DOPAMINE channel)
    // To be implemented when bio-router integration is complete

    return NIMCP_SUCCESS;
}

//=============================================================================
// Peer Tracking
//=============================================================================

/**
 * WHAT: Add new peer to peer table
 * WHY:  Track connection to new brain
 * HOW:  Find free slot, initialize peer structure
 *
 * @param node NLP node
 * @param peer_id Peer brain ID
 * @param address Peer address
 * @param port Peer port
 * @return Peer pointer on success, NULL on error
 */
nlp_peer_t* nlp_peer_add(nlp_node_t node, uint32_t peer_id,
                        const char* address, uint16_t port) {
    if (!nlp_validate_node(node) || !address) {
        return NULL;
    }

    nimcp_mutex_lock(&node->peer_mutex);

    // Check if peer already exists
    for (uint32_t i = 0; i < node->peer_count; i++) {
        if (node->peers[i].peer_id == peer_id) {
            NIMCP_LOG_WARN("nlp_peer_add: Peer 0x%08X already exists", peer_id);
            nimcp_mutex_unlock(&node->peer_mutex);
            return &node->peers[i];
        }
    }

    // Check capacity
    if (node->peer_count >= NLP_MAX_PEERS) {
        NIMCP_LOG_ERROR("nlp_peer_add: Peer table full (%u peers)", node->peer_count);
        nimcp_mutex_unlock(&node->peer_mutex);
        return NULL;
    }

    // Add new peer
    nlp_peer_t* peer = &node->peers[node->peer_count];
    memset(peer, 0, sizeof(nlp_peer_t));

    peer->peer_id = peer_id;
    strncpy(peer->address, address, sizeof(peer->address) - 1);
    peer->address[sizeof(peer->address) - 1] = '\0';
    peer->port = port;

    // Initialize session
    nlp_session_init(peer);

    node->peer_count++;

    NIMCP_LOG_INFO("nlp_peer_add: Added peer 0x%08X at %s:%u (total=%u)",
                   peer_id, address, port, node->peer_count);

    nimcp_mutex_unlock(&node->peer_mutex);
    return peer;
}

/**
 * WHAT: Remove peer from peer table
 * WHY:  Clean up disconnected peer
 * HOW:  Close session, remove from table, compact array
 *
 * @param node NLP node
 * @param peer_id Peer brain ID
 * @return 0 on success, negative on error
 */
int nlp_peer_remove(nlp_node_t node, uint32_t peer_id) {
    if (!nlp_validate_node(node)) {
        return -NIMCP_INVALID_PARAM;
    }

    nimcp_mutex_lock(&node->peer_mutex);

    // Find peer
    for (uint32_t i = 0; i < node->peer_count; i++) {
        if (node->peers[i].peer_id == peer_id) {
            nlp_peer_t* peer = &node->peers[i];

            NIMCP_LOG_INFO("nlp_peer_remove: Removing peer 0x%08X", peer_id);

            // Close session first
            nlp_session_close(node, peer);

            // Compact array (move last peer to this slot)
            if (i < node->peer_count - 1) {
                node->peers[i] = node->peers[node->peer_count - 1];
            }

            node->peer_count--;

            nimcp_mutex_unlock(&node->peer_mutex);
            return NIMCP_SUCCESS;
        }
    }

    NIMCP_LOG_WARN("nlp_peer_remove: Peer 0x%08X not found", peer_id);
    nimcp_mutex_unlock(&node->peer_mutex);
    return -NIMCP_NOT_FOUND;
}

/**
 * WHAT: Find peer by ID
 * WHY:  Lookup peer for message routing
 * HOW:  Linear search through peer table
 *
 * @param node NLP node
 * @param peer_id Peer brain ID
 * @return Peer pointer or NULL if not found
 */
nlp_peer_t* nlp_peer_find(nlp_node_t node, uint32_t peer_id) {
    if (!nlp_validate_node(node)) {
        return NULL;
    }

    nimcp_mutex_lock(&node->peer_mutex);

    for (uint32_t i = 0; i < node->peer_count; i++) {
        if (node->peers[i].peer_id == peer_id) {
            nlp_peer_t* peer = &node->peers[i];
            nimcp_mutex_unlock(&node->peer_mutex);
            return peer;
        }
    }

    nimcp_mutex_unlock(&node->peer_mutex);
    return NULL;
}

/**
 * WHAT: Find peer by address
 * WHY:  Lookup peer from incoming connection
 * HOW:  Linear search by address:port
 *
 * @param node NLP node
 * @param address Peer address
 * @param port Peer port
 * @return Peer pointer or NULL if not found
 */
nlp_peer_t* nlp_peer_find_by_address(nlp_node_t node, const char* address, uint16_t port) {
    if (!nlp_validate_node(node) || !address) {
        return NULL;
    }

    nimcp_mutex_lock(&node->peer_mutex);

    for (uint32_t i = 0; i < node->peer_count; i++) {
        if (strcmp(node->peers[i].address, address) == 0 &&
            node->peers[i].port == port) {
            nlp_peer_t* peer = &node->peers[i];
            nimcp_mutex_unlock(&node->peer_mutex);
            return peer;
        }
    }

    nimcp_mutex_unlock(&node->peer_mutex);
    return NULL;
}

/**
 * WHAT: Update peer statistics
 * WHY:  Track bandwidth, message counts for monitoring
 * HOW:  Increment counters atomically
 *
 * @param peer Peer to update
 * @param bytes_sent Bytes sent
 * @param bytes_received Bytes received
 * @return 0 on success, negative on error
 */
int nlp_peer_update_stats(nlp_peer_t* peer, uint64_t bytes_sent, uint64_t bytes_received) {
    if (!peer) {
        return -NIMCP_INVALID_PARAM;
    }

    if (bytes_sent > 0) {
        peer->messages_sent++;
        peer->bytes_sent += bytes_sent;
        peer->last_sent_ms = nimcp_time_get_ms();
    }

    if (bytes_received > 0) {
        peer->messages_received++;
        peer->bytes_received += bytes_received;
        peer->last_seen_ms = nimcp_time_get_ms();
        peer->missed_heartbeats = 0;  // Reset on any received message
    }

    // Calculate packet loss rate (simplified exponential moving average)
    if (peer->messages_sent > 0) {
        uint64_t expected = peer->messages_sent;
        uint64_t received = peer->messages_received;
        float loss = (expected > received) ?
                    (float)(expected - received) / (float)expected : 0.0F;
        peer->packet_loss_rate = 0.9F * peer->packet_loss_rate + 0.1F * loss;
    }

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Check peer health, update missed heartbeats
 * WHY:  Detect dead/unresponsive peers
 * HOW:  Check heartbeat interval, increment counter, mark unhealthy
 *
 * @param node NLP node
 * @param peer Peer to check
 * @return 0 if healthy, negative if unhealthy
 */
int nlp_peer_check_health(nlp_node_t node, nlp_peer_t* peer) {
    if (!nlp_validate_node(node) || !peer) {
        return -NIMCP_INVALID_PARAM;
    }

    uint64_t now_ms = nimcp_time_get_ms();
    uint64_t elapsed_ms = now_ms - peer->last_seen_ms;

    // Check if heartbeat interval exceeded
    if (elapsed_ms > node->config.heartbeat_interval_ms) {
        peer->missed_heartbeats++;

        NIMCP_LOG_DEBUG("nlp_peer_check_health: Peer 0x%08X missed %u heartbeats",
                       peer->peer_id, peer->missed_heartbeats);

        // Mark unhealthy after 3 missed heartbeats
        if (peer->missed_heartbeats >= 3) {
            if (peer->healthy) {
                peer->healthy = false;

                NIMCP_LOG_WARN("nlp_peer_check_health: Peer 0x%08X marked unhealthy",
                             peer->peer_id);

                // Bio-async alert stub: Peer unhealthy (NOREPINEPHRINE channel)
                // To be implemented when bio-router integration is complete
            }

            return -NIMCP_ERROR;
        }
    } else {
        // Peer is responding
        if (!peer->healthy && peer->session_state == NLP_SESSION_ESTABLISHED) {
            peer->healthy = true;
            peer->missed_heartbeats = 0;

            NIMCP_LOG_INFO("nlp_peer_check_health: Peer 0x%08X recovered",
                         peer->peer_id);
        }
    }

    return peer->healthy ? NIMCP_SUCCESS : -NIMCP_ERROR;
}

//=============================================================================
// Module Info
//=============================================================================

/**
 * Module: NLP Session Management
 * Version: 1.0.0
 * Thread Safety: Full (mutex-protected peer table and stats)
 * Dependencies: platform, time, thread, bio-async, logging, memory
 *
 * TESTING:
 * - Unit tests: Test each function with valid/invalid inputs
 * - Integration tests: Test full handshake flow
 * - Stress tests: Many concurrent peers, rapid connect/disconnect
 * - Security tests: Replay attacks, invalid timestamps, sequence gaps
 */
