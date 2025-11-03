# NIMCP P2P Node: Heartbeat, Logging, and Data Structure Refactoring Design

**Author:** Claude (Anthropic)
**Date:** 2025-11-01
**Status:** Design Document
**Phases:** 7-9 - Heartbeat + Logging + Remove Antipatterns

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Phase 7: Heartbeat/Keepalive Mechanism](#phase-7-heartbeatkeepalive-mechanism)
3. [Phase 8: Comprehensive Logging Strategy](#phase-8-comprehensive-logging-strategy)
4. [Phase 9: Remove Dual Data Structure Antipattern](#phase-9-remove-dual-data-structure-antipattern)
5. [Implementation Order](#implementation-order)
6. [Testing Strategy](#testing-strategy)

---

## Executive Summary

This document provides a comprehensive design for three critical enhancements to the NIMCP P2P node:

1. **Heartbeat/Keepalive Mechanism** - Detect dead connections through periodic heartbeat messages
2. **Comprehensive Logging** - Integrate nimcp_logging throughout the P2P node for observability
3. **Data Structure Refactoring** - Eliminate the dual data structure antipattern (hash table + array)

### Key Design Principles

- **Performance**: No logging in hot paths, efficient heartbeat checks
- **Simplicity**: Use existing NIMCP protocol messages, leverage existing hash table
- **Correctness**: State machine-driven heartbeat tracking with clear invariants
- **Observability**: Comprehensive logging of all connection lifecycle events

---

## Phase 7: Heartbeat/Keepalive Mechanism

### 1. Heartbeat Message Protocol

#### Message Structure

Use existing NIMCP 2.0 Control Message with type `CTRL_MSG_HEARTBEAT` (0x0D):

```c
// Already defined in nimcp_protocol.h
typedef enum {
    // ... other types ...
    CTRL_MSG_HEARTBEAT = 0x0D,
    // ...
} control_msg_type_t;
```

**Heartbeat Payload** (optional parameters in TLV format):
- **TLV Type 0x01**: Node Status (1 byte) - health indicator
- **TLV Type 0x02**: Peer Count (4 bytes) - number of active connections
- **TLV Type 0x03**: Uptime (8 bytes) - milliseconds since node start

**Wire Format:**
```
[Control Message Header: 24 bytes]
[Optional TLV Parameters: variable]
```

#### Protocol Behavior

**Heartbeat Transmission:**
1. Send heartbeat every `keepalive_interval` milliseconds (from config)
2. **Optimization**: Don't send heartbeat if data was sent in last interval
3. Use sequence numbers to detect missed heartbeats

**Heartbeat Reception:**
1. Update peer's `last_heartbeat_time` timestamp
2. Reset `missed_heartbeat_count` to 0
3. Extract optional status information from TLV payload

**Connection Failure Detection:**
1. If no data received for `keepalive_interval * heartbeat_timeout_factor` ms
2. Mark connection as failed
3. Trigger disconnect and optional reconnection

---

### 2. Heartbeat State Machine (Per-Peer)

#### Peer State Extensions

Add heartbeat tracking fields to `peer_info_t`:

```c
/**
 * @brief Enhanced peer information structure with heartbeat tracking
 */
typedef struct {
    // Existing fields
    char ip[16];
    uint16_t port;
    int socket_fd;
    bool connected;

    // NEW: Heartbeat tracking
    uint64_t last_heartbeat_time;      /**< Timestamp of last heartbeat (ms) */
    uint64_t last_data_time;           /**< Timestamp of last data sent/recv (ms) */
    uint32_t missed_heartbeat_count;   /**< Consecutive missed heartbeats */
    heartbeat_state_t heartbeat_state; /**< Current heartbeat state */
    uint32_t heartbeat_sequence;       /**< Outgoing heartbeat sequence number */
} peer_info_t;
```

#### Heartbeat State Enumeration

```c
/**
 * @brief Heartbeat state for connection health tracking
 */
typedef enum {
    HB_STATE_HEALTHY,      /**< Receiving heartbeats regularly */
    HB_STATE_DEGRADED,     /**< Missed 1-2 heartbeats */
    HB_STATE_FAILING,      /**< Missed 3-4 heartbeats */
    HB_STATE_DEAD          /**< Connection considered dead */
} heartbeat_state_t;
```

#### State Transitions

```
HEALTHY → DEGRADED → FAILING → DEAD
   ↑         ↑          ↑
   └─────────┴──────────┴─── (heartbeat received)
```

**State Transition Logic:**

```c
/**
 * @brief Update heartbeat state based on missed heartbeat count
 */
static void update_heartbeat_state(peer_info_t* peer, const node_config_t* config) {
    if (peer->missed_heartbeat_count == 0) {
        peer->heartbeat_state = HB_STATE_HEALTHY;
    } else if (peer->missed_heartbeat_count <= 2) {
        peer->heartbeat_state = HB_STATE_DEGRADED;
    } else if (peer->missed_heartbeat_count <= config->max_missed_heartbeats) {
        peer->heartbeat_state = HB_STATE_FAILING;
    } else {
        peer->heartbeat_state = HB_STATE_DEAD;
    }
}
```

---

### 3. Configuration Parameters

Add to `node_config_t` in `/home/bbrelin/src/repos/nimcp/src/include/nimcp_p2pnode.h`:

```c
typedef struct {
    uint16_t listen_port;
    uint32_t max_peers;

    // Existing
    uint32_t keepalive_interval;   /**< Already present */

    // NEW: Heartbeat-specific configuration
    uint32_t heartbeat_timeout_ms;    /**< Timeout for heartbeat response */
    uint32_t max_missed_heartbeats;   /**< Max missed before disconnect (default: 5) */
    bool enable_heartbeat_stats;      /**< Collect detailed heartbeat statistics */

    // Existing
    uint32_t discovery_interval;
    uint32_t reconnect_interval;
    uint32_t max_retries;
    uint32_t ping_interval;
} node_config_t;
```

**Default Values:**
- `keepalive_interval`: 1000 ms (1 second)
- `heartbeat_timeout_ms`: 5000 ms (5 seconds)
- `max_missed_heartbeats`: 5
- `enable_heartbeat_stats`: false

**Calculation:**
```
Time until connection declared dead = keepalive_interval * max_missed_heartbeats
Example: 1000ms * 5 = 5 seconds
```

---

### 4. Event Loop Integration

#### Threading Model

**Option A: Single-threaded event loop with timerfd**
```c
/**
 * @brief Event loop structure
 */
typedef struct {
    int epoll_fd;              /**< epoll file descriptor */
    int timer_fd;              /**< timerfd for heartbeat timer */
    bool running;              /**< Event loop running flag */
    uint64_t next_heartbeat;   /**< Next heartbeat send time */
} event_loop_t;
```

**Option B: Separate heartbeat thread (simpler initial implementation)**
```c
/**
 * @brief Heartbeat thread function
 */
static void* heartbeat_thread_fn(void* arg) {
    p2p_node_t node = (p2p_node_t)arg;

    while (node->running) {
        uint64_t now = get_time_ms();

        // Check all peers and send heartbeats
        check_and_send_heartbeats(node, now);

        // Sleep until next heartbeat interval
        usleep(node->config.keepalive_interval * 1000);
    }

    return NULL;
}
```

**Recommendation**: Start with **Option B** (separate thread) for simplicity, migrate to Option A (event loop) in future optimization.

#### Heartbeat Check Algorithm

```c
/**
 * @brief Check all peers and send heartbeats if needed
 *
 * COMPLEXITY: O(n) where n = peer_count
 * CALLED BY: Heartbeat thread every keepalive_interval
 */
static void check_and_send_heartbeats(p2p_node_t node, uint64_t now) {
    if (!node) return;

    // Iterate through all peers (using hash table iterator)
    hash_table_iterate(&node->peer_table, heartbeat_check_iterator, &now);
}

/**
 * @brief Iterator callback for heartbeat checking
 */
static bool heartbeat_check_iterator(const void* key, size_t key_size,
                                     void* value, size_t value_size,
                                     void* user_data) {
    peer_info_t* peer = (peer_info_t*)value;
    uint64_t now = *(uint64_t*)user_data;
    p2p_node_t node = // ... get from context

    // Check if peer needs heartbeat
    uint64_t time_since_last_data = now - peer->last_data_time;

    if (time_since_last_data >= node->config.keepalive_interval) {
        // Send heartbeat
        send_heartbeat_to_peer(node, peer);
    }

    // Check if peer has missed heartbeats
    uint64_t time_since_heartbeat = now - peer->last_heartbeat_time;

    if (time_since_heartbeat > node->config.keepalive_interval) {
        peer->missed_heartbeat_count++;
        update_heartbeat_state(peer, &node->config);

        // Log degradation
        if (peer->heartbeat_state == HB_STATE_DEGRADED) {
            NIMCP_LOGGING_WARN("Peer %s:%u degraded, missed %u heartbeats",
                              peer->ip, peer->port, peer->missed_heartbeat_count);
        }

        // Disconnect if dead
        if (peer->heartbeat_state == HB_STATE_DEAD) {
            NIMCP_LOGGING_ERROR("Peer %s:%u dead, disconnecting after %u missed heartbeats",
                               peer->ip, peer->port, peer->missed_heartbeat_count);
            disconnect_dead_peer(node, peer);
        }
    }

    return true; // Continue iteration
}
```

---

### 5. Heartbeat Message Implementation

#### Sending Heartbeats

```c
/**
 * @brief Send heartbeat message to peer
 *
 * COMPLEXITY: O(1)
 * OPTIMIZATION: Only sends if no recent data transmission
 */
static bool send_heartbeat_to_peer(p2p_node_t node, peer_info_t* peer) {
    if (!node || !peer || !peer->connected) return false;

    // Create heartbeat control message
    control_message_t msg = {
        .version = PROTOCOL_VERSION,
        .msg_type = CTRL_MSG_HEARTBEAT,
        .flags = 0,
        .message_length = sizeof(control_message_t),
        .source_node_id = node->node_id,
        .target_specifier = 0, // Direct peer
        .sequence_number = peer->heartbeat_sequence++,
        .param_count = 0 // No TLV parameters for basic heartbeat
    };

    // Serialize message
    uint8_t buffer[256];
    int bytes = control_message_serialize(&msg, NULL, buffer, sizeof(buffer));
    if (bytes < 0) {
        NIMCP_LOGGING_ERROR("Failed to serialize heartbeat for %s:%u",
                           peer->ip, peer->port);
        return false;
    }

    // Send via socket
    ssize_t sent = send(peer->socket_fd, buffer, bytes, MSG_NOSIGNAL);
    if (sent < 0) {
        NIMCP_LOGGING_ERROR("Failed to send heartbeat to %s:%u: %s",
                           peer->ip, peer->port, strerror(errno));
        return false;
    }

    // Update last data time (we just sent data)
    peer->last_data_time = get_time_ms();

    NIMCP_LOGGING_DEBUG("Sent heartbeat seq=%u to %s:%u",
                       peer->heartbeat_sequence - 1, peer->ip, peer->port);

    return true;
}
```

#### Receiving Heartbeats

```c
/**
 * @brief Handle received heartbeat message
 *
 * CALLED BY: Message dispatch handler
 */
static void handle_heartbeat_message(p2p_node_t node, peer_info_t* peer,
                                     const control_message_t* msg) {
    if (!node || !peer || !msg) return;

    uint64_t now = get_time_ms();

    // Update heartbeat tracking
    peer->last_heartbeat_time = now;
    peer->last_data_time = now;
    peer->missed_heartbeat_count = 0;

    // Restore healthy state
    heartbeat_state_t old_state = peer->heartbeat_state;
    peer->heartbeat_state = HB_STATE_HEALTHY;

    // Log state recovery
    if (old_state != HB_STATE_HEALTHY) {
        NIMCP_LOGGING_INFO("Peer %s:%u recovered from %s to HEALTHY",
                          peer->ip, peer->port,
                          heartbeat_state_name(old_state));
    }

    NIMCP_LOGGING_DEBUG("Received heartbeat seq=%u from %s:%u",
                       msg->sequence_number, peer->ip, peer->port);
}
```

---

### 6. Optimization: Suppress Heartbeats on Active Connections

**Rationale**: If we're actively sending data, we don't need separate heartbeat messages.

```c
/**
 * @brief Update last data time when sending any message
 *
 * CALL THIS: After every successful send() operation
 */
static inline void update_peer_data_time(peer_info_t* peer) {
    if (peer) {
        peer->last_data_time = get_time_ms();
    }
}
```

**Integration Points:**
1. After sending any protocol message
2. After sending event packets
3. After sending control messages

**Logic:**
```c
// In send_heartbeat_to_peer():
uint64_t time_since_last_data = now - peer->last_data_time;

if (time_since_last_data < node->config.keepalive_interval / 2) {
    // Recent data sent, skip heartbeat
    return true;
}
```

---

## Phase 8: Comprehensive Logging Strategy

### 1. Logging Architecture

#### Log Levels

Using existing nimcp_logging levels from `/home/bbrelin/src/repos/nimcp/src/include/logging/nimcp_logging.h`:

```c
typedef enum {
    LOG_LEVEL_DEBUG,   // Verbose debugging (peer iteration, heartbeat checks)
    LOG_LEVEL_INFO,    // Important events (connection established/lost)
    LOG_LEVEL_WARN,    // Warnings (degraded connections, retries)
    LOG_LEVEL_ERROR,   // Errors (connection failures, send errors)
    LOG_LEVEL_FATAL    // Fatal errors (initialization failures)
} log_level_t;
```

#### Logging Macros

Use existing macros from nimcp_logging.h:
- `NIMCP_LOGGING_DEBUG(...)`
- `NIMCP_LOGGING_INFO(...)`
- `NIMCP_LOGGING_WARN(...)`
- `NIMCP_LOGGING_ERROR(...)`
- `NIMCP_LOGGING_FATAL(...)`

---

### 2. Logging Events Catalog

#### Connection Lifecycle Events

```c
// CONNECTION ESTABLISHED (INFO level)
NIMCP_LOGGING_INFO("Connection established: %s:%u (socket=%d, peers=%u/%u)",
                  peer->ip, peer->port, peer->socket_fd,
                  node->peer_count, node->config.max_peers);

// CONNECTION LOST (ERROR level)
NIMCP_LOGGING_ERROR("Connection lost: %s:%u (reason: %s, uptime=%lums)",
                   peer->ip, peer->port, disconnect_reason,
                   now - peer->connect_time);

// CONNECTION ATTEMPT (DEBUG level)
NIMCP_LOGGING_DEBUG("Attempting connection to %s:%u (retry %u/%u)",
                   ip, port, retry_count, config->max_retries);

// CONNECTION FAILED (WARN level)
NIMCP_LOGGING_WARN("Connection failed: %s:%u (error: %s)",
                  ip, port, strerror(errno));
```

#### Message Events

```c
// MESSAGE SENT (DEBUG level - only if verbose mode enabled)
if (node->config.verbose_logging) {
    NIMCP_LOGGING_DEBUG("Sent %s message to %s:%u (size=%d bytes, seq=%u)",
                       msg_type_name(type), peer->ip, peer->port,
                       bytes_sent, sequence_num);
}

// MESSAGE RECEIVED (DEBUG level - only if verbose mode)
if (node->config.verbose_logging) {
    NIMCP_LOGGING_DEBUG("Received %s message from %s:%u (size=%d bytes, seq=%u)",
                       msg_type_name(type), peer->ip, peer->port,
                       bytes_recv, sequence_num);
}

// MESSAGE COUNT SUMMARY (INFO level - periodic)
NIMCP_LOGGING_INFO("Message statistics: sent=%lu, recv=%lu, errors=%lu (period=%us)",
                  stats->total_sent, stats->total_recv, stats->errors,
                  stats_period_seconds);
```

#### Queue Events

```c
// QUEUE OVERFLOW (WARN level)
NIMCP_LOGGING_WARN("Send queue overflow for %s:%u (dropped=%u, queue_size=%u/%u)",
                  peer->ip, peer->port, dropped_count,
                  queue_size, queue_capacity);

// QUEUE BACKPRESSURE (DEBUG level)
NIMCP_LOGGING_DEBUG("Send queue backpressure: %s:%u (queue=%u/%u, utilization=%.1f%%)",
                   peer->ip, peer->port, queue_size, queue_capacity,
                   (queue_size * 100.0f) / queue_capacity);
```

#### Heartbeat Events

```c
// HEARTBEAT SENT (DEBUG level)
NIMCP_LOGGING_DEBUG("Sent heartbeat to %s:%u (seq=%u, state=%s)",
                   peer->ip, peer->port, peer->heartbeat_sequence,
                   heartbeat_state_name(peer->heartbeat_state));

// HEARTBEAT RECEIVED (DEBUG level)
NIMCP_LOGGING_DEBUG("Received heartbeat from %s:%u (seq=%u)",
                   peer->ip, peer->port, msg->sequence_number);

// HEARTBEAT DEGRADED (WARN level)
NIMCP_LOGGING_WARN("Heartbeat degraded: %s:%u (missed=%u, state=%s→%s)",
                  peer->ip, peer->port, peer->missed_heartbeat_count,
                  heartbeat_state_name(old_state),
                  heartbeat_state_name(new_state));

// HEARTBEAT TIMEOUT (ERROR level)
NIMCP_LOGGING_ERROR("Heartbeat timeout: %s:%u (missed=%u, last_seen=%lums ago)",
                   peer->ip, peer->port, peer->missed_heartbeat_count,
                   now - peer->last_heartbeat_time);
```

#### Thread Lifecycle Events

```c
// THREAD START (INFO level)
NIMCP_LOGGING_INFO("Starting %s thread (id=%lu)",
                  thread_name, (unsigned long)thread_id);

// THREAD STOP (INFO level)
NIMCP_LOGGING_INFO("Stopping %s thread (id=%lu, runtime=%lums)",
                  thread_name, (unsigned long)thread_id, runtime_ms);

// THREAD ERROR (ERROR level)
NIMCP_LOGGING_ERROR("Thread error in %s: %s (id=%lu)",
                   thread_name, error_msg, (unsigned long)thread_id);
```

#### Node Lifecycle Events

```c
// NODE START (INFO level)
NIMCP_LOGGING_INFO("P2P node starting (port=%u, max_peers=%u, version=%d)",
                  config->listen_port, config->max_peers, PROTOCOL_VERSION);

// NODE STOP (INFO level)
NIMCP_LOGGING_INFO("P2P node stopping (uptime=%lus, connections=%lu, bytes_sent=%lu, bytes_recv=%lu)",
                  uptime_seconds, node->total_connections,
                  node->bytes_sent, node->bytes_received);

// LISTEN SOCKET BOUND (DEBUG level)
NIMCP_LOGGING_DEBUG("Listen socket bound (fd=%d, port=%u)",
                   node->listen_socket, config->listen_port);
```

---

### 3. Performance Considerations

#### Hot Path Optimization

**RULE**: No logging in hot paths unless debugging mode is enabled.

```c
// DON'T log in tight loops or per-packet handlers
for (uint32_t i = 0; i < packet_count; i++) {
    // WRONG: This will flood logs
    // NIMCP_LOGGING_DEBUG("Processing packet %u", i);

    process_packet(&packets[i]);
}

// DO log aggregated statistics periodically
if (should_log_stats(now)) {
    NIMCP_LOGGING_INFO("Processed %u packets in last %us",
                      packet_count, stats_interval);
}
```

#### Conditional Compilation for Debug Logs

```c
// In debug builds, enable verbose logging
#ifdef NIMCP_DEBUG
    #define LOG_VERBOSE(fmt, ...) NIMCP_LOGGING_DEBUG(fmt, ##__VA_ARGS__)
#else
    #define LOG_VERBOSE(fmt, ...) ((void)0)
#endif

// Usage:
LOG_VERBOSE("Entering send_message() for peer %s:%u", ip, port);
```

#### Rate Limiting for Frequent Events

```c
/**
 * @brief Rate-limited logging to prevent log floods
 */
static bool should_log_event(const char* event_id, uint32_t max_per_second) {
    static hash_table_t* rate_limit_table = NULL;

    // Initialize table on first call
    if (!rate_limit_table) {
        hash_table_config_t config = {
            .initial_buckets = 64,
            .key_type = HASH_KEY_STRING,
            .hash_algorithm = HASH_ALG_FNV1A
        };
        rate_limit_table = hash_table_create(&config);
    }

    // Check rate limit
    uint64_t* last_log_time = hash_table_lookup_string(rate_limit_table, event_id);
    uint64_t now = get_time_ms();

    if (last_log_time && (now - *last_log_time) < (1000 / max_per_second)) {
        return false; // Rate limited
    }

    // Update timestamp
    hash_table_insert_string(rate_limit_table, event_id, &now, sizeof(now));
    return true;
}

// Usage:
if (should_log_event("heartbeat_debug", 10)) { // Max 10/second
    NIMCP_LOGGING_DEBUG("Heartbeat check iteration");
}
```

---

### 4. Structured Logging Format

#### Log Entry Format

```
[TIMESTAMP] [LEVEL] [COMPONENT] Message
```

Example:
```
[2025-11-01 14:32:15] [INFO] [P2P_NODE] Connection established: 192.168.1.10:8080 (socket=5, peers=3/10)
[2025-11-01 14:32:20] [WARN] [HEARTBEAT] Heartbeat degraded: 192.168.1.10:8080 (missed=2, state=HEALTHY→DEGRADED)
[2025-11-01 14:32:25] [ERROR] [P2P_NODE] Connection lost: 192.168.1.10:8080 (reason: timeout, uptime=45000ms)
```

#### Component Tags

```c
#define LOG_TAG_P2P_NODE   "P2P_NODE"
#define LOG_TAG_HEARTBEAT  "HEARTBEAT"
#define LOG_TAG_PROTOCOL   "PROTOCOL"
#define LOG_TAG_THREAD     "THREAD"

// Usage:
NIMCP_LOGGING_INFO("[%s] Starting node on port %u", LOG_TAG_P2P_NODE, port);
```

---

### 5. Logging Integration Points

#### File: /home/bbrelin/src/repos/nimcp/src/lib/nimcp_p2pnode.c

**Add logging at these locations:**

1. **Node Creation** (`p2p_node_create`)
   ```c
   NIMCP_LOGGING_INFO("Creating P2P node (port=%u, max_peers=%u)",
                     config->listen_port, config->max_peers);
   ```

2. **Node Start** (`p2p_node_start`)
   ```c
   NIMCP_LOGGING_INFO("Starting P2P node (port=%u)", node->config.listen_port);
   // ... after listen socket setup ...
   NIMCP_LOGGING_DEBUG("Listen socket bound (fd=%d, port=%u)",
                      node->listen_socket, node->config.listen_port);
   ```

3. **Node Stop** (`p2p_node_stop`)
   ```c
   NIMCP_LOGGING_INFO("Stopping P2P node (connections=%lu, bytes_sent=%lu, bytes_recv=%lu)",
                     node->total_connections, node->bytes_sent, node->bytes_received);
   ```

4. **Peer Connect** (`p2p_node_connect_peer`)
   ```c
   // Before connection attempt
   NIMCP_LOGGING_DEBUG("Attempting connection to %s:%u", peer_ip, peer_port);

   // On success
   NIMCP_LOGGING_INFO("Connection established: %s:%u (socket=%d, peers=%u/%u)",
                     peer_ip, peer_port, sock, node->peer_count, node->config.max_peers);

   // On failure
   NIMCP_LOGGING_WARN("Connection failed: %s:%u (error: %s)",
                     peer_ip, peer_port, strerror(errno));
   ```

5. **Peer Disconnect** (`p2p_node_disconnect_peer`)
   ```c
   NIMCP_LOGGING_INFO("Disconnecting peer: %s:%u", peer_ip, peer_port);
   ```

---

## Phase 9: Remove Dual Data Structure Antipattern

### 1. Current Architecture (Antipattern)

**Problem:** Dual data structures in `/home/bbrelin/src/repos/nimcp/src/lib/nimcp_p2pnode.c`:

```c
struct p2p_node_struct {
    // ... other fields ...

    // ANTIPATTERN: Two data structures for same data
    peer_hash_table_t peer_table;  // For O(1) lookups
    peer_info_t* peers;            // For iteration
    uint32_t peer_count;
};
```

**Issues:**
1. **Redundancy**: Same peer data stored twice
2. **Synchronization burden**: Must update both structures on every change
3. **Memory waste**: ~2x memory usage
4. **Bug potential**: Structures can get out of sync
5. **Complexity**: Two separate update paths

---

### 2. Refactored Architecture (Hash Table Only)

**Solution:** Use hash table exclusively, iterate via buckets.

```c
struct p2p_node_struct {
    // ... other fields ...

    // Single source of truth
    peer_hash_table_t peer_table;
};
```

**Benefits:**
1. **Single source of truth**: Impossible to desynchronize
2. **Memory savings**: ~50% reduction in peer storage
3. **Simpler code**: One update path
4. **Same performance**: O(1) lookups, O(n) iteration

---

### 3. Hash Table Iterator for Peer Traversal

#### Generic Iterator Interface

The existing hash table in nimcp_hash_table.h already provides:

```c
/**
 * WHAT: Iterate over all entries in table
 * WHY: Process all stored values
 * HOW: Traverse all buckets and chains, call callback
 */
void hash_table_iterate(hash_table_t* table,
                        hash_table_iterator_fn_t callback,
                        void* user_data);

typedef bool (*hash_table_iterator_fn_t)(const void* key, size_t key_size,
                                          void* value, size_t value_size,
                                          void* user_data);
```

#### P2P-Specific Iterator Wrapper

```c
/**
 * @brief Iterator callback type for peer traversal
 *
 * @param peer Pointer to peer_info_t
 * @param user_data Opaque user context
 * @return true to continue iteration, false to stop
 */
typedef bool (*peer_iterator_fn_t)(peer_info_t* peer, void* user_data);

/**
 * @brief Iterate over all connected peers
 *
 * COMPLEXITY: O(n) where n = number of peers
 *
 * @param node P2P node instance
 * @param callback Function to call for each peer
 * @param user_data Opaque pointer passed to callback
 */
static void iterate_peers(p2p_node_t node,
                         peer_iterator_fn_t callback,
                         void* user_data) {
    if (!node || !callback) return;

    // Context for hash table iterator
    struct {
        peer_iterator_fn_t callback;
        void* user_data;
    } ctx = { .callback = callback, .user_data = user_data };

    // Adapt to hash table iterator interface
    hash_table_iterate(&node->peer_table, peer_iterator_adapter, &ctx);
}

/**
 * @brief Adapter between hash_table_iterator and peer_iterator
 */
static bool peer_iterator_adapter(const void* key, size_t key_size,
                                  void* value, size_t value_size,
                                  void* user_data) {
    struct { peer_iterator_fn_t callback; void* user_data; }* ctx = user_data;
    peer_info_t* peer = (peer_info_t*)value;

    return ctx->callback(peer, ctx->user_data);
}
```

#### Usage Examples

**Example 1: Disconnect all peers**

```c
// Callback function
static bool disconnect_peer_callback(peer_info_t* peer, void* user_data) {
    if (peer->socket_fd >= 0) {
        close(peer->socket_fd);
        peer->socket_fd = -1;
    }
    peer->connected = false;
    return true; // Continue iteration
}

// Disconnect all
iterate_peers(node, disconnect_peer_callback, NULL);
```

**Example 2: Count connected peers**

```c
static bool count_connected_callback(peer_info_t* peer, void* user_data) {
    uint32_t* count = (uint32_t*)user_data;
    if (peer->connected) {
        (*count)++;
    }
    return true;
}

uint32_t count = 0;
iterate_peers(node, count_connected_callback, &count);
```

**Example 3: Send broadcast message**

```c
typedef struct {
    uint8_t* message;
    size_t message_len;
    uint32_t sent_count;
} broadcast_ctx_t;

static bool broadcast_callback(peer_info_t* peer, void* user_data) {
    broadcast_ctx_t* ctx = (broadcast_ctx_t*)user_data;

    if (!peer->connected) return true; // Skip disconnected

    ssize_t sent = send(peer->socket_fd, ctx->message, ctx->message_len, 0);
    if (sent > 0) {
        ctx->sent_count++;
    }

    return true; // Continue to next peer
}

broadcast_ctx_t ctx = { .message = msg, .message_len = len, .sent_count = 0 };
iterate_peers(node, broadcast_callback, &ctx);
```

---

### 4. Migration Plan: Removing `peers` Array

#### Step-by-Step Refactoring

**STEP 1: Identify all array usage locations**

```bash
# Search for direct array access
grep -n "node->peers\[" src/lib/nimcp_p2pnode.c
grep -n "->peers\[" src/lib/nimcp_p2pnode.c
```

**STEP 2: Replace array iteration with hash table iteration**

**Before:**
```c
// Old array-based iteration
for (uint32_t i = 0; i < node->peer_count; i++) {
    peer_info_t* peer = &node->peers[i];
    if (peer->connected) {
        // Do something with peer
    }
}
```

**After:**
```c
// New hash table iteration
static bool process_peer_callback(peer_info_t* peer, void* user_data) {
    if (peer->connected) {
        // Do something with peer
    }
    return true;
}

iterate_peers(node, process_peer_callback, NULL);
```

**STEP 3: Remove array allocation in `p2p_node_create`**

**Before:**
```c
// Line 497-498 in nimcp_p2pnode.c
node->peers = calloc(node->config.max_peers, sizeof(peer_info_t));
return node->peers != NULL;
```

**After:**
```c
// Remove array allocation entirely
// Hash table already initialized at line 556
return true;
```

**STEP 4: Update `add_peer_to_node` function**

**Before:**
```c
static bool add_peer_to_node(p2p_node_t node, const char* ip,
                             uint16_t port, int socket_fd) {
    if (!node || !ip) return false;
    if (node->peer_count >= node->config.max_peers) return false;

    // Initialize peer info in array
    peer_info_t* peer = &node->peers[node->peer_count];
    initialize_peer_info(peer, ip, port, socket_fd);

    // Generate key for hash table
    char key[32];
    generate_peer_key(key, sizeof(key), ip, port);

    // Add to hash table
    if (!hash_table_insert_peer(&node->peer_table, key, peer)) {
        return false;
    }

    node->peer_count++;
    return true;
}
```

**After:**
```c
static bool add_peer_to_node(p2p_node_t node, const char* ip,
                             uint16_t port, int socket_fd) {
    if (!node || !ip) return false;

    // Check peer count limit
    if (hash_table_size(&node->peer_table) >= node->config.max_peers) {
        return false;
    }

    // Allocate new peer structure
    peer_info_t* peer = malloc(sizeof(peer_info_t));
    if (!peer) return false;

    // Initialize peer info
    initialize_peer_info(peer, ip, port, socket_fd);

    // Generate key
    char key[32];
    generate_peer_key(key, sizeof(key), ip, port);

    // Add to hash table (transfers ownership)
    if (!hash_table_insert_peer(&node->peer_table, key, peer)) {
        free(peer);
        return false;
    }

    return true;
}
```

**STEP 5: Update `compact_peer_array` to hash table removal**

**Before:**
```c
static void compact_peer_array(p2p_node_t node, uint32_t index) {
    if (!node || index >= node->peer_count) return;

    // Move last peer to this slot
    if (index < node->peer_count - 1) {
        node->peers[index] = node->peers[node->peer_count - 1];
    }

    node->peer_count--;
}
```

**After:**
```c
// Function no longer needed - hash table handles removal
// Delete this function entirely
```

**STEP 6: Update `p2p_node_disconnect_peer`**

**Before:**
```c
bool p2p_node_disconnect_peer(p2p_node_t node, const char* peer_ip, uint16_t peer_port) {
    if (!node || !peer_ip) return false;

    char key[32];
    generate_peer_key(key, sizeof(key), peer_ip, peer_port);

    peer_info_t* peer = hash_table_lookup_peer(&node->peer_table, key);
    if (!peer) return false;

    close_peer_socket(peer);
    hash_table_remove_peer(&node->peer_table, key);

    int index = find_peer_index(node, peer_ip, peer_port);
    if (index < 0) return false;

    compact_peer_array(node, (uint32_t)index);
    return true;
}
```

**After:**
```c
bool p2p_node_disconnect_peer(p2p_node_t node, const char* peer_ip, uint16_t peer_port) {
    if (!node || !peer_ip) return false;

    char key[32];
    generate_peer_key(key, sizeof(key), peer_ip, peer_port);

    // Lookup peer
    peer_info_t* peer = hash_table_lookup_peer(&node->peer_table, key);
    if (!peer) return false;

    // Close socket
    close_peer_socket(peer);

    // Remove from hash table (will free peer via destructor)
    return hash_table_remove_peer(&node->peer_table, key);
}
```

**STEP 7: Update `disconnect_all_peers`**

**Before:**
```c
static void disconnect_all_peers(p2p_node_t node) {
    if (!node) return;

    for (uint32_t i = 0; i < node->peer_count; i++) {
        disconnect_peer_during_shutdown(&node->peers[i]);
    }

    node->peer_count = 0;
}
```

**After:**
```c
static bool disconnect_peer_iterator(peer_info_t* peer, void* user_data) {
    if (peer->socket_fd >= 0) {
        close(peer->socket_fd);
        peer->socket_fd = -1;
    }
    peer->connected = false;
    return true;
}

static void disconnect_all_peers(p2p_node_t node) {
    if (!node) return;

    // Close all sockets via iteration
    iterate_peers(node, disconnect_peer_iterator, NULL);

    // Clear hash table (will free all peers via destructor)
    hash_table_clear(&node->peer_table);
}
```

**STEP 8: Remove array deallocation in `p2p_node_destroy`**

**Before:**
```c
void p2p_node_destroy(p2p_node_t node) {
    if (!node) return;

    if (node->running) {
        p2p_node_stop(node);
    }

    destroy_peer_hash_table(&node->peer_table);
    free(node->peers);  // <-- REMOVE THIS
    close_listen_socket(node);
    free(node);
}
```

**After:**
```c
void p2p_node_destroy(p2p_node_t node) {
    if (!node) return;

    if (node->running) {
        p2p_node_stop(node);
    }

    destroy_peer_hash_table(&node->peer_table);
    // Array removed - no deallocation needed
    close_listen_socket(node);
    free(node);
}
```

**STEP 9: Remove `peer_count` field**

**Before:**
```c
struct p2p_node_struct {
    // ...
    peer_hash_table_t peer_table;
    peer_info_t* peers;        // <-- REMOVE
    uint32_t peer_count;       // <-- REMOVE
    // ...
};
```

**After:**
```c
struct p2p_node_struct {
    // ...
    peer_hash_table_t peer_table;
    // Removed: peers array and peer_count
    // ...
};

// Replace all peer_count usage with:
uint32_t peer_count = hash_table_size(&node->peer_table);
```

**STEP 10: Remove helper functions**

Delete these functions that are no longer needed:
- `find_peer_index()` (line 900-912)
- `compact_peer_array()` (line 922-932)
- `allocate_peer_storage()` (line 493-499)

---

### 5. Memory Management with Hash Table

#### Peer Memory Ownership

**Configure hash table with destructor:**

```c
/**
 * @brief Destructor for peer_info_t in hash table
 */
static void peer_destructor(void* value, size_t value_size) {
    peer_info_t* peer = (peer_info_t*)value;

    // Close socket if open
    if (peer && peer->socket_fd >= 0) {
        close(peer->socket_fd);
    }

    // Free peer structure
    free(peer);
}

/**
 * @brief Initialize peer hash table with destructor
 */
static void init_peer_hash_table(peer_hash_table_t* table) {
    if (!table) return;

    hash_table_config_t config = {
        .initial_buckets = HASH_TABLE_SIZE,
        .key_type = HASH_KEY_STRING,
        .hash_algorithm = HASH_ALG_DJB2,
        .value_destructor = peer_destructor,  // Auto-cleanup
        .case_insensitive = false
    };

    table->handle = hash_table_create(&config);
    table->num_entries = 0;
}
```

**Benefits:**
- Automatic cleanup when peer removed
- No memory leaks
- Simplified error handling

---

### 6. Line-by-Line Migration Checklist

**File:** `/home/bbrelin/src/repos/nimcp/src/lib/nimcp_p2pnode.c`

| Line | Current Code | Action | Replacement |
|------|-------------|--------|-------------|
| 127-128 | `peer_info_t* peers;`<br>`uint32_t peer_count;` | DELETE | Remove both fields |
| 193 | `table->num_entries = 0;` | KEEP | Hash table maintains count |
| 497-498 | `node->peers = calloc(...);` | DELETE | No array allocation |
| 515 | `node->peer_count = 0;` | DELETE | Use `hash_table_size()` |
| 619 | `free(node->peers);` | DELETE | Hash table owns memory |
| 777 | `if (node->peer_count >= ...)` | MODIFY | Use `hash_table_size()` |
| 780-781 | `peer_info_t* peer = &node->peers[node->peer_count];` | MODIFY | Allocate with `malloc()` |
| 792 | `node->peer_count++;` | DELETE | Hash table updates count |
| 825 | `if (node->peer_count >= ...)` | MODIFY | Use `hash_table_size()` |
| 900-912 | `find_peer_index()` function | DELETE | No longer needed |
| 922-932 | `compact_peer_array()` function | DELETE | No longer needed |
| 975-980 | Array compaction code | DELETE | Hash table handles removal |
| 1071-1076 | Array iteration in `disconnect_all_peers()` | MODIFY | Use `iterate_peers()` |

---

### 7. Performance Impact Analysis

#### Memory Impact

**Before:**
```
Array: max_peers * sizeof(peer_info_t) = 100 * 64 bytes = 6,400 bytes
Hash table entries: num_peers * (32 + sizeof(peer_info_t*)) = variable
Total: ~10KB for 100 peers
```

**After:**
```
Hash table only: num_peers * (32 + sizeof(peer_info_t)) = variable
Total: ~7KB for 100 peers (30% savings)
```

#### Lookup Performance

**Before:**
- Lookup: O(1) via hash table ✓
- Iteration: O(n) via array ✓

**After:**
- Lookup: O(1) via hash table ✓
- Iteration: O(n) via hash table buckets ✓

**No performance degradation.**

#### Complexity Analysis

| Operation | Before | After | Change |
|-----------|--------|-------|--------|
| Lookup peer | O(1) | O(1) | Same |
| Add peer | O(1) | O(1) | Same |
| Remove peer | O(n) | O(1) | **Better** |
| Iterate all | O(n) | O(n) | Same |
| Get count | O(1) | O(1) | Same |

---

## Implementation Order

### Recommended Sequence

**Phase 1: Heartbeat (Week 1)**
1. Add heartbeat fields to `peer_info_t`
2. Implement heartbeat state machine
3. Add heartbeat thread
4. Implement heartbeat send/receive
5. Add timeout detection
6. Unit tests

**Phase 2: Logging (Week 1)**
7. Add logging initialization in `p2p_node_create`
8. Add logging to connection lifecycle
9. Add logging to heartbeat events
10. Add logging to message send/receive
11. Configure log levels and rate limiting
12. Integration tests

**Phase 3: Refactoring (Week 2)**
13. Add `iterate_peers()` wrapper
14. Replace array iterations with hash table iterations
15. Remove array allocation
16. Update disconnect logic
17. Remove unused functions
18. Remove array fields from struct
19. Regression tests
20. Performance benchmarks

---

## Testing Strategy

### Unit Tests

#### Heartbeat Tests

```c
// Test: Heartbeat state transitions
void test_heartbeat_state_transitions(void) {
    peer_info_t peer = {0};
    node_config_t config = { .max_missed_heartbeats = 5 };

    // HEALTHY → DEGRADED
    peer.missed_heartbeat_count = 1;
    update_heartbeat_state(&peer, &config);
    assert(peer.heartbeat_state == HB_STATE_DEGRADED);

    // DEGRADED → FAILING
    peer.missed_heartbeat_count = 3;
    update_heartbeat_state(&peer, &config);
    assert(peer.heartbeat_state == HB_STATE_FAILING);

    // FAILING → DEAD
    peer.missed_heartbeat_count = 6;
    update_heartbeat_state(&peer, &config);
    assert(peer.heartbeat_state == HB_STATE_DEAD);

    // Recovery: DEAD → HEALTHY
    peer.missed_heartbeat_count = 0;
    update_heartbeat_state(&peer, &config);
    assert(peer.heartbeat_state == HB_STATE_HEALTHY);
}

// Test: Heartbeat suppression on active connection
void test_heartbeat_suppression(void) {
    p2p_node_t node = create_test_node();
    peer_info_t peer = {0};
    uint64_t now = get_time_ms();

    // Recent data sent
    peer.last_data_time = now - 100; // 100ms ago
    node->config.keepalive_interval = 1000; // 1 second

    // Should not send heartbeat
    bool sent = should_send_heartbeat(node, &peer, now);
    assert(!sent);

    // Old data
    peer.last_data_time = now - 2000; // 2 seconds ago

    // Should send heartbeat
    sent = should_send_heartbeat(node, &peer, now);
    assert(sent);
}
```

#### Iterator Tests

```c
// Test: Iterate all peers
void test_iterate_all_peers(void) {
    p2p_node_t node = create_test_node();

    // Add 10 peers
    for (int i = 0; i < 10; i++) {
        char ip[16];
        snprintf(ip, sizeof(ip), "192.168.1.%d", i + 1);
        add_peer_to_node(node, ip, 8080 + i, -1);
    }

    // Count via iteration
    uint32_t count = 0;
    iterate_peers(node, count_callback, &count);

    assert(count == 10);
    assert(hash_table_size(&node->peer_table) == 10);
}

// Test: Early termination
void test_iterator_early_stop(void) {
    p2p_node_t node = create_test_node();
    add_test_peers(node, 10);

    // Stop after 5 peers
    struct { uint32_t count; uint32_t limit; } ctx = { .count = 0, .limit = 5 };
    iterate_peers(node, stop_at_limit_callback, &ctx);

    assert(ctx.count == 5);
}
```

### Integration Tests

```c
// Test: Heartbeat timeout triggers disconnect
void test_heartbeat_timeout_disconnect(void) {
    p2p_node_t node = create_test_node();
    node->config.keepalive_interval = 100; // 100ms
    node->config.max_missed_heartbeats = 3;

    // Add peer
    peer_info_t* peer = add_test_peer(node, "192.168.1.10", 8080);

    // Simulate missed heartbeats
    for (int i = 0; i < 4; i++) {
        usleep(110000); // 110ms
        check_and_send_heartbeats(node, get_time_ms());
    }

    // Peer should be disconnected
    assert(peer->heartbeat_state == HB_STATE_DEAD);
    assert(!peer->connected);
}

// Test: Logging captures all events
void test_logging_integration(void) {
    // Capture logs to buffer
    char log_buffer[4096] = {0};
    set_log_capture_buffer(log_buffer, sizeof(log_buffer));

    p2p_node_t node = create_test_node();

    // Trigger various events
    p2p_node_start(node);
    p2p_node_connect_peer(node, "192.168.1.10", 8080);
    p2p_node_disconnect_peer(node, "192.168.1.10", 8080);
    p2p_node_stop(node);

    // Verify log entries
    assert(strstr(log_buffer, "Starting P2P node"));
    assert(strstr(log_buffer, "Connection established"));
    assert(strstr(log_buffer, "Disconnecting peer"));
    assert(strstr(log_buffer, "Stopping P2P node"));
}
```

### Stress Tests

```c
// Test: Handle many peers
void test_many_peers(void) {
    p2p_node_t node = create_test_node();
    node->config.max_peers = 1000;

    // Add 1000 peers
    for (int i = 0; i < 1000; i++) {
        char ip[16];
        snprintf(ip, sizeof(ip), "10.%d.%d.%d",
                (i >> 16) & 0xFF, (i >> 8) & 0xFF, i & 0xFF);
        add_peer_to_node(node, ip, 8080 + i, -1);
    }

    // Verify all added
    assert(hash_table_size(&node->peer_table) == 1000);

    // Measure iteration time
    uint64_t start = get_time_us();
    uint32_t count = 0;
    iterate_peers(node, count_callback, &count);
    uint64_t duration = get_time_us() - start;

    printf("Iterated 1000 peers in %lu us\n", duration);
    assert(count == 1000);
    assert(duration < 1000); // Should be < 1ms
}

// Test: Rapid connect/disconnect
void test_churn(void) {
    p2p_node_t node = create_test_node();

    // 1000 iterations of connect/disconnect
    for (int i = 0; i < 1000; i++) {
        p2p_node_connect_peer(node, "192.168.1.10", 8080);
        p2p_node_disconnect_peer(node, "192.168.1.10", 8080);
    }

    // Should have no peers
    assert(hash_table_size(&node->peer_table) == 0);

    // No memory leaks (verify with valgrind)
}
```

---

## Appendix A: Configuration Examples

### Minimal Configuration
```c
node_config_t config = {
    .listen_port = 8080,
    .max_peers = 10,
    .keepalive_interval = 1000,      // 1 second
    .heartbeat_timeout_ms = 5000,    // 5 seconds
    .max_missed_heartbeats = 5,
    .enable_heartbeat_stats = false
};
```

### High-Reliability Configuration
```c
node_config_t config = {
    .listen_port = 8080,
    .max_peers = 100,
    .keepalive_interval = 500,       // 500ms - faster detection
    .heartbeat_timeout_ms = 2000,    // 2 seconds
    .max_missed_heartbeats = 3,      // Fail fast
    .enable_heartbeat_stats = true
};
```

### Low-Bandwidth Configuration
```c
node_config_t config = {
    .listen_port = 8080,
    .max_peers = 50,
    .keepalive_interval = 5000,      // 5 seconds - less traffic
    .heartbeat_timeout_ms = 30000,   // 30 seconds - tolerant
    .max_missed_heartbeats = 10,
    .enable_heartbeat_stats = false
};
```

---

## Appendix B: Helper Functions Reference

### Heartbeat Helpers

```c
// Get current time in milliseconds
uint64_t get_time_ms(void);

// Get heartbeat state name
const char* heartbeat_state_name(heartbeat_state_t state);

// Check if heartbeat should be sent
bool should_send_heartbeat(p2p_node_t node, peer_info_t* peer, uint64_t now);

// Update heartbeat state
void update_heartbeat_state(peer_info_t* peer, const node_config_t* config);
```

### Logging Helpers

```c
// Initialize logging
void init_p2p_logging(const char* log_file);

// Format peer identifier
void format_peer_id(char* buffer, size_t size, const peer_info_t* peer);

// Get message type name
const char* msg_type_name(msg_type_t type);
```

### Iterator Helpers

```c
// Iterate peers with callback
void iterate_peers(p2p_node_t node, peer_iterator_fn_t callback, void* user_data);

// Count connected peers
uint32_t count_connected_peers(p2p_node_t node);

// Find peer by socket FD
peer_info_t* find_peer_by_socket(p2p_node_t node, int socket_fd);
```

---

## Appendix C: Error Codes

### Heartbeat Errors

| Code | Name | Description |
|------|------|-------------|
| 0 | HB_SUCCESS | Heartbeat sent successfully |
| -1 | HB_ERROR_SOCKET | Socket send failed |
| -2 | HB_ERROR_SERIALIZE | Message serialization failed |
| -3 | HB_ERROR_PEER_DEAD | Peer already marked dead |
| -4 | HB_ERROR_TIMEOUT | Heartbeat timeout |

---

## Appendix D: Metrics and Observability

### Heartbeat Metrics

```c
typedef struct {
    uint64_t heartbeats_sent;
    uint64_t heartbeats_received;
    uint64_t heartbeat_timeouts;
    uint64_t connections_failed;
    uint64_t connections_recovered;
    float avg_rtt_ms;
} heartbeat_stats_t;
```

### Logging Metrics

```c
typedef struct {
    uint64_t debug_logs;
    uint64_t info_logs;
    uint64_t warn_logs;
    uint64_t error_logs;
    uint64_t fatal_logs;
    uint64_t rate_limited;
} logging_stats_t;
```

---

**END OF DESIGN DOCUMENT**
