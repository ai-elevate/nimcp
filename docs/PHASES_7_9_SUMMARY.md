# Phases 7-9 Implementation Summary

## Quick Reference Guide

This document provides quick access to the key deliverables from the comprehensive design document.

---

## 1. Heartbeat Message Structure

### Protocol Definition

**Message Type:** `CTRL_MSG_HEARTBEAT` (0x0D) - Already defined in `nimcp_protocol.h`

**Wire Format:**
```
┌──────────────────────────────────┐
│  Control Message Header (24B)    │
├──────────────────────────────────┤
│  TLV 0x01: Node Status (1B)      │
│  TLV 0x02: Peer Count (4B)       │
│  TLV 0x03: Uptime (8B)           │
└──────────────────────────────────┘
```

### Sending Logic
```c
// Send heartbeat every keepalive_interval milliseconds
// OPTIMIZATION: Skip if data sent recently
if ((now - peer->last_data_time) >= node->config.keepalive_interval) {
    send_heartbeat_to_peer(node, peer);
}
```

### Receiving Logic
```c
// On heartbeat received:
peer->last_heartbeat_time = now;
peer->missed_heartbeat_count = 0;
peer->heartbeat_state = HB_STATE_HEALTHY;
```

---

## 2. Heartbeat State Machine (Per-Peer)

### State Definition

```c
typedef enum {
    HB_STATE_HEALTHY,      // Receiving heartbeats regularly
    HB_STATE_DEGRADED,     // Missed 1-2 heartbeats
    HB_STATE_FAILING,      // Missed 3-4 heartbeats
    HB_STATE_DEAD          // Connection considered dead
} heartbeat_state_t;
```

### State Transitions

```
    Heartbeat received
    ───────────────────────────────────┐
    │                                   │
    ▼                                   │
HEALTHY ──(miss)──> DEGRADED ──(miss)──> FAILING ──(miss)──> DEAD
  (0)               (1-2)                 (3-4)               (5+)
```

### Peer Structure Extensions

```c
typedef struct {
    // Existing fields
    char ip[16];
    uint16_t port;
    int socket_fd;
    bool connected;

    // NEW: Heartbeat tracking
    uint64_t last_heartbeat_time;      // Timestamp of last heartbeat
    uint64_t last_data_time;           // Timestamp of last data sent/recv
    uint32_t missed_heartbeat_count;   // Consecutive missed heartbeats
    heartbeat_state_t heartbeat_state; // Current state
    uint32_t heartbeat_sequence;       // Outgoing sequence number
} peer_info_t;
```

### Update Logic

```c
static void update_heartbeat_state(peer_info_t* peer, const node_config_t* config) {
    if (peer->missed_heartbeat_count == 0) {
        peer->heartbeat_state = HB_STATE_HEALTHY;
    } else if (peer->missed_heartbeat_count <= 2) {
        peer->heartbeat_state = HB_STATE_DEGRADED;
    } else if (peer->missed_heartbeat_count <= config->max_missed_heartbeats) {
        peer->heartbeat_state = HB_STATE_FAILING;
    } else {
        peer->heartbeat_state = HB_STATE_DEAD;
        // Trigger disconnect
    }
}
```

---

## 3. Comprehensive Logging Strategy

### Log Level Usage

| Level | Usage | Examples |
|-------|-------|----------|
| DEBUG | Verbose diagnostics | Heartbeat checks, iteration details |
| INFO | Lifecycle events | Connection established, node started |
| WARN | Degraded conditions | Connection degraded, queue backpressure |
| ERROR | Failures | Connection lost, send failed |
| FATAL | Critical failures | Initialization failed |

### Key Logging Events

#### Connection Events
```c
// Connection established
NIMCP_LOGGING_INFO("Connection established: %s:%u (socket=%d, peers=%u/%u)",
                  peer->ip, peer->port, peer->socket_fd,
                  node->peer_count, node->config.max_peers);

// Connection lost
NIMCP_LOGGING_ERROR("Connection lost: %s:%u (reason: %s, uptime=%lums)",
                   peer->ip, peer->port, disconnect_reason,
                   now - peer->connect_time);
```

#### Heartbeat Events
```c
// Heartbeat degraded
NIMCP_LOGGING_WARN("Heartbeat degraded: %s:%u (missed=%u, state=%s→%s)",
                  peer->ip, peer->port, peer->missed_heartbeat_count,
                  heartbeat_state_name(old_state),
                  heartbeat_state_name(new_state));

// Heartbeat timeout
NIMCP_LOGGING_ERROR("Heartbeat timeout: %s:%u (missed=%u, last_seen=%lums ago)",
                   peer->ip, peer->port, peer->missed_heartbeat_count,
                   now - peer->last_heartbeat_time);
```

#### Performance Considerations

**DO NOT log in hot paths:**
```c
// BAD: Logs every packet
for (int i = 0; i < count; i++) {
    NIMCP_LOGGING_DEBUG("Processing packet %d", i); // NO!
}

// GOOD: Aggregate statistics
if (should_log_stats(now)) {
    NIMCP_LOGGING_INFO("Processed %u packets in %us", count, interval);
}
```

**Use rate limiting:**
```c
if (should_log_event("heartbeat_check", 10)) { // Max 10/second
    NIMCP_LOGGING_DEBUG("Heartbeat check iteration");
}
```

---

## 4. Refactored Data Structure (Hash Table Only)

### Before (Antipattern)
```c
struct p2p_node_struct {
    peer_hash_table_t peer_table;  // For lookups
    peer_info_t* peers;            // For iteration - REDUNDANT
    uint32_t peer_count;           // - REDUNDANT
};
```

### After (Single Source of Truth)
```c
struct p2p_node_struct {
    peer_hash_table_t peer_table;  // Single source
    // Removed: peers array and peer_count
};
```

### Memory Savings
- **Before:** ~10KB for 100 peers
- **After:** ~7KB for 100 peers
- **Savings:** 30% reduction

---

## 5. Peer Iteration Without Array

### Iterator Interface

```c
/**
 * @brief Iterator callback for peer traversal
 */
typedef bool (*peer_iterator_fn_t)(peer_info_t* peer, void* user_data);

/**
 * @brief Iterate over all peers
 */
static void iterate_peers(p2p_node_t node,
                         peer_iterator_fn_t callback,
                         void* user_data);
```

### Usage Examples

**Example 1: Count connected peers**
```c
static bool count_callback(peer_info_t* peer, void* user_data) {
    uint32_t* count = (uint32_t*)user_data;
    if (peer->connected) (*count)++;
    return true; // Continue
}

uint32_t count = 0;
iterate_peers(node, count_callback, &count);
```

**Example 2: Disconnect all peers**
```c
static bool disconnect_callback(peer_info_t* peer, void* user_data) {
    close_peer_socket(peer);
    return true;
}

iterate_peers(node, disconnect_callback, NULL);
hash_table_clear(&node->peer_table);
```

**Example 3: Send broadcast message**
```c
typedef struct {
    uint8_t* message;
    size_t len;
    uint32_t sent;
} broadcast_ctx_t;

static bool broadcast_callback(peer_info_t* peer, void* user_data) {
    broadcast_ctx_t* ctx = (broadcast_ctx_t*)user_data;
    if (peer->connected) {
        send(peer->socket_fd, ctx->message, ctx->len, 0);
        ctx->sent++;
    }
    return true;
}

broadcast_ctx_t ctx = { .message = msg, .len = len, .sent = 0 };
iterate_peers(node, broadcast_callback, &ctx);
printf("Broadcast to %u peers\n", ctx.sent);
```

---

## 6. Line-by-Line Removal Plan

### File: `/home/bbrelin/src/repos/nimcp/src/lib/nimcp_p2pnode.c`

**Step-by-step refactoring:**

1. **Remove struct fields** (lines 127-128)
   ```c
   // DELETE:
   peer_info_t* peers;
   uint32_t peer_count;
   ```

2. **Remove array allocation** (lines 497-498)
   ```c
   // DELETE:
   node->peers = calloc(node->config.max_peers, sizeof(peer_info_t));
   ```

3. **Remove array initialization** (line 515)
   ```c
   // DELETE:
   node->peer_count = 0;
   ```

4. **Replace peer count checks** (lines 777, 825)
   ```c
   // BEFORE:
   if (node->peer_count >= node->config.max_peers)

   // AFTER:
   if (hash_table_size(&node->peer_table) >= node->config.max_peers)
   ```

5. **Replace peer allocation** (lines 780-781)
   ```c
   // BEFORE:
   peer_info_t* peer = &node->peers[node->peer_count];

   // AFTER:
   peer_info_t* peer = malloc(sizeof(peer_info_t));
   ```

6. **Remove peer count increment** (line 792)
   ```c
   // DELETE:
   node->peer_count++;
   ```

7. **Delete helper functions**
   - `find_peer_index()` (lines 900-912)
   - `compact_peer_array()` (lines 922-932)
   - `allocate_peer_storage()` (lines 493-499)

8. **Replace disconnect logic** (lines 975-980)
   ```c
   // BEFORE:
   int index = find_peer_index(node, peer_ip, peer_port);
   compact_peer_array(node, index);

   // AFTER:
   hash_table_remove_peer(&node->peer_table, key);
   ```

9. **Replace iteration** (lines 1071-1076)
   ```c
   // BEFORE:
   for (uint32_t i = 0; i < node->peer_count; i++) {
       disconnect_peer_during_shutdown(&node->peers[i]);
   }

   // AFTER:
   iterate_peers(node, disconnect_callback, NULL);
   hash_table_clear(&node->peer_table);
   ```

10. **Remove array deallocation** (line 619)
    ```c
    // DELETE:
    free(node->peers);
    ```

---

## 7. Configuration Reference

### Node Configuration Structure

```c
typedef struct {
    uint16_t listen_port;
    uint32_t max_peers;

    // Heartbeat configuration
    uint32_t keepalive_interval;       // Heartbeat interval (ms)
    uint32_t heartbeat_timeout_ms;     // NEW: Timeout for response
    uint32_t max_missed_heartbeats;    // NEW: Max missed before disconnect
    bool enable_heartbeat_stats;       // NEW: Collect statistics

    // Existing fields
    uint32_t discovery_interval;
    uint32_t reconnect_interval;
    uint32_t max_retries;
    uint32_t ping_interval;
} node_config_t;
```

### Default Values

```c
node_config_t default_config = {
    .listen_port = 8080,
    .max_peers = 100,
    .keepalive_interval = 1000,           // 1 second
    .heartbeat_timeout_ms = 5000,         // 5 seconds
    .max_missed_heartbeats = 5,           // 5 * 1s = 5s timeout
    .enable_heartbeat_stats = false,
    .discovery_interval = 10000,
    .reconnect_interval = 5000,
    .max_retries = 3,
    .ping_interval = 1000
};
```

### Configuration Scenarios

**Fast failure detection:**
```c
.keepalive_interval = 500,        // 500ms
.max_missed_heartbeats = 3,       // Fail after 1.5s
```

**Low bandwidth:**
```c
.keepalive_interval = 5000,       // 5 seconds
.max_missed_heartbeats = 10,      // Fail after 50s
```

**High reliability:**
```c
.keepalive_interval = 1000,       // 1 second
.max_missed_heartbeats = 5,       // Fail after 5s
.enable_heartbeat_stats = true,
```

---

## 8. Implementation Timeline

### Week 1: Heartbeat + Logging

**Days 1-3: Heartbeat**
- [ ] Add heartbeat fields to `peer_info_t`
- [ ] Implement state machine
- [ ] Create heartbeat thread
- [ ] Implement send/receive logic
- [ ] Add timeout detection
- [ ] Write unit tests

**Days 4-5: Logging**
- [ ] Initialize logging in `p2p_node_create`
- [ ] Add connection lifecycle logging
- [ ] Add heartbeat event logging
- [ ] Add message send/receive logging (rate-limited)
- [ ] Configure log levels
- [ ] Write integration tests

### Week 2: Refactoring

**Days 1-2: Iterator**
- [ ] Implement `iterate_peers()` wrapper
- [ ] Replace all array iterations
- [ ] Test iterator functionality

**Days 3-4: Array Removal**
- [ ] Remove array allocation
- [ ] Update `add_peer_to_node()`
- [ ] Update disconnect logic
- [ ] Remove helper functions
- [ ] Remove struct fields

**Day 5: Testing**
- [ ] Run regression tests
- [ ] Performance benchmarks
- [ ] Memory leak checks (valgrind)
- [ ] Stress tests

---

## 9. Testing Checklist

### Unit Tests

- [ ] Heartbeat state transitions (HEALTHY → DEGRADED → FAILING → DEAD)
- [ ] Heartbeat suppression on active connections
- [ ] Iterator counts all peers correctly
- [ ] Iterator early termination works
- [ ] Peer allocation/deallocation (no leaks)
- [ ] Hash table size tracking

### Integration Tests

- [ ] Heartbeat timeout triggers disconnect
- [ ] Heartbeat recovery restores HEALTHY state
- [ ] Logging captures all lifecycle events
- [ ] Logging rate limiting works
- [ ] Broadcast message via iterator
- [ ] Disconnect all peers via iterator

### Stress Tests

- [ ] 1000 peers iteration performance
- [ ] Rapid connect/disconnect (churn test)
- [ ] Memory usage with many peers
- [ ] Heartbeat thread CPU usage
- [ ] Log file growth rate
- [ ] Concurrent operations thread safety

---

## 10. Quick Start Code Snippets

### Initialize Node with Heartbeat

```c
#include "nimcp_p2pnode.h"
#include "logging/nimcp_logging.h"

int main() {
    // Initialize logging
    log_init("/var/log/nimcp/p2pnode.log");

    // Configure node with heartbeat
    node_config_t config = {
        .listen_port = 8080,
        .max_peers = 100,
        .keepalive_interval = 1000,
        .heartbeat_timeout_ms = 5000,
        .max_missed_heartbeats = 5
    };

    // Create and start node
    p2p_node_t node = p2p_node_create(&config);
    if (!node) {
        NIMCP_LOGGING_FATAL("Failed to create P2P node");
        return 1;
    }

    if (!p2p_node_start(node)) {
        NIMCP_LOGGING_FATAL("Failed to start P2P node");
        p2p_node_destroy(node);
        return 1;
    }

    NIMCP_LOGGING_INFO("P2P node running on port %u", config.listen_port);

    // Run event loop...

    // Cleanup
    p2p_node_stop(node);
    p2p_node_destroy(node);
    log_close();

    return 0;
}
```

### Iterate and Process Peers

```c
// Define callback
static bool process_peer(peer_info_t* peer, void* user_data) {
    if (!peer->connected) return true; // Skip disconnected

    // Check heartbeat health
    if (peer->heartbeat_state == HB_STATE_DEGRADED) {
        printf("WARN: Peer %s:%u degraded\n", peer->ip, peer->port);
    }

    // Send data or perform processing
    // ...

    return true; // Continue iteration
}

// Use iterator
iterate_peers(node, process_peer, NULL);
```

### Handle Heartbeat Events

```c
// In message dispatcher
static void handle_message(p2p_node_t node, peer_info_t* peer,
                          const msg_header_t* header) {
    switch (header->type) {
        case MSG_TYPE_HEARTBEAT:
            // Update peer heartbeat state
            peer->last_heartbeat_time = get_time_ms();
            peer->missed_heartbeat_count = 0;
            peer->heartbeat_state = HB_STATE_HEALTHY;

            NIMCP_LOGGING_DEBUG("Heartbeat from %s:%u (seq=%u)",
                               peer->ip, peer->port, header->sequence);
            break;

        // ... other message types ...
    }
}
```

---

## Files Modified

### Headers
- `/home/bbrelin/src/repos/nimcp/src/include/nimcp_p2pnode.h`
  - Add heartbeat fields to `node_config_t`
  - No changes to public API

### Implementation
- `/home/bbrelin/src/repos/nimcp/src/lib/nimcp_p2pnode.c`
  - Add heartbeat tracking fields to `peer_info_t`
  - Implement heartbeat state machine
  - Add heartbeat thread
  - Integrate nimcp_logging throughout
  - Remove `peers` array and `peer_count` field
  - Replace array operations with hash table iteration

### Tests
- `/home/bbrelin/src/repos/nimcp/src/tests/test_p2pnode.cpp`
  - Add heartbeat state tests
  - Add iterator tests
  - Add logging verification tests

---

## Performance Characteristics

### Complexity Analysis

| Operation | Before | After | Change |
|-----------|--------|-------|--------|
| Lookup peer | O(1) | O(1) | Same |
| Add peer | O(1) | O(1) | Same |
| Remove peer | O(n) | O(1) | **Improved** |
| Iterate all | O(n) | O(n) | Same |
| Get count | O(1) | O(1) | Same |
| Send heartbeat | - | O(1) | New |
| Check heartbeats | - | O(n) | New (periodic) |

### Memory Usage

| Structure | Before | After | Savings |
|-----------|--------|-------|---------|
| 100 peers | ~10 KB | ~7 KB | 30% |
| 1000 peers | ~100 KB | ~70 KB | 30% |

### CPU Impact

- **Heartbeat thread:** < 1% CPU (1000ms interval)
- **Logging:** < 0.1% CPU (with rate limiting)
- **Iterator:** Same as array iteration (O(n))

---

## References

**Full Design Document:**
`/home/bbrelin/src/repos/nimcp/docs/HEARTBEAT_LOGGING_REFACTORING_DESIGN.md`

**Related Files:**
- `/home/bbrelin/src/repos/nimcp/src/include/nimcp_protocol.h` - Protocol definitions
- `/home/bbrelin/src/repos/nimcp/src/include/logging/nimcp_logging.h` - Logging API
- `/home/bbrelin/src/repos/nimcp/src/include/utils/nimcp_hash_table.h` - Hash table API
- `/home/bbrelin/src/repos/nimcp/src/include/utils/nimcp_thread.h` - Thread utilities

---

**Last Updated:** 2025-11-01
**Status:** Ready for Implementation
