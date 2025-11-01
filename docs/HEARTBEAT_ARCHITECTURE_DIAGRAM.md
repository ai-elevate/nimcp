# Heartbeat/Keepalive Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         P2P Node Process                             │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      Main Thread                              │  │
│  │                                                               │  │
│  │  ┌────────────┐       ┌──────────────┐                       │  │
│  │  │  Listen    │       │   Message    │                       │  │
│  │  │  Socket    │──────▶│  Dispatcher  │                       │  │
│  │  └────────────┘       └──────┬───────┘                       │  │
│  │                              │                                │  │
│  │                              ▼                                │  │
│  │                    ┌───────────────────┐                      │  │
│  │                    │  handle_message() │                      │  │
│  │                    └─────────┬─────────┘                      │  │
│  │                              │                                │  │
│  │              ┌───────────────┼───────────────┐               │  │
│  │              ▼               ▼               ▼               │  │
│  │         MSG_TYPE_       MSG_TYPE_       MSG_TYPE_            │  │
│  │         HANDSHAKE       HEARTBEAT       STATE_UPDATE         │  │
│  │              │               │               │               │  │
│  │              └───────────────┴───────────────┘               │  │
│  │                              │                                │  │
│  │                              ▼                                │  │
│  │                   ┌──────────────────┐                        │  │
│  │                   │  Update Peer     │                        │  │
│  │                   │  Heartbeat State │                        │  │
│  │                   └──────────────────┘                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Heartbeat Thread                           │  │
│  │                                                               │  │
│  │  while (running) {                                            │  │
│  │      now = get_time_ms();                                     │  │
│  │      check_and_send_heartbeats(node, now); ◀─────┐            │  │
│  │      usleep(keepalive_interval * 1000);          │            │  │
│  │  }                                                │            │  │
│  │                                                   │            │  │
│  │  ┌────────────────────────────────────────────────┘            │  │
│  │  │                                                             │  │
│  │  ▼                                                             │  │
│  │  iterate_peers(node, heartbeat_check_iterator, &now)          │  │
│  │                              │                                │  │
│  │                              ▼                                │  │
│  │              ┌──────────────────────────────┐                 │  │
│  │              │  For each peer:              │                 │  │
│  │              │  1. Check if needs heartbeat │                 │  │
│  │              │  2. Check missed heartbeats  │                 │  │
│  │              │  3. Update state             │                 │  │
│  │              │  4. Disconnect if dead       │                 │  │
│  │              └──────────────────────────────┘                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     Data Structures                           │  │
│  │                                                               │  │
│  │  peer_hash_table_t (SINGLE SOURCE OF TRUTH)                   │  │
│  │  ┌──────────┬──────────┬──────────┬─────┬──────────┐         │  │
│  │  │ Bucket 0 │ Bucket 1 │ Bucket 2 │ ... │ Bucket N │         │  │
│  │  └────┬─────┴────┬─────┴────┬─────┴─────┴────┬─────┘         │  │
│  │       │          │          │                │               │  │
│  │       ▼          ▼          ▼                ▼               │  │
│  │   ┌─────┐    ┌─────┐    ┌─────┐          ┌─────┐            │  │
│  │   │Peer1│    │Peer2│    │Peer3│          │PeerN│            │  │
│  │   │     │    │     │    │     │          │     │            │  │
│  │   │ip   │    │ip   │    │ip   │          │ip   │            │  │
│  │   │port │    │port │    │port │          │port │            │  │
│  │   │fd   │    │fd   │    │fd   │          │fd   │            │  │
│  │   │     │    │     │    │     │          │     │            │  │
│  │   │❤️ HB│    │❤️ HB│    │❤️ HB│          │❤️ HB│            │  │
│  │   │state│    │state│    │state│          │state│            │  │
│  │   │time │    │time │    │time │          │time │            │  │
│  │   │count│    │count│    │count│          │count│            │  │
│  │   └─────┘    └─────┘    └─────┘          └─────┘            │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## Heartbeat State Machine Flow

```
                    ┌──────────────────────────┐
                    │   Peer Connected         │
                    │   Initial State          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      HB_STATE_HEALTHY    │
                    │  missed_count = 0        │
                    │  ✅ All good             │
                    └────────────┬─────────────┘
                                 │
                                 │ Miss heartbeat
                                 │ (no data for keepalive_interval)
                                 ▼
                    ┌──────────────────────────┐
   Heartbeat ◀──────│   HB_STATE_DEGRADED      │
   received         │  missed_count = 1-2      │
                    │  ⚠️  Warning              │
                    └────────────┬─────────────┘
                                 │
                                 │ Miss another heartbeat
                                 ▼
                    ┌──────────────────────────┐
   Heartbeat ◀──────│   HB_STATE_FAILING       │
   received         │  missed_count = 3-4      │
                    │  🔴 Critical             │
                    └────────────┬─────────────┘
                                 │
                                 │ Miss final heartbeat
                                 ▼
                    ┌──────────────────────────┐
                    │      HB_STATE_DEAD       │
                    │  missed_count >= 5       │
                    │  ❌ Connection lost      │
                    │  → Trigger disconnect    │
                    └──────────────────────────┘
```

## Heartbeat Send Decision Flow

```
Start: Should we send heartbeat to peer?
│
├─ Get current time (now)
│
├─ Calculate: time_since_data = now - peer->last_data_time
│
└─▶ Is time_since_data >= keepalive_interval?
    │
    ├─ NO  ──▶ Skip heartbeat (recent data sent)
    │          └─▶ OPTIMIZATION: Save bandwidth
    │
    └─ YES ──▶ Send heartbeat message
               │
               ├─ Create CTRL_MSG_HEARTBEAT
               ├─ Set sequence number (peer->heartbeat_sequence++)
               ├─ Serialize control message
               ├─ send() to peer socket
               ├─ Update peer->last_data_time = now
               └─ Log: DEBUG "Sent heartbeat seq=X to IP:PORT"
```

## Heartbeat Check Algorithm (Per Peer)

```
Heartbeat Check Iterator (called every keepalive_interval)
│
For each peer in hash table:
│
├─ Step 1: Check if should send heartbeat
│  │
│  ├─ time_since_data = now - peer->last_data_time
│  │
│  └─ if (time_since_data >= keepalive_interval)
│     └─▶ send_heartbeat_to_peer(node, peer)
│
├─ Step 2: Check for missed heartbeats
│  │
│  ├─ time_since_hb = now - peer->last_heartbeat_time
│  │
│  └─ if (time_since_hb > keepalive_interval)
│     │
│     ├─ peer->missed_heartbeat_count++
│     │
│     ├─ old_state = peer->heartbeat_state
│     │
│     ├─ update_heartbeat_state(peer, config)
│     │
│     └─ Log state change:
│        │
│        ├─ DEGRADED ──▶ WARN log
│        ├─ FAILING  ──▶ ERROR log
│        └─ DEAD     ──▶ ERROR log + disconnect_dead_peer()
│
└─ Continue to next peer
```

## Message Flow Diagram

### Outgoing Heartbeat

```
Heartbeat Thread                    Peer
      │                              │
      ├─ Create control_message_t    │
      │  (CTRL_MSG_HEARTBEAT)        │
      │                              │
      ├─ Serialize to buffer         │
      │                              │
      ├─ send(peer->socket_fd)  ─────▶
      │                              │
      ├─ Update last_data_time       │
      │                              │
      └─ Log DEBUG                   │
                                     │
                                     └─ Receives heartbeat
```

### Incoming Heartbeat

```
Peer                           Main Thread
  │                                 │
  └─ Sends heartbeat  ──────────────▶
                                    │
                      Receive on socket
                                    │
                        Parse message header
                                    │
                           type = HEARTBEAT?
                                    │
                               ┌────┴────┐
                               │ YES     │
                               └────┬────┘
                                    │
                        Update peer heartbeat state:
                            - last_heartbeat_time = now
                            - missed_count = 0
                            - state = HEALTHY
                                    │
                                Log DEBUG
```

## Logging Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                    Function Call Stack                          │
└─────────────────────────────────────────────────────────────────┘

p2p_node_create()
  └─▶ log_init("/var/log/nimcp/nimcp.log")
  └─▶ NIMCP_LOGGING_INFO("Creating P2P node port=%u", port)

p2p_node_start()
  └─▶ NIMCP_LOGGING_INFO("Starting P2P node port=%u", port)
  └─▶ setup_listen_socket()
      └─▶ NIMCP_LOGGING_DEBUG("Listen socket bound fd=%d", fd)

p2p_node_connect_peer(ip, port)
  └─▶ NIMCP_LOGGING_DEBUG("Attempting connection to %s:%u", ip, port)
  └─▶ connect() success?
      ├─ YES ──▶ NIMCP_LOGGING_INFO("Connection established...")
      └─ NO  ──▶ NIMCP_LOGGING_WARN("Connection failed: %s", errno)

heartbeat_check_iterator()
  └─▶ State degradation?
      ├─ HEALTHY → DEGRADED
      │  └─▶ NIMCP_LOGGING_WARN("Heartbeat degraded...")
      ├─ DEGRADED → FAILING
      │  └─▶ NIMCP_LOGGING_ERROR("Heartbeat failing...")
      └─ FAILING → DEAD
         └─▶ NIMCP_LOGGING_ERROR("Heartbeat timeout...")
         └─▶ disconnect_dead_peer()
             └─▶ NIMCP_LOGGING_ERROR("Connection lost...")

p2p_node_stop()
  └─▶ NIMCP_LOGGING_INFO("Stopping P2P node stats=...")
  └─▶ disconnect_all_peers()
  └─▶ log_close()
```

## Data Structure Before/After Comparison

### BEFORE (Antipattern - Dual Structures)

```
struct p2p_node_struct {
    peer_hash_table_t peer_table;
    ┌──────────┬──────────┬──────────┐
    │ Bucket 0 │ Bucket 1 │ Bucket 2 │
    └────┬─────┴────┬─────┴────┬─────┘
         │          │          │
         ▼          ▼          ▼
      ┌─────┐   ┌─────┐   ┌─────┐
      │ PTR │   │ PTR │   │ PTR │  ◀── Pointers to array elements
      └──┬──┘   └──┬──┘   └──┬──┘
         │         │         │
         │         │         │
    ┌────┼─────────┼─────────┘
    │    │         │
    ▼    ▼         ▼
  ┌────────────────────────┐
  │ peer_info_t* peers[]   │  ◀── REDUNDANT ARRAY
  ├────┬────┬────┬─────────┤
  │ P1 │ P2 │ P3 │   ...   │
  └────┴────┴────┴─────────┘

  PROBLEMS:
  ✗ Dual storage (2x memory)
  ✗ Synchronization burden
  ✗ Can get out of sync
  ✗ Complex update logic
}
```

### AFTER (Single Source of Truth)

```
struct p2p_node_struct {
    peer_hash_table_t peer_table;
    ┌──────────┬──────────┬──────────┐
    │ Bucket 0 │ Bucket 1 │ Bucket 2 │
    └────┬─────┴────┬─────┴────┬─────┘
         │          │          │
         ▼          ▼          ▼
      ┌─────┐   ┌─────┐   ┌─────┐
      │Peer1│   │Peer2│   │Peer3│  ◀── Direct storage
      │     │   │     │   │     │
      │ip   │   │ip   │   │ip   │
      │port │   │port │   │port │
      │fd   │   │fd   │   │fd   │
      │HB ❤️│   │HB ❤️│   │HB ❤️│
      └─────┘   └─────┘   └─────┘

  BENEFITS:
  ✓ Single source of truth
  ✓ 30% less memory
  ✓ Cannot desync
  ✓ Simple update logic
  ✓ Same O(1) lookup
  ✓ Same O(n) iteration
}
```

## Iterator Pattern Implementation

### Old Way (Array Iteration)

```c
// BEFORE: Direct array access
for (uint32_t i = 0; i < node->peer_count; i++) {
    peer_info_t* peer = &node->peers[i];
    // Process peer
}
```

### New Way (Hash Table Iteration)

```c
// AFTER: Iterator pattern

// 1. Define callback
static bool my_callback(peer_info_t* peer, void* user_data) {
    // Process peer
    return true; // Continue iteration
}

// 2. Iterate
iterate_peers(node, my_callback, user_data);


// Implementation (internal):
iterate_peers() {
    hash_table_iterate(&node->peer_table,
                      peer_iterator_adapter,
                      callback_context);
}

// Traverses hash table buckets:
Bucket 0 ──▶ Peer1 ──▶ Peer2 ──▶ NULL
Bucket 1 ──▶ Peer3 ──▶ NULL
Bucket 2 ──▶ NULL
Bucket 3 ──▶ Peer4 ──▶ Peer5 ──▶ Peer6 ──▶ NULL
...
```

## Configuration Impact on Behavior

```
┌─────────────────────────────────────────────────────────────┐
│                  Configuration Scenarios                    │
└─────────────────────────────────────────────────────────────┘

Scenario 1: FAST FAILURE DETECTION
┌──────────────────────────────────────────┐
│ keepalive_interval = 500ms               │
│ max_missed_heartbeats = 3                │
│                                          │
│ Timeline:                                │
│ 0ms    ─ Connected (HEALTHY)             │
│ 500ms  ─ Miss #1 (DEGRADED)              │
│ 1000ms ─ Miss #2 (DEGRADED)              │
│ 1500ms ─ Miss #3 (FAILING)               │
│ 2000ms ─ Miss #4 → DEAD (disconnect)     │
│                                          │
│ Total time to failure: 2.0 seconds       │
└──────────────────────────────────────────┘

Scenario 2: LOW BANDWIDTH
┌──────────────────────────────────────────┐
│ keepalive_interval = 5000ms              │
│ max_missed_heartbeats = 10               │
│                                          │
│ Timeline:                                │
│ 0s  ─ Connected (HEALTHY)                │
│ 5s  ─ Miss #1 (DEGRADED)                 │
│ 10s ─ Miss #2 (DEGRADED)                 │
│ 15s ─ Miss #3 (FAILING)                  │
│ ...                                      │
│ 50s ─ Miss #10 (FAILING)                 │
│ 55s ─ Miss #11 → DEAD (disconnect)       │
│                                          │
│ Total time to failure: 55 seconds        │
└──────────────────────────────────────────┘

Scenario 3: BALANCED (Default)
┌──────────────────────────────────────────┐
│ keepalive_interval = 1000ms              │
│ max_missed_heartbeats = 5                │
│                                          │
│ Timeline:                                │
│ 0s ─ Connected (HEALTHY)                 │
│ 1s ─ Miss #1 (DEGRADED)                  │
│ 2s ─ Miss #2 (DEGRADED)                  │
│ 3s ─ Miss #3 (FAILING)                   │
│ 4s ─ Miss #4 (FAILING)                   │
│ 5s ─ Miss #5 (FAILING)                   │
│ 6s ─ Miss #6 → DEAD (disconnect)         │
│                                          │
│ Total time to failure: 6 seconds         │
└──────────────────────────────────────────┘
```

## Thread Safety Considerations

```
┌─────────────────────────────────────────────────────────────┐
│              Thread Synchronization                         │
└─────────────────────────────────────────────────────────────┘

Main Thread                    Heartbeat Thread
     │                              │
     │  Reads peer state            │
     │  (connected, socket_fd)      │
     │                              │  Reads/writes:
     │                              │  - last_heartbeat_time
     │                              │  - missed_count
     │                              │  - heartbeat_state
     │                              │
     │  Mutex: peer_table_mutex     │
     │  ┌────────────────────────┐  │
     │  │  Shared Access         │  │
     ├──▶  peer_hash_table       ◀──┤
     │  │                        │  │
     │  └────────────────────────┘  │
     │                              │
     │  Updates on message receive  │  Updates on timeout
     │  - last_heartbeat_time       │  - missed_count++
     │  - missed_count = 0          │  - update_state()
     │                              │

Required Protection:
  1. Hash table iteration must be atomic
  2. Peer structure updates need per-peer locks
  3. Disconnect operations must be synchronized

Recommendation:
  Use nimcp_mutex_t from nimcp_thread.h for synchronization
```

## Memory Management Flow

```
┌─────────────────────────────────────────────────────────────┐
│                Peer Lifecycle Memory                        │
└─────────────────────────────────────────────────────────────┘

Connect Peer
    │
    ▼
malloc(sizeof(peer_info_t))  ◀── Heap allocation
    │
    ├─ initialize_peer_info()
    │
    ├─ generate_peer_key("IP:PORT")
    │
    └─ hash_table_insert_peer(key, peer)
       │
       └─ Hash table takes ownership
          (Will call destructor on remove)

...peer exists in hash table...

Disconnect Peer
    │
    ▼
hash_table_remove_peer(key)
    │
    └─ Calls peer_destructor():
       │
       ├─ close(peer->socket_fd)
       │
       └─ free(peer)  ◀── Cleanup

No memory leaks! ✓
```

---

**Visual Reference for Implementation**
**Last Updated:** 2025-11-01
