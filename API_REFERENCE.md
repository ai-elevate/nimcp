# NIMCP API Reference

**Version:** 2.5.0
**Last Updated:** 2025-11-01

This document provides comprehensive reference documentation for all public APIs in the NIMCP (Neuromorphic Infant Machine Cognitive Platform) library.

## Table of Contents

1. [Core Neural Network APIs](#core-neural-network-apis)
2. [Brain & Cognitive Systems](#brain--cognitive-systems)
3. [Adaptive Learning](#adaptive-learning)
4. [Event Processing](#event-processing)
5. [P2P Networking](#p2p-networking)
6. [Data I/O & Streaming](#data-io--streaming)
7. [Attention & Salience](#attention--salience)
8. [Memory Consolidation](#memory-consolidation)
9. [Introspection & Monitoring](#introspection--monitoring)
10. [Higher-Level Cognitive APIs](#higher-level-cognitive-apis)
11. [Utility APIs](#utility-apis)
12. [Python Bindings](#python-bindings)

---

## Core Neural Network APIs

### Neural Network Creation & Management
**Header:** `nimcp_neuralnet.h`

#### Types
```c
typedef struct nimcp_neuralnet_t nimcp_neuralnet_t;

typedef struct {
    uint32_t num_inputs;
    uint32_t num_outputs;
    uint32_t num_hidden;
    float learning_rate;
    float stdp_a_plus;
    float stdp_a_minus;
    float tau_plus;
    float tau_minus;
} nimcp_neuralnet_config_t;

typedef struct {
    uint32_t total_neurons;
    uint32_t total_synapses;
    float avg_activity;
    float network_stability;
} network_stats_t;
```

#### Functions

**`nimcp_neuralnet_t* nimcp_neuralnet_create(const nimcp_neuralnet_config_t* config)`**
- **Description:** Creates a new spiking neural network
- **Parameters:**
  - `config`: Network configuration (inputs, outputs, hidden neurons, learning parameters)
- **Returns:** Network handle or NULL on failure
- **Thread Safety:** Thread-safe

**`void nimcp_neuralnet_destroy(nimcp_neuralnet_t* net)`**
- **Description:** Destroys a neural network and frees resources
- **Parameters:**
  - `net`: Network handle
- **Thread Safety:** Must not be called concurrently with same network

**`void nimcp_neuralnet_forward(nimcp_neuralnet_t* net, const float* inputs, float* outputs)`**
- **Description:** Performs forward propagation through the network
- **Parameters:**
  - `net`: Network handle
  - `inputs`: Input array (size = num_inputs)
  - `outputs`: Output array (size = num_outputs)
- **Thread Safety:** Not thread-safe, external synchronization required

**`void nimcp_neuralnet_backward(nimcp_neuralnet_t* net, const float* target, float* error)`**
- **Description:** Performs backpropagation and learning
- **Parameters:**
  - `net`: Network handle
  - `target`: Target output values
  - `error`: Computed error (output)
- **Thread Safety:** Not thread-safe

**`void nimcp_neuralnet_get_stats(nimcp_neuralnet_t* net, network_stats_t* stats)`**
- **Description:** Retrieves network statistics
- **Parameters:**
  - `net`: Network handle
  - `stats`: Output statistics structure
- **Thread Safety:** Thread-safe (read-only operation)

**`int nimcp_neuralnet_save(nimcp_neuralnet_t* net, const char* filename)`**
- **Description:** Saves network to file
- **Returns:** 0 on success, -1 on failure

**`nimcp_neuralnet_t* nimcp_neuralnet_load(const char* filename)`**
- **Description:** Loads network from file
- **Returns:** Network handle or NULL on failure

---

## Brain & Cognitive Systems

### Brain API
**Header:** `nimcp_brain.h`

Provides high-level cognitive brain abstraction that combines multiple neural networks with memory consolidation, salience detection, and learning.

#### Types
```c
typedef struct nimcp_brain_t nimcp_brain_t;

typedef struct {
    uint32_t num_inputs;
    uint32_t num_outputs;
    uint32_t hidden_layers[16];
    uint32_t num_hidden_layers;
    uint32_t consolidation_interval;
    float salience_threshold;
    char task_name[128];
} nimcp_brain_config_t;

typedef struct {
    uint64_t total_experiences;
    uint32_t memories_consolidated;
    float avg_salience;
    float learning_progress;
} brain_stats_t;
```

#### Functions

**`nimcp_brain_t* nimcp_brain_create(const nimcp_brain_config_t* config)`**
- **Description:** Creates a cognitive brain with integrated learning systems
- **Features:** Multi-layer networks, automatic consolidation, salience tracking
- **Returns:** Brain handle or NULL on failure

**`void nimcp_brain_destroy(nimcp_brain_t* brain)`**
- **Description:** Destroys brain and all subsystems

**`void nimcp_brain_process(nimcp_brain_t* brain, const float* inputs, float* outputs)`**
- **Description:** Process inputs through the brain, update learning
- **Side Effects:** Updates internal state, triggers consolidation if needed

**`void nimcp_brain_consolidate(nimcp_brain_t* brain)`**
- **Description:** Manually trigger memory consolidation
- **Use Case:** Consolidate important experiences into long-term memory

**`void nimcp_brain_get_stats(nimcp_brain_t* brain, brain_stats_t* stats)`**
- **Description:** Get brain statistics and learning progress

**`int nimcp_brain_save(nimcp_brain_t* brain, const char* filename)`**
- **Description:** Save complete brain state to file

**`nimcp_brain_t* nimcp_brain_load(const char* filename)`**
- **Description:** Load brain from saved state

---

## Adaptive Learning

### Adaptive Threshold Spiking Network
**Header:** `nimcp_adaptive.h`

Implements adaptive threshold spiking neurons with dynamic sparsity control.

#### Types
```c
typedef struct adaptive_network_t adaptive_network_t;

typedef struct {
    uint32_t num_inputs;
    uint32_t num_outputs;
    uint32_t num_hidden;
    float k_factor;              // Winner-take-all factor
    float min_threshold;
    float max_threshold;
    float target_sparsity;       // 0.0 - 1.0
} adaptive_network_config_t;
```

#### Functions

**`adaptive_network_t* adaptive_network_create(const adaptive_network_config_t* config)`**
- **Description:** Create adaptive spiking network with homeostatic plasticity
- **Returns:** Network handle or NULL

**`void adaptive_network_forward(adaptive_network_t* net, const float* inputs, float* outputs)`**
- **Description:** Forward pass with adaptive thresholds
- **Features:** k-winner-take-all competition, dynamic threshold adjustment

**`float adaptive_network_get_sparsity(adaptive_network_t* net)`**
- **Description:** Get current activation sparsity (fraction of neurons active)
- **Returns:** Sparsity value (0.0 = dense, 1.0 = sparse)

**`void adaptive_network_prune(adaptive_network_t* net, float threshold)`**
- **Description:** Prune weak synapses below threshold

**`void adaptive_network_learn_supervised(adaptive_network_t* net, const float* inputs, const float* targets, float learning_rate)`**
- **Description:** Supervised learning with target outputs

**`void adaptive_network_learn_unsupervised(adaptive_network_t* net, const float* inputs, float learning_rate)`**
- **Description:** Unsupervised Hebbian learning

---

## Event Processing

### Event-Driven Processing
**Header:** `nimcp_events.h`

Event-based spike encoding and processing for neuromorphic communication.

#### Types
```c
typedef struct event_generator_t event_generator_t;
typedef struct event_receiver_t event_receiver_t;

typedef struct {
    uint64_t timestamp;
    uint32_t source_neuron;
    uint16_t feature_code;
    uint8_t  confidence;
    uint8_t  flags;
} event_packet_t;

typedef void (*event_callback_t)(const event_packet_t* event, void* user_data);
```

#### Event Generator Functions

**`event_generator_t* event_generator_create(const event_generator_config_t* config, event_callback_t callback, void* user_data)`**
- **Description:** Create event generator attached to neural network
- **Parameters:**
  - `config`: Generator configuration
  - `callback`: Function called when events are generated
  - `user_data`: User context passed to callback

**`void event_generator_on_spike(event_generator_t* gen, uint32_t neuron_id)`**
- **Description:** Notify generator of neuron spike
- **Use Case:** Connect to network spike output

#### Event Receiver Functions

**`event_receiver_t* event_receiver_create(const event_receiver_config_t* config, nimcp_neuralnet_t* target_network)`**
- **Description:** Create event receiver that injects spikes into network

**`void event_receiver_process(event_receiver_t* recv, const event_packet_t* packet)`**
- **Description:** Process incoming event packet
- **Side Effects:** May inject current into target neurons

**`void event_receiver_add_filter(event_receiver_t* recv, uint16_t feature_code, bool enabled)`**
- **Description:** Enable/disable filtering for specific feature codes

---

## P2P Networking

### Peer-to-Peer Node
**Header:** `nimcp_p2pnode.h`

Distributed neural network communication via P2P networking.

#### Types
```c
typedef struct p2p_node_t p2p_node_t;

typedef struct {
    char node_name[64];
    uint16_t listen_port;
    uint32_t max_peers;
    uint32_t connection_timeout_ms;
} p2p_node_config_t;

typedef struct {
    char ip[64];
    uint16_t port;
    uint32_t latency_ms;
    bool connected;
} peer_info_t;
```

#### Functions

**`p2p_node_t* p2p_node_create(const p2p_node_config_t* config)`**
- **Description:** Create P2P node for distributed learning

**`int p2p_node_start(p2p_node_t* node)`**
- **Description:** Start listening for connections
- **Returns:** 0 on success

**`int p2p_node_connect(p2p_node_t* node, const char* peer_ip, uint16_t peer_port)`**
- **Description:** Connect to remote peer
- **Returns:** 0 on success, -1 on failure

**`int p2p_node_broadcast(p2p_node_t* node, const uint8_t* data, size_t size)`**
- **Description:** Broadcast data to all connected peers

**`int p2p_node_get_peers(p2p_node_t* node, peer_info_t* peers, uint32_t* count)`**
- **Description:** Get list of connected peers

**`void p2p_node_stop(p2p_node_t* node)`**
- **Description:** Stop node and disconnect all peers

---

## Data I/O & Streaming

### Data I/O
**Header:** `nimcp_dataio.h`

CSV data loading, batch processing, and training utilities.

#### Types
```c
typedef struct nimcp_dataset_t nimcp_dataset_t;

typedef struct {
    float** inputs;
    float** outputs;
    uint32_t count;
} nimcp_batch_t;
```

#### Functions

**`nimcp_dataset_t* nimcp_dataio_load_csv(const char* filepath, bool has_header)`**
- **Description:** Load dataset from CSV file
- **Returns:** Dataset handle or NULL

**`int nimcp_dataio_read_batch(nimcp_dataset_t* dataset, nimcp_batch_t* batch, uint32_t batch_size)`**
- **Description:** Read next batch from dataset
- **Returns:** Number of samples read

**`void nimcp_dataio_reset(nimcp_dataset_t* dataset)`**
- **Description:** Reset dataset iterator to beginning

**`int nimcp_dataio_train(nimcp_brain_t* brain, nimcp_dataset_t* dataset, uint32_t epochs, float validation_split)`**
- **Description:** Train brain on dataset
- **Parameters:**
  - `validation_split`: Fraction for validation (0.0-1.0)
- **Returns:** 0 on success

**`int nimcp_dataio_save_csv(const char* filepath, const float** data, uint32_t rows, uint32_t cols, const char** headers)`**
- **Description:** Save data to CSV file

### Streaming API
**Header:** `nimcp_stream.h`

Real-time streaming input processing.

#### Types
```c
typedef struct nimcp_stream_t nimcp_stream_t;

typedef enum {
    STREAM_MODE_SYNCHRONOUS,    // Immediate processing
    STREAM_MODE_BACKGROUND,     // Background thread
    STREAM_MODE_BATCHED        // Batch accumulation
} stream_mode_t;

typedef void (*stream_decision_callback_t)(const float* decision, float confidence, void* user_data);
```

#### Functions

**`nimcp_stream_t* nimcp_stream_create(nimcp_brain_t* brain, stream_mode_t mode, const stream_config_t* config)`**
- **Description:** Create streaming processor

**`int nimcp_stream_feed(nimcp_stream_t* stream, const float* features, uint32_t size)`**
- **Description:** Feed new input to stream
- **Returns:** 0 on success

**`int nimcp_stream_get_decision(nimcp_stream_t* stream, float* decision, float* confidence, uint32_t timeout_ms)`**
- **Description:** Get processed decision output
- **Blocking:** May block up to timeout_ms

**`void nimcp_stream_pause(nimcp_stream_t* stream)`**
- **Description:** Pause processing (for BACKGROUND mode)

**`void nimcp_stream_resume(nimcp_stream_t* stream)`**
- **Description:** Resume processing

**`void nimcp_stream_flush(nimcp_stream_t* stream)`**
- **Description:** Flush pending inputs

---

## Attention & Salience

### Salience Detection
**Header:** `nimcp_salience.h`

Attention mechanisms and salience (importance) evaluation.

#### Types
```c
typedef struct salience_evaluator_t salience_evaluator_t;

typedef enum {
    SALIENCE_MODE_FAST,      // Fast heuristic
    SALIENCE_MODE_ACCURATE   // Full computation
} salience_mode_t;

typedef struct {
    float novelty;           // How new is this input
    float surprise;          // How unexpected
    float relevance;         // How relevant to current task
    float urgency;           // How urgent to process
    float overall_salience;  // Combined score
} salience_score_t;
```

#### Functions

**`salience_evaluator_t* salience_evaluator_create(nimcp_brain_t* brain, salience_mode_t mode)`**
- **Description:** Create salience evaluator

**`void salience_evaluate(salience_evaluator_t* eval, const float* features, uint32_t size, salience_score_t* score)`**
- **Description:** Evaluate salience of input
- **Output:** Detailed salience scores

**`float salience_get_novelty(salience_evaluator_t* eval, const float* features, uint32_t size)`**
- **Description:** Get novelty score only
- **Returns:** Novelty (0.0 = familiar, 1.0 = novel)

**`float salience_get_surprise(salience_evaluator_t* eval, const float* expected, const float* actual, uint32_t size)`**
- **Description:** Measure surprise (prediction error)
- **Returns:** Surprise level

---

## Memory Consolidation

### Memory Systems
**Header:** `nimcp_consolidation.h`

Memory consolidation for moving short-term experiences to long-term storage.

#### Types
```c
typedef struct consolidation_system_t consolidation_system_t;

typedef struct {
    float* pattern;
    uint32_t size;
    float salience;
    uint64_t timestamp;
    uint32_t rehearsal_count;
} memory_trace_t;
```

#### Functions

**`consolidation_system_t* consolidation_create(nimcp_brain_t* brain, uint32_t max_traces)`**
- **Description:** Create memory consolidation system

**`void consolidation_add_trace(consolidation_system_t* sys, const float* pattern, uint32_t size, float salience)`**
- **Description:** Add short-term memory trace

**`void consolidation_consolidate(consolidation_system_t* sys)`**
- **Description:** Consolidate high-salience traces to long-term memory
- **Process:** Rehearsal, integration into network weights

**`int consolidation_recall(consolidation_system_t* sys, const float* cue, uint32_t size, float* output, float* confidence)`**
- **Description:** Recall memory from cue
- **Returns:** 1 if recalled, 0 if not found

**`uint32_t consolidation_get_trace_count(consolidation_system_t* sys)`**
- **Description:** Get number of stored traces

---

## Introspection & Monitoring

### Network Introspection
**Header:** `nimcp_introspection.h`

Runtime inspection and monitoring of network internals.

#### Functions

**`uint32_t nimcp_introspection_get_neuron_count(nimcp_neuralnet_t* net)`**
- **Description:** Get total neuron count

**`float nimcp_introspection_get_neuron_activation(nimcp_neuralnet_t* net, uint32_t neuron_id)`**
- **Description:** Get current activation of specific neuron

**`int nimcp_introspection_get_active_neurons(nimcp_neuralnet_t* net, uint32_t* neuron_ids, uint32_t max_count)`**
- **Description:** Get list of currently active neurons
- **Returns:** Number of active neurons

**`uint32_t nimcp_introspection_get_synapse_count(nimcp_neuralnet_t* net)`**
- **Description:** Get total synapse count

**`float nimcp_introspection_get_synapse_weight(nimcp_neuralnet_t* net, uint32_t from_neuron, uint32_t to_neuron)`**
- **Description:** Get weight of specific synapse

**`void nimcp_introspection_dump_state(nimcp_neuralnet_t* net, FILE* output)`**
- **Description:** Dump complete network state for debugging

---

## Higher-Level Cognitive APIs

### Curiosity-Driven Learning
**Header:** `nimcp_curiosity.h`

Intrinsic motivation and autonomous exploration.

#### Types
```c
typedef struct curiosity_engine_t curiosity_engine_t;

typedef struct {
    float exploration_rate;
    float novelty_threshold;
    float learning_progress_weight;
} curiosity_config_t;
```

#### Functions

**`curiosity_engine_t* curiosity_create(const curiosity_config_t* config)`**
- **Description:** Create curiosity-driven learning engine

**`float curiosity_get_intrinsic_reward(curiosity_engine_t* engine, const float* state, const float* next_state, float external_reward)`**
- **Description:** Compute intrinsic motivation reward
- **Returns:** Combined intrinsic + extrinsic reward

**`void curiosity_update(curiosity_engine_t* engine, const float* observation, float prediction_error)`**
- **Description:** Update curiosity model based on experience

### Knowledge Acquisition
**Header:** `nimcp_knowledge.h`

Semantic knowledge representation and retrieval.

#### Types
```c
typedef struct knowledge_base_t knowledge_base_t;

typedef struct {
    char concept[128];
    float embedding[512];
    char description[256];
} concept_t;
```

#### Functions

**`knowledge_base_t* knowledge_create(uint32_t embedding_dim)`**
- **Description:** Create knowledge base

**`int knowledge_add_concept(knowledge_base_t* kb, const concept_t* concept)`**
- **Description:** Add semantic concept

**`int knowledge_query(knowledge_base_t* kb, const float* query_embedding, concept_t* results, uint32_t max_results)`**
- **Description:** Query similar concepts
- **Returns:** Number of results

**`float knowledge_get_similarity(const concept_t* a, const concept_t* b)`**
- **Description:** Compute semantic similarity

### Ethical Reasoning
**Header:** `nimcp_ethics.h`

Ethical constraint checking and decision evaluation.

#### Types
```c
typedef struct ethics_engine_t ethics_engine_t;

typedef struct {
    bool is_ethical;
    float confidence;
    char explanation[256];
    float harm_score;
    float fairness_score;
} ethical_evaluation_t;
```

#### Functions

**`ethics_engine_t* ethics_create(const char* rules_file)`**
- **Description:** Create ethics engine with rule set

**`void ethics_evaluate_action(ethics_engine_t* engine, const float* state, const float* action, ethical_evaluation_t* result)`**
- **Description:** Evaluate if action is ethical

**`bool ethics_is_allowed(ethics_engine_t* engine, const float* action)`**
- **Description:** Quick ethical check
- **Returns:** true if allowed

---

## Utility APIs

### Queue Management
**Header:** `utils/nimcp_queue_manager.h`

Multi-channel priority queue system.

#### Types
```c
typedef struct nimcp_queue_manager_t nimcp_queue_manager_t;
typedef struct nimcp_message_t nimcp_message_t;

typedef struct {
    uint32_t num_channels;
    uint32_t max_queue_size;
    uint32_t num_priorities;
} nimcp_queue_manager_config_t;
```

#### Functions

**`nimcp_queue_manager_t* nimcp_queue_manager_create(const nimcp_queue_manager_config_t* config)`**

**`int nimcp_queue_manager_enqueue(nimcp_queue_manager_t* manager, uint32_t channel, const nimcp_message_t* message, uint32_t priority)`**

**`int nimcp_queue_manager_dequeue(nimcp_queue_manager_t* manager, uint32_t channel, nimcp_message_t* message, uint32_t priority)`**

**`bool nimcp_queue_manager_is_empty(nimcp_queue_manager_t* manager, uint32_t channel)`**

**`bool nimcp_queue_manager_is_full(nimcp_queue_manager_t* manager, uint32_t channel)`**

### Thread Pool
**Header:** `utils/nimcp_thread_pool.h`

Worker thread pool for parallel processing.

#### Functions

**`nimcp_thread_pool_t* nimcp_thread_pool_create(uint32_t num_threads)`**

**`int nimcp_thread_pool_submit(nimcp_thread_pool_t* pool, void (*task)(void*), void* arg)`**

**`void nimcp_thread_pool_wait(nimcp_thread_pool_t* pool)`**

**`void nimcp_thread_pool_destroy(nimcp_thread_pool_t* pool)`**

### Memory Tracking
**Header:** `utils/nimcp_memory.h`

Memory allocation tracking and leak detection.

#### Functions

**`void* nimcp_malloc(size_t size, const char* file, int line)`**

**`void nimcp_free(void* ptr, const char* file, int line)`**

**`void nimcp_memory_report(FILE* output)`**
- **Description:** Print memory allocation report

**`size_t nimcp_memory_get_allocated(void)`**
- **Description:** Get total allocated bytes

### Data Structures
**Header:** `utils/nimcp_hash_table.h`, `utils/nimcp_btree.h`, `utils/nimcp_graph.h`

Standard data structures with neuromorphic optimizations.

#### Hash Table

**`nimcp_hash_table_t* nimcp_hash_table_create(uint32_t capacity)`**

**`int nimcp_hash_table_insert(nimcp_hash_table_t* table, const char* key, void* value)`**

**`void* nimcp_hash_table_get(nimcp_hash_table_t* table, const char* key)`**

**`void nimcp_hash_table_remove(nimcp_hash_table_t* table, const char* key)`**

#### B-Tree

**`nimcp_btree_t* nimcp_btree_create(uint32_t order)`**

**`void nimcp_btree_insert(nimcp_btree_t* tree, uint64_t key, void* value)`**

**`void* nimcp_btree_search(nimcp_btree_t* tree, uint64_t key)`**

#### Graph

**`nimcp_graph_t* nimcp_graph_create(uint32_t num_vertices)`**

**`void nimcp_graph_add_edge(nimcp_graph_t* graph, uint32_t from, uint32_t to, float weight)`**

**`int nimcp_graph_shortest_path(nimcp_graph_t* graph, uint32_t from, uint32_t to, uint32_t* path, uint32_t* path_length)`**

---

## Python Bindings

### Python Module
**Header:** `nimcp_module.h`

Python bindings for all core functionality.

#### Example Usage

```python
import nimcp

# Create network
config = nimcp.NetworkConfig(
    num_inputs=10,
    num_outputs=5,
    num_hidden=20,
    learning_rate=0.01
)
net = nimcp.NeuralNetwork(config)

# Forward pass
inputs = [0.1, 0.2, 0.3, ...]
outputs = net.forward(inputs)

# Create brain
brain_config = nimcp.BrainConfig(
    num_inputs=10,
    num_outputs=5,
    hidden_layers=[20, 15],
    task_name="classification"
)
brain = nimcp.Brain(brain_config)

# Process data
result = brain.process(inputs)
stats = brain.get_stats()

# P2P networking
node = nimcp.P2PNode("my_node", port=8080, max_peers=10)
node.start()
node.connect("192.168.1.100", 8080)
node.broadcast(data)
```

---

## API Conventions

### Error Handling

- **NULL Returns:** Creation functions return NULL on failure
- **Integer Returns:** Functions return 0 on success, -1 on error
- **Boolean Returns:** Functions return true/false for status checks

### Thread Safety

- **Creation/Destruction:** Not thread-safe
- **Forward Pass:** Not thread-safe on same network
- **Statistics:** Thread-safe (read-only)
- **Queue Operations:** Thread-safe with internal locking

### Memory Management

- **Ownership:** Caller owns returned pointers from create functions
- **Cleanup:** Always call corresponding destroy function
- **Input Buffers:** Library does not take ownership of input arrays
- **Output Buffers:** Caller must allocate output buffers

### Naming Conventions

- **Prefix:** All public APIs prefixed with `nimcp_`
- **Structs:** `*_t` suffix for types
- **Constants:** UPPERCASE_WITH_UNDERSCORES
- **Functions:** lowercase_with_underscores

---

## Version History

### 2.5.0 (Current)
- Added Phase 3 security enhancements
- Added comprehensive stress testing
- Enhanced static analysis integration
- Added dependency vulnerability scanning
- Improved code coverage (85% target)

### 2.4.0
- Added Phase 2 fuzzing infrastructure
- Added error injection testing
- Enhanced security hardening
- Added SECURITY.md policy

### 2.3.0
- Added Phase 1 sanitizers (ASAN, UBSAN, TSAN)
- Removed unsafe string functions
- Added compiler hardening flags
- Enhanced CI security jobs

---

## Additional Resources

- **README.md:** General project overview
- **SECURITY.md:** Security policy and reporting
- **SECURITY_AUDIT.md:** Pre-release security checklist
- **BUILD_SECURITY.md:** Security build instructions
- **Examples:** See `examples/` directory for usage examples

---

**Copyright © 2025 NIMCP Project**
**License:** See LICENSE file
