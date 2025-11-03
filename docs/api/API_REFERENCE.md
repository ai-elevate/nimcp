# NIMCP API Reference

**Version:** 2.5.1
**Last Updated:** 2025-11-03

This document provides comprehensive reference documentation for all public APIs in the NIMCP (Neuromorphic Infant Machine Cognitive Platform) library.

## Recent Updates (2025-11-03)

- **Added:** Knowledge B-tree indexing for efficient confidence-based queries (O(log n) range queries)
- **Added:** `knowledge_get_by_confidence_range()` - Query knowledge items by confidence level
- **Added:** `knowledge_get_all_ordered_by_confidence()` - Get all knowledge sorted by confidence
- **Added:** `knowledge_add_item()` - Test helper API for direct knowledge insertion
- **Fixed:** B-tree key extraction pattern using stable stored keys instead of thread-local buffers
- **Status:** 600+ tests passing; 2/3 knowledge B-tree tests passing (1 performance issue under investigation)

## Table of Contents

1. [Core Neural Network APIs](#core-neural-network-apis)
2. [Brain & Cognitive Systems](#brain--cognitive-systems)
3. [Learning Systems](#learning-systems)
   - [Adaptive Learning](#adaptive-learning)
   - [Neuromodulator System](#neuromodulator-system)
   - [BCM Learning Rule](#bcm-learning-rule)
4. [Event Processing](#event-processing)
5. [P2P Networking](#p2p-networking)
6. [Data I/O & Streaming](#data-io--streaming)
7. [Attention & Salience](#attention--salience)
8. [Memory Consolidation](#memory-consolidation)
9. [Introspection & Monitoring](#introspection--monitoring)
10. [Higher-Level Cognitive APIs](#higher-level-cognitive-apis)
11. [Thread Safety & Synchronization](#thread-safety--synchronization)
12. [Utility APIs](#utility-apis)
13. [Language Bindings](#language-bindings)
    - [Python Bindings](#python-bindings)
    - [C++ Bindings](#c-bindings)
    - [Java Bindings](#java-bindings)
    - [Rust Bindings](#rust-bindings)
    - [Go Bindings](#go-bindings)
    - [Perl Bindings](#perl-bindings)
    - [C# Bindings](#c-bindings-1)

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

### Neuromodulator System
**Header:** `nimcp_neuromodulators.h`

Implements biologically-inspired neuromodulator systems (dopamine, serotonin, acetylcholine, norepinephrine) with thread-safe concurrent access.

#### Types
```c
typedef struct neuromodulator_system_t neuromodulator_system_t;

typedef enum {
    NEUROMOD_DOPAMINE = 0,       // Reward, motivation, learning
    NEUROMOD_SEROTONIN,          // Mood regulation, patience
    NEUROMOD_ACETYLCHOLINE,      // Attention, arousal
    NEUROMOD_NOREPINEPHRINE,     // Alertness, stress response
    NEUROMOD_COUNT               // Total count
} neuromodulator_type_t;

typedef struct {
    float learning_rate_multiplier;
    float exploration_bias;
    float attention_focus;
    float memory_consolidation;
} modulation_effects_t;

typedef struct {
    float concentrations[NEUROMOD_COUNT];
    float moving_averages[NEUROMOD_COUNT];
    uint64_t release_counts[NEUROMOD_COUNT];
    uint64_t update_count;
} neuromodulator_stats_t;
```

#### Functions

**`neuromodulator_system_t neuromodulator_system_create(void)`**
- **Description:** Create thread-safe neuromodulator system
- **Thread Safety:** Thread-safe creation
- **Returns:** System handle

**`void neuromodulator_system_destroy(neuromodulator_system_t system)`**
- **Description:** Destroy neuromodulator system and release resources
- **Thread Safety:** Must not be called concurrently with other operations

**`void neuromodulator_release(neuromodulator_system_t system, neuromodulator_type_t type, float amount)`**
- **Description:** Release neuromodulator (thread-safe)
- **Parameters:**
  - `type`: Which neuromodulator to release
  - `amount`: Amount to release (0.0-1.0)
- **Thread Safety:** Thread-safe with reader-writer lock
- **Performance:** ~100ns write lock overhead

**`float neuromodulator_get_level(neuromodulator_system_t system, neuromodulator_type_t type)`**
- **Description:** Get current concentration level
- **Returns:** Concentration (0.0-1.0)
- **Thread Safety:** Thread-safe read operation (~50ns overhead)
- **Use Case:** Query before making learning decisions

**`modulation_effects_t neuromodulator_get_effects(neuromodulator_system_t system)`**
- **Description:** Get current modulation effects on learning
- **Returns:** Combined effects of all neuromodulators
- **Thread Safety:** Thread-safe (uses thread-local buffer, 0ns contention)
- **Performance:** Lock-free access to pre-computed effects

**`bool neuromodulator_update(neuromodulator_system_t system, float dt)`**
- **Description:** Update neuromodulator concentrations over time
- **Parameters:**
  - `dt`: Time delta in seconds
- **Thread Safety:** Thread-safe with write lock
- **Returns:** true on success

**`void neuromodulator_get_stats(neuromodulator_system_t system, neuromodulator_stats_t* stats)`**
- **Description:** Get statistics (thread-safe)
- **Output:** Concentrations, averages, release counts
- **Thread Safety:** Uses atomic counters for lock-free increments

#### Design Patterns Used
- **Monitor Pattern:** Reader-writer lock protecting shared state
- **Thread-Local Storage Pattern:** Zero-contention effect buffers
- **Atomic Operations:** Lock-free statistics counters

---

### BCM Learning Rule
**Header:** `nimcp_bcm.h`

Bienenstock-Cooper-Munro learning rule with sliding threshold and thread-safe synapse updates.

#### Types
```c
typedef struct {
    float weight;                    // Synapse weight
    float threshold;                 // BCM sliding threshold
    float avg_post_activity;         // Average postsynaptic activity
    float eligibility;               // Eligibility trace for delayed learning
    nimcp_spinlock_t lock;          // Spinlock for thread-safe updates
} bcm_synapse_t;

typedef struct {
    float learning_rate;             // Base learning rate
    float threshold_tau;             // Time constant for threshold adaptation
    float trace_decay;               // Eligibility trace decay
    float min_weight;               // Weight bounds
    float max_weight;
} bcm_params_t;
```

#### Functions

**`bcm_synapse_t bcm_synapse_init(float initial_weight, float initial_threshold)`**
- **Description:** Initialize BCM synapse with spinlock
- **Parameters:**
  - `initial_weight`: Starting weight value
  - `initial_threshold`: Starting threshold
- **Thread Safety:** Spinlock initialized for concurrent access
- **Returns:** Initialized synapse structure

**`void bcm_apply_rule(bcm_synapse_t* synapse, float pre_activity, float post_activity, float dt, const bcm_params_t* params)`**
- **Description:** Apply BCM learning rule (thread-safe)
- **Algorithm:** Δw = η × pre × post × (post - θ)
- **Parameters:**
  - `synapse`: Synapse to update (locked during update)
  - `pre_activity`: Presynaptic activity
  - `post_activity`: Postsynaptic activity
  - `dt`: Time step
  - `params`: Learning parameters
- **Thread Safety:** Spinlock-protected (~10-20ns overhead)
- **Performance:** Optimized for <100 cycle critical sections

**`void bcm_update_threshold(bcm_synapse_t* synapse, float post_activity, float dt, const bcm_params_t* params)`**
- **Description:** Update BCM sliding threshold
- **Formula:** θ(t) = E[post²(t)]
- **Thread Safety:** Spinlock-protected

**`void bcm_update_eligibility(bcm_synapse_t* synapse, float pre_activity, float post_activity, float dt, const bcm_params_t* params)`**
- **Description:** Update eligibility trace for three-factor learning
- **Use Case:** Enables dopamine-modulated BCM learning
- **Thread Safety:** Spinlock-protected

**`void bcm_apply_modulation(bcm_synapse_t* synapse, float modulation, const bcm_params_t* params)`**
- **Description:** Apply neuromodulator to eligibility trace
- **Parameters:**
  - `modulation`: Neuromodulator signal (typically dopamine)
- **Algorithm:** Δw = modulation × eligibility
- **Thread Safety:** Spinlock-protected

**`void bcm_synapse_destroy(bcm_synapse_t* synapse)`**
- **Description:** Destroy spinlock and cleanup
- **Thread Safety:** Must not be called while synapse is in use

#### Design Patterns Used
- **Factory Pattern:** bcm_synapse_init() constructs with all invariants
- **Monitor Pattern:** Spinlock protects mutable state
- **Template Method Pattern:** Modular update functions

#### Performance Characteristics
- **Spinlock Acquisition:** ~10-20ns (busy-wait)
- **BCM Update:** ~50-100ns total (including lock)
- **Best For:** Brief critical sections (<100 CPU cycles)
- **Not For:** Long-running computations (use rwlock instead)

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

Multi-domain knowledge acquisition system with B-tree indexed queries for efficient retrieval.

#### Types
```c
typedef struct knowledge_system_struct* knowledge_system_t;

typedef enum {
    KNOWLEDGE_DOMAIN_LANGUAGE,
    KNOWLEDGE_DOMAIN_LITERATURE,
    KNOWLEDGE_DOMAIN_ART,
    KNOWLEDGE_DOMAIN_ETHICS,
    KNOWLEDGE_DOMAIN_HISTORY,
    KNOWLEDGE_DOMAIN_SCIENCE,
    KNOWLEDGE_DOMAIN_MATHEMATICS,
    KNOWLEDGE_DOMAIN_SOCIAL,
    KNOWLEDGE_DOMAIN_TECHNICAL,
    KNOWLEDGE_DOMAIN_PHILOSOPHY,
    KNOWLEDGE_DOMAIN_GENERAL
} knowledge_domain_t;

typedef struct {
    char concept[256];
    knowledge_domain_t domain;
    char definition[1024];
    char context[512];
    char** examples;
    uint32_t num_examples;
    char** related_concepts;
    uint32_t num_related;
    float confidence;              // How well understood (0-1)
    uint64_t learned_timestamp;
    uint32_t reinforcement_count;
    char confidence_key[16];       // B-tree key field
} knowledge_item_t;

typedef struct {
    knowledge_domain_t domain;
    uint32_t concepts_known;
    uint32_t estimated_total;
    float coverage_percentage;
    float avg_confidence;
    char gaps[5][256];
    uint32_t num_gaps;
} domain_knowledge_t;
```

#### System Management

**`knowledge_system_t knowledge_system_create(const char* learner_name)`**
- **Description:** Create knowledge system
- **Parameters:** `learner_name` - Name for the learner
- **Returns:** Knowledge system handle or NULL on error
- **Thread Safety:** Thread-safe

**`void knowledge_system_destroy(knowledge_system_t system)`**
- **Description:** Destroy knowledge system and free resources
- **Thread Safety:** Must not be called concurrently with same system

#### Learning Functions

**`uint32_t knowledge_learn_from_text(knowledge_system_t system, const char* text, knowledge_domain_t domain)`**
- **Description:** Learn from text incrementally
- **Returns:** Number of concepts learned

**`bool knowledge_retrieve(knowledge_system_t system, const char* concept, knowledge_item_t* item)`**
- **Description:** Retrieve knowledge about a concept
- **Returns:** true if found

**`bool knowledge_reinforce(knowledge_system_t system, const char* concept, const char* new_example)`**
- **Description:** Strengthen understanding through repetition
- **Returns:** true on success

#### B-Tree Indexed Queries (New in v2.5.1)

**`uint32_t knowledge_get_by_confidence_range(knowledge_system_t system, float min_confidence, float max_confidence, knowledge_item_t** results_out)`**
- **Description:** Query knowledge items within confidence range using B-tree
- **Parameters:**
  - `system` - Knowledge system handle
  - `min_confidence` - Minimum confidence threshold (0.0-1.0)
  - `max_confidence` - Maximum confidence threshold (0.0-1.0)
  - `results_out` - Output array (caller must free with nimcp_free)
- **Returns:** Number of items in range
- **Complexity:** O(log n + k) where k = results in range
- **Thread Safety:** Thread-safe
- **Use Cases:**
  - Find well-understood concepts: `(0.8, 1.0)`
  - Find weak knowledge needing reinforcement: `(0.0, 0.4)`
  - Find moderately confident items: `(0.4, 0.7)`

**`uint32_t knowledge_get_all_ordered_by_confidence(knowledge_system_t system, knowledge_item_t** results_out)`**
- **Description:** Get all knowledge items sorted by confidence (low to high)
- **Parameters:**
  - `system` - Knowledge system handle
  - `results_out` - Output array (caller must free with nimcp_free)
- **Returns:** Number of items
- **Complexity:** O(n) via B-tree in-order traversal
- **Thread Safety:** Thread-safe
- **Use Cases:** Review knowledge progression, identify learning gaps

#### Testing API

**`bool knowledge_add_item(knowledge_system_t system, const knowledge_item_t* item)`**
- **Description:** Add knowledge item directly (for testing)
- **Parameters:**
  - `system` - Knowledge system handle
  - `item` - Knowledge item to add
- **Returns:** true on success
- **Note:** Only available when NIMCP_TESTING is defined

#### Assessment

**`bool knowledge_assess_domain(knowledge_system_t system, knowledge_domain_t domain, domain_knowledge_t* assessment)`**
- **Description:** Assess knowledge coverage in a domain
- **Returns:** true on success

**`uint32_t knowledge_get_summary(knowledge_system_t system, domain_knowledge_t* all_domains, uint32_t max_domains)`**
- **Description:** Get overall knowledge summary across all domains
- **Returns:** Number of domains assessed

#### Persistence

**`bool knowledge_save(knowledge_system_t system, const char* filepath)`**
- **Description:** Save knowledge to file (persistent memory)
- **Returns:** true on success

**`knowledge_system_t knowledge_load(const char* filepath)`**
- **Description:** Load knowledge from file
- **Returns:** Knowledge system or NULL on error

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

## Thread Safety & Synchronization

### Thread Utilities
**Header:** `utils/nimcp_thread.h`

POSIX thread wrappers with NIMCP conventions and comprehensive error handling.

#### Types
```c
typedef pthread_mutex_t nimcp_mutex_t;
typedef pthread_cond_t nimcp_cond_t;
typedef pthread_rwlock_t nimcp_rwlock_t;        // Reader-writer lock
typedef pthread_spinlock_t nimcp_spinlock_t;    // Spinlock
typedef pthread_t nimcp_thread_t;
```

#### Reader-Writer Lock API

**WHAT:** Multiple concurrent readers OR single exclusive writer
**WHY:** Enables parallel read access to shared data (10x faster than mutex for read-heavy workloads)
**WHEN TO USE:** Read-heavy data structures, configuration, statistics

**`nimcp_result_t nimcp_rwlock_init(nimcp_rwlock_t* rwlock)`**
- **Description:** Initialize reader-writer lock
- **Returns:** NIMCP_SUCCESS or error code
- **Thread Safety:** Not thread-safe (call before sharing)

**`nimcp_result_t nimcp_rwlock_rdlock(nimcp_rwlock_t* rwlock)`**
- **Description:** Acquire read lock (multiple readers allowed)
- **Performance:** ~50-100ns (no blocking with other readers!)
- **Use Case:** Reading configuration, querying statistics
- **Blocking:** Blocks only if writer holds lock

**`nimcp_result_t nimcp_rwlock_wrlock(nimcp_rwlock_t* rwlock)`**
- **Description:** Acquire write lock (exclusive access)
- **Performance:** ~50-100ns + wait for readers to finish
- **Use Case:** Updating shared state, releasing neuromodulators
- **Blocking:** Blocks until all readers and writers release

**`nimcp_result_t nimcp_rwlock_unlock(nimcp_rwlock_t* rwlock)`**
- **Description:** Release read or write lock
- **Performance:** ~50-100ns
- **Important:** MUST be called after rdlock or wrlock

**`nimcp_result_t nimcp_rwlock_destroy(nimcp_rwlock_t* rwlock)`**
- **Description:** Destroy rwlock
- **Thread Safety:** Must not be in use

#### Spinlock API

**WHAT:** Busy-wait lock for very brief critical sections
**WHY:** Faster than mutex for <100 cycle critical sections (~10-20ns overhead)
**WHEN TO USE:** Protecting individual synapse updates, brief counter increments
**WHEN NOT TO USE:** Long-running operations, I/O, memory allocation

**`nimcp_result_t nimcp_spinlock_init(nimcp_spinlock_t* spinlock)`**
- **Description:** Initialize spinlock
- **Returns:** NIMCP_SUCCESS or error code

**`nimcp_result_t nimcp_spinlock_lock(nimcp_spinlock_t* spinlock)`**
- **Description:** Acquire spinlock (busy-wait)
- **Performance:** ~10-20ns if uncontended
- **Behavior:** Spins (burns CPU) until lock acquired
- **Best For:** <100 CPU cycles of work

**`nimcp_result_t nimcp_spinlock_unlock(nimcp_spinlock_t* spinlock)`**
- **Description:** Release spinlock
- **Performance:** ~5-10ns

**`nimcp_result_t nimcp_spinlock_destroy(nimcp_spinlock_t* spinlock)`**
- **Description:** Destroy spinlock

#### Atomic Operations

**WHAT:** Lock-free atomic operations for counters and flags
**WHY:** ~50x faster than mutex-protected increments (~5ns vs ~250ns)
**WHEN TO USE:** Statistics counters, reference counts, flags

```c
#include <stdatomic.h>

// Atomic types
typedef atomic_uint_fast64_t nimcp_atomic_counter_t;
typedef atomic_bool nimcp_atomic_flag_t;

// Atomic operations
uint64_t atomic_fetch_add(&counter, 1);           // Atomic increment
bool atomic_load(&flag);                          // Atomic read
atomic_store(&flag, true);                        // Atomic write
```

#### Thread-Local Storage

**WHAT:** Per-thread private storage (zero contention)
**WHY:** Eliminates lock contention for thread-private data
**WHEN TO USE:** Per-thread buffers, temporary computation results

```c
// Declaration
_Thread_local modulation_effects_t thread_effect_buffer;

// Usage (zero synchronization overhead!)
thread_effect_buffer.learning_rate = 1.5f;
```

#### Design Patterns for Thread Safety

**Monitor Pattern:**
```c
struct protected_data {
    nimcp_rwlock_t rwlock;
    float shared_value;
};

// Read operation (parallel with other readers)
nimcp_rwlock_rdlock(&data->rwlock);
float value = data->shared_value;
nimcp_rwlock_unlock(&data->rwlock);

// Write operation (exclusive)
nimcp_rwlock_wrlock(&data->rwlock);
data->shared_value = new_value;
nimcp_rwlock_unlock(&data->rwlock);
```

**Spinlock for Brief Updates:**
```c
bcm_synapse_t synapse;
nimcp_spinlock_init(&synapse.lock);

// Brief critical section (<100 cycles)
nimcp_spinlock_lock(&synapse.lock);
synapse.weight += delta;
nimcp_spinlock_unlock(&synapse.lock);
```

**Atomic Counters for Statistics:**
```c
atomic_uint_fast64_t update_count = 0;

// Lock-free increment (5ns, zero contention)
atomic_fetch_add(&update_count, 1);
```

#### Performance Comparison

| Operation | Latency | Parallel? | Best For |
|-----------|---------|-----------|----------|
| **RWLock Read** | ~50-100ns | ✅ Multiple readers | Read-heavy data |
| **RWLock Write** | ~50-100ns | ❌ Exclusive | Infrequent updates |
| **Spinlock** | ~10-20ns | ❌ Exclusive | <100 cycle sections |
| **Mutex** | ~50-100ns | ❌ Exclusive | General purpose |
| **Atomic Increment** | ~5ns | ✅ Lock-free | Counters, flags |
| **Thread-Local** | 0ns | ✅ No sync needed | Per-thread data |

#### Thread Safety Guarantees

**Thread-Safe Operations:**
- ✅ Neuromodulator release/query (rwlock)
- ✅ BCM synapse updates (spinlock)
- ✅ Statistics increments (atomics)
- ✅ Queue operations (internal locking)

**Not Thread-Safe:**
- ❌ Network creation/destruction
- ❌ Forward pass on same network
- ❌ Parameter updates

**Best Practices:**
1. Use RWLock for read-heavy data structures
2. Use Spinlock for <100 cycle critical sections
3. Use Atomics for counters and flags
4. Use Thread-Local for per-thread buffers
5. Minimize critical section duration
6. Never hold lock while doing I/O or allocation

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

## Language Bindings

NIMCP provides bindings for 7 languages, each following language-specific conventions while maintaining API consistency.

### Python Bindings

**Directory:** `src/python/`
**Module:** `nimcp`
**Build:** `python setup.py install`

#### Example Usage

```python
import nimcp

# Create neural network
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

# Create brain with neuromodulators
brain_config = nimcp.BrainConfig(
    num_inputs=10,
    num_outputs=5,
    hidden_layers=[20, 15],
    task_name="classification"
)
brain = nimcp.Brain(brain_config)

# Process with neuromodulation
result = brain.process(inputs)
brain.release_dopamine(0.8)  # Reward signal
stats = brain.get_stats()

# BCM learning
synapse = nimcp.BCMSynapse(initial_weight=0.5)
params = nimcp.BCMParams(learning_rate=0.01)
synapse.apply_rule(pre=0.8, post=0.9, dt=0.001, params=params)

# P2P networking
node = nimcp.P2PNode("my_node", port=8080, max_peers=10)
node.start()
node.connect("192.168.1.100", 8080)
node.broadcast(data)
```

#### Threading

Python bindings release GIL for compute-intensive operations:
- `forward()` - Releases GIL
- `update()` - Releases GIL
- `consolidate()` - Releases GIL

---

### C++ Bindings

**Directory:** `src/bindings/cpp/`
**Namespace:** `nimcp`
**Build:** `g++ -std=c++17 -lnimcp_core -lnimcp_cpp`

C++ bindings provide RAII wrappers, exceptions, and modern C++ idioms.

#### Example Usage

```cpp
#include <nimcp/neural_network.hpp>
#include <nimcp/brain.hpp>
#include <nimcp/neuromodulators.hpp>
#include <nimcp/bcm.hpp>

using namespace nimcp;

// RAII neural network
auto config = NetworkConfig{
    .num_inputs = 10,
    .num_outputs = 5,
    .num_hidden = 20,
    .learning_rate = 0.01f
};
NeuralNetwork net(config);  // Automatic cleanup via destructor

// Forward pass with std::vector
std::vector<float> inputs = {0.1f, 0.2f, 0.3f, ...};
auto outputs = net.forward(inputs);  // Returns std::vector<float>

// Neuromodulator system (thread-safe)
NeuromodulatorSystem neuro_sys;
neuro_sys.release(NeuromodulatorType::Dopamine, 0.8f);
auto effects = neuro_sys.get_effects();

// BCM synapse with RAII
BCMSynapse synapse(0.5f, 1.0f);  // weight, threshold
BCMParams params{
    .learning_rate = 0.01f,
    .threshold_tau = 1000.0f
};
synapse.apply_rule(0.8f, 0.9f, 0.001f, params);

// Exception handling
try {
    auto brain = Brain::create(brain_config);
    brain->process(inputs);
} catch (const NIMCPException& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}

// Smart pointers
std::unique_ptr<Brain> brain = Brain::create(config);
std::shared_ptr<P2PNode> node = std::make_shared<P2PNode>("node1", 8080);
```

#### Features
- ✅ RAII resource management
- ✅ Exception-based error handling
- ✅ STL containers (std::vector, std::string)
- ✅ Smart pointers (unique_ptr, shared_ptr)
- ✅ Move semantics
- ✅ Template methods

---

### Java Bindings

**Directory:** `src/bindings/java/`
**Package:** `com.nimcp`
**Build:** `mvn package`

JNI-based bindings with Java conventions and automatic resource management.

#### Example Usage

```java
import com.nimcp.*;

// Neural network with try-with-resources
try (NeuralNetwork net = new NeuralNetwork(config)) {
    float[] inputs = {0.1f, 0.2f, 0.3f};
    float[] outputs = net.forward(inputs);

    // Network statistics
    NetworkStats stats = net.getStats();
    System.out.println("Activity: " + stats.getAverageActivity());
}

// Brain with neuromodulators
BrainConfig config = new BrainConfig.Builder()
    .setNumInputs(10)
    .setNumOutputs(5)
    .setTaskName("classification")
    .build();

try (Brain brain = Brain.create(config)) {
    float[] result = brain.process(inputs);
    brain.releaseDopamine(0.8f);

    // Get neuromodulator stats
    NeuromodulatorStats stats = brain.getNeuromodulatorStats();
}

// BCM Learning
BCMSynapse synapse = new BCMSynapse(0.5f, 1.0f);
BCMParams params = new BCMParams.Builder()
    .setLearningRate(0.01f)
    .setThresholdTau(1000.0f)
    .build();

synapse.applyRule(0.8f, 0.9f, 0.001f, params);

// P2P networking
try (P2PNode node = new P2PNode("node1", 8080)) {
    node.start();
    node.connect("192.168.1.100", 8080);
    node.broadcast(data);

    List<PeerInfo> peers = node.getPeers();
}

// Exception handling
try {
    brain.process(inputs);
} catch (NIMCPException e) {
    e.printStackTrace();
}
```

#### Features
- ✅ Builder pattern for configuration
- ✅ try-with-resources (AutoCloseable)
- ✅ Java exceptions
- ✅ JNI performance optimization
- ✅ Thread-safe operations

---

### Rust Bindings

**Directory:** `src/bindings/rust/`
**Crate:** `nimcp`
**Build:** `cargo build --release`

Idiomatic Rust bindings with safe FFI, ownership, and zero-cost abstractions.

#### Example Usage

```rust
use nimcp::{NeuralNetwork, Brain, NeuromodulatorSystem, BCMSynapse};
use nimcp::config::{NetworkConfig, BrainConfig, BCMParams};

// Neural network with ownership
let config = NetworkConfig {
    num_inputs: 10,
    num_outputs: 5,
    num_hidden: 20,
    learning_rate: 0.01,
    ..Default::default()
};

let mut net = NeuralNetwork::create(&config)?;  // Returns Result<NeuralNetwork, NIMCPError>

// Forward pass with slices
let inputs = vec![0.1, 0.2, 0.3];
let outputs = net.forward(&inputs)?;

// Neuromodulator system (thread-safe)
let mut neuro_sys = NeuromodulatorSystem::create();
neuro_sys.release(NeuromodulatorType::Dopamine, 0.8);
let effects = neuro_sys.get_effects();

// BCM synapse
let mut synapse = BCMSynapse::new(0.5, 1.0);
let params = BCMParams {
    learning_rate: 0.01,
    threshold_tau: 1000.0,
    ..Default::default()
};

synapse.apply_rule(0.8, 0.9, 0.001, &params);

// Brain with Result handling
let config = BrainConfig::builder()
    .num_inputs(10)
    .num_outputs(5)
    .task_name("classification")
    .build();

let mut brain = Brain::create(&config)?;
let result = brain.process(&inputs)?;

// Error handling with ?
fn run_network() -> Result<(), NIMCPError> {
    let mut net = NeuralNetwork::create(&config)?;
    let outputs = net.forward(&inputs)?;
    Ok(())
}

// Thread-safe operations with Arc<Mutex<T>>
use std::sync::{Arc, Mutex};
use std::thread;

let neuro_sys = Arc::new(Mutex::new(NeuromodulatorSystem::create()));
let neuro_clone = Arc::clone(&neuro_sys);

thread::spawn(move || {
    let mut sys = neuro_clone.lock().unwrap();
    sys.release(NeuromodulatorType::Dopamine, 0.5);
});
```

#### Features
- ✅ Ownership and borrowing
- ✅ Result<T, E> error handling
- ✅ Zero-cost abstractions
- ✅ Safe FFI with bindgen
- ✅ Thread safety with Arc/Mutex
- ✅ Traits (Default, Clone, Debug)

---

### Go Bindings

**Directory:** `src/bindings/go/`
**Package:** `github.com/nimcp/nimcp-go`
**Build:** `go build`

CGO-based bindings with Go conventions, goroutine safety, and channels.

#### Example Usage

```go
package main

import (
    "github.com/nimcp/nimcp-go"
    "fmt"
    "sync"
)

func main() {
    // Neural network with defer cleanup
    config := &nimcp.NetworkConfig{
        NumInputs:    10,
        NumOutputs:   5,
        NumHidden:    20,
        LearningRate: 0.01,
    }

    net, err := nimcp.NewNeuralNetwork(config)
    if err != nil {
        panic(err)
    }
    defer net.Destroy()

    // Forward pass
    inputs := []float32{0.1, 0.2, 0.3}
    outputs, err := net.Forward(inputs)
    if err != nil {
        panic(err)
    }

    // Neuromodulator system (goroutine-safe)
    neuroSys := nimcp.NewNeuromodulatorSystem()
    defer neuroSys.Destroy()

    neuroSys.Release(nimcp.Dopamine, 0.8)
    effects := neuroSys.GetEffects()

    // BCM synapse
    synapse := nimcp.NewBCMSynapse(0.5, 1.0)
    defer synapse.Destroy()

    params := &nimcp.BCMParams{
        LearningRate:  0.01,
        ThresholdTau: 1000.0,
    }

    synapse.ApplyRule(0.8, 0.9, 0.001, params)

    // Brain with error handling
    brainConfig := &nimcp.BrainConfig{
        NumInputs:  10,
        NumOutputs: 5,
        TaskName:   "classification",
    }

    brain, err := nimcp.NewBrain(brainConfig)
    if err != nil {
        panic(err)
    }
    defer brain.Destroy()

    result, err := brain.Process(inputs)
    brain.ReleaseDopamine(0.8)

    // Concurrent learning with goroutines
    var wg sync.WaitGroup
    neuroSys := nimcp.NewNeuromodulatorSystem()

    for i := 0; i < 4; i++ {
        wg.Add(1)
        go func(agent int) {
            defer wg.Done()
            neuroSys.Release(nimcp.Dopamine, 0.5)
            fmt.Printf("Agent %d released dopamine\n", agent)
        }(i)
    }

    wg.Wait()
}
```

#### Features
- ✅ defer for cleanup
- ✅ Error handling with (value, error)
- ✅ Goroutine-safe operations
- ✅ Channel-based async
- ✅ Go naming conventions
- ✅ Context support

---

### Perl Bindings

**Directory:** `src/bindings/perl/`
**Module:** `NIMCP`
**Build:** `perl Makefile.PL && make`

XS-based bindings with Perl conventions and CPAN compatibility.

#### Example Usage

```perl
use NIMCP;
use strict;
use warnings;

# Neural network
my $config = NIMCP::NetworkConfig->new(
    num_inputs    => 10,
    num_outputs   => 5,
    num_hidden    => 20,
    learning_rate => 0.01
);

my $net = NIMCP::NeuralNetwork->new($config);

# Forward pass
my @inputs = (0.1, 0.2, 0.3);
my @outputs = $net->forward(\@inputs);

# Neuromodulator system
my $neuro_sys = NIMCP::NeuromodulatorSystem->new();
$neuro_sys->release('dopamine', 0.8);
my $effects = $neuro_sys->get_effects();

print "Learning rate multiplier: ", $effects->{learning_rate_multiplier}, "\n";

# BCM synapse
my $synapse = NIMCP::BCMSynapse->new(0.5, 1.0);  # weight, threshold
my $params = {
    learning_rate  => 0.01,
    threshold_tau => 1000.0
};

$synapse->apply_rule(0.8, 0.9, 0.001, $params);

# Brain
my $brain_config = NIMCP::BrainConfig->new(
    num_inputs  => 10,
    num_outputs => 5,
    task_name   => 'classification'
);

my $brain = NIMCP::Brain->new($brain_config);
my $result = $brain->process(\@inputs);
$brain->release_dopamine(0.8);

# Error handling
eval {
    $brain->process(\@inputs);
};
if ($@) {
    die "NIMCP error: $@";
}

# Multithreading with threads
use threads;
use threads::shared;

my $neuro_sys = NIMCP::NeuromodulatorSystem->new();

my @threads;
for my $i (1..4) {
    push @threads, threads->create(sub {
        $neuro_sys->release('dopamine', 0.5);
        print "Thread $i released dopamine\n";
    });
}

$_->join() for @threads;
```

#### Features
- ✅ XS for performance
- ✅ Perl OO conventions
- ✅ Hash refs for config
- ✅ eval for exceptions
- ✅ threads support
- ✅ CPAN compatible

---

### C# Bindings

**Directory:** `src/bindings/csharp/`
**Namespace:** `NIMCP`
**Build:** `dotnet build`

P/Invoke bindings with .NET conventions, IDisposable, and async/await support.

#### Example Usage

```csharp
using NIMCP;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;

// Neural network with using statement
var config = new NetworkConfig
{
    NumInputs = 10,
    NumOutputs = 5,
    NumHidden = 20,
    LearningRate = 0.01f
};

using (var net = new NeuralNetwork(config))
{
    float[] inputs = { 0.1f, 0.2f, 0.3f };
    float[] outputs = net.Forward(inputs);

    // Get stats
    var stats = net.GetStats();
    Console.WriteLine($"Activity: {stats.AverageActivity}");
}

// Neuromodulator system (thread-safe)
using (var neuroSys = new NeuromodulatorSystem())
{
    neuroSys.Release(NeuromodulatorType.Dopamine, 0.8f);
    var effects = neuroSys.GetEffects();

    Console.WriteLine($"Learning rate: {effects.LearningRateMultiplier}");
}

// BCM synapse
using (var synapse = new BCMSynapse(0.5f, 1.0f))
{
    var params = new BCMParams
    {
        LearningRate = 0.01f,
        ThresholdTau = 1000.0f
    };

    synapse.ApplyRule(0.8f, 0.9f, 0.001f, params);
}

// Brain with LINQ-style API
var brainConfig = new BrainConfig
{
    NumInputs = 10,
    NumOutputs = 5,
    TaskName = "classification"
};

using (var brain = Brain.Create(brainConfig))
{
    float[] result = brain.Process(inputs);
    brain.ReleaseDopamine(0.8f);

    // Get neuromodulator stats
    var neuroStats = brain.GetNeuromodulatorStats();
}

// Exception handling
try
{
    brain.Process(inputs);
}
catch (NIMCPException ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}

// Async operations
public async Task<float[]> ProcessAsync(Brain brain, float[] inputs)
{
    return await Task.Run(() => brain.Process(inputs));
}

// Parallel learning with Tasks
var neuroSys = new NeuromodulatorSystem();
var tasks = new List<Task>();

for (int i = 0; i < 4; i++)
{
    int agent = i;
    tasks.Add(Task.Run(() =>
    {
        neuroSys.Release(NeuromodulatorType.Dopamine, 0.5f);
        Console.WriteLine($"Agent {agent} released dopamine");
    }));
}

await Task.WhenAll(tasks);

// Event-driven updates
brain.OnLearningComplete += (sender, e) =>
{
    Console.WriteLine($"Learning complete: {e.Accuracy}");
};
```

#### Features
- ✅ IDisposable pattern
- ✅ Properties and indexers
- ✅ async/await support
- ✅ Event-driven API
- ✅ LINQ compatibility
- ✅ Exception handling
- ✅ .NET Standard 2.0

---

## Language Binding Comparison

| Feature | Python | C++ | Java | Rust | Go | Perl | C# |
|---------|--------|-----|------|------|----|----|-----|
| **Memory Mgmt** | GC | RAII | GC | Ownership | GC | RC | GC |
| **Thread Safety** | GIL | Manual | JVM | Arc/Mutex | Goroutines | threads | Task |
| **Error Handling** | Exception | Exception | Exception | Result<T,E> | (val, err) | eval | Exception |
| **Performance** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Build System** | pip/setup.py | CMake | Maven/Gradle | Cargo | go build | CPAN | dotnet |
| **Best For** | ML/AI | HPC | Enterprise | Systems | Cloud | Scripting | Enterprise |

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
- **Thread Safety:** Added comprehensive thread safety with rwlocks, spinlocks, and atomics
- **Neuromodulator System:** Biologically-inspired neuromodulators with concurrent access
- **BCM Learning:** Bienenstock-Cooper-Munro learning rule with thread-safe updates
- **Language Bindings:** Added bindings for C++, Java, Rust, Go, Perl, and C#
- **Performance Optimizations:**
  - RWLock: ~10x faster than mutex for read-heavy workloads
  - Spinlock: ~10-20ns overhead for brief critical sections
  - Atomics: ~50x faster than mutex for counters
  - Thread-Local Storage: Zero-contention effect buffers
- **Security Enhancements:**
  - Phase 3 security hardening complete
  - Comprehensive stress testing
  - Enhanced static analysis integration
  - Dependency vulnerability scanning
- **Code Quality:**
  - 85%+ code coverage
  - TDD implementation for all new features
  - Explicit WHAT/WHY comments
  - Design patterns (Monitor, Thread-Local Storage, Factory)

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
