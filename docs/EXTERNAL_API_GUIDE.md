# NIMCP External API Guide

**Version 2.6.3** | For users integrating NIMCP into their applications

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Brain API](#brain-api)
3. [Training Pipeline](#training-pipeline)
4. [Training Callbacks](#training-callbacks)
5. [Dynamic Brain Resizing](#dynamic-brain-resizing)
6. [Brain Snapshots](#brain-snapshots)
7. [Copy-on-Write Cache](#copy-on-write-cache)
8. [Working Memory](#working-memory)
9. [Global Workspace](#global-workspace)
10. [Complex Oscillations](#complex-oscillations)
11. [Brain Probe](#brain-probe)
12. [Config File Loading](#config-file-loading)
13. [Network API](#network-api)
14. [Ethics Module](#ethics-module)
15. [Knowledge Graph](#knowledge-graph)
16. [Tensor Operations](#tensor-operations)
17. [Health Agent API](#health-agent-api)
18. [Error Handling](#error-handling)
19. [Python Bindings](#python-bindings)
20. [Complete Examples](#complete-examples)

---

## Quick Start

### Installation

```bash
# Build NIMCP
cd nimcp/build
cmake ..
make nimcp -j4
```

### Minimal Example

```c
#include "nimcp.h"

int main(void) {
    // Create a small classifier brain
    nimcp_brain_t brain = nimcp_brain_create(
        "classifier",           // Name
        NIMCP_BRAIN_SMALL,      // Size preset
        NIMCP_TASK_CLASSIFICATION,
        10,                     // Input dimensions
        3                       // Output classes
    );

    if (!brain) {
        printf("Failed to create brain\n");
        return 1;
    }

    // Train on examples
    float features[] = {0.5f, 0.3f, 0.8f, 0.1f, 0.9f, 0.2f, 0.7f, 0.4f, 0.6f, 0.0f};
    nimcp_brain_learn_example(brain, features, 10, "class_a", 1.0f);

    // Make prediction
    char label[64];
    float confidence;
    nimcp_brain_predict(brain, features, 10, label, &confidence);
    printf("Predicted: %s (%.2f%% confidence)\n", label, confidence * 100);

    // Cleanup
    nimcp_brain_destroy(brain);
    return 0;
}
```

---

## Brain API

The Brain API provides high-level learning and inference capabilities.

### Brain Size Presets

| Preset | Neurons | Memory | Inference Time |
|--------|---------|--------|----------------|
| `NIMCP_BRAIN_TINY` | 100 | <1MB | ~0.1ms |
| `NIMCP_BRAIN_SMALL` | 1K | ~10MB | ~0.5ms |
| `NIMCP_BRAIN_MEDIUM` | 10K | ~50MB | ~5ms |
| `NIMCP_BRAIN_LARGE` | 100K | ~500MB | ~50ms |

### Task Templates

| Template | Use Case |
|----------|----------|
| `NIMCP_TASK_CLASSIFICATION` | Multi-class classification |
| `NIMCP_TASK_REGRESSION` | Continuous value prediction |
| `NIMCP_TASK_PATTERN_MATCHING` | Pattern recognition |
| `NIMCP_TASK_SEQUENCE` | Temporal sequence learning |
| `NIMCP_TASK_ASSOCIATION` | Association learning |

### Creating a Brain

```c
nimcp_brain_t nimcp_brain_create(
    const char* name,           // Human-readable identifier
    nimcp_brain_size_t size,    // Size preset
    nimcp_brain_task_t task,    // Task template
    uint32_t num_inputs,        // Input dimension
    uint32_t num_outputs        // Output dimension
);
```

**Returns:** Brain handle, or `NULL` on error.

### Learning from Examples

```c
nimcp_status_t nimcp_brain_learn_example(
    nimcp_brain_t brain,
    const float* features,      // Input feature array
    uint32_t num_features,      // Number of features
    const char* label,          // Output label/class
    float confidence            // Example importance (0.0-1.0)
);
```

**Returns:** `NIMCP_OK` on success.

### Making Predictions

```c
nimcp_status_t nimcp_brain_predict(
    nimcp_brain_t brain,
    const float* features,      // Input features
    uint32_t num_features,      // Number of features
    char* out_label,            // Buffer for predicted label (min 64 bytes)
    float* out_confidence       // Prediction confidence (0.0-1.0)
);
```

### Raw Inference

For numeric outputs or embeddings:

```c
nimcp_status_t nimcp_brain_infer(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    float* outputs,             // Pre-allocated output array
    uint32_t num_outputs
);
```

### Persistence

```c
// Save brain state
nimcp_status_t nimcp_brain_save(nimcp_brain_t brain, const char* filepath);

// Load brain from file
nimcp_brain_t nimcp_brain_load(const char* filepath);
```

### Cleanup

```c
void nimcp_brain_destroy(nimcp_brain_t brain);
```

---

## Training Pipeline

For advanced training with custom loss functions, optimizers, and learning rate schedulers.

### Loss Functions

| Type | Description |
|------|-------------|
| `NIMCP_API_LOSS_MSE` | Mean Squared Error (regression) |
| `NIMCP_API_LOSS_CROSS_ENTROPY` | Cross-Entropy (classification) |
| `NIMCP_API_LOSS_BINARY_CE` | Binary Cross-Entropy |
| `NIMCP_API_LOSS_HUBER` | Huber Loss (robust regression) |
| `NIMCP_API_LOSS_MAE` | Mean Absolute Error |
| `NIMCP_API_LOSS_FOCAL` | Focal Loss (imbalanced classes) |
| `NIMCP_API_LOSS_KL_DIV` | KL Divergence |

### Optimizers

| Type | Description |
|------|-------------|
| `NIMCP_API_OPT_SGD` | Stochastic Gradient Descent |
| `NIMCP_API_OPT_MOMENTUM` | SGD with Momentum |
| `NIMCP_API_OPT_ADAM` | Adam optimizer |
| `NIMCP_API_OPT_ADAMW` | AdamW (weight decay) |
| `NIMCP_API_OPT_RMSPROP` | RMSprop |
| `NIMCP_API_OPT_ADAGRAD` | Adagrad |

### Learning Rate Schedulers

| Type | Description |
|------|-------------|
| `NIMCP_API_SCHED_CONSTANT` | Constant learning rate |
| `NIMCP_API_SCHED_STEP` | Step decay |
| `NIMCP_API_SCHED_EXPONENTIAL` | Exponential decay |
| `NIMCP_API_SCHED_COSINE` | Cosine annealing |
| `NIMCP_API_SCHED_WARMUP_COSINE` | Warmup + Cosine annealing |
| `NIMCP_API_SCHED_REDUCE_ON_PLATEAU` | Reduce when metric plateaus |
| `NIMCP_API_SCHED_CYCLIC` | Cyclic learning rate |

### Configuration

```c
// Get default configuration
nimcp_training_config_t config = nimcp_training_config_default();

// Customize
config.loss_type = NIMCP_API_LOSS_CROSS_ENTROPY;
config.optimizer_type = NIMCP_API_OPT_ADAM;
config.learning_rate = 0.001f;
config.weight_decay = 1e-4f;
config.enable_gradient_clipping = true;
config.gradient_clip_value = 1.0f;

// Apply to brain
nimcp_brain_configure_training(brain, &config);
```

### Training Configuration Structure

```c
typedef struct {
    nimcp_api_loss_t loss_type;
    nimcp_api_optimizer_t optimizer_type;
    nimcp_api_scheduler_t scheduler_type;

    float learning_rate;        // Initial LR (default: 0.001)
    float weight_decay;         // L2 regularization (default: 0.0)
    float momentum;             // For SGD/Momentum (default: 0.9)
    float beta1;                // Adam beta1 (default: 0.9)
    float beta2;                // Adam beta2 (default: 0.999)
    float epsilon;              // Adam epsilon (default: 1e-8)

    uint32_t scheduler_step_size;
    float scheduler_gamma;      // LR decay factor (default: 0.1)
    uint32_t warmup_steps;      // Warmup steps (default: 0)

    bool enable_gradient_clipping;
    float gradient_clip_value;  // Max gradient norm (default: 1.0)

    bool enable_biological_modulation;
    float biological_blend;     // Bio modulation strength (0-1)

    // Network type dispatch (new in 2.6.3)
    nimcp_network_type_t network_type;     // NIMCP_NETWORK_ADAPTIVE (default)

    // SNN-specific (when network_type == NIMCP_NETWORK_SNN)
    nimcp_snn_train_method_t snn_method;   // STDP, R_STDP, EPROP, SURROGATE, HOMEOSTATIC
    float snn_eligibility_tau;             // Eligibility trace decay (ms, default: 20.0)
    float snn_reward_tau;                  // Reward signal decay (ms, default: 100.0)
    float snn_surrogate_beta;              // Surrogate gradient steepness (default: 5.0)

    // LNN-specific (when network_type == NIMCP_NETWORK_LNN)
    nimcp_lnn_train_method_t lnn_method;   // ADJOINT, BPTT, RTRL, EPROP
    uint32_t lnn_bptt_truncation;          // BPTT truncation length (default: 100)
    bool lnn_use_adjoint_checkpointing;    // Memory-efficient checkpointing (default: true)
} nimcp_training_config_t;
```

### Network Types

| Type | Description |
|------|-------------|
| `NIMCP_NETWORK_ADAPTIVE` | Standard adaptive network (backprop) - default |
| `NIMCP_NETWORK_SNN` | Spiking Neural Network (STDP/eProp/surrogate) |
| `NIMCP_NETWORK_LNN` | Liquid Neural Network (adjoint ODE) |
| `NIMCP_NETWORK_CNN` | Convolutional Neural Network |
| `NIMCP_NETWORK_HYBRID` | Mixed architecture (multiple network types) |

### Training Results

```c
typedef struct {
    float loss;                 // Loss value
    float learning_rate;        // Current learning rate
    uint32_t step;              // Training step number
    bool early_stopped;         // Early stopping triggered
    float gradient_norm;        // Gradient norm (if clipping enabled)
} nimcp_training_result_t;
```

---

## Training Callbacks

Event-driven training monitoring and control.

### Callback Events

| Event | Description |
|-------|-------------|
| `NIMCP_CB_STEP_COMPLETE` | Training step finished |
| `NIMCP_CB_EPOCH_COMPLETE` | Epoch finished |
| `NIMCP_CB_LOSS_COMPUTED` | Loss calculated |
| `NIMCP_CB_WEIGHTS_UPDATED` | Weights modified |
| `NIMCP_CB_LR_CHANGED` | Learning rate changed |
| `NIMCP_CB_CONVERGENCE` | Early stopping triggered |
| `NIMCP_CB_DIVERGENCE` | Training instability |
| `NIMCP_CB_CHECKPOINT` | Checkpoint saved |

### Callback Actions

| Action | Description |
|--------|-------------|
| `NIMCP_CB_ACTION_CONTINUE` | Continue training normally |
| `NIMCP_CB_ACTION_STOP` | Stop training loop |
| `NIMCP_CB_ACTION_SKIP` | Skip current step |
| `NIMCP_CB_ACTION_ROLLBACK` | Rollback to checkpoint |
| `NIMCP_CB_ACTION_REDUCE_LR` | Reduce learning rate |
| `NIMCP_CB_ACTION_INCREASE_LR` | Increase learning rate |

### Enabling Callbacks

```c
// Get default callback configuration
nimcp_callback_config_t config = nimcp_callback_config_default();
config.enable_early_stopping = true;
config.patience = 10;

// Enable callbacks (must be called after configure_training)
nimcp_brain_enable_callbacks(brain, &config);

// Register a callback
nimcp_callback_action_t my_logger(
    nimcp_callback_event_t event,
    const nimcp_callback_metrics_t* m,
    void* user_data)
{
    printf("Step %lu: loss=%.4f lr=%.6f\n", m->step, m->loss, m->learning_rate);
    return NIMCP_CB_ACTION_CONTINUE;
}

uint32_t cb_id = nimcp_brain_register_callback(
    brain, NIMCP_CB_STEP_COMPLETE, my_logger, NULL, "logger");

// Unregister when done
nimcp_brain_unregister_callback(brain, cb_id);

// Disable all callbacks
nimcp_brain_disable_callbacks(brain);
```

### Callback Statistics

```c
uint64_t total_fired;
float avg_time_us;
uint32_t early_stops;
nimcp_brain_get_callback_stats(brain, &total_fired, &avg_time_us, &early_stops);
```

---

## Dynamic Brain Resizing

Dynamically adjust brain neuron count at runtime.

```c
// Manual resize to specific neuron count
bool success = nimcp_brain_resize(brain, 5000);

// Auto-resize based on hardware capabilities and utilization
bool resized = nimcp_brain_auto_resize(brain);

// Query current neuron count
uint32_t count = nimcp_brain_get_neuron_count(brain);

// Get utilization metrics
float utilization, saturation;
nimcp_brain_get_utilization_metrics(brain, &utilization, &saturation);
printf("Utilization: %.1f%%, Saturation: %.1f%%\n",
       utilization * 100, saturation * 100);
```

---

## Brain Snapshots

Named, timestamped backups for versioning and A/B testing.

```c
// Save named snapshots at key points
nimcp_brain_snapshot_save(brain, "before_training", "Baseline state");
// ... train ...
nimcp_brain_snapshot_save(brain, "after_epoch_1", "After 1 epoch");
// ... more training ...
nimcp_brain_snapshot_save(brain, "final", "Production model");

// Restore from a snapshot
nimcp_brain_t restored = nimcp_brain_snapshot_restore(brain, "before_training");

// List all available snapshots
nimcp_brain_snapshot_info_t infos[100];
uint32_t count;
nimcp_brain_snapshot_list(brain, infos, 100, &count);
for (uint32_t i = 0; i < count; i++) {
    printf("%s: %s (%.1f KB)\n", infos[i].name,
           infos[i].description, infos[i].file_size / 1024.0f);
}

// Delete a snapshot
nimcp_brain_snapshot_delete(brain, "after_epoch_1");
```

---

## Copy-on-Write Cache

Efficient memory sharing for brain cloning and checkpointing.

### COW Clone

```c
// Create a lightweight clone (86% memory savings)
nimcp_brain_t clone = nimcp_brain_clone_cow(original);
// clone shares memory with original - writes trigger copy
nimcp_brain_learn_example(clone, ...);  // Triggers copy on first write
```

### COW Snapshot/Restore

```c
// Instant snapshot (<1ms, ~48 bytes overhead)
nimcp_brain_snapshot_t checkpoint = nimcp_brain_snapshot_cow(brain);

// Train and evaluate...
train_epochs(brain, 100);

if (performance < threshold) {
    // Instant rollback
    nimcp_brain_restore_cow(brain, checkpoint);
}

// Clean up snapshot
nimcp_brain_snapshot_destroy(checkpoint);
```

**Performance:**
- Clone time: <10ms (vs ~1000ms for full copy)
- Memory savings: 86% for replicas, 99% for snapshots
- Restore time: <1ms (pointer swapping)

---

## Working Memory

Active representation buffer for reasoning (Miller's 7±2 capacity).

```c
// Add item to working memory with salience priority
float features[64] = {...};
nimcp_brain_working_memory_add(brain, features, 64, 0.8f);  // High salience

// Get highest-salience item
uint32_t size;
const float* item = nimcp_brain_working_memory_get(brain, 0, &size);

// Get statistics
uint32_t current_size, capacity;
nimcp_brain_working_memory_stats(brain, &current_size, &capacity);
printf("Working memory: %u/%u items\n", current_size, capacity);

// Refresh item to prevent temporal decay
nimcp_brain_working_memory_refresh(brain, 0);
```

> **Note:** Requires `enable_working_memory=true` in brain config.

---

## Global Workspace

Conscious access and cross-module information broadcasting (Global Workspace Theory).

### Cognitive Module Identifiers

| Module | Description |
|--------|-------------|
| `NIMCP_MODULE_PERCEPTION` | Sensory processing |
| `NIMCP_MODULE_WORKING_MEMORY` | Active memory |
| `NIMCP_MODULE_EXECUTIVE` | Executive control |
| `NIMCP_MODULE_ETHICS` | Ethical reasoning |
| `NIMCP_MODULE_ATTENTION` | Attention |
| `NIMCP_MODULE_EMOTION` | Emotional processing |
| `NIMCP_MODULE_SALIENCE` | Salience detection |

### Competition and Broadcasting

```c
// Compete for conscious access
float content[256] = {...};
nimcp_status_t status = nimcp_brain_workspace_compete(
    brain, NIMCP_MODULE_PERCEPTION, content, 256, 0.85f
);
if (status == NIMCP_OK) {
    printf("Content reached conscious access!\n");
}

// Read current broadcast
float broadcast[256];
uint32_t dim;
nimcp_cognitive_module_t source;
if (nimcp_brain_workspace_read(brain, broadcast, 256, &dim, &source) == NIMCP_OK) {
    printf("Broadcast from module %d, dimension %u\n", source, dim);
}

// Subscribe/unsubscribe modules
nimcp_brain_workspace_subscribe(brain, NIMCP_MODULE_EXECUTIVE);
nimcp_brain_workspace_unsubscribe(brain, NIMCP_MODULE_EXECUTIVE);

// Check broadcast status
bool has_broadcast;
nimcp_brain_workspace_has_broadcast(brain, &has_broadcast);

// Statistics
uint32_t total_broadcasts, total_competitions;
float avg_strength;
nimcp_brain_workspace_stats(brain, &total_broadcasts, &total_competitions, &avg_strength);
```

> **Note:** Requires `enable_global_workspace=true` in brain config.

---

## Complex Oscillations

Phase coding and neural synchrony with complex number support.

```c
// Enable complex oscillation features (~15% memory overhead)
nimcp_enable_complex_oscillations(brain, true);

// Check if enabled
bool enabled = nimcp_is_complex_oscillations_enabled(brain);

// Get oscillation phasor for a neuron
nimcp_oscillation_phasor_t phasor = nimcp_get_oscillation_phasor(brain, 42);
printf("Neuron 42: amplitude=%.2f, phase=%.2f rad\n",
       phasor.amplitude, phasor.phase);

// Compute phase coherence across neurons
uint32_t neurons[] = {10, 20, 30, 40, 50};
float coherence = nimcp_get_phase_coherence(brain, neurons, 5);
if (coherence > 0.6f) {
    printf("High synchronization detected!\n");
}

// Measure theta-gamma phase-amplitude coupling (PAC)
float pac = nimcp_get_pac_modulation(brain, 6.0f, 40.0f);
if (pac > 0.4f) {
    printf("Strong theta-gamma coupling (memory encoding active)\n");
}
```

**Phase Coherence Scale:**

| Value | Meaning |
|-------|---------|
| 0.0 | Random phases (no synchronization) |
| 0.3 | Weak synchronization |
| 0.6 | Moderate synchronization |
| 1.0 | Perfect phase locking |

---

## Brain Probe

Comprehensive brain state snapshot for monitoring and debugging.

```c
nimcp_brain_probe_t probe;
nimcp_brain_probe(brain, &probe);

printf("Brain: %s (%u neurons, %u synapses)\n",
       probe.task_name, probe.num_neurons, probe.num_synapses);
printf("Inferences: %lu, Learning steps: %lu\n",
       probe.total_inferences, probe.total_learning_steps);
printf("Accuracy: %.1f%%, Memory: %.1f MB\n",
       probe.accuracy * 100, probe.memory_bytes / (1024.0 * 1024.0));

// COW statistics (if this is a clone)
if (probe.is_cow_clone) {
    printf("COW: %u refs, %.1f KB shared, %.1f KB private\n",
           probe.cow_ref_count,
           probe.cow_shared_bytes / 1024.0,
           probe.cow_private_bytes / 1024.0);
}

// Broadcast probe data via bio-async message system
nimcp_brain_broadcast_probe(brain);
```

---

## Config File Loading

Create brains from YAML or JSON configuration files.

```c
nimcp_brain_t brain = nimcp_brain_create_from_config("model.yaml");
```

**Example YAML config:**

```yaml
brain:
  name: "classifier"
  size: small           # tiny, small, medium, large
  task: classification  # classification, regression, pattern_matching, sequence, association
  architecture:
    num_inputs: 784
    num_outputs: 10
    num_hidden: 256
    learning_rate: 0.01
```

---

## Network API

Low-level neural network control for advanced users.

### Creating Networks

```c
nimcp_network_t nimcp_network_create(
    uint32_t num_inputs,        // Number of input neurons
    uint32_t num_outputs,       // Number of output neurons
    uint32_t num_hidden,        // Number of hidden neurons
    float learning_rate         // Learning rate (typically 0.001 - 0.1)
);
```

**Returns:** Network handle, or `NULL` on error.

### Network Operations

```c
// Forward pass
nimcp_status_t nimcp_network_forward(
    nimcp_network_t network,
    const float* inputs,
    uint32_t num_inputs,
    float* outputs,             // Pre-allocated output array
    uint32_t num_outputs
);

// Train on a single example
nimcp_status_t nimcp_network_train(
    nimcp_network_t network,
    const float* inputs,
    uint32_t num_inputs,
    const float* targets,
    uint32_t num_targets
);
```

### Cleanup

```c
void nimcp_network_destroy(nimcp_network_t network);
```

---

## Ethics Module

Built-in ethical decision-making support.

### Creating Ethics Module

```c
nimcp_ethics_t nimcp_ethics_create(void);
```

### Evaluating Actions

```c
// Check if an action is ethically acceptable
nimcp_status_t nimcp_ethics_check(
    nimcp_ethics_t ethics,
    const float* situation,     // Situation features array
    uint32_t num_features,      // Number of features
    float* out_score            // Score: -1.0=harmful, 0.0=neutral, 1.0=beneficial
);
```

**Returns:** `NIMCP_OK` on success.

### Cleanup

```c
void nimcp_ethics_destroy(nimcp_ethics_t ethics);
```

---

## Knowledge Graph

Structured knowledge representation and reasoning.

### Creating Knowledge Graphs

```c
nimcp_knowledge_t nimcp_knowledge_create(void);
```

### Adding Knowledge

```c
// Add a fact (subject-predicate-object triple)
nimcp_status_t nimcp_knowledge_add_fact(
    nimcp_knowledge_t knowledge,
    const char* subject,        // Subject entity
    const char* predicate,      // Relationship type
    const char* object          // Object entity
);
```

### Querying

```c
// Query the knowledge graph
nimcp_status_t nimcp_knowledge_query(
    nimcp_knowledge_t knowledge,
    const char* query,          // Query string
    char* out_result,           // Result buffer (pre-allocated, min 1024 bytes)
    uint32_t max_result_len     // Maximum result length
);
```

### Cleanup

```c
void nimcp_knowledge_destroy(nimcp_knowledge_t knowledge);
```

---

## Tensor Operations

Efficient tensor computations with SIMD acceleration.

### Including Tensor Header

```c
#include "utils/tensor/nimcp_tensor.h"
```

### Creating Tensors

```c
// Create tensor with shape
uint32_t dims[] = {3, 4};  // 3x4 matrix
nimcp_tensor_t* tensor = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);

// Create from data
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
nimcp_tensor_t* from_data = nimcp_tensor_from_array(data, dims, 2, NIMCP_DTYPE_F32);

// Special tensors
nimcp_tensor_t* zeros = nimcp_tensor_zeros(dims, 2, NIMCP_DTYPE_F32);
nimcp_tensor_t* ones = nimcp_tensor_ones(dims, 2, NIMCP_DTYPE_F32);
nimcp_tensor_t* eye = nimcp_tensor_eye(4, NIMCP_DTYPE_F32);  // 4x4 identity
```

### Element Access

```c
// Get/set by indices
uint32_t indices[] = {1, 2};
float val = nimcp_tensor_get(tensor, indices);
nimcp_tensor_set(tensor, indices, 3.14f);

// Flat indexing
float val = nimcp_tensor_get_flat(tensor, 5);
nimcp_tensor_set_flat(tensor, 5, 2.71f);
```

### Mathematical Operations

```c
// Element-wise operations (return new tensors)
nimcp_tensor_t* sum = nimcp_tensor_add(a, b);
nimcp_tensor_t* diff = nimcp_tensor_sub(a, b);
nimcp_tensor_t* prod = nimcp_tensor_mul(a, b);
nimcp_tensor_t* quot = nimcp_tensor_div(a, b);

// In-place operations (modify first tensor)
nimcp_tensor_add_(a, b);   // a += b
nimcp_tensor_sub_(a, b);   // a -= b
nimcp_tensor_mul_(a, b);   // a *= b
nimcp_tensor_mul_scalar_(a, 2.0);  // a *= 2.0

// Reductions (return scalar tensors)
nimcp_tensor_t* total = nimcp_tensor_sum(t);
nimcp_tensor_t* mean = nimcp_tensor_mean(t);
nimcp_tensor_t* max_val = nimcp_tensor_max(t);
nimcp_tensor_t* min_val = nimcp_tensor_min(t);

// Extract scalar from reduction result
double scalar = nimcp_tensor_get_flat(total, 0);
```

### Linear Algebra

```c
// Matrix multiplication
nimcp_tensor_t* c = nimcp_tensor_matmul(a, b);

// Dot product
nimcp_tensor_t* dot = nimcp_tensor_dot(a, b);

// Transpose
nimcp_tensor_t* t = nimcp_tensor_transpose(a, 0, 1);

// Norms
double fro = nimcp_tensor_norm_fro(t);    // Frobenius norm
double lp = nimcp_tensor_norm_p(t, 2.0);  // L-p norm
```

### Activation Functions

```c
nimcp_tensor_t* relu_out = nimcp_tensor_relu(t);
nimcp_tensor_t* sig_out = nimcp_tensor_sigmoid(t);
nimcp_tensor_t* tanh_out = nimcp_tensor_tanh(t);
nimcp_tensor_t* softmax_out = nimcp_tensor_softmax(t, -1);  // Along last dim
```

### Cleanup

```c
nimcp_tensor_destroy(tensor);
```

---

## Health Agent API

The Health Agent provides autonomous health monitoring with integration to cognitive modules (Portia, Dragonfly, Swarm, Memory systems).

### Include Header

```c
#include "utils/fault_tolerance/nimcp_health_agent.h"
```

### Creating a Health Agent

```c
// Get default configuration
health_agent_config_t config;
nimcp_health_agent_default_config(&config);

// Customize if needed
strncpy(config.agent_name, "my_health_agent", sizeof(config.agent_name));
config.heartbeat_interval_ms = 100;
config.watchdog_timeout_ms = 500;

// Create agent
nimcp_health_agent_t* agent = nimcp_health_agent_create(&config);
if (!agent) {
    printf("Failed to create health agent\n");
    return -1;
}
```

### Starting and Stopping

```c
// Start the agent (spawns monitoring thread)
nimcp_health_agent_start(agent);

// Send heartbeats regularly from main code
while (running) {
    nimcp_health_agent_heartbeat(agent);
    // ... your main loop ...
}

// Stop when done
nimcp_health_agent_stop(agent);
nimcp_health_agent_destroy(agent);
```

### Portia USE Functions (Platform Resource Control)

```c
// Set platform tier (0=full, 1=high, 2=medium, 3=low, 4=minimal)
nimcp_health_agent_use_portia_set_tier(agent, 2);

// Set degradation level (0=none, 1=minor, 2=moderate, 3=severe, 4=emergency)
nimcp_health_agent_use_portia_degrade(agent, 1);

// Get recommended neuron count based on resources
uint32_t neurons;
nimcp_health_agent_use_portia_get_recommended_neurons(agent, &neurons);

// Get full Portia status
uint32_t power, thermal, degradation;
nimcp_health_agent_use_portia_get_status(agent, &power, &thermal, &degradation);
```

### Dragonfly USE Functions (Predictive Anomaly Tracking)

```c
// Track an anomaly with Dragonfly
health_agent_message_t msg = {0};
msg.type = HEALTH_MSG_MEMORY_CORRUPTION;
msg.severity = HEALTH_SEVERITY_CRITICAL;
msg.source = HEALTH_SOURCE_MEMORY;

uint32_t target_id;
nimcp_health_agent_use_dragonfly_track_anomaly(agent, &msg, &target_id);

// Get prediction for the anomaly
float time_to_failure, confidence;
nimcp_health_agent_use_dragonfly_predict(agent, target_id, &time_to_failure, &confidence);
printf("Predicted failure in %.1f seconds (%.0f%% confidence)\n",
       time_to_failure, confidence * 100);

// Start pursuing the anomaly
nimcp_health_agent_use_dragonfly_pursue(agent);

// Get current Dragonfly mode (0=idle, 1=scanning, 2=tracking, 3=pursuing, 4=intercepting)
uint32_t mode;
nimcp_health_agent_use_dragonfly_get_mode(agent, &mode);
```

### Swarm Immune USE Functions (Distributed Threat Detection)

```c
// Detect threats in data
const uint8_t* data = /* suspicious data */;
size_t data_len = /* length */;
bool detected;
uint32_t threat_id;

nimcp_health_agent_use_swarm_detect_threat(agent, data, data_len, source_id, &detected, &threat_id);

if (detected) {
    // Generate coordinated response
    uint32_t response_id;
    nimcp_health_agent_use_swarm_generate_response(agent, threat_id, &response_id);
}

// Check component behavior for anomalies
float anomaly_score;
nimcp_health_agent_use_swarm_check_behavior(agent, component_id, &anomaly_score);
```

### Swarm Memory USE Functions (Distributed Pattern Storage)

```c
// Store a health pattern
const void* pattern_data = /* pattern */;
size_t pattern_size = /* size */;
char pattern_id[65];

nimcp_health_agent_use_swarm_memory_store(
    agent, pattern_data, pattern_size,
    0,  // pattern_type: 0=episodic, 1=semantic, 2=procedural, 3=threat, 4=spatial
    2,  // importance: 0-3
    pattern_id
);

// Trigger memory replay for consolidation
int replayed = nimcp_health_agent_use_swarm_memory_replay(agent, 10);

// Trigger memory consolidation
nimcp_health_agent_use_swarm_memory_consolidate(agent);

// Get memory statistics
uint64_t total, consolidated;
float avg_strength;
nimcp_health_agent_use_swarm_memory_get_stats(agent, &total, &consolidated, &avg_strength);
```

### Engram USE Functions (Memory Encoding/Recall)

```c
// Encode a health event as an engram
health_agent_message_t event = {0};
event.type = HEALTH_MSG_ANOMALY_DETECTED;
event.severity = HEALTH_SEVERITY_WARNING;
event.source = HEALTH_SOURCE_NEURAL;

uint64_t engram_id;
nimcp_health_agent_use_engram_encode(agent, &event, &engram_id);

// Recall similar past events
uint64_t recalled_ids[10];
uint32_t num_recalled;
nimcp_health_agent_use_engram_recall(agent, &event, recalled_ids, 10, &num_recalled);

// Get engram statistics
uint32_t active, consolidated;
float strength;
nimcp_health_agent_use_engram_get_stats(agent, &active, &consolidated, &strength);
```

### Recovery Actions

```c
// Trigger garbage collection
nimcp_health_agent_trigger_gc(agent, false);  // true to force

// Create a checkpoint
nimcp_health_agent_create_checkpoint(agent, "pre_experiment");

// Rollback to checkpoint (0 for latest)
nimcp_health_agent_rollback(agent, 0);

// Reduce system load
nimcp_health_agent_reduce_load(agent, 0.5f);  // 50% reduction

// Restore normal load
nimcp_health_agent_restore_load(agent);
```

### Getting Statistics

```c
health_agent_stats_t stats;
nimcp_health_agent_get_stats(agent, &stats);

printf("Uptime: %llu ms\n", stats.uptime_ms);
printf("Checks performed: %llu\n", stats.checks_performed);
printf("Anomalies detected: %llu\n", stats.anomalies_detected);
printf("Recoveries: %llu/%llu succeeded\n",
       stats.recoveries_succeeded, stats.recoveries_triggered);
```

### Python Bindings

```python
import nimcp

# Create health agent
agent = nimcp.HealthAgent(name="py_agent", heartbeat_ms=100)
agent.start()

# Portia integration
agent.use_portia_set_tier(2)
power, thermal, degrade = agent.use_portia_get_status()
neurons = agent.use_portia_get_recommended_neurons()

# Dragonfly integration
target_id = agent.use_dragonfly_track_anomaly(
    nimcp.HEALTH_MSG_MEMORY_CORRUPTION,
    nimcp.HEALTH_SEVERITY_CRITICAL,
    nimcp.HEALTH_SOURCE_MEMORY,
    "Memory corruption detected"
)
ttf, conf = agent.use_dragonfly_predict(target_id)

# Swarm memory
pattern_id = agent.use_swarm_memory_store(b"pattern", 0, 2)
total, consolidated, strength = agent.use_swarm_memory_get_stats()

# Recovery
agent.trigger_gc(force=False)
agent.create_checkpoint("my_checkpoint")
agent.rollback(0)  # 0 = latest

# Statistics
stats = agent.get_stats()
print(f"Anomalies: {stats.anomalies_detected}")

agent.stop()
```

### Message Types

| Constant | Value | Description |
|----------|-------|-------------|
| `HEALTH_MSG_ANOMALY_DETECTED` | 0 | Generic anomaly |
| `HEALTH_MSG_CYTOKINE_SIGNAL` | 1 | Inflammatory signal |
| `HEALTH_MSG_EMERGENCY` | 2 | Emergency condition |
| `HEALTH_MSG_RECOVERY_REQUEST` | 3 | Recovery requested |
| `HEALTH_MSG_STATE_CORRUPTION` | 4 | State corrupted |
| `HEALTH_MSG_HEARTBEAT_TIMEOUT` | 5 | System may be hung |
| `HEALTH_MSG_DEADLOCK_DETECTED` | 6 | Deadlock found |
| `HEALTH_MSG_NAN_DETECTED` | 7 | NaN in computations |
| `HEALTH_MSG_MEMORY_CORRUPTION` | 8 | Memory corrupted |
| `HEALTH_MSG_RESOURCE_EXHAUSTION` | 9 | Resources running low |

### Severity Levels

| Level | Value | Description |
|-------|-------|-------------|
| `HEALTH_SEVERITY_INFO` | 0 | Informational only |
| `HEALTH_SEVERITY_WARNING` | 1 | Potential issue |
| `HEALTH_SEVERITY_ERROR` | 2 | Definite problem, recoverable |
| `HEALTH_SEVERITY_CRITICAL` | 3 | Severe, may need rollback |
| `HEALTH_SEVERITY_FATAL` | 4 | System integrity compromised |

### Anomaly Sources

| Source | Value | Description |
|--------|-------|-------------|
| `HEALTH_SOURCE_MEMORY` | 1 | Memory subsystem |
| `HEALTH_SOURCE_THREADING` | 2 | Thread/lock subsystem |
| `HEALTH_SOURCE_NEURAL` | 3 | Neural computation |
| `HEALTH_SOURCE_KG` | 4 | Knowledge graph |
| `HEALTH_SOURCE_IMMUNE` | 5 | Immune system |
| `HEALTH_SOURCE_IO` | 6 | I/O operations |
| `HEALTH_SOURCE_BRAIN_REGION` | 7 | Brain region |
| `HEALTH_SOURCE_CHECKPOINT` | 8 | Checkpoint system |
| `HEALTH_SOURCE_HEARTBEAT` | 9 | Heartbeat monitor |

---

## Error Handling

### Status Codes

| Code | Meaning |
|------|---------|
| `NIMCP_OK` | Success |
| `NIMCP_ERROR` | Generic error |
| `NIMCP_ERROR_NULL_ARG` | NULL argument provided |
| `NIMCP_ERROR_INVALID` | Invalid argument value |
| `NIMCP_ERROR_MEMORY` | Memory allocation failed |
| `NIMCP_ERROR_IO` | I/O operation failed |

### Error Checking Pattern

```c
nimcp_status_t status = nimcp_brain_learn_example(brain, features, n, label, conf);
if (status != NIMCP_OK) {
    printf("Error: %d\n", status);
    // Handle error...
}
```

---

## Python Bindings

NIMCP provides comprehensive Python bindings via the `nimcp` module.

### Installation

```bash
cd nimcp/build
cmake ..
make nimcp_python -j4
# Module is built as nimcp.so in build/src/python/
```

### Module Functions

```python
import nimcp

# Library lifecycle
nimcp.init()                    # Initialize NIMCP library
print(nimcp.version())          # "2.6.3"
print(nimcp.version_int())      # 20603
print(nimcp.get_error())        # Last error message
nimcp.shutdown()                # Cleanup
```

### Constants

```python
# Brain sizes
nimcp.BRAIN_TINY                # 100 neurons
nimcp.BRAIN_SMALL               # 1K neurons
nimcp.BRAIN_MEDIUM              # 10K neurons
nimcp.BRAIN_LARGE               # 100K neurons

# Task types
nimcp.TASK_CLASSIFICATION
nimcp.TASK_REGRESSION
nimcp.TASK_PATTERN_MATCHING
nimcp.TASK_SEQUENCE
nimcp.TASK_ASSOCIATION

# Network types
nimcp.NETWORK_ADAPTIVE          # Standard backprop
nimcp.NETWORK_SNN               # Spiking neural network
nimcp.NETWORK_LNN               # Liquid neural network
nimcp.NETWORK_CNN               # Convolutional
nimcp.NETWORK_HYBRID            # Mixed architecture

# Status codes
nimcp.OK
nimcp.ERROR
nimcp.ERROR_NULL_ARG
nimcp.ERROR_INVALID
nimcp.ERROR_MEMORY
nimcp.ERROR_IO
```

### Brain Creation and Inference

```python
# Create a brain
brain = nimcp.Brain("classifier",
                    size=nimcp.BRAIN_SMALL,
                    task=nimcp.TASK_CLASSIFICATION,
                    inputs=10, outputs=3)

# Learn from examples
brain.learn([0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.0],
            "class_a", confidence=1.0)

# Predict
label, confidence = brain.decide([0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.0])
print(f"Predicted: {label} ({confidence:.1%})")

# Raw inference (numeric outputs)
outputs = brain.infer([0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 0.0], 3)
print(f"Raw outputs: {outputs}")

# Save/load
brain.save("model.nimcp")
loaded = nimcp.Brain.load("model.nimcp")

# Create from config file
brain = nimcp.Brain.create_from_config("model.yaml")

# Load pre-trained model
brain = nimcp.Brain.from_pretrained("nimcp_foundation_medium_v1.0")
```

### Training Pipeline

```python
# Configure training
config = nimcp.TrainingConfig.default()
config.loss_type = nimcp.LOSS_CROSS_ENTROPY
config.optimizer_type = nimcp.OPT_ADAM
config.learning_rate = 0.001
brain.configure_training(config)

# Single training step
features = [0.1] * 784
targets = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # One-hot class 3
result = brain.train_step(features, targets)
print(f"Loss: {result.loss:.4f}, LR: {result.learning_rate:.6f}")

# Batch training
batch_features = [[0.1] * 784 for _ in range(32)]
batch_targets = [[0] * 10 for _ in range(32)]
result = brain.train_batch(batch_features, batch_targets)

# Training statistics
steps, loss, lr = brain.get_training_stats()
new_lr = brain.step_scheduler(validation_accuracy)
```

### Dynamic Resizing

```python
print(f"Neurons: {brain.get_neuron_count()}")
brain.resize(5000)
brain.auto_resize()
util, sat = brain.get_utilization_metrics()
print(f"Utilization: {util:.1%}, Saturation: {sat:.1%}")
```

### Snapshots and COW

```python
# Named snapshots
brain.snapshot_save("baseline", "Before training")
brain.snapshot_save("epoch_10", "After 10 epochs")
for snap in brain.snapshot_list():
    print(f"{snap['name']}: {snap['description']}")
restored = brain.snapshot_restore("baseline")
brain.snapshot_delete("epoch_10")

# Copy-on-write cloning
clone = brain.clone_cow()  # 86% memory savings
result = clone.decide(features)  # Inference on clone
```

### Working Memory and Global Workspace

```python
# Working memory
brain.working_memory_add([0.1, 0.2, 0.3], 0.8)
data, size = brain.working_memory_get(0)
current, capacity = brain.working_memory_stats()
brain.working_memory_refresh(0)

# Global workspace
won = brain.workspace_compete(nimcp.MODULE_PERCEPTION, content, 0.85)
content, dim, source = brain.workspace_read()
brain.workspace_subscribe(nimcp.MODULE_EXECUTIVE)
has_broadcast = brain.workspace_has_broadcast()
broadcasts, competitions, strength = brain.workspace_stats()
```

### Complex Oscillations

```python
brain.enable_complex_oscillations(True)
amp, phase = brain.get_oscillation_phasor(42)
coherence = brain.get_phase_coherence([10, 20, 30, 40, 50])
pac = brain.get_pac_modulation(6.0, 40.0)
```

### Brain Probe

```python
stats = brain.probe()
print(f"Neurons: {stats['num_neurons']}")
print(f"Accuracy: {stats['accuracy']:.1%}")
print(f"Memory: {stats['memory_bytes'] / 1024 / 1024:.1f} MB")
if stats['is_cow_clone']:
    print(f"COW shared: {stats['cow_shared_bytes']} bytes")
```

---

## Complete Examples

### Example 1: Image Classifier

```c
#include "nimcp.h"
#include <stdio.h>

#define IMAGE_SIZE 784  // 28x28 flattened
#define NUM_CLASSES 10

int main(void) {
    // Create classifier brain
    nimcp_brain_t brain = nimcp_brain_create(
        "mnist_classifier",
        NIMCP_BRAIN_MEDIUM,
        NIMCP_TASK_CLASSIFICATION,
        IMAGE_SIZE,
        NUM_CLASSES
    );

    if (!brain) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }

    // Configure training
    nimcp_training_config_t config = nimcp_training_config_default();
    config.loss_type = NIMCP_API_LOSS_CROSS_ENTROPY;
    config.optimizer_type = NIMCP_API_OPT_ADAM;
    config.learning_rate = 0.001f;
    config.scheduler_type = NIMCP_API_SCHED_COSINE;
    nimcp_brain_configure_training(brain, &config);

    // Training loop (pseudo-code)
    // for each (image, label) in training_data:
    //     nimcp_brain_learn_example(brain, image, IMAGE_SIZE, label, 1.0f);

    // Inference
    float test_image[IMAGE_SIZE] = { /* ... */ };
    char predicted_class[64];
    float confidence;

    nimcp_brain_predict(brain, test_image, IMAGE_SIZE, predicted_class, &confidence);
    printf("Predicted digit: %s (%.1f%% confidence)\n",
           predicted_class, confidence * 100);

    // Save trained model
    nimcp_brain_save(brain, "mnist_model.nimcp");

    nimcp_brain_destroy(brain);
    return 0;
}
```

### Example 2: Regression Model

```c
#include "nimcp.h"

int main(void) {
    // Create regression brain
    nimcp_brain_t brain = nimcp_brain_create(
        "price_predictor",
        NIMCP_BRAIN_SMALL,
        NIMCP_TASK_REGRESSION,
        5,   // 5 input features
        1    // 1 output (price)
    );

    // Configure for regression
    nimcp_training_config_t config = nimcp_training_config_default();
    config.loss_type = NIMCP_API_LOSS_MSE;
    config.optimizer_type = NIMCP_API_OPT_ADAMW;
    config.learning_rate = 0.01f;
    nimcp_brain_configure_training(brain, &config);

    // Training (examples)
    float features1[] = {1500.0f, 3.0f, 2.0f, 1985.0f, 0.25f};
    nimcp_brain_learn_example(brain, features1, 5, "350000", 1.0f);

    // Predict
    float new_house[] = {2000.0f, 4.0f, 2.5f, 2010.0f, 0.5f};
    float price[1];
    nimcp_brain_infer(brain, new_house, 5, price, 1);
    printf("Predicted price: $%.0f\n", price[0]);

    nimcp_brain_destroy(brain);
    return 0;
}
```

### Example 3: Pattern Recognition with Tensor Ops

```c
#include "nimcp.h"
#include "utils/tensor/nimcp_tensor.h"

int main(void) {
    // Create tensors
    uint32_t dims[] = {3, 3};
    nimcp_tensor_t* pattern = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);
    nimcp_tensor_t* input = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);

    // Set pattern values
    float pattern_data[] = {1, 0, 1, 0, 1, 0, 1, 0, 1};  // Checkerboard
    for (int i = 0; i < 9; i++) {
        nimcp_tensor_set_flat(pattern, i, pattern_data[i]);
    }

    // Compute similarity via dot product
    nimcp_tensor_t* similarity = nimcp_tensor_dot(pattern, input);
    double score = nimcp_tensor_get_flat(similarity, 0);

    // Normalize by norms
    double pattern_norm = nimcp_tensor_norm_fro(pattern);
    double input_norm = nimcp_tensor_norm_fro(input);
    if (input_norm > 0) {
        double cosine_sim = score / (pattern_norm * input_norm);
        printf("Pattern similarity: %.2f\n", cosine_sim);
    }

    // Cleanup
    nimcp_tensor_destroy(pattern);
    nimcp_tensor_destroy(input);
    nimcp_tensor_destroy(similarity);

    return 0;
}
```

---

## Performance Tips

1. **Use appropriate brain size** - Start with `NIMCP_BRAIN_SMALL` and scale up only if needed
2. **Batch training** - Use `nimcp_brain_train_batch()` for large datasets
3. **SIMD acceleration** - Tensor operations automatically use AVX2/AVX-512/NEON when available
4. **Memory management** - Always destroy brains and tensors when done
5. **Persistence** - Save trained models to avoid retraining

---

## Version Information

```c
// Get version string
const char* version = nimcp_version();  // "2.6.3"

// Get version as integer
int ver_int = nimcp_version_int();      // 20603

// Initialize / shutdown
nimcp_init();
// ... use NIMCP ...
nimcp_shutdown();

// Get last error
const char* err = nimcp_get_error();
```

---

*Last updated: 2026-02-16*
