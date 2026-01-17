# NIMCP External API Guide

**Version 2.6.1** | For users integrating NIMCP into their applications

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Brain API](#brain-api)
3. [Training Pipeline](#training-pipeline)
4. [Network API](#network-api)
5. [Ethics Module](#ethics-module)
6. [Knowledge Graph](#knowledge-graph)
7. [Tensor Operations](#tensor-operations)
8. [Health Agent API](#health-agent-api)
9. [Error Handling](#error-handling)
10. [Complete Examples](#complete-examples)

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
} nimcp_training_config_t;
```

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

## Network API

Low-level neural network control for advanced users.

### Creating Networks

```c
nimcp_network_t nimcp_network_create(
    const char* name,
    uint32_t num_inputs,
    uint32_t num_outputs,
    uint32_t num_hidden_layers,
    const uint32_t* hidden_sizes
);
```

### Network Operations

```c
// Forward pass
nimcp_status_t nimcp_network_forward(
    nimcp_network_t network,
    const float* inputs,
    float* outputs
);

// Backward pass (training)
nimcp_status_t nimcp_network_backward(
    nimcp_network_t network,
    const float* target,
    float learning_rate
);
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
// Get ethical score for an action
float nimcp_ethics_evaluate(
    nimcp_ethics_t ethics,
    const char* action,
    const char* context
);

// Check if action is permissible
bool nimcp_ethics_is_permissible(
    nimcp_ethics_t ethics,
    const char* action,
    float threshold          // Minimum acceptable score
);
```

### Cleanup

```c
void nimcp_ethics_destroy(nimcp_ethics_t ethics);
```

---

## Knowledge Graph

Structured knowledge representation and reasoning.

### Creating Knowledge Graphs

```c
nimcp_knowledge_t nimcp_knowledge_create(const char* name);
```

### Adding Knowledge

```c
// Add entity
nimcp_status_t nimcp_knowledge_add_entity(
    nimcp_knowledge_t kg,
    const char* entity_name,
    const char* entity_type
);

// Add relationship
nimcp_status_t nimcp_knowledge_add_relation(
    nimcp_knowledge_t kg,
    const char* subject,
    const char* predicate,
    const char* object
);
```

### Querying

```c
// Query relationships
nimcp_status_t nimcp_knowledge_query(
    nimcp_knowledge_t kg,
    const char* subject,
    const char* predicate,
    char* result,           // Buffer for result
    size_t result_size
);
```

### Cleanup

```c
void nimcp_knowledge_destroy(nimcp_knowledge_t kg);
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
const char* version = nimcp_version();  // "2.6.1"

// Get version as integer
int ver_int = nimcp_version_int();      // 20601
```

---

*Last updated: 2025-12-30*
