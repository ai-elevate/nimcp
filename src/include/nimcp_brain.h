//=============================================================================
// nimcp_brain.h - High-Level Brain API (Application-Friendly)
//=============================================================================

#ifndef NIMCP_BRAIN_H
#define NIMCP_BRAIN_H

#include "nimcp_adaptive.h"
#include "nimcp_export.h"

/**
 * @file nimcp_brain.h
 * @brief Simple, high-level API for creating lightweight learning systems
 *
 * This is the recommended API for application developers. It provides:
 * - Simple creation with sensible defaults
 * - Easy pattern learning from examples or LLMs
 * - Fast inference (<1ms for small models)
 * - Serialization for persistence
 * - Interpretability for debugging
 *
 * Example usage:
 * ```c
 * // Create a small brain for ethics decisions
 * brain_t brain = brain_create("ethics", BRAIN_SIZE_SMALL);
 *
 * // Learn from examples
 * brain_learn_example(brain, situation_features, "allow", 0.95);
 *
 * // Make decisions
 * brain_decision_t decision = brain_decide(brain, new_situation);
 * printf("Decision: %s (confidence: %.2f)\n",
 *        decision.label, decision.confidence);
 *
 * // Save for later
 * brain_save(brain, "ethics_brain.nimcp");
 * ```
 */

//=============================================================================
// Brain Sizes & Presets
//=============================================================================

/**
 * @brief Pre-configured brain sizes
 */
typedef enum {
    BRAIN_SIZE_TINY,      /**< 100 neurons, <1MB,  ~0.1ms inference */
    BRAIN_SIZE_SMALL,     /**< 1K neurons,  ~10MB, ~0.5ms inference */
    BRAIN_SIZE_MEDIUM,    /**< 10K neurons, ~50MB, ~5ms inference */
    BRAIN_SIZE_LARGE,     /**< 100K neurons, ~500MB, ~50ms inference */
    BRAIN_SIZE_CUSTOM     /**< User-defined size */
} brain_size_t;

/**
 * @brief Brain task templates
 */
typedef enum {
    BRAIN_TASK_CLASSIFICATION,   /**< Multi-class classification */
    BRAIN_TASK_REGRESSION,        /**< Continuous value prediction */
    BRAIN_TASK_PATTERN_MATCHING,  /**< Pattern recognition */
    BRAIN_TASK_SEQUENCE,          /**< Temporal sequence learning */
    BRAIN_TASK_ASSOCIATION,       /**< Association learning (Hebbian) */
    BRAIN_TASK_CUSTOM             /**< Custom task */
} brain_task_t;

//=============================================================================
// Brain Handle & Configuration
//=============================================================================

/**
 * @brief Opaque brain handle
 */
typedef struct brain_struct* brain_t;

/**
 * @brief Simple brain configuration
 */
typedef struct {
    brain_size_t size;           /**< Brain size preset */
    brain_task_t task;           /**< Task template */
    uint32_t num_inputs;         /**< Input dimension */
    uint32_t num_outputs;        /**< Output dimension */
    float learning_rate;         /**< Learning rate (0.001-0.1) */
    float sparsity_target;       /**< Target sparsity (0.7-0.95) */
    bool enable_explanations;    /**< Enable interpretability */
    char task_name[64];          /**< Name for this brain */
} brain_config_t;

/**
 * @brief Create brain with preset size and task
 *
 * @param task_name Human-readable name (e.g., "ethics", "coordination")
 * @param size Brain size preset
 * @param task Task template
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @return Brain handle or NULL on error
 */
brain_t brain_create(const char* task_name,
                     brain_size_t size,
                     brain_task_t task,
                     uint32_t num_inputs,
                     uint32_t num_outputs);

/**
 * @brief Create brain with custom configuration
 *
 * @param config Custom configuration
 * @return Brain handle or NULL on error
 */
brain_t brain_create_custom(const brain_config_t* config);

/**
 * @brief Destroy brain
 *
 * @param brain Brain to destroy
 */
void brain_destroy(brain_t brain);

//=============================================================================
// Learning API
//=============================================================================

/**
 * @brief Simple feature vector for learning
 */
typedef struct {
    float* features;             /**< Feature values */
    uint32_t num_features;       /**< Number of features */
    char label[64];              /**< Semantic label */
    float confidence;            /**< Training confidence (0-1) */
} brain_example_t;

/**
 * @brief Learn from single labeled example
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @param label Target label/output
 * @param confidence Training weight (0-1)
 * @return Loss value
 */
float brain_learn_example(brain_t brain,
                         const float* features,
                         uint32_t num_features,
                         const char* label,
                         float confidence);

/**
 * @brief Learn from batch of examples
 *
 * @param brain Brain handle
 * @param examples Array of examples
 * @param num_examples Example count
 * @return Average loss
 */
float brain_learn_batch(brain_t brain,
                       const brain_example_t* examples,
                       uint32_t num_examples);

/**
 * @brief LLM teacher function signature
 *
 * @param input Input features
 * @param num_features Feature count
 * @param context User context
 * @param output_label Output buffer for decision label
 * @param max_label_len Maximum label length
 * @return Confidence in decision (0-1)
 */
typedef float (*llm_teacher_fn_t)(
    const float* input,
    uint32_t num_features,
    void* context,
    char* output_label,
    uint32_t max_label_len
);

/**
 * @brief Learn by querying an LLM
 *
 * Allows brain to learn from any external decision maker:
 * - LLM APIs (Claude, GPT, etc.)
 * - Rule engines
 * - Human experts
 * - Other ML models
 *
 * @param brain Brain handle
 * @param input Input features
 * @param num_features Feature count
 * @param llm_fn Teacher function
 * @param llm_context Context for teacher
 * @return Loss value
 */
float brain_learn_from_llm(brain_t brain,
                          const float* input,
                          uint32_t num_features,
                          llm_teacher_fn_t llm_fn,
                          void* llm_context);

//=============================================================================
// Inference API
//=============================================================================

/**
 * @brief Decision result
 */
typedef struct {
    char label[64];              /**< Decision label */
    float confidence;            /**< Confidence (0-1) */
    float* output_vector;        /**< Raw output vector */
    uint32_t output_size;        /**< Output vector size */

    // Interpretability (if enabled)
    uint32_t num_active_neurons; /**< Active neuron count */
    uint32_t* active_neuron_ids; /**< Active neuron IDs */
    float sparsity;              /**< Actual sparsity */
    char explanation[256];       /**< Human-readable explanation */

    uint64_t inference_time_us;  /**< Inference time (microseconds) */
} brain_decision_t;

/**
 * @brief Make decision for input
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @return Decision result (caller must free with brain_free_decision)
 */
brain_decision_t* brain_decide(brain_t brain,
                              const float* features,
                              uint32_t num_features);

/**
 * @brief Free decision result
 *
 * @param decision Decision to free
 */
void brain_free_decision(brain_decision_t* decision);

/**
 * @brief Batch inference
 *
 * @param brain Brain handle
 * @param inputs Array of input vectors
 * @param num_inputs Number of inputs
 * @param features_per_input Features per input
 * @param decisions Output decisions array (allocated by caller)
 * @return true on success
 */
bool brain_decide_batch(brain_t brain,
                       const float** inputs,
                       uint32_t num_inputs,
                       uint32_t features_per_input,
                       brain_decision_t* decisions);

//=============================================================================
// Persistence API
//=============================================================================

/**
 * @brief Save brain to file
 *
 * @param brain Brain handle
 * @param filepath Path to save to
 * @return true on success
 */
bool brain_save(brain_t brain, const char* filepath);

/**
 * @brief Load brain from file
 *
 * @param filepath Path to load from
 * @return Brain handle or NULL on error
 */
brain_t brain_load(const char* filepath);

/**
 * @brief Get brain memory footprint
 *
 * @param brain Brain handle
 * @return Memory usage in bytes
 */
size_t brain_get_memory_usage(brain_t brain);

//=============================================================================
// Analysis & Monitoring API
//=============================================================================

/**
 * @brief Brain statistics
 */
typedef struct {
    char task_name[64];          /**< Brain name */
    brain_size_t size;           /**< Size preset */
    uint32_t num_neurons;        /**< Total neurons */
    uint32_t num_synapses;       /**< Total synapses */
    uint32_t num_active_synapses; /**< Non-pruned synapses */

    uint64_t total_inferences;   /**< Inference count */
    uint64_t total_learning_steps; /**< Learning steps */

    float avg_sparsity;          /**< Average sparsity */
    float avg_inference_time_us; /**< Avg inference (μs) */
    float current_learning_rate; /**< Current learning rate */

    float accuracy;              /**< Validation accuracy */
    size_t memory_bytes;         /**< Memory usage */
} brain_stats_t;

/**
 * @brief Get brain statistics
 *
 * @param brain Brain handle
 * @param stats Output statistics
 * @return true on success
 */
bool brain_get_stats(brain_t brain, brain_stats_t* stats);

/**
 * @brief Print brain info to stdout
 *
 * @param brain Brain handle
 */
void brain_print_info(brain_t brain);

/**
 * @brief Get most important neurons
 *
 * @param brain Brain handle
 * @param top_n Number of neurons to return
 * @param neuron_ids Output array of neuron IDs
 * @param importances Output array of importance scores
 * @return Number of neurons returned
 */
uint32_t brain_get_top_neurons(brain_t brain,
                               uint32_t top_n,
                               uint32_t* neuron_ids,
                               float* importances);

/**
 * @brief Explain why brain made a decision
 *
 * @param brain Brain handle
 * @param features Input that led to decision
 * @param num_features Feature count
 * @param explanation Output buffer
 * @param max_length Max explanation length
 * @return true on success
 */
bool brain_explain_decision(brain_t brain,
                           const float* features,
                           uint32_t num_features,
                           char* explanation,
                           uint32_t max_length);

//=============================================================================
// Optimization API
//=============================================================================

/**
 * @brief Prune weak connections
 *
 * @param brain Brain handle
 * @param threshold Prune synapses with weight < threshold
 * @return Number of synapses pruned
 */
uint32_t brain_prune(brain_t brain, float threshold);

/**
 * @brief Optimize brain for inference
 *
 * Performs:
 * - Aggressive pruning
 * - Quantization
 * - Sparsity optimization
 *
 * @param brain Brain handle
 * @return true on success
 */
bool brain_optimize_for_inference(brain_t brain);

/**
 * @brief Get recommended pruning threshold
 *
 * @param brain Brain handle
 * @param target_sparsity Desired sparsity (0-1)
 * @return Recommended threshold
 */
float brain_recommend_pruning_threshold(brain_t brain, float target_sparsity);

//=============================================================================
// Convenience Macros
//=============================================================================

/**
 * @brief Create default brain for classification
 */
#define BRAIN_CREATE_CLASSIFIER(name, inputs, outputs) \
    brain_create(name, BRAIN_SIZE_SMALL, BRAIN_TASK_CLASSIFICATION, inputs, outputs)

/**
 * @brief Create default brain for pattern matching
 */
#define BRAIN_CREATE_PATTERN_MATCHER(name, inputs) \
    brain_create(name, BRAIN_SIZE_SMALL, BRAIN_TASK_PATTERN_MATCHING, inputs, 1)

/**
 * @brief Create tiny brain for embedded use
 */
#define BRAIN_CREATE_TINY(name, task, inputs, outputs) \
    brain_create(name, BRAIN_SIZE_TINY, task, inputs, outputs)

//=============================================================================
// Error Handling
//=============================================================================

/**
 * @brief Get last error message
 *
 * @return Error string or NULL if no error
 */
const char* brain_get_last_error(void);

/**
 * @brief Clear last error
 */
void brain_clear_error(void);

//=============================================================================
// Internal Access API (for NIMCP 2.5 Consciousness Subsystems)
//=============================================================================

/**
 * @brief Get underlying adaptive network
 *
 * WARNING: For internal use by introspection/salience/consolidation only!
 * Direct network access bypasses brain abstraction layer.
 *
 * @param brain Brain handle
 * @return Adaptive network handle (do not free!)
 */
adaptive_network_t brain_get_network(brain_t brain);

#endif // NIMCP_BRAIN_H
