//=============================================================================
// nimcp_adaptive.h - Adaptive Threshold Spiking (SpikingBrain-inspired)
//=============================================================================

#ifndef NIMCP_ADAPTIVE_H
#define NIMCP_ADAPTIVE_H

#include <stdbool.h>
#include <stdint.h>
#include "common/nimcp_export.h"
#include "core/neuralnet/nimcp_neuralnet.h"

/**
 * @file nimcp_adaptive.h
 * @brief Adaptive threshold spiking neurons inspired by SpikingBrain
 *
 * Implements efficient "pseudo-spiking" using:
 * - Dynamic thresholds based on input statistics
 * - High sparsity (70-90% neurons inactive)
 * - Integer spike counts (not temporal trains)
 * - GPU-friendly computation
 * - Optional bitwise spike encoding
 *
 * This is NOT biologically realistic spiking, but computationally efficient
 * pattern learning suitable for practical applications.
 */

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Adaptive Threshold Configuration
//=============================================================================

/**
 * @brief Spike encoding schemes
 */
typedef enum {
    SPIKE_ENCODING_INTEGER, /**< Direct integer counts (fastest) */
    SPIKE_ENCODING_BINARY,  /**< Binary {0,1} expansion */
    SPIKE_ENCODING_TERNARY, /**< Ternary {-1,0,1} with E/I */
    SPIKE_ENCODING_BITWISE  /**< Bitwise binary encoding (most sparse) */
} spike_encoding_t;

/**
 * @brief Adaptive spiking parameters
 */
typedef struct {
    float k_factor;             /**< Firing rate control (default: 0.5) */
    float sparsity_target;      /**< Target sparsity 0-1 (default: 0.7) */
    spike_encoding_t encoding;  /**< Spike encoding scheme */
    bool enable_soft_reset;     /**< Use soft reset dynamics */
    bool enable_adaptation;     /**< Enable threshold adaptation */
    uint32_t adaptation_window; /**< Window for statistics (timesteps) */
    float min_threshold;        /**< Minimum threshold value */
    float max_threshold;        /**< Maximum threshold value */
} adaptive_spike_params_t;

/**
 * @brief Adaptive neuron state
 */
typedef struct {
    float membrane_potential;    /**< Current potential */
    float adaptive_threshold;    /**< Dynamic threshold */
    int32_t spike_count;         /**< Integer spike count */
    uint32_t* spike_train;       /**< Optional sparse spike train */
    uint32_t spike_train_length; /**< Length of spike train */

    // Statistics for adaptation
    float activation_mean;     /**< Running mean of inputs */
    float activation_variance; /**< Running variance */
    uint32_t sample_count;     /**< Number of samples seen */
} adaptive_neuron_state_t;

//=============================================================================
// Adaptive Neural Network
//=============================================================================

/**
 * @brief Adaptive neural network configuration
 */
typedef struct {
    network_config_t base_config;         /**< Base network configuration */
    adaptive_spike_params_t spike_params; /**< Adaptive spiking parameters */
    bool enable_sparsity;                 /**< Enable sparse activation */
    float pruning_threshold;              /**< Prune weights below this */
    uint32_t update_frequency;            /**< How often to adapt thresholds */

    // Checkpoint/persistence configuration
    const char* checkpoint_path;          /**< Path to checkpoint file (NULL = no checkpoint) */
    bool auto_load;                       /**< Auto-load from checkpoint on create (default: true) */
    bool auto_save;                       /**< Auto-save to checkpoint periodically (default: false) */
    uint32_t auto_save_interval;          /**< Auto-save every N learning steps (0 = disabled) */
} adaptive_network_config_t;

/**
 * @brief Adaptive neural network handle
 */
typedef struct adaptive_network_struct* adaptive_network_t;

/**
 * @brief Create adaptive neural network
 *
 * @param config Network configuration
 * @return Adaptive network handle or NULL on error
 */
adaptive_network_t adaptive_network_create(const adaptive_network_config_t* config);

/**
 * @brief Destroy adaptive neural network
 *
 * @param network Network to destroy
 */
void adaptive_network_destroy(adaptive_network_t network);

/**
 * @brief Process input through adaptive network
 *
 * @param network Adaptive network
 * @param input Input vector
 * @param input_size Size of input
 * @param output Output vector (allocated by caller)
 * @param output_size Size of output buffer
 * @param timestamp Current time
 * @return Number of active neurons (for sparsity tracking)
 */
uint32_t adaptive_network_forward(adaptive_network_t network, const float* input,
                                  uint32_t input_size, float* output, uint32_t output_size,
                                  uint64_t timestamp);

/**
 * @brief Read-only forward pass (does not update statistics)
 *
 * Phase 3: COW-safe inference - allows multiple brains to share network
 * WHY: Enables indefinite sharing during inference-only workloads
 * HOW: Performs computation without mutating network state
 *
 * @param network Adaptive network (not modified)
 * @param input Input vector
 * @param input_size Input dimension
 * @param output Output buffer
 * @param output_size Output dimension
 * @param timestamp Current time (unused in read-only mode)
 * @return Number of active neurons
 */
uint32_t adaptive_network_forward_readonly(const adaptive_network_t network, const float* input,
                                           uint32_t input_size, float* output, uint32_t output_size,
                                           uint64_t timestamp);

/**
 * @brief Get network sparsity
 *
 * @param network Adaptive network
 * @return Current sparsity ratio (0.0-1.0)
 */
float adaptive_network_get_sparsity(adaptive_network_t network);

/**
 * @brief Prune weak connections
 *
 * @param network Adaptive network
 * @param threshold Prune synapses with |weight| < threshold
 * @return Number of synapses pruned
 */
uint32_t adaptive_network_prune(adaptive_network_t network, float threshold);

/**
 * @brief Get network configuration
 *
 * Phase 2.8: Brain resizing support
 * WHY: Enable reading current config for resize operations
 * HOW: Return pointer to internal config (read-only)
 *
 * @param network Adaptive network
 * @return Pointer to network configuration (read-only), or NULL on error
 */
const adaptive_network_config_t* adaptive_network_get_config(adaptive_network_t network);

/**
 * @brief Get number of neurons in network
 *
 * WHY: Enable community detection and topology analysis
 * HOW: Return neuron count from base network
 *
 * @param network Adaptive network
 * @return Number of neurons, or 0 on error
 */
uint32_t adaptive_network_get_num_neurons(adaptive_network_t network);

/**
 * @brief Get underlying base neural network
 *
 * WHY: Enable access to low-level network structure for analysis
 * HOW: Return pointer to base neural network
 *
 * @param network Adaptive network
 * @return Base neural network handle, or NULL on error
 */
neural_network_t adaptive_network_get_base_network(adaptive_network_t network);

/**
 * @brief Get synapse weight between two neurons
 *
 * WHY: Enable community detection to analyze connectivity
 * HOW: Search neuron synapses for connection
 *
 * @param network Adaptive network
 * @param from_neuron Source neuron ID
 * @param to_neuron Target neuron ID
 * @return Synapse weight, or 0.0f if no connection
 */
float adaptive_network_get_synapse_weight(adaptive_network_t network, uint32_t from_neuron, uint32_t to_neuron);

//=============================================================================
// Adaptive Threshold Spiking Functions
//=============================================================================

/**
 * @brief Compute adaptive threshold for input
 *
 * Implements: V_th(x) = (1/k) × mean(|x|)
 *
 * @param input Input vector
 * @param size Input size
 * @param k_factor Firing rate control parameter
 * @return Adaptive threshold value
 */
float adaptive_compute_threshold(const float* input, uint32_t size, float k_factor);

/**
 * @brief Convert continuous value to integer spike count
 *
 * Implements: s_INT = round(x / V_th(x))
 *
 * @param value Input value
 * @param threshold Adaptive threshold
 * @return Integer spike count
 */
int32_t adaptive_value_to_spikes(float value, float threshold);

/**
 * @brief Encode spike count into sparse spike train
 *
 * @param spike_count Integer spike count
 * @param encoding Encoding scheme
 * @param spike_train Output buffer for spike train
 * @param max_length Maximum spike train length
 * @return Actual spike train length
 */
uint32_t adaptive_encode_spikes(int32_t spike_count, spike_encoding_t encoding,
                                uint8_t* spike_train, uint32_t max_length);

/**
 * @brief Decode spike train back to value
 *
 * @param spike_train Spike train
 * @param length Spike train length
 * @param encoding Encoding scheme
 * @param threshold Threshold used for encoding
 * @return Decoded value
 */
float adaptive_decode_spikes(const uint8_t* spike_train, uint32_t length, spike_encoding_t encoding,
                             float threshold);

//=============================================================================
// Pattern Learning & Distillation
//=============================================================================

/**
 * @brief Learning mode for pattern distillation
 */
typedef enum {
    LEARN_MODE_SUPERVISED,   /**< Supervised learning with labels */
    LEARN_MODE_UNSUPERVISED, /**< Unsupervised pattern extraction */
    LEARN_MODE_DISTILLATION, /**< Distill from teacher model/LLM */
    LEARN_MODE_REINFORCEMENT /**< Reinforcement from rewards */
} learning_mode_t;

/**
 * @brief Training example for pattern learning
 */
typedef struct {
    float* input;         /**< Input features */
    uint32_t input_size;  /**< Input vector size */
    float* target;        /**< Target output (supervised) */
    uint32_t target_size; /**< Target vector size */
    float confidence;     /**< Confidence/importance weight */
    char label[64];       /**< Optional semantic label */
} training_example_t;

/**
 * @brief Learn from training example
 *
 * @param network Adaptive network
 * @param example Training example
 * @param mode Learning mode
 * @param learning_rate Learning rate for this update
 * @return Loss/error value
 */
float adaptive_network_learn(adaptive_network_t network, const training_example_t* example,
                             learning_mode_t mode, float learning_rate);

/**
 * @brief Learn from batch of examples
 *
 * @param network Adaptive network
 * @param examples Array of training examples
 * @param num_examples Number of examples
 * @param mode Learning mode
 * @param learning_rate Learning rate
 * @return Average loss over batch
 */
float adaptive_network_learn_batch(adaptive_network_t network, const training_example_t* examples,
                                   uint32_t num_examples, learning_mode_t mode,
                                   float learning_rate);

/**
 * @brief Distill patterns from teacher function
 *
 * This allows learning from any external decision maker (LLM, rule engine, etc.)
 *
 * @param network Adaptive network (student)
 * @param input Input to query teacher with
 * @param input_size Input size
 * @param teacher_fn Teacher function callback
 * @param teacher_context Context passed to teacher
 * @param learning_rate Learning rate
 * @return Loss/error
 */
typedef float* (*teacher_function_t)(const float* input, uint32_t input_size, void* context);

float adaptive_network_distill(adaptive_network_t network, const float* input, uint32_t input_size,
                               teacher_function_t teacher_fn, void* teacher_context,
                               float learning_rate);

//=============================================================================
// Model Persistence
//=============================================================================

/**
 * @brief Serialization format
 */
typedef enum {
    SERIALIZE_FORMAT_BINARY,     /**< Compact binary format */
    SERIALIZE_FORMAT_JSON,       /**< Human-readable JSON */
    SERIALIZE_FORMAT_SAFETENSORS /**< SafeTensors format */
} serialize_format_t;

/**
 * @brief Save network to file
 *
 * @param network Adaptive network
 * @param filepath Path to save file
 * @param format Serialization format
 * @return true on success
 */
bool adaptive_network_save(adaptive_network_t network, const char* filepath,
                           serialize_format_t format);

/**
 * @brief Load network from file
 *
 * @param filepath Path to load from
 * @return Loaded network or NULL on error
 */
adaptive_network_t adaptive_network_load(const char* filepath);

/**
 * @brief Get network size in bytes
 *
 * @param network Adaptive network
 * @return Memory footprint in bytes
 */
size_t adaptive_network_get_size(adaptive_network_t network);

//=============================================================================
// Interpretability & Analysis
//=============================================================================

/**
 * @brief Neuron importance ranking
 */
typedef struct {
    uint32_t neuron_id;        /**< Neuron identifier */
    float importance;          /**< Importance score */
    float avg_activation;      /**< Average activation */
    uint32_t activation_count; /**< Times activated */
    char* most_active_for;     /**< What patterns it responds to */
} neuron_importance_t;

/**
 * @brief Pattern activation analysis
 */
typedef struct {
    uint32_t num_active_neurons; /**< Active neuron count */
    float sparsity;              /**< Current sparsity */
    float confidence;            /**< Prediction confidence */
    uint32_t* active_neuron_ids; /**< IDs of active neurons */
    float* activation_strengths; /**< Activation magnitudes */
} activation_analysis_t;

/**
 * @brief Analyze which neurons fire for input
 *
 * @param network Adaptive network
 * @param input Input pattern
 * @param input_size Input size
 * @param analysis Output analysis (allocated by caller)
 * @return true on success
 */
bool adaptive_network_analyze_activation(adaptive_network_t network, const float* input,
                                         uint32_t input_size, activation_analysis_t* analysis);

/**
 * @brief Rank neurons by importance
 *
 * @param network Adaptive network
 * @param rankings Output rankings array
 * @param max_rankings Maximum rankings to return
 * @return Number of rankings returned
 */
uint32_t adaptive_network_rank_neurons(adaptive_network_t network, neuron_importance_t* rankings,
                                       uint32_t max_rankings);

/**
 * @brief Get explanation for decision
 *
 * @param network Adaptive network
 * @param input Input that produced decision
 * @param input_size Input size
 * @param explanation Output buffer for human-readable explanation
 * @param max_length Maximum explanation length
 * @return Length of explanation
 */
uint32_t adaptive_network_explain(adaptive_network_t network, const float* input,
                                  uint32_t input_size, char* explanation, uint32_t max_length);

//=============================================================================
// Performance Statistics
//=============================================================================

/**
 * @brief Network performance statistics
 */
typedef struct {
    uint64_t total_inferences;     /**< Total forward passes */
    uint64_t total_learning_steps; /**< Total learning updates */
    float avg_sparsity;            /**< Average sparsity */
    float avg_inference_time_us;   /**< Avg inference time (μs) */
    float avg_learning_time_us;    /**< Avg learning time (μs) */
    size_t memory_usage_bytes;     /**< Current memory usage */
    float accuracy;                /**< Accuracy on validation set */
    uint32_t num_pruned_synapses;  /**< Total pruned synapses */
} network_performance_t;

/**
 * @brief Get performance statistics
 *
 * @param network Adaptive network
 * @param stats Output statistics structure
 * @return true on success
 */
bool adaptive_network_get_performance(adaptive_network_t network, network_performance_t* stats);

/**
 * @brief Reset performance counters
 *
 * @param network Adaptive network
 */
void adaptive_network_reset_stats(adaptive_network_t network);

//=============================================================================
// Introspection & Network State Access (for NIMCP 2.5 Consciousness APIs)
//=============================================================================

/**
 * @brief Get total number of neurons in network
 *
 * @param network Adaptive network
 * @return Number of neurons, or 0 on error
 */
uint32_t adaptive_network_get_neuron_count(adaptive_network_t network);

/**
 * @brief Get activation level of specific neuron
 *
 * @param network Adaptive network
 * @param neuron_id Neuron identifier (0 to num_neurons-1)
 * @param activation Output: neuron activation value
 * @return true on success, false if neuron_id invalid
 */
bool adaptive_network_get_neuron_activation(adaptive_network_t network, uint32_t neuron_id,
                                            float* activation);

/**
 * @brief Get list of active neurons above threshold
 *
 * @param network Adaptive network
 * @param threshold Activity threshold (neurons with activation > threshold)
 * @param neuron_ids Output: array of active neuron IDs (caller allocates)
 * @param activations Output: array of activation values (caller allocates)
 * @param max_neurons Maximum neurons to return
 * @return Number of active neurons found
 */
uint32_t adaptive_network_get_active_neurons(adaptive_network_t network, float threshold,
                                             uint32_t* neuron_ids, float* activations,
                                             uint32_t max_neurons);

/**
 * @brief Get number of synaptic connections for a neuron
 *
 * @param network Adaptive network
 * @param neuron_id Neuron identifier
 * @param num_connections Output: number of connections
 * @return true on success
 */
bool adaptive_network_get_connection_count(adaptive_network_t network, uint32_t neuron_id,
                                           uint32_t* num_connections);

/**
 * @brief Get total weight (sum of connection weights) for a neuron
 *
 * @param network Adaptive network
 * @param neuron_id Neuron identifier
 * @param total_weight Output: sum of absolute weights
 * @return true on success
 */
bool adaptive_network_get_total_weight(adaptive_network_t network, uint32_t neuron_id,
                                       float* total_weight);

/**
 * @brief Get network base handle for direct access
 * WARNING: For internal use by consciousness subsystems only
 *
 * @param network Adaptive network
 * @return Base neural network handle (do not free!)
 */
neural_network_t adaptive_network_get_base_network(adaptive_network_t network);

#ifdef __cplusplus
}
#endif

#endif  // NIMCP_ADAPTIVE_H
