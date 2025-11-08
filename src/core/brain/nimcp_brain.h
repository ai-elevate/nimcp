//=============================================================================
// nimcp_brain.h - High-Level Brain API (Application-Friendly)
//=============================================================================

#ifndef NIMCP_BRAIN_H
#define NIMCP_BRAIN_H

#include "plasticity/adaptive/nimcp_adaptive.h"
#include "networking/distributed/nimcp_distributed_cognition.h"
#include "common/nimcp_export.h"

// Forward declarations for opaque types (full headers only in .c file)
// Using void* to avoid typedef conflicts between modules
// The .c file will include headers and cast appropriately

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
    BRAIN_SIZE_TINY,   /**< 100 neurons, <1MB,  ~0.1ms inference */
    BRAIN_SIZE_SMALL,  /**< 1K neurons,  ~10MB, ~0.5ms inference */
    BRAIN_SIZE_MEDIUM, /**< 10K neurons, ~50MB, ~5ms inference */
    BRAIN_SIZE_LARGE,  /**< 100K neurons, ~500MB, ~50ms inference */
    BRAIN_SIZE_CUSTOM  /**< User-defined size */
} brain_size_t;

/**
 * @brief Brain task templates
 */
typedef enum {
    BRAIN_TASK_CLASSIFICATION,   /**< Multi-class classification */
    BRAIN_TASK_REGRESSION,       /**< Continuous value prediction */
    BRAIN_TASK_PATTERN_MATCHING, /**< Pattern recognition */
    BRAIN_TASK_SEQUENCE,         /**< Temporal sequence learning */
    BRAIN_TASK_ASSOCIATION,      /**< Association learning (Hebbian) */
    BRAIN_TASK_CUSTOM            /**< Custom task */
} brain_task_t;

//=============================================================================
// Brain Handle & Configuration
//=============================================================================

/**
 * @brief Opaque brain handle
 */
typedef struct brain_struct* brain_t;

/**
 * @brief Comprehensive brain configuration
 *
 * WHAT: Extended configuration for all advanced subsystems
 * WHY:  Enable full integration of consciousness, glial, sensory, and cognitive modules
 * HOW:  Optional flags allow selective feature activation
 */
typedef struct {
    // === CORE CONFIGURATION ===
    brain_size_t size;        /**< Brain size preset */
    brain_task_t task;        /**< Task template */
    uint32_t num_inputs;      /**< Input dimension */
    uint32_t num_outputs;     /**< Output dimension */
    float learning_rate;      /**< Learning rate (0.001-0.1) */
    float sparsity_target;    /**< Target sparsity (0.7-0.95) */
    bool enable_explanations; /**< Enable interpretability */
    char task_name[64];       /**< Name for this brain */

    // === PHASE 3: DISTRIBUTED COGNITION ===
    bool enable_distributed;  /**< Enable P2P cognitive coordination */
    p2p_node_t p2p_node;      /**< P2P network node (if distributed) */

    // === PHASE 5/6: BIOLOGICAL REALISM ===
    bool enable_glial;        /**< Enable glial integration (astrocytes, oligodendrocytes, microglia) */
    bool enable_oscillations; /**< Enable brain wave analysis (delta, theta, alpha, beta, gamma) */
    uint32_t num_astrocytes;  /**< Number of astrocytes (default: neurons/5) */
    uint32_t num_oligodendrocytes; /**< Number of oligodendrocytes (default: neurons/7) */
    uint32_t num_microglia;   /**< Number of microglia (default: neurons/10) */

    // === PHASE 5.3: SENSORY PROCESSING ===
    bool enable_visual_cortex; /**< Enable visual cortex (V1) for image processing */
    bool enable_audio_cortex;  /**< Enable audio cortex (A1) for sound processing */

    // === CONSCIOUSNESS & COGNITION ===
    bool enable_introspection; /**< Enable self-awareness and uncertainty estimation */
    bool enable_ethics;        /**< Enable ethical reasoning (Golden Rule, empathy) */
    bool enable_salience;      /**< Enable fast attention/relevance evaluation */
    bool enable_consolidation; /**< Enable memory consolidation (sleep-like learning) */
    bool enable_curiosity;     /**< Enable exploration and knowledge gap detection */
    bool enable_knowledge;     /**< Enable multi-domain knowledge acquisition */
    bool enable_wellbeing;     /**< Enable distress detection and ethical safeguards */

    // === ADVANCED PLASTICITY ===
    bool enable_eligibility_traces; /**< Enable temporal credit assignment (Phase 5.1) */
    bool enable_pink_noise;    /**< Enable pink noise neuromodulation (Phase 4) */
    bool enable_spike_nlp;     /**< Enable NLP via spike encoding (Phase 5.1) */
    bool enable_fractal_topology; /**< Enable scale-free network topology (Phase 2) */

    // === PHASE 8: UNIFIED MULTI-MODAL PROCESSING ===
    bool enable_multimodal_integration; /**< Enable multi-modal sensory integration */
    uint32_t visual_feature_dim;  /**< Visual feature dimension (0 = no visual) */
    uint32_t audio_feature_dim;   /**< Audio feature dimension (0 = no audio) */
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
brain_t brain_create(const char* task_name, brain_size_t size, brain_task_t task,
                     uint32_t num_inputs, uint32_t num_outputs);

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
// Phase 2: Copy-on-Write Brain Cloning
//=============================================================================

/**
 * @brief Clone brain using copy-on-write (COW) optimization
 *
 * WHAT: Creates lightweight clone sharing memory with original
 * WHY:  Enable efficient replication with 86% memory savings
 * HOW:  Shares network structure, copies on first write
 *
 * @param original Brain to clone
 * @return Cloned brain or NULL on error
 */
brain_t brain_clone_cow(brain_t original);

//=============================================================================
// Phase 3: Distributed Brain API
//=============================================================================

/**
 * @brief Create distributed brain with P2P coordination
 *
 * WHAT: Creates a brain that can coordinate with peer brains over network
 * WHY:  Enable multi-brain collaborative learning and shared chemical signals
 * HOW:  Integrates distributed cognition coordinator for neuromod/glial sync
 *
 * @param task_name Human-readable name
 * @param size Brain size preset
 * @param task Task template
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @param p2p_node P2P network node for coordination
 * @return Distributed brain handle or NULL on error
 *
 * THREAD SAFETY: Thread-safe creation
 * PERFORMANCE: O(n) where n = num_neurons + network initialization
 */
NIMCP_EXPORT brain_t brain_create_distributed(
    const char* task_name,
    brain_size_t size,
    brain_task_t task,
    uint32_t num_inputs,
    uint32_t num_outputs,
    p2p_node_t p2p_node
);

/**
 * @brief Enable distributed coordination on existing brain
 *
 * WHAT: Retrofits an existing brain with distributed cognition
 * WHY:  Allow conversion of standalone brain to distributed mode
 * HOW:  Creates distrib_cognition coordinator and starts sync threads
 *
 * @param brain Brain handle
 * @param p2p_node P2P network node
 * @return true on success, false on failure
 *
 * THREAD SAFETY: Thread-safe if brain not actively processing
 */
NIMCP_EXPORT bool brain_enable_distributed(brain_t brain, p2p_node_t p2p_node);

/**
 * @brief Synchronize neuromodulators with peer brains
 *
 * WHAT: Manually trigger neuromodulator broadcast to network
 * WHY:  Allow explicit control of sync timing for performance tuning
 * HOW:  Calls distrib_cognition_broadcast_neuromod for all neuromod types
 *
 * @param brain Distributed brain handle
 * @return true on success, false if not distributed or error
 *
 * THREAD SAFETY: Thread-safe
 * PERFORMANCE: O(P × N) where P=peers, N=neuromod types
 */
NIMCP_EXPORT bool brain_sync_neuromodulators(brain_t brain);

/**
 * @brief Get distributed cognition statistics
 *
 * WHAT: Query network sync stats (broadcasts, latency, peer count)
 * WHY:  Monitor distributed brain performance and health
 * HOW:  Forwards query to underlying distrib_cognition coordinator
 *
 * @param brain Distributed brain handle
 * @param stats Output statistics structure
 * @return true on success, false if not distributed
 */
NIMCP_EXPORT bool brain_get_distributed_stats(
    brain_t brain,
    distrib_cognition_stats_t* stats
);

/**
 * @brief Check if brain is distributed
 *
 * @param brain Brain handle
 * @return true if distributed coordination enabled, false otherwise
 */
NIMCP_EXPORT bool brain_is_distributed(brain_t brain);

//=============================================================================
// Learning API
//=============================================================================

/**
 * @brief Simple feature vector for learning
 */
typedef struct {
    float* features;       /**< Feature values */
    uint32_t num_features; /**< Number of features */
    char label[64];        /**< Semantic label */
    float confidence;      /**< Training confidence (0-1) */
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
float brain_learn_example(brain_t brain, const float* features, uint32_t num_features,
                          const char* label, float confidence);

/**
 * @brief Learn from batch of examples
 *
 * @param brain Brain handle
 * @param examples Array of examples
 * @param num_examples Example count
 * @return Average loss
 */
float brain_learn_batch(brain_t brain, const brain_example_t* examples, uint32_t num_examples);

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
typedef float (*llm_teacher_fn_t)(const float* input, uint32_t num_features, void* context,
                                  char* output_label, uint32_t max_label_len);

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
float brain_learn_from_llm(brain_t brain, const float* input, uint32_t num_features,
                           llm_teacher_fn_t llm_fn, void* llm_context);

//=============================================================================
// Inference API
//=============================================================================

/**
 * @brief Decision result
 */
typedef struct {
    char label[64];       /**< Decision label */
    float confidence;     /**< Confidence (0-1) */
    float* output_vector; /**< Raw output vector */
    uint32_t output_size; /**< Output vector size */

    // Interpretability (if enabled)
    uint32_t num_active_neurons; /**< Active neuron count */
    uint32_t* active_neuron_ids; /**< Active neuron IDs */
    float sparsity;              /**< Actual sparsity */
    char explanation[256];       /**< Human-readable explanation */

    uint64_t inference_time_us; /**< Inference time (microseconds) */
} brain_decision_t;

/**
 * @brief Make decision for input
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @return Decision result (caller must free with brain_free_decision)
 */
brain_decision_t* brain_decide(brain_t brain, const float* features, uint32_t num_features);

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
bool brain_decide_batch(brain_t brain, const float** inputs, uint32_t num_inputs,
                        uint32_t features_per_input, brain_decision_t* decisions);

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
    char task_name[64];           /**< Brain name */
    brain_size_t size;            /**< Size preset */
    uint32_t num_neurons;         /**< Total neurons */
    uint32_t num_synapses;        /**< Total synapses */
    uint32_t num_active_synapses; /**< Non-pruned synapses */

    uint64_t total_inferences;     /**< Inference count */
    uint64_t total_learning_steps; /**< Learning steps */

    float avg_sparsity;          /**< Average sparsity */
    float avg_inference_time_us; /**< Avg inference (μs) */
    float current_learning_rate; /**< Current learning rate */

    float accuracy;      /**< Validation accuracy */
    size_t memory_bytes; /**< Memory usage */
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
 * @brief Copy-on-Write statistics
 */
typedef struct {
    bool is_cow_clone;        /**< True if this brain is a COW clone */
    uint32_t cow_ref_count;   /**< Reference count (1 for original, 2+ for shared) */
    size_t cow_shared_bytes;  /**< Bytes shared via COW */
    size_t cow_private_bytes; /**< Bytes private to this brain */
} brain_cow_stats_t;

/**
 * @brief Get COW statistics for brain
 *
 * @param brain Brain handle
 * @param cow_stats Output COW statistics
 * @return true on success
 */
bool brain_get_cow_stats(brain_t brain, brain_cow_stats_t* cow_stats);

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
uint32_t brain_get_top_neurons(brain_t brain, uint32_t top_n, uint32_t* neuron_ids,
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
bool brain_explain_decision(brain_t brain, const float* features, uint32_t num_features,
                            char* explanation, uint32_t max_length);

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

//=============================================================================
// Phase 8: Unified Multi-Modal Processing API
//=============================================================================

/**
 * @brief Multi-modal processing input bundle
 *
 * WHAT: Container for all possible input modalities
 * WHY:  Enable unified processing of visual, audio, and direct inputs
 * HOW:  Pass NULL for unused modalities
 *
 * DESIGN: Flexible multi-modal input - any combination of modalities allowed
 */
typedef struct {
    // Visual input (optional)
    const uint8_t* visual_data;  /**< Raw image data (grayscale or RGB) */
    uint32_t visual_width;       /**< Image width in pixels */
    uint32_t visual_height;      /**< Image height in pixels */
    uint32_t visual_channels;    /**< 1=grayscale, 3=RGB */

    // Audio input (optional)
    const float* audio_data;     /**< Audio samples (normalized -1 to 1) */
    uint32_t audio_samples;      /**< Number of audio samples */
    uint8_t audio_channels;      /**< 1=mono, 2=stereo */

    // Direct input (optional)
    const float* direct_data;    /**< Direct feature vector */
    uint32_t direct_dim;         /**< Direct feature dimension */

    // Temporal information
    uint64_t timestamp_ms;       /**< Timestamp for temporal alignment */
} brain_multimodal_input_t;

/**
 * @brief Unified multi-modal processing result
 *
 * WHAT: Comprehensive output with confidence and explanations
 * WHY:  Provide full cognitive state after processing
 * HOW:  Populated by cognitive modules during integration
 */
typedef struct {
    // Core decision
    float* output_vector;        /**< Output feature vector */
    uint32_t output_dim;         /**< Output dimension */
    char decision_label[64];     /**< Human-readable decision label */
    float confidence;            /**< Overall confidence [0,1] */

    // Cognitive assessments
    float introspection_uncertainty; /**< Epistemic uncertainty [0,1] */
    float salience_score;        /**< Output salience [0,1] */
    bool ethical_approved;       /**< Passed ethical review */
    float novelty_score;         /**< Input novelty [0,1] */

    // Attention breakdown
    float visual_attention;      /**< Visual modality weight [0,1] */
    float audio_attention;       /**< Audio modality weight [0,1] */
    float direct_attention;      /**< Direct input weight [0,1] */

    // Explanation
    char explanation[256];       /**< Human-readable explanation */
} brain_multimodal_output_t;

/**
 * @brief Process multi-modal input through unified cognitive architecture
 *
 * WHAT: Unified processing pipeline integrating all sensory and cognitive modules
 * WHY:  Enable coordinated multi-modal perception and cognition
 * HOW:  Sensory extraction → Integration → Neural processing → Cognitive checks → Output
 *
 * ARCHITECTURE:
 * 1. SENSORY STAGE:
 *    - Visual cortex extracts CNN features (if visual_data present)
 *    - Audio cortex extracts FFT features (if audio_data present)
 *    - Direct features passed through (if direct_data present)
 *
 * 2. INTEGRATION STAGE:
 *    - Multi-modal integration layer combines features
 *    - Attention weighting by modality
 *    - Unified representation → network input
 *
 * 3. NEURAL PROCESSING:
 *    - Feed integrated features to neural network
 *    - STDP learning updates synapses
 *    - Glial modulation affects transmission
 *    - Brain oscillations coordinate activity
 *    - Pink noise adds exploration
 *
 * 4. COGNITIVE PROCESSING:
 *    - Introspection assesses confidence
 *    - Salience identifies important patterns
 *    - Ethics validates output
 *    - Curiosity detects novelty
 *    - Knowledge applies constraints
 *
 * 5. OUTPUT INTEGRATION:
 *    - Consolidation strengthens memories
 *    - Ethical filtering blocks harmful outputs
 *    - Salience weighting prioritizes relevant outputs
 *    - Extract final decision with explanations
 *
 * @param brain Brain handle
 * @param input Multi-modal input bundle
 * @param output Output structure (pre-allocated)
 * @return true on success, false on failure
 *
 * USAGE:
 * ```c
 * // Create brain with multi-modal support
 * brain_config_t config = brain_default_config("perception", BRAIN_SIZE_MEDIUM,
 *                                               BRAIN_TASK_CLASSIFICATION, 256, 10);
 * config.enable_visual_cortex = true;
 * config.enable_audio_cortex = true;
 * config.enable_ethics = true;
 * config.enable_introspection = true;
 * config.enable_multimodal_integration = true;
 * config.visual_feature_dim = 128;
 * config.audio_feature_dim = 64;
 *
 * brain_t brain = brain_create_custom(&config);
 *
 * // Process camera + microphone input
 * uint8_t camera_frame[640 * 480];  // Grayscale
 * float microphone_samples[1024];    // Audio
 *
 * brain_multimodal_input_t input = {
 *     .visual_data = camera_frame,
 *     .visual_width = 640,
 *     .visual_height = 480,
 *     .visual_channels = 1,
 *     .audio_data = microphone_samples,
 *     .audio_samples = 1024,
 *     .audio_channels = 1,
 *     .direct_data = NULL,
 *     .direct_dim = 0,
 *     .timestamp_ms = nimcp_time_ms()
 * };
 *
 * brain_multimodal_output_t output = {0};
 * output.output_vector = nimcp_malloc(10 * sizeof(float));
 * output.output_dim = 10;
 *
 * brain_process_multimodal(brain, &input, &output);
 *
 * printf("Decision: %s (confidence: %.2f)\n", output.decision_label, output.confidence);
 * printf("Ethical: %s, Novelty: %.2f\n",
 *        output.ethical_approved ? "YES" : "NO", output.novelty_score);
 * printf("Visual attention: %.2f, Audio attention: %.2f\n",
 *        output.visual_attention, output.audio_attention);
 * printf("Explanation: %s\n", output.explanation);
 * ```
 *
 * COMPLEXITY:
 * - Visual processing: O(W·H·K²·F) where K=kernel, F=filters
 * - Audio processing: O(N·log(N)) FFT
 * - Integration: O(D_v + D_a + D_d)
 * - Neural step: O(N·S) where N=neurons, S=avg synapses
 * - Overall: O(sensory + neural + cognitive)
 *
 * PERFORMANCE: ~10-50ms typical for medium brain with camera+audio
 *
 * THREAD SAFETY: Not thread-safe (brain state modified)
 *
 * ERROR HANDLING:
 * - Returns false if brain NULL or not configured for multi-modal
 * - Returns false if all input modalities are NULL
 * - Gracefully handles missing optional modalities
 *
 * MEMORY: No allocation (uses pre-allocated output buffer)
 *
 * @version 2.7.0 Phase 8 - Unified Multi-Modal Architecture
 * @author NIMCP Development Team
 * @date 2025-11-08
 */
bool brain_process_multimodal(
    brain_t brain,
    const brain_multimodal_input_t* input,
    brain_multimodal_output_t* output
);

//=============================================================================
// Comprehensive Module Access API
// NOTE: Module accessors will be added incrementally as modules are initialized
// For now, modules are accessible via void* cast from brain internals
//=============================================================================

/*
 * Future accessor functions (to be implemented with proper initialization):
 * - glial_integration_t* brain_get_glial(brain_t brain);
 * - brain_oscillation_analyzer_t* brain_get_oscillations(brain_t brain);
 * - visual_cortex_t* brain_get_visual_cortex(brain_t brain);
 * - audio_cortex_t* brain_get_audio_cortex(brain_t brain);
 * - introspection_context_t* brain_get_introspection(brain_t brain);
 * - ethics_engine_t* brain_get_ethics(brain_t brain);
 * - salience_evaluator_t* brain_get_salience(brain_t brain);
 * - consolidation_handle_t brain_get_consolidation(brain_t brain);
 * - curiosity_engine_t brain_get_curiosity(brain_t brain);
 * - knowledge_system_t brain_get_knowledge(brain_t brain);
 * - wellbeing_monitor_t brain_get_wellbeing(brain_t brain);
 * - eligibility_trace_system_t* brain_get_eligibility_traces(brain_t brain);
 * - neuromod_pink_t* brain_get_pink_noise(brain_t brain);
 * - spike_nlp_t* brain_get_spike_nlp(brain_t brain);
 *
 * These will be uncommented and implemented once module initialization is complete.
 */

#endif  // NIMCP_BRAIN_H
