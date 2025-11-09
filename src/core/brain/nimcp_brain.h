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

// === PHASE 10: ADVANCED COGNITIVE SYSTEMS ===

// Phase 10.1: Working Memory
typedef struct working_memory working_memory_t;

// Phase 10.2: Emotional Tagging
typedef struct emotional_system emotional_system_t;
typedef struct emotional_tag emotional_tag_t;
typedef struct emotional_state emotional_state_t;

// Phase 10.3: Executive Functions
typedef struct executive_controller executive_controller_t;

// Phase 10.4: Sleep-Wake Cycle
typedef struct sleep_system_struct* sleep_system_t;

// Phase 10.5: Mental Health Monitoring
typedef struct mental_health_monitor mental_health_monitor_t;

// Phase 10.6: Theory of Mind
// Forward declare Theory of Mind (opaque pointer)
typedef struct theory_of_mind_s* theory_of_mind_t;

// Phase 10.7: Natural Explanations
typedef struct explanation_generator_s* explanation_generator_t;  // Opaque pointer
typedef struct natural_explanation natural_explanation_t;

// Phase 10.8: Meta-Learning
typedef struct meta_learner_s* meta_learner_t;  // Opaque pointer

// Phase 10.9: Predictive Processing
typedef struct predictive_network_s* predictive_network_t;  // Opaque pointer

// Phase 10.11: Mirror Neurons (Social Cognition & Imitation Learning)
typedef struct mirror_neurons_system* mirror_neurons_t;  // Opaque pointer

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
    bool enable_speech_cortex; /**< Enable speech cortex (STG/Wernicke) for language processing */

    // === CONSCIOUSNESS & COGNITION ===
    bool enable_introspection; /**< Enable self-awareness and uncertainty estimation */
    bool enable_ethics;        /**< Enable ethical reasoning (Golden Rule, empathy) */
    bool enable_salience;      /**< Enable fast attention/relevance evaluation */
    bool enable_consolidation; /**< Enable memory consolidation (sleep-like learning) */
    bool enable_curiosity;     /**< Enable exploration and knowledge gap detection */
    bool enable_knowledge;     /**< Enable multi-domain knowledge acquisition */
    bool enable_wellbeing;     /**< Enable distress detection and ethical safeguards */
    bool enable_logic;         /**< Enable symbolic logic and reasoning (Phase 9.4) */

    // === ADVANCED PLASTICITY ===
    bool enable_eligibility_traces; /**< Enable temporal credit assignment (Phase 5.1) */
    bool enable_pink_noise;    /**< Enable pink noise neuromodulation (Phase 4) */
    bool enable_spike_nlp;     /**< Enable NLP via spike encoding (Phase 5.1) */
    bool enable_fractal_topology; /**< Enable scale-free network topology (Phase 2) */

    // === PHASE 8: UNIFIED MULTI-MODAL PROCESSING ===
    bool enable_multimodal_integration; /**< Enable multi-modal sensory integration */
    uint32_t visual_feature_dim;  /**< Visual feature dimension (0 = no visual) */
    uint32_t audio_feature_dim;   /**< Audio feature dimension (0 = no audio) */
    uint32_t speech_feature_dim;  /**< Speech feature dimension (0 = no speech) */
    uint32_t language_feature_dim; /**< Language/text feature dimension (0 = no language, Phase 9.4) */

    // === PHASE 9.2: EPISTEMIC FILTERING ===
    bool enable_epistemic_filter; /**< Enable cognitive bias prevention */

    // === PHASE 9.3: WELLBEING MONITORING ===
    bool enable_wellbeing_monitoring; /**< Enable self-preservation and distress monitoring */
    uint64_t wellbeing_check_interval_ms; /**< Interval between wellbeing checks (0 = always) */

    // === PHASE 10: ADVANCED COGNITIVE SYSTEMS ===

    // Phase 10.1: Working Memory
    bool enable_working_memory;       /**< Enable Miller's 7±2 working memory buffer */
    uint32_t working_memory_capacity; /**< Working memory capacity (default: 7) */
    float working_memory_decay_tau_ms; /**< Decay time constant in ms (default: 1000) */

    // Phase 10.2: Emotional Tagging
    bool enable_emotional_tagging;    /**< Enable emotional context for memories */
    bool enable_emotional_memories;   /**< Enable emotional encoding boost */

    // Phase 10.3: Executive Functions
    bool enable_executive_control;    /**< Enable task switching and planning */
    bool enable_task_switching;       /**< Enable multi-task management */
    bool enable_planning;             /**< Enable goal-directed planning */

    // Phase 10.4: Sleep-Wake Cycle
    bool enable_sleep_wake_cycle;     /**< Enable sleep/wake states and consolidation */
    float sleep_pressure_threshold;   /**< Sleep needed threshold (default: 0.8) */
    bool enable_memory_replay;        /**< Enable memory replay during sleep */
    bool enable_synaptic_homeostasis; /**< Enable synaptic downscaling */
    bool enable_rem_creativity;       /**< Enable REM creative recombination */

    // Phase 10.5: Mental Health Monitoring
    bool enable_mental_health_monitoring; /**< Enable disorder detection */
    bool enable_auto_intervention;    /**< Enable automatic interventions */
    bool shutdown_on_critical_disorder; /**< Shutdown if critical disorder detected */

    // Phase 10.6: Theory of Mind
    bool enable_theory_of_mind;       /**< Enable social cognition and empathy */
    bool enable_empathy_responses;    /**< Enable empathetic emotional responses */
    bool enable_false_belief_tracking; /**< Enable false belief understanding */

    // Phase 10.7: Natural Explanations
    bool enable_natural_explanations; /**< Enable human-readable explanations */
    bool enable_causal_explanations;  /**< Enable causal chain generation */

    // Phase 10.8: Meta-Learning
    bool enable_meta_learning;        /**< Enable learning-to-learn */
    bool enable_adaptive_meta_lr;     /**< Enable adaptive learning rates per region */
    uint32_t meta_task_batch_size;    /**< Task batch size for meta-learning */
    uint32_t meta_k_shot;             /**< K-shot learning parameter (1, 5, or 10) */

    // Phase 10.9: Predictive Processing
    bool enable_predictive_processing; /**< Enable predictive coding */
    bool enable_active_inference;     /**< Enable active inference for actions */

    // Phase 10.11: Mirror Neurons (Social Cognition & Imitation Learning)
    bool enable_mirror_neurons;       /**< Enable observation-based learning */
    uint32_t mirror_neuron_count;     /**< Number of mirror neurons (default: 1000) */
    uint32_t mirror_max_actions;      /**< Max distinct actions to track (default: 100) */
    uint32_t mirror_max_agents;       /**< Max agents to observe (default: 10) */
    float mirror_learning_rate;       /**< Association learning rate (default: 0.01) */
    float mirror_match_threshold;     /**< Action matching threshold (default: 0.7) */

    // === PERSISTENCE & CHECKPOINTING ===
    const char* checkpoint_path;      /**< Path to checkpoint file (NULL = no checkpoint) */
    bool auto_load;                   /**< Auto-load from checkpoint on create (default: true) */
    bool auto_save;                   /**< Auto-save to checkpoint periodically (default: false) */
    uint32_t auto_save_interval;      /**< Auto-save every N decisions (0 = disabled) */

    // === SNAPSHOTS ===
    const char* snapshot_dir;         /**< Directory for snapshots (default: "./snapshots") */
    bool enable_auto_snapshots;       /**< Enable automatic snapshots (default: false) */
    uint32_t auto_snapshot_interval;  /**< Auto-snapshot every N decisions (0 = disabled) */
    bool compress_snapshots;          /**< Compress snapshots (default: true) */
    bool encrypt_snapshots;           /**< Encrypt snapshots (default: true) */
    const char* encryption_key;       /**< Encryption key (32 bytes hex, NULL = derive from system) */
    bool save_initial_snapshot;       /**< Save snapshot at creation (default: true) */
    bool save_final_snapshot;         /**< Save snapshot at destruction (default: true) */
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

/**
 * @brief Get working memory from brain (Phase 10.2 accessor)
 *
 * WHAT: Retrieve pointer to brain's working memory subsystem
 * WHY:  Allow API wrapper functions to access working memory
 * HOW:  Return brain->working_memory field
 *
 * @param brain Brain instance
 * @return Working memory pointer or NULL if not enabled
 */
working_memory_t* brain_get_working_memory(brain_t brain);

/**
 * WHAT: Retrieve pointer to brain's sleep/wake subsystem (Phase 10.4)
 * WHY:  Allow external control of sleep cycles and pressure monitoring
 * HOW:  Return brain->sleep_system field
 *
 * @param brain Brain instance
 * @return Sleep system pointer or NULL if invalid brain
 */
sleep_system_t brain_get_sleep_system(brain_t brain);

/**
 * WHAT: Retrieve pointer to brain's Theory of Mind subsystem (Phase 10.6)
 * WHY:  Allow external access to social cognition and empathy functions
 * HOW:  Return brain->theory_of_mind field
 *
 * @param brain Brain instance
 * @return Theory of Mind pointer or NULL if not enabled/invalid brain
 */
theory_of_mind_t brain_get_theory_of_mind(brain_t brain);

/**
 * @brief Get explanation generator from brain (Phase 10.7)
 *
 * @param brain Brain instance
 * @return Explanation generator pointer or NULL if not enabled/invalid brain
 */
explanation_generator_t brain_get_explanation_generator(brain_t brain);

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
// Phase 2.8: Distributed Copy-on-Write Brain Cloning
//=============================================================================

// Forward declaration for distributed COW types
typedef struct distributed_cow_config distributed_cow_config_t;
typedef struct distributed_cow_stats distributed_cow_stats_t;

/**
 * @brief Create distributed COW clone on remote node
 *
 * WHAT: Creates a COW clone that fetches network data from remote master
 * WHY:  Enable efficient distributed deployment without full network copy
 * HOW:  Establishes P2P connection, initializes cache, registers with master
 *
 * @param original Brain to clone (must be on master node)
 * @param remote_host Master node hostname/IP
 * @param remote_port Master node port
 * @param config Distributed COW configuration (NULL for defaults)
 * @return Distributed COW clone or NULL on error
 *
 * MEMORY: ~1-10MB overhead (cache + metadata)
 * LATENCY: ~10-100ms (connection + handshake)
 * BANDWIDTH: Lazy loading - only fetches neurons needed for inference
 *
 * Example:
 * ```c
 * // On master node (192.168.1.100:5000)
 * brain_t master = brain_create("model", BRAIN_SIZE_MEDIUM,
 *                               BRAIN_TASK_CLASSIFICATION, 256, 10);
 * brain_enable_distributed_cow_master(master, p2p_node);
 *
 * // On remote node (192.168.1.101)
 * brain_t clone = brain_clone_cow_distributed(master, "192.168.1.100", 5000, NULL);
 * // Clone shares network with master, fetches on demand
 * brain_decision_t* decision = brain_decide(clone, input, 256);
 * ```
 */
NIMCP_EXPORT brain_t brain_clone_cow_distributed(
    brain_t original,
    const char* remote_host,
    uint16_t remote_port,
    const distributed_cow_config_t* config
);

/**
 * @brief Enable distributed COW serving on master node
 *
 * WHAT: Configures brain to serve network segments to remote clones
 * WHY:  Allow brain to act as master for distributed COW
 * HOW:  Registers network segment handlers, starts P2P listener
 *
 * @param brain Brain to enable as master
 * @param p2p_node P2P node for serving (must be started)
 * @return true on success, false on failure
 */
NIMCP_EXPORT bool brain_enable_distributed_cow_master(
    brain_t brain,
    p2p_node_t p2p_node
);

/**
 * @brief Get distributed COW statistics
 *
 * @param brain Brain handle
 * @param stats Output statistics structure
 * @return true on success, false if not distributed COW
 */
NIMCP_EXPORT bool brain_get_distributed_cow_stats(
    brain_t brain,
    distributed_cow_stats_t* stats
);

/**
 * @brief Check if brain is distributed COW clone
 *
 * @param brain Brain handle
 * @return true if distributed COW clone, false otherwise
 */
NIMCP_EXPORT bool brain_is_distributed_cow(brain_t brain);

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

//=============================================================================
// Snapshot API - Compressed & Encrypted State Snapshots
//=============================================================================

/**
 * @brief Snapshot metadata
 */
typedef struct {
    char name[128];              /**< Snapshot name */
    char description[512];       /**< Human-readable description */
    uint64_t timestamp;          /**< Creation timestamp (Unix epoch) */
    uint32_t file_size;          /**< Compressed size in bytes */
    bool is_compressed;          /**< Compression enabled */
    bool is_encrypted;           /**< Encryption enabled */
} brain_snapshot_info_t;

/**
 * @brief Save brain snapshot with compression and encryption
 *
 * WHAT: Creates a named, timestamped snapshot of complete brain state
 * WHY:  Enable backups, A/B testing, version control, disaster recovery
 * HOW:  Saves to snapshot_dir with compression and encryption by default
 *
 * Snapshots are saved as: snapshot_dir/name_timestamp.snapshot
 * - Compressed with zlib (default)
 * - Encrypted with AES-256 (default)
 * - Include full brain state (network, subsystems, knowledge)
 *
 * @param brain Brain instance
 * @param name Snapshot name (e.g., "before_experiment", "v1.0")
 * @param description Optional description (can be NULL)
 * @return true on success, false on error
 */
bool brain_save_snapshot(brain_t brain, const char* name, const char* description);

/**
 * @brief Restore brain from snapshot
 *
 * WHAT: Loads brain state from named snapshot
 * WHY:  Restore previous state, rollback changes, A/B testing
 * HOW:  Decompresses and decrypts snapshot, restores all state
 *
 * @param brain Brain instance to restore into (or NULL to create new)
 * @param name Snapshot name
 * @return Brain instance (restored if provided, new if NULL), or NULL on error
 */
brain_t brain_restore_snapshot(brain_t brain, const char* name);

/**
 * @brief List available snapshots
 *
 * WHAT: Enumerate all snapshots for this brain
 * WHY:  Allow users to see available restore points
 * HOW:  Scans snapshot_dir, parses metadata
 *
 * @param brain Brain instance (for snapshot_dir config)
 * @param infos Output array of snapshot info (allocated by caller)
 * @param max_count Maximum number of snapshots to return
 * @param out_count Output: actual number of snapshots found
 * @return true on success, false on error
 */
bool brain_list_snapshots(brain_t brain, brain_snapshot_info_t* infos,
                         uint32_t max_count, uint32_t* out_count);

/**
 * @brief Delete snapshot
 *
 * @param brain Brain instance (for snapshot_dir config)
 * @param name Snapshot name to delete
 * @return true on success, false on error
 */
bool brain_delete_snapshot(brain_t brain, const char* name);

//=============================================================================
// Phase 9.0: Pre-Trained Models API
//=============================================================================

/**
 * @brief Load pre-trained baseline model
 *
 * WHAT: Loads a pre-trained NIMCP baseline model with trained weights
 * WHY:  Enables instant NIMCP integration without 48-hour training
 * HOW:  Downloads model on first use, caches locally
 *
 * @param model_id Model identifier (e.g., "nimcp_baseline_medium")
 *                 Available models:
 *                 - "nimcp_baseline_small":  1K neurons, 4.2MB, 0.3ms inference
 *                 - "nimcp_baseline_medium": 10K neurons, 42MB, 0.8ms inference (RECOMMENDED)
 *                 - "nimcp_baseline_large":  100K neurons, 420MB, 3ms inference
 * @param task Task template for output layer configuration
 * @return Brain handle with pre-trained weights or NULL on error
 *
 * Example:
 * ```c
 * // Load pre-trained model (instant, no training!)
 * brain_t brain = brain_create_pretrained("nimcp_baseline_medium",
 *                                         BRAIN_TASK_CLASSIFICATION);
 *
 * // Use immediately - works out of the box!
 * brain_output_t output = brain_process_multimodal(brain, &input);
 * ```
 *
 * Note: Models are downloaded from https://models.nimcp.ai and cached in:
 *       - Linux/macOS: ~/.nimcp/models/
 *       - Windows: %LOCALAPPDATA%\\NIMCP\\models\\
 */
brain_t brain_create_pretrained(const char* model_id, brain_task_t task);

/**
 * @brief Fine-tuning configuration
 */
typedef struct {
    float learning_rate;      /**< Learning rate (default: 0.001) */
    uint32_t num_epochs;      /**< Number of training epochs (default: 5) */
    bool freeze_sensory;      /**< Freeze visual/audio/speech cortices (default: true) */
    bool freeze_cognitive;    /**< Freeze ethics/logic/introspection (default: true) */
    bool finetune_classifier; /**< Fine-tune output layer (default: true) */
    uint32_t batch_size;      /**< Batch size (default: 32) */
    bool verbose;             /**< Print training progress (default: true) */
} brain_finetune_config_t;

/**
 * @brief Fine-tune pre-trained model on domain-specific data
 *
 * WHAT: Adapts pre-trained baseline to specific domain with minimal data
 * WHY:  Bridges gap between general baseline and domain requirements
 * HOW:  Selective layer unfreezing + lower learning rate + few-shot learning
 *
 * @param brain Pre-trained brain to fine-tune
 * @param training_data Array of input examples (num_samples × input_dim)
 * @param labels Array of target labels or outputs (num_samples × output_dim)
 * @param num_samples Number of training examples (10-100 for quick adaptation)
 * @param config Fine-tuning configuration (NULL for defaults)
 * @return true on success
 *
 * Example:
 * ```c
 * // Load pre-trained model
 * brain_t brain = brain_create_pretrained("nimcp_baseline_medium",
 *                                         BRAIN_TASK_CLASSIFICATION);
 *
 * // Fine-tune on 50 domain-specific examples (10 minutes)
 * brain_finetune_config_t config = {
 *     .learning_rate = 0.001,
 *     .num_epochs = 5,
 *     .freeze_sensory = true,  // Keep visual/audio frozen
 *     .freeze_cognitive = true, // Keep ethics/logic frozen
 *     .finetune_classifier = true  // Only adapt final layers
 * };
 *
 * brain_finetune(brain, my_data, my_labels, 50, &config);
 *
 * // Save fine-tuned model
 * brain_save(brain, "my_finetuned_model.brain");
 * ```
 *
 * Strategies:
 * - Quick Adaptation (10-100 examples): Freeze all, fine-tune classifier only
 * - Domain Adaptation (100-1000 examples): Unfreeze sensory, fine-tune features
 * - Full Fine-Tuning (1000+ examples): Unfreeze all, low learning rate
 */
bool brain_finetune(brain_t brain, const float* training_data, const float* labels,
                    uint32_t num_samples, const brain_finetune_config_t* config);

/**
 * @brief Model information
 */
typedef struct {
    char model_id[64];           /**< Model identifier */
    char version[16];            /**< Model version (e.g., "v2.7.0") */
    bool is_available;           /**< Model is available locally */
    bool update_available;       /**< Newer version available online */
    char latest_version[16];     /**< Latest available version */
    size_t file_size_bytes;      /**< Model file size */
    char description[256];       /**< Model description */
    char training_date[32];      /**< Training date (ISO 8601) */
} brain_model_info_t;

/**
 * @brief Get information about a pre-trained model
 *
 * @param model_id Model identifier
 * @param info Output model information
 * @return true on success
 */
bool brain_get_model_info(const char* model_id, brain_model_info_t* info);

/**
 * @brief Check if model exists locally
 *
 * @param model_id Model identifier
 * @return true if model is cached locally
 */
bool brain_model_exists(const char* model_id);

/**
 * @brief Download pre-trained model
 *
 * @param model_id Model identifier
 * @return true on success
 */
bool brain_download_model(const char* model_id);

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
 * WHY:  Enable unified processing of visual, audio, language, and direct inputs
 * HOW:  Pass NULL for unused modalities, combine any subset of modalities
 *
 * DESIGN: Flexible multi-modal input - any combination of modalities allowed
 *
 * HUMAN COMMUNICATION USE CASE (Phase 9.4):
 * When communicating with humans, the brain should process ALL available cues:
 * - Visual: Facial expressions, gestures, body language, environmental context
 * - Audio: Tone of voice, prosody, emotion, speech patterns, background sounds
 * - Language: Spoken/written words, semantic meaning, intent, context
 *
 * EXAMPLE: Video call with human
 * - visual_data: Camera frame showing face (expressions, gaze, gestures)
 * - audio_data: Microphone capturing voice (tone, emotion, pitch)
 * - language_text: Transcribed or typed text (semantic content)
 *
 * The brain integrates ALL modalities to understand:
 * - WHAT is being said (language)
 * - HOW it's being said (audio tone/emotion)
 * - NON-VERBAL cues (visual expressions/gestures)
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

    // Language input (optional) - Phase 9.4: Human Communication
    const char* language_text;   /**< Text input (UTF-8 string) */
    uint32_t language_length;    /**< Text length in bytes */
    const float* language_embeddings; /**< Pre-computed embeddings (optional) */
    uint32_t language_embed_dim; /**< Embedding dimension (if pre-computed) */

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

    // Epistemic filtering (Phase 9.2: Bias Prevention)
    float epistemic_quality;     /**< Evidence quality [0,1] */
    float skepticism_score;      /**< Applied skepticism [0,1] */
    float credibility_score;     /**< Claim credibility [0,1] */
    float conspiracy_score;      /**< Conspiracy pattern detection [0,1] */
    bool bias_detected;          /**< Cognitive bias detected */
    bool requires_verification;  /**< Needs further verification */
    char epistemic_reasoning[256]; /**< Explanation of epistemic assessment */

    // Attention breakdown
    float visual_attention;      /**< Visual modality weight [0,1] */
    float audio_attention;       /**< Audio modality weight [0,1] */
    float speech_attention;      /**< Speech modality weight [0,1] - Phase 8.8 */
    float language_attention;    /**< Language modality weight [0,1] - Phase 9.4 */
    float direct_attention;      /**< Direct input weight [0,1] */

    // Language output (Phase 9.4: Human Communication)
    char* language_response;     /**< Generated text response (caller must free) */
    uint32_t language_response_length; /**< Response length in bytes */
    float language_confidence;   /**< Language generation confidence [0,1] */

    // Logical reasoning output (Phase 9.4: Communication)
    bool logical_consistency;    /**< Logical consistency check passed */
    float reasoning_confidence;  /**< Confidence in logical inference [0,1] */
    char logical_reasoning[256]; /**< Logical inference explanation */

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
 * // EXAMPLE 1: Human Communication (Visual + Audio + Language)
 * // Use case: Video call with human - process facial expressions, tone, and words
 *
 * brain_config_t config = brain_default_config("human_comm", BRAIN_SIZE_MEDIUM,
 *                                               BRAIN_TASK_CLASSIFICATION, 512, 128);
 * config.enable_visual_cortex = true;      // Facial expressions, gestures
 * config.enable_audio_cortex = true;       // Tone, emotion, prosody
 * config.enable_language_cortex = true;    // Semantic meaning (Phase 9.4)
 * config.enable_ethics = true;             // Ethical communication
 * config.enable_introspection = true;      // Uncertainty about interpretation
 * config.enable_logic = true;              // Logical reasoning (Phase 9.4)
 * config.enable_epistemic_filter = true;   // Detect bias, misinformation
 * config.enable_wellbeing_monitoring = true; // Self-preservation
 * config.enable_multimodal_integration = true;
 * config.visual_feature_dim = 128;
 * config.audio_feature_dim = 64;
 * config.language_feature_dim = 256;
 *
 * brain_t brain = brain_create_custom(&config);
 *
 * // Capture from video call: camera frame, microphone, transcribed text
 * uint8_t camera_frame[640 * 480 * 3];  // RGB image of person
 * float microphone_samples[1024];        // Audio of voice
 * const char* spoken_text = "I'm feeling really stressed about this project";
 *
 * brain_multimodal_input_t input = {
 *     // Visual cues: facial expression, body language
 *     .visual_data = camera_frame,
 *     .visual_width = 640,
 *     .visual_height = 480,
 *     .visual_channels = 3,  // RGB
 *
 *     // Audio cues: tone, pitch, emotion in voice
 *     .audio_data = microphone_samples,
 *     .audio_samples = 1024,
 *     .audio_channels = 1,
 *
 *     // Language cues: semantic meaning of words
 *     .language_text = spoken_text,
 *     .language_length = strlen(spoken_text),
 *     .language_embeddings = NULL,  // Brain will compute
 *     .language_embed_dim = 0,
 *
 *     .direct_data = NULL,
 *     .direct_dim = 0,
 *     .timestamp_ms = nimcp_time_get_ms()
 * };
 *
 * brain_multimodal_output_t output = {0};
 * output.output_vector = malloc(128 * sizeof(float));
 * output.output_dim = 128;
 *
 * // Process through FULL 7-stage cognitive pipeline:
 * // Stage 0: Wellbeing check (pre-processing)
 * // Stage 1: Introspection
 * // Stage 2: Ethics filtering
 * // Stage 3: Salience detection
 * // Stage 4: Knowledge integration
 * // Stage 5: Curiosity-driven exploration
 * // Stage 6: Wellbeing check (post-processing)
 * brain_process_multimodal(brain, &input, &output);
 *
 * // Brain understood from ALL three modalities:
 * printf("Decision: %s (confidence: %.2f)\n", output.decision_label, output.confidence);
 * printf("Ethical: %s, Bias detected: %s\n",
 *        output.ethical_approved ? "YES" : "NO",
 *        output.bias_detected ? "YES" : "NO");
 *
 * // Attention breakdown shows which modality was most important
 * printf("Attention: Visual=%.2f, Audio=%.2f, Language=%.2f\n",
 *        output.visual_attention, output.audio_attention, output.language_attention);
 *
 * // Generated response (if language output enabled)
 * if (output.language_response) {
 *     printf("Brain response: %s (conf: %.2f)\n",
 *            output.language_response, output.language_confidence);
 *     free(output.language_response);  // Caller must free
 * }
 *
 * printf("Explanation: %s\n", output.explanation);
 * printf("Epistemic: %s\n", output.epistemic_reasoning);
 *
 * // Logical reasoning
 * printf("Logical consistency: %s (conf: %.2f)\n",
 *        output.logical_consistency ? "YES" : "NO",
 *        output.reasoning_confidence);
 * printf("Logical inference: %s\n", output.logical_reasoning);
 *
 * // The brain processed:
 * // - VISUAL: stressed facial expression, tense posture (30% attention)
 * // - AUDIO: anxious tone, rapid speech (40% attention)
 * // - LANGUAGE: words "stressed", "project" (30% attention)
 * // - LOGIC: Inferred "person needs support" from combined cues
 * // = Integrated understanding of human's emotional state + logical inference
 * ```
 *
 * COMPLEXITY:
 * - Visual processing: O(W·H·K²·F) where K=kernel, F=filters
 * - Audio processing: O(N·log(N)) FFT
 * - Language processing: O(T·E) where T=tokens, E=embedding dim
 * - Integration: O(D_v + D_a + D_l + D_d) where D_l=language features
 * - Neural step: O(N·S) where N=neurons, S=avg synapses
 * - Cognitive pipeline (7 stages): O(N + K + C) where K=knowledge, C=cognitive checks
 * - Overall: O(sensory + neural + cognitive)
 *
 * PERFORMANCE: ~10-50ms typical for medium brain with camera+audio+text
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
