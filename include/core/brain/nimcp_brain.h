//=============================================================================
// nimcp_brain.h - High-Level Brain API (Application-Friendly)
//=============================================================================

#ifndef NIMCP_BRAIN_H
#define NIMCP_BRAIN_H

#include "plasticity/adaptive/nimcp_adaptive.h"
#include "networking/distributed/nimcp_distributed_cognition.h"
#include "common/nimcp_export.h"
#include "core/neuron_models/nimcp_neuron_model.h"  // For ode_integration_method_t
#include "information/nimcp_shannon.h"  // Phase C4: Shannon Information Theory
#include "utils/quantum/nimcp_quantum_shannon.h"  // Phase C4.1: Quantum-Shannon diffusion
#include "information/nimcp_cross_modal.h"  // Phase C4.7: Cross-modal information flow

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for opaque types (full headers only in .c file)
// Using void* to avoid typedef conflicts between modules
// The .c file will include headers and cast appropriately

// === PHASE 10: ADVANCED COGNITIVE SYSTEMS ===

// Phase 10.1: Working Memory
typedef struct working_memory working_memory_t;

// Phase M2: Systems Consolidation
typedef struct systems_consolidation_system systems_consolidation_system_t;

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

// Global Workspace Architecture (Global Workspace Theory - Baars, Dehaene)
typedef struct global_workspace_struct* global_workspace_t;  // Opaque pointer

// Neuromodulator System (for mental health interventions)
typedef struct neuromodulator_system_struct* neuromodulator_system_t;  // Opaque pointer

// Phase 9.4: Symbolic Logic & Reasoning
typedef struct symbolic_logic symbolic_logic_t;

// Pink Noise Neuromodulator - typedef defined in plasticity/neuromodulators/nimcp_neuromod_pink_noise.h

// === PHASE 7: ADVANCED SUBSYSTEM ACCESSORS ===
// Forward declarations for opaque pointers used by accessor functions
// Note: glial_integration_t already defined via nimcp_distributed_cognition.h
typedef struct brain_oscillation_analyzer_struct brain_oscillation_analyzer_t;
typedef struct introspection_context_struct* introspection_context_t;
typedef struct ethics_engine_struct* ethics_engine_t;
typedef struct salience_evaluator_struct* salience_evaluator_t;
typedef struct consolidation_handle_struct* consolidation_handle_t;
typedef struct curiosity_engine_struct* curiosity_engine_t;
typedef struct knowledge_system_struct* knowledge_system_t;

// === PHASE E: HIGHER-ORDER COGNITIVE & SOCIAL SYSTEMS ===
// Forward declarations - actual typedefs in respective headers
// Phase E5: Shadow Emotions - defined in cognitive/nimcp_shadow_emotions.h
// Phase E6: Bias Detection - defined in cognitive/nimcp_bias_detection.h

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
 * @brief Forward declaration for brain stats (full definition below)
 */
typedef struct brain_stats_struct brain_stats_t;

/**
 * @brief Astrocyte network statistics
 */
typedef struct {
    uint32_t num_astrocytes;           /**< Total astrocytes in network */
    float avg_calcium_um;              /**< Average calcium level (µM) */
    uint32_t num_tripartite_synapses;  /**< Synapses with astrocyte modulation */
    uint64_t total_modulations;        /**< Total synaptic modulations applied */
    float avg_modulation_strength;     /**< Average modulation factor (0.8-1.2) */
} astrocyte_stats_t;

//=============================================================================
// Persistence Format Versioning
//=============================================================================

/**
 * @brief Brain metadata file format version header
 *
 * WHAT: Version identifier written at start of .meta files
 * WHY:  Enable backward compatibility and format evolution
 * HOW:  Check magic + version on load, support migrations
 *
 * FORMAT: [magic:4 bytes][major:2][minor:2][flags:4][reserved:4]
 * Total: 16 bytes header
 *
 * NOTE: SHA-256 checksum field will be added in v1.1 when CHECKSUM flag is set
 */
typedef struct {
    uint8_t magic[4];       /**< Magic bytes: {'N', 'I', 'M', 'P'} */
    uint16_t version_major; /**< Major version (incompatible changes) */
    uint16_t version_minor; /**< Minor version (compatible changes) */
    uint32_t flags;         /**< Format flags (compressed, encrypted, etc.) */
    uint32_t reserved;      /**< Reserved for future use */
} nimcp_file_header_t;

// Current format version
#define NIMCP_FORMAT_VERSION_MAJOR 1
#define NIMCP_FORMAT_VERSION_MINOR 0

// Header magic bytes
#define NIMCP_MAGIC_0 'N'
#define NIMCP_MAGIC_1 'I'
#define NIMCP_MAGIC_2 'M'
#define NIMCP_MAGIC_3 'P'

// Format flags
#define NIMCP_FORMAT_FLAG_COMPRESSED  0x00000001  /**< Data is compressed */
#define NIMCP_FORMAT_FLAG_ENCRYPTED   0x00000002  /**< Data is encrypted */
#define NIMCP_FORMAT_FLAG_CHECKSUM    0x00000004  /**< File has checksum */

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

    // === PART A: DIFFERENTIAL EQUATIONS & PDEs ===

    /**
     * ODE Integration Method (A1.x)
     *
     * WHAT: Numerical integration algorithm for neuron membrane dynamics
     * WHY: Control accuracy/speed tradeoff for differential equation solving
     * HOW: Passed to neuron update functions during network simulation
     *
     * OPTIONS:
     * - ODE_EULER (default): Fast, first-order accuracy
     *   - Use when: Real-time performance critical, dt < 0.5ms
     *   - Speed: 1.0× (baseline)
     *
     * - ODE_RK2: Balanced, second-order accuracy
     *   - Use when: Good accuracy needed, dt = 0.5-2ms
     *   - Speed: ~0.5× (2× slower than Euler)
     *
     * - ODE_RK4: Best accuracy, fourth-order
     *   - Use when: Scientific precision required, dt = 1-5ms
     *   - Speed: ~0.25× (4× slower than Euler)
     *
     * BACKWARD COMPATIBILITY:
     * - Default value is ODE_EULER (0)
     * - Existing code continues to use Euler method
     * - No breaking changes to API or behavior
     *
     * EXAMPLE:
     * ```c
     * brain_config_t config = brain_default_config(...);
     * config.neuron_integration = ODE_RK4;  // High accuracy
     * brain_t brain = brain_create_custom(&config);
     * ```
     */
    ode_integration_method_t neuron_integration;  /**< Neuron ODE integration method (default: ODE_EULER) */

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

    // Attention Mechanism
    bool enable_multihead_attention;  /**< Enable attention mechanism for selective processing */
    uint32_t num_attention_heads;     /**< Number of attention heads (default: 8) */
    uint32_t attention_key_dim;       /**< Key/query dimension (default: 64) */
    bool enable_thalamic_gate;        /**< Enable thalamic gating for top-down control */
    bool enable_salience_weighting;   /**< Enable salience-based attention */

    // Brain Regions Architecture (hierarchical cortical organization)
    bool enable_brain_regions;        /**< Enable modular brain regions with cortical layers */
    uint32_t num_brain_regions;       /**< Number of brain regions to create (default: 4) */
    uint32_t neurons_per_region;      /**< Neurons per region (default: 1000) */

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

    // Global Workspace Architecture (Global Workspace Theory)
    bool enable_global_workspace;     /**< Enable Global Workspace for conscious access */
    uint32_t workspace_capacity_dim;  /**< Workspace content dimension (default: 256) */
    float workspace_ignition_threshold; /**< Ignition threshold for conscious access (default: 0.6) */
    uint32_t workspace_refractory_ms; /**< Refractory period between broadcasts in ms (default: 50) */
    bool workspace_enable_history;    /**< Enable workspace history tracking (default: true) */
    uint32_t workspace_history_depth; /**< History buffer depth (default: 10) */

    // === PART B: GEOMETRIC METHODS ===

    /**
     * Hyperbolic Knowledge Embeddings (B1.1)
     *
     * WHAT: Store knowledge in Poincaré hyperbolic space
     * WHY: 200x memory reduction vs Euclidean embeddings
     * HOW: Hierarchical concepts naturally embed in hyperbolic geometry
     *
     * PERFORMANCE:
     * - Memory: 200x reduction (10K concepts: 40MB → 200KB)
     * - Speed: 0.9x (slightly slower distance computation)
     * - Accuracy: Better for hierarchical data
     *
     * SYNERGY:
     * - Combines with MPS (Part C3.1) for 20,000x total memory reduction
     * - Compatible with RK4 integration (Part A1.1)
     *
     * EXAMPLE:
     * ```c
     * config.use_hyperbolic_knowledge = true;
     * config.hyperbolic_curvature = -1.0f;  // Standard Poincaré disk
     * config.hyperbolic_embedding_dim = 32;  // vs 6400 for Euclidean!
     * ```
     */
    bool use_hyperbolic_knowledge;     /**< Use hyperbolic embeddings for knowledge (B1.1) */
    float hyperbolic_curvature;        /**< Curvature constant (default: -1.0) */
    uint32_t hyperbolic_embedding_dim; /**< Embedding dimension (default: 32) */

    /**
     * Riemannian Gradient Descent (B2.1)
     *
     * WHAT: Natural gradient descent on learned manifold structure
     * WHY: 2-10x faster convergence than standard gradient descent
     * HOW: Follow geodesics on parameter space manifold
     *
     * PERFORMANCE:
     * - Speed: 2-10x faster convergence (fewer iterations)
     * - Memory: 1.1x overhead (Fisher info matrix approximation)
     * - Accuracy: Better, follows natural parameter space
     *
     * EXAMPLE:
     * ```c
     * config.use_natural_gradient = true;
     * config.fisher_damping = 1e-4f;  // Numerical stability
     * ```
     */
    bool use_natural_gradient;         /**< Use natural gradient descent (B2.1) */
    float fisher_damping;              /**< Fisher info matrix damping (default: 1e-4) */

    /**
     * Manifold Structure Learning (B3.1)
     *
     * WHAT: Learn geometric structure of data manifold
     * WHY: Better generalization, faster learning
     * HOW: Estimate local metric tensor from data
     *
     * EXAMPLE:
     * ```c
     * config.learn_manifold_structure = true;
     * config.manifold_neighborhood_size = 10;
     * ```
     */
    bool learn_manifold_structure;     /**< Learn data manifold structure (B3.1) */
    uint32_t manifold_neighborhood_size; /**< Neighborhood size for manifold learning (default: 10) */

    // === PART C: QUANTUM-INSPIRED ALGORITHMS ===

    /**
     * Matrix Product States (MPS) Weight Compression (C3.1)
     *
     * WHAT: Compress neural network weights using tensor network decomposition
     * WHY: 10-100x memory reduction with <1% accuracy loss
     * HOW: Decompose weight matrices into chains of small tensors
     *
     * ALGORITHM:
     * - Weight matrix W[N×M] → MPS chain: A[1] · A[2] · ... · A[k]
     * - Each A[i] has size: (bond_dim × bond_dim × phys_dim)
     * - Total parameters: O(k × bond_dim² × phys_dim) << O(N×M)
     *
     * PERFORMANCE:
     * - Memory: 10-100x reduction (depends on bond_dim)
     *   - bond_dim=5:  50-100x compression, >98% accuracy
     *   - bond_dim=10: 10-20x compression, >99% accuracy
     *   - bond_dim=20: 5-10x compression, >99.9% accuracy
     * - Speed: 0.4x (slower matrix-vector multiply)
     * - Accuracy: Controlled by bond_dim
     *
     * SYNERGY WITH OTHER PHASES:
     * - A1.1 (RK4): Compress weights, accurate dynamics → 100x memory, 10x accuracy
     * - B1.1 (Hyperbolic): 200x × 100x = 20,000x total memory reduction
     * - Compatible with all plasticity mechanisms (STDP, BCM, eligibility traces)
     *
     * USE CASES:
     * - Large networks (100K+ synapses) with limited memory
     * - Embedded systems requiring compact models
     * - Research mode with maximum biological realism
     *
     * TRADE-OFFS:
     * - Higher bond_dim = more memory, better accuracy
     * - Lower bond_dim = less memory, faster, slight accuracy loss
     *
     * EXAMPLE:
     * ```c
     * // Recommended: Balanced compression
     * config.use_mps_weights = true;
     * config.mps_bond_dimension = 10;          // 10-20x compression
     * config.mps_adaptive_bond_dim = true;     // Optimize per-synapse
     *
     * // High compression (embedded systems)
     * config.mps_bond_dimension = 5;           // 50-100x compression
     *
     * // High accuracy (research)
     * config.mps_bond_dimension = 20;          // 5-10x compression, >99.9% accuracy
     * ```
     */
    bool use_mps_weights;              /**< Use MPS tensor compression for weights (C3.1) */
    uint32_t mps_bond_dimension;       /**< MPS bond dimension (default: 10) */
    bool mps_adaptive_bond_dim;        /**< Adapt bond dimension per synapse (default: true) */
    float mps_svd_tolerance;           /**< SVD truncation tolerance (default: 1e-6) */

    /**
     * Quantum Annealing for Network Optimization (C1.1)
     *
     * WHAT: Use quantum annealing for discrete optimization problems
     * WHY: Find better local optima than gradient descent alone
     * HOW: Simulated quantum annealing with transverse field
     *
     * USE CASES:
     * - Network topology optimization
     * - Discrete synapse pruning decisions
     * - Combinatorial cognitive tasks
     *
     * EXAMPLE:
     * ```c
     * config.enable_quantum_annealing = true;
     * config.annealing_temperature_init = 10.0f;
     * config.annealing_temperature_final = 0.1f;
     * ```
     */
    bool enable_quantum_annealing;     /**< Use quantum annealing for optimization (C1.1) */
    float annealing_temperature_init;  /**< Initial annealing temperature (default: 10.0) */
    float annealing_temperature_final; /**< Final annealing temperature (default: 0.1) */
    uint32_t annealing_steps;          /**< Number of annealing steps (default: 1000) */
    uint32_t quantum_annealing_frequency; /**< Run annealing every N learning steps (default: 100) */

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

    // === BIOLOGICAL SECURITY (Phase 11) ===
    /**
     * Biological Attack Defense
     *
     * WHAT: Enable runtime monitoring for biological attacks
     * WHY:  Protect against excitotoxicity, synaptic poisoning, neuromodulator hijacking
     * HOW:  Monitor activity levels, validate weight changes, rate-limit neuromodulators
     *
     * ATTACK TYPES DEFENDED:
     * - Excitotoxicity: Runaway excitation (>95% neurons active)
     * - Synaptic poisoning: Malicious weight updates (>10% per step)
     * - Neuromodulator hijacking: Dopamine manipulation (>20% per step)
     * - Homeostatic bypass: Mass BCM/eligibility disable (>10% synapses)
     *
     * PERFORMANCE IMPACT:
     * - Activity monitoring: +5% overhead (O(N) per forward pass)
     * - Weight validation: +2% overhead (O(1) per weight update)
     * - Total: ~7% performance overhead when enabled
     *
     * RECOMMENDED: Enable for production, disable for benchmarking
     *
     * EXAMPLE:
     * ```c
     * config.enable_bio_security = true;
     * config.activity_warning_threshold = 0.8f;  // 80% activity
     * config.activity_danger_threshold = 0.95f;  // 95% activity
     * config.max_weight_delta = 0.1f;            // 10% per step
     * config.max_neuromod_rate = 0.2f;           // 20% per step
     * ```
     */
    bool enable_bio_security;            /**< Enable biological security monitoring (default: true) */
    float activity_warning_threshold;    /**< Warning threshold for network activity (default: 0.8) */
    float activity_danger_threshold;     /**< Danger threshold for emergency response (default: 0.95) */
    float max_weight_delta_per_step;     /**< Maximum weight change per step (default: 0.1) */
    float max_neuromod_rate_per_step;    /**< Maximum neuromodulator change rate (default: 0.2) */
    float max_plasticity_disable_ratio;  /**< Alert if >X ratio synapses disabled (default: 0.1) */
    bool emergency_inhibit_on_attack;    /**< Activate emergency inhibition on attack (default: true) */

    // === MULTI-GPU SUPPORT (Phase 11) ===
    /**
     * Multi-GPU Distributed Computation
     *
     * WHAT: Enable transparent multi-GPU execution for large networks
     * WHY:  Scale beyond single GPU memory/compute limitations
     * HOW:  Partition network across GPUs, coordinate computation, sync results
     *
     * WHEN TO ENABLE:
     * - Networks > 1M neurons
     * - Multiple GPUs available
     * - Need faster training/inference
     *
     * PERFORMANCE:
     * - Ideal speedup: N×GPUs (with good load balance)
     * - Communication overhead: 5-15%
     * - Memory per GPU: total_memory / num_gpus
     *
     * USAGE:
     * ```c
     * config.enable_multi_gpu = true;
     * config.multi_gpu_device_count = 4;       // Use 4 GPUs (0=all)
     * config.multi_gpu_partition_strategy = MULTIGPU_PARTITION_HYBRID;
     * config.enable_peer_to_peer = true;       // 3-5x faster transfers
     * ```
     */
    bool enable_multi_gpu;                /**< Enable multi-GPU distributed computation (default: false) */
    uint32_t multi_gpu_device_count;      /**< Number of GPUs to use (0=all available, default: 0) */
    uint32_t multi_gpu_partition_strategy; /**< Partitioning strategy (0=layer, 1=neuron, 2=hybrid, default: 2) */
    bool enable_peer_to_peer;             /**< Enable P2P GPU transfers (3-5x faster, default: true) */
    bool multi_gpu_verbose_logging;       /**< Log multi-GPU operations (default: false) */

    // === PHASE 12: PERSONALITY AND IDENTITY ===

    /**
     * Personality and Identity Configuration (Phase 12)
     *
     * WHAT: Configure unique personality, gender, and sexual identity
     * WHY:  Each brain should be a unique individual with distinct traits
     * HOW:  Specify explicit traits or generate randomly
     *
     * BEHAVIORAL EFFECTS:
     * - Personality traits modulate cognitive processes
     * - High openness → increased curiosity and exploratory learning
     * - High conscientiousness → better planning and goal persistence
     * - High extraversion → more social interactions and higher arousal
     * - High agreeableness → more empathetic and cooperative responses
     * - High neuroticism → stronger negative emotions and stress sensitivity
     *
     * IDENTITY INTEGRATION:
     * - Gender identity affects self-model and pronoun usage
     * - Sexual orientation affects social cognition and relationship models
     * - Identity certainty affects self-concept stability
     *
     * USAGE:
     * ```c
     * // Option 1: Random personality (default: female, random traits)
     * config.use_random_personality = true;
     * config.personality_seed = 12345;  // Reproducible
     *
     * // Option 2: Specify explicit traits
     * config.use_random_personality = false;
     * config.explicit_openness = 0.8f;  // Creative
     * config.explicit_conscientiousness = 0.6f;  // Organized
     * config.explicit_extraversion = 0.7f;  // Social
     * config.explicit_agreeableness = 0.9f;  // Compassionate
     * config.explicit_neuroticism = 0.3f;  // Emotionally stable
     * config.explicit_gender = 1;  // Female (0=male, 1=female, 2=non-binary)
     * config.explicit_sexuality = 2;  // Bisexual
     * ```
     */
    bool use_random_personality;          /**< Generate random personality (default: true) */
    uint32_t personality_seed;            /**< RNG seed for personality (0=time-based) */

    // Explicit personality specification (only used if use_random_personality = false)
    float explicit_openness;              /**< [0-1] Openness to experience */
    float explicit_conscientiousness;     /**< [0-1] Organization and discipline */
    float explicit_extraversion;          /**< [0-1] Sociability and energy */
    float explicit_agreeableness;         /**< [0-1] Compassion and cooperation */
    float explicit_neuroticism;           /**< [0-1] Emotional sensitivity */
    uint32_t explicit_gender;             /**< Gender identity (0=male, 1=female, 2=non-binary, etc.) */
    uint32_t explicit_sexuality;          /**< Sexual orientation (0=hetero, 1=homo, 2=bi, etc.) */

    // Personality generation configuration
    float personality_trait_mean;         /**< Mean for random traits (default: 0.5) */
    float personality_trait_stddev;       /**< Stddev for random traits (default: 0.15) */
    float female_probability;             /**< P(female) when random (default: 1.0) */
    float male_probability;               /**< P(male) when random (default: 0.0) */
    float non_binary_probability;         /**< P(non-binary) when random (default: 0.0) */

    // === PHASE C2.1: QUANTUM WALKS FOR NEUROMODULATOR DIFFUSION ===
    /**
     * Quantum Walk Configuration
     *
     * WHAT: Quantum random walks for O(√N) speedup in neuromodulator diffusion
     * WHY:  Classical diffusion is O(d²), quantum walk is O(d) for distance d
     * HOW:  Replace/augment classical diffusion with quantum walk evolution
     *
     * PERFORMANCE IMPACT:
     * - Quadratic speedup for neuromodulator propagation
     * - 10x faster dopamine/serotonin spread in large networks
     * - Memory overhead: 2x (complex amplitudes)
     *
     * BIOLOGICAL INTERPRETATION:
     * - Models rapid mood/attention shifts
     * - Captures non-local neuromodulator effects
     * - Explains fast reward prediction error propagation
     */
    bool enable_quantum_walk_diffusion;   /**< Use quantum walks for neuromodulator diffusion (default: false) */
    uint32_t quantum_walk_steps;          /**< Steps per diffusion update (default: 50) */
    float quantum_classical_mixing;       /**< Hybrid mixing ratio [0=pure quantum, 1=classical] (default: 0.2) */
    uint32_t quantum_coin_type;           /**< Coin operator: 0=Hadamard, 1=Grover, 2=Fourier (default: 0) */
    float quantum_decoherence_rate;       /**< Decoherence strength [0=none, 1=instant classical] (default: 0.05) */

    // === PHASE C2.2: COMPLEX OSCILLATION TRACKING ===
    /**
     * Complex Oscillation Support
     *
     * WHAT: Phasor-based phase and amplitude tracking per neuron
     * WHY:  Phase relationships encode binding, memory, spatial information
     * HOW:  Track neural_phasor_t (complex number) for each neuron
     *
     * FEATURES:
     * - Instantaneous phase and amplitude per neuron
     * - Phase coherence across neuron populations
     * - Phase-amplitude coupling (PAC) detection
     * - Cross-frequency phase relationships
     *
     * NEUROSCIENCE:
     * - Hippocampal place cells: theta phase = position
     * - Working memory: gamma phase = item order
     * - Grid cells: phase interference patterns
     * - Theta-gamma PAC: memory encoding/retrieval
     *
     * PERFORMANCE:
     * - Memory overhead: 8 bytes/neuron (2x float)
     * - Update cost: ~10ns/neuron (phase increment)
     * - Coherence: ~0.8µs for 1000 neurons
     *
     * INTEGRATION:
     * - Builds on nimcp_brain_oscillations.h (frequency bands)
     * - Uses nimcp_complex_math.h (phasor operations)
     * - Opt-in: disabled by default for backward compatibility
     */
    bool complex_oscillation_enabled;     /**< Enable complex phasor tracking (default: false) */
    float complex_phase_update_rate;      /**< Phase increment per step in radians (default: 0.1) */
    float complex_amplitude_decay;        /**< Amplitude decay factor per step (default: 0.95) */
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
// Phase 2.8: Dynamic Brain Resizing
//=============================================================================

/**
 * @brief Resize brain to new neuron count
 *
 * WHAT: Expand brain capacity while preserving all learned knowledge
 * WHY:  Enable continuous learning without hitting capacity limits
 * HOW:  Create new network, transfer neurons/weights, swap atomically
 *
 * GUARANTEES:
 * - Zero knowledge loss: All weights, biases, states preserved exactly
 * - Atomic operation: Network swap is instantaneous
 * - Memory safe: Old network destroyed only after successful transfer
 *
 * @param brain Brain to resize
 * @param new_neuron_count New total neuron count (must be > current)
 * @return true on success, false on error (brain unchanged)
 */
bool brain_resize(brain_t brain, uint32_t new_neuron_count);

/**
 * @brief Internal helper for brain_resize - update subsystems after network swap
 *
 * WHAT: Updates glial integration and other subsystems to reference new network
 * WHY:  brain_resize.c can't access full brain struct, so needs helper in main module
 * HOW:  Destroys/recreates glial integration with new network reference
 *
 * @param brain Brain handle
 * @param new_base_network New neural network after resize
 * @param new_neuron_count New neuron count after resize
 * @return true on success
 *
 * NOTE: This is an internal function only called by brain_resize()
 */
bool brain_resize_update_subsystems_internal(brain_t brain, neural_network_t new_base_network, uint32_t new_neuron_count);

/**
 * @brief Auto-resize brain based on utilization metrics
 *
 * WHAT: Automatically grow brain when capacity is saturated
 * WHY:  Enable continuous learning without manual intervention
 * HOW:  Evaluate metrics, decide if growth needed, resize if so
 *
 * GROWTH TRIGGERS:
 * - High utilization: >90% neurons active for >1000 steps
 * - Weight saturation: >80% weights near ±1.0
 *
 * @param brain Brain to potentially resize
 * @return true if resize occurred, false if no resize needed
 */
bool brain_auto_resize(brain_t brain);

/**
 * @brief Get brain current neuron count
 *
 * WHAT: Return current brain capacity in neurons
 * WHY:  Allow monitoring of brain size for metrics/logging
 *
 * @param brain Brain to query
 * @return Neuron count, or 0 on error
 */
uint32_t brain_get_neuron_count(brain_t brain);

/**
 * @brief Get brain utilization metrics
 *
 * WHAT: Return current capacity utilization statistics
 * WHY:  Enable monitoring and debugging of auto-resize logic
 *
 * @param brain Brain to analyze
 * @param utilization Output: neuron utilization ratio [0.0, 1.0]
 * @param saturation Output: weight saturation ratio [0.0, 1.0]
 * @return true on success, false on error
 */
bool brain_get_utilization_metrics(brain_t brain, float* utilization, float* saturation);

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
 * @brief Get global workspace from brain
 *
 * WHAT: Retrieve pointer to brain's global workspace subsystem
 * WHY:  Allow cognitive modules to access workspace for competition and broadcasting
 * HOW:  Return brain->global_workspace field
 *
 * @param brain Brain instance
 * @return Global workspace pointer or NULL if not enabled
 */
global_workspace_t* brain_get_global_workspace(brain_t brain);

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
 * @brief Apply reward-based reinforcement learning to all synapses
 *
 * WHAT: Apply eligibility-trace-based learning with reward signal
 * WHY:  Enable temporal credit assignment and reinforcement learning
 * HOW:  Use eligibility traces to propagate reward to recently active synapses
 *
 * BIOLOGY: Implements three-factor learning rule (Hebbian + Reward + Dopamine)
 * - Eligibility traces mark recently active synapses ("synaptic tags")
 * - Dopamine bursts trigger consolidation ("capture")
 * - Reward signal modulates weight changes
 *
 * @param brain Brain handle
 * @param reward Reward signal (0-1 for positive, -1-0 for punishment)
 * @return Number of synapses modified
 */
uint32_t brain_apply_reward_learning(brain_t brain, float reward);

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
 * @brief Observe action performed by another agent (Phase 10.11)
 *
 * WHAT: Record observed action in mirror neuron system for observational learning
 * WHY:  Enable learning from watching others (imitation, social cognition)
 * HOW:  Convert input features to observed action and send to mirror neurons
 *
 * This is the OBSERVATION PATHWAY for mirror neurons. When the brain observes
 * another agent performing an action, this function records it for learning.
 *
 * USE CASES:
 * - Robot watching human demonstration
 * - Agent observing another agent's behavior
 * - Learning from video/sensor data of actions
 * - Social learning and imitation
 *
 * COMPLEXITY: O(n) where n = num_features
 * THREAD-SAFE: No (requires external synchronization)
 *
 * @param brain Brain handle
 * @param features Observed action features (sensor data, visual features, etc.)
 * @param num_features Number of features
 * @param agent_id ID of agent being observed (must be > 0, as 0 = self)
 * @return true on success, false on error
 */
bool brain_observe_action(brain_t brain, const float* features, uint32_t num_features,
                          uint32_t agent_id);

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

/**
 * @brief Simple prediction/inference function
 *
 * Lightweight wrapper around brain_process_multimodal for simple prediction tasks.
 * For full multimodal capabilities, use brain_process_multimodal directly.
 *
 * @param brain Brain handle
 * @param input Input feature vector
 * @param input_size Number of input features
 * @param output Output vector (allocated by caller)
 * @param output_size Size of output vector
 * @return true on success, false on error
 */
bool brain_predict(brain_t brain, const float* input, uint32_t input_size,
                  float* output, uint32_t output_size);

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
struct brain_stats_struct {
    char task_name[64];           /**< Brain name */
    brain_size_t size;            /**< Size preset */
    uint32_t num_neurons;         /**< Total neurons */
    uint32_t num_synapses;        /**< Total synapses */
    uint32_t num_active_synapses; /**< Non-pruned synapses */

    uint64_t total_inferences;     /**< Inference count */
    uint64_t total_learning_steps; /**< Learning steps */
    uint64_t quantum_annealing_runs; /**< Quantum annealing optimization runs */

    float avg_sparsity;          /**< Average sparsity */
    float avg_inference_time_us; /**< Avg inference (μs) */
    float current_learning_rate; /**< Current learning rate */

    float accuracy;      /**< Validation accuracy */
    size_t memory_bytes; /**< Memory usage */
};

/**
 * @brief Get brain statistics
 *
 * @param brain Brain handle
 * @param stats Output statistics
 * @return true on success
 */
bool brain_get_stats(brain_t brain, brain_stats_t* stats);

/**
 * @brief Mark brain as a snapshot with preserved stats
 *
 * WHAT: Sets snapshot flag and preserves current stats
 * WHY:  Snapshots should preserve stats at snapshot time, not reflect future changes
 * HOW:  Stores current stats in brain->snapshot_stats
 *
 * @param brain Brain to mark as snapshot
 * @param stats Stats to preserve
 */
void brain_mark_as_snapshot(brain_t brain, const brain_stats_t* stats);

/**
 * @brief Get number of input features for this brain
 *
 * @param brain Brain handle
 * @return Number of input features, or 0 if brain is NULL
 */
uint32_t brain_get_num_inputs(brain_t brain);

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
 * @brief Get systems consolidation subsystem
 *
 * WHAT: Access the brain's systems consolidation component
 * WHY:  Allow other modules (e.g., mental health) to interact with memory consolidation
 * HOW:  Return pointer to systems consolidation subsystem
 *
 * @param brain Brain handle
 * @return Pointer to systems consolidation, or NULL if brain is NULL or consolidation not initialized
 */
systems_consolidation_system_t* brain_get_systems_consolidation(brain_t brain);

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

/**
 * @brief Get neuromodulator system from brain
 *
 * WHAT: Accessor for brain's neuromodulator system
 * WHY:  Mental health monitoring needs to read/write neurotransmitter levels
 * HOW:  Returns opaque handle to neuromodulator system
 *
 * @param brain Brain handle
 * @return Neuromodulator system handle (do not free!), or NULL if not initialized
 */
neuromodulator_system_t brain_get_neuromodulator_system(brain_t brain);

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

    // Phase 11: Cognitive Module Outputs (when relevant)
    // Global Workspace outputs (when broadcast active)
    bool has_workspace_broadcast;         /**< Is global workspace broadcasting? */
    uint8_t workspace_source_module;      /**< Which module is broadcasting (cognitive_module_t) */
    float workspace_broadcast_strength;   /**< Strength of current broadcast [0,1] */
    uint32_t workspace_num_competitors;   /**< Number of modules competing for workspace */

    // Executive Function outputs (when active)
    uint32_t working_memory_items;        /**< Number of items in working memory */
    float working_memory_utilization;     /**< Working memory load [0,1] */
    char top_wm_item_description[128];    /**< Description of most salient WM item */

    // Theory of Mind outputs (when agent detected)
    bool has_mental_state_inference;      /**< Did ToM infer mental states? */
    char inferred_belief[128];            /**< Inferred agent belief */
    char inferred_intention[128];         /**< Inferred agent intention */
    float tom_confidence;                 /**< Confidence in ToM inference [0,1] */

    // Curiosity outputs (when exploration triggered)
    float curiosity_drive;                /**< Current exploration drive [0,1] */
    bool exploration_triggered;           /**< Should explore this input? */
    char curiosity_reason[128];           /**< Why exploration is recommended */

    // Predictive Processing outputs (when prediction available)
    bool has_prediction;                  /**< Is prediction available? */
    float prediction_error;               /**< Mismatch between prediction and actual [0,1] */
    float prediction_confidence;          /**< Confidence in prediction [0,1] */

    // Knowledge outputs (when facts retrieved)
    bool has_knowledge_retrieval;         /**< Were facts retrieved? */
    uint32_t num_facts_retrieved;         /**< Number of knowledge facts accessed */
    char retrieved_concept[64];           /**< Most relevant retrieved concept */

    // NLP outputs (when language processed)
    bool has_nlp_interpretation;          /**< Was language interpreted? */
    char nlp_intent[64];                  /**< Detected intent (e.g., "question", "command") */
    char nlp_sentiment[32];               /**< Sentiment (e.g., "positive", "negative", "neutral") */
    float nlp_comprehension_score;        /**< Language comprehension quality [0,1] */

    // Emotion Recognition outputs (Phase 11: Part I.1)
    bool has_emotion_detected;            /**< Was emotion detected from input? */
    char detected_emotion[32];            /**< Detected emotion name (e.g., "anger", "fear") */
    float emotion_confidence;             /**< Emotion detection confidence [0,1] */
    float emotion_valence;                /**< Emotional valence (-1=negative, +1=positive) */
    float emotion_arousal;                /**< Emotional arousal (0=calm, 1=excited) */
    float emotion_intensity;              /**< Emotion intensity [0,1] */
    bool emotion_is_negative;             /**< Is negative emotion requiring support? */

    // Empathetic Response outputs (Phase 11: Part I.2)
    bool has_empathetic_response;         /**< Was empathetic response generated? */
    char empathetic_response[1024];       /**< Generated empathetic response text */
    float empathy_score;                  /**< Predicted empathy of response [0,1] */
    bool requires_human_escalation;       /**< Crisis detected - hand-off to human */
    char escalation_reason[256];          /**< Why escalation needed (if applicable) */
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

/**
 * @brief Get symbolic logic system
 * @param brain Brain handle
 * @return Symbolic logic handle or NULL if not initialized
 */
symbolic_logic_t* brain_get_symbolic_logic(brain_t brain);

/**
 * @brief Get pink noise neuromodulator
 * @param brain Brain handle
 * @return Pink noise handle or NULL if not initialized
 */
void* brain_get_pink_noise(brain_t brain);

/**
 * @brief Get glial integration subsystem
 * @param brain Brain handle
 * @return Glial integration pointer or NULL if not initialized
 */
glial_integration_t* brain_get_glial(brain_t brain);

/**
 * @brief Get brain oscillation analyzer subsystem
 * @param brain Brain handle
 * @return Brain oscillation analyzer pointer or NULL if not initialized
 */
brain_oscillation_analyzer_t* brain_get_oscillations(brain_t brain);

/**
 * @brief Get introspection subsystem
 * @param brain Brain handle
 * @return Introspection context handle or NULL if not initialized
 */
introspection_context_t brain_get_introspection(brain_t brain);

/**
 * @brief Get ethics engine subsystem
 * @param brain Brain handle
 * @return Ethics engine handle or NULL if not initialized
 */
ethics_engine_t brain_get_ethics(brain_t brain);

/**
 * @brief Get salience evaluator subsystem
 * @param brain Brain handle
 * @return Salience evaluator handle or NULL if not initialized
 */
salience_evaluator_t brain_get_salience(brain_t brain);

/**
 * @brief Get consolidation subsystem
 * @param brain Brain handle
 * @return Consolidation handle or NULL if not initialized
 */
consolidation_handle_t brain_get_consolidation(brain_t brain);

/**
 * @brief Get curiosity engine subsystem
 * @param brain Brain handle
 * @return Curiosity engine handle or NULL if not initialized
 */
curiosity_engine_t brain_get_curiosity(brain_t brain);

/**
 * @brief Get knowledge system subsystem
 * @param brain Brain handle
 * @return Knowledge system handle or NULL if not initialized
 */
knowledge_system_t brain_get_knowledge(brain_t brain);

/**
 * @brief Get mirror neuron activations for Theory of Mind integration
 *
 * WHAT: Extract current mirror neuron activation pattern
 * WHY:  Enable ToM to infer agent intentions from mirror neuron activity
 * HOW:  Query mirror neuron system and return activation array
 *
 * BIOLOGICAL RATIONALE:
 * Mirror neurons fire when observing actions, enabling understanding of others'
 * intentions (Rizzolatti & Craighero, 2004). Theory of Mind uses these activations
 * to infer mental states: "What action are they performing, and why?"
 *
 * Use case: External systems can use these activations to:
 * - Infer observed agent's intentions
 * - Compute empathy responses
 * - Predict next actions in social contexts
 * - Enable social learning from observation
 *
 * @param brain Brain handle
 * @param activations Output buffer for activation values (must have space for max_size floats)
 * @param max_size Maximum number of activations to return (buffer size)
 * @param out_size Output: actual number of activations written
 * @return true on success, false on error (invalid params or mirror neurons not enabled)
 *
 * COMPLEXITY: O(n) where n = num_mirror_neuron_actions
 * THREAD-SAFE: Yes (read-only operation)
 */
bool brain_get_mirror_activations(brain_t brain, float* activations,
                                  uint32_t max_size, uint32_t* out_size);

/**
 * @brief Compute empathy response from mirror neuron activations
 *
 * WHAT: Generate empathetic emotional response based on observed actions
 * WHY:  Mirror neurons enable emotional contagion and empathy (Preston & de Waal, 2002)
 * HOW:  Map mirror neuron activation pattern → inferred emotion → empathy response
 *
 * BIOLOGICAL RATIONALE:
 * Empathy arises from mirror neuron activation during observation of others' actions
 * and emotional expressions. This function models the pathway:
 * Visual observation → Mirror neuron activation → Emotion inference → Empathetic response
 *
 * @param brain Brain handle
 * @param observed_features Features representing observed behavior
 * @param num_features Number of features
 * @param empathy_valence Output: empathy valence (-1.0 to 1.0, negative=distress, positive=joy)
 * @param empathy_arousal Output: empathy arousal (0.0 to 1.0, how strong the empathy)
 * @param empathy_confidence Output: confidence in empathy response (0.0 to 1.0)
 * @return true on success, false on error
 *
 * COMPLEXITY: O(n) where n = num_features
 * THREAD-SAFE: Yes (read-only operation)
 */
bool brain_compute_empathy(brain_t brain, const float* observed_features,
                          uint32_t num_features, float* empathy_valence,
                          float* empathy_arousal, float* empathy_confidence);

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
 * - spike_nlp_t* brain_get_spike_nlp(brain_t brain);
 *
 * These will be uncommented and implemented once module initialization is complete.
 */

// Shannon Information Theory API (Phase C4)
//=============================================================================

/**
 * @brief Enable Shannon information flow monitoring
 *
 * WHAT: Activate real-time Shannon metrics during learning/inference
 * WHY:  Monitor channel capacity, detect bottlenecks, optimize information flow
 * HOW:  Sets enable_shannon_monitoring flag in brain
 *
 * PERFORMANCE IMPACT: ~5-10% overhead during learning/inference
 *
 * @param brain Brain handle
 * @param enable true to enable, false to disable
 */
void brain_enable_shannon_monitoring(brain_t brain, bool enable);

/**
 * @brief Get last Shannon network metrics
 *
 * WHAT: Retrieve most recent Shannon analysis results
 * WHY:  Allow external monitoring of information flow characteristics
 * HOW:  Returns copy of last_shannon_metrics from brain
 *
 * @param brain Brain handle
 * @param metrics Output metrics structure
 * @return true on success, false on error
 */
bool brain_get_shannon_metrics(brain_t brain, shannon_network_metrics_t* metrics);

/**
 * @brief Set custom Shannon configuration
 *
 * WHAT: Override default Shannon analysis parameters
 * WHY:  Tune accuracy vs performance tradeoff
 * HOW:  Updates brain->shannon_config
 *
 * @param brain Brain handle
 * @param config Custom Shannon configuration
 */
void brain_set_shannon_config(brain_t brain, const shannon_config_t* config);

//=============================================================================
// Phase C4.1: Quantum-Shannon Diffusion API
//=============================================================================

/**
 * @brief Enable quantum-Shannon accelerated diffusion
 *
 * WHAT: Activate √N speedup quantum walk diffusion with Shannon monitoring
 * WHY:  Quadratic speedup for neuromodulator propagation + real-time bottleneck detection
 * HOW:  Creates quantum_shannon_diffusion_t on brain network
 *
 * PERFORMANCE IMPACT: 2-50x speedup (topology dependent), 3× memory overhead
 *
 * @param brain Brain handle
 * @param enable true to enable, false to disable
 * @param source_neuron_id Initial source neuron (0 = auto-select middle neuron)
 * @param source_information_bits Initial information content (default: 10.0 bits)
 * @return true on success, false on error
 */
bool brain_enable_quantum_shannon_diffusion(brain_t brain, bool enable, uint32_t source_neuron_id, float source_information_bits);

/**
 * @brief Set quantum-Shannon mixing ratio
 *
 * WHAT: Control quantum vs classical diffusion blend
 * WHY:  Tune performance vs accuracy tradeoff
 * HOW:  Sets mixing_ratio [0=pure quantum, 1=pure classical]
 *
 * @param brain Brain handle
 * @param mixing_ratio Mix ratio [0.0-1.0]
 */
void brain_set_quantum_shannon_mixing(brain_t brain, float mixing_ratio);

/**
 * @brief Set quantum-Shannon evolution steps
 *
 * WHAT: Control how many quantum steps per diffusion update
 * WHY:  More steps = better spreading, but slower
 * HOW:  Sets evolution_steps parameter
 *
 * @param brain Brain handle
 * @param steps Number of steps (10-1000, default: 100)
 */
void brain_set_quantum_shannon_steps(brain_t brain, uint32_t steps);

/**
 * @brief Get last quantum-Shannon diffusion metrics
 *
 * WHAT: Retrieve most recent Shannon metrics from quantum diffusion
 * WHY:  Monitor speedup, bottlenecks, and information flow
 * HOW:  Returns last_quantum_shannon_metrics from brain
 *
 * @param brain Brain handle
 * @param metrics Output metrics structure
 * @return true on success, false on error
 */
bool brain_get_quantum_shannon_metrics(brain_t brain, shannon_diffusion_metrics_t* metrics);

/**
 * @brief Evolve quantum-Shannon diffusion manually
 *
 * WHAT: Manually trigger quantum-Shannon evolution
 * WHY:  For fine-grained control or testing
 * HOW:  Calls quantum_shannon_evolve() with configured steps
 *
 * @param brain Brain handle
 * @param num_steps Number of evolution steps (0 = use configured value)
 * @return true on success, false on error
 */
bool brain_evolve_quantum_shannon(brain_t brain, uint32_t num_steps);

//=============================================================================
// Phase C4.7: Cross-Modal Information Flow API
//=============================================================================

/**
 * @brief Enable cross-modal information flow monitoring
 *
 * WHAT: Activate real-time tracking of information flow between sensory modalities
 * WHY:  Monitor multi-sensory integration, detect bottlenecks, optimize routing
 * HOW:  Sets enable_cross_modal_monitoring flag and creates routing graph
 *
 * BIOLOGICAL BASIS: Superior temporal sulcus (audiovisual), superior colliculus (multisensory)
 * PERFORMANCE IMPACT: ~2-5% overhead during multimodal processing
 *
 * @param brain Brain handle
 * @param enable true to enable, false to disable
 */
void brain_enable_cross_modal_monitoring(brain_t brain, bool enable);

/**
 * @brief Get cross-modal routing graph
 *
 * WHAT: Retrieve current cross-modal information routing graph
 * WHY:  Allow external analysis of multi-sensory integration pathways
 * HOW:  Returns pointer to brain's cross_modal_graph
 *
 * NOTE: Graph may be NULL if cross-modal monitoring not enabled
 *
 * @param brain Brain handle
 * @return Cross-modal routing graph (NULL if not enabled)
 */
cross_modal_routing_graph_t* brain_get_cross_modal_graph(brain_t brain);

/**
 * @brief Get last multi-modal integration metrics
 *
 * WHAT: Retrieve most recent cross-modal integration metrics
 * WHY:  Monitor synergy, redundancy, and integration efficiency
 * HOW:  Returns copy of last_cross_modal_metrics from brain
 *
 * @param brain Brain handle
 * @param metrics Output metrics structure
 * @return true on success, false on error
 */
bool brain_get_cross_modal_metrics(brain_t brain, multi_modal_integration_t* metrics);

/**
 * @brief Set cross-modal bottleneck detection threshold
 *
 * WHAT: Configure threshold for identifying cross-modal bottlenecks
 * WHY:  Tune sensitivity of bottleneck detection
 * HOW:  Sets cross_modal_bottleneck_threshold in brain
 *
 * @param brain Brain handle
 * @param threshold Efficiency threshold [0.0-1.0] (default: 0.5)
 */
void brain_set_cross_modal_threshold(brain_t brain, float threshold);

//=============================================================================
// Community Detection & Network Topology Analysis
//=============================================================================

/**
 * @brief Get or create network analyzer for topology analysis
 *
 * WHAT: Lazy initialization of network analyzer for real-time topology monitoring
 * WHY:  Enable continuous analysis of network organization during learning
 * HOW:  Create analyzer on first access, cache for reuse
 *
 * @param brain Brain to analyze
 * @return Network analyzer pointer (opaque) or NULL on error
 */
void* brain_get_network_analyzer(brain_t brain);

/**
 * @brief Detect functional modules (communities) in brain network
 *
 * WHAT: Run Louvain algorithm to identify functional communities
 * WHY:  Understand modular organization and functional specialization
 * HOW:  Builds graph from brain topology and runs community detection
 *
 * ALGORITHM: Louvain method (greedy modularity optimization)
 * COMPLEXITY: O(n log n) where n = number of neurons
 *
 * RESULTS: Stored in brain->functional_modules
 * - Number of communities detected
 * - Community assignment for each neuron
 * - Modularity score Q (0.3+ is good, 0.5+ is excellent)
 *
 * @param brain Brain handle
 * @return true on success, false on error
 */
NIMCP_EXPORT bool brain_detect_communities(brain_t brain);

/**
 * @brief Get community assignment for a neuron
 *
 * WHAT: Query which functional module a neuron belongs to
 * WHY:  Analyze neuron roles and functional specialization
 * HOW:  Lookup in brain->functional_modules
 *
 * NOTE: Requires brain_detect_communities() to be called first
 *
 * @param brain Brain handle
 * @param neuron_id Neuron index
 * @return Community ID or UINT32_MAX if not found/invalid
 */
NIMCP_EXPORT uint32_t brain_get_neuron_community(brain_t brain, uint32_t neuron_id);

/**
 * @brief Detect hub neurons (high connectivity or betweenness)
 *
 * WHAT: Identify neurons with exceptional connectivity
 * WHY:  Find critical neurons for network function
 * HOW:  Computes degree centrality, identifies outliers
 *
 * ALGORITHM: Hub = degree > mean + threshold*std
 * THRESHOLD: 2.0 is typical (neurons 2 std deviations above mean)
 *
 * RESULTS: Stored in brain->network_hubs
 * - List of hub neuron IDs
 * - Hub scores (degree centrality)
 * - Fast lookup: is_hub[neuron_id]
 *
 * @param brain Brain handle
 * @param threshold Standard deviations above mean (typical: 1.5-2.5)
 * @return true on success, false on error
 */
NIMCP_EXPORT bool brain_detect_hubs(brain_t brain, float threshold);

/**
 * @brief Check if a neuron is a hub
 *
 * WHAT: Fast lookup for hub status
 * WHY:  Quickly check if neuron is critical for network function
 * HOW:  Lookup in brain->network_hubs->is_hub
 *
 * NOTE: Requires brain_detect_hubs() to be called first
 *
 * @param brain Brain handle
 * @param neuron_id Neuron index
 * @return true if neuron is a hub, false otherwise
 */
NIMCP_EXPORT bool brain_is_hub_neuron(brain_t brain, uint32_t neuron_id);

/**
 * @brief Compute comprehensive topology metrics
 *
 * WHAT: Calculate network quality metrics (modularity, clustering, path length, small-world)
 * WHY:  Assess brain network health and efficiency
 * HOW:  Computes Q, C, L, σ and other graph statistics
 *
 * METRICS:
 * - Modularity Q: community structure quality (>0.3 is good)
 * - Clustering C: local connectivity (>0.1 is healthy)
 * - Path length L: communication efficiency (low is good)
 * - Small-world σ: efficiency ratio (>1.0 is small-world)
 * - Diameter: longest shortest path
 * - Density: fraction of possible edges present
 * - Components: number of disconnected subgraphs (should be 1)
 *
 * RESULTS: Stored in brain->topology_metrics
 *
 * @param brain Brain handle
 * @return true on success, false on error
 */
NIMCP_EXPORT bool brain_compute_topology_metrics(brain_t brain);

/**
 * @brief Validate brain topology for common problems
 *
 * WHAT: Check for network health issues
 * WHY:  Detect problems like disconnected components, poor modularity
 * HOW:  Runs comprehensive topology checks
 *
 * CHECKS:
 * 1. Disconnected components (should be 1 component)
 * 2. Poor modularity (Q should be > 0.2)
 * 3. Non-small-world (σ should be > 1.0)
 * 4. Low clustering (C should be > 0.1)
 * 5. Poorly connected hubs (hub degree should be > 5)
 *
 * @param brain Brain handle
 * @return true if topology is healthy, false if problems detected
 */
NIMCP_EXPORT bool brain_validate_topology(brain_t brain);

#ifdef __cplusplus
}
#endif

#endif  // NIMCP_BRAIN_H
