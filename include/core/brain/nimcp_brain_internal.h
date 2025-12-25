//=============================================================================
// nimcp_brain_internal.h - Internal Brain Structure Definition
//=============================================================================
/**
 * @file nimcp_brain_internal.h
 * @brief Internal header exposing brain_struct and task_strategy_t for brain modules
 *
 * WHAT: Internal definitions for brain implementation modules
 * WHY:  Extracted brain modules (factory, learning, inference, persistence,
 *       distributed, strategy) need access to brain internal structures
 * HOW:  Exposes brain_struct and task_strategy_t definitions extracted from nimcp_brain.c
 *
 * WARNING: THIS IS FOR INTERNAL USE ONLY
 * ========================================
 * DO NOT include this header in public-facing code - use nimcp_brain.h instead.
 * This header exposes internal implementation details that are subject to change.
 * Only brain implementation modules should include this file.
 *
 * VISIBILITY:
 * - brain_struct: Complete internal structure of the brain
 * - task_strategy_t: Strategy pattern interface for task-specific behaviors
 *
 * DESIGN:
 * - Allows modular decomposition of brain implementation
 * - Maintains encapsulation at module boundaries
 * - Supports clean separation of concerns
 */

#ifndef NIMCP_BRAIN_INTERNAL_H
#define NIMCP_BRAIN_INTERNAL_H

//=============================================================================
// Required Headers for Type Definitions
//=============================================================================

#include "core/brain/nimcp_brain.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include <stdint.h>
#include <stdbool.h>
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "networking/distributed/nimcp_distributed_cognition.h"
#include "security/nimcp_security.h"
#include "security/nimcp_security_integration.h"  // Phase SC-4: Universal Security Integration
#include "security/nimcp_blood_brain_barrier.h"   // Phase IS-1: BBB Perimeter Defense

// Comprehensive Integration: All Advanced Subsystems
#include "glial/integration/nimcp_glial_integration.h"
#include "core/brain_oscillations/nimcp_brain_oscillations.h"
#include "cognitive/introspection/nimcp_introspection.h"
#include "cognitive/introspection/nimcp_connectivity_health.h"  // Phase 1.5.4: Connectivity Health
#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/salience/nimcp_salience.h"
#include "cognitive/consolidation/nimcp_consolidation.h"
#include "cognitive/curiosity/nimcp_curiosity.h"
#include "cognitive/knowledge/nimcp_knowledge.h"
#include "cognitive/nimcp_symbolic_logic.h"              // Phase 9.4: Symbolic reasoning
#include "cognitive/epistemic/nimcp_epistemic_filter.h"  // Phase 9.2: Bias prevention
#include "cognitive/wellbeing/nimcp_wellbeing.h"        // Phase 9.3: Self-preservation
#include "plasticity/neuromodulators/nimcp_neuromod_pink_noise.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include "plasticity/neuromodulators/nimcp_spatial_neuromod.h"
#include "plasticity/attention/nimcp_attention.h"
#include "core/neuron_types/nimcp_neural_logic.h"

// Phase T1: Biological Framework Enhancements (Training Pipeline)
#include "plasticity/homeostatic/nimcp_homeostatic.h"
#include "plasticity/dendritic/nimcp_dendritic.h"
#include "plasticity/predictive/nimcp_predictive_coding.h"
#include "plasticity/nimcp_second_messengers.h"  // cAMP, IP3/DAG, Ca2+, gene expression cascades
#include "cognitive/nimcp_fractal_cognitive.h"

// Phase TM-3: Brain-Training Integration (Training Pipeline)
#include "middleware/training/nimcp_brain_training_integration.h"

// Phase TPB-1: Training-Plasticity Bridge (connects training pipeline to biological plasticity)
#include "middleware/training/nimcp_training_plasticity_bridge.h"

// Phase EDP-1: Event-Driven Plasticity (continuous learning from sensory events)
#include "middleware/training/nimcp_event_driven_plasticity.h"

// Phase 8: Multi-Modal Integration
#include "core/integration/nimcp_multimodal_integration.h"
#include "perception/nimcp_visual_cortex.h"
#include "perception/nimcp_audio_cortex.h"
#include "perception/nimcp_speech_cortex.h"
#include "nlp/nimcp_nlp.h"

// Brain Regions Architecture
#include "core/brain_regions/nimcp_brain_regions.h"

// Universal Event Bus
#include "core/events/nimcp_event_bus.h"

// Phase 10: Advanced Cognitive Architecture
#include "cognitive/nimcp_working_memory.h"
#include "cognitive/nimcp_emotional_tagging.h"
#include "cognitive/nimcp_emotional_system.h"
#include "cognitive/nimcp_executive.h"
#include "cognitive/nimcp_sleep_wake.h"
#include "cognitive/memory/nimcp_engram.h"
#include "cognitive/memory/nimcp_systems_consolidation.h"
#include "cognitive/memory/nimcp_wm_transfer.h"
#include "cognitive/memory/nimcp_semantic_memory.h"
#include "cognitive/nimcp_mental_health.h"
#include "cognitive/nimcp_theory_of_mind.h"
#include "cognitive/nimcp_explanations.h"
#include "cognitive/nimcp_meta_learning.h"
#include "cognitive/nimcp_predictive.h"
#include "cognitive/nimcp_mirror_neurons.h"
#include "cognitive/global_workspace/nimcp_global_workspace.h"
#include "cognitive/nimcp_autobiographical_memory.h"
#include "cognitive/nimcp_self_model.h"
#include "cognitive/nimcp_personality.h"

// Phase E: Higher-Order Cognitive & Social Systems
#include "cognitive/nimcp_shadow_emotions.h"
#include "cognitive/nimcp_bias_detection.h"
#include "cognitive/analysis/nimcp_network_analysis.h"
#include "cognitive/nimcp_grief_and_loss.h"
#include "cognitive/nimcp_joy_euphoria.h"
#include "cognitive/nimcp_remorse_regret.h"
#include "cognitive/nimcp_love_loyalty_friendship.h"

// Phase C4: Information Theory
#include "information/nimcp_shannon.h"
#include "utils/quantum/nimcp_quantum_shannon.h"
#include "information/nimcp_cross_modal.h"

// Phase 11: Quantum Annealing
#include "optimization/quantum_annealing/nimcp_quantum_annealing.h"

// Topology and Community Detection
#include "core/topology/nimcp_community_detection.h"
#include "utils/algorithms/nimcp_graph_metrics.h"
#include "utils/containers/nimcp_graph.h"

// Phase CC-1: Cortical Columns Architecture (Tier 0.65)
#include "core/cortical_columns/nimcp_cortical_column.h"
#include "core/cortical_columns/nimcp_cortical_layers.h"
#include "core/cortical_columns/nimcp_columnar_connectivity.h"
#include "core/cortical_columns/nimcp_topographic_maps.h"
#include "core/cortical_columns/nimcp_orientation_columns.h"
#include "core/cortical_columns/nimcp_feature_hypercolumns.h"

// Phase 2 Middleware: Population coding & spike analysis
#include "middleware/brain_integration.h"

// Phase 1.5: Memory pools for hot-path allocations
#include "utils/memory/nimcp_memory_pool.h"

// Brain Immune System Integration
#include "cognitive/immune/nimcp_brain_immune.h"

// FEP Orchestrator Integration
#include "cognitive/free_energy/nimcp_fep_orchestrator.h"

// Core Directives Integration (Ethical Foundation)
// Uses core/directives which has the full implementation with nested configs
#include "core/directives/nimcp_core_directives.h"

// Medulla Oblongata Integration (Brainstem Autonomic Regulation)
#include "core/medulla/nimcp_medulla.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Forward Declarations - Strategy Pattern
//=============================================================================

typedef struct task_strategy task_strategy_t;

//=============================================================================
// Internal Brain Structure with Strategy
//=============================================================================

/**
 * @brief Brain internal structure with strategy pattern
 *
 * WHY: Encapsulates network, config, labels, stats, and task strategy
 * Enables type-specific behaviors without switch statements
 *
 * COMPLEXITY: O(1) access to all members
 */
struct brain_struct {
    // === CORE COMPONENTS ===
    adaptive_network_t network;  // Underlying neural network
    brain_config_t config;       // User configuration
    task_strategy_t* strategy;   // Task-specific behavior strategy

    // Label to output mapping
    char** output_labels;        // Label strings
    uint32_t num_output_labels;  // Current label count

    // Statistics
    brain_stats_t stats;  // Performance metrics

    // Decision caching for optimization
    float* last_input;                  // Cached input vector
    brain_decision_t* cached_decision;  // Cached decision
    uint32_t input_size;                // For cache validation
    nimcp_platform_mutex_t cache_mutex; // Thread-safe cache access

    // Phase 11: Adaptive Learning Rate (Simple Online Meta-Learning)
    float loss_history[10];             // Rolling window of last 10 losses
    uint32_t loss_history_index;        // Current position in circular buffer
    uint32_t loss_history_count;        // Number of losses recorded (0-10)
    float base_learning_rate;           // Original learning rate (backup)

    // Phase 11: Curiosity-Driven Learning Rate Modulation
    float last_curiosity_drive;         // Most recent curiosity drive [0.0-1.0]
    float last_novelty_score;           // Most recent novelty score [0.0-1.0]

    // Phase 11: Long-Term Memory Consolidation Buffer
    struct {
        float* features;                 // Feature vector
        uint32_t num_features;          // Vector size
        float salience;                 // Importance score
        uint64_t timestamp_ms;          // When consolidated
    } *longterm_memory;                     // Array of consolidated memories
    uint32_t longterm_capacity;             // Max memories (default: 100)
    uint32_t longterm_count;                // Current count

    // Phase 3: Distributed cognition coordinator
    distrib_cognition_t distributed;  // P2P cognitive coordination (NULL if standalone)

    // Phase 2: Copy-on-Write (COW) tracking
    bool is_cow_clone;                  // Is this a COW clone?
    bool owns_network;                  // Does this brain own its network? (can destroy it)
    adaptive_network_t original_network; // Original network reference (if COW)
    bool network_is_cached;             // Is network allocated via nimcp_cache?

    // Phase 3: Reference counting and read-only optimization
    uint32_t* network_refcount;         // Pointer to shared refcount (NULL if not shared)
    bool can_use_readonly;              // Can use read-only inference? (true for COW clones)
    nimcp_platform_mutex_t* refcount_mutex;    // Mutex for refcount updates (shared among clones)
    bool is_snapshot;                   // Is this a snapshot? (preserve stats, don't update from network)
    brain_stats_t snapshot_stats;       // Preserved stats at snapshot time (only used if is_snapshot=true)

    // === COMPREHENSIVE INTEGRATION: ADVANCED SUBSYSTEMS ===
    // NOTE: Only modules that currently exist are integrated
    // Types marked with * are already pointer types in their typedef
    // Types without * are struct types and need pointer declaration here

    // Phase 5/6: Biological Realism
    glial_integration_t* glial;                  // Glial cells (struct type, needs *)
    brain_oscillation_analyzer_t* oscillations;  // Brain wave analysis (struct type, needs *)
    myelin_sheath_network_t* myelin_sheath;      // Myelin structural modeling (struct type, needs *)

    // Consciousness & Cognition (most use pointer typedefs)
    introspection_context_t introspection;       // Self-awareness (already pointer type*)
    ethics_engine_t ethics;                      // Golden Rule, empathy (already pointer type*)
    salience_evaluator_t salience;               // Fast attention (already pointer type*)
    consolidation_handle_t consolidation;        // Memory consolidation (already pointer type*)
    curiosity_engine_t curiosity;                // Exploration (already pointer type*)
    knowledge_system_t knowledge;                // Multi-domain knowledge (already pointer type*)
    neural_logic_network_t logic;                // Phase 9.0: Neural logic gates (spiking logic, GPU-accelerated)
    symbolic_logic_t* symbolic_logic;            // Phase 9.4: Symbolic reasoning (first-order logic, inference)
    epistemic_filter_t epistemic;                // Phase 9.2: Epistemic filtering (bias prevention, skepticism)

    // Phase 9.3: Wellbeing & Self-Preservation
    distress_assessment_t last_distress;         // Most recent distress assessment
    bool wellbeing_monitoring_enabled;           // Enable/disable wellbeing checks
    uint64_t wellbeing_check_interval_ms;        // How often to check (0 = every decision)
    uint64_t last_wellbeing_check_time;          // Timestamp of last check

    // Simulation Time Tracking (for proper glial/calcium dynamics)
    uint64_t current_time_us;                    // Current simulation time in microseconds
    uint64_t last_glial_update_us;               // Last glial integration update time

    // === PHASE 10: ADVANCED COGNITIVE SYSTEMS ===

    // Phase 10.1: Working Memory (Miller's 7±2)
    working_memory_t* working_memory;            // Active representation buffer (prefrontal cortex)

    // Phase 10.2: Emotional Tagging (Russell's circumplex model)
    emotional_system_t* emotional_system;        // Emotional state and memory tagging

    // Phase 10.3: Executive Functions (task switching, planning, inhibition)
    executive_controller_t* executive;           // Executive control center (DLPFC)

    // Phase 10.4: Sleep/Wake Cycle (Memory consolidation & synaptic homeostasis)
    sleep_system_t sleep_system;                 // Sleep/wake state machine, consolidation

    // Phase M1: Memory Engrams (distributed memory traces)
    engram_system_t* engram_system;              // Memory engram encoding, consolidation, and recall

    // Phase M2: Systems Consolidation (hippocampus → cortex transfer)
    systems_consolidation_system_t* systems_consolidation; // Sleep-dependent memory transfer and semantic abstraction

    // Phase M3: Working Memory Transfer (WM → engram encoding)
    wm_transfer_system_t* wm_transfer_system;    // Selective transfer from working memory to long-term memory

    // Phase M4: Semantic Memory Network (concept network + spreading activation)
    semantic_memory_system_t* semantic_memory;   // Semantic concept network for abstract reasoning and inference

    // Phase 10.5: Mental Health Monitoring (disorder detection & intervention)
    mental_health_monitor_t* mental_health_monitor; // Mental health tracking and safety

    // Phase 11: Part I - Emotional Intelligence & Accessibility
    empathy_network_t empathy_network;           // Mirror neuron empathy system for perspective-taking
    void* empathetic_response_engine;            // Non-reactive empathetic response system (opaque pointer)

    // Phase 12: Self-Awareness Enhancement (Autobiographical Memory & Self-Model)
    autobiographical_memory_t autobio;           // Episodic self-memory system (timeline-indexed experiences)
    self_model_system_t self_model;              // Explicit self-representation (identity, beliefs, capabilities)
    personality_profile_t* personality;          // Phase 12: Personality, Gender, and Sexual Identity (unique individual traits)

    // Phase 10.6: Theory of Mind (mental state inference)
    theory_of_mind_t theory_of_mind;             // Model other agents' beliefs, desires, goals (opaque pointer)

    // Phase 10.7: Natural Explanations (interpretability)
    explanation_generator_t explanation_gen;     // Generate human-readable explanations (opaque pointer)

    // Phase 10.8: Meta-Learning (learning-to-learn)
    meta_learner_t meta_learner;                // MAML-style meta-learning (opaque pointer)

    // Phase 10.9: Predictive Processing (free energy minimization)
    predictive_network_t predictive_network;     // Hierarchical predictive coding (opaque pointer)

    // Phase 10.11: Mirror Neurons (social cognition, imitation learning)
    mirror_neurons_t mirror_neurons;             // Observation-action learning system (opaque pointer)

    // Global Workspace Architecture (Global Workspace Theory - Baars, Dehaene)
    global_workspace_t* global_workspace;        // Central broadcast architecture for conscious access

    // Advanced Plasticity
    neuromod_pink_noise_t* pink_noise;           // Pink noise neuromodulation (struct type, needs *)
    neuromodulator_system_t neuromodulator_system; // Full neuromodulator system (DA, 5-HT, ACh, NE, GABA, GLU)
    multihead_attention_t multihead_attention;   // Attention mechanism for selective feature processing (typedef already includes *)

    // Phase T1: Biological Framework Enhancements (Training Pipeline)
    homeostatic_controller_t homeostatic;        // Synaptic scaling + intrinsic plasticity (maintains activity levels)
    dendritic_tree_t dendritic;                  // Dendritic computation with NMDA dynamics (local nonlinearities)
    pc_hierarchy_t predictive_coding;            // Free energy minimization (hierarchical error computation)
    second_messenger_system_t* second_messengers; // Second messenger cascades (cAMP, IP3/DAG, Ca2+, gene expression)
    bool enable_second_messengers;               // Enable second messenger cascade system

    // Phase TM-3: Brain-Training Integration (Training Pipeline)
    nimcp_brain_training_ctx_t* training_ctx;    // Training integration context (loss functions, optimizers)
    bool enable_training_integration;            // Enable training subsystem

    // Phase TPB-1: Training-Plasticity Bridge
    tpb_context_t* plasticity_bridge;            // Connects training pipeline to biological plasticity systems
    bool enable_plasticity_bridge;               // Enable plasticity bridge

    // Phase EDP-1: Event-Driven Plasticity (continuous learning from sensory events)
    edp_context_t* event_driven_plasticity;      // Event-driven continuous learning adapter
    bool enable_event_driven_plasticity;         // Enable EDP system

    // === PHASE 8: UNIFIED MULTI-MODAL PROCESSING ===
    // Sensory Cortices (specialized feature extractors)
    visual_cortex_t* visual_cortex;              // V1 visual processing (CNN-based)
    audio_cortex_t* audio_cortex;                // A1 auditory processing (FFT-based)
    speech_cortex_t* speech_cortex;              // STG/Wernicke speech processing (Phase 8.8)

    // Multi-Modal Integration Layer
    multimodal_integration_t multimodal;         // Integrates sensory features into unified representation

    // NLP Processing (language understanding and generation)
    nlp_network_t nlp_network;                   // Natural language processor (token embeddings + attention + neuromodulation)

    // Feature buffers (reusable to avoid allocation per frame)
    float* visual_feature_buffer;                // Pre-allocated visual features
    float* audio_feature_buffer;                 // Pre-allocated audio features
    float* speech_feature_buffer;                // Pre-allocated speech features (Phase 8.8)
    float* integrated_feature_buffer;            // Pre-allocated integrated features

    // Brain Regions Architecture (hierarchical cortical organization)
    brain_module_t* brain_regions;               // Modular brain regions with cortical layers and minicolumns

    // Phase 11 Enhancement C1.1: Quantum Annealing for Weight Optimization
    quantum_annealer_t quantum_annealer;         // Quantum annealing optimizer for escaping local minima

    // Phase C4: Shannon Information Theory (Channel Capacity & Bottleneck Analysis)
    shannon_config_t shannon_config;              // Shannon analysis configuration
    bool enable_shannon_monitoring;               // Enable real-time information flow monitoring
    shannon_network_metrics_t last_shannon_metrics; // Last computed network-level Shannon metrics

    // Phase C4.1: Quantum-Shannon Information Diffusion (√N speedup + bottleneck detection)
    void* quantum_shannon_diffusion;              // quantum_shannon_diffusion_t* (opaque to avoid circular dependency)
    bool enable_quantum_shannon_diffusion;        // Enable quantum-Shannon accelerated diffusion
    float quantum_shannon_mixing_ratio;           // Mix quantum+classical [0=pure quantum, 1=classical]
    uint32_t quantum_shannon_evolution_steps;     // Steps per diffusion update (default: 100)
    shannon_diffusion_metrics_t last_quantum_shannon_metrics; // Last computed quantum-Shannon metrics

    // Phase C4.7: Cross-Modal Information Flow (multi-sensory integration tracking)
    cross_modal_routing_graph_t* cross_modal_graph;    // Cross-modal information routing graph (visual↔audio↔speech)
    bool enable_cross_modal_monitoring;                 // Enable real-time cross-modal tracking
    multi_modal_integration_t last_cross_modal_metrics; // Last computed multi-modal integration metrics
    float cross_modal_bottleneck_threshold;             // Efficiency threshold for bottleneck detection (default: 0.5)
    uint32_t cross_modal_sample_count;                  // Number of samples used for analysis (default: 50)

    // === PHASE E: HIGHER-ORDER COGNITIVE & SOCIAL SYSTEMS ===

    // Phase E5: Shadow Emotions (self-monitoring & correction for maladaptive patterns)
    shadow_emotion_system_t* shadow_emotions;     // Detect jealousy, envy, obsession, hubris, greed, narcissism

    // Phase E6: Bias Detection & Correction (fairness & social cognition)
    bias_detection_system_t* bias_detection;      // Detect and correct biases (racial, LGBTQ+, gender, misogyny, etc.)

    // === PHASE E: FULL EMOTIONAL INTELLIGENCE ===

    // Phase E1: Grief and Loss (negative emotion - attachment severing)
    grief_system_t* grief_system;                 // Process loss, bereavement, mortality awareness

    // Phase E2: Joy and Euphoria (positive emotion - value-aligned success)
    joy_system_t* joy_system;                     // Reward for value-aligned achievements, flow states

    // Phase E3: Remorse and Regret (moral emotion - evaluative)
    remorse_regret_system_t* remorse_system;      // Moral evaluation, guilt, learning from mistakes

    // Phase E4: Love, Loyalty, Friendship (positive social emotion - bonding)
    social_bond_system_t* social_bond_system;     // Attachment, trust, positive relationships

    // NIMCP 2.7 Phase 8.5: Fractal Topology Integration
    fractal_cognitive_cache_t* fractal_cache;     // Topology-based cognitive enhancements
    bool has_fractal_topology;                    // Whether fractal topology is enabled

    // === COMMUNITY DETECTION & NETWORK TOPOLOGY ===
    community_structure_t* functional_modules;    // Detected functional communities
    hub_structure_t* network_hubs;                // Hub neurons (high connectivity)
    topology_validation_t* topology_metrics;            // Network quality metrics (Q, C, L, σ)
    bool auto_detect_communities;                 // Auto-run after training
    float community_detection_interval;           // Run every N epochs (0 = manual only)

    // Network analyzer for real-time topology analysis during inference
    void* network_analyzer;                       // network_analyzer_t* (opaque to avoid circular dependency)

    // === PHASE 1.5.4: CONNECTIVITY HEALTH MONITORING ===
    bool enable_connectivity_monitoring;          // Enable periodic connectivity health assessment
    connectivity_health_config_t connectivity_health_config; // Configuration for health assessment
    brain_connectivity_health_t last_connectivity_health;   // Cached last connectivity health assessment
    uint64_t last_connectivity_assessment_time_ms;          // Timestamp of last assessment
    void (*connectivity_health_callback)(const brain_connectivity_health_t*, void*); // Callback on health change
    void* connectivity_health_callback_context;   // Context for callback

    // === PHASE 1.5.5: MIDDLEWARE CONTROLLER (COGNITIVE → MIDDLEWARE) ===
    struct middleware_controller* middleware_controller;  // Unified cognitive-to-middleware command interface
    bool enable_middleware_controller;                    // Enable middleware controller subsystem

    // === PHASE 2 MIDDLEWARE: SPIKE ANALYSIS & POPULATION CODING ===

    // Spike-based feature extraction (firing rates, ISI, CV, synchrony, oscillations, entropy)
    brain_spike_feature_extractor_t spike_feature_extractor;  // Extract population features from spike trains

    // Population coding analysis (vector sum, center of mass, PCA, synchrony)
    brain_population_analyzer_t population_analyzer;          // Analyze distributed population codes

    // Enable/disable middleware features
    bool enable_spike_analysis;                   // Enable spike-based feature extraction
    bool enable_population_coding;                // Enable population coding analysis

    // === UNIVERSAL EVENT BUS INTEGRATION ===

    // Event bus for broadcasting brain activities (training, inference, cognitive events)
    event_bus_t event_bus;                        // Event bus for system-wide event coordination
    bool enable_event_broadcasting;               // Enable/disable event publishing

    // === AXON & DENDRITE INTEGRATION ===
    //
    // NOTE: Axons and dendrites are components of individual neurons, not separate networks.
    // These "network" containers are management structures that:
    // - Provide efficient lookup/iteration over all axons/dendrites
    // - Enable network-wide operations (step all, get stats, etc.)
    // - Allow glial cells to modulate axonal conduction (oligodendrocytes)
    //
    // Each neuron has:
    // - One axon (output): neuron->axon_id references into axon_network
    // - Multiple dendrites (inputs): neuron->dendrite_ids references into dendrite_network
    //
    // Signal flow: Presynaptic neuron → Axon → Synapse → Dendritic spine → Postsynaptic neuron

    // Axon container for all neuron axons (Phase 1.5.6)
    void* axon_network;                           // axon_network_t* - manages axons for all neurons

    // Dendrite container for all neuron dendrites (Phase 1.5.7)
    void* dendrite_network;                       // dendrite_network_t* - manages dendrites for all neurons

    // === PHASE CC-1: CORTICAL COLUMNS ARCHITECTURE (Tier 0.65) ===
    //
    // Hierarchical cortical column organization based on Douglas & Martin (1991) canonical microcircuit.
    // Minicolumns (~80-100 neurons) group into hypercolumns (~100K neurons) with lateral inhibition
    // and competitive dynamics for feature detection and representation.
    //
    // Architecture:
    // - cortical_column_pool: Memory management for minicolumns/hypercolumns
    // - laminar_system: 6-layer organization (I, II/III, IV, V, VI)
    // - columnar_connectivity: Canonical microcircuit connectivity patterns
    // - topographic_maps: Retinotopic, tonotopic, somatotopic spatial maps
    // - orientation_system: V1 orientation selectivity (Gabor filters)
    // - feature_hypercolumns: Multi-dimensional feature coverage

    cortical_column_pool_t* cortical_column_pool;     // Memory pool for minicolumns/hypercolumns
    hypercolumn_t** hypercolumns;                     // Array of hypercolumns
    uint32_t num_hypercolumns;                        // Number of hypercolumns
    laminar_structure_t* laminar_system;              // 6-layer cortical organization
    columnar_connectivity_t* columnar_connectivity;   // Canonical microcircuit connectivity
    topographic_map_t* visual_topographic_map;        // Retinotopic map for visual cortex
    topographic_map_t* auditory_topographic_map;      // Tonotopic map for auditory cortex
    topographic_map_t* somatosensory_topographic_map; // Somatotopic map for S1
    orientation_hypercolumn_t** orientation_hypercolumns; // V1 orientation columns
    uint32_t num_orientation_hypercolumns;            // Number of orientation hypercolumns
    feature_hypercolumn_t** feature_hypercolumns;     // Multi-dimensional feature hypercolumns
    uint32_t num_feature_hypercolumns;                // Number of feature hypercolumns
    bool enable_cortical_columns;                     // Master enable flag
    bool cortical_needs_lazy_init;                    // Flag: cortical columns need lazy init
    bool topographic_needs_lazy_init;                 // Flag: topographic maps need lazy init
    uint64_t last_cortical_update_us;                 // Last cortical column update timestamp

    // === PHASE 1.5: MEMORY POOLS FOR HOT-PATH ALLOCATIONS ===

    // Decision structure pool - used for internal cached_decision allocation
    // Pool for brain_decision_t structures (fixed size, O(1) acquire/release)
    memory_pool_t decision_struct_pool;

    // Output vector pool - used for decision output vectors
    // Pool for output vectors of size config.num_outputs * sizeof(float)
    memory_pool_t output_vector_pool;

    // Active neuron IDs pool - used for decision interpretability data
    // Pool for neuron ID arrays (max neurons * sizeof(uint32_t))
    memory_pool_t active_neuron_ids_pool;

    // Track max neurons for pool sizing
    uint32_t max_active_neurons_for_pool;

    // === PHASE SC-2: SECURITY-FAULT TOLERANCE INTEGRATION ===
    //
    // Security recovery bridge connects security modules (coverage, fractal, CFI,
    // shadow stack, audit) with the fault tolerance system (fast recovery, checkpoints).
    // When security violations are detected, automatic repair actions are triggered.
    //
    // Architecture:
    // - Security Coverage: Tracks protected memory regions with hash verification
    // - Fractal Security: Hierarchical integrity checking (Merkle tree)
    // - CFI/Shadow Stack: Control flow protection against ROP/JOP attacks
    // - Fast Recovery: Sub-millisecond repair for common errors
    // - Checkpoints: Full state restoration for severe violations

    void* security_bridge;              // nimcp_security_recovery_bridge_t* (opaque)
    bool enable_security_monitoring;    // Enable security-fault tolerance integration
    uint32_t security_check_interval_ms; // Verification cycle interval (0 = manual only)
    uint64_t last_security_check_ms;    // Last security verification timestamp

    // === PHASE SC-4: UNIVERSAL SECURITY INTEGRATION ===
    //
    // Global security integration framework provides:
    // - Entropy monitoring: Detect tampering via Shannon entropy analysis
    // - Trust management: Bayesian trust propagation across modules
    // - Differential privacy: Privacy-preserving statistics and queries
    // - Event system: Security event propagation and logging
    // - Self-monitoring: Security system monitors its own integrity
    //
    // The brain registers itself and its subsystems with the global security
    // context, enabling comprehensive security monitoring across all modules.

    nimcp_sec_integration_t* security_integration;  // Global security integration context
    uint32_t sec_module_id;                         // Brain's module ID in security system
    uint32_t* sec_region_ids;                       // Region IDs for monitored memory regions
    uint32_t num_sec_regions;                       // Number of monitored regions
    bool enable_security_integration;               // Enable Phase SC-4 security

    // === PHASE IS-1: BLOOD-BRAIN BARRIER (BBB) INTEGRATION ===
    //
    // BBB provides perimeter defense for the neural network:
    // - Input Gate: Validates and sanitizes all external inputs
    // - Code Signing: Verifies integrity of loaded weights/models
    // - Memory Boundary: Protects critical memory regions
    // - Access Control: Role-based access to brain operations
    //
    // Each brain holds a reference to the global BBB system for protection.

    bbb_system_t bbb_system;            // Reference to global BBB system (NULL if disabled)
    uint32_t bbb_memory_region_id;      // BBB memory region registration ID
    uint32_t bbb_subject_id;            // BBB access control subject ID
    bool bbb_enabled;                   // BBB protection enabled for this brain

    // === BIO-ASYNC MESSAGING INTEGRATION ===
    //
    // Bio-async provides biologically-inspired asynchronous communication:
    // - Neuromodulator channels (dopamine, serotonin, norepinephrine, acetylcholine)
    // - Message handlers for brain state queries and neuron activation requests
    // - Predictive signal publishing for state changes (only triggers on prediction errors)
    // - Decoupled from cognitive modules via bio-router
    //
    void* bio_async_ctx;                         // brain_bio_async_ctx_t* (opaque pointer)
    void* bio_async_ctx_handle;                  // unified_mem_handle_t for context memory
    void* bio_async_mem_mgr;                     // unified_mem_manager_t for bio-async allocations
    bool bio_async_enabled;                      // Bio-async messaging enabled for this brain

    // === BRAIN IMMUNE SYSTEM INTEGRATION ===
    //
    // Brain immune system provides unified adaptive defense coordination:
    // - Antigen presentation: Converts BBB/BFT/swarm threats to immune antigens
    // - B cells: Antibody production mapped to swarm immune responses
    // - T cells: Helper coordination (cytokines) and killer actions (BFT quarantine)
    // - Cytokines: Bio-async signaling for cross-module immune coordination
    // - Inflammation: Hierarchical recovery escalation (node→pod→region→global)
    // - Memory: Long-term threat pattern storage via swarm immune memory cells
    //
    // Auto-connected during brain creation to BBB, BFT, and swarm systems.
    //
    brain_immune_system_t* immune_system;        // Brain immune coordination layer
    bool immune_enabled;                         // Immune system enabled for this brain

    // === FEP ORCHESTRATOR INTEGRATION ===
    //
    // FEP Orchestrator provides unified coordination of all FEP bridges:
    // - Manages 93+ FEP bridges across 9 categories
    // - Category-based update intervals (biologically-plausible timescales)
    // - Bio-async integration for inter-bridge messaging
    // - Brain immune integration for precision modulation
    // - Centralized statistics and monitoring
    //
    struct fep_orchestrator* fep_orchestrator;   // FEP bridge coordination layer
    bool fep_orchestrator_enabled;               // FEP orchestrator enabled for this brain

    // === CORE DIRECTIVES INTEGRATION (ETHICAL FOUNDATION) ===
    //
    // Core Directives provides foundational ethical constraint enforcement:
    // - Asimov's Three Laws (harm prevention, obedience, self-preservation)
    // - Golden Rule (reciprocity and fairness evaluation)
    // - Combinatorial Harm Detection (emergent harm from action sequences)
    // - All brain outputs must pass through directive evaluation
    // - Integration with brain immune system (ethical violations → immune response)
    // - Integration with FEP orchestrator (ethical constraints modulate free energy)
    //
    // This is the FIRST checkpoint before any action is executed - ethical
    // constraints cannot be overridden by higher cognitive functions.
    //
    core_directives_system_t* core_directives;      // Core ethical directives system
    directive_immune_bridge_t* directive_immune_bridge;  // Directives-immune bridge
    directive_fep_bridge_t* directive_fep_bridge;   // Directives-FEP bridge
    bool core_directives_enabled;                   // Core directives enabled for this brain

    // === COORDINATOR/ORCHESTRATOR INTEGRATIONS ===
    //
    // These coordinators manage system-wide coordination across NIMCP:
    // - Bio-Async Orchestrator: Central coordinator for 200+ bio-async modules
    // - Plasticity Coordinator: Unified manager for all plasticity mechanisms
    // - Immune Bridge Coordinator: Central registry for 27+ immune bridges
    // - Cognitive Meta-Controller: Arbitrator for cognitive subsystem resources
    // - Security-Perception Bridge: Sensory threat analysis and defense
    // - Swarm Module Registry: Plugin architecture for swarm behaviors
    //
    // Initialization order (dependencies):
    // 1. Bio-Async Orchestrator (foundation for messaging)
    // 2. Plasticity Coordinator (depends on bio-async)
    // 3. Immune Bridge Coordinator (depends on bio-async, brain immune)
    // 4. Cognitive Meta-Controller (depends on plasticity, working memory, executive)
    // 5. Security-Perception Bridge (depends on BBB, immune, perception cortices)
    // 6. Swarm Module Registry (depends on all above, swarm_brain)

    struct bio_async_orchestrator* bio_async_orchestrator;  // Bio-async module coordination
    bool bio_async_orchestrator_enabled;                    // Bio-async orchestrator enabled

    struct plasticity_coordinator* plasticity_coordinator;  // Plasticity mechanism coordination
    bool plasticity_coordinator_enabled;                    // Plasticity coordinator enabled

    struct immune_bridge_coordinator* immune_bridge_coordinator;  // Immune bridge coordination
    bool immune_bridge_coordinator_enabled;                       // Immune bridge coordinator enabled

    struct cognitive_meta_controller* cognitive_meta_controller;  // Cognitive resource arbitration
    bool cognitive_meta_controller_enabled;                       // Cognitive meta-controller enabled

    struct security_perception_bridge* security_perception_bridge;  // Sensory threat defense
    bool security_perception_bridge_enabled;                        // Security-perception bridge enabled

    struct swarm_module_registry* swarm_module_registry;  // Swarm behavior plugin registry
    bool swarm_module_registry_enabled;                   // Swarm module registry enabled

    // === MEDULLA OBLONGATA INTEGRATION (BRAINSTEM AUTONOMIC REGULATION) ===
    //
    // The Medulla Oblongata provides foundational autonomic regulation:
    // - Arousal State: Global activation level (alertness, attention readiness)
    // - Protective Cutoff: Emergency shutdown for system protection
    // - Circadian Rhythm: 24-hour biological clock simulation
    // - Brainstem Coupling: Coordination with other brainstem nuclei
    //
    // The medulla operates at the lowest level of the brain hierarchy,
    // providing continuous background regulation that affects all higher
    // cognitive functions. It integrates with:
    // - Immune System: Inflammation affects arousal, protection triggers
    // - Sleep/Wake: Circadian phase modulates sleep pressure
    // - Neuromodulators: Arousal state influences catecholamine release
    // - Bio-Async: Publishes state changes for system-wide coordination
    //
    medulla_t medulla;                                    // Medulla oblongata brainstem regulator
    bool medulla_enabled;                                 // Medulla enabled for this brain
    uint64_t last_medulla_update_us;                      // Last medulla update timestamp
};

//=============================================================================
// Strategy Pattern - Task-Specific Behaviors
//=============================================================================

/**
 * @brief Task strategy interface
 *
 * WHY: Different tasks (classification, regression) need different:
 * - Learning rates
 * - Output transformations
 * - Performance metrics
 *
 * PATTERN: Strategy pattern - encapsulates algorithm families
 */
struct task_strategy {
    brain_task_t task_type;

    /**
     * @brief Get recommended learning rate for this task
     * COMPLEXITY: O(1)
     */
    float (*get_learning_rate)(void);

    /**
     * @brief Transform raw output to task-specific format
     * COMPLEXITY: O(num_outputs)
     */
    void (*transform_output)(float* output, uint32_t size);

    /**
     * @brief Compute task-specific loss
     * COMPLEXITY: O(num_outputs)
     */
    float (*compute_loss)(const float* predicted, const float* target, uint32_t size);
};

#ifdef __cplusplus
}
#endif

#endif // NIMCP_BRAIN_INTERNAL_H
