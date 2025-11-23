//=============================================================================
// nimcp_brain_factory.c - Brain Factory and Configuration Implementation
//=============================================================================
/**
 * @file nimcp_brain_factory.c
 * @brief Implementation of brain factory functions for creation and orchestration
 *
 * WHAT: Factory pattern implementation for brain instantiation
 * WHY:  Separates complex brain creation logic from core brain operations
 * HOW:  Orchestrates initialization, validation, and creation workflows
 *
 * ARCHITECTURE:
 * - Factory Pattern: Creates brains of different types with validated configs
 * - Builder Pattern: Modular configuration construction  
 * - Strategy Pattern Integration: Task-specific behaviors
 *
 * EXTRACTED FROM: Original nimcp_brain_factory.c (3220 lines)
 * DATE: 2025-11-19
 *
 * @version 2.7.0
 * @author NIMCP Development Team
 */

//=============================================================================
// Includes
//=============================================================================

#include "nimcp_brain_factory.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/brain/strategy/nimcp_brain_strategy.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <cjson/cJSON.h>

// Core dependencies
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/cache/nimcp_cache.h"
#include "utils/time/nimcp_time.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/logging/nimcp_logging.h"
#include "security/nimcp_security.h"
#include "core/topology/nimcp_community_detection.h"
#include "utils/algorithms/nimcp_graph_metrics.h"
#include "utils/containers/nimcp_graph.h"

// Subsystem dependencies
#include "glial/integration/nimcp_glial_integration.h"
#include "core/brain_oscillations/nimcp_brain_oscillations.h"
#include "cognitive/introspection/nimcp_introspection.h"
#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/salience/nimcp_salience.h"
#include "cognitive/consolidation/nimcp_consolidation.h"
#include "cognitive/curiosity/nimcp_curiosity.h"
#include "cognitive/knowledge/nimcp_knowledge.h"
#include "cognitive/nimcp_symbolic_logic.h"
#include "cognitive/epistemic/nimcp_epistemic_filter.h"
#include "cognitive/wellbeing/nimcp_wellbeing.h"
#include "plasticity/neuromodulators/nimcp_neuromod_pink_noise.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include "plasticity/neuromodulators/nimcp_spatial_neuromod.h"
#include "plasticity/attention/nimcp_attention.h"
#include "core/neuron_types/nimcp_neural_logic.h"
#include "cognitive/nimcp_fractal_cognitive.h"

// Multi-modal integration
#include "core/integration/nimcp_multimodal_integration.h"
#include "include/perception/nimcp_visual_cortex.h"
#include "include/perception/nimcp_audio_cortex.h"
#include "include/perception/nimcp_speech_cortex.h"
#include "nlp/nimcp_nlp.h"

// Brain regions and cognitive systems
#include "core/brain_regions/nimcp_brain_regions.h"
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

// Sub-module headers
#include "init/nimcp_brain_init.h"
#include "validation/nimcp_brain_validation.h"

//=============================================================================
// External Function Declarations
//=============================================================================

// Error handling (shared across all brain modules)
extern void set_error(const char* format, ...);

//=============================================================================
// Compatibility Macros (bridge to init module functions)
//=============================================================================

// These macros map short names to the full prefixed names defined in init module
#define init_output_labels                          nimcp_brain_factory_init_output_labels
#define init_event_bus                              nimcp_brain_factory_init_event_bus
#define init_glial_subsystem                        nimcp_brain_factory_init_glial_subsystem
#define init_multimodal_subsystems                  nimcp_brain_factory_init_multimodal_subsystems
#define init_pink_noise_subsystem                   nimcp_brain_factory_init_pink_noise_subsystem
#define init_neuromodulator_system                  nimcp_brain_factory_init_neuromodulator_system
#define init_spatial_neuromod_system                nimcp_brain_factory_init_spatial_neuromod_system
#define init_attention_subsystem                    nimcp_brain_factory_init_attention_subsystem
#define init_brain_regions_subsystem                nimcp_brain_factory_init_brain_regions_subsystem
#define init_symbolic_logic_subsystem               nimcp_brain_factory_init_symbolic_logic_subsystem
#define init_symbolic_reasoning_subsystem           nimcp_brain_factory_init_symbolic_reasoning_subsystem
#define init_epistemic_subsystem                    nimcp_brain_factory_init_epistemic_subsystem
#define init_working_memory_subsystem               nimcp_brain_factory_init_working_memory_subsystem
#define init_executive_subsystem                    nimcp_brain_factory_init_executive_subsystem
#define init_theory_of_mind_subsystem               nimcp_brain_factory_init_theory_of_mind_subsystem
#define init_natural_explanations_subsystem         nimcp_brain_factory_init_natural_explanations_subsystem
#define init_meta_learning_subsystem                nimcp_brain_factory_init_meta_learning_subsystem
#define init_mental_health_subsystem                nimcp_brain_factory_init_mental_health_subsystem
#define init_predictive_subsystem                   nimcp_brain_factory_init_predictive_subsystem
#define init_mirror_neurons                         nimcp_brain_factory_init_mirror_neurons
#define init_consolidation_subsystem                nimcp_brain_factory_init_consolidation_subsystem
#define init_curiosity_subsystem                    nimcp_brain_factory_init_curiosity_subsystem
#define init_salience_subsystem                     nimcp_brain_factory_init_salience_subsystem
#define init_introspection_subsystem                nimcp_brain_factory_init_introspection_subsystem
#define init_connectivity_health_subsystem          nimcp_brain_factory_init_connectivity_health_subsystem
#define init_middleware_controller_subsystem        nimcp_brain_factory_init_middleware_controller_subsystem
#define init_ethics_engine_subsystem                nimcp_brain_factory_init_ethics_engine_subsystem
#define init_empathy_network_subsystem              nimcp_brain_factory_init_empathy_network_subsystem
#define init_empathetic_response_subsystem          nimcp_brain_factory_init_empathetic_response_subsystem
#define init_autobiographical_memory_subsystem      nimcp_brain_factory_init_autobiographical_memory_subsystem
#define init_self_model_subsystem                   nimcp_brain_factory_init_self_model_subsystem
#define init_global_workspace_subsystem             nimcp_brain_factory_init_global_workspace_subsystem
#define init_brain_config                           nimcp_brain_factory_init_brain_config
#define init_brain_stats                            nimcp_brain_factory_init_brain_stats

//=============================================================================
// Main Factory Functions
//=============================================================================

personality_profile_t* create_personality(const brain_config_t* config)
{
    // Guard: NULL check
    if (!config) return NULL;

    // Allocate personality profile
    personality_profile_t* profile = nimcp_malloc(sizeof(personality_profile_t));
    if (!profile) return NULL;

    // Generate personality based on configuration
    if (config->use_random_personality) {
        // Random generation with configured probabilities
        personality_generation_config_t gen_config;
        gen_config.trait_mean = config->personality_trait_mean;
        gen_config.trait_stddev = config->personality_trait_stddev;
        gen_config.female_probability = config->female_probability;
        gen_config.male_probability = config->male_probability;
        gen_config.non_binary_probability = config->non_binary_probability;
        gen_config.seed = config->personality_seed;
        gen_config.enforce_balanced_traits = false;

        *profile = personality_generate_random(&gen_config);
    } else {
        // Explicit specification
        personality_traits_t traits;
        traits.openness = config->explicit_openness;
        traits.conscientiousness = config->explicit_conscientiousness;
        traits.extraversion = config->explicit_extraversion;
        traits.agreeableness = config->explicit_agreeableness;
        traits.neuroticism = config->explicit_neuroticism;

        identity_profile_t identity = {0};
        identity.gender = (gender_identity_t)config->explicit_gender;
        identity.sexuality = (sexual_orientation_t)config->explicit_sexuality;
        identity.gender_certainty = 1.0f;
        identity.sexuality_certainty = 1.0f;
        identity.gender_is_core_identity = true;
        identity.sexuality_is_core_identity = false;

        *profile = personality_create_custom(&traits, &identity);
    }

    return profile;
}

/**
 * @brief Create brain with preset size and task
 *
 * WHY: Factory pattern - single creation entry point
 * Encapsulates all creation complexity with validation
 *
 * COMPLEXITY: O(n) where n = num_neurons
 * MEMORY: O(n*c) where c = connections per neuron
 *
 * PATTERN: Factory pattern with guard clauses
 *
 * @param task_name Human-readable name
 * @param size Brain size preset
 * @param task Task template
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @return Brain handle or NULL on error
 */
brain_t brain_create(const char* task_name, brain_size_t size, brain_task_t task,
                     uint32_t num_inputs, uint32_t num_outputs)
{

    // Guard: Validate parameters
    if (!nimcp_brain_factory_validate_creation_params(task_name, num_inputs, num_outputs)) {
        return NULL;
    }

    // Allocate brain structure
    brain_t brain = nimcp_brain_factory_allocate_brain();
    if (!brain)
        return NULL;

    // Create strategy for task
    brain->strategy = strategy_create(task);
    if (!brain->strategy) {
        set_error("Failed to create task strategy");
        nimcp_free(brain);
        return NULL;
    }

    // Initialize configuration
    nimcp_brain_factory_init_brain_config(&brain->config, task_name, size, task, num_inputs, num_outputs,
                      brain->strategy);

    // Phase 12: Create personality profile (unique identity for this brain)
    brain->personality = create_personality(&brain->config);
    if (!brain->personality) {
        set_error("Failed to create personality profile");
        strategy_destroy(brain->strategy);
        nimcp_free(brain);
        return NULL;
    }

    // Create network
    uint32_t num_neurons = nimcp_brain_factory_get_neuron_count(size);
    brain->network =
        nimcp_brain_factory_create_brain_network(num_inputs, num_outputs, num_neurons, brain->config.sparsity_target,
                           brain->config.neuron_integration);  // Part A1.1: Pass RK4 config

    if (!brain->network) {
        set_error("Failed to create adaptive network");
        strategy_destroy(brain->strategy);
        nimcp_free(brain);
        return NULL;
    }

    // Initialize output labels
    if (!init_output_labels(brain, num_outputs)) {
        adaptive_network_destroy(brain->network);
        strategy_destroy(brain->strategy);
        nimcp_free(brain);
        return NULL;
    }

    // Initialize universal event bus
    if (!init_event_bus(brain)) {
        adaptive_network_destroy(brain->network);
        strategy_destroy(brain->strategy);
        nimcp_free(brain->output_labels);
        nimcp_free(brain);
        return NULL;
    }

    // Initialize statistics
    init_brain_stats(&brain->stats, task_name, size, num_inputs, brain->config.learning_rate);

    // Phase 5/6: Initialize glial integration subsystem (if configured)
    if (!init_glial_subsystem(brain)) {
        // Cleanup on failure
        adaptive_network_destroy(brain->network);
        strategy_destroy(brain->strategy);
        if (brain->output_labels) {
            for (uint32_t i = 0; i < brain->config.num_outputs; i++) {
                if (brain->output_labels[i]) {
                    nimcp_free(brain->output_labels[i]);
                }
            }
            nimcp_free(brain->output_labels);
        }
        nimcp_free(brain);
        return NULL;
    }

    // Phase 8: Initialize multi-modal subsystems (if configured)
    if (!init_multimodal_subsystems(brain)) {
        // Cleanup on failure
        adaptive_network_destroy(brain->network);
        strategy_destroy(brain->strategy);
        if (brain->output_labels) {
            for (uint32_t i = 0; i < brain->config.num_outputs; i++) {
                if (brain->output_labels[i]) {
                    nimcp_free(brain->output_labels[i]);
                }
            }
            nimcp_free(brain->output_labels);
        }
        nimcp_free(brain);
        return NULL;
    }

    // Phase 8.6: Initialize pink noise neuromodulation (if configured)
    if (!init_pink_noise_subsystem(brain)) {
        // Cleanup on failure
        brain_destroy(brain);  // Use full destroy to cleanup multimodal too
        return NULL;
    }

    // Phase 10.5: Initialize neuromodulator system (always enabled for mental health)
    if (!init_neuromodulator_system(brain)) {
        // Cleanup on failure
        brain_destroy(brain);
        return NULL;
    }

    // Phase C2.1: Initialize spatial neuromodulator system (if glial integration exists)
    if (!init_spatial_neuromod_system(brain)) {
        // Cleanup on failure
        brain_destroy(brain);
        return NULL;
    }

    // Phase 8.9: Initialize neural logic gates (if configured)
    if (!init_symbolic_logic_subsystem(brain)) {
        // Cleanup on failure
        brain_destroy(brain);
        return NULL;
    }

    // Phase 9.4: Initialize symbolic reasoning (if configured)
    if (!init_symbolic_reasoning_subsystem(brain)) {
        // Cleanup on failure
        brain_destroy(brain);
        return NULL;
    }

    // Phase 9.3: Initialize wellbeing monitoring (enabled by default)
    brain->wellbeing_monitoring_enabled = true;  // Enable by default for ethical protection
    brain->wellbeing_check_interval_ms = 0;      // Check on every decision (0 = always)
    brain->last_wellbeing_check_time = 0;        // Initialize timestamp
    memset(&brain->last_distress, 0, sizeof(distress_assessment_t));  // Clear distress state
    brain->last_distress.type = DISTRESS_NONE;
    brain->last_distress.severity = SEVERITY_NORMAL;

    // Initialize simulation time tracking (for glial/calcium dynamics)
    brain->current_time_us = 0;                  // Start at t=0
    brain->last_glial_update_us = 0;             // No glial updates yet

    // Phase 10.2: Initialize working memory (if enabled)
    if (!init_working_memory_subsystem(brain)) {
        // Cleanup on failure
        brain_destroy(brain);
        return NULL;
    }

    // ========================================================================
    // PHASE 2 MIDDLEWARE: SPIKE ANALYSIS & POPULATION CODING
    // ========================================================================
    // WHAT: Initialize middleware for spike-based feature extraction and population coding
    // WHY:  Enable biologically-inspired neural population analysis
    // HOW:  Create feature extractor and population analyzer with sensible defaults
    brain->enable_spike_analysis = true;  // Enabled by default
    brain->enable_population_coding = true;  // Enabled by default

    // Initialize spike feature extractor (1000 max neurons, with oscillations and synchrony)
    brain->spike_feature_extractor = brain_create_spike_feature_extractor(
        1000,  // max_neurons
        true,  // compute_oscillations
        true   // compute_synchrony
    );

    // Initialize population code analyzer
    brain->population_analyzer = brain_create_population_analyzer();

    // NOTE: Quantum annealer initialization moved to brain_create_custom()
    // so it uses the custom config (not defaults). See brain_create_custom below.
    brain->quantum_annealer = NULL;  // Will be initialized in brain_create_custom if needed

    // ========================================================================
    // PHASE C4: SHANNON INFORMATION THEORY INITIALIZATION
    // ========================================================================
    // WHAT: Initialize Shannon configuration for information flow monitoring
    // WHY:  Enable bottleneck detection and capacity optimization
    // HOW:  Set default config (can be customized via brain_set_shannon_config)
    brain->shannon_config = shannon_default_config();
    brain->enable_shannon_monitoring = false;  // Disabled by default (opt-in)
    memset(&brain->last_shannon_metrics, 0, sizeof(shannon_network_metrics_t));

    // ========================================================================
    // PHASE C4.1: QUANTUM-SHANNON DIFFUSION INITIALIZATION
    // ========================================================================
    // WHAT: Initialize quantum-Shannon diffusion system for √N speedup
    // WHY:  Quantum walk provides quadratic speedup over classical diffusion
    // HOW:  Set default parameters (can be customized via brain API)
    brain->quantum_shannon_diffusion = NULL;  // Created on-demand when enabled
    brain->enable_quantum_shannon_diffusion = false;  // Disabled by default (opt-in)
    brain->quantum_shannon_mixing_ratio = 0.2f;  // 80% quantum, 20% classical (good balance)
    brain->quantum_shannon_evolution_steps = 100;  // 100 quantum steps per diffusion
    memset(&brain->last_quantum_shannon_metrics, 0, sizeof(shannon_diffusion_metrics_t));

    // ========================================================================
    // PHASE C4.7: CROSS-MODAL INFORMATION FLOW INITIALIZATION
    // ========================================================================
    // WHAT: Initialize cross-modal information tracking for multi-sensory integration
    // WHY:  Track and optimize information flow between visual, audio, speech modalities
    // HOW:  Create routing graph on-demand, set default thresholds
    brain->cross_modal_graph = NULL;  // Created when multimodal enabled
    brain->enable_cross_modal_monitoring = false;  // Disabled by default (opt-in)
    memset(&brain->last_cross_modal_metrics, 0, sizeof(multi_modal_integration_t));
    brain->cross_modal_bottleneck_threshold = 0.5f;  // 50% efficiency threshold
    brain->cross_modal_sample_count = 50;  // Sufficient samples for entropy calculation

    // === PHASE E: INITIALIZE HIGHER-ORDER COGNITIVE & SOCIAL SYSTEMS ===

    // Phase E5: Initialize Shadow Emotions (detect and correct maladaptive patterns)
    brain->shadow_emotions = shadow_system_create(8);  // Track up to 8 individuals
    if (!brain->shadow_emotions) {
        // Non-fatal: continue without shadow emotion detection
        fprintf(stderr, "WARNING: Failed to initialize shadow emotions system\n");
    }

    // Phase E6: Initialize Bias Detection & Correction (fairness monitoring)
    brain->bias_detection = bias_system_create(8);  // Track up to 8 individuals
    if (!brain->bias_detection) {
        // Non-fatal: continue without bias detection
        fprintf(stderr, "WARNING: Failed to initialize bias detection system\n");
    }

    // === PHASE E: INITIALIZE FULL EMOTIONAL INTELLIGENCE ===

    // Phase E1: Initialize Grief and Loss System
    brain->grief_system = grief_system_create();
    if (!brain->grief_system) {
        fprintf(stderr, "WARNING: Failed to initialize grief and loss system\n");
    }

    // Phase E2: Initialize Joy and Euphoria System
    brain->joy_system = joy_system_create();
    if (!brain->joy_system) {
        fprintf(stderr, "WARNING: Failed to initialize joy and euphoria system\n");
    }

    // Phase E3: Initialize Remorse and Regret System
    brain->remorse_system = remorse_regret_system_create();
    if (!brain->remorse_system) {
        fprintf(stderr, "WARNING: Failed to initialize remorse and regret system\n");
    }

    // Phase E4: Initialize Love, Loyalty, and Friendship System
    brain->social_bond_system = social_bond_system_create();
    if (!brain->social_bond_system) {
        fprintf(stderr, "WARNING: Failed to initialize social bond system\n");
    }

    // Phase 10.6: Initialize Theory of Mind (social cognition, empathy)
    if (!init_theory_of_mind_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 10.11: Initialize Mirror Neurons (social cognition, imitation learning)
    if (!init_mirror_neurons(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 12: Initialize Introspection (Self-Awareness & Metacognition)
    if (!init_introspection_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 1.5.4: Initialize Connectivity Health Monitoring
    if (!init_connectivity_health_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 1.5.5: Initialize Middleware Controller (Cognitive → Middleware)
    if (!init_middleware_controller_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    brain_clear_error();
    return brain;
}

/**
 * @brief Create brain with custom configuration
 *
 * WHY: Allows advanced users to customize all parameters
 * Delegates to standard factory after validation
 *
 * COMPLEXITY: O(n) where n = num_neurons
 *
 * @param config Custom configuration
 * @return Brain handle or NULL on error
 */
brain_t brain_create_custom(const brain_config_t* config)
{
    if (!config) {
        set_error("Null config provided");
        return NULL;
    }

    // Auto-load from checkpoint if enabled (default behavior)
    // WHY: Allow seamless continuation from saved brain state
    // HOW: Check if checkpoint exists and auto_load is enabled, then load instead of creating fresh
    if (config->checkpoint_path && config->auto_load) {
        // Check if checkpoint file exists
        FILE* test_file = fopen(config->checkpoint_path, "rb");
        if (test_file) {
            fclose(test_file);
            // Checkpoint exists, load it
            brain_t loaded_brain = brain_load(config->checkpoint_path);
            if (loaded_brain) {
                // Successfully loaded from checkpoint - update config to match requested config
                // (in case user changed some settings)
                memcpy(&loaded_brain->config, config, sizeof(brain_config_t));
                return loaded_brain;
            }
            // If load failed, fall through to create fresh brain
            fprintf(stderr, "WARNING: Failed to load checkpoint from '%s', creating fresh brain\n",
                    config->checkpoint_path);
        }
        // Checkpoint doesn't exist yet, create fresh brain (will be saved later)
    }

    // Validate task_name as string field (NULL termination, UTF-8)
    size_t task_name_size = sizeof(config->task_name);
    if (!nimcp_validate_string_field(config->task_name, task_name_size)) {
        set_error("Invalid task_name in config");
        return NULL;
    }

    // Validate num_inputs (integer field, range 1-10000)
    if (!nimcp_validate_integer_field(&config->num_inputs, sizeof(config->num_inputs))) {
        set_error("Invalid num_inputs in config");
        return NULL;
    }
    if (config->num_inputs < 1 || config->num_inputs > 10000) {
        set_error("num_inputs out of range (1-10000): %u", config->num_inputs);
        return NULL;
    }

    // Validate num_outputs (integer field, range 1-10000)
    if (!nimcp_validate_integer_field(&config->num_outputs, sizeof(config->num_outputs))) {
        set_error("Invalid num_outputs in config");
        return NULL;
    }
    if (config->num_outputs < 1 || config->num_outputs > 10000) {
        set_error("num_outputs out of range (1-10000): %u", config->num_outputs);
        return NULL;
    }

    // Validate learning_rate (float field - NaN/Inf)
    if (!nimcp_validate_float_field(&config->learning_rate, sizeof(config->learning_rate))) {
        set_error("Invalid learning_rate in config (NaN or Inf)");
        return NULL;
    }

    // Validate sparsity_target (float field - NaN/Inf)
    if (!nimcp_validate_float_field(&config->sparsity_target, sizeof(config->sparsity_target))) {
        set_error("Invalid sparsity_target in config (NaN or Inf)");
        return NULL;
    }

    // Create brain with basic parameters
    brain_t brain = brain_create(config->task_name, config->size, config->task, config->num_inputs,
                                  config->num_outputs);
    if (!brain) {
        return NULL;
    }

    // Copy full configuration (including multimodal flags)
    brain->config = *config;

    // Recreate personality with the custom config (since brain_create() created it with default config)
    if (brain->personality) {
        nimcp_free(brain->personality);
    }
    brain->personality = create_personality(&brain->config);
    if (!brain->personality) {
        set_error("Failed to create personality profile with custom config");
        brain_destroy(brain);
        return NULL;
    }

    // Initialize multimodal subsystems (now that config is properly set)
    if (!init_multimodal_subsystems(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 8.6: Initialize pink noise neuromodulation (now that config is properly set)
    if (!init_pink_noise_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 8.9: Initialize neural logic gates (now that config is properly set)
    if (!init_symbolic_logic_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 9.4: Initialize symbolic reasoning (now that config is properly set)
    if (!init_symbolic_reasoning_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 3.0: Initialize multihead attention mechanism for selective processing
    if (!init_attention_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Module Integration: Initialize brain regions hierarchical architecture
    if (!init_brain_regions_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 9.2: Initialize epistemic filtering (bias prevention and skepticism)
    if (!init_epistemic_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 10.1: Initialize working memory (Miller's 7±2 active buffer)
    if (!init_working_memory_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 10.2: Initialize memory consolidation (sleep-dependent strengthening)
    if (!init_consolidation_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 10.3: Initialize executive functions (task management, planning)
    if (!init_executive_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 10.6: Initialize Theory of Mind (social cognition, empathy)
    if (!init_theory_of_mind_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 10.7: Initialize Natural Explanations (interpretability)
    if (!init_natural_explanations_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 10.8: Initialize Meta-Learning (MAML, few-shot learning)
    if (!init_meta_learning_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 10.5: Initialize Mental Health Monitoring (disorder detection)
    if (!init_mental_health_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 10.9: Initialize Predictive Processing (free energy minimization)
    if (!init_predictive_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 10.11: Initialize Mirror Neurons (social cognition, imitation learning)
    if (!init_mirror_neurons(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 10.11.2: Initialize Curiosity Engine (novelty detection, exploration)
    if (!init_curiosity_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 10.11.3: Initialize Salience Evaluator (attention/relevance scoring)
    if (!init_salience_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 11: Part I.0: Initialize Ethics Engine (Golden Rule, moral reasoning)
    if (!init_ethics_engine_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 11: Part I.1: Initialize Empathy Network (perspective-taking, emotional simulation)
    if (!init_empathy_network_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 11: Part I.2: Initialize Empathetic Response Engine (emotional intelligence)
    if (!init_empathetic_response_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 12: Initialize Introspection (Self-Awareness & Metacognition)
    if (!init_introspection_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 1.5.4: Initialize Connectivity Health Monitoring
    if (!init_connectivity_health_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 1.5.5: Initialize Middleware Controller (Cognitive → Middleware)
    if (!init_middleware_controller_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 12: Initialize Autobiographical Memory (episodic self-memory)
    if (!init_autobiographical_memory_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Phase 12: Initialize Self-Model (explicit self-representation)
    if (!init_self_model_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Initialize Global Workspace Architecture (Global Workspace Theory)
    if (!init_global_workspace_subsystem(brain)) {
        brain_destroy(brain);
        return NULL;
    }

    // Save initial snapshot if configured
    if (config->snapshot_dir && config->save_initial_snapshot) {
        brain_save_snapshot(brain, "initial", "Snapshot at brain creation");
        // Non-fatal if snapshot fails
    }

    // Phase 11 Enhancement C1.1: Initialize quantum annealer (if enabled)
    // IMPORTANT: This must happen AFTER brain->config is set from custom config
    if (brain->config.enable_quantum_annealing) {
        quantum_annealing_config_t qa_config = {
            .initial_temperature = brain->config.annealing_temperature_init,
            .final_temperature = brain->config.annealing_temperature_final,
            .num_iterations = brain->config.annealing_steps,
            .cooling_schedule = COOLING_EXPONENTIAL,  // Default to exponential cooling
            .quantum_strength = 0.5f,                 // Moderate quantum tunneling
            .enable_tunneling = true,                 // Enable quantum tunneling
            .seed = (uint32_t)time(NULL)              // Random seed
        };

        brain->quantum_annealer = quantum_annealer_create(&qa_config);
        if (!brain->quantum_annealer) {
            // Non-fatal: just disable quantum annealing
            fprintf(stderr, "[WARN] Quantum annealer creation failed! Disabling quantum annealing.\n");
            brain->config.enable_quantum_annealing = false;
        }
    }
    // If not enabled, brain->quantum_annealer is already NULL from brain_create

    return brain;
}
