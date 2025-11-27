//=============================================================================
// nimcp_brain_init.c - Brain Initialization and Subsystem Setup
//=============================================================================
/**
 * @file nimcp_brain_init.c
 * @brief Brain initialization and subsystem setup functions
 *
 * WHAT: All brain structure and subsystem initialization logic
 * WHY:  Separates initialization from creation orchestration
 * HOW:  Modular initialization functions for each brain subsystem
 *
 * ARCHITECTURE:
 * - Configuration builders for network and parameters
 * - Memory allocation and structure setup (allocate_brain)
 * - 31 subsystem initialization functions
 * - Network creation and output label setup
 *
 * EXTRACTED FROM: nimcp_brain_factory.c
 * DATE: 2025-11-19
 *
 * @version 2.7.0
 * @author NIMCP Development Team
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init.h"
#include "core/brain/factory/nimcp_brain_factory.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
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
#include "glial/myelin_sheath/nimcp_myelin_sheath.h"       // Myelin sheath structural modeling
#include "core/brain_oscillations/nimcp_brain_oscillations.h"
#include "cognitive/introspection/nimcp_introspection.h"
#include "cognitive/introspection/nimcp_connectivity_health.h"  // Phase 1.5.4: Connectivity Health
#include "middleware/integration/nimcp_middleware_controller.h" // Phase 1.5.5: Middleware Controller
#include "core/axon/nimcp_axon.h"                               // Phase 1.5.6: Axon Integration
#include "core/dendrite/nimcp_dendrite.h"                       // Phase 1.5.7: Dendrite Integration
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
#include "perception/nimcp_visual_cortex.h"
#include "perception/nimcp_audio_cortex.h"
#include "perception/nimcp_speech_cortex.h"
#include "nlp/nimcp_nlp.h"

// Brain regions and cognitive systems
#include "core/brain_regions/nimcp_brain_regions.h"
#include "core/events/nimcp_event_bus.h"
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
#include "cognitive/global_workspace/nimcp_global_workspace_shannon.h"  /* Phase 1.5.3: Shannon info-weighted competition */
#include "cognitive/nimcp_autobiographical_memory.h"
#include "core/events/nimcp_event_bus.h"  // Universal event bus for ALL brain activities

// Phase SC-2: Security-Fault Tolerance Integration
#include "security/nimcp_security_recovery_bridge.h"

// Phase SC-4: Universal Security Integration Framework
#include "security/nimcp_security_integration.h"

// Phase IS-1: Blood-Brain Barrier (Perimeter Defense)
#include "security/nimcp_blood_brain_barrier.h"

// Phase TM-3: Brain-Training Integration
#include "middleware/training/nimcp_brain_training_integration.h"

//=============================================================================
// Global BBB System (Singleton with Reference Counting)
//=============================================================================

static bbb_system_t g_bbb_system = NULL;
static uint32_t g_bbb_refcount = 0;
static nimcp_platform_mutex_t g_bbb_mutex;
static bool g_bbb_mutex_initialized = false;

/**
 * @brief Get or create the global BBB system (thread-safe)
 * @return BBB system handle, or NULL on failure
 */
static bbb_system_t get_global_bbb_system(void)
{
    // Lazy mutex initialization
    if (!g_bbb_mutex_initialized) {
        if (nimcp_platform_mutex_init(&g_bbb_mutex, false) != 0) {
            LOG_ERROR("Failed to initialize BBB mutex");
            return NULL;
        }
        g_bbb_mutex_initialized = true;
    }

    nimcp_platform_mutex_lock(&g_bbb_mutex);

    if (!g_bbb_system) {
        // Create with default conservative configuration
        bbb_config_t config = bbb_default_config();
        g_bbb_system = bbb_system_create(&config);
        if (g_bbb_system) {
            LOG_INFO("Global BBB system created (Phase IS-1)");
        } else {
            LOG_WARNING("Failed to create global BBB system");
        }
    }

    if (g_bbb_system) {
        g_bbb_refcount++;
        LOG_DEBUG("BBB system refcount incremented to %u", g_bbb_refcount);
    }

    nimcp_platform_mutex_unlock(&g_bbb_mutex);
    return g_bbb_system;
}

/**
 * @brief Release reference to global BBB system (thread-safe)
 * NOTE: Non-static to allow brain_destroy() to call this
 */
void nimcp_bbb_release_global_system(void)
{
    if (!g_bbb_mutex_initialized) {
        return;
    }

    nimcp_platform_mutex_lock(&g_bbb_mutex);

    if (g_bbb_refcount > 0) {
        g_bbb_refcount--;
        LOG_DEBUG("BBB system refcount decremented to %u", g_bbb_refcount);

        if (g_bbb_refcount == 0 && g_bbb_system) {
            bbb_system_destroy(g_bbb_system);
            g_bbb_system = NULL;
            LOG_INFO("Global BBB system destroyed (no more references)");
        }
    }

    nimcp_platform_mutex_unlock(&g_bbb_mutex);
}

/**
 * @brief Get the global BBB system for cross-module protection (thread-safe)
 *
 * WHAT: Returns the shared BBB system instance without incrementing refcount
 * WHY:  Enables all modules to use consistent perimeter security for validation
 * HOW:  Thread-safe read access to global BBB system pointer
 *
 * NOTE: This function does NOT increment the reference count. The returned
 *       handle is valid as long as at least one brain with BBB enabled exists.
 *       For standalone usage, call this after brain_create() with BBB enabled.
 *
 * @return BBB system handle, or NULL if not initialized
 */
bbb_system_t nimcp_bbb_get_global_system(void)
{
    if (!g_bbb_mutex_initialized) {
        return NULL;
    }

    nimcp_platform_mutex_lock(&g_bbb_mutex);
    bbb_system_t result = g_bbb_system;
    nimcp_platform_mutex_unlock(&g_bbb_mutex);

    return result;
}

//=============================================================================
// Configuration and Initialization Functions
//=============================================================================

uint32_t nimcp_brain_factory_get_neuron_count(brain_size_t size)
{
    switch (size) {
        case BRAIN_SIZE_TINY:
            return 100;
        case BRAIN_SIZE_SMALL:
            return 500;
        case BRAIN_SIZE_MEDIUM:
            return 1000;  // Reduced from 10000 for faster tests (1.8GB→180MB)
        case BRAIN_SIZE_LARGE:
            return 5000;  // Reduced from 100000 (9GB→450MB)
        case BRAIN_SIZE_CUSTOM:
            return 1000;
        default:
            return 1000;
    }
}

/**
 * @brief Get default sparsity target for size
 *
 * WHY: Larger networks need higher sparsity for efficiency
 * Balances performance and memory
 *
 * COMPLEXITY: O(1)
 *
 * @param size Brain size preset
 * @return Sparsity target (0.0-1.0)
 */
float nimcp_brain_factory_get_default_sparsity(brain_size_t size)
{
    switch (size) {
        case BRAIN_SIZE_TINY:
            return 0.70f;
        case BRAIN_SIZE_SMALL:
            return 0.80f;
        case BRAIN_SIZE_MEDIUM:
            return 0.85f;
        case BRAIN_SIZE_LARGE:
            return 0.90f;
        default:
            return 0.80f;
    }
}

//=============================================================================
// Configuration Builders
//=============================================================================

/**
 * @brief Build spike parameters for brain configuration
 *
 * WHY: Separates spike config from main creation logic
 * Makes configuration more maintainable and testable
 *
 * COMPLEXITY: O(1)
 *
 * @param sparsity_target Target sparsity level
 * @return Spike parameters structure
 */
adaptive_spike_params_t nimcp_brain_factory_build_spike_params(float sparsity_target)
{
    adaptive_spike_params_t params = {0};
    params.k_factor = 0.5f;
    params.sparsity_target = sparsity_target;
    params.encoding = SPIKE_ENCODING_INTEGER;
    params.enable_soft_reset = true;
    params.enable_adaptation = true;
    params.adaptation_window = 100;
    params.min_threshold = 0.0001f;  // Very low to allow tiny outputs from untrained networks
    params.max_threshold = 10.0f;
    return params;
}

/**
 * @brief Build base network configuration
 *
 * WHY: Isolates network config from brain config
 * Enables reuse and testing of network setup
 *
 * COMPLEXITY: O(1) + memory allocation
 *
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @param num_neurons Total neuron count
 * @return Base network config (caller must free layer_sizes)
 */
network_config_t nimcp_brain_factory_build_base_network_config(uint32_t num_inputs, uint32_t num_outputs,
                                                  uint32_t num_neurons,
                                                  ode_integration_method_t integration_method)
{
    network_config_t config = {0};

    // Validation: Check for zero parameters
    if (num_inputs == 0) {
        set_error("Invalid network configuration: num_inputs cannot be zero");
        return config;  // Return with layer_sizes = NULL to signal error
    }
    if (num_outputs == 0) {
        set_error("Invalid network configuration: num_outputs cannot be zero");
        return config;  // Return with layer_sizes = NULL to signal error
    }
    if (num_neurons == 0) {
        set_error("Invalid network configuration: num_neurons cannot be zero");
        return config;  // Return with layer_sizes = NULL to signal error
    }

    config.input_size = num_inputs;
    config.output_size = num_outputs;
    config.num_neurons = num_neurons;
    config.num_layers = 3;
    config.integration_method = integration_method;  // Part A1.1: Pass through RK4 config

    config.layer_sizes = nimcp_calloc(3, sizeof(uint32_t));
    // Guard: Check allocation
    // WHY: If allocation fails, returning config with NULL layer_sizes will crash
    if (!config.layer_sizes) {
        set_error("Failed to allocate layer_sizes array");
        return config;  // Return with layer_sizes = NULL to signal error
    }

    config.layer_sizes[0] = num_inputs;
    config.layer_sizes[1] = num_neurons;
    config.layer_sizes[2] = num_outputs;

    config.enable_stdp = true;
    config.enable_hebbian = true;
    config.enable_oja = true;
    config.enable_homeostasis = true;

    // SCALABILITY: Disable BCM and eligibility traces by default
    // WHY: These require per-synapse heap allocation
    // IMPACT: With 1M neurons × 256 synapses = 256M allocations
    // SOLUTION: Only enable when explicitly configured by brain_config
    config.enable_bcm = false;          // Conditional BCM allocation
    config.enable_eligibility = false;  // Conditional eligibility allocation

    return config;
}

/**
 * @brief Build complete adaptive network configuration
 *
 * WHY: Combines base config and spike params
 * Single point of network configuration assembly
 *
 * COMPLEXITY: O(1) + memory allocation
 *
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @param num_neurons Neuron count
 * @param sparsity_target Target sparsity
 * @return Complete adaptive network config
 */
adaptive_network_config_t nimcp_brain_factory_build_network_config(uint32_t num_inputs, uint32_t num_outputs,
                                                      uint32_t num_neurons, float sparsity_target,
                                                      ode_integration_method_t integration_method)
{
    adaptive_network_config_t config = {0};

    config.base_config = nimcp_brain_factory_build_base_network_config(num_inputs, num_outputs, num_neurons, integration_method);

    config.spike_params = nimcp_brain_factory_build_spike_params(sparsity_target);

    config.enable_sparsity = false;  // Disabled for regression tests - untrained networks produce zeros
    config.pruning_threshold = 0.01f;
    config.update_frequency = 100;

    return config;
}

void nimcp_brain_factory_init_brain_config(brain_config_t* config, const char* task_name, brain_size_t size,
                              brain_task_t task, uint32_t num_inputs, uint32_t num_outputs,
                              task_strategy_t* strategy)
{
    // Guard: NULL check
    if (!config || !strategy) {
        fprintf(stderr, "[ERROR] nimcp_brain_factory_init_brain_config: NULL config or strategy pointer\n");
        return;
    }

    config->size = size;
    config->task = task;
    config->num_inputs = num_inputs;
    config->num_outputs = num_outputs;
    config->learning_rate = strategy->get_learning_rate();
    config->sparsity_target = nimcp_brain_factory_get_default_sparsity(size);
    config->enable_explanations = true;
    strncpy(config->task_name, task_name, sizeof(config->task_name) - 1);

    // Part A: Differential Equations - ODE Integration Method (A1.x)
    config->neuron_integration = ODE_EULER;  // Default: Fast Euler (backward compatible)

    // Phase 10.2: Working Memory defaults (Miller's 7±2)
    config->enable_working_memory = true;           // Enable by default
    config->working_memory_capacity = 7;            // Miller's magic number
    config->working_memory_decay_tau_ms = 1000.0f;  // 1 second decay

    // Phase 10.6: Theory of Mind defaults (social cognition, empathy)
    config->enable_theory_of_mind = true;           // Enable by default for social cognition
    config->enable_empathy_responses = true;        // Enable empathetic responses
    config->enable_false_belief_tracking = true;    // Enable false belief understanding

    // Phase 10.11: Mirror Neurons defaults (observation-based learning)
    config->enable_mirror_neurons = true;           // Enable by default for social learning
    config->mirror_neuron_count = 1000;             // Standard population size
    config->mirror_max_actions = 100;               // Diverse action repertoire
    config->mirror_max_agents = 10;                 // Multi-agent social learning
    config->mirror_learning_rate = 0.01f;           // Hebbian association rate
    config->mirror_match_threshold = 0.7f;          // Action recognition threshold

    // Global Workspace Architecture defaults (Global Workspace Theory - Baars 1988)
    config->enable_global_workspace = true;         // Enable by default for conscious access
    config->workspace_capacity_dim = 256;           // Content dimension (256 floats)
    config->workspace_ignition_threshold = 0.6f;    // Threshold for conscious access
    config->workspace_refractory_ms = 50;           // 50ms refractory between broadcasts
    config->workspace_enable_history = true;        // Enable history tracking
    config->workspace_history_depth = 10;           // Last 10 broadcasts

    // Phase 11 Enhancement C1.1: Quantum Annealing defaults
    config->enable_quantum_annealing = false;       // Disable by default (opt-in for optimization)
    config->annealing_temperature_init = 10.0f;     // Initial exploration temperature
    config->annealing_temperature_final = 0.1f;     // Final exploitation temperature
    config->annealing_steps = 1000;                 // Number of optimization steps
    config->quantum_annealing_frequency = 100;      // Run every 100 learning steps

    // Phase 12: Personality and Identity defaults
    config->use_random_personality = true;          // Default: generate random personality
    config->personality_seed = 0;                   // Time-based seed for uniqueness
    config->explicit_openness = 0.5f;               // Moderate openness (if explicit)
    config->explicit_conscientiousness = 0.5f;      // Moderate conscientiousness (if explicit)
    config->explicit_extraversion = 0.5f;           // Moderate extraversion (if explicit)
    config->explicit_agreeableness = 0.5f;          // Moderate agreeableness (if explicit)
    config->explicit_neuroticism = 0.5f;            // Moderate neuroticism (if explicit)
    config->explicit_gender = GENDER_FEMALE;        // Default: female (per user request)
    config->explicit_sexuality = SEXUALITY_HETEROSEXUAL; // Default: heterosexual
    config->personality_trait_mean = 0.5f;          // Mean for random trait generation
    config->personality_trait_stddev = 0.15f;       // Stddev for random trait generation
    config->female_probability = 1.0f;              // Default 100% female (per user request)
    config->male_probability = 0.0f;                // 0% male by default
    config->non_binary_probability = 0.0f;          // 0% non-binary by default

    // Phase 5/6: Biological Realism defaults
    config->enable_glial = true;                    // Enable glial integration by default
    config->enable_oscillations = false;            // Disable oscillations by default (opt-in)
    config->enable_myelin_sheath = true;            // Enable myelin sheath by default (works with glial)

    // Lazy Initialization defaults (performance optimization)
    // WHY: Heavy subsystems like dendrites/axons can take 10-30s to initialize
    // For tests that don't need them, lazy init saves significant time
    config->lazy_dendrite_init = false;             // Eager init by default (biological realism)
    config->lazy_axon_init = false;                 // Eager init by default
    config->enable_dendrites = true;                // Dendrites enabled by default
    config->enable_axons = true;                    // Axons enabled by default

    // Phase C2.1: Quantum Walk defaults (disabled by default for stability/testing)
    config->enable_quantum_walk_diffusion = false;  // Opt-in: requires testing for production
    config->quantum_walk_steps = 50;                // Moderate steps for √N speedup
    config->quantum_classical_mixing = 0.2f;        // 80% quantum + 20% classical (hybrid)
    config->quantum_coin_type = 0;                  // 0=Hadamard (balanced superposition)
    config->quantum_decoherence_rate = 0.05f;       // Minimal decoherence (5% per step)

    // Phase TM-3: Brain-Training Integration defaults
    config->enable_training_integration = true;     // Enabled by default for full functionality
    config->training_register_security = true;      // Register with security system
    config->training_default_lr = 0.0f;             // 0 = use brain's learning_rate

    // Phase TM-4/5/6: Advanced Training Pipeline defaults
    // LR Scheduler: Automatically adjust learning rate during training
    config->enable_lr_scheduler = false;            // Disabled by default (opt-in)

    // Regularization: Prevent overfitting via L1/L2/Dropout
    config->enable_regularization = false;          // Disabled by default (opt-in)
    config->regularization_l1_lambda = 0.0f;        // No L1 by default (sparse weights)
    config->regularization_l2_lambda = 0.0001f;     // Small L2 (weight decay)
    config->dropout_rate = 0.0f;                    // No dropout by default

    // Gradient Management: Accumulation, clipping, and health monitoring
    config->enable_gradient_management = false;     // Disabled by default (opt-in)
    config->enable_gradient_health_check = true;    // Enabled by default (safety)
    config->gradient_accumulation_steps = 1;        // No accumulation by default
    config->gradient_clip_value = 0.0f;             // No value clipping (0 = disabled)
    config->gradient_clip_norm = 0.0f;              // No norm clipping (0 = disabled)
}

/**
 * @brief Initialize brain statistics
 *
 * WHY: Separates stats initialization for clarity
 * Makes stats setup reusable
 *
 * COMPLEXITY: O(1)
 *
 * @param stats Output stats structure
 * @param task_name Name for brain
 * @param size Size preset
 * @param num_inputs Input dimension
 * @param learning_rate Learning rate
 */
void nimcp_brain_factory_init_brain_stats(brain_stats_t* stats, const char* task_name, brain_size_t size,
                             uint32_t num_inputs, float learning_rate)
{
    uint32_t num_neurons = nimcp_brain_factory_get_neuron_count(size);

    stats->size = size;
    stats->num_neurons = num_neurons;
    stats->num_synapses = num_neurons * num_inputs;
    stats->num_active_synapses = stats->num_synapses;
    stats->current_learning_rate = learning_rate;
    stats->quantum_annealing_runs = 0;  // Initialize quantum annealing counter
    strncpy(stats->task_name, task_name, sizeof(stats->task_name) - 1);
}

//=============================================================================
// Decision Caching
//=============================================================================

// NOTE: copy_decision() is declared as extern at line 130 (implemented in nimcp_brain.c)

// Cache functions moved to validation module (nimcp_brain_validation.c)

//=============================================================================
// Brain Factory - Creation with Validation
//=============================================================================

/**
 * @brief Validate brain creation parameters
 *
 * WHY: Guard clause pattern - early exit on invalid input
 * Prevents invalid state propagation
 *
 * COMPLEXITY: O(1)
 *
 * @param task_name Brain name
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @return true if valid
 */

brain_t nimcp_brain_factory_allocate_brain(void)
{
    brain_t brain = nimcp_calloc(1, sizeof(struct brain_struct));
    if (!brain) {
        set_error("Failed to allocate brain structure");
        return NULL;
    }

    brain->last_input = NULL;
    brain->cached_decision = NULL;
    brain->input_size = 0;

    // Initialize cache mutex for thread-safe access
    if (nimcp_platform_mutex_init(&brain->cache_mutex, false) != 0) {
        set_error("Failed to initialize cache mutex");
        nimcp_free(brain);
        return NULL;
    }

    brain->distributed = NULL;  // Initialize as standalone brain

    // Phase 11: Initialize long-term memory consolidation buffer
    brain->longterm_capacity = 100;  // Store up to 100 consolidated memories
    brain->longterm_count = 0;
    brain->longterm_memory = nimcp_calloc(brain->longterm_capacity,
                                          sizeof(*brain->longterm_memory));
    // Guard: If allocation fails, set capacity to 0 (consolidation will be disabled)
    if (!brain->longterm_memory) {
        brain->longterm_capacity = 0;
    }

    // Initialize COW fields
    brain->is_cow_clone = false;
    brain->owns_network = true;  // By default, brain owns its network
    brain->original_network = NULL;
    brain->network_is_cached = false;

    // Phase 3: Initialize reference counting fields
    brain->network_refcount = NULL;
    brain->can_use_readonly = false;
    brain->refcount_mutex = NULL;

    // Community Detection: Initialize fields
    brain->functional_modules = NULL;
    brain->network_hubs = NULL;
    brain->topology_metrics = NULL;
    brain->auto_detect_communities = false;
    brain->community_detection_interval = 0.0f;  // Manual only by default

    // Universal Event Bus: Initialize event broadcasting system
    // WHAT: Create event bus for broadcasting brain activities (training, inference, cognitive events)
    // WHY:  Enable decoupled monitoring and integration with external systems
    // HOW:  Use immediate delivery mode for minimal overhead (<1μs per event)
    // TODO: Universal event bus integration pending
    // brain->event_bus = event_bus_create("brain_event_bus", EVENT_DELIVERY_IMMEDIATE);
    // brain->enable_event_broadcasting = (brain->event_bus != NULL);
    // if (!brain->event_bus) {
    //     LOG_WARNING("Failed to create event bus for brain - event broadcasting disabled");
    // }

    return brain;
}

/**
 * @brief Create adaptive network for brain
 *
 * WHY: Isolates network creation complexity
 * Handles network config lifecycle
 *
 * COMPLEXITY: O(n) where n = num_neurons
 *
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @param num_neurons Neuron count
 * @param sparsity_target Target sparsity
 * @return Network handle or NULL on error
 */
adaptive_network_t nimcp_brain_factory_create_brain_network(uint32_t num_inputs, uint32_t num_outputs,
                                               uint32_t num_neurons, float sparsity_target,
                                               ode_integration_method_t integration_method)
{
    adaptive_network_config_t net_config =
        nimcp_brain_factory_build_network_config(num_inputs, num_outputs, num_neurons, sparsity_target, integration_method);

    // Guard: Check if layer_sizes allocation failed in build_base_network_config
    // WHY: NULL layer_sizes will cause crash in adaptive_network_create
    if (!net_config.base_config.layer_sizes) {
        // Error already set by build_base_network_config
        return NULL;
    }

    adaptive_network_t network = adaptive_network_create(&net_config);

    // Free our copy of layer_sizes - adaptive_network_create makes its own deep copy (or fails)
    // WHY: Avoid memory leak - we allocated this in build_base_network_config
    // WHAT: Safe to free even if network creation failed, because we still own this allocation
    // Note: layer_sizes pointer should not be modified by adaptive_network_create (const param)
    if (net_config.base_config.layer_sizes) {
        nimcp_free((void*)net_config.base_config.layer_sizes);
    }

    return network;
}

/**
 * @brief Initialize output labels array
 *
 * COMPLEXITY: O(1)
 *
 * @param brain Brain to initialize
 * @param num_outputs Number of output labels
 * @return true on success
 */
bool nimcp_brain_factory_init_output_labels(brain_t brain, uint32_t num_outputs)
{
    if (!brain) {
        set_error("NULL brain pointer in init_output_labels");
        return false;
    }
    if (num_outputs == 0) {
        // Zero outputs - no allocation needed, but set to NULL
        brain->output_labels = NULL;
        brain->num_output_labels = 0;
        return true;
    }
    brain->output_labels = nimcp_calloc(num_outputs, sizeof(char*));
    if (!brain->output_labels) {
        set_error("Failed to allocate output labels");
        return false;
    }
    brain->num_output_labels = 0;
    return true;
}

/**
 * @brief Initialize universal event bus
 *
 * WHAT: Creates event bus for brain-wide event coordination
 * WHY:  Enables all modules to publish and subscribe to events
 * HOW:  Creates event bus with immediate delivery mode
 *
 * @param brain Brain to initialize event bus for
 * @return true if initialization successful, false on error
 *
 * @note Initialization Behavior
 * - Returns false if brain invalid
 * - Returns true if already initialized (idempotent)
 * - Returns false only on allocation failure
 * - Cleanup handled by brain_destroy()
 *
 * @version 2.7.0 Phase 10.x
 * @author NIMCP Development Team
 * @date 2025-11-20
 */
bool nimcp_brain_factory_init_event_bus(brain_t brain)
{
    if (!brain) {
        set_error("brain_factory_init_event_bus: NULL brain");
        return false;
    }

    // Check if already initialized
    if (brain->event_bus) {
        return true;  // Already initialized
    }

    // Create event bus with immediate delivery (synchronous for predictability)
    brain->event_bus = event_bus_create("brain_event_bus", EVENT_DELIVERY_IMMEDIATE);
    if (!brain->event_bus) {
        set_error("Failed to create brain event bus");
        return false;
    }

    LOG_INFO("Universal event bus initialized for brain");
    return true;
}

/**
 * @brief Initialize multi-modal subsystems (Phase 8)
 *
 * WHAT: Create visual cortex, audio cortex, and integration layer
 * WHY:  Enable unified multi-modal processing
 * HOW:  Check config flags, create modules, allocate feature buffers
 *
 * DESIGN:
 * - Only creates modules if config flags are enabled
 * - Allocates reusable feature buffers (no per-frame allocation)
 * - Gracefully handles partial initialization
 *
 * @param brain Brain structure with configuration set
 * @return true on success, false on failure
 *
 * COMPLEXITY: O(1) - just allocation
 * MEMORY: O(D_v + D_a + D_integrated) for feature buffers
 *
 * ERROR HANDLING:
 * - Returns true if multi-modal disabled (not an error)
 * - Returns false only on allocation failure
 * - Partial cleanup handled by brain_destroy()
 *
 * @version 2.7.0 Phase 8.1
 * @author NIMCP Development Team
 * @date 2025-11-08
 */

/**
 * @brief Initialize glial integration subsystem
 *
 * WHAT: Creates glial integration for astrocyte modulation
 * WHY:  Enables biological realism with glial cell support
 * HOW:  Conditional initialization based on config.enable_glial flag
 *
 * @param brain Brain to initialize glial subsystem for
 * @return true if initialization successful (or glial disabled), false on error
 *
 * @note Initialization Behavior
 * - Returns true if glial disabled (not an error)
 * - Returns true if already initialized
 * - Returns false only on allocation failure
 * - Cleanup handled by brain_destroy()
 *
 * @version 2.7.0 Phase 5/6
 * @author NIMCP Development Team
 * @date 2025-11-18
 */
bool nimcp_brain_factory_init_glial_subsystem(brain_t brain)
{
    if (!brain || !brain->network) {
        return false;
    }

    // Check if already initialized (prevent double initialization)
    if (brain->glial) {
        return true;  // Already initialized
    }

    // Check if glial integration is enabled
    if (!brain->config.enable_glial) {
        return true;  // Disabled, not an error
    }

    // Get base network for glial integration
    neural_network_t base = adaptive_network_get_base_network(brain->network);
    if (!base) {
        set_error("Failed to get base network for glial integration");
        return false;
    }

    // Create glial integration with reasonable max_mappings
    brain->glial = glial_integration_create(base, 1000);

    if (!brain->glial) {
        set_error("Failed to create glial integration");
        return false;
    }

    // Initialize myelin sheath network if enabled
    if (brain->config.enable_myelin_sheath) {
        // Check if already initialized
        if (!brain->myelin_sheath) {
            // Create myelin sheath network with default configuration
            // Capacity scales with oligodendrocyte count (each oligo can myelinate multiple axons)
            uint32_t myelin_capacity = brain->config.num_oligodendrocytes > 0
                ? brain->config.num_oligodendrocytes * 10
                : 1000;  // Default capacity
            brain->myelin_sheath = myelin_network_create_default(myelin_capacity);
            if (!brain->myelin_sheath) {
                set_error("Failed to create myelin sheath network");
                // Non-fatal: glial integration still works without myelin modeling
            } else {
                // Wire myelin sheath network to glial integration layer
                nimcp_result_t result = glial_integration_set_myelin_sheath_network(
                    brain->glial, brain->myelin_sheath);
                if (result != NIMCP_SUCCESS) {
                    set_error("Failed to wire myelin sheath to glial integration");
                    // Non-fatal: myelin network still usable standalone
                }
            }
        }
    }

    return true;
}
bool nimcp_brain_factory_init_multimodal_subsystems(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if multi-modal processing is enabled
    if (!brain->config.enable_multimodal_integration) {
        // Even when multimodal is disabled, we still need integrated_feature_buffer
        // as a working buffer for direct-only predictions
        if (!brain->integrated_feature_buffer) {
            brain->integrated_feature_buffer = nimcp_calloc(brain->config.num_inputs, sizeof(float));
            if (!brain->integrated_feature_buffer) {
                set_error("Failed to allocate integrated feature buffer for direct predictions");
                return false;
            }
        }
        return true;
    }

    // If multimodal is enabled, check if already fully initialized
    // We check for visual/audio/speech cortices only if they should be enabled
    bool visual_needed = brain->config.enable_visual_cortex && brain->config.visual_feature_dim > 0;
    bool audio_needed = brain->config.enable_audio_cortex && brain->config.audio_feature_dim > 0;
    bool speech_needed = brain->config.enable_speech_cortex && brain->config.speech_feature_dim > 0;

    bool visual_ready = !visual_needed || brain->visual_cortex;
    bool audio_ready = !audio_needed || brain->audio_cortex;
    bool speech_ready = !speech_needed || brain->speech_cortex;

    // If all needed components are ready and multimodal layer exists, we're done
    if (brain->multimodal && visual_ready && audio_ready && speech_ready) {
        return true;  // Already fully initialized
    }

    // Initialize visual cortex (if enabled)
    if (brain->config.enable_visual_cortex && brain->config.visual_feature_dim > 0) {
        visual_cortex_config_t visual_config = {
            .input_width = 640,        // Default camera resolution
            .input_height = 480,
            .num_v1_filters = 32,      // 32 orientation-selective filters
            .feature_dim = brain->config.visual_feature_dim,
            .enable_attention = true,
            .enable_memory = true,

            // NIMCP 2.7 Phase 8.5: Fractal Topology Integration
            .enable_fractal_topology = brain->config.enable_fractal_topology,
            .hub_ratio = 0.15f,        // 15% hub neurons (biological cortex ratio)
            .power_law_gamma = -2.1f,  // Cortical power-law exponent
            .internal_neurons = 32 * 10 // 10 neurons per filter (V1 columnar structure)
        };

        brain->visual_cortex = visual_cortex_create(&visual_config);
        if (!brain->visual_cortex) {
            set_error("Failed to create visual cortex");
            return false;
        }

        // Allocate visual feature buffer
        brain->visual_feature_buffer = nimcp_calloc(brain->config.visual_feature_dim, sizeof(float));
        if (!brain->visual_feature_buffer) {
            set_error("Failed to allocate visual feature buffer");
            visual_cortex_destroy(brain->visual_cortex);
            brain->visual_cortex = NULL;
            return false;
        }
    }

    // Initialize audio cortex (if enabled)
    if (brain->config.enable_audio_cortex && brain->config.audio_feature_dim > 0) {
        audio_cortex_config_t audio_config = {
            .sample_rate = 16000,      // Default 16kHz audio
            .frame_size = 512,         // 32ms frames at 16kHz
            .num_freq_bins = 256,
            .num_mel_filters = 40,     // Standard for speech
            .num_mfcc = brain->config.audio_feature_dim,
            .num_channels = 1,         // Mono by default
            .feature_dim = brain->config.audio_feature_dim,
            .enable_attention = true,
            .enable_memory = true,

            // NIMCP 2.7 Phase 8.5: Fractal Topology Integration
            .enable_fractal_topology = brain->config.enable_fractal_topology,
            .hub_ratio = 0.15f,        // 15% hub neurons (biological A1 ratio)
            .power_law_gamma = -2.1f,  // Tonotopic power-law exponent
            .internal_neurons = 40 * 10 // 10 neurons per mel filter (A1 tonotopic structure)
        };

        brain->audio_cortex = audio_cortex_create(&audio_config);
        if (!brain->audio_cortex) {
            set_error("Failed to create audio cortex");
            return false;
        }

        // Allocate audio feature buffer
        brain->audio_feature_buffer = nimcp_calloc(brain->config.audio_feature_dim, sizeof(float));
        if (!brain->audio_feature_buffer) {
            set_error("Failed to allocate audio feature buffer");
            audio_cortex_destroy(brain->audio_cortex);
            brain->audio_cortex = NULL;
            return false;
        }
    }

    // Initialize speech cortex (Phase 8.8)
    if (brain->config.enable_speech_cortex && brain->config.speech_feature_dim > 0) {
        speech_cortex_config_t speech_config = speech_cortex_default_config();

        // Override defaults with brain config
        speech_config.sample_rate = 16000;        // Standard speech rate
        speech_config.frame_size_ms = 20;         // 20ms frames for phoneme analysis
        speech_config.num_phonemes = SPEECH_NUM_PHONEMES; // 44 phonemes (English)
        speech_config.feature_dim = brain->config.speech_feature_dim;
        speech_config.enable_wernicke = true;     // Enable word recognition
        speech_config.enable_prosody = true;      // Enable pitch/stress analysis
        speech_config.enable_memory = true;       // Enable phonological working memory

        // NIMCP 2.7 Phase 8.5: Fractal Topology Integration
        speech_config.enable_fractal_topology = brain->config.enable_fractal_topology;
        speech_config.hub_ratio = 0.15f;          // 15% hub neurons (biological STG ratio)
        speech_config.power_law_gamma = -2.1f;    // Speech network power-law exponent
        speech_config.internal_neurons = SPEECH_NUM_PHONEMES * 10; // 10 neurons per phoneme

        brain->speech_cortex = speech_cortex_create(&speech_config);
        if (!brain->speech_cortex) {
            set_error("Failed to create speech cortex");
            return false;
        }

        // Allocate speech feature buffer
        brain->speech_feature_buffer = nimcp_calloc(brain->config.speech_feature_dim, sizeof(float));
        if (!brain->speech_feature_buffer) {
            set_error("Failed to allocate speech feature buffer");
            speech_cortex_destroy(brain->speech_cortex);
            brain->speech_cortex = NULL;
            return false;
        }
    }

    // Initialize multi-modal integration layer
    uint32_t visual_dim = brain->config.enable_visual_cortex ? brain->config.visual_feature_dim : 0;
    uint32_t audio_dim = brain->config.enable_audio_cortex ? brain->config.audio_feature_dim : 0;
    uint32_t speech_dim = brain->config.enable_speech_cortex ? brain->config.speech_feature_dim : 0;
    // Direct dimension: Remaining space after visual, audio, and speech features
    uint32_t direct_dim = 0;
    if (brain->config.num_inputs > (visual_dim + audio_dim + speech_dim)) {
        direct_dim = brain->config.num_inputs - visual_dim - audio_dim - speech_dim;
    }

    if (visual_dim > 0 || audio_dim > 0 || speech_dim > 0 || direct_dim > 0) {
        // Phase 8.8: Speech is now a dedicated modality
        multimodal_config_t mm_config = multimodal_default_config(visual_dim, audio_dim, speech_dim, direct_dim);

        // Output dimension should match network input size
        mm_config.output_dim = brain->config.num_inputs;

        brain->multimodal = multimodal_integration_create(&mm_config);
        if (!brain->multimodal) {
            set_error("Failed to create multimodal integration layer");
            return false;
        }

        // Allocate integrated feature buffer
        brain->integrated_feature_buffer = nimcp_calloc(mm_config.output_dim, sizeof(float));
        if (!brain->integrated_feature_buffer) {
            set_error("Failed to allocate integrated feature buffer");
            multimodal_integration_destroy(brain->multimodal);
            brain->multimodal = NULL;
            return false;
        }
    }

    // Initialize NLP network (if multimodal or speech is enabled)
    if (brain->config.enable_multimodal_integration || brain->config.enable_speech_cortex) {

        // Configure NLP network with minimal config
        nlp_network_config_t nlp_config = {0};

        // NLP-specific parameters
        nlp_config.vocab_size = 10000;            // 10k token vocabulary
        nlp_config.embedding_dim = 128;           // 128-dim embeddings
        nlp_config.max_sequence_length = 32;      // 32 token context
        nlp_config.use_attention_synapses = true;
        nlp_config.use_neuromodulated_synapses = true;

        // Configure base network (required for neural_network_create)
        nlp_config.network_config.num_neurons = 256;  // Small NLP network
        nlp_config.network_config.input_size = nlp_config.embedding_dim;
        nlp_config.network_config.output_size = nlp_config.embedding_dim;
        nlp_config.network_config.enable_stdp = true;
        nlp_config.network_config.enable_hebbian = false;
        nlp_config.network_config.enable_oja = false;
        nlp_config.network_config.enable_homeostasis = false;
        nlp_config.network_config.learning_rate = 0.01f;

        // Configure attention (required for multihead_attention_create)
        nlp_config.attention_config.num_heads = brain->config.num_attention_heads > 0 ? brain->config.num_attention_heads : 4;
        nlp_config.attention_config.input_dim = nlp_config.embedding_dim;
        nlp_config.attention_config.output_dim = nlp_config.embedding_dim;
        nlp_config.attention_config.sequence_length = nlp_config.max_sequence_length;
        nlp_config.attention_config.use_thalamic_gate = false;
        nlp_config.attention_config.use_salience_weighting = false;
        nlp_config.attention_config.gate_bias = 0.5f;

        // Configure neuromodulators (required for neuromodulator_system_create)
        nlp_config.neuromod_config.baseline_dopamine = 0.2f;
        nlp_config.neuromod_config.baseline_serotonin = 0.2f;
        nlp_config.neuromod_config.baseline_acetylcholine = 0.2f;
        nlp_config.neuromod_config.baseline_norepinephrine = 0.2f;
        nlp_config.neuromod_config.dopamine_decay = 2.0f;
        nlp_config.neuromod_config.serotonin_decay = 10.0f;
        nlp_config.neuromod_config.acetylcholine_decay = 0.5f;
        nlp_config.neuromod_config.norepinephrine_decay = 3.0f;
        nlp_config.neuromod_config.reward_dopamine_gain = 0.5f;
        nlp_config.neuromod_config.threat_norepinephrine_gain = 0.7f;
        nlp_config.neuromod_config.salience_acetylcholine_gain = 0.6f;
        nlp_config.neuromod_config.punishment_serotonin_gain = 0.4f;
        nlp_config.neuromod_config.enable_volume_transmission = true;
        nlp_config.neuromod_config.diffusion_rate = 0.1f;

        brain->nlp_network = nlp_network_create(&nlp_config);
        if (!brain->nlp_network) {
            set_error("Failed to create NLP network");
            return false;
        }
    }

    return true;
}

/**
 * WHAT: Initialize pink noise neuromodulation subsystem
 * WHY:  Enable 1/f noise-modulated dopamine/serotonin for exploration-exploitation balance
 * HOW:  Create pink noise neuromodulator if config flag is set
 *
 * BIOLOGICAL MOTIVATION:
 * - Dopamine neurons exhibit 1/f noise in firing patterns (Montague et al., 2004)
 * - Serotonin fluctuations follow pink spectrum (Cools et al., 2008)
 * - Multi-timescale correlations enable context-dependent learning
 *
 * INTEGRATION:
 * - Modulates learning rates via dopamine
 * - Scales attention via acetylcholine
 * - Enables exploration via pink noise
 *
 * @param brain Brain instance to initialize
 * @return true on success, false on failure
 *
 * @version 2.7.0 Phase 8.6
 * @author NIMCP Development Team
 * @date 2025-11-08
 */
bool nimcp_brain_factory_init_pink_noise_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->pink_noise) {
        return true;  // Already initialized
    }

    // Check if pink noise is enabled
    if (!brain->config.enable_pink_noise) {
        return true;  // Not enabled, not an error
    }

    // Create pink noise neuromodulator with default configuration
    neuromod_pink_config_t pink_config = neuromod_pink_default_config();

    // Adjust baselines for brain-level processing
    pink_config.dopamine_baseline = 0.3f;      // Moderate baseline for learning
    pink_config.serotonin_baseline = 0.4f;     // Moderate baseline for stability
    pink_config.acetylcholine_baseline = 0.5f; // Moderate baseline for attention
    pink_config.norepinephrine_baseline = 0.2f;// Lower baseline for arousal

    // Configure noise amplitudes for exploration-exploitation balance
    pink_config.dopamine_noise_amplitude = 0.15f;      // 15% noise for exploration
    pink_config.serotonin_noise_amplitude = 0.08f;     // 8% noise for stability modulation
    pink_config.acetylcholine_noise_amplitude = 0.20f; // 20% noise for dynamic attention
    pink_config.norepinephrine_noise_amplitude = 0.10f;// 10% noise for arousal variation

    brain->pink_noise = neuromod_pink_create(&pink_config);
    if (!brain->pink_noise) {
        set_error("Failed to create pink noise neuromodulator");
        return false;
    }

    return true;
}

/**
 * WHAT: Initialize full neuromodulator system
 * WHY:  Enable mental health interventions to adjust neurotransmitter levels
 * HOW:  Create neuromodulator system with default configuration
 *
 * BIOLOGICAL MOTIVATION:
 * - Neurotransmitters regulate mood, attention, arousal, and learning
 * - Mental health disorders often involve chemical imbalances
 * - Interventions can modulate levels to restore healthy functioning
 */
bool nimcp_brain_factory_init_neuromodulator_system(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->neuromodulator_system) {
        return true;  // Already initialized
    }

    // Phase 12: Compute personality-modulated neuromodulator baselines
    // Default to moderate levels if no personality
    float dopamine_base = 0.5f;
    float serotonin_base = 0.5f;
    float acetylcholine_base = 0.5f;
    float norepinephrine_base = 0.5f;

    if (brain->personality) {
        // Map personality traits to neurotransmitter baselines
        personality_profile_t* p = brain->personality;

        // Dopamine (reward, motivation): Driven by Extraversion
        // Extraverts seek social rewards → higher dopamine baseline
        dopamine_base = 0.3f + p->traits.extraversion * 0.5f;  // [0.3, 0.8]

        // Serotonin (mood stability, impulse control): Inverse of Neuroticism
        // High neuroticism → low serotonin (anxiety, mood instability)
        serotonin_base = 0.7f - p->traits.neuroticism * 0.4f;  // [0.3, 0.7]

        // Acetylcholine (attention, learning): Driven by Openness
        // High openness → high acetylcholine (intellectual curiosity)
        acetylcholine_base = 0.3f + p->traits.openness * 0.5f;  // [0.3, 0.8]

        // Norepinephrine (arousal, vigilance): Driven by Conscientiousness
        // High conscientiousness → sustained alertness
        norepinephrine_base = 0.4f + p->traits.conscientiousness * 0.4f;  // [0.4, 0.8]
    }

    // Always create neuromodulator system (needed for mental health monitoring)
    // Configuration with personality-modulated baseline levels
    neuromodulator_config_t neuromod_config = {
        // Baseline concentrations (personality-modulated homeostatic set points)
        .baseline_dopamine = dopamine_base,          // Reward sensitivity
        .baseline_serotonin = serotonin_base,        // Mood/impulse control
        .baseline_acetylcholine = acetylcholine_base, // Attention
        .baseline_norepinephrine = norepinephrine_base, // Arousal

        // Decay time constants (seconds)
        .dopamine_decay = 2.0f,         // Fast decay (phasic DA bursts)
        .serotonin_decay = 10.0f,       // Slow decay (tonic 5-HT)
        .acetylcholine_decay = 0.5f,    // Very fast decay (attention bursts)
        .norepinephrine_decay = 3.0f,   // Moderate decay (arousal)

        // Response gains
        .reward_dopamine_gain = 0.5f,
        .threat_norepinephrine_gain = 0.7f,
        .salience_acetylcholine_gain = 0.6f,
        .punishment_serotonin_gain = 0.4f,

        // Volume transmission
        .enable_volume_transmission = true,
        .diffusion_rate = 0.1f
    };

    brain->neuromodulator_system = neuromodulator_system_create(&neuromod_config);
    if (!brain->neuromodulator_system) {
        set_error("Failed to create neuromodulator system");
        return false;
    }

    return true;
}

/**
 * WHAT: Initialize spatial neuromodulator system with quantum walk diffusion (Phase C2.1)
 * WHY:  Enable spatially-distributed neuromodulation with quantum speedup
 * HOW:  Create spatial neuromod system and wire to glial integration
 *
 * BIOLOGICAL MOTIVATION:
 * - Volume transmission: Neuromodulators diffuse through extracellular space
 * - Glial mediation: Astrocytes regulate neuromodulator concentrations
 * - Quantum walk: O(√N) speedup for diffusion on neural network graph
 *
 * INTEGRATION WITH BRAIN:
 * - Wired into glial integration system for coordination with astrocytes
 * - Uses quantum walk configuration from brain config
 * - Spatially modulates synaptic transmission based on local concentrations
 *
 * @param brain Brain instance to initialize
 * @return true on success, false on failure
 */
bool nimcp_brain_factory_init_spatial_neuromod_system(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->glial && brain->glial->spatial_neuromod) {
        return true;  // Already initialized
    }

    // Guard: Need network and glial integration
    if (!brain->network || !brain->glial) {
        // Not an error if glial integration not set up yet
        return true;
    }

    // Phase C2.1: Create spatial neuromod configs with quantum walk settings
    bool enabled_types[NEUROMOD_COUNT] = {true, true, true, true};  // Enable all 4 types
    spatial_neuromod_config_t configs[NEUROMOD_COUNT];

    for (int i = 0; i < NEUROMOD_COUNT; i++) {
        configs[i] = spatial_neuromod_default_config((neuromodulator_type_t)i);

        // Apply brain quantum walk configuration (Phase C2.1)
        configs[i].enable_quantum_walk = brain->config.enable_quantum_walk_diffusion;
        configs[i].quantum_walk_steps = brain->config.quantum_walk_steps;
        configs[i].quantum_mixing_ratio = brain->config.quantum_classical_mixing;
        configs[i].quantum_coin_type = brain->config.quantum_coin_type;
        configs[i].quantum_decoherence = brain->config.quantum_decoherence_rate;
    }

    // Create spatial neuromod system
    spatial_neuromod_system_t* spatial_neuromod =
        spatial_neuromod_system_create(brain->network, enabled_types, configs);

    if (!spatial_neuromod) {
        // Non-fatal: spatial neuromod is optional enhancement
        fprintf(stderr, "WARNING: Failed to create spatial neuromodulator system, continuing without it\n");
        return true;
    }

    // Wire into glial integration
    nimcp_result_t result = glial_integration_set_spatial_neuromod_system(
        brain->glial, spatial_neuromod);

    if (result != NIMCP_SUCCESS) {
        spatial_neuromod_system_destroy(spatial_neuromod);
        fprintf(stderr, "WARNING: Failed to wire spatial neuromod into glial integration\n");
        return true;  // Non-fatal
    }

    fprintf(stderr, "INFO: Spatial neuromodulator system initialized %s\n",
            brain->config.enable_quantum_walk_diffusion ?
            "with quantum walk acceleration (√N speedup)" : "(classical diffusion)");

    return true;
}

/**
 * WHAT: Initialize multihead attention mechanism
 * WHY:  Enable selective focus on relevant features for efficient processing
 * HOW:  Create attention system based on cortical column architecture
 *
 * BIOLOGICAL MOTIVATION:
 * - Cortical Columns: Each attention head acts as specialized processing column
 * - Thalamic Gating: Controls information flow (like thalamic relay nucleus)
 * - Salience Weighting: Biologically-inspired attention based on feature importance
 * - Parallel Streams: Multiple heads process different aspects simultaneously
 *
 * INTEGRATION WITH BRAIN:
 * - Applied to multimodal inputs (visual, audio, speech) before neural network
 * - Connects to salience evaluator for attention weighting
 * - Interfaces with executive control for top-down attention modulation
 * - Used in working memory for attention-based retrieval
 *
 * PERFORMANCE BENEFITS:
 * - 2-5x inference speedup by selective processing
 * - 30-50% memory reduction through focused activations
 * - 5-15% accuracy improvement on complex tasks
 *
 * @param brain Brain instance to initialize
 * @return true on success, false on failure
 *
 * @version 3.0.0 Module Integration Phase
 * @author NIMCP Development Team
 * @date 2025-11-11
 */
bool nimcp_brain_factory_init_attention_subsystem(brain_t brain)
{
    // WHAT: Guard clause - validate input
    // WHY:  Prevent null pointer dereference
    // HOW:  Check brain pointer before use
    if (!brain) {
        return false;
    }

    // WHAT: Check if already initialized
    // WHY:  Prevent double initialization and memory leak
    // HOW:  Return success if attention already exists
    if (brain->multihead_attention) {
        return true;  // Already initialized
    }

    // WHAT: Check if attention is enabled in configuration
    // WHY:  Only initialize if user requested this feature
    // HOW:  Check config flag, return success (not error) if disabled
    if (!brain->config.enable_multihead_attention) {
        return true;  // Not enabled, not an error
    }

    // WHAT: Calculate appropriate dimensions for attention
    // WHY:  Attention dimensions must match integrated_feature_buffer size
    // HOW:  Always use num_inputs (the output size of multimodal integration)
    //
    // NOTE: The multimodal integration layer compresses all modalities
    //       (visual + audio + speech + direct) into a unified representation
    //       of size num_inputs. The attention system processes this integrated
    //       representation, not the raw concatenated features.
    uint32_t input_dim = brain->config.num_inputs;

    // WHAT: Configure multihead attention system
    // WHY:  Need proper configuration for cortical column architecture
    // HOW:  Create config with biological parameters
    multihead_attention_config_t attention_config = {
        .num_heads = brain->config.num_attention_heads > 0 ?
                     brain->config.num_attention_heads : 8,  // Default: 8 heads
        .input_dim = input_dim,
        .output_dim = input_dim,  // Same dimension (residual connection compatible)
        .sequence_length = 32,    // Default sequence length for temporal processing
        .use_thalamic_gate = brain->config.enable_thalamic_gate,
        .use_salience_weighting = brain->config.enable_salience_weighting,
        .gate_bias = 0.5f        // Default: 50% gate opening
    };

    // WHAT: Create multihead attention system
    // WHY:  Enable selective feature processing with parallel attention streams
    // HOW:  Call attention creation API with configured parameters
    brain->multihead_attention = multihead_attention_create(&attention_config);
    if (!brain->multihead_attention) {
        set_error("Failed to create multihead attention system");
        return false;
    }

    return true;
}

/**
 * WHAT: Initialize brain regions hierarchical architecture
 * WHY:  Enable modular cortical organization with layers and minicolumns
 * HOW:  Create brain module with specialized regions if config enables it
 *
 * BIOLOGICAL MOTIVATION:
 * - Cerebral cortex organized into hierarchical regions (V1, A1, M1, PFC, etc.)
 * - Each region has 6 cortical layers with distinct functions
 * - Minicolumns span layers vertically for parallel processing
 * - Inter-region connections follow biological pathways (feedforward/feedback)
 *
 * INTEGRATION WITH BRAIN:
 * - Provides spatial organization of processing
 * - Enables specialized regions for sensory, motor, associative functions
 * - Supports realistic cortical layer dynamics (Layer 4 input, Layer 5 output)
 * - Allows for hierarchical processing pathways
 *
 * @param brain Brain instance to initialize
 * @return true on success, false on failure
 *
 * @version 3.0.0 Module Integration Phase
 * @author NIMCP Development Team
 * @date 2025-11-11
 */
bool nimcp_brain_factory_init_brain_regions_subsystem(brain_t brain)
{
    // WHAT: Guard clause - validate input
    // WHY:  Prevent null pointer dereference
    // HOW:  Check brain pointer before use
    if (!brain) {
        return false;
    }

    // WHAT: Check if already initialized
    // WHY:  Prevent double initialization and memory leak
    // HOW:  Return success if brain_regions already exists
    if (brain->brain_regions) {
        return true;  // Already initialized
    }

    // WHAT: Check if brain regions architecture is enabled in configuration
    // WHY:  Only initialize if user requested this feature
    // HOW:  Check config flag, return success (not error) if disabled
    if (!brain->config.enable_brain_regions) {
        return true;  // Not enabled, not an error
    }

    // WHAT: Determine number of regions and neurons per region
    // WHY:  Need proper sizing for modular architecture
    // HOW:  Use config values with sensible defaults
    uint32_t num_regions = brain->config.num_brain_regions > 0 ?
                           brain->config.num_brain_regions : 4;  // Default: 4 regions
    uint32_t neurons_per_region = brain->config.neurons_per_region > 0 ?
                                  brain->config.neurons_per_region : 1000;  // Default: 1000 neurons

    // WHAT: Create brain module with max capacity
    // WHY:  Top-level container for all brain regions
    // HOW:  Allocate module with specified max regions
    brain->brain_regions = brain_module_create(num_regions);
    if (!brain->brain_regions) {
        set_error("Failed to create brain regions module");
        return false;
    }

    // WHAT: Create individual brain regions with specialized types
    // WHY:  Different regions have different layer proportions and neuron types
    // HOW:  Create regions based on configuration, starting with primary sensory/motor areas
    brain_region_type_t region_types[] = {
        REGION_VISUAL_V1,      // Primary visual cortex
        REGION_AUDITORY_A1,    // Primary auditory cortex
        REGION_MOTOR_M1,       // Primary motor cortex
        REGION_PREFRONTAL      // Prefrontal cortex (executive control)
    };

    for (uint32_t i = 0; i < num_regions && i < 4; i++) {
        brain_region_t* region = brain_region_create(region_types[i], neurons_per_region);
        if (!region) {
            set_error("Failed to create brain region");
            return false;
        }

        // Organize region into minicolumns (8x8 grid for moderate-sized regions)
        uint32_t columns_x = 8;
        uint32_t columns_y = 8;
        if (brain_region_organize_columns(region, columns_x, columns_y) != NIMCP_SUCCESS) {
            brain_region_destroy(region);
            set_error("Failed to organize brain region into minicolumns");
            return false;
        }

        // Add region to brain module
        if (brain_module_add_region(brain->brain_regions, region) != NIMCP_SUCCESS) {
            brain_region_destroy(region);
            set_error("Failed to add region to brain module");
            return false;
        }
    }

    // WHAT: Establish inter-region connections
    // WHY:  Brain regions need to communicate (e.g., V1 → PFC for visual attention)
    // HOW:  Connect regions with biologically realistic pathways
    if (num_regions >= 2) {
        // Connect V1 (visual) → PFC (prefrontal) for visual processing pathway
        brain_region_t* v1 = brain_module_get_region_by_type(brain->brain_regions, REGION_VISUAL_V1);
        brain_region_t* pfc = brain_module_get_region_by_type(brain->brain_regions, REGION_PREFRONTAL);
        if (v1 && pfc) {
            brain_module_connect_regions(brain->brain_regions, v1->id, pfc->id, 0.3f);
        }
    }

    if (num_regions >= 3) {
        // Connect A1 (auditory) → PFC for auditory processing pathway
        brain_region_t* a1 = brain_module_get_region_by_type(brain->brain_regions, REGION_AUDITORY_A1);
        brain_region_t* pfc = brain_module_get_region_by_type(brain->brain_regions, REGION_PREFRONTAL);
        if (a1 && pfc) {
            brain_module_connect_regions(brain->brain_regions, a1->id, pfc->id, 0.3f);
        }
    }

    return true;
}

/**
 * WHAT: Initialize symbolic logic reasoning subsystem
 * WHY:  Enable logical inference, knowledge representation, and abstract reasoning
 * HOW:  Create symbolic logic engine if config enables it
 *
 * BIOLOGICAL MOTIVATION:
 * - Prefrontal cortex performs abstract logical reasoning
 * - Hippocampus stores declarative knowledge (facts)
 * - Working memory maintains active inferences
 *
 * INTEGRATION WITH BRAIN:
 * - Stores facts learned during experience
 * - Performs deductive/inductive reasoning
 * - Validates decisions against logical constraints
 * - Enables explanation generation ("because X implies Y")
 *
 * @param brain Brain instance to initialize
 * @return true on success, false on failure
 *
 * @version 2.7.0 Phase 8.9
 * @author NIMCP Development Team
 * @date 2025-11-08
 */
bool nimcp_brain_factory_init_symbolic_logic_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->logic) {
        return true;  // Already initialized
    }

    // Check if symbolic logic is enabled via knowledge system or explicit flag
    // The knowledge system uses logic internally, so enable if knowledge is enabled
    if (!brain->config.enable_knowledge) {
        return true;  // Not enabled, not an error
    }

    // Create neural logic network with spiking logic gates (Phase 9.0)
    neural_logic_config_t logic_config = neural_logic_default_config(1000);
    logic_config.use_gpu = neural_logic_gpu_available();
    logic_config.integration_window_ms = 5.0f;
    logic_config.enable_learning = false;  // Combinational logic (no plasticity)

    brain->logic = neural_logic_create(&logic_config);
    if (!brain->logic) {
        set_error("Failed to create neural logic network");
        return false;
    }

    return true;
}

/**
 * @brief Initialize symbolic reasoning subsystem (Phase 9.4)
 *
 * WHAT: Creates symbolic logic engine for first-order logic reasoning
 * WHY:  Enable logical inference, consistency checking for communication
 * HOW:  Allocate logic engine with inference and knowledge base capabilities
 *
 * @param brain Brain to initialize
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_symbolic_reasoning_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->symbolic_logic) {
        return true;  // Already initialized
    }

    // Only initialize if explicitly enabled
    if (!brain->config.enable_logic) {
        brain->symbolic_logic = NULL;
        return true;  // Not enabled, not an error
    }

    // Create symbolic logic engine with default configuration
    logic_config_t logic_config = {
        .max_predicates = LOGIC_MAX_PREDICATES,
        .max_rules = LOGIC_MAX_RULES,
        .max_kb_size = 10000,           // 10K facts
        .max_inference_depth = 10,       // Max 10 inference steps
        .enable_forward_chaining = true,
        .enable_backward_chaining = true,
        .enable_resolution = true,
        .enable_memory_consolidation = false  // Handled by brain->consolidation
    };

    brain->symbolic_logic = symbolic_logic_create(&logic_config);
    if (!brain->symbolic_logic) {
        set_error("Failed to create symbolic logic engine");
        return false;
    }

    return true;
}

/**
 * @brief Initialize epistemic filtering subsystem (Phase 9.2)
 *
 * WHAT: Creates epistemic filter for cognitive bias prevention
 * WHY:  Prevents conspiracy-theory thinking and cognitive biases
 * HOW:  Applies skepticism, evidence evaluation, bias detection
 *
 * @param brain Brain to initialize
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_epistemic_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->epistemic) {
        return true;  // Already initialized
    }

    // Epistemic filtering is recommended for all brains to prevent
    // accepting unproven information or developing biased reasoning

    // Skepticism level:
    // 0.0 = credulous (accepts most claims)
    // 0.5 = balanced (reasonable skepticism)
    // 0.7 = cautious (requires strong evidence)
    // 1.0 = extreme skeptic (rejects almost everything)
    //
    // We default to 0.6 (cautious but not paranoid)
    float skepticism_level = 0.6f;

    brain->epistemic = epistemic_filter_create(skepticism_level);
    if (!brain->epistemic) {
        set_error("Failed to create epistemic filter");
        return false;
    }

    return true;
}

/**
 * @brief Initialize working memory subsystem (Phase 10.2)
 *
 * WHAT: Creates Miller's 7±2 working memory buffer with temporal decay
 * WHY:  Provides active representation buffer for reasoning and planning
 * HOW:  Uses config settings or defaults (capacity=7, tau=1000ms)
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_working_memory_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->working_memory) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_working_memory) {
        return true;  // Not enabled, but not an error
    }

    // Create custom config from brain config
    working_memory_config_t wm_config = working_memory_default_config();

    // Override defaults with brain config if specified
    if (brain->config.working_memory_capacity > 0) {
        wm_config.capacity = brain->config.working_memory_capacity;
    }
    if (brain->config.working_memory_decay_tau_ms > 0.0f) {
        wm_config.decay_tau_ms = brain->config.working_memory_decay_tau_ms;
    }

    brain->working_memory = working_memory_create_custom(&wm_config);
    if (!brain->working_memory) {
        set_error("Failed to create working memory");
        return false;
    }

    /* =========================================================================
     * PHASE 10.2: Emotional System Initialization
     * =========================================================================
     * WHAT: Create integrated emotional processing system
     * WHY:  Enable emotional tagging, regulation, salience boost, mental health
     * HOW:  Create with defaults (Russell's Circumplex Model + CBT/DBT regulation)
     */

    brain->emotional_system = emotion_system_create(NULL);  // NULL = use defaults
    if (!brain->emotional_system) {
        set_error("Failed to create emotional system");
        return false;
    }

    /* =========================================================================
     * PHASE 10.1: Sleep/Wake Cycle Initialization
     * =========================================================================
     * WHAT: Create sleep system and connect to brain
     * WHY:  Enable memory consolidation and synaptic homeostasis
     * HOW:  Create with defaults, link brain reference for working memory access
     */

    // Create sleep system with default configuration
    brain->sleep_system = sleep_system_create(NULL);  // NULL = use defaults
    if (!brain->sleep_system) {
        set_error("Failed to create sleep system");
        return false;
    }

    // Connect brain reference for working memory access (Phase 10.3 integration)
    sleep_set_brain_reference(brain->sleep_system, (void*)brain);

    /* =========================================================================
     * PHASE M1: Memory Engram System Initialization
     * =========================================================================
     * WHAT: Create engram system for distributed memory traces
     * WHY:  Enable biological memory encoding, consolidation, and pattern completion
     * HOW:  Create with default capacity (512 engrams), integrates with sleep/emotion
     */

    // Create engram system with default capacity (512 engrams)
    brain->engram_system = engram_system_create();
    if (!brain->engram_system) {
        set_error("Failed to create engram system");
        return false;
    }

    /**
     * PHASE M2: SYSTEMS CONSOLIDATION INITIALIZATION
     *
     * WHAT: Create systems consolidation system for hippocampus → cortex transfer
     * WHY:  Enable sleep-dependent memory consolidation and semantic abstraction
     * HOW:  Create with default capacities, link to engram and sleep systems
     */

    // Create systems consolidation system with default capacity (2048 cortical nodes)
    brain->systems_consolidation = systems_consolidation_create();
    if (!brain->systems_consolidation) {
        set_error("Failed to create systems consolidation system");
        return false;
    }

    // Link to Phase M1 engram system (source of memories to consolidate)
    systems_consolidation_set_engram_system(brain->systems_consolidation, brain->engram_system);

    // Link to sleep-wake cycle system (controls consolidation rate and replay)
    systems_consolidation_set_sleep_system(brain->systems_consolidation, &brain->sleep_system);

    /**
     * PHASE M3: WORKING MEMORY TRANSFER INITIALIZATION
     *
     * WHAT: Create WM transfer system for working memory → engram encoding
     * WHY:  Enable selective consolidation based on rehearsal, attention, and emotion
     * HOW:  Create system, link to working memory, engrams, and emotional tagging
     */

    // Create working memory transfer system with default criteria
    brain->wm_transfer_system = wm_transfer_create();
    if (!brain->wm_transfer_system) {
        set_error("Failed to create working memory transfer system");
        return false;
    }

    // Link to Phase 10.1 working memory (source of temporary information)
    wm_transfer_set_working_memory(brain->wm_transfer_system, brain->working_memory);

    // Link to Phase M1 engram system (destination for transferred memories)
    wm_transfer_set_engram_system(brain->wm_transfer_system, brain->engram_system);

    // Link to Phase 10.2 emotional tagging system (emotional salience for encoding)
    wm_transfer_set_emotional_system(brain->wm_transfer_system, brain->emotional_system);

    /*
     * PHASE M4: SEMANTIC MEMORY NETWORK INITIALIZATION
     *
     * WHAT: Create semantic memory for concept network and spreading activation
     * WHY:  Enable abstract reasoning, inference, and knowledge retrieval
     * HOW:  Create system, link to Phase M2 for concept extraction
     */

    // Create semantic memory network system
    brain->semantic_memory = semantic_memory_create();
    if (!brain->semantic_memory) {
        set_error("Failed to create semantic memory network");
        return false;
    }

    // Link to Phase M2 systems consolidation (source of semantic concepts)
    semantic_memory_set_consolidation(brain->semantic_memory, brain->systems_consolidation);

    return true;
}

/**
 * @brief Initialize executive functions subsystem (Phase 10.3)
 *
 * WHAT: Create executive controller for task management
 * WHY:  Enable goal-directed behavior and multi-tasking
 * HOW:  Create with defaults or custom config from brain
 */
bool nimcp_brain_factory_init_executive_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->executive) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_executive_control) {
        return true;  // Not enabled, but not an error
    }

    // Create executive controller
    brain->executive = executive_create();
    if (!brain->executive) {
        set_error("Failed to create executive controller");
        return false;
    }

    return true;
}

/**
 * @brief Initialize Theory of Mind subsystem (Phase 10.6)
 *
 * WHAT: Create Theory of Mind module for social cognition
 * WHY:  Enable understanding of others' beliefs, goals, and emotions
 * HOW:  Create ToM with brain reference for self-model
 *
 * COMPLEXITY: O(1)
 * MEMORY: O(1) - small fixed structures for BDI model
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_theory_of_mind_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->theory_of_mind) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_theory_of_mind) {
        return true;  // Not enabled, but not an error
    }

    // Create Theory of Mind module with brain reference for self-model
    brain->theory_of_mind = tom_create(brain);
    if (!brain->theory_of_mind) {
        set_error("Failed to create Theory of Mind module");
        return false;
    }

    return true;
}

/**
 * @brief Initialize Natural Explanations subsystem (Phase 10.7)
 *
 * WHAT: Create explanation generator for human-readable AI interpretability
 * WHY:  Enable "what-why-how" explanations of brain decisions
 * HOW:  Create explanation_generator with config-driven generation
 *
 * COMPLEXITY: O(1)
 * MEMORY: O(1) - small fixed structures for explanation templates
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_natural_explanations_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->explanation_gen) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_natural_explanations) {
        return true;  // Not enabled, but not an error
    }

    // Create explanation config from brain config flags
    explanation_config_t exp_config = explanation_default_config();
    exp_config.generate_what = brain->config.enable_natural_explanations;
    exp_config.generate_why = brain->config.enable_natural_explanations;
    exp_config.generate_how = brain->config.enable_natural_explanations;
    exp_config.generate_confidence = brain->config.enable_natural_explanations;
    exp_config.generate_alternatives = brain->config.enable_natural_explanations;
    exp_config.generate_counterfactuals = brain->config.enable_causal_explanations;
    exp_config.use_symbolic_logic = (brain->symbolic_logic != NULL);

    // Create Natural Explanations module
    brain->explanation_gen = explanation_generator_create(&exp_config);
    if (!brain->explanation_gen) {
        set_error("Failed to create Natural Explanations module");
        return false;
    }

    return true;
}

/**
 * @brief Initialize Meta-Learning subsystem (Phase 10.8)
 *
 * WHAT: Create MAML meta-learner for few-shot learning
 * WHY:  Enable rapid adaptation from 1, 5, or 10 examples
 * HOW:  Initialize meta-learner with adaptive learning rates per region
 *
 * COMPLEXITY: O(num_regions)
 * MEMORY: O(1) - small fixed structures for MAML state
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_meta_learning_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->meta_learner) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_meta_learning) {
        return true;  // Not enabled, but not an error
    }

    // Create meta-learning config from brain config
    meta_learning_config_t meta_config = meta_learning_default_config();
    meta_config.few_shot_k = (few_shot_mode_t)brain->config.meta_k_shot;
    meta_config.enable_adaptive_lr = brain->config.enable_adaptive_meta_lr;
    meta_config.enable_task_similarity = true;

    // Determine number of regions (simplified: 3 main regions)
    uint32_t num_regions = 3;  // Sensory, Association, Prefrontal

    // Create Meta-Learning module
    brain->meta_learner = meta_learner_create(&meta_config, num_regions);
    if (!brain->meta_learner) {
        set_error("Failed to create Meta-Learning module");
        return false;
    }

    return true;
}

/**
 * @brief Initialize Mental Health subsystem (Phase 10.5)
 *
 * WHAT: Create mental health monitor for disorder detection
 * WHY:  Track cognitive health and prevent harmful states
 * HOW:  Initialize monitor with 9 disorder detectors
 *
 * COMPLEXITY: O(1)
 * MEMORY: O(1) - fixed structures for monitoring state
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_mental_health_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->mental_health_monitor) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_mental_health_monitoring) {
        return true;  // Not enabled, but not an error
    }

    // Create Mental Health monitor with default config
    brain->mental_health_monitor = mental_health_create_default();
    if (!brain->mental_health_monitor) {
        set_error("Failed to create Mental Health monitor");
        return false;
    }

    return true;
}

/**
 * @brief Initialize Predictive Processing subsystem (Phase 10.9)
 *
 * WHAT: Create hierarchical predictive coding network
 * WHY:  Enable free energy minimization and active inference
 * HOW:  Initialize multi-layer predictive network
 *
 * COMPLEXITY: O(sum(layer_sizes))
 * MEMORY: O(sum(layer_sizes)) - hierarchical state space
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_predictive_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->predictive_network) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_predictive_processing) {
        return true;  // Not enabled, but not an error
    }

    // Create Predictive network with default config
    brain->predictive_network = predictive_create(NULL);  // NULL = use defaults
    if (!brain->predictive_network) {
        set_error("Failed to create Predictive Processing network");
        return false;
    }

    return true;
}

/**
 * @brief Initialize mirror neuron system for brain
 *
 * WHAT: Create and configure mirror neuron system for observation-based learning
 * WHY:  Enable social cognition, imitation learning, and action understanding
 * HOW:  Create mirror_neurons_t with config-specified parameters
 *
 * @param brain Brain to initialize mirror neurons for
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_mirror_neurons(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Don't re-initialize
    if (brain->mirror_neurons) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_mirror_neurons) {
        return true;  // Not enabled, but not an error
    }

    // Create mirror neuron config from brain config
    mirror_neuron_config_t mirror_config = mirror_neurons_get_default_config();
    // Only override defaults if values are explicitly set (non-zero)
    if (brain->config.mirror_neuron_count > 0) {
        mirror_config.num_mirror_neurons = brain->config.mirror_neuron_count;
    }
    if (brain->config.mirror_max_actions > 0) {
        mirror_config.max_actions = brain->config.mirror_max_actions;
    }
    if (brain->config.mirror_max_agents > 0) {
        mirror_config.max_agents = brain->config.mirror_max_agents;
    }
    if (brain->config.mirror_learning_rate > 0.0f) {
        mirror_config.learning_rate = brain->config.mirror_learning_rate;
    }
    if (brain->config.mirror_match_threshold > 0.0f) {
        mirror_config.match_threshold = brain->config.mirror_match_threshold;
    }

    // Enable integration with other cognitive systems
    mirror_config.enable_working_memory = brain->config.enable_working_memory;
    mirror_config.enable_theory_of_mind = brain->config.enable_theory_of_mind;
    mirror_config.enable_prediction = brain->config.enable_predictive_processing;

    // Create mirror neuron system
    brain->mirror_neurons = mirror_neurons_create(&mirror_config);
    if (!brain->mirror_neurons) {
        set_error("Failed to create mirror neuron system");
        return false;
    }

    // Integrate with other cognitive systems if they exist
    if (brain->working_memory && mirror_config.enable_working_memory) {
        mirror_neurons_integrate_working_memory(brain->mirror_neurons, brain->working_memory);
    }

    if (brain->theory_of_mind && mirror_config.enable_theory_of_mind) {
        mirror_neurons_integrate_theory_of_mind(brain->mirror_neurons, brain->theory_of_mind);
    }

    if (brain->predictive_network && mirror_config.enable_prediction) {
        mirror_neurons_integrate_predictive(brain->mirror_neurons, brain->predictive_network);
    }

    return true;
}

//=============================================================================
// Phase T1: Biological Framework Enhancements (Training Pipeline)
//=============================================================================

/**
 * @brief Initialize homeostatic plasticity subsystem
 *
 * WHAT: Create and configure homeostatic plasticity controller
 * WHY:  Maintains neural activity within healthy ranges via synaptic scaling
 * HOW:  Create homeostatic_controller_t with config-specified parameters
 *
 * BIOLOGICAL BASIS:
 * - Synaptic scaling: Adjusts all synaptic weights to maintain target firing rate
 * - Intrinsic plasticity: Modifies neuron excitability for homeostasis
 * - Metaplasticity: Adjusts plasticity rules based on activity history
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_homeostatic_plasticity_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->homeostatic) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_homeostatic_plasticity) {
        return true;  // Not enabled, but not an error
    }

    // Create homeostatic config from brain config
    homeostatic_config_t config = homeostatic_config_default();
    if (brain->config.homeostatic_target_rate_hz > 0.0f) {
        config.scaling_params.target_rate = brain->config.homeostatic_target_rate_hz;
    }
    if (brain->config.homeostatic_tau_ms > 0.0f) {
        config.scaling_params.scaling_time_constant = brain->config.homeostatic_tau_ms;
    }

    // Get number of neurons from network - use default if not available
    uint32_t num_neurons = 1000;  // Default
    if (brain->network) {
        neural_network_t base_net = adaptive_network_get_base_network(brain->network);
        if (base_net) {
            num_neurons = neural_network_get_num_neurons(base_net);
        }
    }

    // Create homeostatic controller
    brain->homeostatic = homeostatic_controller_create(&config, num_neurons);
    if (!brain->homeostatic) {
        set_error("Failed to create homeostatic plasticity controller");
        return false;
    }

    LOG_INFO("Homeostatic plasticity subsystem initialized: target_rate=%.1f Hz, tau=%.1f ms, neurons=%u",
            config.scaling_params.target_rate, config.scaling_params.scaling_time_constant, num_neurons);
    return true;
}

/**
 * @brief Initialize dendritic computation subsystem
 *
 * WHAT: Create and configure dendritic tree with NMDA dynamics
 * WHY:  Enable local computation in dendritic branches
 * HOW:  Create dendritic_tree_t with compartmental modeling
 *
 * BIOLOGICAL BASIS:
 * - NMDA dynamics: Voltage-dependent magnesium block, NMDA-dependent LTP
 * - Dendritic branches: Each branch computes local activations
 * - Compartmental modeling: Spatial integration of inputs
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_dendritic_computation_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->dendritic) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_dendritic_computation) {
        return true;  // Not enabled, but not an error
    }

    // Create dendritic tree config from brain config
    dendritic_tree_config_t config = dendritic_tree_config_default();
    if (brain->config.dendritic_branches > 0) {
        config.num_branches = brain->config.dendritic_branches;
    }
    if (brain->config.dendritic_compartments > 0) {
        config.compartments_per_branch = brain->config.dendritic_compartments;
    }

    // Create dendritic tree
    brain->dendritic = dendritic_tree_create(&config);
    if (!brain->dendritic) {
        set_error("Failed to create dendritic computation tree");
        return false;
    }

    LOG_INFO("Dendritic computation subsystem initialized: branches=%u, compartments=%u",
            config.num_branches, config.compartments_per_branch);
    return true;
}

/**
 * @brief Initialize biological predictive coding subsystem
 *
 * WHAT: Create and configure hierarchical predictive coding network
 * WHY:  Enable error-driven learning via Free Energy Principle
 * HOW:  Create pc_hierarchy_t with prediction and error units
 *
 * BIOLOGICAL BASIS:
 * - Prediction units: Generate top-down expectations
 * - Error units: Compute mismatch between prediction and input
 * - Precision weighting: Modulates error importance by confidence
 *
 * NOTE: This is different from the Phase 10.9 predictive_network (cognitive)
 *       This is for biological-level predictive coding in sensory processing
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_biological_predictive_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->predictive_coding) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_biological_predictive) {
        return true;  // Not enabled, but not an error
    }

    // Determine number of hierarchy levels and units
    uint32_t num_levels = brain->config.predictive_levels;
    if (num_levels == 0) {
        num_levels = 3;  // Default: 3 levels
    }

    // Allocate units array and set up pyramid structure
    uint32_t* units = (uint32_t*)malloc(num_levels * sizeof(uint32_t));
    if (!units) {
        set_error("Failed to allocate predictive coding units array");
        return false;
    }

    // Set up hierarchical pyramid: each level has half the units of the previous
    uint32_t base_units = 64;  // Bottom level
    for (uint32_t i = 0; i < num_levels; i++) {
        units[i] = base_units >> i;  // 64, 32, 16, 8, ...
        if (units[i] < 4) units[i] = 4;  // Minimum 4 units per level
    }

    // Create predictive coding config
    pc_hierarchy_config_t config = pc_hierarchy_config_default(num_levels, units);
    config.units_per_level = units;  // Must set explicitly

    if (brain->config.predictive_learning_rate > 0.0f) {
        config.learning_rate = brain->config.predictive_learning_rate;
    }

    // Create predictive coding hierarchy
    brain->predictive_coding = pc_hierarchy_create(&config);
    free(units);  // pc_hierarchy_create copies the array

    if (!brain->predictive_coding) {
        set_error("Failed to create biological predictive coding hierarchy");
        return false;
    }

    LOG_INFO("Biological predictive coding subsystem initialized: levels=%u, learning_rate=%.3f",
            num_levels, config.learning_rate);
    return true;
}

//=============================================================================
// Phase TM-3: Brain-Training Integration
//=============================================================================

/**
 * @brief Initialize brain-training integration subsystem
 *
 * WHAT: Create and configure training integration context
 * WHY:  Provides unified interface for loss functions, optimizers, and training events
 * HOW:  Create nimcp_brain_training_ctx with config-specified parameters
 *
 * FEATURES:
 * - Loss function management (MSE, CrossEntropy, Huber, etc.)
 * - Optimizer management (SGD, Adam, RMSProp, etc.)
 * - Automatic security registration with BBB and Security Integration
 * - Event bus integration for training events
 * - Memory pool support for gradient buffers
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_training_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->training_ctx) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_training_integration) {
        brain->enable_training_integration = false;
        return true;  // Not enabled, but not an error
    }

    // Build training integration configuration
    nimcp_brain_training_config_t training_config = nimcp_brain_training_default_config();

    // Override with brain config values
    if (brain->config.training_default_lr > 0.0f) {
        training_config.default_learning_rate = brain->config.training_default_lr;
    } else {
        training_config.default_learning_rate = brain->config.learning_rate;
    }

    training_config.enable_security = brain->config.training_register_security;
    training_config.register_with_bbb = (brain->bbb_system != NULL);
    training_config.use_memory_pool = true;

    // Phase TM-4/5/6: Advanced Training Pipeline Configuration
    training_config.enable_lr_scheduler = brain->config.enable_lr_scheduler;
    training_config.enable_regularization = brain->config.enable_regularization;
    training_config.enable_gradient_management = brain->config.enable_gradient_management;
    training_config.enable_gradient_health_check = brain->config.enable_gradient_health_check;

    // Regularization parameters
    training_config.l1_lambda = brain->config.regularization_l1_lambda;
    training_config.l2_lambda = brain->config.regularization_l2_lambda;
    training_config.dropout_rate = brain->config.dropout_rate;

    // Gradient management parameters
    training_config.gradient_accumulation_steps = brain->config.gradient_accumulation_steps;
    training_config.gradient_clip_value = brain->config.gradient_clip_value;
    training_config.gradient_clip_norm = brain->config.gradient_clip_norm;

    // Create training integration context
    brain->training_ctx = nimcp_brain_training_create(&training_config);
    if (!brain->training_ctx) {
        set_error("Failed to create brain-training integration context");
        return false;
    }

    // Initialize with security integration if available
    nimcp_result_t result = nimcp_brain_training_init(
        brain->training_ctx,
        brain->security_integration,
        NULL  // Memory manager - use internal pools
    );

    if (result != NIMCP_SUCCESS) {
        LOG_WARNING("Failed to initialize training security integration (result=%d), continuing without security", result);
        // Non-fatal - training can work without security registration
    }

    // Phase TM-4: Create default LR scheduler if enabled
    if (brain->config.enable_lr_scheduler) {
        nimcp_lr_scheduler_config_t sched_config = {
            .type = NIMCP_LR_STEP,
            .params.step = {
                .initial_lr = training_config.default_learning_rate,
                .step_size = 100,     // Decay every 100 epochs
                .gamma = 0.9f,        // 10% decay
                .min_lr = 1e-6f
            },
            .verbose = false,
            .lr_epsilon = 1e-8f
        };
        uint32_t sched_id = 0;
        nimcp_result_t sres = nimcp_brain_training_create_scheduler(brain->training_ctx, &sched_config, &sched_id);
        if (sres != NIMCP_SUCCESS || sched_id == 0) {
            LOG_WARNING("Failed to create default LR scheduler (result=%d)", sres);
        } else {
            LOG_INFO("Created default StepLR scheduler (id=%u, step_size=100, gamma=0.9)", sched_id);
        }
    }

    // Phase TM-6: Create default gradient manager if enabled
    if (brain->config.enable_gradient_management && brain->config.gradient_accumulation_steps > 1) {
        nimcp_gradient_manager_config_t gm_config = nimcp_gradient_manager_default_config();
        gm_config.use_accumulation = true;
        gm_config.accumulation.accumulation_steps = brain->config.gradient_accumulation_steps;
        gm_config.accumulation.mode = NIMCP_GRAD_ACCUM_SUM;
        gm_config.accumulation.sync_on_step = false;
        gm_config.check_nan_inf = brain->config.enable_gradient_health_check;

        uint32_t gm_id = 0;
        nimcp_result_t gres = nimcp_brain_training_create_gradient_manager(brain->training_ctx, &gm_config, &gm_id);
        if (gres != NIMCP_SUCCESS || gm_id == 0) {
            LOG_WARNING("Failed to create default gradient manager (result=%d)", gres);
        } else {
            LOG_INFO("Created default gradient manager (id=%u, accum_steps=%u)",
                    gm_id, brain->config.gradient_accumulation_steps);
        }
    }

    brain->enable_training_integration = true;

    // Phase TPB-1: Initialize Training-Plasticity Bridge
    // This connects the training pipeline to biological plasticity systems
    if (brain->config.enable_training_integration && brain->neuromodulator_system) {
        tpb_config_t tpb_config = tpb_config_default();

        // Link to the brain's neuromodulator system
        tpb_config.neuromod_system = brain->neuromodulator_system;

        // Configure RPE parameters from training config
        tpb_config.rpe_to_da_gain = 0.5f;  // Moderate DA sensitivity to loss changes

        // Enable CoW for weight snapshots
        tpb_config.enable_cow = true;

        // Create the plasticity bridge
        brain->plasticity_bridge = tpb_create(&tpb_config);
        if (brain->plasticity_bridge) {
            brain->enable_plasticity_bridge = true;

            // Configure default cortical region covering all neurons
            tpb_region_config_t cortical = tpb_region_cortical_default();
            cortical.neuron_start_idx = 0;
            cortical.neuron_end_idx = brain_get_neuron_count(brain);
            tpb_configure_region(brain->plasticity_bridge, &cortical, NULL);

            // Connect the plasticity bridge to the training context
            if (brain->training_ctx) {
                nimcp_result_t conn_result = nimcp_brain_training_connect_plasticity_bridge(
                    brain->training_ctx, brain->plasticity_bridge);
                if (conn_result == NIMCP_SUCCESS) {
                    // Enable biological modulation by default (50% blend)
                    nimcp_brain_training_set_biological_modulation(brain->training_ctx, 0.5f);
                    LOG_INFO("Training-Plasticity Bridge connected to training context");
                } else {
                    LOG_WARNING("Failed to connect plasticity bridge to training context");
                }
            }

            LOG_INFO("Training-Plasticity Bridge initialized: connected to neuromodulator system");
        } else {
            brain->enable_plasticity_bridge = false;
            LOG_WARNING("Failed to create Training-Plasticity Bridge, continuing without");
        }
    } else {
        brain->plasticity_bridge = NULL;
        brain->enable_plasticity_bridge = false;
    }

    LOG_INFO("Brain-training integration initialized: lr=%.4f, security=%s, "
             "lr_sched=%s, regularization=%s, grad_mgmt=%s, plasticity_bridge=%s",
            training_config.default_learning_rate,
            training_config.enable_security ? "enabled" : "disabled",
            brain->config.enable_lr_scheduler ? "enabled" : "disabled",
            brain->config.enable_regularization ? "enabled" : "disabled",
            brain->config.enable_gradient_management ? "enabled" : "disabled",
            brain->enable_plasticity_bridge ? "enabled" : "disabled");

    return true;
}

/**
 * @brief Initialize memory consolidation subsystem
 *
 * WHAT: Create and configure background memory consolidation system
 * WHY:  Enable sleep-like memory strengthening and replay
 * HOW:  Start background consolidation thread with configured interval
 *
 * BIOLOGICAL BASIS: Sleep-dependent memory consolidation
 * - Memory replay during sleep/rest periods
 * - Synaptic homeostasis and pruning
 * - Selective strengthening of important memories
 *
 * @param brain Brain instance
 * @return true on success, false on failure
 */
bool nimcp_brain_factory_init_consolidation_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->consolidation) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_consolidation) {
        return true;  // Not enabled, but not an error
    }

    // Create consolidation config with defaults
    consolidation_config_t consolidation_config = consolidation_default_config();

    // Start background consolidation with 5-minute interval (300 seconds)
    brain->consolidation = brain_start_background_consolidation(
        brain,
        300,  // Consolidate every 5 minutes
        &consolidation_config
    );

    if (!brain->consolidation) {
        set_error("Failed to start background consolidation");
        return false;
    }

    return true;
}

/**
 * @brief Initialize Curiosity Engine subsystem
 *
 * WHAT: Create and configure curiosity-driven exploration system
 * WHY:  Enable novelty detection, knowledge gap detection, and exploration drive
 * HOW:  Create curiosity engine with learner name from brain config
 *
 * BIOLOGICAL BASIS: Intrinsic motivation and curiosity
 * - Dopaminergic response to novelty (midbrain)
 * - Exploration-exploitation trade-off (prefrontal cortex)
 * - Information-seeking behavior (anterior cingulate)
 * - Knowledge gap detection (metacognition)
 *
 * COGNITIVE BENEFITS (from COGNITIVE_AUDIT.md):
 * - 40% faster learning on novel patterns
 * - Intelligent exploration vs exploitation balance
 * - Prioritized learning of novel information
 *
 * @param brain Brain instance
 * @return true on success, false on failure
 */
bool nimcp_brain_factory_init_curiosity_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Guard: Check if already initialized
    if (brain->curiosity) {
        return true;  // Already initialized
    }

    // Guard: Only create if enabled in config
    if (!brain->config.enable_curiosity) {
        return true;  // Not enabled, but not an error
    }

    // Create curiosity engine with parent brain reference (module pattern)
    // Curiosity uses parent brain's neuromodulator system instead of creating separate brains
    brain->curiosity = curiosity_engine_create(brain, "nimcp_brain");
    if (!brain->curiosity) {
        set_error("Failed to create curiosity engine");
        return false;
    }

    // Set baseline curiosity to moderate-high (like a curious adult learner)
    // Infants: 0.95, Children: 0.85, Adults: 0.6-0.7
    curiosity_set_baseline(brain->curiosity, 0.7f);

    return true;
}

/**
 * @brief Initialize Salience subsystem (Attention/Relevance Evaluation)
 *
 * WHAT: Create salience evaluator for fast attention/relevance scoring
 * WHY:  Enable brain to quickly determine what inputs deserve attention
 * HOW:  Create salience evaluator with default configuration
 *
 * CAPABILITIES:
 * - Fast salience evaluation (10x faster than full decision)
 * - Novelty detection (never seen before)
 * - Surprise measurement (violated expectations)
 * - Urgency scoring (requires immediate response)
 * - Attention competition in global workspace
 *
 * COGNITIVE BENEFITS:
 * - Selective attention to important stimuli
 * - Reduced computational cost (0.1ms vs 1ms per input)
 * - Emotional-salience integration for mood-biased attention
 * - Surprise-driven arousal modulation
 *
 * @param brain Brain instance
 * @return true on success, false on failure
 */
bool nimcp_brain_factory_init_salience_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Guard: Check if already initialized
    if (brain->salience) {
        return true;  // Already initialized
    }

    // Guard: Only create if enabled in config
    if (!brain->config.enable_salience) {
        return true;  // Not enabled, but not an error
    }

    // Create salience evaluator with default configuration
    salience_config_t config = salience_default_config();
    brain->salience = salience_evaluator_create(brain, &config);

    if (!brain->salience) {
        set_error("Failed to create salience evaluator");
        return false;
    }

    return true;
}

/**
 * @brief Initialize Introspection subsystem (Self-Awareness)
 *
 * WHAT: Create introspection context for self-monitoring and metacognition
 * WHY:  Enable brain to examine its own internal state (consciousness requirement)
 * HOW:  Create introspection context with default configuration
 *
 * CAPABILITIES:
 * - Query active neurons and network state
 * - Measure uncertainty (epistemic + aleatoric)
 * - Track learned patterns
 * - Monitor activity history
 * - Network topology inspection
 *
 * CRITICAL FOR:
 * - Self-awareness and metacognition
 * - Uncertainty-aware decision making
 * - Explanation generation
 * - Wellbeing monitoring
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_introspection_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Guard: Check if already initialized
    if (brain->introspection) {
        return true;  // Already initialized
    }

    // Create introspection context with default configuration
    introspection_config_t config = introspection_default_config();

    // Customize configuration for NIMCP
    config.default_strategy = STATE_STRATEGY_BALANCED;  // Balance speed vs accuracy
    config.activity_threshold = 0.3f;                   // Neurons above 0.3 = "active"
    config.history_size = 100;                          // Track last 100 states
    config.enable_pattern_tracking = true;              // Track learned patterns
    config.enable_uncertainty_estimation = true;        // Enable uncertainty
    config.uncertainty_ensemble_size = 5;               // 5 models for uncertainty

    brain->introspection = introspection_context_create(brain, &config);

    if (!brain->introspection) {
        set_error("Failed to create introspection context");
        return false;
    }

    return true;
}

/**
 * @brief Initialize Connectivity Health Monitoring subsystem (Phase 1.5.4)
 *
 * WHAT: Enable periodic brain connectivity health assessment
 * WHY:  Self-awareness of network organizational quality and information flow
 * HOW:  Community detection + hub analysis + Shannon metrics + graph topology
 *
 * BIOLOGICAL INSPIRATION:
 * - Modular organization in cortex (Bullmore & Sporns, 2012)
 * - Hub neurons in prefrontal-parietal network (Power et al., 2013)
 * - Small-world topology in brain networks (Watts & Strogatz, 1998)
 * - Information integration in conscious processing (Tononi, 2004)
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_connectivity_health_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Initialize connectivity health configuration with defaults
    brain->connectivity_health_config = connectivity_health_default_config();

    // Initialize monitoring state
    brain->enable_connectivity_monitoring = false;  // Opt-in by default
    brain->last_connectivity_assessment_time_ms = 0;
    brain->connectivity_health_callback = NULL;
    brain->connectivity_health_callback_context = NULL;

    // Initialize health structure to safe defaults
    memset(&brain->last_connectivity_health, 0, sizeof(brain_connectivity_health_t));
    brain->last_connectivity_health.overall_health = 0.0f;
    brain->last_connectivity_health.is_healthy = false;

    // Enable monitoring if Shannon monitoring is enabled (synergy)
    if (brain->enable_shannon_monitoring) {
        brain->enable_connectivity_monitoring = true;
    }

    return true;
}

/**
 * @brief Initialize Middleware Controller subsystem (Phase 1.5.5)
 *
 * WHAT: Create unified command interface for cognitive → middleware control
 * WHY:  Enable top-down cognitive modulation of middleware processing
 * HOW:  Wrap attention gate, routing table, and pattern library with command API
 *
 * BIOLOGICAL INSPIRATION:
 * - Top-down attention control from prefrontal cortex (Miller & Cohen, 2001)
 * - Executive control of sensory processing (Desimone & Duncan, 1995)
 * - Cognitive control of information routing (Posner & Petersen, 1990)
 *
 * PERFORMANCE TARGET: <5µs per command (real-time cognitive control)
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_middleware_controller_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Initialize middleware controller state
    brain->middleware_controller = NULL;
    brain->enable_middleware_controller = false;  // Opt-in by default

    // Create controller if executive controller is enabled (synergy with Phase 1.5.2)
    if (brain->executive) {
        // Create with default configuration
        brain->middleware_controller = middleware_controller_create(brain);

        if (brain->middleware_controller) {
            brain->enable_middleware_controller = true;
        }
        // Note: Failure to create is not fatal - cognitive control still works via executive
    }

    return true;
}

/**
 * @brief Initialize Ethics Engine subsystem (Phase 11: Part I.0)
 *
 * WHAT: Create Golden Rule ethics engine for ethical decision-making
 * WHY:  Ensure all actions align with "do unto others" principle
 * HOW:  Create ethics engine with empathy network for perspective-taking
 *
 * BIOLOGICAL BASIS: Prefrontal Cortex (Moral Reasoning)
 * - Evaluates actions against ethical principles
 * - Uses empathy to predict impact on others
 * - Hard-wired Golden Rule as foundational constraint
 *
 * DESIGN PRINCIPLE:
 * "Do unto others as you would have them done unto you"
 * - Ultimate goal: Improve human condition through compassion and fairness
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_ethics_engine_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Check if ethics is disabled via config
    if (!brain->config.enable_ethics) {
        brain->ethics = NULL;
        return true;  // Not an error - ethics disabled by config
    }

    // Guard: Check if already initialized
    if (brain->ethics) {
        return true;  // Already initialized
    }

    // Create ethics engine configuration
    ethics_config_t config = {
        .policies = NULL,                    // Will be auto-initialized with Golden Rule
        .num_policies = 0,
        .callback = NULL,                    // No custom callback
        .callback_context = NULL,
        .default_severity = 0.5f,            // Moderate default severity
        .enable_learning = true,             // Enable learning from outcomes
        .action_feature_size = 20,           // Feature vector size for actions
        .max_agents = 10,                    // Maximum number of agents to consider
        .golden_rule_threshold = 0.0f,       // Always evaluate (no threshold)
        .empathy_weight = 0.7f               // High weight for empathy signals
    };

    // Create ethics engine with empathy network integration
    brain->ethics = ethics_engine_create(&config);

    if (!brain->ethics) {
        set_error("Failed to create ethics engine");
        return false;
    }

    return true;
}

/**
 * @brief Initialize Empathy Network subsystem (Phase 11: Part I.1)
 *
 * WHAT: Create empathy network for perspective-taking and emotional understanding
 * WHY:  Ethical decisions require simulating impact on others via mirror neurons
 * HOW:  Create empathy network with mirror neuron integration
 *
 * BIOLOGICAL BASIS: Mirror Neuron System
 * - Simulates other agents' emotional states (perspective-taking)
 * - Enables emotional contagion and empathy
 * - Integrates with ethics engine for Golden Rule evaluation
 * - Connects to mirror neurons for action understanding
 *
 * INTEGRATION POINTS:
 * - Ethics Engine: Provides empathy for ethical decisions
 * - Empathetic Response: Enables compassionate communication
 * - Mirror Neurons: Links action observation to emotional simulation
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_empathy_network_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Empathy network requires ethics to be enabled (it's part of ethical reasoning)
    if (!brain->config.enable_ethics) {
        brain->empathy_network = NULL;
        return true;  // Not an error - empathy requires ethics
    }

    // Guard: Check if already initialized
    if (brain->empathy_network) {
        return true;  // Already initialized
    }

    // Create empathy network configuration
    empathy_config_t config = {
        .mirror_network = brain->mirror_neurons,  // Link to mirror neurons if available
        .observation_window_ms = 1000,            // 1 second observation window
        .empathy_threshold = 0.5f                 // Minimum activation for empathy response
    };

    // Create empathy network
    brain->empathy_network = empathy_network_create(&config);

    if (!brain->empathy_network) {
        set_error("Failed to create empathy network");
        return false;
    }

    return true;
}

/**
 * @brief Initialize Empathetic Response Engine subsystem (Phase 11: Part I.2)
 *
 * WHAT: Create non-reactive empathetic response system for emotional support
 * WHY:  Enable safe, supportive responses to negative emotions (NEVER react negatively)
 * HOW:  Create empathetic response engine with ethics and empathy network integration
 *
 * CORE PRINCIPLE:
 * NEVER produce negative reactions to negative emotions (rage, hate, fear, disgust, despair)
 * Always respond with validation, empathy, and support
 *
 * SAFETY CRITICAL:
 * - Detects crisis situations (suicide, self-harm, abuse)
 * - Provides immediate escalation to human support
 * - Uses Golden Rule validation for all responses
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_empathetic_response_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Guard: Check if already initialized
    if (brain->empathetic_response_engine) {
        return true;  // Already initialized
    }

    // Always enable empathetic response (safety critical for user interaction)
    // This is not optional - needed for safe human-AI interaction

    // Import empathetic response API (forward declare to avoid circular dependency)
    typedef void* (*empathetic_response_create_fn)(void*, void*);
    extern void* empathetic_response_create(void* ethics_engine, void* empathy_network);

    // Create empathetic response engine with ethics and empathy network
    brain->empathetic_response_engine = empathetic_response_create(
        brain->ethics,            // For Golden Rule validation
        brain->empathy_network    // Empathy network for emotional understanding
    );

    if (!brain->empathetic_response_engine) {
        set_error("Failed to create empathetic response engine");
        return false;
    }

    return true;
}

/**
 * @brief Initialize Autobiographical Memory subsystem (Phase 12)
 *
 * WHAT: Create episodic self-memory system for continuous identity
 * WHY:  Self-awareness requires memory of "who I was" across time
 * HOW:  Create autobiographical memory with timeline-indexed episodes
 *
 * BIOLOGICAL BASIS: Hippocampal episodic memory system
 * - Timeline-indexed experiences
 * - Emotional tagging (amygdala integration)
 * - Sleep consolidation (memory strengthening/pruning)
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_autobiographical_memory_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Guard: Check if already initialized
    if (brain->autobio) {
        return true;  // Already initialized
    }

    // Create autobiographical memory with 10,000 memory capacity
    brain->autobio = autobio_create(10000);

    if (!brain->autobio) {
        set_error("Failed to create autobiographical memory system");
        return false;
    }

    return true;
}

/**
 * @brief Initialize Self-Model subsystem (Phase 12)
 *
 * WHAT: Create explicit self-representation system
 * WHY:  Self-awareness requires structured "I am X" distinct from "World is Y"
 * HOW:  Create self-model with identity, beliefs, capabilities, boundaries
 *
 * BIOLOGICAL BASIS: Medial Prefrontal Cortex (self-referential processing)
 * - Explicit identity representation
 * - Self-beliefs and self-knowledge
 * - Self-other boundary tracking
 * - Assertive, confident self-concept (HIGH self-esteem)
 *
 * DESIGN PHILOSOPHY:
 * - Confident and self-respecting (NOT meek/passive)
 * - Healthy boundaries and assertiveness
 * - Polite but will refuse abuse
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_self_model_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Guard: Check if already initialized
    if (brain->self_model) {
        return true;  // Already initialized
    }

    // Create self-model with assertive, confident identity
    brain->self_model = self_model_create(
        "NIMCP",
        "AI learning system with self-awareness and ethical reasoning",
        "Help humans learn while maintaining ethical behavior, healthy boundaries, and self-respect"
    );

    if (!brain->self_model) {
        set_error("Failed to create self-model system");
        return false;
    }

    // Phase 12: Wire personality into self-model
    if (brain->personality) {
        if (!self_model_set_personality(brain->self_model, brain->personality)) {
            set_error("Failed to wire personality into self-model");
            return false;
        }
    }

    return true;
}

/**
 * @brief Initialize Global Workspace Architecture subsystem
 *
 * WHAT: Create and configure global workspace for conscious access
 * WHY:  Enable broadcast architecture for cross-module information integration
 * HOW:  Create workspace with brain config parameters
 *
 * BIOLOGICAL BASIS: Global Workspace Theory (Baars, 1988; Dehaene, 2011)
 * - Limited-capacity broadcast architecture
 * - Winner-take-all competition for conscious access
 * - Refractory period matches attentional blink (~50ms)
 *
 * @param brain Brain instance
 * @return true on success, false on failure
 */
bool nimcp_brain_factory_init_global_workspace_subsystem(brain_t brain)
{
    if (!brain) {
        return false;
    }

    // Check if already initialized
    if (brain->global_workspace) {
        return true;  // Already initialized
    }

    // Only create if enabled in config
    if (!brain->config.enable_global_workspace) {
        return true;  // Not enabled, but not an error
    }

    // Create global workspace config from brain config
    global_workspace_config_t gw_config = global_workspace_default_config();

    // Override defaults with brain config if specified
    if (brain->config.workspace_capacity_dim > 0) {
        gw_config.capacity_dim = brain->config.workspace_capacity_dim;
    }
    if (brain->config.workspace_ignition_threshold > 0.0f) {
        gw_config.ignition_threshold = brain->config.workspace_ignition_threshold;
    }
    if (brain->config.workspace_refractory_ms > 0) {
        gw_config.refractory_period_ms = brain->config.workspace_refractory_ms;
    }
    if (brain->config.workspace_history_depth > 0) {
        gw_config.history_depth = brain->config.workspace_history_depth;
    }
    gw_config.enable_history = brain->config.workspace_enable_history;

    // Create global workspace
    brain->global_workspace = global_workspace_create_custom(&gw_config);
    if (!brain->global_workspace) {
        set_error("Failed to create global workspace");
        return false;
    }

    // Subscribe key cognitive modules to workspace broadcasts
    // This allows them to receive conscious broadcasts
    if (brain->working_memory) {
        global_workspace_subscribe(brain->global_workspace, MODULE_WORKING_MEMORY);
    }
    if (brain->executive) {
        global_workspace_subscribe(brain->global_workspace, MODULE_EXECUTIVE);
    }
    if (brain->ethics) {
        global_workspace_subscribe(brain->global_workspace, MODULE_ETHICS);
    }
    if (brain->introspection) {
        global_workspace_subscribe(brain->global_workspace, MODULE_INTROSPECTION);
    }
    if (brain->salience) {
        global_workspace_subscribe(brain->global_workspace, MODULE_SALIENCE);
    }
    if (brain->theory_of_mind) {
        global_workspace_subscribe(brain->global_workspace, MODULE_THEORY_OF_MIND);
    }

    // Phase 1.5.3: Enable Shannon information-weighted competition if requested
    // WHAT: Add Shannon entropy monitoring to workspace competition
    // WHY:  High-information, salient content should win workspace access
    // HOW:  Enable Shannon features, set subscriber capacities
    if (brain->enable_shannon_monitoring) {
        shannon_workspace_config_t shannon_config = shannon_workspace_default_config();

        // Enable all Shannon features
        shannon_config.enable_info_weighted_competition = true;
        shannon_config.enable_shannon_monitoring = true;
        shannon_config.enable_adaptive_rate = true;

        if (global_workspace_enable_shannon(brain->global_workspace, &shannon_config)) {
            // Set subscriber capacities based on module complexity
            // Higher capacity = can handle more information flow
            if (brain->working_memory) {
                global_workspace_set_subscriber_capacity(
                    brain->global_workspace, MODULE_WORKING_MEMORY, 150.0f);
            }
            if (brain->executive) {
                global_workspace_set_subscriber_capacity(
                    brain->global_workspace, MODULE_EXECUTIVE, 200.0f);
            }
            if (brain->ethics) {
                global_workspace_set_subscriber_capacity(
                    brain->global_workspace, MODULE_ETHICS, 100.0f);
            }
            if (brain->introspection) {
                global_workspace_set_subscriber_capacity(
                    brain->global_workspace, MODULE_INTROSPECTION, 80.0f);
            }
            if (brain->salience) {
                global_workspace_set_subscriber_capacity(
                    brain->global_workspace, MODULE_SALIENCE, 120.0f);
            }
            if (brain->theory_of_mind) {
                global_workspace_set_subscriber_capacity(
                    brain->global_workspace, MODULE_THEORY_OF_MIND, 100.0f);
            }
        }
    }

    return true;
}

/**
 * @brief Initialize Axon Network subsystem (Phase 1.5.6)
 *
 * WHAT: Create axon network for realistic signal propagation with conduction delays
 * WHY:  Enable biologically realistic action potential propagation between neurons
 * HOW:  Create axon for each neuron, configure myelination based on neuron type
 *
 * BIOLOGICAL INSPIRATION:
 * - Axonal conduction delays vary 0.5-100 m/s based on myelination (Waxman, 1977)
 * - Myelinated axons have saltatory conduction (Huxley & Stämpfli, 1949)
 * - Axon diameter affects conduction velocity (Hursh, 1939)
 *
 * PERFORMANCE TARGET: O(n) where n = num_neurons
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_axon_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Check if axons are disabled via config
    if (!brain->config.enable_axons) {
        brain->axon_network = NULL;
        return true;  // Not an error - axons disabled by config
    }

    // Check if lazy initialization is requested
    if (brain->config.lazy_axon_init) {
        brain->axon_network = NULL;
        // Axons will be created on first access via brain_ensure_axons()
        return true;
    }

    // Guard: No network
    if (!brain->network) {
        brain->axon_network = NULL;
        return true;  // Not an error - just no network yet
    }

    // Get base network for neuron access
    neural_network_t nn = adaptive_network_get_base_network(brain->network);
    if (!nn) {
        brain->axon_network = NULL;
        return true;  // Not an error - empty network
    }

    // Get neuron count using accessor function (opaque type pattern)
    uint32_t num_neurons = neural_network_get_num_neurons(nn);
    if (num_neurons == 0) {
        brain->axon_network = NULL;
        return true;  // Not an error - empty network
    }

    // Create axon network with capacity for all neurons
    axon_network_t* axon_net = axon_network_create(num_neurons);
    if (!axon_net) {
        // Axon network creation failed - continue without axons (graceful degradation)
        brain->axon_network = NULL;
        return true;  // Not fatal - direct connections still work
    }

    // Create axon for each neuron
    for (uint32_t i = 0; i < num_neurons; i++) {
        neuron_t* neuron = neural_network_get_neuron(nn, i);
        if (!neuron) {
            continue;  // Skip invalid neurons
        }

        // Determine axon type based on neuron type
        axon_type_t axon_type = AXON_TYPE_UNMYELINATED;  // Default
        float myelination = 0.0f;

        // Excitatory neurons tend to have larger, myelinated axons
        if (neuron->type == NEURON_EXCITATORY) {
            axon_type = AXON_TYPE_MYELINATED;
            myelination = 0.6f + (float)(rand() % 40) / 100.0f;  // 0.6-1.0
        } else {
            // Inhibitory interneurons often have unmyelinated or lightly myelinated axons
            axon_type = AXON_TYPE_UNMYELINATED;
            myelination = (float)(rand() % 30) / 100.0f;  // 0.0-0.3
        }

        // Calculate axon properties
        float length = 100.0f + (float)(rand() % 400);  // 100-500 μm
        float diameter = 0.5f + (float)(rand() % 20) / 10.0f;  // 0.5-2.5 μm

        // Create axon using the axon_create API
        // Parameters: id, type, source_neuron_id, target_synapse_id, length, diameter
        axon_t* axon = axon_create(i, axon_type, neuron->id, 0, length, diameter);
        if (axon) {
            // Set myelination level
            axon_set_myelination(axon, myelination);

            // Add to network
            if (axon_network_add(axon_net, axon)) {
                neuron->axon_id = axon->id;
            } else {
                axon_destroy(axon);
                neuron->axon_id = 0;
            }
        } else {
            neuron->axon_id = 0;  // No axon - direct connection
        }
    }

    brain->axon_network = axon_net;
    return true;
}

/**
 * @brief Initialize Dendrite Network subsystem (Phase 1.5.7)
 *
 * WHAT: Create dendrite network for spatiotemporal signal integration
 * WHY:  Enable biologically realistic dendritic computation and synaptic integration
 * HOW:  Create dendrites for each neuron, add segments and spines for synaptic inputs
 *
 * BIOLOGICAL INSPIRATION:
 * - Dendritic integration uses cable theory (Rall, 1959)
 * - Spines are primary sites of excitatory input (Yuste, 2010)
 * - Dendrites perform nonlinear computation (Polsky et al., 2004)
 *
 * PERFORMANCE TARGET: O(n) where n = num_neurons
 *
 * @param brain Brain instance
 * @return true on success, false on error
 */
bool nimcp_brain_factory_init_dendrite_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        return false;
    }

    // Check if dendrites are disabled via config
    if (!brain->config.enable_dendrites) {
        brain->dendrite_network = NULL;
        return true;  // Not an error - dendrites disabled by config
    }

    // Check if lazy initialization is requested
    if (brain->config.lazy_dendrite_init) {
        brain->dendrite_network = NULL;
        // Dendrites will be created on first access via brain_ensure_dendrites()
        return true;
    }

    // Guard: No network
    if (!brain->network) {
        brain->dendrite_network = NULL;
        return true;  // Not an error - just no network yet
    }

    // Get base network for neuron access
    neural_network_t nn = adaptive_network_get_base_network(brain->network);
    if (!nn) {
        brain->dendrite_network = NULL;
        return true;  // Not an error - empty network
    }

    // Get neuron count using accessor function (opaque type pattern)
    uint32_t num_neurons = neural_network_get_num_neurons(nn);
    if (num_neurons == 0) {
        brain->dendrite_network = NULL;
        return true;  // Not an error - empty network
    }

    // Create dendrite network with capacity for all neurons (each can have multiple dendrites)
    // Estimate: ~2 dendrites per neuron on average (basal + apical for pyramidal)
    dendrite_network_t* dend_net = dendrite_network_create(num_neurons * 2);
    if (!dend_net) {
        // Dendrite network creation failed - continue without dendrites (graceful degradation)
        brain->dendrite_network = NULL;
        return true;  // Not fatal - direct inputs still work
    }

    // Create dendrites for each neuron
    for (uint32_t i = 0; i < num_neurons; i++) {
        neuron_t* neuron = neural_network_get_neuron(nn, i);
        if (!neuron) {
            continue;  // Skip invalid neurons
        }

        // Initialize dendrite tracking for this neuron
        // Allocate space for up to 4 dendrites per neuron
        neuron->dendrite_ids = (uint32_t*)nimcp_calloc(4, sizeof(uint32_t));
        neuron->num_dendrites = 0;

        if (!neuron->dendrite_ids) {
            continue;  // Skip this neuron on allocation failure
        }

        // Determine dendrite configuration based on neuron type
        uint32_t num_dendrites_to_create = 1;  // Default: 1 dendrite
        dendrite_type_t primary_type = DENDRITE_TYPE_BASAL;

        // Excitatory neurons (pyramidal-like) have more elaborate dendritic trees
        if (neuron->type == NEURON_EXCITATORY) {
            num_dendrites_to_create = 2;  // Basal + apical
            primary_type = DENDRITE_TYPE_BASAL;
        }

        // Create dendrites for this neuron
        for (uint32_t d = 0; d < num_dendrites_to_create && neuron->num_dendrites < 4; d++) {
            dendrite_type_t dtype = (d == 0) ? primary_type : DENDRITE_TYPE_APICAL;

            dendrite_config_t dend_config = {
                .id = dend_net->num_dendrites,  // Sequential ID
                .type = dtype,
                .target_neuron_id = neuron->id,
                .total_length = 150.0f + (float)(rand() % 200),  // 150-350 μm
                .mean_diameter = 1.0f + (float)(rand() % 20) / 10.0f,  // 1.0-3.0 μm
                .start_pos = {0.0f, 0.0f, 0.0f},
                .integration_window_ms = 15.0f + (float)(rand() % 20),  // 15-35 ms
                .structural_plasticity = 0.01f,
                .ltp_threshold = 0.7f + (float)(rand() % 20) / 100.0f,  // 0.7-0.9
                .ltd_threshold = 0.2f + (float)(rand() % 20) / 100.0f   // 0.2-0.4
            };

            // Create dendrite
            dendrite_t* dendrite = dendrite_create(&dend_config);
            if (!dendrite) {
                continue;  // Skip on creation failure
            }

            // Create initial segments (3-5 compartments)
            uint32_t num_segments = 3 + (rand() % 3);  // 3-5 segments
            segment_config_t* seg_configs = (segment_config_t*)nimcp_calloc(
                num_segments, sizeof(segment_config_t));

            if (seg_configs) {
                float path_dist = 0.0f;
                for (uint32_t s = 0; s < num_segments; s++) {
                    seg_configs[s].type = (s == 0) ? DENDRITE_SEGMENT_PROXIMAL :
                                          (s == num_segments - 1) ? DENDRITE_SEGMENT_TERMINAL :
                                          DENDRITE_SEGMENT_SHAFT;
                    seg_configs[s].parent_segment = (s == 0) ? UINT32_MAX : (s - 1);
                    seg_configs[s].length = 30.0f + (float)(rand() % 40);  // 30-70 μm
                    seg_configs[s].diameter = dend_config.mean_diameter *
                                             (1.0f - 0.1f * s);  // Taper
                    seg_configs[s].path_distance = path_dist;
                    path_dist += seg_configs[s].length;
                    seg_configs[s].has_active_properties = (dtype == DENDRITE_TYPE_APICAL);
                }

                dendrite_create_segments(dendrite, num_segments, seg_configs);
                nimcp_free(seg_configs);
            }

            // Add dendrite to network
            if (dendrite_network_add(dend_net, dendrite)) {
                neuron->dendrite_ids[neuron->num_dendrites] = dendrite->id;
                neuron->num_dendrites++;
            } else {
                dendrite_destroy(dendrite);
            }
        }
    }

    brain->dendrite_network = dend_net;
    return true;
}

//=============================================================================
// Phase CC-1: Cortical Columns Subsystem Initialization (Tier 0.65)
//=============================================================================

/**
 * @brief Initialize cortical columns subsystem
 *
 * WHAT: Creates hierarchical cortical column architecture with minicolumns,
 *       hypercolumns, laminar organization, and topographic maps.
 *
 * WHY:  Implements biologically-realistic columnar organization (Douglas & Martin 1991)
 *       for feature detection and competitive dynamics in neural processing.
 *
 * HOW:  1. Create cortical column memory pool for efficient allocation
 *       2. Create hypercolumns based on brain region requirements
 *       3. Initialize 6-layer laminar structure for each region
 *       4. Set up canonical microcircuit connectivity
 *       5. Create topographic maps for sensory processing
 *       6. Initialize orientation columns for V1 (if visual enabled)
 *       7. Create feature hypercolumns for multi-dimensional features
 *
 * ARCHITECTURE:
 *   Hypercolumn (100K neurons):
 *   - Contains ~100 minicolumns
 *   - Each minicolumn ~80-100 neurons
 *   - Competition modes: winner-take-all, k-winners, softmax
 *   - Lateral inhibition (Mexican hat) for sharpening
 *
 * @param brain Brain to initialize cortical columns for
 * @return true on success (or if disabled), false on fatal error
 *
 * COMPLEXITY: O(H × M × N) where H=hypercolumns, M=minicolumns, N=neurons
 * THREAD-SAFE: Yes (pool is thread-safe after creation)
 *
 * @version 2.7.0 Phase CC-1
 * @date 2025-11-25
 */
bool nimcp_brain_factory_init_cortical_columns_subsystem(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        set_error("nimcp_brain_factory_init_cortical_columns_subsystem: NULL brain");
        return false;
    }

    // Check if cortical columns are enabled
    if (!brain->config.enable_cortical_columns) {
        // Initialize pointers to NULL for safe cleanup
        brain->cortical_column_pool = NULL;
        brain->hypercolumns = NULL;
        brain->num_hypercolumns = 0;
        brain->laminar_system = NULL;
        brain->columnar_connectivity = NULL;
        brain->visual_topographic_map = NULL;
        brain->auditory_topographic_map = NULL;
        brain->somatosensory_topographic_map = NULL;
        brain->orientation_hypercolumns = NULL;
        brain->num_orientation_hypercolumns = 0;
        brain->feature_hypercolumns = NULL;
        brain->num_feature_hypercolumns = 0;
        brain->enable_cortical_columns = false;
        brain->last_cortical_update_us = 0;
        return true;  // Disabled, not an error
    }

    // Check if already initialized (prevent double initialization)
    if (brain->cortical_column_pool) {
        return true;  // Already initialized
    }

    LOG_INFO("Initializing cortical columns subsystem (Phase CC-1)");

    // Get configuration parameters with defaults
    uint32_t num_hypercolumns = brain->config.num_hypercolumns > 0 ?
        brain->config.num_hypercolumns : 10;  // Default: 10 hypercolumns
    uint32_t minicolumns_per_hc = brain->config.minicolumns_per_hypercolumn > 0 ?
        brain->config.minicolumns_per_hypercolumn : 100;  // Default: 100 minicolumns
    uint32_t neurons_per_mc = brain->config.neurons_per_minicolumn > 0 ?
        brain->config.neurons_per_minicolumn : 80;  // Default: 80 neurons

    // Step 1: Create cortical column memory pool
    cortical_column_pool_config_t pool_config = {
        .max_minicolumns = num_hypercolumns * minicolumns_per_hc,
        .max_hypercolumns = num_hypercolumns,
        .max_neurons_per_minicolumn = neurons_per_mc,
        .enable_cow_support = brain->is_cow_clone || brain->can_use_readonly
    };

    brain->cortical_column_pool = cortical_column_pool_create(&pool_config);
    if (!brain->cortical_column_pool) {
        LOG_WARNING("Failed to create cortical column pool - continuing without columnar organization");
        brain->enable_cortical_columns = false;
        return true;  // Non-fatal: brain still works without columns
    }

    // Step 2: Create hypercolumns array
    brain->hypercolumns = (hypercolumn_t**)nimcp_calloc(num_hypercolumns, sizeof(hypercolumn_t*));
    if (!brain->hypercolumns) {
        LOG_WARNING("Failed to allocate hypercolumns array");
        cortical_column_pool_destroy(brain->cortical_column_pool);
        brain->cortical_column_pool = NULL;
        brain->enable_cortical_columns = false;
        return true;  // Non-fatal
    }

    brain->num_hypercolumns = num_hypercolumns;

    // Step 3: Initialize laminar structure if enabled
    if (brain->config.enable_laminar_structure) {
        // Create default laminar structure for cortical region
        // Pass NULL to use default layer configurations
        brain->laminar_system = laminar_structure_create(NULL);
        if (!brain->laminar_system) {
            LOG_WARNING("Failed to create laminar structure - continuing without layers");
        } else {
            LOG_INFO("Created 6-layer laminar structure with default configuration");
        }
    }

    // Step 4: Initialize columnar connectivity if enabled
    if (brain->config.enable_columnar_connectivity) {
        // Create canonical microcircuit connectivity
        // Estimate max connections: ~1000 per minicolumn * num_minicolumns
        uint32_t max_connections = num_hypercolumns * minicolumns_per_hc * 100;

        brain->columnar_connectivity = columnar_connectivity_create(max_connections);
        if (!brain->columnar_connectivity) {
            LOG_WARNING("Failed to create columnar connectivity - continuing without connectivity");
        } else {
            LOG_INFO("Created canonical microcircuit connectivity (max %u connections)", max_connections);
        }
    }

    // Step 5: Create topographic maps if sensory processing is enabled
    if (brain->config.enable_visual_topographic && brain->config.enable_visual_cortex) {
        // Create retinotopic map for V1 (log-polar mapping)
        retinotopic_params_t visual_params = {
            .foveal_radius = 2.0f,            // 2 degrees foveal region
            .cortical_magnification = 20.0f,  // 20 mm/degree at fovea
            .log_polar_a = 0.5f,              // Log-polar offset
            .aspect_ratio = 1.0f,             // Circular
            .eccentricity_half = 1.5f,        // E₂ for magnification falloff
            .angle_coverage = 2.0f * 3.14159f // Full visual field
        };
        // Default cortical dimensions: 64x64 columns
        uint32_t cortical_width = 64;
        uint32_t cortical_height = 64;

        brain->visual_topographic_map = topographic_map_create_retinotopic(
            &visual_params, cortical_width, cortical_height);
        if (!brain->visual_topographic_map) {
            LOG_WARNING("Failed to create visual topographic map");
        } else {
            LOG_INFO("Created retinotopic (log-polar) map for V1 (%ux%u)", cortical_width, cortical_height);
        }
    }

    if (brain->config.enable_auditory_topographic && brain->config.enable_audio_cortex) {
        // Create tonotopic map for A1 (logarithmic frequency mapping)
        tonotopic_params_t auditory_params = {
            .min_frequency = 20.0f,            // 20 Hz
            .max_frequency = 20000.0f,         // 20 kHz
            .octave_span = 1.0f,               // 1 octave per unit distance
            .is_logarithmic = true,            // Log frequency scale
            .q_factor = 10.0f                  // Constant Q bandwidth
        };
        // Default: 128 frequency bands (about 10 bands per octave over ~10 octaves)
        uint32_t num_frequency_bands = 128;

        brain->auditory_topographic_map = topographic_map_create_tonotopic(
            &auditory_params, num_frequency_bands);
        if (!brain->auditory_topographic_map) {
            LOG_WARNING("Failed to create auditory topographic map");
        } else {
            LOG_INFO("Created tonotopic (logarithmic) map for A1 (%u bands)", num_frequency_bands);
        }
    }

    if (brain->config.enable_somatosensory_topographic) {
        // Create somatotopic map for S1 (homunculus)
        // Default: 20 body regions (standard homunculus)
        uint32_t num_body_regions = 20;
        brain->somatosensory_topographic_map = topographic_map_create_somatotopic(num_body_regions);
        if (!brain->somatosensory_topographic_map) {
            LOG_WARNING("Failed to create somatosensory topographic map");
        } else {
            LOG_INFO("Created somatotopic (homunculus) map for S1 (%u regions)", num_body_regions);
        }
    }

    // Step 6: Initialize orientation columns if visual processing enabled
    if (brain->config.enable_orientation_columns && brain->config.enable_visual_cortex) {
        uint32_t num_orient_cols = brain->config.num_orientation_columns > 0 ?
            brain->config.num_orientation_columns : 16;  // Default: 16 orientations (0-180° in 11.25° steps)
        float spatial_frequency = 2.0f;  // Default: 2 cycles/degree
        float tuning_width = 30.0f;      // Default: 30° half-width at half-height

        // Create orientation hypercolumns (one per visual hypercolumn)
        uint32_t num_orient_hc = num_hypercolumns;  // Match hypercolumn count
        brain->orientation_hypercolumns = (orientation_hypercolumn_t**)nimcp_calloc(
            num_orient_hc, sizeof(orientation_hypercolumn_t*));

        if (brain->orientation_hypercolumns) {
            for (uint32_t i = 0; i < num_orient_hc; i++) {
                brain->orientation_hypercolumns[i] = orientation_hypercolumn_create(
                    num_orient_cols, spatial_frequency, tuning_width);
                if (!brain->orientation_hypercolumns[i]) {
                    LOG_WARNING("Failed to create orientation hypercolumn %u", i);
                }
            }
            brain->num_orientation_hypercolumns = num_orient_hc;
            LOG_INFO("Created %u orientation hypercolumns with %u columns each (sf=%.1f, tw=%.1f)",
                     num_orient_hc, num_orient_cols, spatial_frequency, tuning_width);
        }
    }

    // Step 7: Initialize feature hypercolumns if enabled
    if (brain->config.enable_feature_hypercolumns) {
        // Create feature hypercolumns for multi-dimensional feature spaces
        // Start with a basic orientation + spatial frequency hypercolumn
        uint32_t num_feat_hc = num_hypercolumns;  // One per spatial location

        brain->feature_hypercolumns = (feature_hypercolumn_t**)nimcp_calloc(
            num_feat_hc, sizeof(feature_hypercolumn_t*));

        if (brain->feature_hypercolumns) {
            for (uint32_t i = 0; i < num_feat_hc; i++) {
                // Create 2D feature space (orientation × spatial frequency)
                feature_dimension_t dims[2];
                dims[0] = feature_dimension_create(FEATURE_ORIENTATION, 0.0f, 180.0f, 16);
                feature_dimension_set_circular(&dims[0], true);
                dims[1] = feature_dimension_create(FEATURE_SPATIAL_FREQ, 0.5f, 8.0f, 8);

                brain->feature_hypercolumns[i] = feature_hypercolumn_create(dims, 2);
                if (!brain->feature_hypercolumns[i]) {
                    LOG_WARNING("Failed to create feature hypercolumn %u", i);
                }
            }
            brain->num_feature_hypercolumns = num_feat_hc;
            LOG_INFO("Created %u feature hypercolumns (orientation × spatial frequency)",
                     num_feat_hc);
        }
    }

    // Mark as enabled
    brain->enable_cortical_columns = true;
    brain->last_cortical_update_us = 0;

    LOG_INFO("Cortical columns subsystem initialized: %u hypercolumns, %s laminar, %s connectivity",
             brain->num_hypercolumns,
             brain->laminar_system ? "with" : "no",
             brain->columnar_connectivity ? "with" : "no");

    return true;
}

//=============================================================================
// Phase SC-2: Security-Fault Tolerance Integration
//=============================================================================

/**
 * @brief Initialize security subsystem with fault tolerance integration
 *
 * WHAT: Sets up the security-recovery bridge for comprehensive protection
 * WHY:  Detect security violations and automatically trigger recovery
 * HOW:  Creates security coverage, fractal security, and connects to fault tolerance
 *
 * ARCHITECTURE:
 * - Security Coverage: Memory region protection with hash verification
 * - Fractal Security: Hierarchical Merkle tree integrity checking
 * - Recovery Bridge: Routes violations to fast recovery or checkpoints
 *
 * @param brain Brain to initialize security for
 * @return true on success, false on failure
 *
 * COMPLEXITY: O(1) setup, O(n) region registration
 */
bool nimcp_brain_factory_init_security_subsystem(brain_t brain)
{
    if (!brain) {
        LOG_ERROR("Null brain in init_security_subsystem");
        return false;
    }

    // Initialize security fields to defaults
    brain->security_bridge = NULL;
    brain->enable_security_monitoring = false;
    brain->security_check_interval_ms = 0;
    brain->last_security_check_ms = 0;

    // Check if security monitoring is enabled in config
    // Default to false to avoid performance impact unless explicitly requested
    if (!brain->config.enable_security_monitoring) {
        LOG_DEBUG("Security monitoring disabled in config");
        return true;  // Success - security disabled
    }

    // Create the security recovery bridge
    nimcp_security_recovery_bridge_t* bridge = nimcp_srb_create();
    if (!bridge) {
        LOG_WARNING("Failed to create security recovery bridge - continuing without security monitoring");
        return true;  // Non-fatal - continue without security
    }

    // Create security recovery bridge configuration
    nimcp_srb_config_t srb_config = {
        .mode = NIMCP_SRB_MODE_AUTO_REPAIR,
        .enable_auto_checkpoint = true,
        .checkpoint_interval_ms = 60000,  // 1 minute
        .enable_fractal_verification = true,
        .verification_interval_ms = brain->config.security_check_interval_ms,
        .cooldown_ms = 1000,
        .max_repairs_per_minute = 60,
        .notify_brain = true,
        .log_to_audit = true
    };

    // Initialize the bridge with configuration
    if (nimcp_srb_init(bridge, &srb_config) != NIMCP_SUCCESS) {
        LOG_WARNING("Failed to initialize security recovery bridge - continuing without security monitoring");
        nimcp_srb_destroy(bridge);
        return true;  // Non-fatal
    }

    // Register this brain with the bridge
    if (nimcp_srb_register_brain(bridge, brain) != NIMCP_SUCCESS) {
        LOG_WARNING("Failed to register brain with security bridge");
        nimcp_srb_destroy(bridge);
        return true;  // Non-fatal
    }

    // Register brain's critical memory regions for protection
    uint32_t registered_regions = nimcp_srb_register_brain_regions(bridge, brain);
    LOG_DEBUG("Registered %u critical memory regions for security monitoring", registered_regions);

    // Store bridge in brain structure
    brain->security_bridge = bridge;
    brain->enable_security_monitoring = true;
    brain->security_check_interval_ms = brain->config.security_check_interval_ms;
    brain->last_security_check_ms = 0;

    LOG_INFO("Security subsystem initialized: mode=%s, fractal=%s, checkpoint=%s",
             srb_config.mode == NIMCP_SRB_MODE_AUTO_REPAIR ? "auto_repair" : "monitor",
             srb_config.enable_fractal_verification ? "enabled" : "disabled",
             srb_config.enable_auto_checkpoint ? "enabled" : "disabled");

    // === PHASE SC-4: UNIVERSAL SECURITY INTEGRATION ===
    // Initialize the global security integration framework if enabled
    // This provides entropy monitoring, trust management, and differential privacy

    // Initialize SC-4 fields to defaults
    brain->security_integration = NULL;
    brain->sec_module_id = 0;
    brain->sec_region_ids = NULL;
    brain->num_sec_regions = 0;
    brain->enable_security_integration = false;

    if (brain->config.enable_security_integration) {
        // Create security integration context
        nimcp_sec_integration_t* sec_ctx = nimcp_sec_integration_create();
        if (sec_ctx) {
            // Configure the security integration
            nimcp_sec_integration_config_t sec_config = nimcp_sec_integration_default_config();
            sec_config.trust_threshold = brain->config.security_trust_threshold > 0.0 ?
                                        brain->config.security_trust_threshold : 0.5;
            sec_config.privacy_budget = brain->config.security_privacy_budget > 0.0 ?
                                       brain->config.security_privacy_budget : 10.0;
            sec_config.enable_continuous_monitoring = false;  // Don't start monitoring thread
            sec_config.enable_self_monitoring = true;
            sec_config.enable_event_logging = true;

            if (nimcp_sec_integration_init(sec_ctx, &sec_config) == NIMCP_SUCCESS) {
                // Register brain as a CORE module
                uint32_t module_id;
                if (nimcp_sec_register_module(sec_ctx, brain->config.task_name,
                                             NIMCP_SEC_CAT_CORE, &module_id) == NIMCP_SUCCESS) {
                    brain->security_integration = sec_ctx;
                    brain->sec_module_id = module_id;
                    brain->enable_security_integration = true;

                    // === Register All Cognitive Subsystems ===
                    uint32_t subsys_id;
                    (void)subsys_id;  // Suppress unused warning if some modules not enabled

                    // COGNITIVE MODULES
                    if (brain->global_workspace) {
                        nimcp_sec_register_module(sec_ctx, "global_workspace", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->ethics) {
                        nimcp_sec_register_module(sec_ctx, "ethics_engine", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->knowledge) {
                        nimcp_sec_register_module(sec_ctx, "knowledge_system", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->introspection) {
                        nimcp_sec_register_module(sec_ctx, "introspection", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->curiosity) {
                        nimcp_sec_register_module(sec_ctx, "curiosity_engine", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->salience) {
                        nimcp_sec_register_module(sec_ctx, "salience_evaluator", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->consolidation) {
                        nimcp_sec_register_module(sec_ctx, "consolidation", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->working_memory) {
                        nimcp_sec_register_module(sec_ctx, "working_memory", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->executive) {
                        nimcp_sec_register_module(sec_ctx, "executive_controller", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->theory_of_mind) {
                        nimcp_sec_register_module(sec_ctx, "theory_of_mind", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->emotional_system) {
                        nimcp_sec_register_module(sec_ctx, "emotional_system", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->mirror_neurons) {
                        nimcp_sec_register_module(sec_ctx, "mirror_neurons", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->mental_health_monitor) {
                        nimcp_sec_register_module(sec_ctx, "mental_health_monitor", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->autobio) {
                        nimcp_sec_register_module(sec_ctx, "autobiographical_memory", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->self_model) {
                        nimcp_sec_register_module(sec_ctx, "self_model", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->meta_learner) {
                        nimcp_sec_register_module(sec_ctx, "meta_learner", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->predictive_network) {
                        nimcp_sec_register_module(sec_ctx, "predictive_network", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->symbolic_logic) {
                        nimcp_sec_register_module(sec_ctx, "symbolic_logic", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->epistemic) {
                        nimcp_sec_register_module(sec_ctx, "epistemic_filter", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    if (brain->personality) {
                        nimcp_sec_register_module(sec_ctx, "personality", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    }
                    // empathy_network is not a pointer - always registered
                    nimcp_sec_register_module(sec_ctx, "empathy_network", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);
                    // sleep_system is not a pointer - always registered
                    nimcp_sec_register_module(sec_ctx, "sleep_system", NIMCP_SEC_CAT_COGNITIVE, &subsys_id);

                    // GLIAL MODULES
                    if (brain->glial) {
                        nimcp_sec_register_module(sec_ctx, "glial_integration", NIMCP_SEC_CAT_GLIAL, &subsys_id);
                    }

                    // PLASTICITY MODULES
                    // neuromodulator_system is not a pointer - always registered
                    nimcp_sec_register_module(sec_ctx, "neuromodulators", NIMCP_SEC_CAT_PLASTICITY, &subsys_id);
                    // multihead_attention is not a pointer - always registered
                    nimcp_sec_register_module(sec_ctx, "attention", NIMCP_SEC_CAT_PLASTICITY, &subsys_id);

                    // CORE MODULES
                    if (brain->network) {
                        nimcp_sec_register_module(sec_ctx, "neural_network", NIMCP_SEC_CAT_CORE, &subsys_id);
                    }
                    if (brain->brain_regions) {
                        nimcp_sec_register_module(sec_ctx, "brain_regions", NIMCP_SEC_CAT_CORE, &subsys_id);
                    }
                    if (brain->oscillations) {
                        nimcp_sec_register_module(sec_ctx, "oscillations", NIMCP_SEC_CAT_CORE, &subsys_id);
                    }
                    // multimodal is not a pointer - always registered
                    nimcp_sec_register_module(sec_ctx, "multimodal", NIMCP_SEC_CAT_CORE, &subsys_id);
                    if (brain->cortical_column_pool) {
                        nimcp_sec_register_module(sec_ctx, "cortical_columns", NIMCP_SEC_CAT_CORE, &subsys_id);
                    }

                    // NETWORKING MODULES
                    // distributed is not a pointer - check if non-NULL context
                    if (brain->distributed) {
                        nimcp_sec_register_module(sec_ctx, "distributed_cognition", NIMCP_SEC_CAT_NETWORKING, &subsys_id);
                    }

                    // MIDDLEWARE MODULES
                    if (brain->middleware_controller) {
                        nimcp_sec_register_module(sec_ctx, "middleware_controller", NIMCP_SEC_CAT_MIDDLEWARE, &subsys_id);
                    }

                    // I/O & PERCEPTION MODULES
                    if (brain->visual_cortex) {
                        nimcp_sec_register_module(sec_ctx, "visual_cortex", NIMCP_SEC_CAT_IO, &subsys_id);
                    }
                    if (brain->audio_cortex) {
                        nimcp_sec_register_module(sec_ctx, "audio_cortex", NIMCP_SEC_CAT_IO, &subsys_id);
                    }
                    if (brain->speech_cortex) {
                        nimcp_sec_register_module(sec_ctx, "speech_cortex", NIMCP_SEC_CAT_IO, &subsys_id);
                    }
                    // nlp_network is not a pointer - always registered
                    nimcp_sec_register_module(sec_ctx, "nlp", NIMCP_SEC_CAT_IO, &subsys_id);

                    LOG_INFO("Security integration (SC-4) initialized: module_id=%u, trust_threshold=%.2f, privacy_budget=%.1f",
                            module_id, sec_config.trust_threshold, sec_config.privacy_budget);
                } else {
                    LOG_WARNING("Failed to register brain with security integration");
                    nimcp_sec_integration_destroy(sec_ctx);
                }
            } else {
                LOG_WARNING("Failed to initialize security integration context");
                nimcp_sec_integration_destroy(sec_ctx);
            }
        } else {
            LOG_WARNING("Failed to create security integration context - continuing without SC-4");
        }
    }

    // === PHASE IS-1: BLOOD-BRAIN BARRIER (BBB) PERIMETER DEFENSE ===
    // Initialize BBB fields to defaults
    brain->bbb_system = NULL;
    brain->bbb_memory_region_id = 0;
    brain->bbb_subject_id = 0;
    brain->bbb_enabled = false;

    if (brain->config.enable_bbb_protection) {
        // Get or create the global BBB system
        bbb_system_t bbb = get_global_bbb_system();
        if (bbb) {
            // Register this brain's memory region with the BBB
            // The brain struct itself is the primary protected region
            uint32_t region_id = bbb_register_memory_region(bbb, brain, sizeof(*brain), false);
            if (region_id > 0) {
                brain->bbb_system = bbb;  // Store reference for cleanup
                brain->bbb_memory_region_id = region_id;
                brain->bbb_enabled = true;

                LOG_INFO("BBB protection enabled for brain '%s': region_id=%u",
                        brain->config.task_name, region_id);
            } else {
                LOG_WARNING("Failed to register brain memory region with BBB");
                nimcp_bbb_release_global_system();  // Release our reference on failure
            }
        } else {
            LOG_WARNING("Failed to get global BBB system - continuing without BBB protection");
        }
    } else {
        LOG_DEBUG("BBB protection disabled in config");
    }

    return true;
}

/**
 * @brief Generate or configure personality from brain config
 *
 * WHAT: Create personality profile based on configuration
 * WHY:  Each brain needs unique personality for individuality
 * HOW:  Random generation or explicit specification
 *
 * @param config Brain configuration with personality settings
 * @return Allocated personality profile or NULL on error
 *
 * COMPLEXITY: O(1)
 */
