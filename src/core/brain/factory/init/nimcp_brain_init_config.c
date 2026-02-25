//=============================================================================
// nimcp_brain_init_config.c - Brain Configuration Functions
//=============================================================================
/**
 * @file nimcp_brain_init_config.c
 * @brief Brain configuration and statistics initialization
 *
 * WHAT: Configuration building and parameter setup for brain initialization
 * WHY:  Separates configuration logic from brain creation
 * HOW:  Provides builder functions for network config and brain stats
 *
 * EXTRACTED FROM: nimcp_brain_init.c
 * DATE: 2025-12-08
 *
 * @version 2.7.0
 * @author NIMCP Development Team
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_config.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>

#define LOG_MODULE "BRAIN_INIT_CONFIG"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "utils/bridge/nimcp_bridge_boilerplate.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

BRIDGE_BOILERPLATE_MESH_ONLY(brain_init_config, MESH_ADAPTER_CATEGORY_SYSTEM)


// P3-56 FIX: Variadic set_error macro for formatted error messages
#ifndef set_error
#define set_error(fmt, ...) LOG_ERROR(LOG_MODULE, fmt, ##__VA_ARGS__)
#endif

//=============================================================================
// Neuron Count and Sparsity Functions
//=============================================================================

uint32_t nimcp_brain_factory_get_neuron_count(brain_size_t size)
{
    switch (size) {
        case BRAIN_SIZE_MICRO:
            return 25;    // Ultra-lightweight for unit tests and swarm drones
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

float nimcp_brain_factory_get_default_sparsity(brain_size_t size)
{
    switch (size) {
        case BRAIN_SIZE_MICRO:
            return 0.60F;   // Lower sparsity for micro brains (more dense connections)
        case BRAIN_SIZE_TINY:
            return 0.70F;
        case BRAIN_SIZE_SMALL:
            return 0.80F;
        case BRAIN_SIZE_MEDIUM:
            return 0.85F;
        case BRAIN_SIZE_LARGE:
            return 0.90F;
        default:
            return 0.80F;
    }
}

//=============================================================================
// Configuration Builders
//=============================================================================

adaptive_spike_params_t nimcp_brain_factory_build_spike_params(float sparsity_target)
{
    adaptive_spike_params_t params = {0};
    params.k_factor = 0.5F;
    params.sparsity_target = sparsity_target;
    params.encoding = SPIKE_ENCODING_INTEGER;
    params.enable_soft_reset = true;
    params.enable_adaptation = true;
    params.adaptation_window = 100;
    params.min_threshold = 0.0001F;  // Very low to allow tiny outputs from untrained networks
    params.max_threshold = 10.0F;
    return params;
}

network_config_t nimcp_brain_factory_build_base_network_config(uint32_t num_inputs, uint32_t num_outputs,
                                                  uint32_t num_neurons,
                                                  ode_integration_method_t integration_method)
{
    network_config_t config = {0};

    // Validation: Check for zero parameters
    if (num_inputs == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "nimcp_brain_factory_build_base_network_config: num_inputs cannot be zero");
        return config;  // Return with layer_sizes = NULL to signal error
    }
    if (num_outputs == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "nimcp_brain_factory_build_base_network_config: num_outputs cannot be zero");
        return config;  // Return with layer_sizes = NULL to signal error
    }
    if (num_neurons == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "nimcp_brain_factory_build_base_network_config: num_neurons cannot be zero");
        return config;  // Return with layer_sizes = NULL to signal error
    }

    config.input_size = num_inputs;
    config.output_size = num_outputs;
    config.num_neurons = num_neurons;
    config.num_layers = 3;
    config.integration_method = integration_method;

    config.layer_sizes = nimcp_calloc(3, sizeof(uint32_t));
    if (!config.layer_sizes) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "nimcp_brain_factory_build_base_network_config: failed to allocate layer_sizes");
        return config;
    }

    config.layer_sizes[0] = num_inputs;
    config.layer_sizes[1] = num_neurons;
    config.layer_sizes[2] = num_outputs;

    config.enable_stdp = true;
    config.enable_hebbian = true;
    config.enable_oja = true;
    config.enable_homeostasis = true;

    // Enable BCM for homeostatic plasticity (prevents weight saturation via sliding threshold)
    config.enable_bcm = true;
    config.enable_eligibility = true;

    // BUGFIX: Set weight bounds to allow random initialization to work
    // Without these, fmaxf(0, fminf(0, weight)) clamps all weights to 0!
    config.min_weight = -1.0F;
    config.max_weight = 1.0F;

    return config;
}

adaptive_network_config_t nimcp_brain_factory_build_network_config(uint32_t num_inputs, uint32_t num_outputs,
                                                      uint32_t num_neurons, float sparsity_target,
                                                      ode_integration_method_t integration_method)
{
    adaptive_network_config_t config = {0};

    config.base_config = nimcp_brain_factory_build_base_network_config(num_inputs, num_outputs, num_neurons, integration_method);
    config.spike_params = nimcp_brain_factory_build_spike_params(sparsity_target);

    config.enable_sparsity = false;  // Disabled for regression tests
    config.pruning_threshold = 0.01F;
    config.update_frequency = 100;

    return config;
}

void nimcp_brain_factory_init_brain_config(brain_config_t* config, const char* task_name, brain_size_t size,
                              brain_task_t task, uint32_t num_inputs, uint32_t num_outputs,
                              task_strategy_t* strategy)
{
    // Guard: NULL check
    if (!config || !strategy) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "nimcp_brain_factory_init_brain_config: config or strategy is NULL");
        return;
    }

    // Save minimal_mode flag (may be pre-set by caller)
    bool minimal = config->minimal_mode;

    config->size = size;
    config->task = task;
    config->num_inputs = num_inputs;
    config->num_outputs = num_outputs;
    config->learning_rate = strategy->get_learning_rate();
    config->sparsity_target = nimcp_brain_factory_get_default_sparsity(size);
    config->enable_explanations = !minimal;  // Skip in minimal mode
    strncpy(config->task_name, task_name, sizeof(config->task_name) - 1);
    config->minimal_mode = minimal;  // Restore after strncpy may have zeroed it

    // Part A: Differential Equations - ODE Integration Method (A1.x)
    config->neuron_integration = ODE_EULER;

    // Phase 10.2: Working Memory defaults (Miller's 7±2)
    config->enable_working_memory = !minimal;
    config->working_memory_capacity = 7;
    config->working_memory_decay_tau_ms = 1000.0F;

    // Phase 10.6: Theory of Mind defaults (social cognition, empathy)
    config->enable_theory_of_mind = !minimal;
    config->enable_empathy_responses = !minimal;
    config->enable_false_belief_tracking = !minimal;

    // Phase 10.11: Mirror Neurons defaults (observation-based learning)
    config->enable_mirror_neurons = !minimal;
    config->mirror_neuron_count = minimal ? 0 : 1000;
    config->mirror_max_actions = 100;
    config->mirror_max_agents = 10;
    config->mirror_learning_rate = 0.01F;
    config->mirror_match_threshold = 0.7F;

    // Global Workspace Architecture defaults (Global Workspace Theory - Baars 1988)
    config->enable_global_workspace = !minimal;
    config->workspace_capacity_dim = 0;  // 0 = auto-derive from brain's input dimension
    config->workspace_ignition_threshold = 0.6F;
    config->workspace_refractory_ms = 50;
    config->workspace_enable_history = !minimal;
    config->workspace_history_depth = 10;

    // Phase 11 Enhancement C1.1: Quantum Annealing defaults
    config->enable_quantum_annealing = false;
    config->annealing_temperature_init = 10.0F;
    config->annealing_temperature_final = 0.1F;
    config->annealing_steps = 1000;
    config->quantum_annealing_frequency = 100;

    // Phase 12: Personality and Identity defaults
    config->use_random_personality = true;
    config->personality_seed = 0;
    config->explicit_openness = 0.5F;
    config->explicit_conscientiousness = 0.5F;
    config->explicit_extraversion = 0.5F;
    config->explicit_agreeableness = 0.5F;
    config->explicit_neuroticism = 0.5F;
    config->explicit_gender = GENDER_FEMALE;
    config->explicit_sexuality = SEXUALITY_HETEROSEXUAL;
    config->personality_trait_mean = 0.5F;
    config->personality_trait_stddev = 0.15F;
    config->female_probability = 1.0F;
    config->male_probability = 0.0F;
    config->non_binary_probability = 0.0F;

    // Cognitive pipeline flags (enable stages in brain_decide)
    // Enables 4 additional stages: natural explanations, curiosity,
    // executive control, and epistemic filtering. Other flags
    // (predictive_processing, ethics, emotional_tagging, mental_health,
    // sleep_wake) have module re-registration issues in multi-brain tests.
    config->enable_natural_explanations = !minimal;
    config->enable_curiosity = !minimal;
    config->enable_executive_control = !minimal;
    config->enable_epistemic_filter = !minimal;

    // Phase 5/6: Biological Realism defaults
    config->enable_glial = !minimal;
    config->enable_oscillations = false;
    config->enable_myelin_sheath = !minimal;

    // Lazy Initialization defaults (performance optimization)
    // Check if lazy_init_mode was pre-set by caller (like minimal mode sets it)
    bool lazy = config->lazy_init_mode || minimal;
    config->lazy_init_mode = lazy;  // Propagate lazy mode
    config->lazy_dendrite_init = lazy;
    config->lazy_axon_init = lazy;
    config->lazy_visual_init = lazy;
    config->lazy_audio_init = lazy;
    config->lazy_speech_init = lazy;
    config->lazy_working_memory_init = lazy;
    config->lazy_theory_of_mind_init = lazy;
    config->lazy_global_workspace_init = lazy;
    config->lazy_ethics_init = lazy;
    config->lazy_mirror_neurons_init = lazy;
    config->lazy_executive_init = lazy;
    config->lazy_consolidation_init = lazy;
    config->lazy_meta_learning_init = lazy;
    config->lazy_neuromod_init = lazy;
    config->lazy_glial_init = lazy;
    config->lazy_cortical_init = lazy;
    config->lazy_topographic_init = lazy;
    config->lazy_pr_memory_init = lazy;
    config->enable_dendrites = !minimal;
    config->enable_axons = !minimal;
    config->enable_pr_memory = !minimal;  // PR memory enabled by default (except minimal mode)

    // === WORLD MODEL CONFIGURATION ===
    // World model enables generative simulation for counterfactual reasoning,
    // policy evaluation, and mental imagery. Based on DreamerV3 and JEPA.
    config->enable_world_model = false;            // Disabled by default (opt-in feature)
    config->lazy_world_model_init = lazy;          // Lazy init follows global lazy setting
    config->enable_omni_world_model = true;        // Omni WM on if world model enabled
    config->enable_multimodal_world_model = true;  // Multimodal WM on if world model enabled

    // Omni World Model defaults (DreamerV3-inspired)
    config->omni_wm_state_dim = 64;                // State dimensionality
    config->omni_wm_action_dim = 32;               // Action dimensionality
    config->omni_wm_obs_dim = 64;                  // Observation dimensionality
    config->omni_wm_latent_dim = 64;               // Latent space dimension
    config->omni_wm_rssm_h_dim = 128;              // RSSM deterministic state
    config->omni_wm_rssm_z_dim = 32;               // RSSM stochastic state
    config->omni_wm_learning_rate = 0.0003F;       // Learning rate
    config->omni_wm_enable_dreaming = true;        // Offline simulation enabled
    config->omni_wm_dream_horizon = 15;            // Dream episode length

    // Multimodal World Model defaults
    config->mm_wm_latent_dim = 256;                // Latent dimension for cross-modal fusion
    config->mm_wm_max_entities = 128;              // Max tracked entities
    config->mm_wm_max_prediction_steps = 50;       // Max prediction horizon
    config->mm_wm_learning_rate = 0.001F;          // Learning rate
    config->mm_wm_enable_bio_async = !minimal;     // Bio-async integration

    // World Model Integration Flags
    config->world_model_connect_active_inference = true;  // Connect to active inference
    config->world_model_connect_imagination = true;       // Connect to imagination engine
    config->world_model_connect_hippocampus = true;       // Connect to hippocampus
    config->world_model_connect_predictive = true;        // Connect to predictive processing

    // === CORTICAL COLUMNS CONFIGURATION ===
    // Disable cortical columns for TINY/SMALL brains (huge memory savings: ~48MB)
    // Only enable for MEDIUM and larger brains that need hierarchical feature processing
    bool large_enough_for_cortical = (config->size >= BRAIN_SIZE_MEDIUM) && !minimal;
    config->enable_cortical_columns = large_enough_for_cortical;
    config->num_hypercolumns = large_enough_for_cortical ? 10 : 0;
    config->minicolumns_per_hypercolumn = large_enough_for_cortical ? 100 : 0;
    config->neurons_per_minicolumn = large_enough_for_cortical ? 80 : 0;
    config->enable_laminar_structure = large_enough_for_cortical;
    config->enable_columnar_connectivity = large_enough_for_cortical;

    // Topographic maps only for brains with sensory processing enabled
    config->enable_visual_topographic = large_enough_for_cortical && config->enable_visual_cortex;
    config->enable_auditory_topographic = large_enough_for_cortical && config->enable_audio_cortex;
    config->enable_somatosensory_topographic = large_enough_for_cortical;
    config->enable_orientation_columns = large_enough_for_cortical && config->enable_visual_cortex;
    config->num_orientation_columns = large_enough_for_cortical ? 16 : 0;
    config->enable_feature_hypercolumns = large_enough_for_cortical;

    // Phase C2.1: Quantum Walk defaults
    config->enable_quantum_walk_diffusion = false;
    config->quantum_walk_steps = 50;
    config->quantum_classical_mixing = 0.2F;
    config->quantum_coin_type = 0;
    config->quantum_decoherence_rate = 0.05F;

    // Phase 10.12: Second Messenger Cascades (biological neuromodulation)
    // Second messengers are fundamental to how neuromodulators actually work in the brain.
    // Without them, dopamine/serotonin/etc. effects are simplified approximations.
    config->enable_second_messengers = !minimal;

    // Phase TM-3: Brain-Training Integration defaults
    config->enable_training_integration = !minimal;
    config->training_register_security = !minimal;
    config->training_default_lr = 0.0F;

    // Phase TM-4/5/6: Advanced Training Pipeline defaults
    config->enable_lr_scheduler = false;
    config->enable_regularization = false;
    config->regularization_l1_lambda = 0.0F;
    config->regularization_l2_lambda = 0.0001F;
    config->dropout_rate = 0.0F;
    config->enable_gradient_management = false;
    config->enable_gradient_health_check = true;
    config->gradient_accumulation_steps = 1;
    config->gradient_clip_value = 0.0F;
    config->gradient_clip_norm = 0.0F;
}

void nimcp_brain_factory_init_brain_stats(brain_stats_t* stats, const char* task_name, brain_size_t size,
                             uint32_t num_inputs, float learning_rate)
{
    uint32_t num_neurons = nimcp_brain_factory_get_neuron_count(size);

    stats->size = size;
    stats->num_neurons = num_neurons;

    // P2-59 FIX: Check for integer overflow in num_neurons * num_inputs
    // Use 64-bit multiplication to detect overflow
    uint64_t synapse_count = (uint64_t)num_neurons * (uint64_t)num_inputs;
    if (synapse_count > UINT32_MAX) {
        LOG_WARN(LOG_MODULE, "Synapse count overflow: %u * %u exceeds uint32_t, clamping",
                 num_neurons, num_inputs);
        stats->num_synapses = UINT32_MAX;
    } else {
        stats->num_synapses = (uint32_t)synapse_count;
    }
    stats->num_active_synapses = stats->num_synapses;
    stats->current_learning_rate = learning_rate;
    stats->quantum_annealing_runs = 0;

    // P2-58 FIX: Guard against task_name==NULL before strncpy
    if (task_name) {
        strncpy(stats->task_name, task_name, sizeof(stats->task_name) - 1);
        stats->task_name[sizeof(stats->task_name) - 1] = '\0';
    } else {
        stats->task_name[0] = '\0';
    }
}
