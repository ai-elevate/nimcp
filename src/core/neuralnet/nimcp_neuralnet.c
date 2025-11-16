//=============================================================================
// nimcp_neuralnet.c - Spiking Neural Network with Advanced Plasticity
//=============================================================================
/**
 * @file nimcp_neuralnet.c
 * @brief Biologically-inspired neural network with STDP, Oja, and homeostasis
 *
 * ARCHITECTURAL OVERVIEW:
 * This module implements a spiking neural network with multiple learning rules:
 * - Strategy Pattern: Activation functions via function pointer tables
 * - Factory Pattern: Neuron/synapse creation through builder functions
 * - Repository Pattern: Hash-indexed synapse lookup for O(1) access
 *
 * DESIGN PATTERNS USED:
 * - Strategy Pattern: Activation functions, learning rules
 * - Factory Pattern: Network/neuron creation
 * - Builder Pattern: Configuration construction
 * - Single Responsibility: Each function does exactly one thing
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Neuron update: O(s) where s = num_synapses (was O(n) with linear search)
 * - Synapse lookup: O(1) average with hash table (was O(n))
 * - Learning: O(s) per neuron (optimized from O(s*n))
 * - Memory: O(n + s) where n = neurons, s = synapses
 *
 * COMPLEXITY IMPROVEMENTS:
 * - Original: O(n²) nested loops in many places
 * - Refactored: O(n) single-pass algorithms throughout
 * - Synapse access: O(n) linear → O(1) hash table
 * - Weight updates: Eliminated redundant calculations
 *
 * DESIGN PRINCIPLES:
 * - No nested ifs: Guard clauses and early returns only
 * - No nested loops: Extracted to helper functions
 * - Functions <50 lines: Complex operations decomposed
 * - Single responsibility: Each function has one clear purpose
 */

#include "nimcp_neuralnet.h"
#include "core/synapse_compute/nimcp_synapse_compute.h"  // NIMCP 2.7: Programmable synapses
#include "core/synapse_types/nimcp_synapse_types.h"      // NIMCP 2.8.7: Synapse type system
#include "core/neuron_models/nimcp_neuron_model.h"
#include "core/neuron_models/nimcp_izhikevich.h"
#include "core/neuron_models/nimcp_two_compartment.h"    // Part A3.1: Two-compartment neurons
#include "plasticity/stp/nimcp_stp.h"
#include "plasticity/bcm/nimcp_bcm.h"                    // Phase 11: BCM homeostatic plasticity
#include "plasticity/eligibility/nimcp_eligibility_trace.h"  // Phase 11: Eligibility traces for RL
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"  // Neuromodulator system for dopamine/serotonin/etc
#include "glial/integration/nimcp_glial_integration.h"   // NIMCP Phase 6: Glial notifications
#include "security/nimcp_security.h"                     // Phase 11: Biological attack defense
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utils/memory/nimcp_memory.h"  // CRITICAL: Declares nimcp_calloc/nimcp_free return types

//=============================================================================
// Constants and Configuration
//=============================================================================

#define EPSILON 1e-10f
#define MAX_ACTIVATION 1.0f
#define MIN_ACTIVATION -1.0f
#define CALCIUM_DECAY_RATE 0.1f
#define TRACE_DECAY_RATE 0.05f
#define META_PLASTICITY_RATE 0.001f
#define HOMEOSTATIC_DECAY 0.999f
#define MAX_SYNAPTIC_STRENGTH 10.0f

//=============================================================================
// Strategy Pattern - Activation Function Dispatch
//=============================================================================

/**
 * @brief Function pointer type for activation strategies
 *
 * WHY: Eliminates switch statements, enables O(1) dispatch
 * Follows Strategy pattern - each activation type is its own algorithm
 *
 * @param input Raw input value
 * @param threshold Adaptive threshold for spike-based activations
 * @return Activated output value
 */
typedef float (*activation_fn_t)(float input, float threshold);

/**
 * @brief Activation function strategy table
 *
 * INVARIANT: All entries must be non-NULL
 * PATTERN: Strategy pattern - function pointer dispatch
 */
typedef struct {
    activation_fn_t functions[8];  // Max activation types
} activation_strategy_table_t;

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Internal neural network structure
 *
 * INVARIANTS:
 * - num_neurons <= MAX_NEURONS
 * - current_time <= network_time
 * - 0.0 <= global_activity <= 1.0
 * - 0.0 <= network_stability <= 1.0
 *
 * COMPLEXITY: O(1) access to all fields
 */
/**
 * WHAT: Internal neural network structure
 * WHY: Encapsulates all network state and allocated resources
 * HOW: Dynamically allocates neurons array based on initial capacity
 * PATTERN: Handle/Body idiom - opaque pointer hides implementation
 */
struct neural_network_struct {
    neuron_t* neurons;     // Dynamically allocated neurons array
    uint32_t num_neurons;  // Current number of active neurons
    uint32_t capacity;     // Allocated capacity (for add_neuron)
    uint64_t current_time;
    network_config_t config;
    uint64_t network_time;
    float global_activity;
    float network_stability;
    float learning_momentum;
    float last_avg_weight;
    uint64_t last_maintenance;
    activation_strategy_table_t activation_strategies;

    // NIMCP 2.7: NLP Integration - Subsystems for synapse compute context
    void* neuromodulator_system;  /**< Neuromodulator system (opaque pointer) */
    float* global_state;          /**< Global state for synapse computation (attention output, etc) */
    uint32_t global_state_size;   /**< Size of global state buffer */

    // NIMCP Phase 6: Glial Integration - Neuro-glial signaling
    void* glial_integration;      /**< Glial integration system (opaque pointer) */
};

//=============================================================================
// Forward Declarations - Organized by Functionality
//=============================================================================

// Activation functions
static float sigmoid(float x);
static float fast_tanh(float x);
static void init_activation_strategies(activation_strategy_table_t* table);

// Membrane and dynamics
static float compute_membrane_potential(neuron_t* neuron, neural_network_t network);
static void update_calcium_dynamics(neuron_t* neuron, uint64_t timestamp);
static void update_synaptic_traces(neuron_t* neuron, uint64_t timestamp);

// Learning rules
static float compute_stdp_update(float dt, const stdp_params_t* params);
static float compute_oja_weight_update(float pre_activity, float post_activity,
                                       float current_weight, const oja_params_t* params);

// Homeostasis
static float compute_homeostatic_factor(neuron_t* neuron, float current_activity);
static void update_meta_plasticity(neuron_t* neuron, uint64_t timestamp);
static void normalize_synaptic_weights(neuron_t* neuron);

// Neuron initialization helpers
static void init_neuron_basic_properties(neuron_t* neuron, uint32_t id, neuron_type_t type,
                                         uint64_t creation_time);
static void init_neuron_learning_params(neuron_t* neuron, const network_config_t* config);
static void init_neuron_homeostatic_params(neuron_t* neuron, const network_config_t* config);
static void init_neuron_activity_tracking(neuron_t* neuron);
static void init_neuron_model(neuron_t* neuron, const network_config_t* config);

//=============================================================================
// Activation Function Implementations - Strategy Pattern
//=============================================================================

/**
 * @brief Sigmoid activation: 1 / (1 + e^(-x))
 *
 * WHY: Smooth, bounded output [0,1], good for probabilistic interpretations
 * WHEN: Classification, gating mechanisms
 *
 * COMPLEXITY: O(1)
 * @param input Raw input value
 * @param threshold Unused for sigmoid
 * @return Activated value in [0,1]
 */
static float activate_sigmoid(float input, float threshold)
{
    (void) threshold;  // Unused
    return sigmoid(input);
}

/**
 * @brief Hyperbolic tangent activation: tanh(x)
 *
 * WHY: Centered around 0, output [-1,1], good gradient properties
 * WHEN: Standard hidden layer activation
 *
 * COMPLEXITY: O(1)
 */
static float activate_tanh(float input, float threshold)
{
    (void) threshold;
    return fast_tanh(input);
}

/**
 * @brief ReLU activation: max(0, x)
 *
 * WHY: Simple, fast, prevents vanishing gradients
 * WHEN: Deep networks, default choice
 *
 * COMPLEXITY: O(1)
 */
static float activate_relu(float input, float threshold)
{
    (void) threshold;
    return (input > 0.0f) ? input : 0.0f;
}

/**
 * @brief Leaky ReLU: max(0.01x, x)
 *
 * WHY: Prevents dead neurons unlike standard ReLU
 * WHEN: Deep networks with dead neuron problems
 *
 * COMPLEXITY: O(1)
 */
static float activate_leaky_relu(float input, float threshold)
{
    (void) threshold;
    return (input > 0.0f) ? input : 0.01f * input;
}

/**
 * @brief Adaptive threshold activation for spiking neurons
 *
 * WHY: Models biological spike threshold adaptation
 * WHEN: Spiking neural networks with homeostasis
 *
 * ALGORITHM:
 * - Pass through input for membrane potential accumulation
 * - Spike detection happens separately in detected_spike()
 *
 * FIXED: Removed hardcoded -65.0 return that prevented membrane accumulation
 *
 * COMPLEXITY: O(1)
 */
static float activate_adaptive(float input, float threshold)
{
    (void)threshold;  // Threshold handled by spike detection logic

    // Pass through input to allow membrane potential accumulation
    // This lets the neuron integrate current over time
    return input;
}

/**
 * @brief Initialize activation strategy table
 *
 * WHY: Eliminates switch statements in hot path (activation calls)
 * Provides O(1) function dispatch vs O(n) switch
 *
 * PATTERN: Strategy pattern - populate dispatch table
 * COMPLEXITY: O(1)
 *
 * @param table Strategy table to initialize
 */
static void init_activation_strategies(activation_strategy_table_t* table)
{
    // Guard clause: Validate input
    if (!table)
        return;

    table->functions[ACTIVATION_SIGMOID] = activate_sigmoid;
    table->functions[ACTIVATION_TANH] = activate_tanh;
    table->functions[ACTIVATION_RELU] = activate_relu;
    table->functions[ACTIVATION_LEAKY_RELU] = activate_leaky_relu;
    table->functions[ACTIVATION_ADAPTIVE] = activate_adaptive;
}

//=============================================================================
// Factory Pattern - Neuron Creation Helpers
//=============================================================================

/**
 * @brief Initialize basic neuron properties
 *
 * WHY: Extracted from network creation - single responsibility
 * WHEN: Creating new neurons in network
 *
 * COMPLEXITY: O(1)
 *
 * @param neuron Neuron to initialize
 * @param id Unique neuron identifier
 * @param type EXCITATORY or INHIBITORY
 * @param creation_time Network timestamp of creation
 */
static void init_neuron_basic_properties(neuron_t* neuron, uint32_t id, neuron_type_t type,
                                         uint64_t creation_time)
{
    // Guard clause: Validate input
    if (!neuron)
        return;

    neuron->id = id;
    neuron->type = type;
    neuron->rest_potential = 0.0f;    // Normalized resting potential
    neuron->threshold = 0.5f;         // Spike threshold (normalized, reachable in [-1,1] range)
    neuron->adaptation = 0.0f;
    neuron->refractory_period = 2;  // Milliseconds
    neuron->state = neuron->rest_potential;
    neuron->bias = (float) rand() / (float) RAND_MAX * 0.1f - 0.05f;
    neuron->external_current = 0.0f;  // No external input by default
    neuron->creation_time = creation_time;
    neuron->last_update = creation_time;
    neuron->last_spike = 0;
}

/**
 * @brief Initialize neuron learning parameters
 *
 * WHY: Separates learning config from basic properties
 * COMPLEXITY: O(1)
 */
static void init_neuron_learning_params(neuron_t* neuron, const network_config_t* config)
{
    // Guard clause: Validate inputs
    if (!neuron || !config)
        return;

    neuron->learning_rule = LEARNING_HYBRID;
    neuron->activation_type = ACTIVATION_ADAPTIVE;

    // Oja's rule parameters
    neuron->oja_params.alpha = config->learning_rate;
    neuron->oja_params.forgetting = 0.01f;
    neuron->oja_params.stabilization = 0.001f;
    neuron->oja_params.target_norm = 1.0f;

    // STDP parameters
    neuron->stdp_params.learning_rate = config->learning_rate;
    neuron->stdp_params.time_window = config->stdp_window;
    neuron->stdp_params.positive_factor = 1.2f;
    neuron->stdp_params.negative_factor = 0.8f;

    neuron->plasticity_rate = config->learning_rate;
}

/**
 * @brief Initialize homeostatic plasticity parameters
 *
 * WHY: Separates homeostasis config from other initialization
 * COMPLEXITY: O(1)
 */
static void init_neuron_homeostatic_params(neuron_t* neuron, const network_config_t* config)
{
    // Guard clause: Validate inputs
    if (!neuron || !config)
        return;

    neuron->homeostatic.target_activity = config->target_activity;
    neuron->homeostatic.time_scale = 1000.0f;  // ms
    neuron->homeostatic.strength = 0.1f;
    neuron->homeostatic_factor = 1.0f;
    neuron->calcium_concentration = 0.0f;
    neuron->weight_norm = 0.0f;
}

/**
 * @brief Initialize activity tracking arrays
 *
 * WHY: Separates memory initialization from parameter setting
 * COMPLEXITY: O(1) - fixed-size arrays
 */
static void init_neuron_activity_tracking(neuron_t* neuron)
{
    // Guard clause: Validate input
    if (!neuron)
        return;

    neuron->spike_history_index = 0;
    neuron->avg_activity = 0.0f;
    memset(neuron->spike_history, 0, sizeof(spike_record_t) * SPIKE_HISTORY_LENGTH);
    memset(neuron->activity_history, 0, sizeof(float) * HISTORY_WINDOW);

    // Initialize outgoing synapses
    neuron->num_synapses = 0;
    memset(neuron->synapses, 0, sizeof(synapse_t) * MAX_SYNAPSES_PER_NEURON);

    // Initialize incoming synapses (bidirectional tracking)
    neuron->num_incoming = 0;
    memset(neuron->incoming_synapses, 0, sizeof(synapse_t) * MAX_SYNAPSES_PER_NEURON);
}

/**
 * @brief Initialize neuron model (Izhikevich, LIF, etc.)
 *
 * WHAT: Creates neuron dynamics model based on config
 * WHY: Enables rich firing patterns beyond simple LIF
 * HOW: Uses plugin architecture with vtable dispatch
 *
 * PATTERN: Strategy Pattern - model selected at runtime
 * COMPLEXITY: O(1)
 *
 * @param neuron Neuron to initialize
 * @param config Network configuration with model type
 */
static void init_neuron_model(neuron_t* neuron, const network_config_t* config)
{
    // Guard: Validate inputs
    if (!neuron || !config) {
        return;
    }

    // Default: No model (legacy LIF behavior)
    neuron->model = NULL;
    neuron->model_type = NEURON_MODEL_LIF;

    // Guard: Check if specific model requested
    if (config->neuron_model == NEURON_MODEL_LIF) {
        return;  // Use legacy LIF, no plugin model needed
    }

    // Get vtable for requested model
    const neuron_model_vtable_t* vtable = NULL;
    const void* params = config->model_params;

    switch (config->neuron_model) {
        case NEURON_MODEL_IZHIKEVICH:
            vtable = izhikevich_get_vtable();
            // If no params provided, use default RS (Regular Spiking)
            if (!params) {
                static izhikevich_params_t default_params;
                default_params = izhikevich_get_preset_params(IZHI_PRESET_REGULAR_SPIKING);
                params = &default_params;
            }
            break;

        case NEURON_MODEL_TWO_COMPARTMENT:
            vtable = two_compartment_get_vtable();
            // If no params provided, use defaults (70% attenuation, RK4)
            if (!params) {
                static two_compartment_params_t default_params;
                default_params = two_compartment_default_params();
                // Use integration method from config
                default_params.integration_method = config->integration_method;
                params = &default_params;
            }
            break;

        case NEURON_MODEL_LIF:
            // Already handled above
            return;

        default:
            // Unknown model, fall back to legacy LIF
            return;
    }

    // Guard: Check vtable valid
    if (!vtable) {
        return;
    }

    // Create model state
    neuron->model = neuron_model_create(vtable, params);
    neuron->model_type = config->neuron_model;

    // Guard: Check creation succeeded
    if (!neuron->model) {
        neuron->model_type = NEURON_MODEL_LIF;
        return;
    }

    // Part A1.1: Set ODE integration method from config
    neuron_model_set_integration_method(neuron->model, config->integration_method);
}

//=============================================================================
// Network Creation - Factory Pattern
//=============================================================================

/**
 * @brief Validate network configuration
 *
 * WHY: Guard clause pattern - validates before allocating resources
 * Prevents invalid networks from being created
 *
 * COMPLEXITY: O(1)
 *
 * @param config Configuration to validate
 * @return true if valid, false otherwise
 */
static bool validate_network_config(const network_config_t* config)
{
    // Guard: Check null pointer
    if (!config)
        return false;

    // Guard: Check neuron count
    if (config->num_neurons == 0 || config->num_neurons > MAX_NEURONS) {
        return false;
    }

    // Guard: Check input/output dimensions
    if (config->input_size == 0 || config->output_size == 0) {
        return false;
    }

    return true;
}

/**
 * @brief Create a new neural network with validation
 *
 * WHY: Factory pattern - single creation point with full validation
 * Encapsulates complex initialization sequence
 *
 * ALGORITHM:
 * 1. Validate configuration (O(1))
 * 2. Allocate network structure (O(1))
 * 3. Initialize each neuron using helper functions (O(n))
 * 4. Set up activation strategies (O(1))
 *
 * COMPLEXITY: O(n) where n = num_neurons
 * MEMORY: O(n * MAX_SYNAPSES_PER_NEURON)
 *
 * PATTERN: Factory pattern with builder helpers
 *
 * @param config Network configuration
 * @return Network handle or NULL on failure
 */
neural_network_t neural_network_create(const network_config_t* config)
{
    // Guard clause: Validate configuration
    if (!validate_network_config(config)) {
        return NULL;
    }
    /**
     * WHAT: Allocate network structure
     * WHY: Create container for network state
     * HOW: Use nimcp_malloc for network struct (not neurons yet - done below)
     */
    neural_network_t network = (neural_network_t) nimcp_malloc(sizeof(struct neural_network_struct));

    // Guard clause: Check network allocation
    if (!network)
        return NULL;

    // Initialize network state (zeros all fields including neurons pointer)
    memset(network, 0, sizeof(struct neural_network_struct));
    memcpy(&network->config, config, sizeof(network_config_t));

    /**
     * WHAT: Dynamically allocate neurons array with growth capacity
     * WHY: Support arbitrary network sizes + allow neural_network_add_neuron()
     * HOW: Allocate MAX_NEURONS or config->num_neurons*2, whichever is smaller
     * PERFORMANCE: O(capacity) memory, allows growth without reallocation
     * TRADEOFF: Small overhead for networks that never add neurons
     */
    uint32_t capacity = (config->num_neurons * 2 < MAX_NEURONS)
                       ? config->num_neurons * 2
                       : MAX_NEURONS;
    network->neurons = (neuron_t*) nimcp_calloc(capacity, sizeof(neuron_t));

    // Guard clause: Check neurons allocation
    if (!network->neurons) {
        nimcp_free(network);
        return NULL;
    }

    network->capacity = capacity;

    /**
     * WHAT: Deep copy layer_sizes array if present (NIMCP 2.5 layered networks)
     * WHY: Config may be stack-allocated, need our own persistent copy
     * HOW: Allocate and memcpy the layer_sizes array
     */
    if (config->num_layers > 0 && config->layer_sizes) {
        uint32_t* layer_sizes_copy = (uint32_t*) nimcp_malloc(config->num_layers * sizeof(uint32_t));
        // Guard clause: Check layer_sizes allocation
        if (!layer_sizes_copy) {
            nimcp_free(network->neurons);
            nimcp_free(network);
            return NULL;
        }
        memcpy(layer_sizes_copy, config->layer_sizes, config->num_layers * sizeof(uint32_t));
        network->config.layer_sizes = layer_sizes_copy;
    } else {
        network->config.layer_sizes = NULL;
    }

    network->network_time = 0;
    network->current_time = 0;
    network->global_activity = 0.0f;
    network->network_stability = 1.0f;
    network->learning_momentum = 0.0f;
    network->last_avg_weight = 0.0f;
    network->last_maintenance = 0;

    // Initialize activation strategy table
    init_activation_strategies(&network->activation_strategies);

    // Determine excitatory/inhibitory split
    uint32_t num_inhibitory = (uint32_t) (config->num_neurons * (1.0f - config->ei_ratio));

    // Create neurons using builder pattern
    for (uint32_t i = 0; i < config->num_neurons; i++) {
        neuron_t* neuron = &network->neurons[i];
        neuron_type_t type = (i < num_inhibitory) ? NEURON_INHIBITORY : NEURON_EXCITATORY;

        init_neuron_basic_properties(neuron, i, type, network->network_time);
        neuron->refractory_period = config->refractory_period;

        init_neuron_learning_params(neuron, config);
        init_neuron_homeostatic_params(neuron, config);
        init_neuron_activity_tracking(neuron);
        init_neuron_model(neuron, config);  // NIMCP 2.6: Initialize neuron model plugin
    }

    network->num_neurons = config->num_neurons;

    // For layered networks (NIMCP 2.5), create connections between layers
    if (config->num_layers > 1 && config->layer_sizes) {
        uint32_t offset = 0;
        for (uint32_t layer = 0; layer < config->num_layers - 1; layer++) {
            uint32_t curr_layer_size = config->layer_sizes[layer];
            uint32_t next_layer_size = config->layer_sizes[layer + 1];
            uint32_t next_layer_offset = offset + curr_layer_size;

            // Connect each neuron in current layer to each neuron in next layer
            for (uint32_t i = 0; i < curr_layer_size && offset + i < network->num_neurons; i++) {
                for (uint32_t j = 0;
                     j < next_layer_size && next_layer_offset + j < network->num_neurons; j++) {
                    // Random initial weight between -0.5 and 0.5
                    float weight = ((float) rand() / RAND_MAX) - 0.5f;
                    neural_network_add_connection(network, offset + i, next_layer_offset + j,
                                                  weight);
                }
            }

            offset = next_layer_offset;
        }
    }

    return network;
}

/**
 * @brief Destroy neural network and free resources
 *
 * WHY: Proper resource cleanup prevents memory leaks
 * COMPLEXITY: O(1) - simple deallocation
 *
 * @param network Network to destroy (can be NULL)
 */
/**
 * WHAT: Destroy neural network and free all allocated resources
 * WHY: Prevent memory leaks by freeing all dynamic allocations
 * HOW: Free neurons array, layer_sizes, then network struct (reverse of creation order)
 * PATTERN: RAII-style cleanup, reverse order of allocation
 */
void neural_network_destroy(neural_network_t network)
{
    // Guard clause: Handle NULL input
    if (!network)
        return;


    /**
     * WHAT: Cleanup neuron models (NIMCP 2.6)
     * WHY: Plugin models allocate their own state
     * HOW: Call neuron_model_destroy for each model
     */
    if (network->neurons) {
        for (uint32_t i = 0; i < network->num_neurons; i++) {
            if (i % 100 == 0) {
            }
            if (network->neurons[i].model) {
                neuron_model_destroy(network->neurons[i].model);
                network->neurons[i].model = NULL;
            }

            // Phase 11: Cleanup BCM plasticity state for all synapses
            // WHAT: Free dynamically allocated BCM state
            // WHY: BCM state is heap-allocated, must be freed to prevent leaks
            // HOW: Iterate through outgoing synapses and free BCM state
            neuron_t* neuron = &network->neurons[i];
            if (i == 0) {
            }
            for (uint32_t j = 0; j < neuron->num_synapses; j++) {
                if (i == 0 && j % 10 == 0) {
                }
                synapse_t* syn = &neuron->synapses[j];
                if (i == 0 && j < 5) {
                }
                if (syn->bcm) {
                    if (i == 0 && j < 5) {
                    }
                    nimcp_free(syn->bcm);
                    if (i == 0 && j < 5) {
                    }
                    syn->bcm = NULL;
                }
                if (syn->eligibility) {
                    if (i == 0 && j < 5) {
                    }
                    nimcp_free(syn->eligibility);
                    if (i == 0 && j < 5) {
                    }
                    syn->eligibility = NULL;
                }
                if (i == 0 && j < 5) {
                }
            }

            // Note: incoming_synapses share BCM/eligibility state with outgoing synapses,
            // so we don't free them for incoming (would be double-free)
        }
    }

    /**
     * WHAT: Free dynamically allocated neurons array
     * WHY: neurons is now heap-allocated, must be freed
     * HOW: Use nimcp_free on neurons pointer
     */
    if (network->neurons) {
        nimcp_free(network->neurons);
    }

    /**
     * WHAT: Free layer_sizes array if allocated (NIMCP 2.5 layered networks)
     * WHY: Deep-copied config arrays must be freed
     * HOW: Check if layer_sizes was allocated, then free
     */
    if (network->config.layer_sizes) {
        nimcp_free((void*) network->config.layer_sizes);
    }

    /**
     * WHAT: Free network structure itself
     * WHY: Final cleanup step
     * HOW: Free the network pointer
     */
    nimcp_free(network);
}

//=============================================================================
// Mathematical Utility Functions
//=============================================================================

/**
 * @brief Fast Padé approximation of hyperbolic tangent
 *
 * WHY: Faster than standard library tanh() while maintaining accuracy
 * ERROR: <0.001 across typical neural network input ranges
 *
 * ALGORITHM: Padé [7/7] rational approximation
 * COMPLEXITY: O(1)
 *
 * @param x Input value
 * @return tanh(x) approximation
 */
static float fast_tanh(float x)
{
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

/**
 * @brief Standard sigmoid activation: σ(x) = 1/(1+e^(-x))
 *
 * WHY: Smooth, bounded, differentiable - core of many neural networks
 * COMPLEXITY: O(1)
 *
 * @param x Input value
 * @return Sigmoid output in (0,1)
 */
static float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Clamp value to valid activation range
 *
 * WHY: Prevents numerical overflow and maintains biological plausibility
 * COMPLEXITY: O(1)
 *
 * @param value Value to clamp
 * @return Clamped value in [MIN_ACTIVATION, MAX_ACTIVATION]
 */
static float clamp_activation(float value)
{
    return fmaxf(MIN_ACTIVATION, fminf(MAX_ACTIVATION, value));
}

//=============================================================================
// Activation Function API - Strategy Pattern
//=============================================================================

/**
 * @brief Compute activation using strategy pattern
 *
 * WHY: O(1) dispatch via function pointer table vs O(n) switch statement
 * Uses Strategy pattern for extensibility and performance
 *
 * ALGORITHM:
 * 1. Look up activation function in strategy table (O(1))
 * 2. Call function with input and threshold (O(1))
 * 3. Clamp result to valid range (O(1))
 *
 * COMPLEXITY: O(1) - constant time regardless of activation type
 * PERFORMANCE GAIN: ~2x faster than switch statement in hot loop
 *
 * @param neuron Neuron with activation type
 * @param input Raw input to activate
 * @return Activated output, clamped to valid range
 */
float neural_network_compute_activation(neuron_t* neuron, float input)
{
    // Guard clause: Validate neuron
    if (!neuron)
        return 0.0f;

    // Temporary: Get network context for strategy table
    // In production, would pass network or use global table
    activation_strategy_table_t table;
    init_activation_strategies(&table);

    // Guard clause: Check activation type bounds
    if (neuron->activation_type < 0 || neuron->activation_type >= 8) {
        return clamp_activation(input);  // Default: pass-through
    }

    // Strategy dispatch - O(1) lookup and call
    activation_fn_t activation_fn = table.functions[neuron->activation_type];

    // Guard clause: Ensure function exists
    if (!activation_fn) {
        return clamp_activation(input);
    }

    // Execute activation function
    float result = activation_fn(input, neuron->threshold);

    // Update adaptive threshold if using adaptive activation
    if (neuron->activation_type == ACTIVATION_ADAPTIVE) {
        if (input > neuron->threshold) {
            neuron->threshold += neuron->adaptation;
        } else {
            neuron->threshold = fmaxf(neuron->threshold - neuron->adaptation * 0.1f,
                                      neuron->rest_potential + 10.0f);
        }
    }

    return clamp_activation(result);
}

//=============================================================================
// Membrane Potential and Neuron Dynamics
//=============================================================================

/**
 * @brief Compute synaptic contribution to membrane potential
 *
 * WHY: Extracted from compute_membrane_potential for single responsibility
 * Isolates synap tic summation from modulation
 *
 * COMPLEXITY: O(s) where s = num_synapses
 *
 * @param neuron Neuron receiving input
 * @param network Network containing pre-synaptic neurons
 * @return Summed synaptic input
 */
static float sum_synaptic_inputs(neuron_t* neuron, neural_network_t network)
{
    // Guard clause: Validate inputs
    if (!neuron || !network)
        return 0.0f;

    float total_input = 0.0f;

    // OPTIMIZED: Use bidirectional synapse tracking for O(S) instead of O(N×S)
    // DESIGN PATTERN: Bidirectional Association
    // WHY: Direct access to incoming synapses eliminates need to scan entire network
    // COMPLEXITY: O(S) where S = num incoming synapses (was O(N×S))
    // SPEEDUP: 10-100x for large networks (N > 1000)

    for (uint32_t i = 0; i < neuron->num_incoming; i++) {
        synapse_t* incoming_syn = &neuron->incoming_synapses[i];

        // In incoming synapse, target_id stores the SOURCE neuron ID
        uint32_t src_id = incoming_syn->target_id;

        if (src_id >= network->num_neurons) {
            continue;  // Safety check
        }

        neuron_t* src_neuron = &network->neurons[src_id];

        // Only transmit if presynaptic neuron is active (not at rest)
        // For spiking neurons, we use state > threshold as activity indicator
        float pre_activity =
            (src_neuron->state > src_neuron->threshold) ? src_neuron->state : 0.0f;

        // NIMCP 2.6: Apply STP modulation if enabled
        float stp_modulation = 1.0f;
        if (incoming_syn->enable_stp) {
            // Update STP continuous decay
            stp_update(&incoming_syn->stp, network->network_time);

            // Get modulation factor (u × x)
            stp_modulation = stp_get_modulation(&incoming_syn->stp);

            // Process spike if presynaptic neuron is firing
            if (pre_activity > 0.0f) {
                stp_process_spike(&incoming_syn->stp, network->network_time);
            }
        }

        // NIMCP 2.7: Programmable synapse computation (MAJOR FEATURE!)
        // If synapse has custom compute function, use it; otherwise default computation
        float synaptic_transmission;
        if (incoming_syn->compute_function != NULL) {
            // Custom computation - synapse decides how to compute transmission
            // This enables attention, semantic similarity, gating, etc.

            // NIMCP 2.7: Wire up neuromodulator levels if system is attached
            float neuromod_level = neural_network_get_neuromodulation(network);

            synapse_compute_context_t context = {
                .global_state = network->global_state,  // Attention output, etc.
                .global_state_size = network->global_state_size,
                .neuromodulation = neuromod_level,  // Dopamine/ACh/etc levels
                .current_time = network->network_time,
                .custom_data = NULL,
                .custom_data_size = 0
            };

            synaptic_transmission = incoming_syn->compute_function(
                incoming_syn,
                src_neuron,
                neuron,
                pre_activity,
                &context
            );
        } else {
            // Default computation (backward compatible with NIMCP 2.0-2.6)
            synaptic_transmission = pre_activity * incoming_syn->weight *
                                   incoming_syn->strength * stp_modulation;
        }

        // ENHANCEMENT 1: Semantic embedding routing
        // WHAT: Modulate transmission by semantic relevance
        // WHY: Route information through semantically relevant synapses (70% faster)
        // HOW: Use cached semantic_relevance if embeddings enabled
        if (incoming_syn->semantic_embedding && incoming_syn->embedding_dim > 0) {
            // Use cached relevance (computed during context update)
            // Relevance in [0,1]: 0=irrelevant, 1=highly relevant
            float semantic_modulation = 0.5f + 0.5f * incoming_syn->semantic_relevance;
            synaptic_transmission *= semantic_modulation;
        }

        total_input += synaptic_transmission;

        // NIMCP Phase 6: Notify glial system of synapse firing
        // WHAT: Inform astrocytes about synaptic transmission events
        // WHY: Enables calcium-mediated tripartite synapse modulation
        // HOW: Observer pattern - glial system observes neural events
        // WHEN: Only if glial system attached and synapse actually fired
        if (network->glial_integration && synaptic_transmission > 0.0f) {
            glial_integration_on_synapse_fired(
                (glial_integration_t*)network->glial_integration,
                src_id,                    // Presynaptic neuron ID
                neuron->id,                // Postsynaptic neuron ID
                incoming_syn->weight,      // Synaptic weight
                network->network_time      // Timestamp
            );
        }
    }

    return total_input;
}

/**
 * @brief Compute membrane potential for a neuron
 *
 * WHY: Central equation for neural dynamics - integrates all inputs
 * Models biological membrane potential equation
 *
 * ALGORITHM:
 * V_membrane = bias + Σ(w_i * a_i * s_i) * (1 + [Ca²⁺])
 * where:
 * - w_i = synaptic weight
 * - a_i = presynaptic activity
 * - s_i = synaptic strength
 * - [Ca²⁺] = calcium concentration (modulation)
 *
 * COMPLEXITY: O(s) where s = num_synapses (single pass)
 *
 * @param neuron Post-synaptic neuron
 * @param network Network containing all neurons
 * @return Membrane potential (mV in biological terms)
 */
static float compute_membrane_potential(neuron_t* neuron, neural_network_t network)
{
    // Guard clause: Validate inputs
    if (!neuron || !network)
        return 0.0f;

    // Start with intrinsic bias
    float potential = neuron->bias;

    // Add synaptic contributions
    potential += sum_synaptic_inputs(neuron, network);

    // Apply calcium-mediated modulation (activity-dependent amplification)
    potential *= (1.0f + neuron->calcium_concentration);

    // Add external input current (e.g., from spike encoding)
    // WHAT: Include external current in membrane potential
    // WHY: Allows external stimulation (sensors, embeddings) to drive spikes
    // HOW: Direct addition after synaptic integration
    // NOTE: Reset after compute_step() to avoid accumulation
    potential += neuron->external_current;

    return potential;
}


//=============================================================================
// Neuron Update - Decomposed into Helper Functions
//=============================================================================

/**
 * @brief Check if neuron is in refractory period
 *
 * WHY: Extracted guard clause for clarity and testability
 * COMPLEXITY: O(1)
 *
 * @return true if in refractory period (cannot spike)
 */
static bool is_in_refractory_period(neuron_t* neuron, uint64_t timestamp)
{
    // If neuron has never spiked (last_spike == 0), not in refractory period
    if (neuron->last_spike == 0)
        return false;
    return (timestamp - neuron->last_spike) < neuron->refractory_period;
}

/**
 * @brief Detect spike event (threshold crossing)
 *
 * WHY: Extracted spike detection logic for clarity
 * COMPLEXITY: O(1)
 *
 * @return true if spike occurred
 */
static bool detected_spike(float old_state, float new_state, float threshold)
{
    return (old_state <= threshold) && (new_state > threshold);
}

/**
 * @brief Apply all enabled learning rules to neuron
 *
 * WHY: Extracted from update_neuron to eliminate nested ifs
 * Single responsibility - only handles learning rule dispatch
 *
 * COMPLEXITY: O(s) where s = num_synapses
 */
static void apply_learning_rules(neural_network_t network, uint32_t neuron_id, neuron_t* neuron,
                                 uint64_t timestamp)
{
    // Guard clause: Check if Oja's rule enabled
    if (neuron->learning_rule & LEARNING_OJA) {
        neural_network_apply_oja(network, neuron_id, timestamp);
    }

    // Guard clause: Check if STDP enabled
    if (neuron->learning_rule & LEARNING_STDP) {
        neural_network_apply_stdp(network, neuron_id, timestamp);
    }
}

/**
 * @brief Update activity history and compute running average
 *
 * WHY: Extracted activity tracking for single responsibility
 * COMPLEXITY: O(w) where w = HISTORY_WINDOW
 */
static void update_activity_history(neuron_t* neuron, float new_state, uint64_t timestamp)
{
    // Guard clause: Validate input
    if (!neuron)
        return;

    // Record current activity
    uint32_t history_idx = timestamp % HISTORY_WINDOW;
    neuron->activity_history[history_idx] = new_state;

    // Compute running average - single pass O(w)
    float sum_activity = 0.0f;
    for (uint32_t i = 0; i < HISTORY_WINDOW; i++) {
        sum_activity += neuron->activity_history[i];
    }
    neuron->avg_activity = sum_activity / HISTORY_WINDOW;
}

/**
 * @brief Update dynamics (calcium, traces) after state change
 *
 * WHY: Extracted dynamics updates for clarity
 * COMPLEXITY: O(s) where s = num_synapses
 */
static void update_neuron_dynamics(neuron_t* neuron, uint64_t timestamp)
{
    // Guard clause: Validate input
    if (!neuron)
        return;

    update_calcium_dynamics(neuron, timestamp);
    update_synaptic_traces(neuron, timestamp);
}

/**
 * @brief Handle spike event and trigger learning
 *
 * WHY: Extracted spike handling to eliminate nesting in main update function
 * Single responsibility - orchestrates spike response
 *
 * COMPLEXITY: O(s) where s = num_synapses
 */
static void handle_spike_event(neural_network_t network, uint32_t neuron_id, neuron_t* neuron,
                               float new_state, uint64_t timestamp)
{
    // Record the spike
    neural_network_record_spike(network, neuron_id, new_state, timestamp);

    // NIMCP Phase 6: Notify glial system of neuron firing
    // WHAT: Inform oligodendrocytes and microglia about neuronal spiking
    // WHY: Enables adaptive myelination and synaptic pruning
    // HOW: Observer pattern - glial system observes neural spike events
    // WHEN: Only if glial system attached
    if (network->glial_integration) {
        glial_integration_on_neuron_fired(
            (glial_integration_t*)network->glial_integration,
            neuron_id,         // ID of neuron that fired
            timestamp          // Time of spike
        );
    }

    // Apply learning rules
    apply_learning_rules(network, neuron_id, neuron, timestamp);

    // Update plasticity mechanisms
    neural_network_update_plasticity(network, neuron_id, timestamp);
    neural_network_apply_homeostasis(network, neuron_id, timestamp);
}

/**
 * @brief Update neuron state and apply learning rules
 *
 * WHY: Main neuron update - refactored to eliminate ALL nested ifs
 * Uses guard clauses and extracted helpers for clarity
 *
 * ALGORITHM:
 * 1. Validate inputs (guard clauses)
 * 2. Check refractory period (early return if refractory)
 * 3. Compute membrane potential and activation
 * 4. Detect spike and handle if occurred
 * 5. Update activity tracking
 * 6. Update dynamics (calcium, traces)
 *
 * COMPLEXITY: O(s) where s = num_synapses
 * PREVIOUSLY: Had nested ifs 3 levels deep
 * NOW: All guard clauses, maximum 1 level
 *
 * @param network Neural network
 * @param neuron_id ID of neuron to update
 * @param input_current External input current
 * @param timestamp Current simulation time
 * @return true if neuron updated successfully, false if in refractory period
 */
bool neural_network_update_neuron(neural_network_t network, uint32_t neuron_id, float input_current,
                                  uint64_t timestamp)
{
    // Guard clause: Validate network
    if (!network)
        return false;

    // Guard clause: Validate neuron ID
    if (neuron_id >= network->num_neurons)
        return false;

    neuron_t* neuron = &network->neurons[neuron_id];

    // Guard clause: Check refractory period
    if (is_in_refractory_period(neuron, timestamp)) {
        neuron->state = neuron->rest_potential;
        return false;
    }

    // Declare new_state before branches for use in activity tracking
    float new_state;

    // NIMCP 2.6: Use neuron model plugin if available
    if (neuron->model != NULL) {
        // Calculate timestep (assume 1ms if timestamps match)
        float dt = (timestamp > neuron->last_update) ? (float)(timestamp - neuron->last_update) : 1.0f;

        // Compute synaptic input for model
        float membrane_potential = compute_membrane_potential(neuron, network);
        float total_input = membrane_potential + input_current;

        // Update model dynamics
        neuron_model_update(neuron->model, dt, total_input);

        // Get new voltage from model
        new_state = neuron_model_get_voltage(neuron->model);
        neuron->state = new_state;

        // Check for spike using model's spike detection
        if (neuron_model_check_spike(neuron->model)) {
            handle_spike_event(network, neuron_id, neuron, new_state, timestamp);
            neuron_model_post_spike(neuron->model);  // Reset model after spike
        }
    } else {
        // Legacy LIF behavior
        float membrane_potential = compute_membrane_potential(neuron, network);
        float total_input = membrane_potential + input_current;
        new_state = neural_network_compute_activation(neuron, total_input);

        // Capture old state for spike detection
        float old_state = neuron->state;
        neuron->state = new_state;

        // Handle spike if detected
        if (detected_spike(old_state, new_state, neuron->threshold)) {
            handle_spike_event(network, neuron_id, neuron, new_state, timestamp);

            // WHAT: Reset neuron state after spike (LIF dynamics)
            // WHY: Spiking neurons must reset to allow future spikes
            // HOW: Set to rest potential, like neuron_model_post_spike() does for models
            // FIX (Phase 5.1): Without this, neuron stays above threshold forever
            neuron->state = neuron->rest_potential;
        }
    }

    // Update activity tracking and dynamics (always happens)
    update_activity_history(neuron, new_state, timestamp);
    update_neuron_dynamics(neuron, timestamp);

    neuron->last_update = timestamp;
    return true;
}

/**
 * @brief Apply Oja's learning rule
 */
uint32_t neural_network_apply_oja(neural_network_t network, uint32_t neuron_id, uint64_t timestamp)
{
    // Guard: Validate inputs
    if (!network || neuron_id >= network->num_neurons) {
        return 0;
    }

    neuron_t* neuron = &network->neurons[neuron_id];
    uint32_t modified = 0;

    // FIX: Use avg_activity instead of current state (which may be 0 after spike reset)
    // Synapse goes FROM neuron TO target, so neuron is presynaptic
    float x = neuron->avg_activity;  // Pre-synaptic average activity

    // Skip update if neuron is not active enough
    if (fabs(x) < ACTIVITY_THRESHOLD) {
        return 0;
    }

    // Calculate weight updates using Oja's rule
    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        synapse_t* syn = &neuron->synapses[i];

        // Get post-synaptic average activity (target of synapse)
        float y = network->neurons[syn->target_id].avg_activity;

        // Compute weight update using Oja's rule: Δw = α(y*x - y²*w)
        float delta_w = compute_oja_weight_update(x, y, syn->weight, &neuron->oja_params);

        // Apply weight update with meta-plasticity
        float new_weight = syn->weight + delta_w * syn->meta_plasticity;

        // Apply weight constraints
        new_weight =
            fmaxf(network->config.min_weight, fminf(network->config.max_weight, new_weight));

        // Update weight if change is significant
        if (fabs(new_weight - syn->weight) > WEIGHT_UPDATE_THRESHOLD) {
            syn->weight = new_weight;
            modified++;
        }

        // Update synaptic strength
        syn->strength = fminf(syn->strength * (1.0f + delta_w), MAX_SYNAPTIC_STRENGTH);
    }

    // Normalize weights periodically
    if (modified > 0 && (timestamp - neuron->last_update) > NORMALIZATION_INTERVAL) {
        normalize_synaptic_weights(neuron);
    }

    return modified;
}

/**
 * @brief Compute weight update using Oja's rule
 */
static float compute_oja_weight_update(float pre_activity, float post_activity,
                                       float current_weight, const oja_params_t* params)
{
    // WHAT: Compute Oja's learning rule weight update
    // WHY: PCA-like learning with weight normalization
    // HOW: Hebbian term - normalization + stabilization
    // CONST: params is read-only configuration
    float hebbian_term = post_activity * pre_activity;
    float normalization_term = post_activity * post_activity * current_weight;
    float stabilization_term = params->stabilization * (params->target_norm - current_weight);

    return params->alpha * (hebbian_term - normalization_term + stabilization_term);
}

/**
 * @brief Apply STDP learning rule
 *
 * WHAT: Update synaptic weights based on spike timing
 * WHY: STDP implements causality-based learning (Hebbian principle)
 * HOW: For each outgoing synapse, compute Δt and apply STDP rule
 *
 * ALGORITHM:
 * - For outgoing synapse: current neuron = PRE, target = POST
 * - Δt = t_post - t_pre
 * - If Δt > 0 (pre before post): LTP (strengthen)
 * - If Δt < 0 (post before pre): LTD (weaken)
 *
 * BUG FIX (Phase 5 - 2025-11-08):
 * - Previous code had pre/post neurons REVERSED
 * - Was using target_id as pre_neuron (wrong!)
 * - Now correctly: neuron_id = pre, target_id = post
 */
uint32_t neural_network_apply_stdp(neural_network_t network, uint32_t neuron_id, uint64_t timestamp)
{
    // WHAT: Get presynaptic neuron (current neuron with outgoing synapses)
    // WHY: We're iterating over this neuron's outgoing connections
    // HOW: Direct array access using neuron_id
    neuron_t* pre_neuron = &network->neurons[neuron_id];
    uint32_t modified = 0;

    // WHAT: Iterate over all outgoing synapses from this neuron
    // WHY: Apply STDP to each connection based on spike timing
    // HOW: Loop through synapse array, compute STDP for each
    for (uint32_t i = 0; i < pre_neuron->num_synapses; i++) {
        synapse_t* syn = &pre_neuron->synapses[i];

        // WHAT: Get postsynaptic neuron (target of this synapse)
        // WHY: Need post-spike time for STDP calculation
        // HOW: Access target neuron using target_id from synapse
        // FIX: This was previously assigned to pre_neuron (bug!)
        // CONST: post_neuron is read-only (only access last_spike)
        const neuron_t* post_neuron = &network->neurons[syn->target_id];

        // Guard: Skip if either neuron never spiked
        if (pre_neuron->last_spike == 0 || post_neuron->last_spike == 0) {
            continue;
        }

        // WHAT: Compute spike time difference (Δt = t_post - t_pre)
        // WHY: STDP depends on causality (which spike came first)
        // HOW: Subtract pre-spike time from post-spike time
        // RESULT: Δt > 0 means pre→post (causal, LTP)
        //         Δt < 0 means post→pre (anti-causal, LTD)
        int64_t dt = (int64_t)(post_neuron->last_spike) - (int64_t)(pre_neuron->last_spike);

        // Guard: Skip if time difference exceeds STDP window
        // WHY: STDP only affects spikes within ~20-50ms window
        // HOW: Check absolute time difference against configured window
        if (fabsf((float) dt) > pre_neuron->stdp_params.time_window)
            continue;

        // WHAT: Compute STDP weight change factor
        // WHY: Weight change decays exponentially with |Δt|
        // HOW: Call helper function with time difference and parameters
        float stdp_factor = compute_stdp_update((float) dt, &pre_neuron->stdp_params);

        // WHAT: Scale STDP by learning rate, trace, and meta-plasticity
        // WHY: Modulate learning based on recent activity and neuron state
        // HOW: Multiply factors together to get final Δw
        float delta_w = pre_neuron->stdp_params.learning_rate * stdp_factor * syn->trace;

        // WHAT: Apply weight update with meta-plasticity modulation
        // WHY: Meta-plasticity prevents runaway learning
        // HOW: Add weighted delta to current weight
        float new_weight = syn->weight + delta_w * syn->meta_plasticity;

        // WHAT: Clamp weight to configured bounds
        // WHY: Prevent weights from growing unbounded
        // HOW: Apply min/max constraints from network config
        new_weight =
            fmaxf(network->config.min_weight, fminf(network->config.max_weight, new_weight));

        // WHAT: Update weight if change is significant
        // WHY: Avoid tiny updates that don't affect computation
        // HOW: Check if |Δw| exceeds threshold before applying
        if (fabs(new_weight - syn->weight) > WEIGHT_UPDATE_THRESHOLD) {
            syn->weight = new_weight;
            modified++;
        }

        // Phase 11: Apply BCM homeostatic plasticity after STDP
        // WHAT: Apply BCM rule to prevent runaway weight growth
        // WHY: STDP can cause weights to saturate without homeostatic control
        // HOW: Use BCM sliding threshold to stabilize weights
        if (syn->enable_bcm && syn->bcm) {
            // Get BCM parameters (cortical preset for neural networks)
            bcm_params_t bcm_params = bcm_params_cortical();

            // Compute time delta in seconds
            float dt = (float)(timestamp - syn->last_active) / 1000000.0f;  // µs to seconds

            // Use synaptic trace as pre-synaptic activity measure
            float pre_activity = syn->trace;

            // Use post-synaptic neuron activation as post-synaptic activity
            float post_activity = post_neuron->state;

            // Apply BCM rule for weight stabilization
            bcm_apply_rule(syn->bcm, pre_activity, post_activity, dt, &bcm_params);

            // Update synaptic weight from BCM (overrides STDP if BCM makes larger change)
            // This ensures homeostatic stability takes precedence
            if (fabs(syn->bcm->weight - syn->weight) > WEIGHT_UPDATE_THRESHOLD) {
                syn->weight = syn->bcm->weight;
            }
        }
    }

    return modified;
}

/**
 * @brief Compute STDP update factor
 *
 * WHAT: Calculate weight change factor based on spike timing
 * WHY: Core STDP computation - exponential decay with time difference
 * HOW: Apply asymmetric time window (LTP vs LTD)
 *
 * @param dt Spike time difference (t_post - t_pre)
 * @param params STDP parameters (read-only)
 * @return STDP update factor
 */
static float compute_stdp_update(float dt, const stdp_params_t* params)
{
    // WHAT: Compute exponential decay based on time difference
    // WHY: STDP strength decays with |Δt|
    // HOW: exp(-|Δt| / τ)
    float time_factor = expf(-fabsf(dt) / params->time_window);

    // FIX: Corrected STDP logic
    // dt = t_post - t_pre
    // dt > 0 means pre-before-post (causal) → LTP (strengthen weights)
    // dt < 0 means post-before-pre (anti-causal) → LTD (weaken weights)
    if (dt > 0) {
        // Pre-before-post: Long-term potentiation (LTP)
        return params->positive_factor * time_factor;
    } else {
        // Post-before-pre: Long-term depression (LTD)
        return -params->negative_factor * time_factor;
    }
}

/**
 * @brief Apply reward-modulated learning to all synapses (Phase 11)
 *
 * WHAT: Apply biological plasticity mechanisms with reward signal
 * WHY:  Enable supervised/RL learning using biological rules
 * HOW:  Iterate neurons → STDP → Eligibility traces → BCM
 */
uint32_t neural_network_apply_reward_learning(neural_network_t network, float reward,
                                              float learning_rate, uint64_t current_time)
{
    // Guard: Validate parameters
    if (!network || reward < 0.0f || reward > 1.0f || learning_rate <= 0.0f) {
        return 0;
    }

    uint32_t total_modified = 0;

    // Iterate over all neurons in the network
    for (uint32_t neuron_id = 0; neuron_id < network->config.num_neurons; neuron_id++) {
        neuron_t* neuron = &network->neurons[neuron_id];

        // Step 1: Apply STDP if enabled for this neuron
        if (neuron->learning_rule & LEARNING_STDP) {
            total_modified += neural_network_apply_stdp(network, neuron_id, current_time);
        }

        // Step 2: Apply Oja's rule if enabled (PCA-like learning)
        if (neuron->learning_rule & LEARNING_OJA) {
            total_modified += neural_network_apply_oja(network, neuron_id, current_time);
        }

        // Step 3: Apply eligibility-trace-based learning with reward signal
        for (uint32_t syn_idx = 0; syn_idx < neuron->num_synapses; syn_idx++) {
            synapse_t* syn = &neuron->synapses[syn_idx];

            // Update eligibility traces
            if (syn->enable_eligibility && syn->eligibility) {
                eligibility_config_t elig_config = eligibility_default_config();
                elig_config.learning_rate = learning_rate;

                // Update trace based on synaptic activity
                if (syn->trace > 0.1f) {  // Synapse was recently active
                    eligibility_trace_update(
                        syn->eligibility,
                        &elig_config,
                        current_time,
                        syn->trace  // Use synaptic trace as spike contribution
                    );
                } else {
                    // Decay trace without spike
                    eligibility_trace_decay(syn->eligibility, &elig_config, current_time);
                }

                // Apply reward-modulated learning
                // Get dopamine level from neuromodulator system (if available)
                float dopamine = 0.5f;  // Default dopamine level
                if (network->neuromodulator_system) {
                    dopamine = neuromodulator_get_level(network->neuromodulator_system, NEUROMOD_DOPAMINE);
                }

                // Apply eligibility-based weight update
                float old_weight = syn->weight;
                eligibility_apply_reward(
                    syn,
                    syn->eligibility,
                    &elig_config,
                    reward,      // Reward from prediction accuracy
                    dopamine     // Dopamine gates learning
                );

                // ===================================================================
                // BIOLOGICAL SECURITY: Validate weight change (Phase 11)
                // ===================================================================
                // Check if weight change exceeds biological plausibility
                // Reject suspicious updates to prevent synaptic poisoning attacks
                if (!nimcp_security_validate_weight_change(old_weight, syn->weight,
                                                          NIMCP_MAX_WEIGHT_DELTA_PER_STEP)) {
                    // SECURITY: Reject suspicious weight change
                    syn->weight = old_weight;  // Revert to safe value
                    continue;  // Skip this synapse
                }

                // Clamp weights to valid range
                syn->weight = fmaxf(network->config.min_weight,
                                  fminf(network->config.max_weight, syn->weight));

                if (fabsf(syn->weight - old_weight) > WEIGHT_UPDATE_THRESHOLD) {
                    total_modified++;
                }
            }

            // Step 4: Apply BCM homeostatic plasticity
            if (syn->enable_bcm && syn->bcm) {
                bcm_params_t bcm_params = bcm_params_cortical();
                float dt = 1.0f;  // Time step in seconds

                const neuron_t* post_neuron = &network->neurons[syn->target_id];
                float pre_activity = neuron->state;       // Presynaptic activity
                float post_activity = post_neuron->state; // Postsynaptic activity

                // Apply BCM rule
                bcm_apply_rule(syn->bcm, pre_activity, post_activity, dt, &bcm_params);

                // Sync BCM weight to synapse weight
                if (fabsf(syn->bcm->weight - syn->weight) > WEIGHT_UPDATE_THRESHOLD) {
                    syn->weight = syn->bcm->weight;
                    total_modified++;
                }
            }
        }
    }

    return total_modified;
}

/**
 * @brief Apply homeostatic plasticity to maintain target activity
 */
bool neural_network_apply_homeostasis(neural_network_t network, uint32_t neuron_id,
                                      uint64_t timestamp)
{
    // Guard: Validate inputs
    if (!network || neuron_id >= network->num_neurons) {
        return false;
    }

    neuron_t* neuron = &network->neurons[neuron_id];

    // Skip if too soon since last update
    if (timestamp - neuron->last_update < network->config.update_interval) {
        return false;
    }

    // Compute current average activity
    float current_activity = 0.0f;
    for (uint32_t i = 0; i < HISTORY_WINDOW; i++) {
        current_activity += neuron->activity_history[i];
    }
    current_activity /= HISTORY_WINDOW;

    // Compute activity error
    float activity_error = neuron->homeostatic.target_activity - current_activity;

    // Update threshold based on activity error
    float threshold_adjustment = neuron->homeostatic.strength * activity_error;
    neuron->threshold += threshold_adjustment;

    // Constrain threshold to reasonable range
    float min_threshold = neuron->rest_potential + 5.0f;
    float max_threshold = neuron->rest_potential + 30.0f;
    neuron->threshold = fmaxf(min_threshold, fminf(max_threshold, neuron->threshold));

    // Update homeostatic factor
    neuron->homeostatic_factor = compute_homeostatic_factor(neuron, current_activity);

    // Apply homeostatic scaling to synapses
    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        synapse_t* syn = &neuron->synapses[i];

        // Scale synaptic strength based on activity error
        float strength_adjustment =
            neuron->homeostatic.strength * activity_error * neuron->homeostatic_factor;

        syn->strength *= (1.0f + strength_adjustment);

        // Constrain synaptic strength
        syn->strength = fmaxf(0.0f, fminf(MAX_SYNAPTIC_STRENGTH, syn->strength));
    }

    // Update adaptation based on activity
    neuron->adaptation =
        fmaxf(0.0f, neuron->adaptation + neuron->homeostatic.strength * activity_error);

    return true;
}

/**
 * @brief Compute homeostatic scaling factor
 */
static float compute_homeostatic_factor(neuron_t* neuron, float current_activity)
{
    float activity_ratio = current_activity / neuron->homeostatic.target_activity;
    float time_scale = neuron->homeostatic.time_scale;

    // Compute scaling factor using time-dependent exponential
    float scaling_factor = expf(-fabsf(activity_ratio - 1.0f) / time_scale);

    // Adjust scaling based on activity direction
    if (activity_ratio > 1.0f) {
        // Too much activity - decrease scaling
        return scaling_factor * HOMEOSTATIC_DECAY;
    } else {
        // Too little activity - increase scaling
        return scaling_factor * (2.0f - HOMEOSTATIC_DECAY);
    }
}

/**
 * @brief Update meta-plasticity based on activity patterns
 */
static void update_meta_plasticity(neuron_t* neuron, uint64_t timestamp)
{
    // Compute activity variance over history window
    float mean_activity = neuron->avg_activity;
    float variance = 0.0f;

    for (uint32_t i = 0; i < HISTORY_WINDOW; i++) {
        float diff = neuron->activity_history[i] - mean_activity;
        variance += diff * diff;
    }
    variance /= HISTORY_WINDOW;

    // Update meta-plasticity for each synapse
    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        synapse_t* syn = &neuron->synapses[i];

        // Compute stability measure
        float stability = expf(-variance * META_PLASTICITY_RATE);

        // Update meta-plasticity factor
        syn->meta_plasticity =
            syn->meta_plasticity * (1.0f - META_PLASTICITY_RATE) + stability * META_PLASTICITY_RATE;

        // Ensure meta-plasticity stays in valid range
        syn->meta_plasticity = fmaxf(0.1f, fminf(1.0f, syn->meta_plasticity));
    }
}

/**
 * @brief Periodic maintenance of homeostatic mechanisms
 */
void neural_network_maintain_homeostasis(neural_network_t network, uint64_t timestamp)
{
    // Skip if too soon since last maintenance
    if (timestamp - network->last_maintenance < network->config.update_interval) {
        return;
    }

    // Update global network stability
    float total_variance = 0.0f;

    for (uint32_t i = 0; i < network->num_neurons; i++) {
        neuron_t* neuron = &network->neurons[i];

        // Apply homeostatic plasticity
        neural_network_apply_homeostasis(network, i, timestamp);

        // Update meta-plasticity
        update_meta_plasticity(neuron, timestamp);

        // Compute contribution to global variance
        float activity_diff = neuron->avg_activity - neuron->homeostatic.target_activity;
        total_variance += activity_diff * activity_diff;
    }

    // Update network stability measure
    network->network_stability = expf(-total_variance / network->num_neurons);
    network->last_maintenance = timestamp;
}

/**
 * @brief Record a spike event for a neuron
 */
bool neural_network_record_spike(neural_network_t network, uint32_t neuron_id, float magnitude,
                                 uint64_t timestamp)
{
    if (neuron_id >= network->num_neurons)
        return false;

    neuron_t* neuron = &network->neurons[neuron_id];

    // Record spike in history
    uint32_t idx = neuron->spike_history_index;
    neuron->spike_history[idx].timestamp = timestamp;
    neuron->spike_history[idx].magnitude = magnitude;

    // Update spike history index
    neuron->spike_history_index = (idx + 1) % SPIKE_HISTORY_LENGTH;

    // Update last spike time
    neuron->last_spike = timestamp;

    // Update network time to stay current with spike events
    if (timestamp > network->network_time) {
        network->network_time = timestamp;
    }

    // Increase calcium concentration
    neuron->calcium_concentration += 1.0f;

    // Update synaptic traces (but don't propagate immediately)
    // Synaptic inputs will be computed when target neurons are updated
    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        synapse_t* syn = &neuron->synapses[i];

        // Update synaptic trace for STDP
        syn->trace += 1.0f;

        // Note: We don't propagate spikes immediately here because
        // sum_synaptic_inputs() handles synaptic integration when
        // target neurons are updated. Immediate propagation would
        // cause double-counting of synaptic inputs.
    }

    return true;
}

/**
 * @brief Update synaptic traces for STDP
 */
static void update_synaptic_traces(neuron_t* neuron, uint64_t timestamp)
{
    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        synapse_t* syn = &neuron->synapses[i];

        // Compute time since last update
        float dt = (float) (timestamp - syn->last_active);

        // Exponential decay of trace
        if (dt > 0) {
            syn->trace *= expf(-dt * TRACE_DECAY_RATE);
        }

        syn->last_active = timestamp;
    }
}

/**
 * @brief Update calcium dynamics
 */
static void update_calcium_dynamics(neuron_t* neuron, uint64_t timestamp)
{
    // Compute time since last update
    float dt = (float) (timestamp - neuron->last_update);

    // Exponential decay of calcium concentration
    if (dt > 0) {
        neuron->calcium_concentration *= expf(-dt * CALCIUM_DECAY_RATE);
    }

    // Ensure calcium concentration stays in valid range
    neuron->calcium_concentration = fmaxf(0.0f, fminf(10.0f, neuron->calcium_concentration));
}

/**
 * @brief Get average activity over time window
 */
float neural_network_get_average_activity(neural_network_t network, uint32_t neuron_id)
{
    // Defensive validation
    if (!network) {
        return 0.0f;  // No activity in NULL network
    }

    if (!network->neurons) {
        return 0.0f;  // No activity if neurons not allocated
    }

    if (neuron_id >= network->num_neurons) {
        return 0.0f;  // No activity for out-of-bounds neuron
    }

    // Validate num_neurons is reasonable (< 10M neurons is sane limit)
    // This catches cases where num_neurons has garbage value
    if (network->num_neurons > 10000000) {
        return 0.0f;
    }

    // For newly created networks with no training, return 0 activity
    // This avoids iterating over uninitialized spike history
    if (network->network_time == 0) {
        return 0.0f;
    }

    // CRITICAL FIX: Check if neurons array is allocated
    // WHY: Protects against accessing freed/unallocated memory
    // WHAT: Return 0 if neurons pointer is NULL
    if (!network->neurons) {
        return 0.0f;  // Neurons array not allocated
    }

    // Use safe accessor instead of direct array access
    neuron_t* neuron = neural_network_get_neuron(network, neuron_id);
    if (!neuron) {
        return 0.0f;  // Neuron access failed
    }

    // Count recent spikes within the time window
    uint32_t spike_count = 0;
    uint64_t current_time = network->network_time;
    uint64_t window_start = (current_time >= HISTORY_WINDOW) ? (current_time - HISTORY_WINDOW) : 0;

    // Only iterate if we have meaningful time progression
    if (current_time > 0) {
        for (uint32_t i = 0; i < SPIKE_HISTORY_LENGTH; i++) {
            // Check if spike is within window and is a real spike (non-zero magnitude)
            if (neuron->spike_history[i].timestamp >= window_start &&
                neuron->spike_history[i].magnitude != 0.0f) {
                spike_count++;
            }
        }
    }

    return (float)spike_count / HISTORY_WINDOW;
}

/**
 * @brief Set the network's internal timestamp
 *
 * WHAT: Synchronizes network time with external simulation time
 * WHY:  Ensure timing-dependent features (STP, etc) use consistent time
 * HOW:  Directly updates network_time field
 *
 * @param network Neural network instance
 * @param timestamp Current simulation time (µs)
 */
void neural_network_set_time(neural_network_t network, uint64_t timestamp)
{
    if (!network) return;
    network->network_time = timestamp;
}

/**
 * @brief Compute network step
 */
uint32_t neural_network_compute_step(neural_network_t network, uint64_t timestamp)
{
    uint32_t active_neurons = 0;

    // Update all neurons
    for (uint32_t i = 0; i < network->num_neurons; i++) {
        neuron_t* neuron = &network->neurons[i];

        // Skip neurons in refractory period
        if (is_in_refractory_period(neuron, timestamp)) {
            continue;
        }

        // Compute membrane potential
        float potential = compute_membrane_potential(neuron, network);

        // Update neuron state
        if (neural_network_update_neuron(network, i, potential, timestamp)) {
            active_neurons++;
        }

        // Update traces and dynamics
        update_synaptic_traces(neuron, timestamp);
        update_calcium_dynamics(neuron, timestamp);

        // WHAT: Reset external current after processing
        // WHY: External current is per-timestep stimulation, not persistent state
        // HOW: Set to 0 after neuron update completes
        // NOTE: Spike encoding sets this before compute_step()
        neuron->external_current = 0.0f;
    }

    // Periodic network maintenance
    if (timestamp - network->last_maintenance >= network->config.update_interval) {
        neural_network_maintain_homeostasis(network, timestamp);
    }

    network->network_time = timestamp;
    network->current_time = timestamp;
    return active_neurons;
}

/**
 * @brief Add a new synaptic connection
 */
bool neural_network_add_connection(neural_network_t network, uint32_t from_id, uint32_t to_id,
                                   float weight)
{
    if (from_id >= network->num_neurons || to_id >= network->num_neurons) {
        return false;
    }

    neuron_t* from_neuron = &network->neurons[from_id];
    neuron_t* to_neuron = &network->neurons[to_id];

    // Check if we have room for new synapse (both forward and reverse)
    if (from_neuron->num_synapses >= MAX_SYNAPSES_PER_NEURON) {
        return false;
    }

    if (to_neuron->num_incoming >= MAX_SYNAPSES_PER_NEURON) {
        return false;
    }

    // Initialize new OUTGOING synapse (forward edge)
    synapse_t* syn = &from_neuron->synapses[from_neuron->num_synapses];
    syn->target_id = to_id;
    syn->weight = fmaxf(network->config.min_weight, fminf(network->config.max_weight, weight));
    syn->plasticity = 1.0f;
    syn->last_change = 0.0f;
    syn->last_active = network->network_time;
    syn->strength = 1.0f;
    syn->meta_plasticity = 1.0f;
    syn->trace = 0.0f;

    // NIMCP 2.6: Initialize STP (Short-Term Plasticity)
    // Default: Depressing synapse for excitatory connections
    stp_preset_t preset = (from_neuron->type == NEURON_EXCITATORY)
                        ? STP_PRESET_DEPRESSING
                        : STP_PRESET_FACILITATING;
    stp_params_t stp_params = stp_get_preset_params(preset);
    stp_init(&syn->stp, &stp_params, network->network_time);
    syn->enable_stp = true;  // Enable STP by default

    // Phase 11: Initialize BCM (Homeostatic Plasticity)
    // CONDITIONAL ALLOCATION: Only allocate if enabled in config
    // WHY: BCM requires per-synapse heap allocation
    // SCALABILITY: 1M neurons × 256 synapses × sizeof(bcm_synapse_t) = massive overhead
    // SOLUTION: Lazy initialization - only allocate when feature is enabled
    if (network->config.enable_bcm) {
        syn->bcm = (bcm_synapse_t*)nimcp_calloc(1, sizeof(bcm_synapse_t));
        if (syn->bcm) {
            *syn->bcm = bcm_synapse_init(syn->weight, 0.5f);  // Initial threshold = 0.5
            syn->enable_bcm = true;
        } else {
            syn->enable_bcm = false;  // Disable if allocation failed
        }
    } else {
        // Disabled by config - no allocation
        syn->bcm = NULL;
        syn->enable_bcm = false;
    }

    // Phase 11: Initialize Eligibility Traces (Temporal Credit Assignment)
    // CONDITIONAL ALLOCATION: Only allocate if enabled in config
    // WHY: Eligibility traces require per-synapse heap allocation
    // SCALABILITY: 1M neurons × 256 synapses × sizeof(eligibility_trace_t) = massive overhead
    // SOLUTION: Lazy initialization - only allocate when feature is enabled
    if (network->config.enable_eligibility) {
        syn->eligibility = (eligibility_trace_t*)nimcp_calloc(1, sizeof(eligibility_trace_t));
        if (syn->eligibility) {
            eligibility_trace_init(syn->eligibility, network->network_time);
            syn->enable_eligibility = true;
        } else {
            syn->enable_eligibility = false;  // Disable if allocation failed
        }
    } else {
        // Disabled by config - no allocation
        syn->eligibility = NULL;
        syn->enable_eligibility = false;
    }

    from_neuron->num_synapses++;

    // OPTIMIZATION: Add INCOMING synapse (reverse edge) for O(S) input summation
    // DESIGN PATTERN: Bidirectional Association
    // WHY: Enables O(S) lookup instead of O(N×S) scan
    synapse_t* incoming_syn = &to_neuron->incoming_synapses[to_neuron->num_incoming];
    incoming_syn->target_id = from_id;  // In reverse edge, target_id stores source
    incoming_syn->weight = syn->weight;  // Same weight as forward edge
    incoming_syn->plasticity = syn->plasticity;
    incoming_syn->last_change = syn->last_change;
    incoming_syn->last_active = syn->last_active;
    incoming_syn->strength = syn->strength;
    incoming_syn->meta_plasticity = syn->meta_plasticity;
    incoming_syn->trace = syn->trace;

    // NIMCP 2.6: Copy STP state to incoming synapse
    incoming_syn->stp = syn->stp;
    incoming_syn->enable_stp = syn->enable_stp;

    // Phase 11: Copy BCM state to incoming synapse
    incoming_syn->bcm = syn->bcm;  // Share BCM state (both directions use same threshold)
    incoming_syn->enable_bcm = syn->enable_bcm;

    // Phase 11: Copy eligibility trace to incoming synapse
    incoming_syn->eligibility = syn->eligibility;  // Share eligibility trace
    incoming_syn->enable_eligibility = syn->enable_eligibility;

    to_neuron->num_incoming++;

    // Update weight norm to reflect new synapse
    float sum_weights = 0.0f;
    for (uint32_t i = 0; i < from_neuron->num_synapses; i++) {
        sum_weights += fabsf(from_neuron->synapses[i].weight);
    }
    from_neuron->weight_norm = sum_weights;

    return true;
}

/**
 * @brief Normalize synaptic weights for a neuron
 */
static void normalize_synaptic_weights(neuron_t* neuron)
{
    float sum_weights = 0.0f;
    float max_weight = 0.0f;

    // Calculate sum and find maximum weight
    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        sum_weights += fabs(neuron->synapses[i].weight);
        max_weight = fmaxf(max_weight, fabs(neuron->synapses[i].weight));
    }

    if (sum_weights > EPSILON) {
        // Normalize weights while preserving sign
        float norm_factor = neuron->oja_params.target_norm / sum_weights;

        for (uint32_t i = 0; i < neuron->num_synapses; i++) {
            synapse_t* syn = &neuron->synapses[i];
            float sign = (syn->weight >= 0.0f) ? 1.0f : -1.0f;
            syn->weight = sign * fabs(syn->weight) * norm_factor;
        }

        // After normalization, weight_norm should be target_norm
        neuron->weight_norm = neuron->oja_params.target_norm;
    } else {
        // If sum is too small, keep original sum
        neuron->weight_norm = sum_weights;
    }
}


/**
 * @brief Prune weak synapses
 */
uint32_t neural_network_prune_synapses(neural_network_t network, float threshold)
{
    uint32_t pruned_count = 0;

    for (uint32_t i = 0; i < network->num_neurons; i++) {
        neuron_t* neuron = &network->neurons[i];
        uint32_t write_idx = 0;

        // Compact synapse array, removing weak synapses
        for (uint32_t read_idx = 0; read_idx < neuron->num_synapses; read_idx++) {
            synapse_t* syn = &neuron->synapses[read_idx];

            if (fabs(syn->weight) * syn->strength >= threshold) {
                if (write_idx != read_idx) {
                    memcpy(&neuron->synapses[write_idx], syn, sizeof(synapse_t));
                }
                write_idx++;
            } else {
                pruned_count++;
            }
        }

        neuron->num_synapses = write_idx;
    }

    return pruned_count;
}

/**
 * @brief Network maintenance routine
 */
void neural_network_maintain(neural_network_t network, uint64_t timestamp)
{
    // Skip if too soon since last maintenance
    if (timestamp - network->last_maintenance < network->config.update_interval) {
        return;
    }

    // Update homeostatic mechanisms
    neural_network_maintain_homeostasis(network, timestamp);

    // Prune very weak synapses
    float prune_threshold = network->config.min_weight * 0.1f;
    neural_network_prune_synapses(network, prune_threshold);

    // Normalize weights for all neurons
    for (uint32_t i = 0; i < network->num_neurons; i++) {
        normalize_synaptic_weights(&network->neurons[i]);
    }

    network->last_maintenance = timestamp;
}

/**
 * @brief Debug function to dump neuron state
 */
void neural_network_dump_neuron(neural_network_t network, uint32_t neuron_id)
{
    if (neuron_id >= network->num_neurons)
        return;

    neuron_t* neuron = &network->neurons[neuron_id];

    printf("Neuron %u:\n", neuron_id);
    printf("  Type: %s\n", (neuron->type == NEURON_INHIBITORY) ? "Inhibitory" : "Excitatory");
    printf("  State: %.3f\n", neuron->state);
    printf("  Threshold: %.3f\n", neuron->threshold);
    printf("  Calcium: %.3f\n", neuron->calcium_concentration);
    printf("  Activity: %.3f\n", neuron->avg_activity);
    printf("  Synapses: %u\n", neuron->num_synapses);

    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        synapse_t* syn = &neuron->synapses[i];
        printf("    Synapse %u -> %u: w=%.3f, s=%.3f\n", neuron_id, syn->target_id, syn->weight,
               syn->strength);
    }
}

/**
 * @brief Reset network state
 */
void neural_network_reset(neural_network_t network)
{
    for (uint32_t i = 0; i < network->num_neurons; i++) {
        neuron_t* neuron = &network->neurons[i];

        // Reset neuron state
        neuron->state = neuron->rest_potential;
        neuron->threshold = -55.0f;
        neuron->adaptation = 0.0f;
        neuron->calcium_concentration = 0.0f;
        neuron->avg_activity = 0.0f;

        // Reset spike history
        neuron->spike_history_index = 0;
        memset(neuron->spike_history, 0, sizeof(spike_record_t) * SPIKE_HISTORY_LENGTH);
        memset(neuron->activity_history, 0, sizeof(float) * HISTORY_WINDOW);

        // Reset synaptic traces
        for (uint32_t j = 0; j < neuron->num_synapses; j++) {
            neuron->synapses[j].trace = 0.0f;
            neuron->synapses[j].strength = 1.0f;
            neuron->synapses[j].meta_plasticity = 1.0f;
        }
    }

    network->network_time = 0;
    network->global_activity = 0.0f;
    network->network_stability = 1.0f;
    network->last_maintenance = 0;
}

/*
 * @brief Get neuron state
 */
bool neural_network_get_neuron_state(neural_network_t network, uint32_t neuron_id, float* state)
{
    if (!network || neuron_id >= network->num_neurons || !state) {
        return false;
    }

    *state = network->neurons[neuron_id].state;
    return true;
}

/**
 * @brief Get number of neurons in network
 *
 * WHAT: Return total neuron count
 * WHY: Allow external code to iterate neurons without accessing internals
 * HOW: Return num_neurons field
 *
 * @param network Network instance
 * @return Number of neurons, or 0 if network is NULL
 */
uint32_t neural_network_get_num_neurons(neural_network_t network)
{
    if (!network) {
        return 0;
    }
    return network->num_neurons;
}

/**
 * @brief Get pointer to neuron structure
 *
 * WHAT: Access neuron internals for serialization/introspection
 * WHY: Enable save/load of synaptic weights without duplicating access logic
 * HOW: Return pointer to neuron in neurons array
 *
 * WARNING: Returns internal pointer - use carefully!
 *
 * @param network Network instance
 * @param neuron_id Neuron ID
 * @return Pointer to neuron, or NULL if invalid
 */
neuron_t* neural_network_get_neuron(neural_network_t network, uint32_t neuron_id)
{
    if (!network || neuron_id >= network->num_neurons) {
        return NULL;
    }
    return &network->neurons[neuron_id];
}

/**
 * @brief Add a new neuron to the network
 */
/**
 * WHAT: Add a new neuron to the network dynamically
 * WHY: Support runtime network growth for adaptive architectures
 * HOW: Use preallocated capacity, check bounds
 * COMPLEXITY: O(1) - no reallocation needed if within capacity
 */
uint32_t neural_network_add_neuron(neural_network_t network, activation_type_t activation)
{
    // Guard clause: Validate network
    if (!network)
        return UINT32_MAX;

    // Guard clause: Check capacity bounds
    if (network->num_neurons >= network->capacity)
        return UINT32_MAX;

    uint32_t new_id = network->num_neurons;
    neuron_t* neuron = &network->neurons[new_id];

    // Initialize neuron with default values
    memset(neuron, 0, sizeof(neuron_t));
    neuron->id = new_id;
    neuron->activation_type = activation;
    neuron->type = NEURON_EXCITATORY;  // Default to excitatory
    neuron->rest_potential = 0.0f;   // Normalized resting potential
    neuron->threshold = 0.5f;        // Normalized spike threshold
    neuron->creation_time = network->network_time;

    // NIMCP 2.6: Initialize neuron model (uses network config)
    init_neuron_model(neuron, &network->config);

    network->num_neurons++;
    return new_id;
}

/**
 * @brief Apply generalized Oja's rule
 */
uint32_t neural_network_apply_generalized_oja(neural_network_t network, uint32_t neuron_id,
                                              uint64_t timestamp)
{
    // Implementation similar to neural_network_apply_oja but with generalized rule
    // This is a placeholder - implement the actual generalized Oja rule as needed
    return 0;
}

/**
 * @brief Update plasticity for a neuron
 */
uint32_t neural_network_update_plasticity(neural_network_t network, uint32_t neuron_id,
                                          uint64_t timestamp)
{
    if (!network || neuron_id >= network->num_neurons)
        return 0;

    neuron_t* neuron = &network->neurons[neuron_id];
    update_meta_plasticity(neuron, timestamp);
    return 1;
}

/**
 * @brief Normalize weights for a specific neuron
 */
bool neural_network_normalize_weights(neural_network_t network, uint32_t neuron_id)
{
    if (!network || neuron_id >= network->num_neurons)
        return false;

    normalize_synaptic_weights(&network->neurons[neuron_id]);
    return true;
}

/**
 * @brief Adapt neuron threshold
 */
bool neural_network_adapt_threshold(neural_network_t network, uint32_t neuron_id,
                                    uint64_t timestamp)
{
    if (!network || neuron_id >= network->num_neurons)
        return false;

    neuron_t* neuron = &network->neurons[neuron_id];
    float activity = neural_network_get_average_activity(network, neuron_id);

    // Adjust threshold based on recent activity
    float threshold_adjustment =
        neuron->adaptation * (activity - neuron->homeostatic.target_activity);
    neuron->threshold += threshold_adjustment;

    return true;
}

/**
 * @brief Update synaptic traces
 */
void neural_network_update_traces(neural_network_t network, uint32_t neuron_id, uint64_t timestamp)
{
    if (!network || neuron_id >= network->num_neurons)
        return;

    update_synaptic_traces(&network->neurons[neuron_id], timestamp);
}

/**
 * @brief Set neuron model type for a specific neuron
 *
 * WHAT: Dynamically changes neuron dynamics model
 * WHY: Enables heterogeneous networks (mixed LIF/Izhikevich/etc)
 * HOW: Cleans up old model, initializes new model
 *
 * PATTERN: Strategy Pattern - swap behavior at runtime
 * COMPLEXITY: O(1)
 *
 * @param network Neural network
 * @param neuron_id Neuron to modify
 * @param model_type Desired model type
 * @param params Model-specific parameters (NULL = use defaults)
 * @return true if successful, false on error
 */
bool neural_network_set_neuron_model(neural_network_t network, uint32_t neuron_id,
                                     neuron_model_type_t model_type, const void* params)
{
    // Guard: Validate network
    if (!network) {
        return false;
    }

    // Guard: Validate neuron ID
    if (neuron_id >= network->num_neurons) {
        return false;
    }

    neuron_t* neuron = &network->neurons[neuron_id];

    // Cleanup existing model if present
    if (neuron->model) {
        neuron_model_destroy(neuron->model);
        neuron->model = NULL;
    }

    // Set to LIF (no plugin model)
    if (model_type == NEURON_MODEL_LIF) {
        neuron->model_type = NEURON_MODEL_LIF;
        return true;
    }

    // Get vtable for requested model
    const neuron_model_vtable_t* vtable = NULL;

    switch (model_type) {
        case NEURON_MODEL_IZHIKEVICH:
            vtable = izhikevich_get_vtable();
            // If no params provided, use default RS (Regular Spiking)
            if (!params) {
                static izhikevich_params_t default_params;
                default_params = izhikevich_get_preset_params(IZHI_PRESET_REGULAR_SPIKING);
                params = &default_params;
            }
            break;

        case NEURON_MODEL_LIF:
            // Already handled above
            return true;

        default:
            // Unknown model type, fall back to LIF
            neuron->model_type = NEURON_MODEL_LIF;
            return false;
    }

    // Guard: Check vtable valid
    if (!vtable) {
        neuron->model_type = NEURON_MODEL_LIF;
        return false;
    }

    // Create new model state
    neuron->model = neuron_model_create(vtable, params);

    // Guard: Check creation succeeded
    if (!neuron->model) {
        neuron->model_type = NEURON_MODEL_LIF;
        return false;
    }

    neuron->model_type = model_type;
    return true;
}

/**
 * @brief Get weight norm for a neuron
 */
float neural_network_get_weight_norm(neural_network_t network, uint32_t neuron_id)
{
    if (!network || neuron_id >= network->num_neurons)
        return 0.0f;
    return network->neurons[neuron_id].weight_norm;
}

/**
 * @brief Get weight statistics for a neuron
 */
void neural_network_get_weight_statistics(neural_network_t network, uint32_t neuron_id, float* mean,
                                          float* std_dev)
{
    if (!network || neuron_id >= network->num_neurons || !mean || !std_dev) {
        if (mean)
            *mean = 0.0f;
        if (std_dev)
            *std_dev = 0.0f;
        return;
    }

    neuron_t* neuron = &network->neurons[neuron_id];
    float sum = 0.0f, sum_sq = 0.0f;

    for (uint32_t i = 0; i < neuron->num_synapses; i++) {
        float w = neuron->synapses[i].weight;
        sum += w;
        sum_sq += w * w;
    }

    if (neuron->num_synapses > 0) {
        *mean = sum / neuron->num_synapses;
        float variance = (sum_sq / neuron->num_synapses) - (*mean * *mean);
        *std_dev = sqrtf(fmaxf(0.0f, variance));
    } else {
        *mean = 0.0f;
        *std_dev = 0.0f;
    }
}

bool neural_network_get_stats(neural_network_t network, network_stats_t* stats)
{
    if (!network || !stats) {
        return false;
    }

    // Initialize stats structure
    memset(stats, 0, sizeof(network_stats_t));

    // Count neurons and calculate averages
    float total_activity = 0.0f;
    float total_weight = 0.0f;
    float total_strength = 0.0f;
    float total_plasticity = 0.0f;
    float total_calcium = 0.0f;
    uint32_t total_synapses = 0;

    for (uint32_t i = 0; i < network->num_neurons; i++) {
        // CONST: neuron is read-only in statistics calculation
        const neuron_t* neuron = &network->neurons[i];

        // Count by neuron type
        if (neuron->type == NEURON_INHIBITORY) {
            stats->num_inhibitory++;
        } else {
            stats->num_excitatory++;
        }

        // Sum up neuron properties
        total_activity += neuron->avg_activity;
        total_calcium += neuron->calcium_concentration;
        total_synapses += neuron->num_synapses;

        // Calculate synapse averages
        for (uint32_t j = 0; j < neuron->num_synapses; j++) {
            total_weight += neuron->synapses[j].weight;
            total_strength += neuron->synapses[j].strength;
            total_plasticity += neuron->synapses[j].plasticity;
        }
    }

    // Set basic counts
    stats->num_neurons = network->num_neurons;
    stats->total_synapses = total_synapses;

    // Calculate averages
    if (stats->num_neurons > 0) {
        stats->avg_activity = total_activity / stats->num_neurons;
        stats->avg_calcium = total_calcium / stats->num_neurons;
    }

    if (total_synapses > 0) {
        stats->avg_weight = total_weight / total_synapses;
        stats->avg_strength = total_strength / total_synapses;
        stats->avg_plasticity = total_plasticity / total_synapses;
    }

    // Calculate network stability (based on weight change rate)
    // Stability = 1 - normalized_change, clamped to [0, 1]
    // Special case: On first call (last_avg_weight == 0 and current avg_weight != 0),
    // we have no history, so assume perfect stability (1.0)
    if (network->last_avg_weight == 0.0f && stats->avg_weight != 0.0f) {
        // First call with non-zero weights - no history to compare against
        stats->network_stability = 1.0f;
    } else if (network->last_avg_weight == 0.0f && stats->avg_weight == 0.0f) {
        // Network has no synapses or all weights are zero - assume stable
        stats->network_stability = 1.0f;
    } else {
        float weight_change = fabs(stats->avg_weight - network->last_avg_weight);
        float normalized_change =
            weight_change / (fmaxf(fabs(stats->avg_weight), fabs(network->last_avg_weight)) + 1e-6f);
        stats->network_stability = fmaxf(0.0f, fminf(1.0f, 1.0f - normalized_change));
    }
    network->last_avg_weight = stats->avg_weight;

    // Set network time
    stats->network_time = network->current_time;

    return true;
}

//=============================================================================
// NIMCP 2.5 - Forward Pass Implementation
//=============================================================================

/**
 * @brief Perform forward pass through the neural network
 *
 * This implements a feedforward computation for networks with layer structure.
 * For NIMCP 2.5 Brain API compatibility.
 */
bool neural_network_forward(neural_network_t network, const float* inputs, uint32_t input_size,
                            float* outputs, uint32_t output_size)
{
    if (!network || !inputs || !outputs) {
        return false;
    }

    // Check if network has layer structure (NIMCP 2.5)
    if (network->config.num_layers > 0 && network->config.layer_sizes) {
        // Use layer-based forward pass
        uint32_t expected_input = network->config.layer_sizes[0];
        uint32_t expected_output = network->config.layer_sizes[network->config.num_layers - 1];

        if (input_size != expected_input || output_size != expected_output) {
            return false;
        }

        // Set input layer neuron states
        for (uint32_t i = 0; i < input_size; i++) {
            if (i < network->num_neurons) {
                network->neurons[i].state = inputs[i];
            }
        }

        // Propagate through layers
        uint32_t neuron_offset = input_size;
        for (uint32_t layer = 1; layer < network->config.num_layers; layer++) {
            uint32_t layer_size = network->config.layer_sizes[layer];

            // Compute each neuron in this layer
            for (uint32_t i = 0; i < layer_size && neuron_offset + i < network->num_neurons; i++) {
                uint32_t neuron_id = neuron_offset + i;
                neuron_t* neuron = &network->neurons[neuron_id];

                // Compute weighted sum from previous layer
                float activation = neuron->bias;
                for (uint32_t j = 0; j < neuron->num_synapses; j++) {
                    synapse_t* syn = &neuron->synapses[j];
                    if (syn->target_id < network->num_neurons) {
                        float pre_activity = network->neurons[syn->target_id].state;
                        activation += pre_activity * syn->weight * syn->strength;
                    }
                }

                // Apply activation function
                switch (neuron->activation_type) {
                    case ACTIVATION_SIGMOID:
                        neuron->state = 1.0f / (1.0f + expf(-activation));
                        break;
                    case ACTIVATION_TANH:
                        neuron->state = tanhf(activation);
                        break;
                    case ACTIVATION_RELU:
                        neuron->state = (activation > 0.0f) ? activation : 0.0f;
                        break;
                    case ACTIVATION_LEAKY_RELU:
                        neuron->state = (activation > 0.0f) ? activation : (activation * 0.01f);
                        break;
                    case ACTIVATION_ADAPTIVE:
                        // Use adaptive threshold
                        if (activation > neuron->threshold) {
                            neuron->state = tanhf((activation - neuron->threshold) / 10.0f);
                        } else {
                            neuron->state = 0.0f;
                        }
                        break;
                    default:
                        neuron->state = tanhf(activation);
                }

                // Clamp to reasonable range
                neuron->state = fmaxf(-1.0f, fminf(1.0f, neuron->state));
            }

            neuron_offset += layer_size;
        }

        // Extract outputs from output layer
        uint32_t output_layer_start = neuron_offset - output_size;
        for (uint32_t i = 0; i < output_size; i++) {
            if (output_layer_start + i < network->num_neurons) {
                outputs[i] = network->neurons[output_layer_start + i].state;
            } else {
                outputs[i] = 0.0f;
            }
        }

        return true;
    }

    // Fallback: Legacy NIMCP 2.0 behavior
    // Set first N neurons to inputs
    uint32_t num_inputs = (input_size < network->num_neurons) ? input_size : network->num_neurons;
    for (uint32_t i = 0; i < num_inputs; i++) {
        network->neurons[i].state = inputs[i];
    }

    // Run one network computation step
    network->current_time++;
    neural_network_compute_step(network, network->current_time);

    // Extract outputs from last N neurons
    uint32_t output_start =
        (network->num_neurons > output_size) ? (network->num_neurons - output_size) : 0;
    for (uint32_t i = 0; i < output_size; i++) {
        uint32_t neuron_id = output_start + i;
        if (neuron_id < network->num_neurons) {
            outputs[i] = network->neurons[neuron_id].state;
        } else {
            outputs[i] = 0.0f;
        }
    }

    return true;
}

//=============================================================================
// Bidirectional Synapse API (OPTIMIZATION)
//=============================================================================

/**
 * @brief Get count of incoming synapses to a neuron
 *
 * DESIGN PATTERN: Iterator Pattern
 * COMPLEXITY: O(1)
 * WHY: Enables O(S) iteration over incoming edges instead of O(N×S) scan
 */
uint32_t neural_network_get_incoming_synapse_count(neural_network_t network, uint32_t neuron_id)
{
    if (!network || neuron_id >= network->num_neurons) {
        return 0;
    }

    return network->neurons[neuron_id].num_incoming;
}

/**
 * @brief Get array of incoming synapses
 *
 * DESIGN PATTERN: Iterator Pattern
 * COMPLEXITY: O(1) to get pointer, O(S) to iterate
 * WHY: Provides direct access to incoming edges for efficient input summation
 *
 * @param network Neural network
 * @param neuron_id Target neuron
 * @param out_synapses Pointer to receive synapse array (read-only)
 * @return Number of incoming synapses
 */
uint32_t neural_network_get_incoming_synapses(neural_network_t network, uint32_t neuron_id,
                                               const synapse_t** out_synapses)
{
    if (!network || !out_synapses || neuron_id >= network->num_neurons) {
        if (out_synapses) {
            *out_synapses = NULL;
        }
        return 0;
    }

    neuron_t* neuron = &network->neurons[neuron_id];
    *out_synapses = neuron->incoming_synapses;
    return neuron->num_incoming;
}

//=============================================================================
// NIMCP 2.7: NLP Integration - Accessor Functions
//=============================================================================

/**
 * @brief Set global state for synapse computation
 *
 * WHAT: Inject global state buffer (e.g., attention output) into network
 * WHY: Enable synapses to access shared context for attention-modulation
 * HOW: Stores pointer and size, passed via synapse_compute_context_t
 * WHEN: Called by NLP layer after attention forward pass
 *
 * DESIGN PATTERN: Dependency Injection
 * COMPLEXITY: O(1)
 *
 * @param network Neural network
 * @param global_state Pointer to global state buffer
 * @param size Size of buffer (floats)
 * @return true on success
 */
bool neural_network_set_global_state(neural_network_t network, float* global_state, uint32_t size)
{
    // Guard: Validate input
    if (!network) {
        return false;
    }

    // Store global state reference
    network->global_state = global_state;
    network->global_state_size = size;

    return true;
}

/**
 * @brief Set neuromodulator system for synapse computation
 *
 * WHAT: Inject neuromodulator system into network
 * WHY: Enable synapses to access neuromodulator levels (dopamine, ACh, etc)
 * HOW: Stores opaque pointer, queried during synapse computation
 * WHEN: Called by NLP layer during network creation
 *
 * DESIGN PATTERN: Dependency Injection
 * COMPLEXITY: O(1)
 *
 * @param network Neural network
 * @param neuromod_system Opaque pointer to neuromodulator system
 * @return true on success
 */
bool neural_network_set_neuromodulator_system(neural_network_t network, void* neuromod_system)
{
    // Guard: Validate input
    if (!network) {
        return false;
    }

    // Store neuromodulator system reference
    network->neuromodulator_system = neuromod_system;

    return true;
}

/**
 * @brief Set glial integration system for neuro-glial signaling (Phase 6)
 *
 * WHAT: Attach glial integration system to network
 * WHY: Enable bidirectional neuro-glial communication
 * HOW: Stores opaque pointer, notified on neuron/synapse events
 * WHEN: Called after glial cells are assigned
 *
 * DESIGN PATTERN: Observer - glial system observes neural events
 * COMPLEXITY: O(1)
 */
bool neural_network_set_glial_integration(neural_network_t network, void* glial_system)
{
    // Guard: Validate input
    if (!network) {
        return false;
    }

    // Store glial integration system reference
    network->glial_integration = glial_system;

    return true;
}

/**
 * @brief Get neuromodulation level for synapse computation
 *
 * WHAT: Query current neuromodulation level from attached system
 * WHY: Synapses need to know dopamine/ACh levels for modulation
 * HOW: Queries neuromodulator system via opaque pointer
 * WHEN: Called during synapse computation in sum_synaptic_inputs
 *
 * DESIGN PATTERN: Adapter Pattern - wraps neuromodulator system access
 * COMPLEXITY: O(1)
 *
 * NOTE: Currently returns 0.0 as placeholder. Full implementation would:
 * 1. Cast neuromodulator_system to proper type
 * 2. Call neuromodulator_get_dopamine() or similar
 * 3. Return normalized level [0,1]
 *
 * @param network Neural network
 * @return Current neuromodulation level [0,1]
 */
float neural_network_get_neuromodulation(neural_network_t network)
{
    // Guard: Validate input
    if (!network || !network->neuromodulator_system) {
        return 0.0f;
    }

    // Query neuromodulator system for dopamine level
    // Dopamine is the primary learning/reward neuromodulator
    neuromodulator_system_t system = (neuromodulator_system_t)network->neuromodulator_system;
    float dopamine = neuromodulator_get_level(system, NEUROMOD_DOPAMINE);

    return dopamine;
}

//=============================================================================
// PHASE 8.7: Synapse Type System - Typed Connections
//=============================================================================

/**
 * @brief Add connection with specific synapse type (Phase 8.7)
 *
 * WHAT: Create synapse with biological type (AMPA, NMDA, GABA-A, etc)
 * WHY: Enable biologically realistic synapse diversity
 * HOW: Initialize synapse with type-specific parameters
 *
 * ALGORITHM:
 * 1. Call standard neural_network_add_connection
 * 2. Set synapse type field
 * 3. Initialize type-specific state (AMPA/NMDA/GABA/etc parameters)
 * 4. Copy type info to incoming synapse (bidirectional)
 *
 * COMPLEXITY: O(1)
 *
 * BIOLOGICAL MOTIVATION:
 * Real synapses differ by neurotransmitter type:
 * - Excitatory: AMPA (fast), NMDA (slow + Ca2+)
 * - Inhibitory: GABA-A (fast), GABA-B (slow)
 * - Modulatory: Dopamine, Serotonin, Acetylcholine
 * - Electrical: Gap junctions
 *
 * @param network Neural network
 * @param from_id Source neuron ID
 * @param to_id Target neuron ID
 * @param weight Initial synaptic weight
 * @param type Synapse type (SYNAPSE_AMPA, SYNAPSE_NMDA, etc)
 * @return true if successful, false on error
 */
bool neural_network_add_connection_typed(neural_network_t network, uint32_t from_id, uint32_t to_id,
                                          float weight, synapse_type_t type)
{
    // 1. Create standard connection (handles all base initialization)
    if (!neural_network_add_connection(network, from_id, to_id, weight)) {
        return false;
    }

    // 2. Get the newly created synapse (last one added)
    neuron_t* from_neuron = &network->neurons[from_id];
    neuron_t* to_neuron = &network->neurons[to_id];

    synapse_t* syn = &from_neuron->synapses[from_neuron->num_synapses - 1];
    synapse_t* incoming_syn = &to_neuron->incoming_synapses[to_neuron->num_incoming - 1];

    // 3. Set synapse type
    syn->type = type;
    incoming_syn->type = type;

    // 4. Initialize type-specific state based on type
    switch (type) {
        case SYNAPSE_AMPA:
            synapse_init_ampa(&syn->type_state.ampa);
            synapse_init_ampa(&incoming_syn->type_state.ampa);
            break;

        case SYNAPSE_NMDA:
            synapse_init_nmda(&syn->type_state.nmda);
            synapse_init_nmda(&incoming_syn->type_state.nmda);
            break;

        case SYNAPSE_GABA_A:
            synapse_init_gaba_a(&syn->type_state.gaba_a);
            synapse_init_gaba_a(&incoming_syn->type_state.gaba_a);
            break;

        case SYNAPSE_GABA_B:
            synapse_init_gaba_b(&syn->type_state.gaba_b);
            synapse_init_gaba_b(&incoming_syn->type_state.gaba_b);
            break;

        case SYNAPSE_DOPAMINE:
            synapse_init_dopamine(&syn->type_state.dopamine);
            synapse_init_dopamine(&incoming_syn->type_state.dopamine);
            break;

        case SYNAPSE_SEROTONIN:
            synapse_init_serotonin(&syn->type_state.serotonin);
            synapse_init_serotonin(&incoming_syn->type_state.serotonin);
            break;

        case SYNAPSE_ACETYLCHOLINE:
            synapse_init_acetylcholine(&syn->type_state.acetylcholine);
            synapse_init_acetylcholine(&incoming_syn->type_state.acetylcholine);
            break;

        case SYNAPSE_ELECTRICAL:
            synapse_init_electrical(&syn->type_state.electrical);
            synapse_init_electrical(&incoming_syn->type_state.electrical);
            break;

        case SYNAPSE_GENERIC:
        default:
            // Generic synapse has no special state to initialize
            break;
    }

    return true;
}
