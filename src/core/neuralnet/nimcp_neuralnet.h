// nimcp_neuralnet.h

#ifndef NIMCP_NEURALNET_H
#define NIMCP_NEURALNET_H

#include "common/nimcp_export.h"
#include "core/neuron_models/nimcp_neuron_model.h"
#include "core/neuron_types/nimcp_neuron_types.h"  // Phase 8.7: Specialized neuron types
#include "plasticity/stp/nimcp_stp.h"
#include "plasticity/bcm/nimcp_bcm.h"  // Phase 11: BCM homeostatic plasticity

// Forward declare neuron_t so we can use it in function pointers before it's fully defined
// NOTE: neuron_t is defined below as typedef struct { ... } neuron_t;
typedef struct neuron_struct neuron_t;

// Forward declarations for programmable synapse types (NIMCP 2.7)
// WHY: Avoid circular dependency with synapse_compute.h
// PATTERN: Use incomplete types, full definitions in synapse_compute.h
// HOW: Pointers to these types are valid, but struct contents are opaque here
struct synapse_compute_context_t;
struct synapse_compute_state_t;

// Forward declaration for eligibility traces (Phase 11)
// WHY: synapse_t uses eligibility_trace_t*, need forward declaration before synapse_t
// Full definition included after synapse_t to avoid circular dependency
typedef struct eligibility_trace_t eligibility_trace_t;

// Phase 8.7: Include synapse type system
// CRITICAL: Must come AFTER neuron_t forward declaration
#include "core/synapse_types/nimcp_synapse_types.h"

// Phase 11: BCM plasticity types
// NOTE: Must come AFTER neuron_t forward declaration
#include "plasticity/bcm/nimcp_bcm.h"

// Function pointer types for synapse computation (NIMCP 2.7)
// DESIGN: Define types here so synapse_t can use them without including synapse_compute.h
// NOTE: We use neuron_t* (typedef) not struct neuron_t* (which doesn't exist yet)
typedef float (*synapse_compute_fn)(
    struct synapse_t* syn,
    const neuron_t* pre_neuron,
    const neuron_t* post_neuron,
    float pre_activity,
    struct synapse_compute_context_t* context
);
typedef void (*synapse_learn_fn)(
    struct synapse_t* syn,
    const neuron_t* pre_neuron,
    const neuron_t* post_neuron,
    float pre_spike_time,
    float post_spike_time,
    float reward_signal,
    struct synapse_compute_context_t* context
);

/**
 * @file nimcp_neuralnet.h
 * @brief Neural network implementation with biological learning mechanisms
 *
 * This module implements a biologically-inspired neural network with:
 * - Hebbian learning (STDP)
 * - Oja's learning rule for PCA
 * - Synaptic plasticity and scaling
 * - Homeostatic plasticity
 * - Excitatory/Inhibitory balance
 * - Activity-dependent plasticity
 * - Adaptive thresholding
 *
 * @author Your Name
 * @date 2025-02-04
 */

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Network size and capacity constants
 *
 * WHAT: Maximum network capacity and per-neuron limits
 * WHY: Support BRAIN_SIZE_LARGE (100K neurons) while preventing unbounded allocation
 * HOW: Neurons array is dynamically allocated up to MAX_NEURONS
 */
#define MAX_NEURONS 100000  // Support BRAIN_SIZE_LARGE (100K neurons)
#define MAX_SYNAPSES_PER_NEURON 256
#define SPIKE_HISTORY_LENGTH 1000
#define HISTORY_WINDOW 100
#define WEIGHT_UPDATE_THRESHOLD 1e-6f
#define ACTIVITY_THRESHOLD 1e-5f
#define NORM_THRESHOLD 1e-7f
#define NORMALIZATION_INTERVAL 1000  // ms

// Phase 8.7: neuron_type_t now defined in nimcp_neuron_types.h
// This provides backward compatibility with NEURON_EXCITATORY/INHIBITORY
// plus new specialized types (V1 simple/complex, A1 frequency-tuned, etc.)

/**
 * @brief Learning rule types
 */
typedef enum {
    LEARNING_HEBBIAN,         /**< Basic Hebbian learning */
    LEARNING_OJA,             /**< Oja's learning rule */
    LEARNING_GENERALIZED_OJA, /**< Generalized Oja's rule */
    LEARNING_STDP,            /**< Spike-timing dependent plasticity */
    LEARNING_HYBRID           /**< Combined learning rules */
} learning_rule_t;

/**
 * @brief Activation function types
 */
typedef enum {
    ACTIVATION_SIGMOID,    /**< Sigmoid activation function */
    ACTIVATION_TANH,       /**< Hyperbolic tangent activation */
    ACTIVATION_RELU,       /**< Rectified Linear Unit */
    ACTIVATION_LEAKY_RELU, /**< Leaky Rectified Linear Unit */
    ACTIVATION_ADAPTIVE    /**< Adaptive threshold function */
} activation_type_t;

/**
 * @brief Network statistics structure
 */
typedef struct {
    uint32_t num_neurons;
    uint32_t num_inhibitory;
    uint32_t num_excitatory;
    uint32_t total_synapses;
    float avg_activity;
    float avg_weight;
    float avg_strength;
    float avg_plasticity;
    float avg_calcium;
    float network_stability;
    uint64_t network_time;
} network_stats_t;

/**
 * @brief Spike timing record structure
 */
typedef struct {
    uint64_t timestamp; /**< Spike timestamp */
    float magnitude;    /**< Spike magnitude */
} spike_record_t;

/**
 * @brief Oja learning parameters
 */
typedef struct {
    float alpha;         /**< Learning rate parameter */
    float forgetting;    /**< Forgetting/normalization factor */
    float stabilization; /**< Stabilization parameter */
    float target_norm;   /**< Target norm for weights */
} oja_params_t;

/**
 * @brief STDP parameters
 */
typedef struct {
    float learning_rate;   /**< STDP learning rate */
    float time_window;     /**< Time window for STDP (ms) */
    float positive_factor; /**< Factor for potentiation */
    float negative_factor; /**< Factor for depression */
} stdp_params_t;

/**
 * @brief Homeostatic plasticity parameters
 */
typedef struct {
    float target_activity; /**< Target average activity */
    float time_scale;      /**< Time scale for averaging */
    float strength;        /**< Strength of homeostatic adjustment */
} homeostatic_params_t;

/**
 * @brief Synapse structure (NIMCP 2.7: Now with programmable computation!)
 *
 * DESIGN EVOLUTION:
 * - NIMCP 2.0-2.5: Synapse = weight
 * - NIMCP 2.6: Synapse = weight + STP state
 * - NIMCP 2.7: Synapse = weight + STP + COMPUTATION
 * - NIMCP 2.8.7 (Phase 8.7): Synapse = weight + STP + COMPUTATION + TYPE
 *
 * This makes synapses first-class computational units with biological type diversity.
 */
typedef struct synapse_t {
    uint32_t target_id;    /**< Target neuron ID */
    float weight;          /**< Synaptic weight */
    float plasticity;      /**< Plasticity coefficient */
    float last_change;     /**< Last weight change (for momentum) */
    uint64_t last_active;  /**< Last activation timestamp */
    float strength;        /**< Synaptic strength (separate from weight) */
    float meta_plasticity; /**< Meta-plasticity factor */
    float trace;           /**< Synaptic trace for STDP */

    // Short-term plasticity (NIMCP 2.6)
    stp_state_t stp;       /**< Short-term plasticity state */
    bool enable_stp;       /**< Enable STP for this synapse */

    // BCM homeostatic plasticity (Phase 11: Plasticity Wiring)
    bcm_synapse_t* bcm;  /**< BCM sliding threshold state (NULL = disabled) */
    bool enable_bcm;     /**< Enable BCM for this synapse */

    // Eligibility traces (Phase 11: Plasticity Wiring)
    eligibility_trace_t* eligibility;  /**< Eligibility trace for RL (NULL = disabled) */
    bool enable_eligibility;           /**< Enable eligibility traces for this synapse */

    // Programmable computation (NIMCP 2.7) - MAJOR FEATURE
    synapse_compute_fn compute_function;  /**< Custom computation (NULL = default) */
    synapse_learn_fn learn_function;      /**< Custom learning (NULL = default) */
    struct synapse_compute_state_t* compute_state; /**< Function-specific state (NULL = none) */

    // Synapse type system (NIMCP 2.8.7 / Phase 8.7) - BIOLOGICAL DIVERSITY
    synapse_type_t type;           /**< Synapse type (AMPA, NMDA, GABA-A, etc) */
    synapse_type_state_t type_state; /**< Type-specific state (conductance, modulation, etc) */

    // ENHANCEMENT 1: Semantic Embeddings (NIMCP 2.9)
    float *semantic_embedding;     /**< Semantic vector [embedding_dim] */
    uint16_t embedding_dim;        /**< Embedding dimension (0 = no embedding, typically 128-512) */
    float semantic_relevance;      /**< Cached relevance score (0-1) for current context */

    // PERFORMANCE NOTE: Function pointers add 24 bytes per synapse (on 64-bit)
    // Type system adds ~40 bytes per synapse (type enum + union state)
    // Semantic embeddings add ~512 bytes per synapse (128D * 4 bytes)
    // For 100K synapses: 2.4 MB (functions) + 4.0 MB (types) + 51.2 MB (embeddings) = 57.6 MB overhead
    // Total synapse size: ~600 bytes/synapse with embeddings. Cost justified for intelligent routing.
} synapse_t;

// Phase 11: Eligibility traces for temporal credit assignment
// NOTE: Must come AFTER synapse_t definition (eligibility functions modify synapse->weight)
#include "plasticity/eligibility/nimcp_eligibility_trace.h"

/**
 * @brief Neuron structure
 *
 * NIMCP 2.7: Changed from anonymous struct to named struct (neuron_struct)
 * WHY: Enables forward declaration for use in synapse compute function pointers
 */
typedef struct neuron_struct {
    uint32_t id;             /**< Unique neuron identifier */
    neuron_type_t type;      /**< Excitatory or Inhibitory */
    float state;             /**< Current activation state */
    float rest_potential;    /**< Resting membrane potential */
    float threshold;         /**< Firing threshold */
    float adaptation;        /**< Adaptive threshold parameter */
    float refractory_period; /**< Refractory period (ms) */
    float bias;              /**< Neuron bias value */
    float external_current;  /**< External input current (e.g., from spike encoding) */

    // Learning parameters
    learning_rule_t learning_rule;     /**< Active learning rule */
    activation_type_t activation_type; /**< Activation function type */
    oja_params_t oja_params;           /**< Oja learning parameters */
    stdp_params_t stdp_params;         /**< STDP parameters */
    homeostatic_params_t homeostatic;  /**< Homeostatic parameters */

    // Synaptic connections (outgoing edges)
    synapse_t synapses[MAX_SYNAPSES_PER_NEURON];
    uint32_t num_synapses;

    // Bidirectional tracking: incoming synapses (OPTIMIZATION for O(S) input summation)
    // DESIGN PATTERN: Bidirectional Association
    // WHY: Eliminates O(N×S) scan in sum_synaptic_inputs, now O(S)
    synapse_t incoming_synapses[MAX_SYNAPSES_PER_NEURON];
    uint32_t num_incoming;

    // Plasticity parameters
    float plasticity_rate;       /**< Learning rate for plasticity */
    float homeostatic_factor;    /**< Homeostatic plasticity factor */
    float calcium_concentration; /**< Calcium concentration (for STDP) */
    float weight_norm;           /**< Current norm of weights */

    // Activity tracking
    spike_record_t spike_history[SPIKE_HISTORY_LENGTH];
    uint32_t spike_history_index;
    float activity_history[HISTORY_WINDOW];
    float avg_activity;     /**< Average activity level */
    uint64_t last_spike;    /**< Last spike timestamp */
    uint64_t last_update;   /**< Last state update timestamp */
    uint64_t creation_time; /**< Neuron creation timestamp */

    // Neuron model - Plugin architecture for LIF/Izhikevich/etc
    neuron_model_state_t model;  /**< Neuron dynamics model (NULL = legacy LIF) */
    neuron_model_type_t model_type; /**< Type of model being used */

    // Phase 8.7: Type-specific parameters for specialized neurons
    // Forward declaration from nimcp_neuron_types.h
    void* type_params;  /**< Type-specific parameters (neuron_type_params_t*) */
} neuron_t;

/**
 * @brief Network configuration parameters
 */
typedef struct {
    // Original NIMCP 2.0 fields
    uint32_t num_neurons;     /**< Initial number of neurons */
    float ei_ratio;           /**< Excitatory/Inhibitory ratio */
    float learning_rate;      /**< Base learning rate */
    float hebbian_rate;       /**< Hebbian learning rate */
    float stdp_window;        /**< STDP time window (ms) */
    float homeostatic_rate;   /**< Homeostatic plasticity rate */
    float target_activity;    /**< Target activity level */
    float adaptation_rate;    /**< Threshold adaptation rate */
    float refractory_period;  /**< Base refractory period (ms) */
    float min_weight;         /**< Minimum synaptic weight */
    float max_weight;         /**< Maximum synaptic weight */
    uint32_t update_interval; /**< Minimum update interval (ms) */

    // NIMCP 2.5 Brain API extensions
    uint32_t input_size;     /**< Number of input neurons */
    uint32_t output_size;    /**< Number of output neurons */
    uint32_t num_layers;     /**< Number of layers */
    uint32_t* layer_sizes;   /**< Size of each layer */
    bool enable_stdp;        /**< Enable STDP learning */
    bool enable_hebbian;     /**< Enable Hebbian learning */
    bool enable_oja;         /**< Enable Oja's rule */
    bool enable_homeostasis; /**< Enable homeostatic plasticity */

    // NIMCP 2.6 Neuron Model Extensions
    neuron_model_type_t neuron_model;  /**< Neuron dynamics model (LIF, Izhikevich, etc) */
    const void* model_params;          /**< Model-specific parameters (izhikevich_params_t*, etc) */

    // Part A1.1: ODE Integration Method (RK4 Support)
    ode_integration_method_t integration_method;  /**< ODE integration algorithm (Euler, RK4) */

    // Advanced Plasticity Controls (conditional allocation for scalability)
    bool enable_bcm;                   /**< Enable BCM plasticity allocation */
    bool enable_eligibility;           /**< Enable eligibility trace allocation */
} network_config_t;

/**
 * @brief Neural network structure
 */
typedef struct neural_network_struct* neural_network_t;

// Core network functions
neural_network_t neural_network_create(const network_config_t* config);
void neural_network_destroy(neural_network_t network);
bool neural_network_update_neuron(neural_network_t network, uint32_t neuron_id, float input_current,
                                  uint64_t timestamp);
bool neural_network_get_neuron_state(neural_network_t network, uint32_t neuron_id, float* state);

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
uint32_t neural_network_get_num_neurons(neural_network_t network);

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
neuron_t* neural_network_get_neuron(neural_network_t network, uint32_t neuron_id);

// NIMCP 2.5 - Forward pass for inference
bool neural_network_forward(neural_network_t network, const float* inputs, uint32_t input_size,
                            float* outputs, uint32_t output_size);
uint32_t neural_network_add_neuron(neural_network_t network, activation_type_t activation);
bool neural_network_add_connection(neural_network_t network, uint32_t from_id, uint32_t to_id,
                                   float weight);

/**
 * @brief Add connection with specific synapse type (Phase 8.7)
 *
 * WHAT: Create synapse with biological type (AMPA, NMDA, GABA-A, etc)
 * WHY: Enable biologically realistic synapse diversity
 * HOW: Initialize synapse with type-specific parameters
 *
 * COMPLEXITY: O(1)
 *
 * @param network Neural network
 * @param from_id Source neuron ID
 * @param to_id Target neuron ID
 * @param weight Initial synaptic weight
 * @param type Synapse type (SYNAPSE_AMPA, SYNAPSE_NMDA, etc)
 * @return true if successful, false on error
 */
bool neural_network_add_connection_typed(neural_network_t network, uint32_t from_id, uint32_t to_id,
                                          float weight, synapse_type_t type);
uint32_t neural_network_compute_step(neural_network_t network, uint64_t timestamp);

// Timing functions
void neural_network_set_time(neural_network_t network, uint64_t timestamp);

// Learning and plasticity functions
uint32_t neural_network_apply_oja(neural_network_t network, uint32_t neuron_id, uint64_t timestamp);
uint32_t neural_network_apply_generalized_oja(neural_network_t network, uint32_t neuron_id,
                                              uint64_t timestamp);
uint32_t neural_network_apply_stdp(neural_network_t network, uint32_t neuron_id,
                                   uint64_t timestamp);
bool neural_network_apply_homeostasis(neural_network_t network, uint32_t neuron_id,
                                      uint64_t timestamp);
uint32_t neural_network_update_plasticity(neural_network_t network, uint32_t neuron_id,
                                          uint64_t timestamp);
bool neural_network_normalize_weights(neural_network_t network, uint32_t neuron_id);

// Activity and state functions
bool neural_network_record_spike(neural_network_t network, uint32_t neuron_id, float magnitude,
                                 uint64_t timestamp);
bool neural_network_adapt_threshold(neural_network_t network, uint32_t neuron_id,
                                    uint64_t timestamp);
float neural_network_compute_activation(neuron_t* neuron, float input);
void neural_network_update_traces(neural_network_t network, uint32_t neuron_id, uint64_t timestamp);

/**
 * @brief Apply reward-modulated learning to all synapses (Phase 11)
 *
 * WHAT: Apply biological plasticity (STDP/BCM/eligibility) with reward signal
 * WHY:  Enable biological learning in supervised/RL contexts
 * HOW:  Iterate all neurons → Apply STDP → Apply eligibility traces → Apply BCM
 *
 * COMPLEXITY: O(N×S) where N = neurons, S = avg synapses per neuron
 *
 * @param network Neural network
 * @param reward Reward signal [0, 1]
 * @param learning_rate Learning rate for weight updates
 * @param current_time Current timestamp
 * @return Number of synapses modified
 */
uint32_t neural_network_apply_reward_learning(neural_network_t network, float reward,
                                              float learning_rate, uint64_t current_time);

// Analysis and monitoring functions
float neural_network_get_average_activity(neural_network_t network, uint32_t neuron_id);
float neural_network_get_weight_norm(neural_network_t network, uint32_t neuron_id);
void neural_network_get_weight_statistics(neural_network_t network, uint32_t neuron_id, float* mean,
                                          float* std_dev);

// NIMCP 2.6: Neuron model configuration
/**
 * @brief Set neuron model type for a specific neuron
 *
 * WHAT: Changes the dynamics model of an existing neuron
 * WHY: Enables heterogeneous networks with mixed neuron types
 * HOW: Cleans up old model, creates new model with specified type
 *
 * COMPLEXITY: O(1)
 *
 * @param network Neural network
 * @param neuron_id Neuron to modify
 * @param model_type Desired model type (NEURON_MODEL_LIF, NEURON_MODEL_IZHIKEVICH, etc)
 * @param params Model-specific parameters (NULL = use defaults for that model)
 * @return true if successful, false on error
 */
bool neural_network_set_neuron_model(neural_network_t network, uint32_t neuron_id,
                                     neuron_model_type_t model_type, const void* params);

/**
 * @brief Get current network statistics
 *
 * @param network Pointer to neural network
 * @param stats Pointer to stats structure to fill
 * @return true if successful, false otherwise
 */
bool neural_network_get_stats(neural_network_t network, network_stats_t* stats);

// Maintenance and utility functions
void neural_network_maintain(neural_network_t network, uint64_t timestamp);
void neural_network_maintain_homeostasis(neural_network_t network, uint64_t timestamp);
void neural_network_reset(neural_network_t network);
uint32_t neural_network_prune_synapses(neural_network_t network, float threshold);
void neural_network_dump_neuron(neural_network_t network, uint32_t neuron_id);

// NIMCP 2.7: NLP Integration - Accessor functions for synapse compute context
/**
 * @brief Set global state for synapse computation (e.g., attention output)
 *
 * WHAT: Attach global state buffer to network for synapse compute functions
 * WHY: Enable synapses to access shared context (attention, etc)
 * HOW: Stores pointer and size, passed to synapses via compute context
 * WHEN: Called by NLP layer after attention computation
 *
 * DESIGN PATTERN: Dependency Injection - external state injected into network
 * COMPLEXITY: O(1)
 *
 * @param network Neural network
 * @param global_state Pointer to global state buffer (e.g., attention output)
 * @param size Size of global state buffer (floats)
 * @return true on success
 */
bool neural_network_set_global_state(neural_network_t network, float* global_state, uint32_t size);

/**
 * @brief Set neuromodulator system for synapse computation
 *
 * WHAT: Attach neuromodulator system to network
 * WHY: Enable synapses to access neuromodulator levels (dopamine, etc)
 * HOW: Stores opaque pointer, queried during synapse computation
 * WHEN: Called by NLP layer during initialization
 *
 * DESIGN PATTERN: Dependency Injection - external system injected
 * COMPLEXITY: O(1)
 *
 * @param network Neural network
 * @param neuromod_system Opaque pointer to neuromodulator system
 * @return true on success
 */
bool neural_network_set_neuromodulator_system(neural_network_t network, void* neuromod_system);

/**
 * @brief Set glial integration system for neuro-glial signaling (Phase 6)
 *
 * WHAT: Attach glial integration system to network
 * WHY: Enable bidirectional neuro-glial communication
 * HOW: Stores opaque pointer, notified on neuron/synapse events
 * WHEN: Called after glial cells are assigned to neurons/synapses
 *
 * DESIGN PATTERN: Observer - glial system observes neural events
 * COMPLEXITY: O(1)
 *
 * @param network Neural network
 * @param glial_system Opaque pointer to glial integration system
 * @return true on success
 */
bool neural_network_set_glial_integration(neural_network_t network, void* glial_system);

/**
 * @brief Get neuromodulation level for synapse computation
 *
 * WHAT: Query current neuromodulation level (dopamine, primarily)
 * WHY: Used by synapse compute context to modulate transmission
 * HOW: Queries attached neuromodulator system
 * WHEN: Called during synapse computation in sum_synaptic_inputs
 *
 * COMPLEXITY: O(1)
 *
 * @param network Neural network
 * @return Current neuromodulation level [0,1] (0 if no system attached)
 */
float neural_network_get_neuromodulation(neural_network_t network);

// Bidirectional synapse access (OPTIMIZATION API)
/**
 * @brief Get count of incoming synapses to a neuron
 *
 * DESIGN PATTERN: Iterator Pattern - access to reverse edges
 * COMPLEXITY: O(1)
 *
 * @param network Neural network
 * @param neuron_id Target neuron
 * @return Number of incoming synapses (0 if invalid neuron or NULL network)
 */
uint32_t neural_network_get_incoming_synapse_count(neural_network_t network, uint32_t neuron_id);

/**
 * @brief Get array of incoming synapses
 *
 * DESIGN PATTERN: Iterator Pattern - provides direct access for O(S) iteration
 * COMPLEXITY: O(1) to get pointer, O(S) to iterate
 *
 * @param network Neural network
 * @param neuron_id Target neuron
 * @param out_synapses Pointer to receive synapse array (read-only)
 * @return Number of incoming synapses (0 if invalid)
 */
uint32_t neural_network_get_incoming_synapses(neural_network_t network, uint32_t neuron_id,
                                               const synapse_t** out_synapses);

//=============================================================================
// ENHANCEMENT 1: Synapse Semantic Embeddings API
//=============================================================================

/**
 * @brief Initialize semantic embedding for a synapse
 *
 * WHAT: Allocates and initializes semantic vector for intelligent routing
 * WHY: Enables synapses to route information based on semantic relevance
 * HOW: Random initialization, later refined by learning
 *
 * @param synapse Synapse to initialize
 * @param dim Embedding dimension (128-512 recommended)
 * @return true on success
 */
bool synapse_init_embedding(synapse_t *synapse, uint16_t dim);

/**
 * @brief Compute semantic similarity between two synapses
 *
 * WHAT: Cosine similarity between embedding vectors
 * WHY: Identifies functionally similar synapses
 * HOW: dot(a,b) / (||a|| * ||b||)
 *
 * @param syn1 First synapse
 * @param syn2 Second synapse
 * @return Similarity (-1 to 1, higher = more similar)
 */
float synapse_semantic_similarity(const synapse_t *syn1, const synapse_t *syn2);

/**
 * @brief Update synapse embedding via gradient descent
 *
 * WHAT: Refines embedding based on usage patterns
 * WHY: Learn semantic routing over time
 *
 * @param synapse Synapse to update
 * @param target_embedding Target vector [dim]
 * @param learning_rate Step size
 * @return true on success
 */
bool synapse_update_embedding(synapse_t *synapse, const float *target_embedding, float learning_rate);

/**
 * @brief Compute relevance of synapse to current context
 *
 * WHAT: Similarity between synapse embedding and context vector
 * WHY: Route information through relevant synapses
 *
 * @param synapse Synapse to evaluate
 * @param context_embedding Context vector [dim]
 * @param context_dim Context dimension
 * @return Relevance score (0-1)
 */
float synapse_compute_relevance(synapse_t *synapse, const float *context_embedding, uint16_t context_dim);

/**
 * @brief Free synapse embedding memory
 *
 * @param synapse Synapse to cleanup
 */
void synapse_destroy_embedding(synapse_t *synapse);

#ifdef __cplusplus
}
#endif

#endif  // NIMCP_NEURALNET_H
