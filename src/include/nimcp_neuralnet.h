// nimcp_neuralnet.h

#ifndef NIMCP_NEURALNET_H
#define NIMCP_NEURALNET_H

#include "nimcp_export.h"
#include "/usr/include/python3.10/Python.h"
extern NIMCP_EXPORT PyTypeObject NeuralNetworkType;

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

#include <stdint.h>
#include <stdbool.h>



/**
 * @brief Network size and capacity constants
 */
#define MAX_NEURONS 1024
#define MAX_SYNAPSES_PER_NEURON 256
#define SPIKE_HISTORY_LENGTH 1000
#define HISTORY_WINDOW 100
#define WEIGHT_UPDATE_THRESHOLD 1e-6f
#define ACTIVITY_THRESHOLD 1e-5f
#define NORM_THRESHOLD 1e-7f
#define NORMALIZATION_INTERVAL 1000 // ms

/**
 * @brief Neuron types for E/I balance
 */
typedef enum {
    NEURON_EXCITATORY,    /**< Excitatory neuron */
    NEURON_INHIBITORY     /**< Inhibitory neuron */
} neuron_type_t;

/**
 * @brief Learning rule types
 */
typedef enum {
    LEARNING_HEBBIAN,      /**< Basic Hebbian learning */
    LEARNING_OJA,          /**< Oja's learning rule */
    LEARNING_GENERALIZED_OJA, /**< Generalized Oja's rule */
    LEARNING_STDP,         /**< Spike-timing dependent plasticity */
    LEARNING_HYBRID        /**< Combined learning rules */
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
    uint64_t timestamp;    /**< Spike timestamp */
    float magnitude;       /**< Spike magnitude */
} spike_record_t;

/**
 * @brief Oja learning parameters
 */
typedef struct {
    float alpha;          /**< Learning rate parameter */
    float forgetting;     /**< Forgetting/normalization factor */
    float stabilization;  /**< Stabilization parameter */
    float target_norm;    /**< Target norm for weights */
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
    float target_activity;  /**< Target average activity */
    float time_scale;       /**< Time scale for averaging */
    float strength;         /**< Strength of homeostatic adjustment */
} homeostatic_params_t;

/**
 * @brief Synapse structure
 */
typedef struct {
    uint32_t target_id;     /**< Target neuron ID */
    float weight;           /**< Synaptic weight */
    float plasticity;       /**< Plasticity coefficient */
    float last_change;      /**< Last weight change (for momentum) */
    uint64_t last_active;   /**< Last activation timestamp */
    float strength;         /**< Synaptic strength (separate from weight) */
    float meta_plasticity;  /**< Meta-plasticity factor */
    float trace;            /**< Synaptic trace for STDP */
} synapse_t;

/**
 * @brief Neuron structure
 */
typedef struct {
    uint32_t id;                          /**< Unique neuron identifier */
    neuron_type_t type;                   /**< Excitatory or Inhibitory */
    float state;                          /**< Current activation state */
    float rest_potential;                 /**< Resting membrane potential */
    float threshold;                      /**< Firing threshold */
    float adaptation;                     /**< Adaptive threshold parameter */
    float refractory_period;              /**< Refractory period (ms) */
    float bias;                           /**< Neuron bias value */
    
    // Learning parameters
    learning_rule_t learning_rule;         /**< Active learning rule */
    activation_type_t activation_type;     /**< Activation function type */
    oja_params_t oja_params;              /**< Oja learning parameters */
    stdp_params_t stdp_params;            /**< STDP parameters */
    homeostatic_params_t homeostatic;     /**< Homeostatic parameters */
    
    // Synaptic connections
    synapse_t synapses[MAX_SYNAPSES_PER_NEURON];
    uint32_t num_synapses;
    
    // Plasticity parameters
    float plasticity_rate;                /**< Learning rate for plasticity */
    float homeostatic_factor;             /**< Homeostatic plasticity factor */
    float calcium_concentration;           /**< Calcium concentration (for STDP) */
    float weight_norm;                    /**< Current norm of weights */
    
    // Activity tracking
    spike_record_t spike_history[SPIKE_HISTORY_LENGTH];
    uint32_t spike_history_index;
    float activity_history[HISTORY_WINDOW];
    float avg_activity;                   /**< Average activity level */
    uint64_t last_spike;                  /**< Last spike timestamp */
    uint64_t last_update;                 /**< Last state update timestamp */
    uint64_t creation_time;               /**< Neuron creation timestamp */
} neuron_t;

/**
 * @brief Network configuration parameters
 */
typedef struct {
    // Original NIMCP 2.0 fields
    uint32_t num_neurons;                 /**< Initial number of neurons */
    float ei_ratio;                       /**< Excitatory/Inhibitory ratio */
    float learning_rate;                  /**< Base learning rate */
    float hebbian_rate;                   /**< Hebbian learning rate */
    float stdp_window;                    /**< STDP time window (ms) */
    float homeostatic_rate;               /**< Homeostatic plasticity rate */
    float target_activity;                /**< Target activity level */
    float adaptation_rate;                /**< Threshold adaptation rate */
    float refractory_period;              /**< Base refractory period (ms) */
    float min_weight;                     /**< Minimum synaptic weight */
    float max_weight;                     /**< Maximum synaptic weight */
    uint32_t update_interval;             /**< Minimum update interval (ms) */

    // NIMCP 2.5 Brain API extensions
    uint32_t input_size;                  /**< Number of input neurons */
    uint32_t output_size;                 /**< Number of output neurons */
    uint32_t num_layers;                  /**< Number of layers */
    uint32_t* layer_sizes;                /**< Size of each layer */
    bool enable_stdp;                     /**< Enable STDP learning */
    bool enable_hebbian;                  /**< Enable Hebbian learning */
    bool enable_oja;                      /**< Enable Oja's rule */
    bool enable_homeostasis;              /**< Enable homeostatic plasticity */
} network_config_t;

/**
 * @brief Neural network structure
 */
typedef struct neural_network_struct* neural_network_t;

// Core network functions
neural_network_t neural_network_create(const network_config_t* config);
void neural_network_destroy(neural_network_t network);
bool neural_network_update_neuron(neural_network_t network, uint32_t neuron_id, float new_state, uint64_t timestamp);
bool neural_network_get_neuron_state(neural_network_t network, uint32_t neuron_id, float* state);

// NIMCP 2.5 - Forward pass for inference
bool neural_network_forward(neural_network_t network, const float* inputs, uint32_t input_size,
                            float* outputs, uint32_t output_size);
uint32_t neural_network_add_neuron(neural_network_t network, activation_type_t activation);
bool neural_network_add_connection(neural_network_t network, uint32_t from_id, uint32_t to_id, float weight);
uint32_t neural_network_compute_step(neural_network_t network, uint64_t timestamp);

// Learning and plasticity functions
uint32_t neural_network_apply_oja(neural_network_t network, uint32_t neuron_id, uint64_t timestamp);
uint32_t neural_network_apply_generalized_oja(neural_network_t network, uint32_t neuron_id, uint64_t timestamp);
uint32_t neural_network_apply_stdp(neural_network_t network, uint32_t neuron_id, uint64_t timestamp);
bool neural_network_apply_homeostasis(neural_network_t network, uint32_t neuron_id, uint64_t timestamp);
uint32_t neural_network_update_plasticity(neural_network_t network, uint32_t neuron_id, uint64_t timestamp);
bool neural_network_normalize_weights(neural_network_t network, uint32_t neuron_id);

// Activity and state functions
bool neural_network_record_spike(neural_network_t network, uint32_t neuron_id, float magnitude, uint64_t timestamp);
bool neural_network_adapt_threshold(neural_network_t network, uint32_t neuron_id, uint64_t timestamp);
float neural_network_compute_activation(neuron_t* neuron, float input);
void neural_network_update_traces(neural_network_t network, uint32_t neuron_id, uint64_t timestamp);

// Analysis and monitoring functions
float neural_network_get_average_activity(neural_network_t network, uint32_t neuron_id);
float neural_network_get_weight_norm(neural_network_t network, uint32_t neuron_id);
void neural_network_get_weight_statistics(neural_network_t network, uint32_t neuron_id, float* mean, float* std_dev);

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

#endif // NIMCP_NEURALNET_H
