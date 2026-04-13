//=============================================================================
// nimcp_snn_gpu.h - GPU Spiking Neural Network Kernels
//=============================================================================
/**
 * @file nimcp_snn_gpu.h
 * @brief GPU-accelerated Spiking Neural Network operations using CUDA
 *
 * WHAT: CUDA kernels for SNN neuron models and spike processing
 * WHY:  Enables massive parallel simulation of spiking networks
 * HOW:  Custom kernels for LIF, Izhikevich, surrogate gradients, STDP
 *
 * ARCHITECTURE:
 * - Neuron models: LIF, Izhikevich, AdEx (Adaptive Exponential)
 * - Surrogate gradients: SuperSpike, fast sigmoid, arctan, triangular
 * - Learning: STDP (pair-based, triplet), eligibility traces
 * - Spike propagation: event-driven processing
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifndef NIMCP_SNN_GPU_H
#define NIMCP_SNN_GPU_H

#include "gpu/context/nimcp_gpu_context.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "common/nimcp_export.h"
#include "gpu/tensor/nimcp_tensor_gpu.h"
#include <stdbool.h>
#include <stdint.h>

//=============================================================================
// Neuron Model Types
//=============================================================================

typedef enum {
    NIMCP_SNN_LIF = 0,            /**< Leaky Integrate-and-Fire */
    NIMCP_SNN_IZHIKEVICH = 1,     /**< Izhikevich model */
    NIMCP_SNN_ADEX = 2,           /**< Adaptive Exponential IF */
    NIMCP_SNN_HH = 3              /**< Hodgkin-Huxley (simplified) */
} nimcp_snn_model_t;

typedef enum {
    NIMCP_SURROGATE_SUPERSPIKE = 0,    /**< SuperSpike surrogate */
    NIMCP_SURROGATE_FAST_SIGMOID = 1,  /**< Fast sigmoid */
    NIMCP_SURROGATE_ARCTAN = 2,        /**< Arctan */
    NIMCP_SURROGATE_TRIANGULAR = 3,    /**< Triangular */
    NIMCP_SURROGATE_GAUSSIAN = 4       /**< Gaussian */
} nimcp_surrogate_type_t;

//=============================================================================
// LIF Neuron Parameters
//=============================================================================

typedef struct {
    float tau_mem;      /**< Membrane time constant (ms) */
    float tau_syn;      /**< Synaptic time constant (ms) */
    float v_thresh;     /**< Spike threshold (mV) */
    float v_reset;      /**< Reset voltage (mV) */
    float v_rest;       /**< Resting potential (mV) */
    float dt;           /**< Timestep (ms) */
    bool hard_reset;    /**< Hard reset (true) or soft reset (false) */
} nimcp_lif_params_t;

//=============================================================================
// Izhikevich Neuron Parameters
//=============================================================================

typedef struct {
    float a;            /**< Recovery time constant */
    float b;            /**< Recovery sensitivity to v */
    float c;            /**< Reset voltage (mV) */
    float d;            /**< Recovery increment after spike */
    float v_thresh;     /**< Spike threshold (mV) */
    float dt;           /**< Timestep (ms) */
} nimcp_izhikevich_params_t;

//=============================================================================
// AdEx Neuron Parameters
//=============================================================================

typedef struct {
    float tau_mem;      /**< Membrane time constant */
    float tau_w;        /**< Adaptation time constant */
    float v_thresh;     /**< Spike threshold */
    float v_reset;      /**< Reset voltage */
    float v_rest;       /**< Resting potential */
    float v_rheo;       /**< Rheobase threshold */
    float delta_T;      /**< Slope factor */
    float a;            /**< Subthreshold adaptation */
    float b;            /**< Spike-triggered adaptation */
    float dt;           /**< Timestep */
} nimcp_adex_params_t;

//=============================================================================
// STDP Parameters
//=============================================================================

typedef struct {
    float A_plus;       /**< LTP amplitude */
    float A_minus;      /**< LTD amplitude */
    float tau_plus;     /**< LTP time constant (ms) */
    float tau_minus;    /**< LTD time constant (ms) */
    float w_max;        /**< Maximum weight */
    float w_min;        /**< Minimum weight */
} nimcp_stdp_params_t;

//=============================================================================
// Triplet STDP Parameters (Pfister & Gerstner 2006)
//=============================================================================

/**
 * @brief Triplet STDP parameters
 *
 * Extends pair-based STDP by considering triplets of spikes for more
 * biologically accurate learning. Based on Pfister & Gerstner (2006).
 *
 * The weight update rule is:
 *   dw = A2_plus * r1 * post_spike + A3_plus * r1 * o2 * post_spike   (LTP)
 *      - A2_minus * o1 * pre_spike - A3_minus * o1 * r2 * pre_spike   (LTD)
 *
 * Where:
 *   r1, r2 = presynaptic traces (fast and slow)
 *   o1, o2 = postsynaptic traces (fast and slow)
 */
typedef struct nimcp_triplet_stdp_params {
    // Pair-based terms (classic STDP)
    float A2_plus;    /**< LTP amplitude for pairs */
    float A2_minus;   /**< LTD amplitude for pairs */
    float tau_plus;   /**< LTP time constant (ms) */
    float tau_minus;  /**< LTD time constant (ms) */

    // Triplet terms
    float A3_plus;    /**< LTP amplitude for triplets (post-post-pre) */
    float A3_minus;   /**< LTD amplitude for triplets (pre-pre-post) */
    float tau_x;      /**< Presynaptic triplet trace time constant (ms) */
    float tau_y;      /**< Postsynaptic triplet trace time constant (ms) */

    // Bounds
    float w_min;      /**< Minimum synaptic weight */
    float w_max;      /**< Maximum synaptic weight */
} nimcp_triplet_stdp_params_t;

/**
 * @brief Triplet STDP state holding spike traces
 *
 * Maintains four traces:
 *   r1 (fast pre): decays with tau_plus, incremented on pre spike
 *   r2 (slow pre): decays with tau_x, incremented on pre spike
 *   o1 (fast post): decays with tau_minus, incremented on post spike
 *   o2 (slow post): decays with tau_y, incremented on post spike
 */
typedef struct nimcp_triplet_stdp_state {
    float* d_r1;      /**< Presynaptic trace (fast, for pairs) */
    float* d_r2;      /**< Presynaptic trace (slow, for triplets) */
    float* d_o1;      /**< Postsynaptic trace (fast, for pairs) */
    float* d_o2;      /**< Postsynaptic trace (slow, for triplets) */
    size_t num_pre;   /**< Number of presynaptic neurons */
    size_t num_post;  /**< Number of postsynaptic neurons */
} nimcp_triplet_stdp_state_t;

/**
 * @brief DAO pattern for STDP state management
 *
 * Encapsulates STDP state and operations with function pointers
 * for flexible, testable learning rule implementations.
 */
typedef struct nimcp_stdp_dao {
    nimcp_triplet_stdp_state_t* state;    /**< Internal state */
    nimcp_triplet_stdp_params_t params;   /**< Learning parameters */
    void* gpu_context;                     /**< GPU context reference */

    /** Update spike traces based on current spikes */
    int (*update_traces)(struct nimcp_stdp_dao* self, const int* pre_spikes,
                         const int* post_spikes, size_t num_pre_spikes,
                         size_t num_post_spikes, float dt);

    /** Compute weight updates without applying them */
    int (*compute_weight_updates)(struct nimcp_stdp_dao* self, float* weights,
                                  const int* pre_indices, const int* post_indices,
                                  size_t num_synapses);

    /** Apply computed weight updates with learning rate */
    int (*apply_updates)(struct nimcp_stdp_dao* self, float* weights,
                         float learning_rate);

    /** Reset all traces to zero */
    int (*reset)(struct nimcp_stdp_dao* self);
} nimcp_stdp_dao_t;

//=============================================================================
// SNN State Structures
//=============================================================================

/**
 * @brief LIF neuron state
 */
typedef struct {
    nimcp_gpu_tensor_t* v;          /**< Membrane potential */
    nimcp_gpu_tensor_t* i_syn;      /**< Synaptic current */
    nimcp_gpu_tensor_t* spikes;     /**< Output spikes (binary) */
    nimcp_lif_params_t params;      /**< Neuron parameters */
} nimcp_lif_state_t;

/**
 * @brief Izhikevich neuron state
 */
typedef struct {
    nimcp_gpu_tensor_t* v;          /**< Membrane potential */
    nimcp_gpu_tensor_t* u;          /**< Recovery variable */
    nimcp_gpu_tensor_t* spikes;     /**< Output spikes */
    nimcp_izhikevich_params_t params;
} nimcp_izhikevich_state_t;

/**
 * @brief AdEx neuron state
 */
typedef struct {
    nimcp_gpu_tensor_t* v;          /**< Membrane potential */
    nimcp_gpu_tensor_t* w;          /**< Adaptation current */
    nimcp_gpu_tensor_t* spikes;     /**< Output spikes */
    nimcp_adex_params_t params;
} nimcp_adex_state_t;

//=============================================================================
// LIF Neuron Model
//=============================================================================

/**
 * @brief Create LIF neuron state
 */
NIMCP_EXPORT nimcp_lif_state_t* nimcp_lif_state_create(
    nimcp_gpu_context_t* ctx,
    size_t n_neurons,
    const nimcp_lif_params_t* params
);

/**
 * @brief Destroy LIF neuron state
 */
NIMCP_EXPORT void nimcp_lif_state_destroy(nimcp_lif_state_t* state);

/**
 * @brief LIF forward pass
 *
 * Implements: dv/dt = (v_rest - v + R*I) / tau_mem
 * Spike if v > v_thresh, then reset to v_reset
 *
 * @param ctx GPU context
 * @param state LIF neuron state
 * @param input Synaptic input current
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_lif_forward(
    nimcp_gpu_context_t* ctx,
    nimcp_lif_state_t* state,
    const nimcp_gpu_tensor_t* input
);

/**
 * @brief Compute I_syn from CSR synapse storage on GPU
 *
 * Sparse gather: for each destination neuron, sum weights of incoming
 * synapses whose source neurons spiked. Uses flat column indices that
 * map (src_pop, src_neuron) to a global spike vector offset.
 *
 * @param ctx GPU context
 * @param d_spike_vector Device pointer: flattened spikes [total_neurons]
 * @param d_weights Device pointer: CSR weights [nnz]
 * @param d_col_indices Device pointer: flat column indices [nnz]
 * @param d_row_ptr Device pointer: CSR row pointers [n_neurons+1]
 * @param d_external_current Device pointer: external currents [n_neurons]
 * @param d_output_isyn Device pointer: output I_syn [n_neurons]
 * @param n_neurons Number of neurons in this population
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_snn_isyn_csr(
    nimcp_gpu_context_t* ctx,
    const float* d_spike_vector,
    const float* d_weights,
    const unsigned int* d_col_indices,
    const unsigned int* d_row_ptr,
    const float* d_external_current,
    float* d_output_isyn,
    size_t n_neurons);

/**
 * @brief LIF backward pass with surrogate gradient
 *
 * @param ctx GPU context
 * @param state LIF neuron state
 * @param grad_output Gradient from upstream
 * @param grad_input Output: gradient w.r.t. input
 * @param surrogate_type Type of surrogate gradient
 * @param beta Surrogate gradient sharpness
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_lif_backward(
    nimcp_gpu_context_t* ctx,
    const nimcp_lif_state_t* state,
    const nimcp_gpu_tensor_t* grad_output,
    nimcp_gpu_tensor_t* grad_input,
    nimcp_surrogate_type_t surrogate_type,
    float beta
);

//=============================================================================
// Izhikevich Neuron Model
//=============================================================================

/**
 * @brief Create Izhikevich neuron state
 */
NIMCP_EXPORT nimcp_izhikevich_state_t* nimcp_izhikevich_state_create(
    nimcp_gpu_context_t* ctx,
    size_t n_neurons,
    const nimcp_izhikevich_params_t* params
);

/**
 * @brief Destroy Izhikevich neuron state
 */
NIMCP_EXPORT void nimcp_izhikevich_state_destroy(nimcp_izhikevich_state_t* state);

/**
 * @brief Izhikevich forward pass
 *
 * Implements:
 *   dv/dt = 0.04*v^2 + 5*v + 140 - u + I
 *   du/dt = a*(b*v - u)
 *   if v >= 30mV: v = c, u = u + d
 */
NIMCP_EXPORT bool nimcp_gpu_izhikevich_forward(
    nimcp_gpu_context_t* ctx,
    nimcp_izhikevich_state_t* state,
    const nimcp_gpu_tensor_t* input
);

//=============================================================================
// AdEx Neuron Model
//=============================================================================

/**
 * @brief Create AdEx neuron state
 */
NIMCP_EXPORT nimcp_adex_state_t* nimcp_adex_state_create(
    nimcp_gpu_context_t* ctx,
    size_t n_neurons,
    const nimcp_adex_params_t* params
);

/**
 * @brief Destroy AdEx neuron state
 */
NIMCP_EXPORT void nimcp_adex_state_destroy(nimcp_adex_state_t* state);

/**
 * @brief AdEx forward pass
 */
NIMCP_EXPORT bool nimcp_gpu_adex_forward(
    nimcp_gpu_context_t* ctx,
    nimcp_adex_state_t* state,
    const nimcp_gpu_tensor_t* input
);

//=============================================================================
// Surrogate Gradient Functions
//=============================================================================

/**
 * @brief Compute surrogate gradient
 *
 * @param ctx GPU context
 * @param v Membrane potential
 * @param v_thresh Spike threshold
 * @param grad Output gradient tensor
 * @param surrogate_type Type of surrogate function
 * @param beta Sharpness parameter
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_surrogate_gradient(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* v,
    float v_thresh,
    nimcp_gpu_tensor_t* grad,
    nimcp_surrogate_type_t surrogate_type,
    float beta
);

//=============================================================================
// Spike Propagation
//=============================================================================

/**
 * @brief Propagate spikes through synapses
 *
 * Computes: I_post = W @ spikes_pre (sparse matrix-vector multiply)
 *
 * @param ctx GPU context
 * @param spikes Pre-synaptic spike tensor (binary)
 * @param weights Synaptic weight matrix
 * @param output Post-synaptic current
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_spike_propagate(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* spikes,
    const nimcp_gpu_tensor_t* weights,
    nimcp_gpu_tensor_t* output
);

/**
 * @brief Event-driven spike propagation (sparse)
 *
 * Only processes non-zero spikes for efficiency
 *
 * @param ctx GPU context
 * @param spike_indices Indices of neurons that spiked
 * @param n_spikes Number of spikes
 * @param weights Synaptic weight matrix
 * @param output Post-synaptic current
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_spike_propagate_sparse(
    nimcp_gpu_context_t* ctx,
    const uint32_t* spike_indices,
    size_t n_spikes,
    const nimcp_gpu_tensor_t* weights,
    nimcp_gpu_tensor_t* output
);

//=============================================================================
// STDP Learning
//=============================================================================

/**
 * @brief Update eligibility traces
 *
 * For E-prop and other eligibility-based learning
 * e = e * decay + spike * (1 - decay)
 */
NIMCP_EXPORT bool nimcp_gpu_eligibility_trace_update(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* trace,
    const nimcp_gpu_tensor_t* spikes,
    float decay
);

/**
 * @brief Pair-based STDP update
 *
 * Classic Hebbian STDP:
 * - Pre before post (dt > 0): potentiation (A_plus * exp(-dt/tau_plus))
 * - Post before pre (dt < 0): depression (A_minus * exp(dt/tau_minus))
 *
 * @param ctx GPU context
 * @param weights Synaptic weights (in-place update)
 * @param pre_spikes Pre-synaptic spikes
 * @param post_spikes Post-synaptic spikes
 * @param pre_trace Pre-synaptic eligibility trace
 * @param post_trace Post-synaptic eligibility trace
 * @param params STDP parameters
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_stdp_pair(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* weights,
    const nimcp_gpu_tensor_t* pre_spikes,
    const nimcp_gpu_tensor_t* post_spikes,
    const nimcp_gpu_tensor_t* pre_trace,
    const nimcp_gpu_tensor_t* post_trace,
    const nimcp_stdp_params_t* params
);

/**
 * @brief Triplet STDP update (legacy interface)
 *
 * Enhanced STDP that considers triplets of spikes
 */
NIMCP_EXPORT bool nimcp_gpu_stdp_triplet(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* weights,
    const nimcp_gpu_tensor_t* pre_spikes,
    const nimcp_gpu_tensor_t* post_spikes,
    nimcp_gpu_tensor_t* pre_trace_fast,
    nimcp_gpu_tensor_t* pre_trace_slow,
    nimcp_gpu_tensor_t* post_trace_fast,
    nimcp_gpu_tensor_t* post_trace_slow,
    const nimcp_stdp_params_t* params
);

//=============================================================================
// Triplet STDP DAO API (Pfister & Gerstner 2006)
//=============================================================================

/**
 * @brief Create triplet STDP DAO
 *
 * Creates a Data Access Object for triplet STDP learning with full
 * state management and parameterized learning rules.
 *
 * @param gpu_ctx GPU context
 * @param num_pre Number of presynaptic neurons
 * @param num_post Number of postsynaptic neurons
 * @param params Triplet STDP parameters
 * @return DAO object or NULL on failure
 */
NIMCP_EXPORT nimcp_stdp_dao_t* nimcp_triplet_stdp_create(
    void* gpu_ctx,
    size_t num_pre,
    size_t num_post,
    nimcp_triplet_stdp_params_t* params
);

/**
 * @brief Destroy triplet STDP DAO
 *
 * Frees all GPU memory and destroys the DAO object.
 *
 * @param dao DAO to destroy
 */
NIMCP_EXPORT void nimcp_triplet_stdp_destroy(nimcp_stdp_dao_t* dao);

/**
 * @brief Perform one step of triplet STDP learning
 *
 * Updates traces based on current spikes and modifies weights accordingly.
 * This is a convenience function that combines trace update and weight update.
 *
 * The triplet rule (Pfister & Gerstner 2006):
 *   - LTP: dw = (A2_plus + A3_plus * o2) * r1 * post_spike
 *   - LTD: dw = -(A2_minus + A3_minus * r2) * o1 * pre_spike
 *
 * @param dao STDP DAO object
 * @param pre_spikes Array of presynaptic neuron indices that spiked
 * @param post_spikes Array of postsynaptic neuron indices that spiked
 * @param num_pre_spikes Number of presynaptic spikes
 * @param num_post_spikes Number of postsynaptic spikes
 * @param weights Weight matrix (num_pre x num_post), modified in-place
 * @param pre_indices Presynaptic indices for synapse connections
 * @param post_indices Postsynaptic indices for synapse connections
 * @param num_synapses Number of synapses to update
 * @param dt Timestep (ms)
 * @param learning_rate Global learning rate multiplier
 * @return 0 on success, -1 on error
 */
NIMCP_EXPORT int nimcp_triplet_stdp_step(
    nimcp_stdp_dao_t* dao,
    const int* pre_spikes,
    const int* post_spikes,
    size_t num_pre_spikes,
    size_t num_post_spikes,
    float* weights,
    const int* pre_indices,
    const int* post_indices,
    size_t num_synapses,
    float dt,
    float learning_rate
);

/**
 * @brief Full triplet STDP update with tensor interface
 *
 * Higher-level interface using GPU tensors instead of raw arrays.
 * Updates traces and applies weight changes in a single call.
 *
 * @param ctx GPU context
 * @param weights Weight tensor (modified in-place)
 * @param pre_spikes Presynaptic spike tensor (binary 0/1)
 * @param post_spikes Postsynaptic spike tensor (binary 0/1)
 * @param r1 Fast presynaptic trace (updated in-place)
 * @param r2 Slow presynaptic trace (updated in-place)
 * @param o1 Fast postsynaptic trace (updated in-place)
 * @param o2 Slow postsynaptic trace (updated in-place)
 * @param params Triplet STDP parameters
 * @param dt Timestep (ms)
 * @param learning_rate Learning rate multiplier
 * @return true on success
 */
NIMCP_EXPORT bool nimcp_gpu_triplet_stdp_full(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* weights,
    const nimcp_gpu_tensor_t* pre_spikes,
    const nimcp_gpu_tensor_t* post_spikes,
    nimcp_gpu_tensor_t* r1,
    nimcp_gpu_tensor_t* r2,
    nimcp_gpu_tensor_t* o1,
    nimcp_gpu_tensor_t* o2,
    const nimcp_triplet_stdp_params_t* params,
    float dt,
    float learning_rate
);

/**
 * @brief Get default triplet STDP parameters
 *
 * Returns biologically plausible default parameters based on
 * experimental data from Pfister & Gerstner (2006).
 *
 * @param params Output parameter structure to fill
 */
NIMCP_EXPORT void nimcp_triplet_stdp_default_params(
    nimcp_triplet_stdp_params_t* params
);

//=============================================================================
// High-Level SNN Layer API (E2E Test Convenience)
//=============================================================================

/**
 * @brief SNN layer configuration for easy creation
 */
typedef struct {
    size_t num_neurons;         /**< Number of neurons in layer */
    float tau_mem;              /**< Membrane time constant (ms) */
    float tau_syn;              /**< Synaptic time constant (ms) */
    float v_rest;               /**< Resting potential (mV) */
    float v_thresh;             /**< Spike threshold (mV) */
    float v_reset;              /**< Reset potential (mV) */
    float dt;                   /**< Timestep (ms) */
    float refractory_period;    /**< Refractory period (ms) */
    nimcp_snn_model_t model;    /**< Neuron model type */
} nimcp_snn_lif_config_t;

/**
 * @brief Unified SNN layer handle
 */
typedef struct nimcp_snn_layer_s {
    nimcp_snn_model_t model;
    size_t num_neurons;
    nimcp_lif_state_t* lif_state;
    nimcp_izhikevich_state_t* izh_state;
    nimcp_adex_state_t* adex_state;
    nimcp_gpu_context_t* ctx;
    float refractory_period;
    nimcp_gpu_tensor_t* refractory_timer;  /**< Time remaining in refractory */
} nimcp_snn_layer_t;

/**
 * @brief Create SNN layer from configuration
 */
NIMCP_EXPORT nimcp_snn_layer_t* nimcp_snn_lif_layer_create(
    nimcp_gpu_context_t* ctx,
    const nimcp_snn_lif_config_t* config
);

/**
 * @brief Destroy SNN layer
 */
NIMCP_EXPORT void nimcp_snn_layer_destroy(nimcp_snn_layer_t* layer);

/**
 * @brief Get membrane potential tensor
 */
NIMCP_EXPORT nimcp_gpu_tensor_t* nimcp_snn_layer_get_membrane(nimcp_snn_layer_t* layer);

/**
 * @brief Get output spikes tensor
 */
NIMCP_EXPORT nimcp_gpu_tensor_t* nimcp_snn_layer_get_spikes(nimcp_snn_layer_t* layer);

/**
 * @brief Get layer size (number of neurons)
 */
NIMCP_EXPORT size_t nimcp_snn_layer_get_size(const nimcp_snn_layer_t* layer);

/**
 * @brief Get membrane time constant
 */
NIMCP_EXPORT float nimcp_snn_layer_get_tau_mem(const nimcp_snn_layer_t* layer);

/**
 * @brief Reset layer state to initial values
 */
NIMCP_EXPORT bool nimcp_snn_layer_reset(
    nimcp_gpu_context_t* ctx,
    nimcp_snn_layer_t* layer
);

/**
 * @brief Forward pass for one timestep
 */
NIMCP_EXPORT bool nimcp_snn_layer_forward(
    nimcp_gpu_context_t* ctx,
    nimcp_snn_layer_t* layer,
    const nimcp_gpu_tensor_t* input
);

/**
 * @brief Forward pass that returns spikes tensor
 * @return Pointer to internal spikes tensor (do not free)
 */
NIMCP_EXPORT nimcp_gpu_tensor_t* nimcp_snn_lif_step(
    nimcp_gpu_context_t* ctx,
    nimcp_snn_layer_t* layer,
    const nimcp_gpu_tensor_t* input
);

/**
 * @brief Create spike tensor from host data
 */
NIMCP_EXPORT nimcp_gpu_tensor_t* nimcp_snn_spike_tensor_create(
    nimcp_gpu_context_t* ctx,
    const uint8_t* data,
    const size_t* dims,
    size_t ndim
);

/**
 * @brief Copy spike tensor to host
 */
NIMCP_EXPORT bool nimcp_snn_spike_tensor_to_host(
    const nimcp_gpu_tensor_t* tensor,
    uint8_t* data
);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Reset neuron state to resting values
 */
NIMCP_EXPORT bool nimcp_gpu_snn_reset_state(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_tensor_t* v,
    float v_rest
);

/**
 * @brief Count spikes in tensor
 */
NIMCP_EXPORT bool nimcp_gpu_spike_count(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* spikes,
    uint32_t* count
);

/**
 * @brief Compute spike rate (spikes per timestep)
 */
NIMCP_EXPORT bool nimcp_gpu_spike_rate(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* spikes,
    size_t n_timesteps,
    nimcp_gpu_tensor_t* rates
);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_SNN_GPU_H
