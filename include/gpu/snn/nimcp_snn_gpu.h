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

#ifdef __cplusplus
extern "C" {
#endif

#include "common/nimcp_export.h"
#include "gpu/context/nimcp_gpu_context.h"
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
 * @brief Triplet STDP update
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
