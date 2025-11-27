/**
 * @file nimcp_gpu_neuron.c
 * @brief GPU Neural Network Implementation (with CPU fallback)
 *
 * WHAT: GPU-accelerated neural network with automatic CPU fallback
 * WHY:  Provide 10-100x speedup when GPU available, graceful degradation otherwise
 * HOW:  Detect GPU at runtime, use CUDA if available, CPU otherwise
 *
 * IMPLEMENTATION STRATEGY:
 * - All functions check GPU availability
 * - CUDA code compiled only when NIMCP_ENABLE_CUDA defined
 * - CPU fallback always available
 * - Same API for both paths
 *
 * @author NIMCP Development Team
 * @date 2025-11-08
 * @version 2.7.0
 */

#include "gpu/nimcp_gpu_neuron.h"
#include "gpu/nimcp_execution_mode.h"
#include "utils/memory/nimcp_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef NIMCP_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief GPU neural network internal structure
 */
struct gpu_neural_network_struct {
    // Configuration
    gpu_network_config_t config;

    // Neuron data (host-side copy)
    gpu_neuron_state_t* neurons_host;
    uint32_t neurons_count;
    uint32_t neurons_capacity;

    // Synapse data (host-side copy)
    gpu_synapse_t* synapses_host;
    uint32_t synapses_count;
    uint32_t synapses_capacity;

    // GPU pointers (NULL if CPU fallback)
#ifdef NIMCP_ENABLE_CUDA
    gpu_neuron_state_t* neurons_device;
    gpu_synapse_t* synapses_device;
    cudaStream_t stream;
#endif

    // Execution mode
    bool using_gpu;

    // Statistics
    uint64_t total_spikes;
    uint64_t total_updates;
};

//=============================================================================
// GPU Detection
//=============================================================================

NIMCP_EXPORT bool gpu_is_available(void)
{
#ifdef NIMCP_ENABLE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

NIMCP_EXPORT uint32_t gpu_get_device_count(void)
{
#ifdef NIMCP_ENABLE_CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return (uint32_t)device_count;
#else
    return 0;
#endif
}

NIMCP_EXPORT bool gpu_get_device_name(uint32_t device_id, char* name, size_t max_len)
{
    if (!name || max_len == 0) {
        return false;
    }

#ifdef NIMCP_ENABLE_CUDA
    struct cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, (int)device_id);
    if (err == cudaSuccess) {
        snprintf(name, max_len, "%s", props.name);
        return true;
    }
#else
    (void)device_id;  // Unused in CPU fallback
#endif

    snprintf(name, max_len, "CPU Fallback");
    return false;
}

//=============================================================================
// Configuration
//=============================================================================

NIMCP_EXPORT gpu_network_config_t gpu_get_optimal_config(uint32_t num_neurons)
{
    gpu_network_config_t config = {0};

    config.num_neurons = num_neurons;
    config.num_synapses = num_neurons * 100;  // Average 100 synapses per neuron

    // GPU kernel configuration
    config.threads_per_block = 256;  // Optimal for most GPUs
    config.max_blocks = (num_neurons + config.threads_per_block - 1) / config.threads_per_block;

    // Memory configuration
    config.spike_queue_capacity = num_neurons * 10;  // 10 spikes per neuron buffer
    config.use_unified_memory = false;  // Explicit transfers faster
    config.pin_host_memory = true;      // Faster CPU↔GPU transfers

    // Execution mode (auto-detect)
    config.exec_mode = gpu_is_available() ? EXEC_MODE_GPU_CUDA : EXEC_MODE_CPU_SEQUENTIAL;

    // Learning configuration
    config.enable_stdp = true;
    config.enable_bcm = false;
    config.global_learning_rate = 0.01f;

    return config;
}

//=============================================================================
// Network Lifecycle
//=============================================================================

NIMCP_EXPORT gpu_neural_network_t gpu_neural_network_create(
    const gpu_network_config_t* config)
{
    if (!config || config->num_neurons == 0) {
        return NULL;
    }

    // Allocate network structure
    gpu_neural_network_t network = nimcp_calloc(1, sizeof(struct gpu_neural_network_struct));
    if (!network) {
        return NULL;
    }

    // Copy configuration
    network->config = *config;
    network->neurons_capacity = config->num_neurons;
    network->synapses_capacity = config->num_synapses;
    network->neurons_count = 0;
    network->synapses_count = 0;
    network->total_spikes = 0;
    network->total_updates = 0;

    // Allocate host memory for neurons (64-byte aligned for gpu_neuron_state_t)
    size_t neurons_size = network->neurons_capacity * sizeof(gpu_neuron_state_t);
    // Round up to multiple of 64 for aligned_alloc requirement
    neurons_size = ((neurons_size + 63) / 64) * 64;
    network->neurons_host = (gpu_neuron_state_t*)nimcp_aligned_alloc(64, neurons_size);
    if (!network->neurons_host) {
        nimcp_free(network);
        return NULL;
    }
    // Zero-initialize the aligned memory
    memset(network->neurons_host, 0, neurons_size);

    /**
     * WHAT: Allocate host memory for synapses with 16-byte alignment
     * WHY:  gpu_synapse_t requires 16-byte alignment (Issue #GPU-NEURON-001)
     * HOW:  Use nimcp_aligned_alloc(16, size) instead of nimcp_calloc
     */
    size_t synapses_size = network->synapses_capacity * sizeof(gpu_synapse_t);
    network->synapses_host = (gpu_synapse_t*)nimcp_aligned_alloc(16, synapses_size);
    if (!network->synapses_host) {
        nimcp_free(network->neurons_host);
        nimcp_free(network);
        return NULL;
    }
    // Zero-initialize the aligned memory
    memset(network->synapses_host, 0, synapses_size);

    // Try to allocate GPU memory if CUDA available
    network->using_gpu = false;
#ifdef NIMCP_ENABLE_CUDA
    if (config->exec_mode == EXEC_MODE_GPU_CUDA || config->exec_mode == EXEC_MODE_HYBRID) {
        // Allocate device memory
        cudaError_t err;

        err = cudaMalloc(&network->neurons_device,
                        network->neurons_capacity * sizeof(gpu_neuron_state_t));
        if (err != cudaSuccess) {
            goto cpu_fallback;
        }

        err = cudaMalloc(&network->synapses_device,
                        network->synapses_capacity * sizeof(gpu_synapse_t));
        if (err != cudaSuccess) {
            cudaFree(network->neurons_device);
            goto cpu_fallback;
        }

        // Create CUDA stream for async operations
        err = cudaStreamCreate(&network->stream);
        if (err != cudaSuccess) {
            cudaFree(network->neurons_device);
            cudaFree(network->synapses_device);
            goto cpu_fallback;
        }

        network->using_gpu = true;
        return network;
    }

cpu_fallback:
#endif

    // CPU fallback mode
    network->using_gpu = false;
    return network;
}

NIMCP_EXPORT void gpu_neural_network_destroy(gpu_neural_network_t network)
{
    if (!network) {
        return;
    }

#ifdef NIMCP_ENABLE_CUDA
    if (network->using_gpu) {
        // Synchronize before cleanup
        cudaStreamSynchronize(network->stream);

        // Free device memory
        if (network->neurons_device) {
            cudaFree(network->neurons_device);
        }
        if (network->synapses_device) {
            cudaFree(network->synapses_device);
        }

        // Destroy stream
        cudaStreamDestroy(network->stream);
    }
#endif

    // Free host memory
    nimcp_aligned_free(network->neurons_host);
    nimcp_free(network->synapses_host);
    nimcp_free(network);
}

//=============================================================================
// Neuron/Synapse Operations
//=============================================================================

NIMCP_EXPORT uint32_t gpu_neural_network_add_neuron(
    gpu_neural_network_t network,
    const gpu_neuron_state_t* initial_state)
{
    if (!network || !initial_state) {
        return UINT32_MAX;
    }

    if (network->neurons_count >= network->neurons_capacity) {
        return UINT32_MAX;  // No space
    }

    uint32_t neuron_id = network->neurons_count;
    network->neurons_host[neuron_id] = *initial_state;
    network->neurons_host[neuron_id].neuron_id = neuron_id;
    network->neurons_count++;

    return neuron_id;
}

NIMCP_EXPORT bool gpu_neural_network_add_synapse(
    gpu_neural_network_t network,
    uint32_t source_id,
    uint32_t target_id,
    float weight,
    float strength)
{
    if (!network) {
        return false;
    }

    if (source_id >= network->neurons_count || target_id >= network->neurons_count) {
        return false;
    }

    if (network->synapses_count >= network->synapses_capacity) {
        return false;
    }

    uint32_t synapse_idx = network->synapses_count;
    network->synapses_host[synapse_idx].source_id = source_id;
    network->synapses_host[synapse_idx].target_id = target_id;
    network->synapses_host[synapse_idx].weight = weight;
    network->synapses_host[synapse_idx].strength = strength;
    network->synapses_count++;

    // Update target neuron's incoming synapse count
    network->neurons_host[target_id].num_incoming++;

    return true;
}

//=============================================================================
// Simulation (CPU Fallback Implementation)
//=============================================================================

/**
 * @brief CPU implementation of neuron update
 */
static uint32_t cpu_update_neurons(
    gpu_neural_network_t network,
    uint64_t timestamp,
    uint64_t delta_t)
{
    uint32_t spike_count = 0;

    for (uint32_t i = 0; i < network->neurons_count; i++) {
        gpu_neuron_state_t* neuron = &network->neurons_host[i];

        // Check refractory period
        if (neuron->last_spike > 0 &&
            (timestamp - neuron->last_spike) < neuron->refractory_period) {
            continue;
        }

        // Compute synaptic input
        float synaptic_input = 0.0f;
        for (uint32_t s = 0; s < network->synapses_count; s++) {
            gpu_synapse_t* syn = &network->synapses_host[s];
            if (syn->target_id == i) {
                gpu_neuron_state_t* pre = &network->neurons_host[syn->source_id];
                if (pre->state > pre->threshold) {
                    synaptic_input += pre->state * syn->weight * syn->strength;
                }
            }
        }

        // Update membrane potential
        float potential = neuron->bias + synaptic_input;
        potential *= (1.0f + neuron->calcium_concentration);

        // Store old state
        float old_state = neuron->state;

        // Update state with leaky integration
        float decay = expf(-(float)delta_t / 20000.0f);  // 20ms time constant
        neuron->membrane_potential = neuron->membrane_potential * decay + potential * (1.0f - decay);
        neuron->state = neuron->membrane_potential;

        // Check for spike
        if (old_state <= neuron->threshold && neuron->state > neuron->threshold) {
            neuron->last_spike = timestamp;
            neuron->spike_count++;
            spike_count++;

            // Update calcium concentration
            neuron->calcium_concentration += 0.1f;
        }

        // Decay calcium
        neuron->calcium_concentration *= 0.99f;
    }

    return spike_count;
}

NIMCP_EXPORT uint32_t gpu_neural_network_update(
    gpu_neural_network_t network,
    uint64_t timestamp,
    uint64_t delta_t)
{
    if (!network) {
        return 0;
    }

    uint32_t spike_count = 0;

#ifdef NIMCP_ENABLE_CUDA
    if (network->using_gpu) {
        // TODO: Launch CUDA kernel
        // For now, fall back to CPU
        spike_count = cpu_update_neurons(network, timestamp, delta_t);
    } else
#endif
    {
        // CPU fallback
        spike_count = cpu_update_neurons(network, timestamp, delta_t);
    }

    network->total_spikes += spike_count;
    network->total_updates++;

    return spike_count;
}

NIMCP_EXPORT uint32_t gpu_neural_network_apply_stdp(
    gpu_neural_network_t network,
    uint64_t timestamp)
{
    if (!network || !network->config.enable_stdp) {
        return 0;
    }

    (void)timestamp;  // Not used in basic implementation

    // Simple STDP implementation
    uint32_t modified_count = 0;
    float learning_rate = network->config.global_learning_rate;

    for (uint32_t i = 0; i < network->synapses_count; i++) {
        gpu_synapse_t* syn = &network->synapses_host[i];
        gpu_neuron_state_t* pre = &network->neurons_host[syn->source_id];
        gpu_neuron_state_t* post = &network->neurons_host[syn->target_id];

        if (pre->last_spike > 0 && post->last_spike > 0) {
            int64_t delta_t = (int64_t)post->last_spike - (int64_t)pre->last_spike;

            if (delta_t > 0 && delta_t < 20000) {  // Pre before post (LTP)
                syn->weight += learning_rate * expf(-(float)delta_t / 20000.0f);
                modified_count++;
            } else if (delta_t < 0 && delta_t > -20000) {  // Post before pre (LTD)
                syn->weight -= learning_rate * expf((float)delta_t / 20000.0f);
                modified_count++;
            }

            // Clamp weights
            if (syn->weight < 0.0f) syn->weight = 0.0f;
            if (syn->weight > 1.0f) syn->weight = 1.0f;
        }
    }

    return modified_count;
}

NIMCP_EXPORT bool gpu_neural_network_synchronize(gpu_neural_network_t network)
{
    if (!network) {
        return false;
    }

#ifdef NIMCP_ENABLE_CUDA
    if (network->using_gpu) {
        cudaError_t err = cudaStreamSynchronize(network->stream);
        return (err == cudaSuccess);
    }
#endif

    return true;  // CPU mode always synchronized
}

//=============================================================================
// Data Access
//=============================================================================

NIMCP_EXPORT bool gpu_neural_network_get_neuron_state(
    gpu_neural_network_t network,
    uint32_t neuron_id,
    gpu_neuron_state_t* state)
{
    if (!network || !state || neuron_id >= network->neurons_count) {
        return false;
    }

    *state = network->neurons_host[neuron_id];
    return true;
}

NIMCP_EXPORT bool gpu_neural_network_set_neuron_state(
    gpu_neural_network_t network,
    uint32_t neuron_id,
    const gpu_neuron_state_t* state)
{
    if (!network || !state || neuron_id >= network->neurons_count) {
        return false;
    }

    network->neurons_host[neuron_id] = *state;
    return true;
}

NIMCP_EXPORT uint32_t gpu_neural_network_get_all_states(
    gpu_neural_network_t network,
    gpu_neuron_state_t* states,
    uint32_t max_neurons)
{
    if (!network || !states) {
        return 0;
    }

    uint32_t count = (max_neurons < network->neurons_count) ? max_neurons : network->neurons_count;
    memcpy(states, network->neurons_host, count * sizeof(gpu_neuron_state_t));

    return count;
}

//=============================================================================
// Statistics
//=============================================================================

NIMCP_EXPORT bool gpu_neural_network_get_stats(
    gpu_neural_network_t network,
    uint64_t* total_spikes,
    float* avg_firing_rate,
    uint64_t* gpu_memory_used)
{
    if (!network) {
        return false;
    }

    if (total_spikes) {
        *total_spikes = network->total_spikes;
    }

    if (avg_firing_rate) {
        if (network->total_updates > 0 && network->neurons_count > 0) {
            *avg_firing_rate = (float)network->total_spikes /
                              (float)(network->total_updates * network->neurons_count);
        } else {
            *avg_firing_rate = 0.0f;
        }
    }

    if (gpu_memory_used) {
        *gpu_memory_used = 0;
#ifdef NIMCP_ENABLE_CUDA
        if (network->using_gpu) {
            *gpu_memory_used = (network->neurons_capacity * sizeof(gpu_neuron_state_t)) +
                              (network->synapses_capacity * sizeof(gpu_synapse_t));
        }
#endif
    }

    return true;
}
