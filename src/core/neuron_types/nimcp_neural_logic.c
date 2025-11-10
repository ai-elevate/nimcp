/**
 * @file nimcp_neural_logic.c  
 * @brief Neural Logic Implementation (GPU + CPU Fallback)
 *
 * WHAT: Spiking neural logic gates with GPU acceleration
 * WHY:  Replace symbolic engine with 100x faster neural logic
 * HOW:  CUDA kernels for GPU, optimized loops for CPU
 *
 * @author NIMCP Development Team
 * @date 2025-11-08
 * @version 2.7.0 Phase 9.0
 */

#include "nimcp_neural_logic.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/validation/nimcp_validate.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"  // Neuromodulator integration
#include "core/brain/nimcp_brain.h"  // Brain reference
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef NIMCP_ENABLE_CUDA
#include <cuda_runtime.h>

// GPU kernel launch wrappers (defined in nimcp_neural_logic_kernels.cu)
extern cudaError_t launch_update_logic_neurons(
    logic_neuron_state_t* neurons_device,
    const float* input_activities_device,
    uint32_t num_neurons,
    uint64_t timestamp,
    uint64_t delta_t,
    uint32_t threads_per_block,
    cudaStream_t stream
);

extern cudaError_t launch_update_variable_bindings(
    variable_binding_state_t* variables_device,
    const float* patterns_device,
    uint32_t num_variables,
    uint32_t pattern_dim,
    uint64_t timestamp,
    uint64_t delta_t,
    uint32_t threads_per_block,
    cudaStream_t stream
);
#endif

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Neural logic network internal structure
 */
struct neural_logic_network_struct {
    // Configuration
    neural_logic_config_t config;

    // Logic neurons (host-side copy)
    logic_neuron_state_t* neurons_host;
    uint32_t neurons_count;
    uint32_t neurons_capacity;

    // Variable bindings (host-side copy)
    variable_binding_state_t* variables_host;
    uint32_t variables_count;
    uint32_t variables_capacity;

    // Neuromodulation
    brain_t brain;  /**< Brain reference for DA + ACh modulation */
    
    // GPU pointers (NULL if CPU fallback)
#ifdef NIMCP_ENABLE_CUDA
    logic_neuron_state_t* neurons_device;
    variable_binding_state_t* variables_device;
    float* input_activities_device;
    cudaStream_t stream;
#endif
    
    // Execution mode
    bool using_gpu;
    
    // Statistics
    uint64_t total_spikes;
    uint64_t total_evaluations;
    float sum_eval_time_us;
};

//=============================================================================
// GPU Detection
//=============================================================================

NIMCP_EXPORT bool neural_logic_gpu_available(void)
{
#ifdef NIMCP_ENABLE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

//=============================================================================
// Configuration
//=============================================================================

NIMCP_EXPORT neural_logic_config_t neural_logic_default_config(uint32_t max_neurons)
{
    neural_logic_config_t config = {0};
    
    config.max_logic_neurons = max_neurons;
    config.max_variables = max_neurons / 4;  // 25% can be variables
    config.variable_pattern_dim = 64;        // 64-dimensional patterns
    
    // GPU configuration
    config.threads_per_block = 256;          // Optimal for most GPUs
    config.use_gpu = neural_logic_gpu_available();
    config.pin_host_memory = true;
    
    // Temporal parameters
    config.timestep_us = 100.0f;             // 100μs = 0.1ms
    config.integration_window_ms = 5.0f;     // 5ms integration window
    
    // Learning configuration
    config.enable_learning = false;          // Disable for now
    config.learning_rate = 0.01f;
    
    return config;
}

//=============================================================================
// Network Lifecycle
//=============================================================================

NIMCP_EXPORT neural_logic_network_t neural_logic_create(
    const neural_logic_config_t* config)
{
    if (!nimcp_validate_pointer(config, "config")) {
        return NULL;
    }
    
    if (config->max_logic_neurons == 0) {
        NIMCP_LOGGING_ERROR("max_logic_neurons must be > 0");
        return NULL;
    }
    
    // Allocate network structure
    neural_logic_network_t network = nimcp_calloc(1, sizeof(struct neural_logic_network_struct));
    if (!nimcp_validate_pointer(network, "network")) {
        return NULL;
    }
    
    // Copy configuration
    network->config = *config;
    network->neurons_capacity = config->max_logic_neurons;
    network->variables_capacity = config->max_variables;
    network->neurons_count = 0;
    network->variables_count = 0;
    network->total_spikes = 0;
    network->total_evaluations = 0;
    network->sum_eval_time_us = 0.0f;
    network->brain = NULL;  // Initialize neuromodulator brain reference
    
    // Allocate host memory for neurons
    network->neurons_host = nimcp_calloc(network->neurons_capacity, 
                                         sizeof(logic_neuron_state_t));
    if (!nimcp_validate_pointer(network->neurons_host, "neurons_host")) {
        nimcp_free(network);
        return NULL;
    }
    
    // Allocate host memory for variables
    network->variables_host = nimcp_calloc(network->variables_capacity,
                                           sizeof(variable_binding_state_t));
    if (!nimcp_validate_pointer(network->variables_host, "variables_host")) {
        nimcp_free(network->neurons_host);
        nimcp_free(network);
        return NULL;
    }
    
    // Try to allocate GPU memory if enabled
    network->using_gpu = false;
#ifdef NIMCP_ENABLE_CUDA
    if (config->use_gpu && neural_logic_gpu_available()) {
        cudaError_t err;
        
        // Allocate device memory for neurons
        err = cudaMalloc(&network->neurons_device,
                        network->neurons_capacity * sizeof(logic_neuron_state_t));
        if (err != cudaSuccess) {
            NIMCP_LOGGING_WARN("GPU malloc failed, using CPU fallback");
            goto cpu_fallback;
        }
        
        // Allocate device memory for variables
        err = cudaMalloc(&network->variables_device,
                        network->variables_capacity * sizeof(variable_binding_state_t));
        if (err != cudaSuccess) {
            NIMCP_LOGGING_WARN("GPU malloc failed, using CPU fallback");
            cudaFree(network->neurons_device);
            goto cpu_fallback;
        }
        
        // Create CUDA stream
        err = cudaStreamCreate(&network->stream);
        if (err != cudaSuccess) {
            NIMCP_LOGGING_WARN("GPU stream creation failed, using CPU fallback");
            cudaFree(network->neurons_device);
            cudaFree(network->variables_device);
            goto cpu_fallback;
        }
        
        network->using_gpu = true;
        NIMCP_LOGGING_INFO("Neural logic using GPU acceleration");
    }
    
cpu_fallback:
#endif
    
    if (!network->using_gpu) {
        NIMCP_LOGGING_INFO("Neural logic using CPU fallback");
    }
    
    return network;
}

NIMCP_EXPORT void neural_logic_destroy(neural_logic_network_t network)
{
    if (!network) {
        return;
    }
    
    // Free GPU memory if allocated
#ifdef NIMCP_ENABLE_CUDA
    if (network->using_gpu) {
        cudaStreamSynchronize(network->stream);
        cudaStreamDestroy(network->stream);
        cudaFree(network->neurons_device);
        cudaFree(network->variables_device);
        if (network->input_activities_device) {
            cudaFree(network->input_activities_device);
        }
    }
#endif
    
    // Free host memory
    nimcp_free(network->variables_host);
    nimcp_free(network->neurons_host);
    nimcp_free(network);
}

//=============================================================================
// Logic Neuron Creation
//=============================================================================

NIMCP_EXPORT uint32_t neural_logic_create_gate(
    neural_logic_network_t network,
    logic_gate_type_t gate_type,
    float threshold)
{
    if (!nimcp_validate_pointer(network, "network")) {
        return UINT32_MAX;
    }
    
    if (network->neurons_count >= network->neurons_capacity) {
        NIMCP_LOGGING_ERROR("Maximum neurons reached");
        return UINT32_MAX;
    }
    
    if (gate_type >= LOGIC_GATE_COUNT) {
        NIMCP_LOGGING_ERROR("Invalid gate type: %d", gate_type);
        return UINT32_MAX;
    }
    
    // Get next neuron ID
    uint32_t neuron_id = network->neurons_count++;
    logic_neuron_state_t* neuron = &network->neurons_host[neuron_id];
    
    // Initialize neuron state
    memset(neuron, 0, sizeof(logic_neuron_state_t));
    neuron->neuron_id = neuron_id;
    neuron->gate_type = gate_type;
    neuron->membrane_potential = 0.0f;
    neuron->output_state = 0.0f;
    neuron->integration_window = network->config.integration_window_ms;
    
    // Set gate-specific parameters
    switch (gate_type) {
        case LOGIC_GATE_AND:
            // AND: Requires both inputs (coincidence detection)
            neuron->threshold = (threshold > 0.0f) ? threshold : 1.8f;  // Slightly less than 2.0
            neuron->excitatory_weight = 1.0f;
            neuron->inhibitory_weight = 0.0f;
            neuron->refractory_period = 1000;  // 1ms
            break;
            
        case LOGIC_GATE_OR:
            // OR: Fires if any input active (low threshold)
            neuron->threshold = (threshold > 0.0f) ? threshold : 0.6f;  // Less than 1.0
            neuron->excitatory_weight = 1.0f;
            neuron->inhibitory_weight = 0.0f;
            neuron->refractory_period = 500;   // 0.5ms
            break;
            
        case LOGIC_GATE_NOT:
            // NOT: Baseline firing, inhibited by input
            neuron->threshold = (threshold > 0.0f) ? threshold : 0.5f;
            neuron->excitatory_weight = 0.0f;
            neuron->inhibitory_weight = -2.0f;  // Strong inhibition
            neuron->membrane_potential = 1.0f;  // Baseline depolarization
            neuron->refractory_period = 1000;   // 1ms
            break;
            
        case LOGIC_GATE_XOR:
            // XOR: Fires if inputs differ (balanced)
            neuron->threshold = (threshold > 0.0f) ? threshold : 0.5f;
            neuron->excitatory_weight = 1.0f;
            neuron->inhibitory_weight = -1.0f;
            neuron->refractory_period = 1000;   // 1ms
            break;
            
        case LOGIC_GATE_IMPLIES:
            // IMPLIES: A → B (fires if A=0 OR B=1)
            neuron->threshold = (threshold > 0.0f) ? threshold : 0.8f;
            neuron->excitatory_weight = 1.0f;
            neuron->inhibitory_weight = -0.5f;
            neuron->refractory_period = 1000;   // 1ms
            break;
            
        default:
            NIMCP_LOGGING_ERROR("Unhandled gate type: %d", gate_type);
            network->neurons_count--;  // Rollback
            return UINT32_MAX;
    }
    
    return neuron_id;
}

NIMCP_EXPORT uint32_t neural_logic_create_variable(
    neural_logic_network_t network,
    const char* variable_name)
{
    if (!nimcp_validate_pointer(network, "network") ||
        !nimcp_validate_pointer(variable_name, "variable_name")) {
        return UINT32_MAX;
    }
    
    if (network->variables_count >= network->variables_capacity) {
        NIMCP_LOGGING_ERROR("Maximum variables reached");
        return UINT32_MAX;
    }
    
    // Simple hash of variable name
    uint32_t hash = 0;
    for (const char* p = variable_name; *p; p++) {
        hash = hash * 31 + (uint32_t)(*p);
    }
    
    // Get next variable ID
    uint32_t var_id = network->variables_count++;
    variable_binding_state_t* var = &network->variables_host[var_id];
    
    // Initialize variable state
    memset(var, 0, sizeof(variable_binding_state_t));
    var->variable_id = hash;
    var->pattern_dim = network->config.variable_pattern_dim;
    var->binding_strength = 0.0f;
    var->decay_rate = 0.001f;  // Slow decay
    var->is_bound = false;
    var->bound_pattern = NULL;  // Allocated on first bind
    
    return var_id;
}

//=============================================================================
// Neuromodulation
//=============================================================================

/**
 * @brief Compute logic gate threshold modulation from neurotransmitters
 *
 * WHAT: Calculate threshold modulation factor based on DA + ACh
 * WHY:  DA modulates logical flexibility, ACh modulates precision
 * HOW:  Read DA and ACh levels, combine into threshold multiplier
 *
 * BIOLOGY:
 * - Dopamine: Logical flexibility vs rigidity
 *   High DA (0.7) → 0.7× threshold (permissive logic, exploratory)
 *   Low DA (0.3) → 1.3× threshold (rigid logic, perseverative)
 *
 * - Acetylcholine: Logical precision
 *   High ACh (0.7) → precise thresholds (accurate logic)
 *   Low ACh (0.3) → imprecise thresholds (sloppy logic, errors)
 *
 * CLINICAL EXAMPLES:
 * - Depression (low DA): Rigid, black-and-white thinking
 * - Mania (high DA): Loose associations, illogical leaps
 * - ADHD (low ACh): Logical errors, misses contradictions
 * - Dementia (low ACh): Impaired reasoning, confabulation
 *
 * COMPLEXITY: O(1)
 *
 * @param brain Brain to read neurotransmitters from
 * @return Threshold multiplier [0.7, 1.3], or 1.0 if no brain
 */
static float get_logic_threshold_modulation(brain_t brain)
{
    // Guard: No brain available
    if (!brain) {
        return 1.0f;
    }

    neuromodulator_system_t neuromod = brain_get_neuromodulator_system(brain);
    if (!neuromod) {
        return 1.0f;
    }

    // Read neurotransmitter levels
    float da = neuromodulator_get_level(neuromod, NEUROMOD_DOPAMINE);
    float ach = neuromodulator_get_level(neuromod, NEUROMOD_ACETYLCHOLINE);

    // DA contribution: [0.3, 0.7] → [1.3, 0.7] (inverse)
    // High DA → lower threshold (permissive, exploratory)
    // Low DA → higher threshold (rigid, perseverative)
    float da_multiplier = 1.3f - (da - 0.3f) * 1.5f;

    // ACh contribution: [0.3, 0.7] → precision scaling
    // High ACh → maintains intended thresholds
    // Low ACh → adds noise/imprecision to reasoning
    // For simplicity, ACh primarily affects precision of DA effect
    float ach_precision = 0.8f + (ach - 0.3f) * 0.5f;  // [0.8, 1.0]

    // Combined: DA sets flexibility, ACh sharpens precision
    // Range: approximately [0.7, 1.3]
    return da_multiplier * ach_precision;
}

//=============================================================================
// CPU Implementation (Fallback)
//=============================================================================

/**
 * @brief CPU implementation of AND gate
 */
static inline float cpu_compute_and_gate(
    float input_a,
    float input_b,
    float threshold,
    float integration_window)
{
    (void)integration_window;  // Unused in simplified model
    
    float sum = input_a + input_b;
    return (sum >= threshold) ? 1.0f : 0.0f;
}

/**
 * @brief CPU implementation of OR gate
 */
static inline float cpu_compute_or_gate(
    float input_a,
    float input_b,
    float threshold)
{
    float sum = input_a + input_b;
    return (sum >= threshold) ? 1.0f : 0.0f;
}

/**
 * @brief CPU implementation of NOT gate
 */
static inline float cpu_compute_not_gate(
    float input,
    float baseline_rate,
    float inhibition_strength)
{
    float output = baseline_rate - (input * inhibition_strength);
    return (output > 0.5f) ? 1.0f : 0.0f;
}

/**
 * @brief CPU implementation of XOR gate
 */
static inline float cpu_compute_xor_gate(
    float input_a,
    float input_b,
    float threshold,
    float balance_tolerance)
{
    (void)balance_tolerance;  // Unused in simplified model
    
    float diff = fabsf(input_a - input_b);
    return (diff >= threshold) ? 1.0f : 0.0f;
}

/**
 * @brief CPU implementation of IMPLIES gate
 */
static inline float cpu_compute_implies_gate(
    float input_a,
    float input_b,
    float antecedent_threshold,
    float consequent_threshold)
{
    // A → B  ≡  ¬A ∨ B
    bool a_active = (input_a >= antecedent_threshold);
    bool b_active = (input_b >= consequent_threshold);
    
    return (!a_active || b_active) ? 1.0f : 0.0f;
}

/**
 * @brief CPU update of single logic neuron
 */
static void cpu_update_logic_neuron(
    logic_neuron_state_t* neuron,
    uint64_t timestamp,
    uint64_t delta_t,
    float threshold_modulation)
{
    // Check refractory period
    if ((timestamp - neuron->last_spike_time) < neuron->refractory_period) {
        return;  // In refractory period
    }

    // Apply neuromodulator modulation to threshold
    float modulated_threshold = neuron->threshold * threshold_modulation;

    // Compute gate output based on type
    float output = 0.0f;

    switch (neuron->gate_type) {
        case LOGIC_GATE_AND:
            output = cpu_compute_and_gate(
                neuron->input_a_activity,
                neuron->input_b_activity,
                modulated_threshold,
                neuron->integration_window
            );
            break;
            
        case LOGIC_GATE_OR:
            output = cpu_compute_or_gate(
                neuron->input_a_activity,
                neuron->input_b_activity,
                modulated_threshold
            );
            break;

        case LOGIC_GATE_NOT:
            output = cpu_compute_not_gate(
                neuron->input_a_activity,
                1.0f,  // baseline_rate
                fabsf(neuron->inhibitory_weight)
            );
            break;

        case LOGIC_GATE_XOR:
            output = cpu_compute_xor_gate(
                neuron->input_a_activity,
                neuron->input_b_activity,
                modulated_threshold,
                0.1f  // balance_tolerance
            );
            break;
            
        case LOGIC_GATE_IMPLIES:
            output = cpu_compute_implies_gate(
                neuron->input_a_activity,
                neuron->input_b_activity,
                0.8f,  // antecedent_threshold
                0.8f   // consequent_threshold
            );
            break;
            
        default:
            output = 0.0f;
            break;
    }
    
    // Update output state
    neuron->output_state = output;
    
    // Fire spike if output is true
    if (output > 0.5f) {
        neuron->last_spike_time = timestamp;
        neuron->total_spikes++;
        neuron->true_outputs++;
    } else {
        neuron->false_outputs++;
    }
    
    // Decay input activities
    float decay = (float)delta_t / 1000.0f;  // Convert μs to ms
    neuron->input_a_activity *= expf(-decay / neuron->integration_window);
    neuron->input_b_activity *= expf(-decay / neuron->integration_window);
}

//=============================================================================
// Simulation
//=============================================================================

NIMCP_EXPORT uint32_t neural_logic_update(
    neural_logic_network_t network,
    uint64_t timestamp,
    uint64_t delta_t)
{
    if (!nimcp_validate_pointer(network, "network")) {
        return 0;
    }
    
    uint32_t spikes_count = 0;
    
#ifdef NIMCP_ENABLE_CUDA
    if (network->using_gpu) {
        // Copy host neurons to device
        cudaError_t err = cudaMemcpyAsync(
            network->neurons_device,
            network->neurons_host,
            network->neurons_count * sizeof(logic_neuron_state_t),
            cudaMemcpyHostToDevice,
            network->stream
        );

        if (err != cudaSuccess) {
            NIMCP_LOGGING_ERROR("Failed to copy neurons to GPU: %s", cudaGetErrorString(err));
            goto cpu_path;
        }

        // Launch GPU kernel
        err = launch_update_logic_neurons(
            network->neurons_device,
            network->input_activities_device,
            network->neurons_count,
            timestamp,
            delta_t,
            network->config.threads_per_block,
            network->stream
        );

        if (err != cudaSuccess) {
            NIMCP_LOGGING_ERROR("GPU kernel launch failed: %s", cudaGetErrorString(err));
            goto cpu_path;
        }

        // Copy results back from device to host
        err = cudaMemcpyAsync(
            network->neurons_host,
            network->neurons_device,
            network->neurons_count * sizeof(logic_neuron_state_t),
            cudaMemcpyDeviceToHost,
            network->stream
        );

        if (err != cudaSuccess) {
            NIMCP_LOGGING_ERROR("Failed to copy neurons from GPU: %s", cudaGetErrorString(err));
            goto cpu_path;
        }

        // Synchronize stream
        cudaStreamSynchronize(network->stream);

        // Count spikes from GPU results
        for (uint32_t i = 0; i < network->neurons_count; i++) {
            logic_neuron_state_t* neuron = &network->neurons_host[i];
            if (neuron->last_spike_time == timestamp) {
                spikes_count++;
            }
        }

        network->total_spikes += spikes_count;
        return spikes_count;
    }

cpu_path:
#endif

    // Compute neuromodulator threshold modulation
    float threshold_modulation = get_logic_threshold_modulation(network->brain);

    // CPU fallback: Update all neurons sequentially
    for (uint32_t i = 0; i < network->neurons_count; i++) {
        logic_neuron_state_t* neuron = &network->neurons_host[i];
        uint32_t spikes_before = neuron->total_spikes;

        cpu_update_logic_neuron(neuron, timestamp, delta_t, threshold_modulation);

        if (neuron->total_spikes > spikes_before) {
            spikes_count++;
        }
    }

    network->total_spikes += spikes_count;

    return spikes_count;
}

NIMCP_EXPORT bool neural_logic_synchronize(neural_logic_network_t network)
{
    if (!nimcp_validate_pointer(network, "network")) {
        return false;
    }
    
#ifdef NIMCP_ENABLE_CUDA
    if (network->using_gpu) {
        cudaStreamSynchronize(network->stream);
        return true;
    }
#endif
    
    return true;  // CPU is always synchronized
}

//=============================================================================
// High-Level Evaluation
//=============================================================================

NIMCP_EXPORT bool neural_logic_evaluate(
    neural_logic_network_t network,
    uint32_t gate_id,
    const float* inputs,
    uint32_t num_inputs,
    float* output)
{
    if (!nimcp_validate_pointer(network, "network") ||
        !nimcp_validate_pointer(inputs, "inputs") ||
        !nimcp_validate_pointer(output, "output")) {
        return false;
    }
    
    if (gate_id >= network->neurons_count) {
        NIMCP_LOGGING_ERROR("Invalid gate_id: %u >= %u", gate_id, network->neurons_count);
        return false;
    }
    
    logic_neuron_state_t* neuron = &network->neurons_host[gate_id];

    // Directly compute gate output (combinational logic, no temporal dynamics needed)
    float input_a = (num_inputs >= 1) ? inputs[0] : 0.0f;
    float input_b = (num_inputs >= 2) ? inputs[1] : 0.0f;

    float result = 0.0f;

    switch (neuron->gate_type) {
        case LOGIC_GATE_AND:
            result = cpu_compute_and_gate(input_a, input_b, neuron->threshold, neuron->integration_window);
            break;

        case LOGIC_GATE_OR:
            result = cpu_compute_or_gate(input_a, input_b, neuron->threshold);
            break;

        case LOGIC_GATE_NOT:
            result = cpu_compute_not_gate(input_a, 1.0f, fabsf(neuron->inhibitory_weight));
            break;

        case LOGIC_GATE_XOR:
            result = cpu_compute_xor_gate(input_a, input_b, neuron->threshold, 0.1f);
            break;

        case LOGIC_GATE_IMPLIES:
            result = cpu_compute_implies_gate(input_a, input_b, 0.8f, 0.8f);
            break;

        default:
            result = 0.0f;
            break;
    }

    // Return output
    *output = result;
    network->total_evaluations++;

    return true;
}

//=============================================================================
// Statistics
//=============================================================================

NIMCP_EXPORT bool neural_logic_get_state(
    neural_logic_network_t network,
    uint32_t neuron_id,
    logic_neuron_state_t* state)
{
    if (!nimcp_validate_pointer(network, "network") ||
        !nimcp_validate_pointer(state, "state")) {
        return false;
    }
    
    if (neuron_id >= network->neurons_count) {
        return false;
    }
    
    // Copy from host memory
    *state = network->neurons_host[neuron_id];
    
    return true;
}

NIMCP_EXPORT bool neural_logic_get_stats(
    neural_logic_network_t network,
    uint32_t* total_gates,
    uint32_t* total_variables,
    uint64_t* total_spikes,
    float* avg_eval_time,
    uint64_t* gpu_memory_used)
{
    if (!nimcp_validate_pointer(network, "network")) {
        return false;
    }
    
    if (total_gates) *total_gates = network->neurons_count;
    if (total_variables) *total_variables = network->variables_count;
    if (total_spikes) *total_spikes = network->total_spikes;
    
    if (avg_eval_time) {
        *avg_eval_time = (network->total_evaluations > 0) ?
            (network->sum_eval_time_us / network->total_evaluations) : 0.0f;
    }
    
    if (gpu_memory_used) {
        *gpu_memory_used = 0;
#ifdef NIMCP_ENABLE_CUDA
        if (network->using_gpu) {
            *gpu_memory_used = 
                (network->neurons_capacity * sizeof(logic_neuron_state_t)) +
                (network->variables_capacity * sizeof(variable_binding_state_t));
        }
#endif
    }
    
    return true;
}

//=============================================================================
// Utility Functions
//=============================================================================

NIMCP_EXPORT const char* neural_logic_gate_name(logic_gate_type_t gate_type)
{
    switch (gate_type) {
        case LOGIC_GATE_AND: return "AND";
        case LOGIC_GATE_OR: return "OR";
        case LOGIC_GATE_NOT: return "NOT";
        case LOGIC_GATE_XOR: return "XOR";
        case LOGIC_GATE_IMPLIES: return "IMPLIES";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Associate brain with neural logic network for neuromodulation
 *
 * WHAT: Set brain reference for DA + ACh modulation of logic gates
 * WHY:  Enable neurochemical modulation (logical flexibility, precision)
 * HOW:  Store brain pointer for neurotransmitter reading
 *
 * BIOLOGY:
 * - Dopamine modulates logical flexibility vs rigidity
 * - Acetylcholine modulates logical precision
 *
 * @param network Neural logic network instance
 * @param brain Brain instance (or NULL to clear)
 */
NIMCP_EXPORT void neural_logic_set_brain(neural_logic_network_t network, brain_t brain)
{
    if (!network) {
        return;
    }
    network->brain = brain;
}

//=============================================================================
// Variable Binding (Stub Implementation)
//=============================================================================

NIMCP_EXPORT bool neural_logic_bind_variable(
    neural_logic_network_t network,
    uint32_t variable_id,
    const float* pattern,
    float binding_strength)
{
    if (!nimcp_validate_pointer(network, "network") ||
        !nimcp_validate_pointer(pattern, "pattern")) {
        return false;
    }
    
    if (variable_id >= network->variables_count) {
        return false;
    }
    
    // TODO: Implement variable binding
    NIMCP_LOGGING_WARN("Variable binding not yet implemented");
    return true;
}

NIMCP_EXPORT bool neural_logic_query_variable(
    neural_logic_network_t network,
    uint32_t variable_id,
    float* pattern,
    uint32_t pattern_dim)
{
    if (!nimcp_validate_pointer(network, "network") ||
        !nimcp_validate_pointer(pattern, "pattern")) {
        return false;
    }
    
    if (variable_id >= network->variables_count) {
        return false;
    }
    
    // TODO: Implement variable querying
    NIMCP_LOGGING_WARN("Variable querying not yet implemented");
    return false;
}

NIMCP_EXPORT bool neural_logic_connect(
    neural_logic_network_t network,
    uint32_t source_id,
    uint32_t target_id,
    float weight)
{
    if (!nimcp_validate_pointer(network, "network")) {
        return false;
    }
    
    // TODO: Implement synapse connectivity
    NIMCP_LOGGING_WARN("Synapse connectivity not yet implemented");
    (void)source_id;
    (void)target_id;
    (void)weight;
    return true;
}
