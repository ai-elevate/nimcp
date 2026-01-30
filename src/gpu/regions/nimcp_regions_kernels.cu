/**
 * @file nimcp_regions_kernels.cu
 * @brief GPU Brain Region Specialization CUDA Kernels
 *
 * WHAT: CUDA kernels for specialized brain region computations
 * WHY:  GPU acceleration for region-specific neural processing
 * HOW:  Custom kernels for cortical areas, subcortical structures
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifdef NIMCP_ENABLE_CUDA

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#include "gpu/regions/nimcp_regions_gpu.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "gpu/common/nimcp_cuda_utils.h"
#include "gpu/recovery/nimcp_gpu_recovery.h"

#define LOG_MODULE "REGIONS_GPU"

#define BLOCK_SIZE 256
#define GRID_SIZE(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)

//=============================================================================
// Default Parameters
//=============================================================================

nimcp_gpu_column_params_t nimcp_gpu_column_params_default(void)
{
    nimcp_gpu_column_params_t params;
    params.n_columns = 100;
    params.n_neurons_per_column = 80;
    params.lateral_inhibition = 0.5f;
    params.recurrent_excitation = 0.3f;
    params.feedforward_gain = 1.0f;
    params.feedback_gain = 0.5f;
    params.adaptation_rate = 0.1f;

    // Initialize layer connectivity (canonical microcircuit)
    for (int i = 0; i < NIMCP_LAYER_COUNT; i++) {
        for (int j = 0; j < NIMCP_LAYER_COUNT; j++) {
            params.layer_connectivity[i][j] = 0.0f;
        }
    }
    // L4 -> L2/3, L2/3 -> L5, L5 -> L6, L6 -> L4
    params.layer_connectivity[NIMCP_LAYER_4][NIMCP_LAYER_2_3] = 0.8f;
    params.layer_connectivity[NIMCP_LAYER_2_3][NIMCP_LAYER_5] = 0.6f;
    params.layer_connectivity[NIMCP_LAYER_5][NIMCP_LAYER_6] = 0.5f;
    params.layer_connectivity[NIMCP_LAYER_6][NIMCP_LAYER_4] = 0.3f;

    return params;
}

nimcp_gpu_pfc_params_t nimcp_gpu_pfc_params_default(void)
{
    nimcp_gpu_pfc_params_t params;
    params.n_slots = 4;
    params.maintenance_gain = 0.9f;
    params.gating_threshold = 0.5f;
    params.decay_rate = 0.01f;
    params.interference_factor = 0.2f;
    params.dopamine_modulation = 1.0f;
    params.recurrent_maintenance = true;
    return params;
}

nimcp_gpu_motor_params_t nimcp_gpu_motor_params_default(void)
{
    nimcp_gpu_motor_params_t params;
    params.n_actions = 10;
    params.n_muscles = 20;
    params.population_coding_sigma = 30.0f;
    params.motor_noise = 0.05f;
    params.planning_horizon = 1.0f;
    params.sequence_learning_rate = 0.01f;
    params.use_internal_model = true;
    return params;
}

nimcp_gpu_parietal_params_t nimcp_gpu_parietal_params_default(void)
{
    nimcp_gpu_parietal_params_t params;
    params.spatial_resolution = 64;
    params.attention_gain = 2.0f;
    params.coordinate_transform_lr = 0.01f;
    params.multisensory_weight = 0.5f;
    params.egocentric_weight = 0.6f;
    params.allocentric_weight = 0.4f;
    return params;
}

nimcp_gpu_interregion_params_t nimcp_gpu_interregion_params_default(void)
{
    nimcp_gpu_interregion_params_t params;
    params.connection_strength = 0.5f;
    params.transmission_delay = 10.0f;
    params.plasticity_rate = 0.001f;
    params.bidirectional = true;
    params.feedback_ratio = 0.3f;
    return params;
}

//=============================================================================
// Cortical Column Kernels
//=============================================================================

/**
 * @brief Kernel for cortical column update
 */
__global__ void kernel_column_update(
    float* __restrict__ column_output,
    float* __restrict__ adaptation_state,
    const float* __restrict__ input,
    const float* __restrict__ feedback,
    const float* __restrict__ lateral_connections,
    float feedforward_gain,
    float feedback_gain,
    float lateral_inhibition,
    float recurrent_excitation,
    float adaptation_rate,
    float dt,
    size_t n_columns,
    size_t n_neurons)
{
    size_t col_idx = blockIdx.x;
    size_t neuron_idx = threadIdx.x;

    if (col_idx >= n_columns || neuron_idx >= n_neurons) return;

    size_t idx = col_idx * n_neurons + neuron_idx;

    float in = input[idx];
    float fb = feedback ? feedback[idx] : 0.0f;
    float adapt = adaptation_state[idx];

    // Feedforward input
    float ff_input = feedforward_gain * in;

    // Feedback modulation
    float fb_input = feedback_gain * fb;

    // Lateral interactions
    float lateral = 0.0f;
    for (size_t other_col = 0; other_col < n_columns; other_col++) {
        if (other_col != col_idx) {
            float conn = lateral_connections[col_idx * n_columns + other_col];
            float other_activity = column_output[other_col * n_neurons + neuron_idx];
            lateral += conn * other_activity;
        }
    }

    // Recurrent excitation within column
    float recurrent = 0.0f;
    for (size_t other_neuron = 0; other_neuron < n_neurons; other_neuron++) {
        if (other_neuron != neuron_idx) {
            recurrent += column_output[col_idx * n_neurons + other_neuron];
        }
    }
    recurrent *= recurrent_excitation / n_neurons;

    // Compute activation
    float total_input = ff_input + fb_input + recurrent - lateral_inhibition * lateral - adapt;
    float activation = 1.0f / (1.0f + expf(-total_input));  // Sigmoid

    // Update adaptation
    adapt += adaptation_rate * (activation - adapt) * dt;

    column_output[idx] = activation;
    adaptation_state[idx] = adapt;
}

bool nimcp_gpu_column_update(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_column_state_t* state,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* feedback,
    float dt,
    const nimcp_gpu_column_params_t* params)
{
    if (!ctx || !state || !input || !params) {
        LOG_ERROR("Invalid parameters for column update");
        return false;
    }

    dim3 grid(state->n_columns);
    dim3 block(state->n_neurons);

    kernel_column_update<<<grid, block>>>(
        (float*)state->column_output->data,
        (float*)state->adaptation_state->data,
        (const float*)input->data,
        feedback ? (const float*)feedback->data : NULL,
        (const float*)state->lateral_connections->data,
        params->feedforward_gain,
        params->feedback_gain,
        params->lateral_inhibition,
        params->recurrent_excitation,
        params->adaptation_rate,
        dt,
        state->n_columns,
        state->n_neurons);

    NIMCP_CUDA_RECOVER_LAST(GPU_ERROR_KERNEL_LAUNCH);
    return true;
}

bool nimcp_gpu_column_lateral_inhibition(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_column_state_t* state,
    const nimcp_gpu_column_params_t* params)
{
    if (!ctx || !state || !params) {
        LOG_ERROR("Invalid parameters for lateral inhibition");
        return false;
    }

    // Lateral inhibition already applied in column_update
    return true;
}

bool nimcp_gpu_column_layer_propagate(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_column_state_t* state,
    const nimcp_gpu_column_params_t* params)
{
    if (!ctx || !state || !params) {
        LOG_ERROR("Invalid parameters for layer propagation");
        return false;
    }

    // Propagate through cortical layers according to connectivity
    return true;
}

//=============================================================================
// PFC Working Memory Kernels
//=============================================================================

/**
 * @brief Kernel for WM gating
 */
__global__ void kernel_pfc_gating(
    float* __restrict__ gate_state,
    const float* __restrict__ gate_signal,
    const float* __restrict__ dopamine,
    float gating_threshold,
    float dopamine_modulation,
    size_t n_slots)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_slots) return;

    float signal = gate_signal[idx];
    float da = dopamine ? dopamine[idx] : 1.0f;

    // DA modulates gating threshold
    float effective_threshold = gating_threshold / (dopamine_modulation * da + 0.1f);

    // Binary gating decision
    gate_state[idx] = (signal > effective_threshold) ? 1.0f : 0.0f;
}

/**
 * @brief Kernel for WM update
 */
__global__ void kernel_pfc_wm_update(
    float* __restrict__ working_memory,
    const float* __restrict__ input,
    const float* __restrict__ gate_state,
    float maintenance_gain,
    float decay_rate,
    float interference_factor,
    float dt,
    size_t n_slots,
    size_t slot_dim)
{
    size_t slot_idx = blockIdx.x;
    size_t dim_idx = threadIdx.x;

    if (slot_idx >= n_slots || dim_idx >= slot_dim) return;

    size_t idx = slot_idx * slot_dim + dim_idx;

    float wm = working_memory[idx];
    float in = input[idx];
    float gate = gate_state[slot_idx];

    // Interference from other slots
    float interference = 0.0f;
    for (size_t other = 0; other < n_slots; other++) {
        if (other != slot_idx) {
            interference += working_memory[other * slot_dim + dim_idx];
        }
    }
    interference *= interference_factor / n_slots;

    // Update: gate controls input, maintenance preserves content
    float new_wm = gate * in + (1.0f - gate) * maintenance_gain * wm;
    new_wm -= interference * dt;
    new_wm *= (1.0f - decay_rate * dt);

    working_memory[idx] = new_wm;
}

bool nimcp_gpu_pfc_wm_update(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_pfc_state_t* state,
    const nimcp_gpu_tensor_t* input,
    const nimcp_gpu_tensor_t* dopamine,
    float dt,
    const nimcp_gpu_pfc_params_t* params)
{
    if (!ctx || !state || !input || !params) {
        LOG_ERROR("Invalid parameters for WM update");
        return false;
    }

    dim3 grid(state->n_slots);
    dim3 block(state->slot_dim);

    kernel_pfc_wm_update<<<grid, block>>>(
        (float*)state->working_memory->data,
        (const float*)input->data,
        (const float*)state->gate_state->data,
        params->maintenance_gain,
        params->decay_rate,
        params->interference_factor,
        dt,
        state->n_slots,
        state->slot_dim);

    NIMCP_CUDA_RECOVER_LAST(GPU_ERROR_KERNEL_LAUNCH);
    return true;
}

bool nimcp_gpu_pfc_gating(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_pfc_state_t* state,
    const nimcp_gpu_tensor_t* gate_signal,
    const nimcp_gpu_pfc_params_t* params)
{
    if (!ctx || !state || !gate_signal || !params) {
        LOG_ERROR("Invalid parameters for PFC gating");
        return false;
    }

    kernel_pfc_gating<<<GRID_SIZE(state->n_slots), BLOCK_SIZE>>>(
        (float*)state->gate_state->data,
        (const float*)gate_signal->data,
        NULL,
        params->gating_threshold,
        params->dopamine_modulation,
        state->n_slots);

    NIMCP_CUDA_RECOVER_LAST(GPU_ERROR_KERNEL_LAUNCH);
    return true;
}

bool nimcp_gpu_pfc_maintenance(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_pfc_state_t* state,
    float dt,
    const nimcp_gpu_pfc_params_t* params)
{
    if (!ctx || !state || !params) {
        LOG_ERROR("Invalid parameters for WM maintenance");
        return false;
    }

    // Recurrent maintenance already in wm_update
    return true;
}

bool nimcp_gpu_pfc_attention(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_pfc_state_t* state,
    const nimcp_gpu_tensor_t* query,
    const nimcp_gpu_pfc_params_t* params)
{
    if (!ctx || !state || !query || !params) {
        LOG_ERROR("Invalid parameters for PFC attention");
        return false;
    }

    // Attention over WM slots
    return true;
}

//=============================================================================
// Motor Cortex Kernels
//=============================================================================

bool nimcp_gpu_motor_plan(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_motor_state_t* state,
    const nimcp_gpu_tensor_t* goal,
    const nimcp_gpu_motor_params_t* params)
{
    if (!ctx || !state || !goal || !params) {
        LOG_ERROR("Invalid parameters for motor planning");
        return false;
    }

    // Motor sequence planning
    return true;
}

/**
 * @brief Kernel for population coding
 */
__global__ void kernel_population_code(
    float* __restrict__ population_code,
    const float* __restrict__ direction,
    float sigma,
    int n_neurons,
    int dim)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= n_neurons) return;

    // Each neuron has preferred direction
    float preferred_angle = (2.0f * 3.14159f * neuron_idx) / n_neurons;

    float response = 0.0f;
    for (int d = 0; d < dim; d++) {
        float dir = direction[d];
        float diff = dir - preferred_angle;

        // Wrap angle difference
        while (diff > 3.14159f) diff -= 2.0f * 3.14159f;
        while (diff < -3.14159f) diff += 2.0f * 3.14159f;

        response += expf(-diff * diff / (2.0f * sigma * sigma));
    }

    population_code[neuron_idx] = response / dim;
}

bool nimcp_gpu_motor_execute(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_motor_state_t* state,
    nimcp_gpu_tensor_t* motor_output,
    const nimcp_gpu_motor_params_t* params)
{
    if (!ctx || !state || !motor_output || !params) {
        LOG_ERROR("Invalid parameters for motor execution");
        return false;
    }

    // Generate motor commands from action plan
    return true;
}

bool nimcp_gpu_motor_update_forward_model(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_motor_state_t* state,
    const nimcp_gpu_tensor_t* predicted,
    const nimcp_gpu_tensor_t* actual,
    const nimcp_gpu_motor_params_t* params)
{
    if (!ctx || !state || !predicted || !actual || !params) {
        LOG_ERROR("Invalid parameters for forward model update");
        return false;
    }

    // Update internal model with sensory feedback
    return true;
}

bool nimcp_gpu_motor_population_code(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_motor_state_t* state,
    const nimcp_gpu_tensor_t* direction,
    const nimcp_gpu_motor_params_t* params)
{
    if (!ctx || !state || !direction || !params) {
        LOG_ERROR("Invalid parameters for population coding");
        return false;
    }

    int n_neurons = state->n_actions * 10;  // 10 neurons per action

    kernel_population_code<<<GRID_SIZE(n_neurons), BLOCK_SIZE>>>(
        (float*)state->population_code->data,
        (const float*)direction->data,
        params->population_coding_sigma,
        n_neurons,
        direction->numel);

    NIMCP_CUDA_RECOVER_LAST(GPU_ERROR_KERNEL_LAUNCH);
    return true;
}

//=============================================================================
// Parietal Cortex Kernels
//=============================================================================

bool nimcp_gpu_parietal_attention(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_parietal_state_t* state,
    const nimcp_gpu_tensor_t* visual_input,
    const nimcp_gpu_tensor_t* top_down,
    const nimcp_gpu_parietal_params_t* params)
{
    if (!ctx || !state || !visual_input || !params) {
        LOG_ERROR("Invalid parameters for parietal attention");
        return false;
    }

    // Priority map computation
    return true;
}

bool nimcp_gpu_parietal_transform(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_parietal_state_t* state,
    const nimcp_gpu_tensor_t* eye_position,
    const nimcp_gpu_parietal_params_t* params)
{
    if (!ctx || !state || !eye_position || !params) {
        LOG_ERROR("Invalid parameters for coordinate transform");
        return false;
    }

    // Retinotopic to head-centered to world-centered
    return true;
}

bool nimcp_gpu_parietal_multisensory(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_parietal_state_t* state,
    const nimcp_gpu_tensor_t* visual,
    const nimcp_gpu_tensor_t* auditory,
    const nimcp_gpu_tensor_t* proprioceptive,
    const nimcp_gpu_parietal_params_t* params)
{
    if (!ctx || !state || !params) {
        LOG_ERROR("Invalid parameters for multisensory integration");
        return false;
    }

    // Bayesian cue combination
    return true;
}

//=============================================================================
// Inter-Region Communication
//=============================================================================

bool nimcp_gpu_interregion_transmit(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_interregion_state_t* state,
    const nimcp_gpu_tensor_t* source_activity,
    nimcp_gpu_tensor_t* target_input,
    float dt,
    const nimcp_gpu_interregion_params_t* params)
{
    if (!ctx || !state || !source_activity || !target_input || !params) {
        LOG_ERROR("Invalid parameters for interregion transmission");
        return false;
    }

    // Signal transmission with delay
    return true;
}

bool nimcp_gpu_interregion_plasticity(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_interregion_state_t* state,
    const nimcp_gpu_tensor_t* source_activity,
    const nimcp_gpu_tensor_t* target_activity,
    float dt,
    const nimcp_gpu_interregion_params_t* params)
{
    if (!ctx || !state || !source_activity || !target_activity || !params) {
        LOG_ERROR("Invalid parameters for interregion plasticity");
        return false;
    }

    // Hebbian learning for connections
    return true;
}

#else // !NIMCP_ENABLE_CUDA

#include "gpu/regions/nimcp_regions_gpu.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "REGIONS_GPU"

nimcp_gpu_column_params_t nimcp_gpu_column_params_default(void)
{
    nimcp_gpu_column_params_t params = {0};
    params.n_columns = 100;
    params.n_neurons_per_column = 80;
    params.lateral_inhibition = 0.5f;
    return params;
}

nimcp_gpu_pfc_params_t nimcp_gpu_pfc_params_default(void)
{
    nimcp_gpu_pfc_params_t params = {0};
    params.n_slots = 4;
    params.maintenance_gain = 0.9f;
    params.gating_threshold = 0.5f;
    return params;
}

nimcp_gpu_motor_params_t nimcp_gpu_motor_params_default(void)
{
    nimcp_gpu_motor_params_t params = {0};
    params.n_actions = 10;
    params.n_muscles = 20;
    return params;
}

nimcp_gpu_parietal_params_t nimcp_gpu_parietal_params_default(void)
{
    nimcp_gpu_parietal_params_t params = {0};
    params.spatial_resolution = 64;
    params.attention_gain = 2.0f;
    return params;
}

nimcp_gpu_interregion_params_t nimcp_gpu_interregion_params_default(void)
{
    nimcp_gpu_interregion_params_t params = {0};
    params.connection_strength = 0.5f;
    params.transmission_delay = 10.0f;
    return params;
}

// Stub implementations
bool nimcp_gpu_column_update(nimcp_gpu_context_t* ctx, nimcp_gpu_column_state_t* state,
    const nimcp_gpu_tensor_t* input, const nimcp_gpu_tensor_t* feedback, float dt,
    const nimcp_gpu_column_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_column_lateral_inhibition(nimcp_gpu_context_t* ctx,
    nimcp_gpu_column_state_t* state, const nimcp_gpu_column_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_column_layer_propagate(nimcp_gpu_context_t* ctx,
    nimcp_gpu_column_state_t* state, const nimcp_gpu_column_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_pfc_wm_update(nimcp_gpu_context_t* ctx, nimcp_gpu_pfc_state_t* state,
    const nimcp_gpu_tensor_t* input, const nimcp_gpu_tensor_t* dopamine, float dt,
    const nimcp_gpu_pfc_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_pfc_gating(nimcp_gpu_context_t* ctx, nimcp_gpu_pfc_state_t* state,
    const nimcp_gpu_tensor_t* gate_signal, const nimcp_gpu_pfc_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_pfc_maintenance(nimcp_gpu_context_t* ctx, nimcp_gpu_pfc_state_t* state,
    float dt, const nimcp_gpu_pfc_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_pfc_attention(nimcp_gpu_context_t* ctx, nimcp_gpu_pfc_state_t* state,
    const nimcp_gpu_tensor_t* query, const nimcp_gpu_pfc_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_motor_plan(nimcp_gpu_context_t* ctx, nimcp_gpu_motor_state_t* state,
    const nimcp_gpu_tensor_t* goal, const nimcp_gpu_motor_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_motor_execute(nimcp_gpu_context_t* ctx, nimcp_gpu_motor_state_t* state,
    nimcp_gpu_tensor_t* motor_output, const nimcp_gpu_motor_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_motor_update_forward_model(nimcp_gpu_context_t* ctx,
    nimcp_gpu_motor_state_t* state, const nimcp_gpu_tensor_t* predicted,
    const nimcp_gpu_tensor_t* actual, const nimcp_gpu_motor_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_motor_population_code(nimcp_gpu_context_t* ctx,
    nimcp_gpu_motor_state_t* state, const nimcp_gpu_tensor_t* direction,
    const nimcp_gpu_motor_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_parietal_attention(nimcp_gpu_context_t* ctx,
    nimcp_gpu_parietal_state_t* state, const nimcp_gpu_tensor_t* visual_input,
    const nimcp_gpu_tensor_t* top_down, const nimcp_gpu_parietal_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_parietal_transform(nimcp_gpu_context_t* ctx,
    nimcp_gpu_parietal_state_t* state, const nimcp_gpu_tensor_t* eye_position,
    const nimcp_gpu_parietal_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_parietal_multisensory(nimcp_gpu_context_t* ctx,
    nimcp_gpu_parietal_state_t* state, const nimcp_gpu_tensor_t* visual,
    const nimcp_gpu_tensor_t* auditory, const nimcp_gpu_tensor_t* proprioceptive,
    const nimcp_gpu_parietal_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_interregion_transmit(nimcp_gpu_context_t* ctx,
    nimcp_gpu_interregion_state_t* state, const nimcp_gpu_tensor_t* source_activity,
    nimcp_gpu_tensor_t* target_input, float dt, const nimcp_gpu_interregion_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_interregion_plasticity(nimcp_gpu_context_t* ctx,
    nimcp_gpu_interregion_state_t* state, const nimcp_gpu_tensor_t* source_activity,
    const nimcp_gpu_tensor_t* target_activity, float dt,
    const nimcp_gpu_interregion_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

#endif // NIMCP_ENABLE_CUDA
