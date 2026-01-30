/**
 * @file nimcp_sleep_kernels.cu
 * @brief GPU Sleep and Memory Consolidation CUDA Kernels
 *
 * WHAT: CUDA kernels for sleep-dependent memory consolidation
 * WHY:  GPU acceleration for offline learning and memory transfer
 * HOW:  Custom kernels for replay, consolidation, synaptic homeostasis
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifdef NIMCP_ENABLE_CUDA

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#include "gpu/sleep/nimcp_sleep_gpu.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "gpu/common/nimcp_cuda_utils.h"
#include "gpu/recovery/nimcp_gpu_recovery.h"

#define LOG_MODULE "SLEEP_GPU"

#define BLOCK_SIZE 256
#define GRID_SIZE(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)

//=============================================================================
// Default Parameters
//=============================================================================

nimcp_gpu_nrem_params_t nimcp_gpu_nrem_params_default(void)
{
    nimcp_gpu_nrem_params_t params;
    params.slow_wave_freq = 0.8f;    // ~0.8 Hz slow oscillation
    params.spindle_freq = 12.0f;     // 12-15 Hz spindles
    params.sharp_wave_rate = 1.0f;   // 1 Hz SWR rate
    params.consolidation_rate = 0.01f;
    params.replay_speed = 10.0f;     // 10x compression
    params.cortical_gain = 0.5f;
    params.replay_iterations = 100;
    params.noise_level = 0.01f;
    return params;
}

nimcp_gpu_rem_params_t nimcp_gpu_rem_params_default(void)
{
    nimcp_gpu_rem_params_t params;
    params.theta_freq = 6.0f;        // 6-8 Hz theta
    params.pgo_rate = 0.5f;
    params.emotional_bias = 0.3f;
    params.integration_rate = 0.05f;
    params.dream_generation_rate = 0.1f;
    params.creativity_factor = 0.2f;
    params.acetylcholine_level = 0.8f;
    return params;
}

nimcp_gpu_homeostasis_params_t nimcp_gpu_homeostasis_params_default(void)
{
    nimcp_gpu_homeostasis_params_t params;
    params.downscaling_rate = 0.1f;
    params.threshold = 0.5f;
    params.preservation_factor = 0.9f;
    params.global_factor = 0.8f;
    params.selective_pruning = true;
    params.min_weight = 0.01f;
    return params;
}

nimcp_gpu_replay_params_t nimcp_gpu_replay_params_default(void)
{
    nimcp_gpu_replay_params_t params;
    params.compression_ratio = 10.0f;
    params.sequence_fidelity = 0.9f;
    params.reverse_replay_prob = 0.3f;
    params.priority_weight = 0.6f;
    params.buffer_size = 1000;
    params.decay_rate = 0.01f;
    return params;
}

//=============================================================================
// Sleep Stage Kernels
//=============================================================================

bool nimcp_gpu_sleep_stage_update(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_sleep_stage_state_t* state,
    float circadian_phase,
    float arousal_signal,
    float dt)
{
    if (!ctx || !state) {
        LOG_ERROR("Invalid parameters for sleep stage update");
        return false;
    }

    // Simple state machine for sleep stages
    // Would implement proper Markov chain transitions
    state->stage_duration += dt;
    state->total_sleep_time += dt;

    return true;
}

bool nimcp_gpu_sleep_transitions(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_sleep_stage_state_t* state,
    nimcp_gpu_tensor_t* transition_probs)
{
    if (!ctx || !state || !transition_probs) {
        LOG_ERROR("Invalid parameters for sleep transitions");
        return false;
    }

    // Compute transition probabilities based on stage duration and pressure
    return true;
}

//=============================================================================
// NREM Consolidation Kernels
//=============================================================================

/**
 * @brief Kernel for slow oscillation generation
 */
__global__ void kernel_slow_oscillation(
    float* __restrict__ oscillation,
    float time,
    float slow_wave_freq,
    size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Slow oscillation: alternating up and down states
    float phase = 2.0f * 3.14159f * slow_wave_freq * time;
    float so = 0.5f * (1.0f + sinf(phase));

    // Nested faster oscillations (spindles during up state)
    float spindle = 0.0f;
    if (so > 0.5f) {
        spindle = 0.3f * sinf(12.0f * 2.0f * 3.14159f * time);
    }

    oscillation[idx] = so + spindle;
}

bool nimcp_gpu_nrem_slow_oscillation(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state,
    float time,
    const nimcp_gpu_nrem_params_t* params)
{
    if (!ctx || !state || !params) {
        LOG_ERROR("Invalid parameters for slow oscillation");
        return false;
    }

    size_t n = state->buffer_size;

    kernel_slow_oscillation<<<GRID_SIZE(n), BLOCK_SIZE>>>(
        (float*)state->slow_oscillation->data,
        time,
        params->slow_wave_freq,
        n);

    NIMCP_CUDA_RECOVER_LAST(GPU_ERROR_KERNEL_LAUNCH);
    return true;
}

/**
 * @brief Kernel for memory replay
 */
__global__ void kernel_nrem_replay(
    float* __restrict__ cortical_weights,
    const float* __restrict__ hippocampal_buffer,
    const float* __restrict__ replay_buffer,
    const float* __restrict__ slow_oscillation,
    const float* __restrict__ priority_scores,
    float consolidation_rate,
    float replay_speed,
    size_t buffer_size,
    size_t memory_dim)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= buffer_size * memory_dim) return;

    size_t mem_idx = idx / memory_dim;
    size_t dim_idx = idx % memory_dim;

    // Replay during slow oscillation up state
    float so = slow_oscillation[mem_idx % buffer_size];
    if (so < 0.5f) return;  // Only replay during up state

    float hippocampal = hippocampal_buffer[idx];
    float priority = priority_scores[mem_idx];

    // Transfer to cortex weighted by priority
    float transfer = consolidation_rate * hippocampal * priority * so;

    // Update cortical weights
    atomicAdd(&cortical_weights[idx], transfer);
}

bool nimcp_gpu_nrem_replay(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state,
    const nimcp_gpu_nrem_params_t* params)
{
    if (!ctx || !state || !params) {
        LOG_ERROR("Invalid parameters for NREM replay");
        return false;
    }

    size_t n = state->buffer_size * state->memory_dim;

    kernel_nrem_replay<<<GRID_SIZE(n), BLOCK_SIZE>>>(
        (float*)state->cortical_weights->data,
        (const float*)state->hippocampal_buffer->data,
        (const float*)state->replay_buffer->data,
        (const float*)state->slow_oscillation->data,
        (const float*)state->priority_scores->data,
        params->consolidation_rate,
        params->replay_speed,
        state->buffer_size,
        state->memory_dim);

    NIMCP_CUDA_RECOVER_LAST(GPU_ERROR_KERNEL_LAUNCH);
    return true;
}

bool nimcp_gpu_nrem_systems_consolidation(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state,
    float dt,
    const nimcp_gpu_nrem_params_t* params)
{
    if (!ctx || !state || !params) {
        LOG_ERROR("Invalid parameters for systems consolidation");
        return false;
    }

    // Systems consolidation: transfer from hippocampus to cortex
    return nimcp_gpu_nrem_replay(ctx, state, params);
}

bool nimcp_gpu_nrem_sharp_wave_ripple(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state,
    nimcp_gpu_tensor_t* ripple_content,
    const nimcp_gpu_nrem_params_t* params)
{
    if (!ctx || !state || !ripple_content || !params) {
        LOG_ERROR("Invalid parameters for SWR");
        return false;
    }

    // Sharp-wave ripple: compressed replay
    return true;
}

//=============================================================================
// REM Processing Kernels
//=============================================================================

bool nimcp_gpu_rem_processing(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state,
    float dt,
    const nimcp_gpu_rem_params_t* params)
{
    if (!ctx || !state || !params) {
        LOG_ERROR("Invalid parameters for REM processing");
        return false;
    }

    // REM: theta rhythm, ACh-dependent processing
    return true;
}

/**
 * @brief Kernel for memory integration
 */
__global__ void kernel_rem_integration(
    float* __restrict__ hippocampal_buffer,
    const float* __restrict__ semantic_memory,
    float integration_rate,
    size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float episodic = hippocampal_buffer[idx];
    float semantic = semantic_memory[idx];

    // Integrate episodic with semantic knowledge
    float integrated = episodic + integration_rate * (semantic - episodic);

    hippocampal_buffer[idx] = integrated;
}

bool nimcp_gpu_rem_memory_integration(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state,
    const nimcp_gpu_tensor_t* semantic_memory,
    const nimcp_gpu_rem_params_t* params)
{
    if (!ctx || !state || !semantic_memory || !params) {
        LOG_ERROR("Invalid parameters for memory integration");
        return false;
    }

    size_t n = state->hippocampal_buffer->numel;

    kernel_rem_integration<<<GRID_SIZE(n), BLOCK_SIZE>>>(
        (float*)state->hippocampal_buffer->data,
        (const float*)semantic_memory->data,
        params->integration_rate,
        n);

    NIMCP_CUDA_RECOVER_LAST(GPU_ERROR_KERNEL_LAUNCH);
    return true;
}

bool nimcp_gpu_rem_emotional_processing(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state,
    const nimcp_gpu_tensor_t* emotional_tags,
    const nimcp_gpu_rem_params_t* params)
{
    if (!ctx || !state || !emotional_tags || !params) {
        LOG_ERROR("Invalid parameters for emotional processing");
        return false;
    }

    // REM emotional memory processing
    return true;
}

bool nimcp_gpu_rem_dream_generation(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_consolidation_state_t* state,
    nimcp_gpu_tensor_t* dream_content,
    const nimcp_gpu_rem_params_t* params)
{
    if (!ctx || !state || !dream_content || !params) {
        LOG_ERROR("Invalid parameters for dream generation");
        return false;
    }

    // Generate novel combinations (creativity)
    return true;
}

//=============================================================================
// Synaptic Homeostasis Kernels
//=============================================================================

/**
 * @brief Kernel for synaptic downscaling
 */
__global__ void kernel_synaptic_downscaling(
    float* __restrict__ weights,
    const float* __restrict__ potentiation_tags,
    float downscaling_rate,
    float preservation_factor,
    float min_weight,
    size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float w = weights[idx];
    float tag = potentiation_tags[idx];

    // Preserve tagged (recently potentiated) synapses
    float protection = tag * preservation_factor;

    // Global downscaling
    float scale = 1.0f - downscaling_rate * (1.0f - protection);

    w *= scale;
    w = fmaxf(min_weight, w);

    weights[idx] = w;
}

bool nimcp_gpu_synaptic_downscaling(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_synaptic_state_t* state,
    const nimcp_gpu_homeostasis_params_t* params)
{
    if (!ctx || !state || !params) {
        LOG_ERROR("Invalid parameters for synaptic downscaling");
        return false;
    }

    size_t n = state->n_synapses;

    kernel_synaptic_downscaling<<<GRID_SIZE(n), BLOCK_SIZE>>>(
        (float*)state->synaptic_weights->data,
        (const float*)state->potentiation_tags->data,
        params->downscaling_rate,
        params->preservation_factor,
        params->min_weight,
        n);

    NIMCP_CUDA_RECOVER_LAST(GPU_ERROR_KERNEL_LAUNCH);
    return true;
}

bool nimcp_gpu_synapse_preservation(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_synaptic_state_t* state,
    const nimcp_gpu_tensor_t* importance_scores,
    const nimcp_gpu_homeostasis_params_t* params)
{
    if (!ctx || !state || !importance_scores || !params) {
        LOG_ERROR("Invalid parameters for synapse preservation");
        return false;
    }

    // Selective preservation of important synapses
    return true;
}

/**
 * @brief Kernel for synapse pruning
 */
__global__ void kernel_synapse_pruning(
    float* __restrict__ weights,
    float threshold,
    float min_weight,
    size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float w = weights[idx];

    // Prune weak synapses
    if (w < threshold) {
        weights[idx] = 0.0f;
    }
}

bool nimcp_gpu_synapse_pruning(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_synaptic_state_t* state,
    const nimcp_gpu_homeostasis_params_t* params)
{
    if (!ctx || !state || !params) {
        LOG_ERROR("Invalid parameters for synapse pruning");
        return false;
    }

    if (!params->selective_pruning) return true;

    size_t n = state->n_synapses;

    kernel_synapse_pruning<<<GRID_SIZE(n), BLOCK_SIZE>>>(
        (float*)state->synaptic_weights->data,
        params->threshold * params->min_weight,
        params->min_weight,
        n);

    NIMCP_CUDA_RECOVER_LAST(GPU_ERROR_KERNEL_LAUNCH);
    return true;
}

bool nimcp_gpu_synaptic_tagging(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_synaptic_state_t* state,
    const nimcp_gpu_tensor_t* recent_activity,
    float dt)
{
    if (!ctx || !state || !recent_activity) {
        LOG_ERROR("Invalid parameters for synaptic tagging");
        return false;
    }

    // Tag recently active synapses
    return true;
}

//=============================================================================
// Memory Replay Functions
//=============================================================================

bool nimcp_gpu_replay_sample(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_consolidation_state_t* state,
    nimcp_gpu_tensor_t* sampled_memories,
    int n_samples,
    const nimcp_gpu_replay_params_t* params)
{
    if (!ctx || !state || !sampled_memories || !params) {
        LOG_ERROR("Invalid parameters for replay sampling");
        return false;
    }

    // Prioritized experience replay sampling
    return true;
}

bool nimcp_gpu_replay_store(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state,
    const nimcp_gpu_tensor_t* experience,
    float priority,
    const nimcp_gpu_replay_params_t* params)
{
    if (!ctx || !state || !experience || !params) {
        LOG_ERROR("Invalid parameters for replay storage");
        return false;
    }

    // Store new experience with priority
    return true;
}

bool nimcp_gpu_replay_update_priorities(
    nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state,
    const nimcp_gpu_tensor_t* td_errors,
    const nimcp_gpu_replay_params_t* params)
{
    if (!ctx || !state || !td_errors || !params) {
        LOG_ERROR("Invalid parameters for priority update");
        return false;
    }

    // Update priorities based on TD errors
    return true;
}

bool nimcp_gpu_replay_compress(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* sequence,
    nimcp_gpu_tensor_t* compressed,
    const nimcp_gpu_replay_params_t* params)
{
    if (!ctx || !sequence || !compressed || !params) {
        LOG_ERROR("Invalid parameters for replay compression");
        return false;
    }

    // Temporal compression of replay sequences
    return true;
}

#else // !NIMCP_ENABLE_CUDA

#include "gpu/sleep/nimcp_sleep_gpu.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "SLEEP_GPU"

nimcp_gpu_nrem_params_t nimcp_gpu_nrem_params_default(void)
{
    nimcp_gpu_nrem_params_t params = {0};
    params.slow_wave_freq = 0.8f;
    params.consolidation_rate = 0.01f;
    return params;
}

nimcp_gpu_rem_params_t nimcp_gpu_rem_params_default(void)
{
    nimcp_gpu_rem_params_t params = {0};
    params.theta_freq = 6.0f;
    params.integration_rate = 0.05f;
    return params;
}

nimcp_gpu_homeostasis_params_t nimcp_gpu_homeostasis_params_default(void)
{
    nimcp_gpu_homeostasis_params_t params = {0};
    params.downscaling_rate = 0.1f;
    params.min_weight = 0.01f;
    return params;
}

nimcp_gpu_replay_params_t nimcp_gpu_replay_params_default(void)
{
    nimcp_gpu_replay_params_t params = {0};
    params.compression_ratio = 10.0f;
    params.buffer_size = 1000;
    return params;
}

// Stub implementations
bool nimcp_gpu_sleep_stage_update(nimcp_gpu_context_t* ctx,
    nimcp_gpu_sleep_stage_state_t* state, float circadian_phase,
    float arousal_signal, float dt)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_sleep_transitions(nimcp_gpu_context_t* ctx,
    nimcp_gpu_sleep_stage_state_t* state, nimcp_gpu_tensor_t* transition_probs)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_nrem_slow_oscillation(nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state, float time, const nimcp_gpu_nrem_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_nrem_replay(nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state, const nimcp_gpu_nrem_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_nrem_systems_consolidation(nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state, float dt, const nimcp_gpu_nrem_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_nrem_sharp_wave_ripple(nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state, nimcp_gpu_tensor_t* ripple_content,
    const nimcp_gpu_nrem_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_rem_processing(nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state, float dt, const nimcp_gpu_rem_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_rem_memory_integration(nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state, const nimcp_gpu_tensor_t* semantic_memory,
    const nimcp_gpu_rem_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_rem_emotional_processing(nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state, const nimcp_gpu_tensor_t* emotional_tags,
    const nimcp_gpu_rem_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_rem_dream_generation(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_consolidation_state_t* state, nimcp_gpu_tensor_t* dream_content,
    const nimcp_gpu_rem_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_synaptic_downscaling(nimcp_gpu_context_t* ctx,
    nimcp_gpu_synaptic_state_t* state, const nimcp_gpu_homeostasis_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_synapse_preservation(nimcp_gpu_context_t* ctx,
    nimcp_gpu_synaptic_state_t* state, const nimcp_gpu_tensor_t* importance_scores,
    const nimcp_gpu_homeostasis_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_synapse_pruning(nimcp_gpu_context_t* ctx,
    nimcp_gpu_synaptic_state_t* state, const nimcp_gpu_homeostasis_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_synaptic_tagging(nimcp_gpu_context_t* ctx,
    nimcp_gpu_synaptic_state_t* state, const nimcp_gpu_tensor_t* recent_activity, float dt)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_replay_sample(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_consolidation_state_t* state, nimcp_gpu_tensor_t* sampled_memories,
    int n_samples, const nimcp_gpu_replay_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_replay_store(nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state, const nimcp_gpu_tensor_t* experience,
    float priority, const nimcp_gpu_replay_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_replay_update_priorities(nimcp_gpu_context_t* ctx,
    nimcp_gpu_consolidation_state_t* state, const nimcp_gpu_tensor_t* td_errors,
    const nimcp_gpu_replay_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

bool nimcp_gpu_replay_compress(nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* sequence, nimcp_gpu_tensor_t* compressed,
    const nimcp_gpu_replay_params_t* params)
{ LOG_WARN("CUDA not enabled"); return false; }

#endif // NIMCP_ENABLE_CUDA
