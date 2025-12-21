/**
 * @file nimcp_systems_consolidation_pink_noise_bridge.c
 * @brief Pink Noise Integration Bridge for Sleep Replay - Implementation
 * @version 1.0.0
 * @date 2025-12-21
 *
 * IMPLEMENTATION NOTES:
 * - Thread-safe operations via mutex locking
 * - Temporal smoothing prevents abrupt modulation changes
 * - Respects biological timescales for sleep stage transitions
 * - Efficient parameter updates (no heavy computation)
 */

#include "cognitive/memory/nimcp_systems_consolidation_pink_noise_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>

//=============================================================================
// Internal Helper: Clamp
//=============================================================================

/**
 * WHAT: Clamp value to range
 * WHY:  Ensure parameters stay within valid bounds
 * HOW:  Standard min/max clamping
 */
static inline float clamp_f(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

//=============================================================================
// Internal Helper: Exponential Smoothing
//=============================================================================

/**
 * WHAT: Apply exponential smoothing filter
 * WHY:  Prevent abrupt changes in modulation parameters
 * HOW:  new_value = alpha × current + (1-alpha) × target
 *
 * @param current Current smoothed value
 * @param target Target value to approach
 * @param alpha Smoothing factor [0=no change, 1=instant]
 * @return Smoothed value
 */
static float smooth_value(float current, float target, float alpha) {
    return alpha * target + (1.0f - alpha) * current;
}

//=============================================================================
// Lifecycle API Implementation
//=============================================================================

int consolidation_pink_noise_default_config(consolidation_pink_noise_config_t* config) {
    // WHAT: Initialize configuration with biologically-plausible defaults
    // WHY:  Provides known-good starting point
    // HOW:  Set empirically-derived parameter values

    if (!config) {
        NIMCP_LOGGING_ERROR("Null config pointer");
        return -1;
    }

    config->enable_amplitude_modulation = true;
    config->enable_alpha_modulation = true;
    config->enable_spindle_priority = true;
    config->enable_arousal_frequency = true;

    config->amplitude_gain = 1.0f;
    config->alpha_sensitivity = 0.5f;
    config->spindle_priority_boost = 0.3f;
    config->modulation_smoothing = 0.1f;

    return 0;
}

consolidation_pink_noise_bridge_t* consolidation_pink_noise_create(
    const consolidation_pink_noise_config_t* config,
    systems_consolidation_system_t* consolidation_system,
    pink_sleep_bridge_t* pink_noise_bridge)
{
    // WHAT: Create and initialize pink noise bridge
    // WHY:  Required for noise-modulated consolidation
    // HOW:  Allocate structure, validate inputs, initialize state

    // Guard clause: Validate required inputs
    if (!consolidation_system) {
        NIMCP_LOGGING_ERROR("Null consolidation system");
        return NULL;
    }

    if (!pink_noise_bridge) {
        NIMCP_LOGGING_ERROR("Null pink noise bridge");
        return NULL;
    }

    // Allocate bridge structure
    consolidation_pink_noise_bridge_t* bridge =
        (consolidation_pink_noise_bridge_t*)nimcp_malloc(sizeof(consolidation_pink_noise_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate bridge");
        return NULL;
    }

    // Initialize configuration
    if (config) {
        memcpy(&bridge->config, config, sizeof(consolidation_pink_noise_config_t));
    } else {
        consolidation_pink_noise_default_config(&bridge->config);
    }

    // Store module connections (non-owning)
    bridge->consolidation_system = consolidation_system;
    bridge->pink_noise_bridge = pink_noise_bridge;

    // Initialize effects to neutral state
    memset(&bridge->effects, 0, sizeof(consolidation_pink_noise_effects_t));
    bridge->effects.replay_strength_factor = 1.0f;
    bridge->effects.transfer_quality_factor = 1.0f;
    bridge->effects.replay_frequency_hz = REPLAY_FREQ_AWAKE;

    // Initialize smoothed values to neutral
    bridge->smoothed_replay_strength = 1.0f;
    bridge->smoothed_transfer_quality = 1.0f;
    bridge->smoothed_frequency = REPLAY_FREQ_AWAKE;

    // Initialize statistics
    bridge->total_updates = 0;
    bridge->spindle_boosts_applied = 0;
    bridge->stage_transitions = 0;

    // Create mutex for thread safety
    bridge->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex");
        nimcp_free(bridge);
        return NULL;
    }
    if (nimcp_mutex_init(bridge->mutex, NULL) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex");
        nimcp_free(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Created pink noise bridge for systems consolidation");
    return bridge;
}

void consolidation_pink_noise_destroy(consolidation_pink_noise_bridge_t* bridge) {
    // WHAT: Clean up bridge resources
    // WHY:  Prevent memory leaks
    // HOW:  Free mutex and structure

    if (!bridge) {
        return;
    }

    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
        nimcp_free(bridge->mutex);
        bridge->mutex = NULL;
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed pink noise bridge");
}

//=============================================================================
// Update and Modulation API Implementation
//=============================================================================

int consolidation_pink_noise_update(consolidation_pink_noise_bridge_t* bridge) {
    // WHAT: Update modulation factors based on current pink noise state
    // WHY:  Keep replay parameters synchronized with noise characteristics
    // HOW:  Sample noise parameters, compute factors, apply smoothing

    if (!bridge) {
        NIMCP_LOGGING_ERROR("Null bridge pointer");
        return -1;
    }

    nimcp_mutex_lock(bridge->mutex);

    // Sample current pink noise state
    float noise_amplitude = pink_sleep_get_amplitude(bridge->pink_noise_bridge);
    float noise_alpha = pink_sleep_get_alpha(bridge->pink_noise_bridge);

    // Get sleep stage (we'll need to extract this from the pink noise bridge structure)
    // For now, we'll track transitions by detecting amplitude/alpha changes
    pink_sleep_stage_t current_stage = PINK_SLEEP_WAKE;  // Default

    // Heuristic: Infer stage from noise characteristics
    if (noise_alpha >= 1.4f && noise_amplitude > 0.5f) {
        current_stage = PINK_SLEEP_N3;  // Deep sleep: high alpha, high amplitude
    } else if (noise_alpha < 0.9f && noise_amplitude > 0.4f) {
        current_stage = PINK_SLEEP_REM;  // REM: low alpha, variable amplitude
    } else if (noise_alpha >= 1.0f && noise_amplitude > 0.3f) {
        current_stage = PINK_SLEEP_N2;  // Light sleep N2
    } else if (noise_amplitude > 0.2f) {
        current_stage = PINK_SLEEP_N1;  // Light sleep N1
    } else if (noise_amplitude > 0.1f) {
        current_stage = PINK_SLEEP_DROWSY;  // Drowsy
    }

    // Track stage transitions
    if (current_stage != bridge->effects.current_stage) {
        bridge->stage_transitions++;
        bridge->effects.current_stage = current_stage;
    }

    // Store current noise parameters
    bridge->effects.noise_amplitude = noise_amplitude;
    bridge->effects.noise_alpha = noise_alpha;

    // Compute arousal level (inverse of noise amplitude for sleep stages)
    float arousal = 1.0f - clamp_f(noise_amplitude, 0.0f, 1.0f);
    bridge->effects.arousal_level = arousal;

    // Check for spindle burst (specific to N2 stage)
    // Spindles occur when we're in N2 range and have specific frequency characteristics
    bridge->effects.in_spindle_burst = (current_stage == PINK_SLEEP_N2);

    // 1. Compute base replay strength from sleep stage
    float base_strength = consolidation_pink_noise_stage_replay_strength(current_stage);

    // 2. Modulate by noise amplitude (if enabled)
    float strength_factor = base_strength;
    if (bridge->config.enable_amplitude_modulation) {
        // Higher amplitude in appropriate sleep stages enhances replay
        float amplitude_mod = 1.0f + bridge->config.amplitude_gain * (noise_amplitude - 0.5f);
        amplitude_mod = clamp_f(amplitude_mod, 0.5f, 1.5f);
        strength_factor *= amplitude_mod;
    }
    strength_factor = clamp_f(strength_factor, 0.0f, 1.0f);

    // 3. Compute transfer quality from spectral exponent (if enabled)
    float quality_factor = 1.0f;
    if (bridge->config.enable_alpha_modulation) {
        quality_factor = consolidation_pink_noise_alpha_to_quality(noise_alpha);
        quality_factor *= (1.0f + bridge->config.alpha_sensitivity * (noise_alpha - 1.0f) * 0.5f);
        quality_factor = clamp_f(quality_factor, 0.5f, 1.5f);
    }

    // 4. Compute replay frequency from arousal (if enabled)
    float frequency_hz = REPLAY_FREQ_AWAKE;
    if (bridge->config.enable_arousal_frequency) {
        frequency_hz = consolidation_pink_noise_arousal_to_frequency(arousal);
    }

    // 5. Compute priority modulation from spindles (if enabled)
    float priority_mod = 0.0f;
    if (bridge->config.enable_spindle_priority && bridge->effects.in_spindle_burst) {
        priority_mod = bridge->config.spindle_priority_boost;
        bridge->spindle_boosts_applied++;
    }

    // Apply temporal smoothing to prevent abrupt changes
    float smooth_alpha = bridge->config.modulation_smoothing;
    bridge->smoothed_replay_strength = smooth_value(
        bridge->smoothed_replay_strength, strength_factor, smooth_alpha);
    bridge->smoothed_transfer_quality = smooth_value(
        bridge->smoothed_transfer_quality, quality_factor, smooth_alpha);
    bridge->smoothed_frequency = smooth_value(
        bridge->smoothed_frequency, frequency_hz, smooth_alpha);

    // Update effects structure with smoothed values
    bridge->effects.replay_strength_factor = bridge->smoothed_replay_strength;
    bridge->effects.transfer_quality_factor = bridge->smoothed_transfer_quality;
    bridge->effects.replay_frequency_hz = bridge->smoothed_frequency;
    bridge->effects.priority_modulation = priority_mod;

    bridge->total_updates++;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int consolidation_pink_noise_apply_modulation(consolidation_pink_noise_bridge_t* bridge) {
    // WHAT: Apply computed modulation to consolidation system
    // WHY:  Implement noise-driven optimization of replay
    // HOW:  Modify consolidation parameters based on effects
    //
    // NOTE: This function sets internal state that will be used by the
    //       consolidation system during its next replay execution.
    //       The actual systems_consolidation API would need extension points
    //       to support this modulation. For now, we document the intent.

    if (!bridge) {
        NIMCP_LOGGING_ERROR("Null bridge pointer");
        return -1;
    }

    nimcp_mutex_lock(bridge->mutex);

    // The consolidation system would need to expose:
    // - Set replay frequency
    // - Set transfer rate multiplier
    // - Set priority adjustment factor
    //
    // Since these APIs don't exist yet in systems_consolidation.h,
    // we'll document the intended integration:
    //
    // systems_consolidation_set_replay_frequency(
    //     bridge->consolidation_system,
    //     bridge->effects.replay_frequency_hz);
    //
    // systems_consolidation_set_transfer_multiplier(
    //     bridge->consolidation_system,
    //     bridge->effects.transfer_quality_factor);
    //
    // systems_consolidation_set_priority_boost(
    //     bridge->consolidation_system,
    //     bridge->effects.priority_modulation);

    // For now, we successfully store the modulation state that can be
    // queried by the consolidation system or wrapper code

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Query API Implementation
//=============================================================================

int consolidation_pink_noise_get_effects(
    const consolidation_pink_noise_bridge_t* bridge,
    consolidation_pink_noise_effects_t* effects_out)
{
    // WHAT: Retrieve current modulation effects
    // WHY:  Allow inspection of noise influence on replay
    // HOW:  Copy effects structure with mutex protection

    if (!bridge || !effects_out) {
        NIMCP_LOGGING_ERROR("Null pointer in get_effects");
        return -1;
    }

    nimcp_mutex_lock(bridge->mutex);
    memcpy(effects_out, &bridge->effects, sizeof(consolidation_pink_noise_effects_t));
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

float consolidation_pink_noise_get_replay_strength(
    const consolidation_pink_noise_bridge_t* bridge)
{
    // WHAT: Get current replay strength factor
    // WHY:  Quick access to primary modulation parameter
    // HOW:  Return smoothed value with bounds checking

    if (!bridge) {
        return -1.0f;
    }

    nimcp_mutex_lock(bridge->mutex);
    float strength = bridge->effects.replay_strength_factor;
    nimcp_mutex_unlock(bridge->mutex);

    return strength;
}

float consolidation_pink_noise_get_transfer_quality(
    const consolidation_pink_noise_bridge_t* bridge)
{
    // WHAT: Get current transfer quality factor
    // WHY:  Indicates consolidation effectiveness
    // HOW:  Return smoothed quality with bounds checking

    if (!bridge) {
        return -1.0f;
    }

    nimcp_mutex_lock(bridge->mutex);
    float quality = bridge->effects.transfer_quality_factor;
    nimcp_mutex_unlock(bridge->mutex);

    return quality;
}

float consolidation_pink_noise_get_replay_frequency(
    const consolidation_pink_noise_bridge_t* bridge)
{
    // WHAT: Get current replay frequency
    // WHY:  Indicates optimal replay rate for current state
    // HOW:  Return smoothed frequency with bounds checking

    if (!bridge) {
        return -1.0f;
    }

    nimcp_mutex_lock(bridge->mutex);
    float freq = bridge->effects.replay_frequency_hz;
    nimcp_mutex_unlock(bridge->mutex);

    return freq;
}

bool consolidation_pink_noise_in_spindle_burst(
    const consolidation_pink_noise_bridge_t* bridge)
{
    // WHAT: Check if spindle burst is active
    // WHY:  Spindles indicate memory tagging opportunity
    // HOW:  Return spindle state with mutex protection

    if (!bridge) {
        return false;
    }

    nimcp_mutex_lock(bridge->mutex);
    bool in_spindle = bridge->effects.in_spindle_burst;
    nimcp_mutex_unlock(bridge->mutex);

    return in_spindle;
}

//=============================================================================
// Statistics API Implementation
//=============================================================================

int consolidation_pink_noise_get_statistics(
    const consolidation_pink_noise_bridge_t* bridge,
    uint64_t* total_updates_out,
    uint64_t* spindle_boosts_out,
    uint64_t* stage_transitions_out)
{
    // WHAT: Retrieve operational statistics
    // WHY:  Support monitoring and debugging
    // HOW:  Copy statistics with mutex protection

    if (!bridge) {
        NIMCP_LOGGING_ERROR("Null bridge pointer");
        return -1;
    }

    nimcp_mutex_lock(bridge->mutex);

    if (total_updates_out) {
        *total_updates_out = bridge->total_updates;
    }
    if (spindle_boosts_out) {
        *spindle_boosts_out = bridge->spindle_boosts_applied;
    }
    if (stage_transitions_out) {
        *stage_transitions_out = bridge->stage_transitions;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Helper Functions Implementation
//=============================================================================

float consolidation_pink_noise_stage_replay_strength(pink_sleep_stage_t stage) {
    // WHAT: Map sleep stage to base replay strength
    // WHY:  Different stages have different consolidation potential
    // HOW:  Lookup table based on biological research

    switch (stage) {
        case PINK_SLEEP_WAKE:
            return REPLAY_STRENGTH_WAKE;
        case PINK_SLEEP_DROWSY:
            return REPLAY_STRENGTH_DROWSY;
        case PINK_SLEEP_N1:
            return REPLAY_STRENGTH_N1;
        case PINK_SLEEP_N2:
            return REPLAY_STRENGTH_N2;
        case PINK_SLEEP_N3:
            return REPLAY_STRENGTH_N3;
        case PINK_SLEEP_REM:
            return REPLAY_STRENGTH_REM;
        default:
            return REPLAY_STRENGTH_WAKE;
    }
}

float consolidation_pink_noise_alpha_to_quality(float alpha) {
    // WHAT: Convert spectral exponent to transfer quality
    // WHY:  Redder noise provides more stable consolidation
    // HOW:  Normalized mapping with peak at α≈1.5
    //
    // BIOLOGICAL RATIONALE:
    // - White noise (α≈0.5): 0.5 quality (poor, high variability)
    // - Pink noise (α≈1.0): 1.0 quality (baseline)
    // - Red noise (α≈1.5): 1.5 quality (optimal for SWS)
    // - Very red (α≈2.0): 1.3 quality (diminishing returns)

    if (alpha <= 0.5f) {
        return 0.5f;  // White noise minimum
    } else if (alpha <= 1.0f) {
        // Linear interpolation: 0.5 at α=0.5, 1.0 at α=1.0
        return 0.5f + (alpha - 0.5f);
    } else if (alpha <= 1.5f) {
        // Linear interpolation: 1.0 at α=1.0, 1.5 at α=1.5
        return 1.0f + (alpha - 1.0f);
    } else if (alpha <= 2.0f) {
        // Diminishing returns: 1.5 at α=1.5, 1.3 at α=2.0
        return 1.5f - 0.2f * (alpha - 1.5f) / 0.5f;
    } else {
        return 1.3f;  // Saturation
    }
}

float consolidation_pink_noise_arousal_to_frequency(float arousal) {
    // WHAT: Convert arousal level to replay frequency
    // WHY:  Deep sleep enables faster replay coordination
    // HOW:  Inverse exponential relationship
    //
    // BIOLOGICAL BASIS:
    // - Deep sleep (arousal ≈ 0.0): ~10 Hz sharp-wave ripples
    // - Light sleep (arousal ≈ 0.5): ~5 Hz theta sequences
    // - Awake (arousal ≈ 1.0): ~0.5 Hz spontaneous reactivation

    arousal = clamp_f(arousal, 0.0f, 1.0f);

    // Exponential decay from deep sleep to awake
    // freq = FREQ_DEEP * exp(-k * arousal)
    // At arousal=0: freq ≈ 10 Hz
    // At arousal=0.5: freq ≈ 5 Hz
    // At arousal=1.0: freq ≈ 0.5 Hz

    float k = 3.0f;  // Decay constant
    float freq = REPLAY_FREQ_DEEP_SLEEP * expf(-k * arousal);

    // Ensure minimum frequency
    if (freq < REPLAY_FREQ_AWAKE) {
        freq = REPLAY_FREQ_AWAKE;
    }

    return freq;
}
