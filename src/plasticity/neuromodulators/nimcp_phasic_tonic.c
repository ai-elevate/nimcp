/**
 * @file nimcp_phasic_tonic.c
 * @brief Implementation of phasic-tonic neuromodulator dynamics (Phase C2.2 Enhancement #2)
 *
 * WHAT: Implements burst vs baseline neurotransmitter release
 * WHY:  Model reward prediction errors and learning signals
 * HOW:  Dual-mode concentration with exponential decay
 *
 * @version Phase C2.2 Enhancement #2
 * @date 2025-11-12
 */

#include "plasticity/neuromodulators/nimcp_phasic_tonic.h"
#include <math.h>
#include <string.h>

// ============================================================================
// Default Configurations
// ============================================================================

phasic_tonic_config_t phasic_tonic_config_dopamine_default(void) {
    phasic_tonic_config_t config = {
        .initial_tonic = DOPAMINE_TONIC_BASELINE,
        .tonic_target = DOPAMINE_TONIC_BASELINE,
        .tonic_range_min = DOPAMINE_TONIC_RANGE_MIN,
        .tonic_range_max = DOPAMINE_TONIC_RANGE_MAX,
        .homeostatic_tau = DOPAMINE_HOMEOSTATIC_TAU,

        .burst_decay_tau = DOPAMINE_BURST_DECAY_TAU,
        .burst_amplitude_scale = 1.0f,
        .max_burst_amplitude = DOPAMINE_PHASIC_PEAK,

        .autoreceptor_sensitivity = 0.3f,  // Moderate D2 autoreceptor feedback
        .feedback_tau = 1.0f
    };
    return config;
}

phasic_tonic_config_t phasic_tonic_config_serotonin_default(void) {
    phasic_tonic_config_t config = {
        .initial_tonic = SEROTONIN_TONIC_BASELINE,
        .tonic_target = SEROTONIN_TONIC_BASELINE,
        .tonic_range_min = SEROTONIN_TONIC_BASELINE * 0.5f,
        .tonic_range_max = SEROTONIN_TONIC_BASELINE * 2.0f,
        .homeostatic_tau = 120.0f,  // Slower than dopamine

        .burst_decay_tau = SEROTONIN_BURST_DECAY_TAU,
        .burst_amplitude_scale = 0.8f,
        .max_burst_amplitude = SEROTONIN_TONIC_BASELINE * 10.0f,

        .autoreceptor_sensitivity = 0.5f,  // Strong 5-HT1A autoreceptor
        .feedback_tau = 2.0f
    };
    return config;
}

phasic_tonic_config_t phasic_tonic_config_norepinephrine_default(void) {
    phasic_tonic_config_t config = {
        .initial_tonic = NOREPINEPHRINE_TONIC_BASELINE,
        .tonic_target = NOREPINEPHRINE_TONIC_BASELINE,
        .tonic_range_min = NOREPINEPHRINE_TONIC_BASELINE * 0.3f,
        .tonic_range_max = NOREPINEPHRINE_TONIC_BASELINE * 3.0f,
        .homeostatic_tau = 30.0f,  // Fast arousal regulation

        .burst_decay_tau = NOREPINEPHRINE_BURST_DECAY_TAU,
        .burst_amplitude_scale = 1.2f,  // Strong bursts for salience
        .max_burst_amplitude = NOREPINEPHRINE_TONIC_BASELINE * 20.0f,

        .autoreceptor_sensitivity = 0.4f,  // α2 autoreceptor
        .feedback_tau = 0.5f
    };
    return config;
}

// ============================================================================
// Initialization
// ============================================================================

void phasic_tonic_init(
    phasic_tonic_state_t* state,
    const phasic_tonic_config_t* config,
    uint64_t current_time_us
) {
    memset(state, 0, sizeof(*state));

    // Tonic initialization
    state->tonic_level = config->initial_tonic;
    state->tonic_target = config->tonic_target;
    state->tonic_min = config->tonic_range_min;
    state->tonic_max = config->tonic_range_max;
    state->homeostatic_tau = config->homeostatic_tau;

    // Phasic initialization
    state->phasic_burst = 0.0f;
    state->burst_decay_tau = config->burst_decay_tau;
    state->in_burst_state = false;
    state->burst_start_time_us = 0;
    state->burst_duration_ms = DOPAMINE_BURST_DURATION_MS;  // Default

    // Burst parameters
    state->burst_amplitude_scale = config->burst_amplitude_scale;
    state->max_burst_amplitude = config->max_burst_amplitude;
    state->min_burst_amplitude = config->tonic_target * 0.5f;  // Half of tonic
    state->burst_threshold = config->tonic_target * 0.1f;

    // Autoreceptor
    state->autoreceptor_sensitivity = config->autoreceptor_sensitivity;
    state->feedback_tau = config->feedback_tau;

    // Output
    state->total_concentration = state->tonic_level;
    state->release_rate = 0.0f;

    // Statistics
    state->burst_count = 0;
    state->last_burst_time_us = current_time_us;
    state->avg_inter_burst_interval = 0.0f;
}

// ============================================================================
// Core Update Function
// ============================================================================

void phasic_tonic_update(
    phasic_tonic_state_t* state,
    float dt,
    uint64_t current_time_us
) {
    // === 1. Update Homeostatic Tonic Regulation ===
    float tonic_alpha = expf(-dt / state->homeostatic_tau);
    state->tonic_level = tonic_alpha * state->tonic_level +
                        (1.0f - tonic_alpha) * state->tonic_target;

    // Clamp tonic to physiological range
    if (state->tonic_level < state->tonic_min) {
        state->tonic_level = state->tonic_min;
    } else if (state->tonic_level > state->tonic_max) {
        state->tonic_level = state->tonic_max;
    }

    // === 2. Update Phasic Burst Decay ===
    if (state->in_burst_state) {
        // Exponential decay
        float decay_alpha = expf(-dt / state->burst_decay_tau);
        state->phasic_burst *= decay_alpha;

        // Check burst termination
        uint64_t elapsed_us = current_time_us - state->burst_start_time_us;
        uint64_t duration_us = (uint64_t)state->burst_duration_ms * 1000ULL;

        if (elapsed_us >= duration_us || state->phasic_burst < state->burst_threshold) {
            state->in_burst_state = false;
            state->phasic_burst = 0.0f;
        }
    }

    // === 3. Apply Autoreceptor Feedback ===
    // High concentration → negative feedback → reduced release
    float total = state->tonic_level + state->phasic_burst;
    float feedback_factor = 1.0f / (1.0f + state->autoreceptor_sensitivity * total);

    // === 4. Compute Release Rate ===
    // Release rate proportional to concentration and feedback
    state->release_rate = total * feedback_factor / dt;

    // === 5. Update Total Concentration ===
    state->total_concentration = state->tonic_level + state->phasic_burst;
}

// ============================================================================
// Burst Triggering
// ============================================================================

bool phasic_tonic_trigger_burst(
    phasic_tonic_state_t* state,
    float amplitude,
    uint32_t duration_ms,
    uint64_t current_time_us
) {
    // Check minimum amplitude threshold
    if (amplitude < state->min_burst_amplitude) {
        return false;  // Too weak to trigger
    }

    // Clamp amplitude to maximum
    if (amplitude > state->max_burst_amplitude) {
        amplitude = state->max_burst_amplitude;
    }

    // Apply amplitude scaling
    amplitude *= state->burst_amplitude_scale;

    // If already bursting, add to current burst (superposition)
    if (state->in_burst_state) {
        state->phasic_burst += amplitude;
        if (state->phasic_burst > state->max_burst_amplitude) {
            state->phasic_burst = state->max_burst_amplitude;
        }
        // Don't reset timing - let current burst continue
    } else {
        // Start new burst
        state->phasic_burst = amplitude;
        state->in_burst_state = true;
        state->burst_start_time_us = current_time_us;
        state->burst_duration_ms = (duration_ms > 0) ? duration_ms : DOPAMINE_BURST_DURATION_MS;

        // Update statistics
        state->burst_count++;

        // Update average inter-burst interval
        if (state->burst_count > 1) {
            float interval = (current_time_us - state->last_burst_time_us) / 1000000.0f;  // Convert to seconds
            float alpha = 0.1f;  // Exponential moving average
            state->avg_inter_burst_interval = alpha * interval +
                                             (1.0f - alpha) * state->avg_inter_burst_interval;
        }

        state->last_burst_time_us = current_time_us;
    }

    // Update total concentration immediately so it reflects the burst
    // (normally this would happen in phasic_tonic_update(), but we need immediate effect)
    state->total_concentration = state->tonic_level + state->phasic_burst;

    return true;
}

void phasic_tonic_induce_dip(
    phasic_tonic_state_t* state,
    float magnitude
) {
    // Clamp magnitude
    if (magnitude < 0.0f) magnitude = 0.0f;
    if (magnitude > 1.0f) magnitude = 1.0f;

    // Temporarily reduce tonic level
    float reduction = state->tonic_level * magnitude;
    state->tonic_level -= reduction;

    // Ensure doesn't go below minimum
    if (state->tonic_level < state->tonic_min) {
        state->tonic_level = state->tonic_min;
    }

    // Update total concentration immediately
    state->total_concentration = state->tonic_level + state->phasic_burst;
}

// ============================================================================
// TD Error Encoding
// ============================================================================

bool phasic_tonic_encode_td_error(
    phasic_tonic_state_t* state,
    float td_error,
    uint64_t current_time_us
) {
    if (td_error > 0.0f) {
        // Positive TD error → phasic burst
        // Scale burst amplitude proportional to error magnitude
        float burst_amplitude = td_error * state->max_burst_amplitude;
        return phasic_tonic_trigger_burst(state, burst_amplitude, 0, current_time_us);

    } else if (td_error < 0.0f) {
        // Negative TD error → tonic dip
        float dip_magnitude = fabs(td_error);  // Convert to positive
        phasic_tonic_induce_dip(state, dip_magnitude * 0.5f);  // 50% scaling
        return false;

    } else {
        // Zero TD error → no change (expected outcome)
        return false;
    }
}

// ============================================================================
// Accessors
// ============================================================================

float phasic_tonic_get_concentration(const phasic_tonic_state_t* state) {
    return state->total_concentration;
}

float phasic_tonic_get_release_rate(const phasic_tonic_state_t* state) {
    return state->release_rate;
}

void phasic_tonic_set_tonic_target(
    phasic_tonic_state_t* state,
    float new_target
) {
    // Clamp to physiological range
    if (new_target < state->tonic_min) {
        new_target = state->tonic_min;
    } else if (new_target > state->tonic_max) {
        new_target = state->tonic_max;
    }

    state->tonic_target = new_target;
}

void phasic_tonic_apply_autoreceptor_modulation(
    phasic_tonic_state_t* state,
    float modulation
) {
    // Clamp modulation to reasonable range
    if (modulation < 0.0f) modulation = 0.0f;
    if (modulation > 2.0f) modulation = 2.0f;

    // Apply to current tonic level (immediate effect)
    state->tonic_level *= modulation;

    // Clamp to range
    if (state->tonic_level < state->tonic_min) {
        state->tonic_level = state->tonic_min;
    } else if (state->tonic_level > state->tonic_max) {
        state->tonic_level = state->tonic_max;
    }
}

// ============================================================================
// Statistics and Monitoring
// ============================================================================

void phasic_tonic_get_burst_statistics(
    const phasic_tonic_state_t* state,
    uint32_t* burst_count,
    float* avg_interval,
    float* time_since_last,
    uint64_t current_time_us
) {
    if (burst_count) {
        *burst_count = state->burst_count;
    }

    if (avg_interval) {
        *avg_interval = state->avg_inter_burst_interval;
    }

    if (time_since_last) {
        uint64_t elapsed_us = current_time_us - state->last_burst_time_us;
        *time_since_last = elapsed_us / 1000000.0f;  // Convert to seconds
    }
}

void phasic_tonic_reset(
    phasic_tonic_state_t* state,
    uint64_t current_time_us
) {
    // Reset to baseline
    state->tonic_level = state->tonic_target;
    state->phasic_burst = 0.0f;
    state->in_burst_state = false;
    state->total_concentration = state->tonic_level;
    state->release_rate = 0.0f;

    // Reset statistics
    state->burst_count = 0;
    state->last_burst_time_us = current_time_us;
    state->avg_inter_burst_interval = 0.0f;
}
