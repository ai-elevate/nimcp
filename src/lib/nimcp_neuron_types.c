/**
 * @file nimcp_neuron_types.c
 * @brief Implementation of extended neuron type specialization system
 *
 * Provides type-specific dynamics for specialized neuron types including:
 * - Visual cortex neurons (edge detectors, orientation-selective)
 * - Auditory cortex neurons (frequency-tuned, onset detectors)
 * - Motor neurons (pattern generators, motoneurons)
 * - Interneurons (fast-spiking, bursting)
 * - Cortical pyramidal neurons (layer-specific)
 */

#include "nimcp_neuron_types.h"
#include "nimcp_neuralnet.h"
#include "utils/nimcp_time.h"
#include "utils/nimcp_validate.h"
#include <string.h>
#include <stdio.h>

// ============================================================================
// INTERNAL HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Apply Gabor filter for visual edge detection
 */
static float apply_gabor_filter(const visual_edge_params_t* params,
                                float input,
                                uint64_t timestamp) {
    // Simplified Gabor filter: orientation-tuned response
    // Real implementation would use spatial convolution

    // Use timestamp for phase modulation
    float time_sec = (float)timestamp / 1000000.0f;
    float phase_modulation = cosf(2.0f * M_PI * params->spatial_frequency * time_sec + params->phase);

    // Orientation tuning (simplified)
    float response = input * (1.0f + 0.5f * phase_modulation);

    // ON vs OFF center
    if (!params->on_center) {
        response = -response;
    }

    return response;
}

/**
 * @brief Apply bandpass filter for auditory frequency tuning
 */
static float apply_bandpass_filter(const auditory_frequency_params_t* params,
                                   float input,
                                   uint64_t timestamp) {
    // Simplified bandpass filter
    // Real implementation would use FFT or IIR filter

    float time_sec = (float)timestamp / 1000000.0f;

    // Center frequency response
    float carrier = sinf(2.0f * M_PI * params->center_frequency * time_sec);

    // Quality factor affects bandwidth
    float envelope = expf(-fabsf(input) / (params->q_factor + 1.0f));

    // Apply temporal integration
    float response = input * carrier * envelope;

    // Adaptation
    float adapted = response * (1.0f - params->adaptation_rate * 0.1f);

    return adapted;
}

/**
 * @brief Generate intrinsic rhythm for motor pattern generators
 */
static float generate_motor_rhythm(const motor_pattern_params_t* params,
                                   float input,
                                   uint64_t timestamp) {
    // Central pattern generator: intrinsic rhythmic activity

    if (!params->pacemaker) {
        return input; // Non-pacemaker neurons just relay input
    }

    float time_sec = (float)timestamp / 1000000.0f;

    // Generate rhythm
    float rhythm = sinf(2.0f * M_PI * params->rhythm_frequency * time_sec + params->phase_offset);

    // Burst modulation
    float burst_period = (params->burst_duration + params->interburst_interval) / 1000.0f;
    float burst_phase = fmodf(time_sec, burst_period);
    float burst_active = (burst_phase < (params->burst_duration / 1000.0f)) ? 1.0f : 0.0f;

    // Combine rhythm with input
    float output = (rhythm * 0.5f + 0.5f) * burst_active + input * 0.3f;

    return output;
}

/**
 * @brief Apply fast-spiking dynamics
 */
static float apply_fast_spiking_dynamics(const fast_spiking_params_t* params,
                                        float input) {
    // Fast membrane dynamics: quick response, minimal integration
    float tau_factor = 1.0f / (params->membrane_time_constant + 0.1f);
    return input * tau_factor;
}

/**
 * @brief Apply bursting dynamics
 */
static float apply_bursting_dynamics(const burst_params_t* params,
                                    float input,
                                    uint64_t timestamp) {
    // Bursting neuron: threshold crossing triggers burst

    if (input < params->burst_threshold) {
        return input; // Below threshold
    }

    // Above threshold: enter burst mode
    // Simplified: use timestamp to determine burst state
    float time_sec = (float)timestamp / 1000000.0f;
    float burst_period = params->interburst_interval / 1000.0f;
    float burst_phase = fmodf(time_sec, burst_period);
    float burst_duration_sec = params->burst_duration / 1000.0f;

    if (burst_phase < burst_duration_sec) {
        // In burst: amplify input
        return input * params->spikes_per_burst / 2.0f;
    } else {
        // Interburst: suppress
        return input * 0.1f;
    }
}

// ============================================================================
// TYPE-SPECIFIC DYNAMICS FUNCTIONS
// ============================================================================

float neuron_type_process_input(neuron_type_extended_t type,
                                const neuron_type_params_t* params,
                                float input,
                                uint64_t timestamp) {
    if (!params) {
        return input; // No params, pass through
    }

    switch (type) {
        // ========== BASIC TYPES ==========
        case NEURON_EXCITATORY:
        case NEURON_INHIBITORY:
            return input; // No special processing

        // ========== VISUAL NEURONS ==========
        case NEURON_VISUAL_EDGE:
            return apply_gabor_filter(&params->visual_edge, input, timestamp);

        case NEURON_VISUAL_ORIENTATION:
            // Simplified orientation selectivity
            return input * (1.0f + 0.3f * params->visual_orientation.direction_selectivity);

        case NEURON_VISUAL_DIRECTION:
        case NEURON_VISUAL_COLOR_OPPONENT:
        case NEURON_VISUAL_RECEPTIVE_FIELD:
            return input; // Simplified: pass through for now

        // ========== AUDITORY NEURONS ==========
        case NEURON_AUDITORY_FREQUENCY:
            return apply_bandpass_filter(&params->auditory_frequency, input, timestamp);

        case NEURON_AUDITORY_ONSET:
            // Onset detection: sharp response to input changes
            return input > params->auditory_onset.onset_threshold ? input * 2.0f : input * 0.1f;

        case NEURON_AUDITORY_ITD:
        case NEURON_AUDITORY_ILD:
        case NEURON_AUDITORY_COMPLEX:
            return input; // Simplified: pass through for now

        // ========== MOTOR NEURONS ==========
        case NEURON_MOTOR_PATTERN_GEN:
            return generate_motor_rhythm(&params->motor_pattern, input, timestamp);

        case NEURON_MOTOR_ALPHA:
        case NEURON_MOTOR_GAMMA:
        case NEURON_MOTOR_RENSHAW:
        case NEURON_MOTOR_PROPRIOCEPTIVE:
            return input; // Simplified: pass through for now

        // ========== INTERNEURONS ==========
        case NEURON_FAST_SPIKING:
            return apply_fast_spiking_dynamics(&params->fast_spiking, input);

        case NEURON_BURST_SPIKING:
            return apply_bursting_dynamics(&params->burst, input, timestamp);

        case NEURON_ADAPTIVE_SPIKING:
        case NEURON_CHANDELIER:
        case NEURON_BASKET:
            return input; // Simplified: pass through for now

        // ========== PYRAMIDAL NEURONS ==========
        case NEURON_PYRAMIDAL_L23:
        case NEURON_PYRAMIDAL_L5_THICK:
        case NEURON_PYRAMIDAL_L5_THIN:
        case NEURON_PYRAMIDAL_L6:
            return input; // Simplified: pass through for now

        // ========== SENSORY NEURONS ==========
        case NEURON_SENSORY_MECHANORECEPTOR:
        case NEURON_SENSORY_NOCICEPTOR:
        case NEURON_SENSORY_THERMORECEPTOR:
        case NEURON_SENSORY_PHOTORECEPTOR:
            return input; // Simplified: pass through for now

        default:
            return input; // Unknown type, pass through
    }
}

float neuron_type_get_threshold(neuron_type_extended_t type,
                                 const neuron_type_params_t* params,
                                 float base_threshold) {
    // Type-specific threshold adjustments
    switch (type) {
        case NEURON_EXCITATORY:
        case NEURON_INHIBITORY:
            return base_threshold;

        case NEURON_FAST_SPIKING:
            // Fast-spiking neurons have lower threshold for rapid firing
            return base_threshold * 0.8f;

        case NEURON_BURST_SPIKING:
            // Bursting neurons have higher threshold to prevent constant firing
            if (params) {
                return params->burst.burst_threshold;
            }
            return base_threshold * 1.2f;

        case NEURON_PYRAMIDAL_L23:
        case NEURON_PYRAMIDAL_L5_THICK:
        case NEURON_PYRAMIDAL_L5_THIN:
        case NEURON_PYRAMIDAL_L6:
            // Pyramidal neurons have slightly higher threshold
            return base_threshold * 1.1f;

        default:
            return base_threshold;
    }
}

float neuron_type_get_refractory_period(neuron_type_extended_t type,
                                         const neuron_type_params_t* params) {
    // Type-specific refractory periods (in milliseconds)
    switch (type) {
        case NEURON_EXCITATORY:
        case NEURON_INHIBITORY:
            return 3.0f; // Standard 3 ms

        case NEURON_FAST_SPIKING:
            // Very short refractory for high-frequency firing
            if (params) {
                // Inverse relationship with max firing rate
                return 1000.0f / (params->fast_spiking.max_firing_rate * 2.0f);
            }
            return 1.0f; // ~1 ms

        case NEURON_BURST_SPIKING:
            // Longer refractory between bursts
            if (params) {
                return params->burst.interburst_interval;
            }
            return 10.0f; // ~10 ms

        case NEURON_PYRAMIDAL_L23:
        case NEURON_PYRAMIDAL_L5_THICK:
        case NEURON_PYRAMIDAL_L5_THIN:
        case NEURON_PYRAMIDAL_L6:
            return 3.5f; // Slightly longer than basic types

        case NEURON_MOTOR_ALPHA:
        case NEURON_MOTOR_GAMMA:
            return 2.0f; // Fast motor neurons

        case NEURON_AUDITORY_ONSET:
            if (params) {
                return params->auditory_onset.refractory_duration;
            }
            return 5.0f; // Onset detectors need recovery time

        default:
            return 3.0f; // Default 3 ms
    }
}

bool neuron_type_is_excitatory(neuron_type_extended_t type) {
    switch (type) {
        // Basic types
        case NEURON_EXCITATORY:
            return true;
        case NEURON_INHIBITORY:
            return false;

        // Interneurons (400-499) are inhibitory
        case NEURON_FAST_SPIKING:
        case NEURON_ADAPTIVE_SPIKING:
        case NEURON_BURST_SPIKING:
        case NEURON_CHANDELIER:
        case NEURON_BASKET:
            return false;

        // Motor Renshaw cells are inhibitory
        case NEURON_MOTOR_RENSHAW:
            return false;

        // All other specialized types are excitatory
        default:
            return true;
    }
}

const char* neuron_type_get_name(neuron_type_extended_t type) {
    switch (type) {
        // Basic types
        case NEURON_EXCITATORY:         return "Excitatory";
        case NEURON_INHIBITORY:         return "Inhibitory";

        // Visual neurons
        case NEURON_VISUAL_EDGE:        return "Visual Edge Detector";
        case NEURON_VISUAL_ORIENTATION: return "Visual Orientation-Selective";
        case NEURON_VISUAL_DIRECTION:   return "Visual Direction-Selective";
        case NEURON_VISUAL_COLOR_OPPONENT: return "Visual Color-Opponent";
        case NEURON_VISUAL_RECEPTIVE_FIELD: return "Visual Receptive Field";

        // Auditory neurons
        case NEURON_AUDITORY_FREQUENCY: return "Auditory Frequency-Tuned";
        case NEURON_AUDITORY_ONSET:     return "Auditory Onset Detector";
        case NEURON_AUDITORY_ITD:       return "Auditory ITD-Sensitive";
        case NEURON_AUDITORY_ILD:       return "Auditory ILD-Sensitive";
        case NEURON_AUDITORY_COMPLEX:   return "Auditory Complex";

        // Motor neurons
        case NEURON_MOTOR_ALPHA:        return "Motor Alpha Motoneuron";
        case NEURON_MOTOR_GAMMA:        return "Motor Gamma Motoneuron";
        case NEURON_MOTOR_PATTERN_GEN:  return "Motor Pattern Generator";
        case NEURON_MOTOR_RENSHAW:      return "Motor Renshaw Cell";
        case NEURON_MOTOR_PROPRIOCEPTIVE: return "Motor Proprioceptive";

        // Interneurons
        case NEURON_FAST_SPIKING:       return "Fast-Spiking Interneuron";
        case NEURON_ADAPTIVE_SPIKING:   return "Adaptive-Spiking Interneuron";
        case NEURON_BURST_SPIKING:      return "Burst-Spiking Neuron";
        case NEURON_CHANDELIER:         return "Chandelier Cell";
        case NEURON_BASKET:             return "Basket Cell";

        // Pyramidal neurons
        case NEURON_PYRAMIDAL_L23:      return "Pyramidal L2/3";
        case NEURON_PYRAMIDAL_L5_THICK: return "Pyramidal L5 Thick-Tufted";
        case NEURON_PYRAMIDAL_L5_THIN:  return "Pyramidal L5 Thin-Tufted";
        case NEURON_PYRAMIDAL_L6:       return "Pyramidal L6";

        // Sensory neurons
        case NEURON_SENSORY_MECHANORECEPTOR: return "Mechanoreceptor";
        case NEURON_SENSORY_NOCICEPTOR:      return "Nociceptor";
        case NEURON_SENSORY_THERMORECEPTOR:  return "Thermoreceptor";
        case NEURON_SENSORY_PHOTORECEPTOR:   return "Photoreceptor";

        default:
            return "Unknown Type";
    }
}

// ============================================================================
// DEFAULT PARAMETERS
// ============================================================================

nimcp_result_t neuron_type_get_default_params(neuron_type_extended_t type,
                                               neuron_type_params_t* out_params) {
    if (!out_params) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Zero out the union
    memset(out_params, 0, sizeof(neuron_type_params_t));

    switch (type) {
        case NEURON_VISUAL_EDGE:
            out_params->visual_edge.orientation = 45.0f;
            out_params->visual_edge.spatial_frequency = 2.0f;
            out_params->visual_edge.phase = 0.0f;
            out_params->visual_edge.aspect_ratio = 0.5f;
            out_params->visual_edge.on_center = true;
            out_params->visual_edge.receptive_field_size = 5.0f;
            break;

        case NEURON_VISUAL_ORIENTATION:
            out_params->visual_orientation.preferred_orientation = 90.0f;
            out_params->visual_orientation.orientation_bandwidth = 30.0f;
            out_params->visual_orientation.direction_selectivity = 0.5f;
            out_params->visual_orientation.surround_suppression = 0.3f;
            break;

        case NEURON_AUDITORY_FREQUENCY:
            out_params->auditory_frequency.center_frequency = 1000.0f; // 1 kHz
            out_params->auditory_frequency.bandwidth = 1.0f; // 1 octave
            out_params->auditory_frequency.q_factor = 5.0f;
            out_params->auditory_frequency.temporal_integration = 10.0f; // 10 ms
            out_params->auditory_frequency.adaptation_rate = 0.1f;
            break;

        case NEURON_AUDITORY_ONSET:
            out_params->auditory_onset.onset_threshold = 0.5f;
            out_params->auditory_onset.refractory_duration = 5.0f; // 5 ms
            out_params->auditory_onset.decay_time_constant = 10.0f; // 10 ms
            out_params->auditory_onset.offset_sensitive = false;
            break;

        case NEURON_MOTOR_PATTERN_GEN:
            out_params->motor_pattern.rhythm_frequency = 2.0f; // 2 Hz
            out_params->motor_pattern.burst_duration = 100.0f; // 100 ms
            out_params->motor_pattern.interburst_interval = 400.0f; // 400 ms
            out_params->motor_pattern.phase_offset = 0.0f;
            out_params->motor_pattern.pacemaker = true;
            break;

        case NEURON_FAST_SPIKING:
            out_params->fast_spiking.max_firing_rate = 500.0f; // 500 Hz (fast-spiking interneurons)
            out_params->fast_spiking.membrane_time_constant = 5.0f; // 5 ms
            out_params->fast_spiking.spike_width = 0.5f; // 0.5 ms
            out_params->fast_spiking.afterhyperpolarization = 10.0f; // 10 mV
            break;

        case NEURON_BURST_SPIKING:
            out_params->burst.burst_threshold = 0.7f;
            out_params->burst.burst_duration = 50.0f; // 50 ms
            out_params->burst.spikes_per_burst = 5.0f;
            out_params->burst.interburst_interval = 200.0f; // 200 ms
            out_params->burst.calcium_decay = 100.0f; // 100 ms
            break;

        case NEURON_EXCITATORY:
        case NEURON_INHIBITORY:
            // No special parameters for basic types
            break;

        default:
            // Other types get default zero values
            break;
    }

    return NIMCP_SUCCESS;
}

// ============================================================================
// VALIDATION
// ============================================================================

nimcp_result_t neuron_type_validate_params(neuron_type_extended_t type,
                                            const neuron_type_params_t* params) {
    if (!params) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    switch (type) {
        case NEURON_VISUAL_EDGE:
            // Validate orientation (0-180 degrees)
            if (params->visual_edge.orientation < 0.0f ||
                params->visual_edge.orientation > 180.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            // Validate spatial frequency (must be positive)
            if (params->visual_edge.spatial_frequency <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            // Validate phase (0-2π)
            if (params->visual_edge.phase < 0.0f ||
                params->visual_edge.phase > 2.0f * M_PI) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            break;

        case NEURON_VISUAL_ORIENTATION:
            // Validate orientation (0-180 degrees)
            if (params->visual_orientation.preferred_orientation < 0.0f ||
                params->visual_orientation.preferred_orientation > 180.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            // Validate bandwidth (must be positive)
            if (params->visual_orientation.orientation_bandwidth <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            // Validate direction selectivity (0-1)
            if (params->visual_orientation.direction_selectivity < 0.0f ||
                params->visual_orientation.direction_selectivity > 1.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            break;

        case NEURON_AUDITORY_FREQUENCY:
            // Validate center frequency (0-20000 Hz, human hearing range)
            if (params->auditory_frequency.center_frequency < 0.0f ||
                params->auditory_frequency.center_frequency > 20000.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            // Validate bandwidth (must be positive)
            if (params->auditory_frequency.bandwidth <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            // Validate Q factor (must be positive)
            if (params->auditory_frequency.q_factor <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            break;

        case NEURON_AUDITORY_ONSET:
            // Validate threshold (0-1)
            if (params->auditory_onset.onset_threshold < 0.0f ||
                params->auditory_onset.onset_threshold > 1.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            // Validate refractory duration (must be positive)
            if (params->auditory_onset.refractory_duration <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            break;

        case NEURON_MOTOR_PATTERN_GEN:
            // Validate rhythm frequency (0-100 Hz, reasonable for CPG)
            if (params->motor_pattern.rhythm_frequency < 0.0f ||
                params->motor_pattern.rhythm_frequency > 100.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            // Validate burst duration (must be positive)
            if (params->motor_pattern.burst_duration <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            // Validate interburst interval (must be positive)
            if (params->motor_pattern.interburst_interval <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            break;

        case NEURON_FAST_SPIKING:
            // Validate max firing rate (must be > 50 Hz for fast-spiking)
            if (params->fast_spiking.max_firing_rate <= 50.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            // Validate membrane time constant (must be positive)
            if (params->fast_spiking.membrane_time_constant <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            break;

        case NEURON_BURST_SPIKING:
            // Validate burst threshold (0-1)
            if (params->burst.burst_threshold < 0.0f ||
                params->burst.burst_threshold > 1.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            // Validate burst duration (must be positive)
            if (params->burst.burst_duration <= 0.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            // Validate spikes per burst (must be >= 1)
            if (params->burst.spikes_per_burst < 1.0f) {
                return NIMCP_ERROR_INVALID_PARAM;
            }
            break;

        case NEURON_EXCITATORY:
        case NEURON_INHIBITORY:
            // No special parameters to validate
            break;

        default:
            // Other types: accept any parameters
            break;
    }

    return NIMCP_SUCCESS;
}
