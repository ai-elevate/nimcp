/**
 * @file nimcp_neuron_types.c
 * @brief Phase 8.7: Implementation of specialized neuron type system
 *
 * This file implements biologically-motivated compute functions for each
 * specialized neuron type defined in Phase 8.7.
 *
 * @author NIMCP Phase 8.7
 * @date 2025-11-08
 */

#include "nimcp_neuron_types.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// INTERNAL HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Compute mean of array
 *
 * WHAT: Calculate average value
 * WHY: Used for uncertainty estimation
 * HOW: Sum all values and divide by count
 *
 * COMPLEXITY: O(n)
 */
static float compute_mean(const float* values, uint32_t count) {
    if (!values || count == 0) return 0.0f;

    float sum = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        sum += values[i];
    }
    return sum / (float)count;
}

/**
 * @brief Compute variance of array
 *
 * WHAT: Calculate variance (spread) of values
 * WHY: Used for uncertainty estimation in metacognitive neurons
 * HOW: variance = E[(X - μ)^2] = E[X^2] - μ^2
 *
 * COMPLEXITY: O(n)
 */
static float compute_variance(const float* values, uint32_t count, float mean) {
    if (!values || count == 0) return 0.0f;

    float sum_sq_diff = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float diff = values[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / (float)count;
}

/**
 * @brief Rotate coordinates for orientation tuning
 *
 * WHAT: Rotate (x,y) by angle theta
 * WHY: Gabor filters need rotation for orientation selectivity
 * HOW: Standard 2D rotation matrix
 */
static void rotate_coords(float x, float y, float theta_rad,
                         float* out_x, float* out_y) {
    float cos_theta = cosf(theta_rad);
    float sin_theta = sinf(theta_rad);
    *out_x = x * cos_theta + y * sin_theta;
    *out_y = -x * sin_theta + y * cos_theta;
}

// ============================================================================
// GENERIC NEURON MODEL IMPLEMENTATIONS
// ============================================================================

float compute_lif_neuron(const lif_params_t* params, float input,
                         float state, float dt) {
    if (!params) return state;

    /**
     * WHAT: Leaky Integrate-and-Fire neuron dynamics
     * WHY: Most common computational neuron model, efficient and interpretable
     * HOW: dV/dt = (E_L - V + R*I) / tau_m
     *      where V = membrane potential, E_L = rest potential,
     *      I = input current, tau_m = membrane time constant
     *
     * BIOLOGICAL MOTIVATION:
     * Models RC circuit of cell membrane with leak conductance.
     * When V reaches threshold, neuron fires and resets.
     *
     * COMPLEXITY: O(1)
     */

    // Euler integration: V(t+dt) = V(t) + dt * dV/dt
    float dV = ((params->rest_potential - state + input) / params->tau_membrane) * dt;
    float new_state = state + dV;

    // Clamp to reasonable range
    if (new_state < params->reset_potential) {
        new_state = params->reset_potential;
    }

    return new_state;
}

void compute_izhikevich_neuron(const izh_params_t* params, float input,
                               float v, float u, float dt,
                               float* out_v, float* out_u) {
    if (!params || !out_v || !out_u) {
        if (out_v) *out_v = v;
        if (out_u) *out_u = u;
        return;
    }

    /**
     * WHAT: Izhikevich neuron model - simple yet biologically realistic
     * WHY: Reproduces 20+ distinct cortical neuron firing patterns
     * HOW: dv/dt = 0.04v^2 + 5v + 140 - u + I
     *      du/dt = a(bv - u)
     *
     * BIOLOGICAL MOTIVATION:
     * Simplified Hodgkin-Huxley model that captures key dynamics:
     * - Fast voltage dynamics (v equation)
     * - Slow recovery dynamics (u equation)
     * - Parameter space maps to distinct neuron types
     *
     * PARAMETERS:
     * a: recovery time scale (smaller = slower recovery)
     * b: sensitivity of recovery to subthreshold fluctuations
     * c: after-spike reset value of v
     * d: after-spike reset increment of u
     *
     * REFERENCE: Izhikevich (2003) "Simple model of spiking neurons"
     * COMPLEXITY: O(1)
     */

    // Euler integration with 0.5ms sub-steps for stability
    const float substep = 0.5f;
    int num_substeps = (int)(dt / substep) + 1;
    float actual_substep = dt / num_substeps;

    float v_curr = v;
    float u_curr = u;

    for (int i = 0; i < num_substeps; i++) {
        // dv/dt = 0.04v^2 + 5v + 140 - u + I
        float dv = (0.04f * v_curr * v_curr + 5.0f * v_curr + 140.0f - u_curr + input) * actual_substep;

        // du/dt = a(bv - u)
        float du = params->a * (params->b * v_curr - u_curr) * actual_substep;

        v_curr += dv;
        u_curr += du;

        // Check for spike (v >= 30 mV)
        if (v_curr >= 30.0f) {
            v_curr = params->c;  // reset voltage
            u_curr += params->d;  // reset recovery
        }
    }

    *out_v = v_curr;
    *out_u = u_curr;
}

// ============================================================================
// VISUAL CORTEX (V1) IMPLEMENTATIONS
// ============================================================================

float compute_v1_simple_cell(const v1_simple_cell_params_t* params,
                             const float* inputs,
                             uint32_t input_width, uint32_t input_height,
                             float center_x, float center_y) {
    if (!params || !inputs) return 0.0f;

    /**
     * WHAT: Gabor filter for oriented edge detection
     * WHY: Models V1 simple cells (Hubel & Wiesel, 1962)
     * HOW: G(x,y) = exp(-(x'^2 + γ^2*y'^2)/(2σ^2)) * cos(2πf*x' + φ)
     *      where (x',y') are coordinates rotated by orientation angle
     *
     * BIOLOGICAL MOTIVATION:
     * V1 simple cells have elongated receptive fields that respond to
     * edges at specific orientations. They are phase-dependent (ON vs OFF).
     *
     * GABOR FILTER COMPONENTS:
     * 1. Gaussian envelope: spatial localization
     * 2. Sinusoidal carrier: orientation/frequency selectivity
     * 3. Rotation: orientation tuning
     *
     * PARAMETERS:
     * - orientation: preferred angle (0-180 degrees)
     * - spatial_frequency: cycles per degree of visual angle
     * - phase: 0 = ON-center, π = OFF-center
     * - sigma: envelope width
     *
     * COMPLEXITY: O(w*h) where w,h are receptive field size
     */

    float response = 0.0f;
    float theta_rad = params->orientation * M_PI / 180.0f;
    float gamma = params->aspect_ratio;
    float sigma_sq = params->sigma * params->sigma;

    // Define receptive field size (typically 3-4 sigma)
    int rf_size = (int)(3.0f * params->sigma);
    int count = 0;

    // Convolve Gabor filter over receptive field
    for (int dy = -rf_size; dy <= rf_size; dy++) {
        for (int dx = -rf_size; dx <= rf_size; dx++) {
            // Input coordinates
            int px = (int)(center_x + dx);
            int py = (int)(center_y + dy);

            // Check bounds
            if (px < 0 || px >= (int)input_width || py < 0 || py >= (int)input_height) {
                continue;
            }

            // Rotate coordinates by orientation angle
            float x_rot, y_rot;
            rotate_coords((float)dx, (float)dy, theta_rad, &x_rot, &y_rot);

            // Gaussian envelope
            float gaussian = expf(-(x_rot * x_rot + gamma * gamma * y_rot * y_rot) / (2.0f * sigma_sq));

            // Sinusoidal carrier
            float sinusoid = cosf(2.0f * M_PI * params->spatial_frequency * x_rot + params->phase);

            // Gabor filter value
            float gabor = gaussian * sinusoid;

            // Apply to input
            float input_val = inputs[py * input_width + px];
            response += gabor * input_val;
            count++;
        }
    }

    // Normalize by receptive field size
    if (count > 0) {
        response /= count;
    }

    // ON-center vs OFF-center
    if (!params->on_center) {
        response = -response;
    }

    // Half-wave rectification (neurons only respond to positive)
    return fmaxf(0.0f, response);
}

float compute_v1_complex_cell(const v1_complex_cell_params_t* params,
                              const float* simple_cell_responses,
                              uint32_t num_simple_cells) {
    if (!params || !simple_cell_responses || num_simple_cells == 0) return 0.0f;

    /**
     * WHAT: Energy model for phase-invariant edge detection
     * WHY: Models V1 complex cells (Hubel & Wiesel, 1962)
     * HOW: E = sqrt(even^2 + odd^2) where even/odd are 90° phase shifted
     *
     * BIOLOGICAL MOTIVATION:
     * Complex cells pool over multiple simple cells with different phases,
     * giving position-invariant edge detection. They respond to edges
     * regardless of exact position within receptive field.
     *
     * ENERGY MODEL:
     * Pool responses from simple cells with phases 0° and 90°
     * Energy = sqrt(R_even^2 + R_odd^2)
     * This gives phase invariance while preserving orientation selectivity
     *
     * COMPLEXITY: O(n) where n = num_simple_cells
     */

    // Pool responses (typically first half are even-symmetric, second half odd)
    float even_sum = 0.0f;
    float odd_sum = 0.0f;
    uint32_t half = num_simple_cells / 2;

    // Even-symmetric responses (phase = 0)
    for (uint32_t i = 0; i < half; i++) {
        even_sum += simple_cell_responses[i];
    }

    // Odd-symmetric responses (phase = π/2)
    for (uint32_t i = half; i < num_simple_cells; i++) {
        odd_sum += simple_cell_responses[i];
    }

    // Normalize
    if (half > 0) {
        even_sum /= half;
        odd_sum /= (num_simple_cells - half);
    }

    // Energy model: sqrt(even^2 + odd^2)
    float energy = sqrtf(even_sum * even_sum + odd_sum * odd_sum);

    // Direction selectivity: suppress responses to non-preferred direction
    // (simplified: assume even/odd sign indicates direction)
    float direction_factor = 1.0f;
    if (params->direction_selectivity > 0.0f) {
        float direction_signal = even_sum - odd_sum;
        if (direction_signal < 0.0f) {
            direction_factor = 1.0f - params->direction_selectivity;
        }
    }

    return energy * direction_factor;
}

// ============================================================================
// AUDITORY CORTEX (A1) IMPLEMENTATIONS
// ============================================================================

float compute_a1_frequency_tuned(const a1_frequency_params_t* params,
                                 const float* audio_input,
                                 uint32_t signal_length,
                                 float sample_rate,
                                 uint64_t timestamp) {
    if (!params || !audio_input || signal_length == 0) return 0.0f;

    /**
     * WHAT: Bandpass filtering for tonotopic frequency selectivity
     * WHY: Models A1 frequency-tuned neurons (Schreiner et al., 2000)
     * HOW: Simple IIR bandpass filter centered at preferred frequency
     *
     * BIOLOGICAL MOTIVATION:
     * Primary auditory cortex (A1) is organized tonotopically - neurons
     * are spatially arranged by preferred frequency, creating a frequency
     * map from low to high.
     *
     * TONOTOPIC ORGANIZATION:
     * - Each neuron responds to narrow frequency band
     * - Q factor determines sharpness of tuning
     * - Temporal integration window smooths response
     *
     * SIMPLIFIED IMPLEMENTATION:
     * Real auditory processing uses cochlear filterbank (gammatone filters)
     * This is simplified bandpass with resonance at center frequency
     *
     * COMPLEXITY: O(n) where n = signal_length
     */

    // Simplified bandpass: multiply by sinusoid at center frequency
    // This extracts energy at that frequency (like heterodyning)
    float response = 0.0f;
    float dt = 1.0f / sample_rate;

    // Determine integration window size (in samples)
    uint32_t window_samples = (uint32_t)(params->integration_window * sample_rate / 1000.0f);
    if (window_samples > signal_length) window_samples = signal_length;
    if (window_samples == 0) window_samples = 1;

    // Bandpass filtering via modulation
    for (uint32_t i = 0; i < window_samples; i++) {
        float t = i * dt;

        // Carrier at center frequency
        float carrier = sinf(2.0f * M_PI * params->center_frequency * t);

        // Modulate input
        float modulated = audio_input[i] * carrier;

        // Low-pass filter the modulated signal (envelope detection)
        // Q factor determines bandwidth
        float alpha = expf(-1.0f / params->q_factor);
        response = alpha * response + (1.0f - alpha) * modulated;
    }

    // Normalize
    response /= window_samples;

    // Adaptation to sustained input
    response *= (1.0f - params->adaptation_rate * 0.1f);

    // Half-wave rectification
    return fmaxf(0.0f, response);
}

float compute_a1_coincidence_detector(const a1_coincidence_params_t* params,
                                      const float* left_input,
                                      const float* right_input,
                                      uint32_t num_samples,
                                      uint64_t timestamp) {
    if (!params || !left_input || !right_input || num_samples == 0) return 0.0f;

    /**
     * WHAT: Coincidence detection for binaural sound localization
     * WHY: Models superior olivary complex neurons (Jeffress, 1948)
     * HOW: Cross-correlation of left/right inputs with high temporal precision
     *
     * BIOLOGICAL MOTIVATION:
     * Medial superior olive (MSO) contains coincidence detector neurons
     * that compute interaural time differences (ITDs) for sound localization.
     * These neurons have very precise temporal integration (~100 µs).
     *
     * JEFFRESS MODEL:
     * Neurons are arranged in delay lines - each neuron fires maximally
     * when inputs from both ears arrive simultaneously. Different delays
     * create spatial map of sound location.
     *
     * ITD COMPUTATION:
     * ITD = (distance to left ear - distance to right ear) / speed_of_sound
     * ITD range: ±700 µs for humans (interaural distance ~21 cm)
     *
     * COMPLEXITY: O(n) where n = num_samples in integration window
     */

    // Compute cross-correlation over short integration window
    float coincidence = 0.0f;
    uint32_t window_size = num_samples;

    // Short integration window for temporal precision
    if (params->integration_window > 0.0f) {
        // Assume sample rate ~44.1 kHz typical for audio
        float sample_rate = 44100.0f;
        uint32_t window_samples = (uint32_t)(params->integration_window * sample_rate / 1000.0f);
        if (window_samples < window_size) {
            window_size = window_samples;
        }
    }

    // Cross-correlation: sum(L[i] * R[i])
    for (uint32_t i = 0; i < window_size; i++) {
        coincidence += left_input[i] * right_input[i];
    }

    // Normalize by window size
    coincidence /= window_size;

    // Apply decay
    float decay_factor = expf(-params->decay_rate * 0.01f);
    coincidence *= decay_factor;

    // Threshold for coincidence detection
    if (coincidence < params->threshold) {
        coincidence = 0.0f;
    }

    return coincidence;
}

// ============================================================================
// COGNITIVE/METACOGNITIVE IMPLEMENTATIONS
// ============================================================================

float compute_metacognitive(const metacognitive_params_t* params,
                            const float* inputs, uint32_t num_inputs,
                            const float* input_history, uint32_t history_size,
                            float* out_confidence, float* out_uncertainty) {
    if (!params || !inputs || num_inputs == 0) {
        if (out_confidence) *out_confidence = 0.0f;
        if (out_uncertainty) *out_uncertainty = 1.0f;
        return 0.0f;
    }

    /**
     * WHAT: Metacognitive monitoring via uncertainty estimation
     * WHY: Models prefrontal confidence monitoring (Fleming & Dolan, 2012)
     * HOW: Estimate uncertainty from input variance, compute confidence
     *
     * BIOLOGICAL MOTIVATION:
     * Metacognition = "thinking about thinking"
     * Prefrontal cortex tracks decision confidence and uncertainty, enabling:
     * - Adaptive behavior based on confidence
     * - Error monitoring and correction
     * - Knowing when to seek more information
     *
     * UNCERTAINTY ESTIMATION:
     * Use variance of inputs as proxy for uncertainty
     * High variance = high uncertainty = low confidence
     * Low variance = low uncertainty = high confidence
     *
     * CONFIDENCE COMPUTATION:
     * Confidence = 1 / (1 + β * uncertainty)
     * where β controls sensitivity to uncertainty
     *
     * SECOND-ORDER PROCESSING:
     * This is "second-order" - monitoring first-order sensory/decision signals
     *
     * COMPLEXITY: O(n) where n = history_size
     */

    // Compute mean of current inputs
    float mean = compute_mean(inputs, num_inputs);

    // Compute variance over history (if available)
    float variance = 0.0f;
    if (input_history && history_size > 0) {
        float history_mean = compute_mean(input_history, history_size);
        variance = compute_variance(input_history, history_size, history_mean);
    } else {
        // Use current inputs only
        variance = compute_variance(inputs, num_inputs, mean);
    }

    // Uncertainty = variance (scaled by beta)
    float uncertainty = variance * params->uncertainty_beta;

    // Confidence = 1 / (1 + uncertainty)
    // This gives sigmoid-like mapping: low uncertainty -> high confidence
    float confidence = 1.0f / (1.0f + uncertainty);

    // Output values if requested
    if (out_confidence) *out_confidence = confidence;
    if (out_uncertainty) *out_uncertainty = uncertainty;

    // Metacognitive activation reflects confidence
    // High confidence -> strong signal, low confidence -> weak signal
    float activation = mean * confidence;

    // Apply confidence threshold
    if (confidence < params->confidence_threshold) {
        activation *= 0.5f;  // Attenuate low-confidence signals
    }

    return activation;
}

float compute_executive_control(const executive_control_params_t* params,
                                float goal_input,
                                float current_state,
                                float dt,
                                bool is_delay_period) {
    if (!params) return current_state;

    /**
     * WHAT: Executive control with goal maintenance and top-down modulation
     * WHY: Models dorsolateral PFC executive functions (Miller & Cohen, 2001)
     * HOW: Sustained activity during delays, goal-directed modulation
     *
     * BIOLOGICAL MOTIVATION:
     * Dorsolateral prefrontal cortex (dlPFC) maintains goal representations
     * and modulates processing throughout the brain:
     * - Working memory: sustained activity during delay periods
     * - Attention: top-down biasing of sensory areas
     * - Cognitive control: maintaining task rules and goals
     *
     * EXECUTIVE CONTROL PROPERTIES:
     * 1. Goal maintenance: persist activity in absence of input
     * 2. Top-down modulation: amplify task-relevant signals
     * 3. Decay: gradual loss of activity (working memory decay)
     * 4. Threshold boosting: increase sensitivity during goal-directed state
     *
     * DELAY PERIOD ACTIVITY:
     * During delays (e.g., between stimulus and response), dlPFC neurons
     * maintain elevated firing to bridge temporal gap.
     *
     * COMPLEXITY: O(1)
     */

    float new_state = current_state;

    if (is_delay_period && params->delay_activity) {
        // DELAY PERIOD: Maintain activity via recurrent excitation
        // dS/dt = -decay * S + maintenance * S
        // This creates sustained firing in absence of input

        float maintenance_input = params->goal_maintenance * current_state;
        float decay = params->decay_rate * current_state;

        // Update: maintain - decay
        new_state = current_state + dt * (maintenance_input - decay);

        // Prevent unbounded growth
        if (new_state > 1.0f) new_state = 1.0f;

    } else {
        // ACTIVE PERIOD: Process goal input with amplification

        // Top-down modulation amplifies goal-relevant signals
        float modulated_input = goal_input * (1.0f + params->modulation_strength);

        // Integrate input with decay
        float integration = modulated_input - params->decay_rate * current_state;
        new_state = current_state + dt * integration;
    }

    // Clamp to [0, 1]
    if (new_state < 0.0f) new_state = 0.0f;
    if (new_state > 1.0f) new_state = 1.0f;

    return new_state;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

nimcp_result_t neuron_type_get_default_params(neuron_type_t type,
                                               neuron_type_params_t* out_params) {
    if (!out_params) return NIMCP_ERROR_INVALID_PARAM;

    // Zero out union
    memset(out_params, 0, sizeof(neuron_type_params_t));

    switch (type) {
        case NEURON_EXCITATORY:
        case NEURON_INHIBITORY:
            // Backward compatibility: use LIF as default
            out_params->lif.tau_membrane = 20.0f;
            out_params->lif.rest_potential = -70.0f;
            out_params->lif.threshold = -55.0f;
            out_params->lif.reset_potential = -75.0f;
            out_params->lif.refractory_period = 2.0f;
            break;

        case NEURON_GENERIC_LIF:
            out_params->lif.tau_membrane = 20.0f;      // 20 ms
            out_params->lif.rest_potential = -70.0f;   // -70 mV
            out_params->lif.threshold = -55.0f;        // -55 mV
            out_params->lif.reset_potential = -75.0f;  // -75 mV
            out_params->lif.refractory_period = 2.0f;  // 2 ms
            break;

        case NEURON_GENERIC_IZHIKEVICH:
            // Regular spiking (RS) cortical neuron
            out_params->izhikevich.a = 0.02f;
            out_params->izhikevich.b = 0.2f;
            out_params->izhikevich.c = -65.0f;
            out_params->izhikevich.d = 8.0f;
            break;

        case NEURON_V1_SIMPLE_CELL:
            out_params->v1_simple.orientation = 45.0f;        // 45 degrees
            out_params->v1_simple.spatial_frequency = 2.0f;   // 2 cycles/degree
            out_params->v1_simple.phase = 0.0f;               // ON-center
            out_params->v1_simple.aspect_ratio = 0.5f;        // Elongated
            out_params->v1_simple.sigma = 2.0f;               // 2 degree width
            out_params->v1_simple.on_center = true;
            break;

        case NEURON_V1_COMPLEX_CELL:
            out_params->v1_complex.orientation = 90.0f;         // Vertical
            out_params->v1_complex.direction_selectivity = 0.5f; // Moderate
            out_params->v1_complex.surround_suppression = 0.3f;  // Weak
            out_params->v1_complex.pooling_size = 4.0f;         // 4 degrees
            break;

        case NEURON_A1_FREQUENCY_TUNED:
            out_params->a1_frequency.center_frequency = 1000.0f;  // 1 kHz
            out_params->a1_frequency.q_factor = 5.0f;             // Moderate tuning
            out_params->a1_frequency.bandwidth = 200.0f;          // 200 Hz
            out_params->a1_frequency.integration_window = 10.0f;  // 10 ms
            out_params->a1_frequency.adaptation_rate = 0.1f;
            break;

        case NEURON_A1_COINCIDENCE_DETECTOR:
            out_params->a1_coincidence.integration_window = 1.0f;   // 1 ms
            out_params->a1_coincidence.temporal_precision = 100.0f; // 100 µs
            out_params->a1_coincidence.threshold = 0.3f;
            out_params->a1_coincidence.decay_rate = 0.1f;
            break;

        case NEURON_METACOGNITIVE:
            out_params->metacognitive.confidence_threshold = 0.5f;  // 50%
            out_params->metacognitive.uncertainty_window = 100.0f;  // 100 ms
            out_params->metacognitive.uncertainty_beta = 1.0f;
            out_params->metacognitive.history_size = 10;
            break;

        case NEURON_EXECUTIVE_CONTROL:
            out_params->executive.goal_maintenance = 0.8f;        // Strong maintenance
            out_params->executive.modulation_strength = 0.5f;     // Moderate modulation
            out_params->executive.decay_rate = 0.05f;             // Slow decay
            out_params->executive.threshold_boost = 0.2f;
            out_params->executive.delay_activity = true;
            break;

        default:
            return NIMCP_ERROR_INVALID_PARAM;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t neuron_type_validate_params(neuron_type_t type,
                                            const neuron_type_params_t* params) {
    if (!params) return NIMCP_ERROR_INVALID_PARAM;

    switch (type) {
        case NEURON_GENERIC_LIF:
            if (params->lif.tau_membrane <= 0.0f) return NIMCP_ERROR_INVALID_PARAM;
            if (params->lif.refractory_period < 0.0f) return NIMCP_ERROR_INVALID_PARAM;
            break;

        case NEURON_GENERIC_IZHIKEVICH:
            // Parameters can have wide range, minimal validation
            break;

        case NEURON_V1_SIMPLE_CELL:
            if (params->v1_simple.orientation < 0.0f || params->v1_simple.orientation > 180.0f)
                return NIMCP_ERROR_INVALID_PARAM;
            if (params->v1_simple.spatial_frequency <= 0.0f)
                return NIMCP_ERROR_INVALID_PARAM;
            if (params->v1_simple.sigma <= 0.0f)
                return NIMCP_ERROR_INVALID_PARAM;
            break;

        case NEURON_V1_COMPLEX_CELL:
            if (params->v1_complex.orientation < 0.0f || params->v1_complex.orientation > 180.0f)
                return NIMCP_ERROR_INVALID_PARAM;
            if (params->v1_complex.direction_selectivity < 0.0f || params->v1_complex.direction_selectivity > 1.0f)
                return NIMCP_ERROR_INVALID_PARAM;
            break;

        case NEURON_A1_FREQUENCY_TUNED:
            if (params->a1_frequency.center_frequency <= 0.0f || params->a1_frequency.center_frequency > 20000.0f)
                return NIMCP_ERROR_INVALID_PARAM;
            if (params->a1_frequency.q_factor <= 0.0f)
                return NIMCP_ERROR_INVALID_PARAM;
            break;

        case NEURON_A1_COINCIDENCE_DETECTOR:
            if (params->a1_coincidence.integration_window <= 0.0f)
                return NIMCP_ERROR_INVALID_PARAM;
            break;

        case NEURON_METACOGNITIVE:
            if (params->metacognitive.confidence_threshold < 0.0f || params->metacognitive.confidence_threshold > 1.0f)
                return NIMCP_ERROR_INVALID_PARAM;
            break;

        case NEURON_EXECUTIVE_CONTROL:
            if (params->executive.goal_maintenance < 0.0f || params->executive.goal_maintenance > 1.0f)
                return NIMCP_ERROR_INVALID_PARAM;
            break;

        default:
            break;
    }

    return NIMCP_SUCCESS;
}

const char* neuron_type_get_name(neuron_type_t type) {
    switch (type) {
        case NEURON_EXCITATORY:            return "Excitatory";
        case NEURON_INHIBITORY:            return "Inhibitory";
        case NEURON_GENERIC_LIF:           return "Generic LIF";
        case NEURON_GENERIC_IZHIKEVICH:    return "Generic Izhikevich";
        case NEURON_V1_SIMPLE_CELL:        return "V1 Simple Cell";
        case NEURON_V1_COMPLEX_CELL:       return "V1 Complex Cell";
        case NEURON_A1_FREQUENCY_TUNED:    return "A1 Frequency-Tuned";
        case NEURON_A1_COINCIDENCE_DETECTOR: return "A1 Coincidence Detector";
        case NEURON_METACOGNITIVE:         return "Metacognitive";
        case NEURON_EXECUTIVE_CONTROL:     return "Executive Control";
        default:                           return "Unknown";
    }
}

bool neuron_type_is_excitatory(neuron_type_t type) {
    switch (type) {
        case NEURON_EXCITATORY:
            return true;
        case NEURON_INHIBITORY:
            return false;
        default:
            // Most specialized types are excitatory
            // Can be overridden per-neuron if needed
            return true;
    }
}
