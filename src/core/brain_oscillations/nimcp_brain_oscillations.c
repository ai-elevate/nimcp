/**
 * @file nimcp_brain_oscillations.c
 * @brief Implementation of brain oscillation analysis
 *
 * WHAT: FFT-based spectral analysis of neural activity
 * WHY:  Detect brain rhythms, infer cognitive states
 * HOW:  Ring buffer for activity history + FFT + power spectrum analysis
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 2.6 (GPU P2P + Spectral Analysis)
 */

#include "core/brain_oscillations/nimcp_brain_oscillations.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/validation/nimcp_validate.h"
#include <string.h>
#include <math.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Oscillation analyzer implementation
 */
struct brain_oscillation_analyzer_struct {
    brain_t brain;                /**< Brain being analyzed */

    // Timing parameters
    uint32_t window_size_ms;      /**< Analysis window in ms */
    uint32_t sampling_rate_hz;    /**< Sampling rate in Hz */
    uint32_t buffer_size;         /**< Activity buffer size (samples) */

    // Activity tracking
    float* activity_buffer;       /**< Ring buffer for neural activity */
    uint32_t buffer_head;         /**< Current write position */
    uint32_t samples_recorded;    /**< Total samples recorded */

    // FFT resources
    fft_plan_t* fft_plan;         /**< FFT plan for spectral analysis */
    fft_complex_t* spectrum;      /**< FFT output spectrum */
    float* power_spectrum;        /**< Power spectrum */
    uint32_t spectrum_size;       /**< Number of frequency bins */

    // Cached results
    brain_wave_power_t last_wave_power;
    cognitive_state_t last_state;
    float last_confidence;
};

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Convert state to string
 */
const char* brain_oscillation_state_to_string(cognitive_state_t state)
{
    switch (state) {
        case COGNITIVE_STATE_UNKNOWN:       return "Unknown";
        case COGNITIVE_STATE_DEEP_SLEEP:    return "Deep Sleep";
        case COGNITIVE_STATE_LIGHT_SLEEP:   return "Light Sleep";
        case COGNITIVE_STATE_RELAXED:       return "Relaxed";
        case COGNITIVE_STATE_FOCUSED:       return "Focused";
        case COGNITIVE_STATE_ATTENTIVE:     return "Attentive";
        case COGNITIVE_STATE_CONSOLIDATING: return "Consolidating";
        default:                            return "Invalid";
    }
}

/**
 * @brief Get recommended window size
 */
uint32_t brain_oscillation_recommended_window(brain_wave_band_t band)
{
    switch (band) {
        case BRAIN_WAVE_DELTA:  return 3000;  // 1 Hz minimum
        case BRAIN_WAVE_THETA:  return 750;   // 4 Hz minimum
        case BRAIN_WAVE_ALPHA:  return 375;   // 8 Hz minimum
        case BRAIN_WAVE_BETA:   return 230;   // 13 Hz minimum
        case BRAIN_WAVE_GAMMA:  return 100;   // 30 Hz minimum
        default:                return 1000;
    }
}

//=============================================================================
// Analyzer Creation
//=============================================================================

/**
 * @brief Create oscillation analyzer
 */
brain_oscillation_analyzer_t* brain_oscillation_create(
    brain_t brain,
    uint32_t window_size_ms,
    uint32_t sampling_rate_hz)
{
    // Guard: Validate inputs
    if (!brain || window_size_ms == 0 || sampling_rate_hz == 0) {
        return NULL;
    }

    // Validate parameters
    if (window_size_ms < 100 || window_size_ms > 10000) {
        return NULL;  // 100ms - 10s range
    }

    if (sampling_rate_hz < 10 || sampling_rate_hz > 10000) {
        return NULL;  // 10 Hz - 10 kHz range
    }

    // Allocate analyzer
    brain_oscillation_analyzer_t* analyzer =
        (brain_oscillation_analyzer_t*)nimcp_calloc(1, sizeof(brain_oscillation_analyzer_t));
    if (!analyzer) {
        return NULL;
    }

    analyzer->brain = brain;
    analyzer->window_size_ms = window_size_ms;
    analyzer->sampling_rate_hz = sampling_rate_hz;

    // Compute buffer size (samples = window_ms * rate_hz / 1000)
    analyzer->buffer_size = (window_size_ms * sampling_rate_hz) / 1000;

    // Round up to power of 2 for FFT
    analyzer->buffer_size = fft_next_power_of_2(analyzer->buffer_size);

    // Allocate activity buffer
    analyzer->activity_buffer = (float*)nimcp_calloc(analyzer->buffer_size, sizeof(float));
    if (!analyzer->activity_buffer) {
        brain_oscillation_destroy(analyzer);
        return NULL;
    }

    // Create FFT plan
    analyzer->fft_plan = fft_plan_create(analyzer->buffer_size, FFT_REAL);
    if (!analyzer->fft_plan) {
        brain_oscillation_destroy(analyzer);
        return NULL;
    }

    // Set Hann window to reduce spectral leakage
    if (!fft_plan_set_window(analyzer->fft_plan, FFT_WINDOW_HANN)) {
        brain_oscillation_destroy(analyzer);
        return NULL;
    }

    // Allocate spectrum buffers
    analyzer->spectrum_size = analyzer->buffer_size / 2 + 1;
    analyzer->spectrum = (fft_complex_t*)nimcp_calloc(analyzer->spectrum_size, sizeof(fft_complex_t));
    analyzer->power_spectrum = (float*)nimcp_calloc(analyzer->spectrum_size, sizeof(float));

    if (!analyzer->spectrum || !analyzer->power_spectrum) {
        brain_oscillation_destroy(analyzer);
        return NULL;
    }

    // Initialize tracking
    analyzer->buffer_head = 0;
    analyzer->samples_recorded = 0;
    analyzer->last_state = COGNITIVE_STATE_UNKNOWN;
    analyzer->last_confidence = 0.0f;

    return analyzer;
}

/**
 * @brief Destroy oscillation analyzer
 */
void brain_oscillation_destroy(brain_oscillation_analyzer_t* analyzer)
{
    if (!analyzer) {
        return;
    }

    if (analyzer->activity_buffer) {
        nimcp_free(analyzer->activity_buffer);
    }

    if (analyzer->fft_plan) {
        fft_plan_destroy(analyzer->fft_plan);
    }

    if (analyzer->spectrum) {
        nimcp_free(analyzer->spectrum);
    }

    if (analyzer->power_spectrum) {
        nimcp_free(analyzer->power_spectrum);
    }

    nimcp_free(analyzer);
}

//=============================================================================
// Activity Recording
//=============================================================================

/**
 * @brief Record activity value
 */
bool brain_oscillation_record_value(
    brain_oscillation_analyzer_t* analyzer,
    float activity)
{
    // Guard: Validate inputs
    if (!analyzer) {
        return false;
    }

    // Add to ring buffer
    analyzer->activity_buffer[analyzer->buffer_head] = activity;
    analyzer->buffer_head = (analyzer->buffer_head + 1) % analyzer->buffer_size;
    analyzer->samples_recorded++;

    return true;
}

/**
 * @brief Record current brain activity
 *
 * NOTE: Placeholder implementation - would integrate with actual brain structure
 * to sample neuron activations
 */
bool brain_oscillation_record_activity(brain_oscillation_analyzer_t* analyzer)
{
    // Guard: Validate analyzer
    if (!analyzer) {
        return false;
    }

    // TODO: Integrate with actual brain structure
    // For now, record a placeholder value
    // In real implementation: compute average neuron activation
    float activity = 0.5f;  // Placeholder

    return brain_oscillation_record_value(analyzer, activity);
}

//=============================================================================
// Spectral Analysis
//=============================================================================

/**
 * @brief Get brain wave power
 */
bool brain_oscillation_get_wave_power(
    brain_oscillation_analyzer_t* analyzer,
    brain_wave_power_t* wave_power)
{
    // Guard: Validate inputs
    if (!analyzer || !wave_power) {
        return false;
    }

    // Check if buffer is full enough
    if (analyzer->samples_recorded < analyzer->buffer_size) {
        return false;  // Need full window
    }

    // Perform FFT
    if (!fft_execute_real(analyzer->fft_plan, analyzer->activity_buffer,
                          analyzer->spectrum)) {
        return false;
    }

    // Compute power spectrum
    if (!fft_power_spectrum(analyzer->spectrum, analyzer->power_spectrum,
                            analyzer->spectrum_size)) {
        return false;
    }

    // Compute sampling rate for frequency conversion
    float sampling_rate = (float)analyzer->sampling_rate_hz;

    // Compute power in each brain wave band
    wave_power->delta_power = fft_brain_wave_power(
        analyzer->power_spectrum, analyzer->spectrum_size,
        sampling_rate, BRAIN_WAVE_DELTA);

    wave_power->theta_power = fft_brain_wave_power(
        analyzer->power_spectrum, analyzer->spectrum_size,
        sampling_rate, BRAIN_WAVE_THETA);

    wave_power->alpha_power = fft_brain_wave_power(
        analyzer->power_spectrum, analyzer->spectrum_size,
        sampling_rate, BRAIN_WAVE_ALPHA);

    wave_power->beta_power = fft_brain_wave_power(
        analyzer->power_spectrum, analyzer->spectrum_size,
        sampling_rate, BRAIN_WAVE_BETA);

    wave_power->gamma_power = fft_brain_wave_power(
        analyzer->power_spectrum, analyzer->spectrum_size,
        sampling_rate, BRAIN_WAVE_GAMMA);

    // Compute total power
    wave_power->total_power = wave_power->delta_power +
                              wave_power->theta_power +
                              wave_power->alpha_power +
                              wave_power->beta_power +
                              wave_power->gamma_power;

    // Find dominant frequency
    wave_power->dominant_freq = fft_dominant_frequency(
        analyzer->power_spectrum, analyzer->spectrum_size, sampling_rate);

    // Determine dominant band
    float max_power = wave_power->delta_power;
    wave_power->dominant_band = BRAIN_WAVE_DELTA;

    if (wave_power->theta_power > max_power) {
        max_power = wave_power->theta_power;
        wave_power->dominant_band = BRAIN_WAVE_THETA;
    }
    if (wave_power->alpha_power > max_power) {
        max_power = wave_power->alpha_power;
        wave_power->dominant_band = BRAIN_WAVE_ALPHA;
    }
    if (wave_power->beta_power > max_power) {
        max_power = wave_power->beta_power;
        wave_power->dominant_band = BRAIN_WAVE_BETA;
    }
    if (wave_power->gamma_power > max_power) {
        wave_power->dominant_band = BRAIN_WAVE_GAMMA;
    }

    // Cache results
    analyzer->last_wave_power = *wave_power;

    return true;
}

/**
 * @brief Get cognitive state
 */
bool brain_oscillation_get_state(
    brain_oscillation_analyzer_t* analyzer,
    cognitive_state_t* state,
    float* confidence)
{
    // Guard: Validate inputs
    if (!analyzer || !state || !confidence) {
        return false;
    }

    // Get wave power
    brain_wave_power_t wave_power;
    if (!brain_oscillation_get_wave_power(analyzer, &wave_power)) {
        return false;
    }

    // Infer state from dominant band and power ratios
    float total = wave_power.total_power;
    if (total < 1e-6f) {
        *state = COGNITIVE_STATE_UNKNOWN;
        *confidence = 0.0f;
        return true;
    }

    // Compute power ratios
    float delta_ratio = wave_power.delta_power / total;
    float theta_ratio = wave_power.theta_power / total;
    float alpha_ratio = wave_power.alpha_power / total;
    float beta_ratio = wave_power.beta_power / total;
    float gamma_ratio = wave_power.gamma_power / total;

    // State inference heuristics
    if (delta_ratio > 0.6f) {
        *state = COGNITIVE_STATE_DEEP_SLEEP;
        *confidence = delta_ratio;
    } else if (theta_ratio > 0.4f) {
        if (delta_ratio > 0.2f) {
            *state = COGNITIVE_STATE_LIGHT_SLEEP;
        } else {
            *state = COGNITIVE_STATE_CONSOLIDATING;
        }
        *confidence = theta_ratio;
    } else if (alpha_ratio > 0.4f) {
        *state = COGNITIVE_STATE_RELAXED;
        *confidence = alpha_ratio;
    } else if (beta_ratio > 0.4f) {
        *state = COGNITIVE_STATE_FOCUSED;
        *confidence = beta_ratio;
    } else if (gamma_ratio > 0.3f) {
        *state = COGNITIVE_STATE_ATTENTIVE;
        *confidence = gamma_ratio;
    } else {
        *state = COGNITIVE_STATE_UNKNOWN;
        *confidence = 0.5f;
    }

    // Cache results
    analyzer->last_state = *state;
    analyzer->last_confidence = *confidence;

    return true;
}

/**
 * @brief Full oscillation analysis
 */
bool brain_oscillation_analyze(
    brain_oscillation_analyzer_t* analyzer,
    oscillation_analysis_t* results)
{
    // Guard: Validate inputs
    if (!analyzer || !results) {
        return false;
    }

    // Initialize results
    memset(results, 0, sizeof(oscillation_analysis_t));

    // Get wave power
    if (!brain_oscillation_get_wave_power(analyzer, &results->wave_power)) {
        return false;
    }

    // Get cognitive state
    if (!brain_oscillation_get_state(analyzer, &results->state,
                                     &results->state_confidence)) {
        return false;
    }

    // Compute spectral entropy (measure of frequency spread)
    float total_power = results->wave_power.total_power;
    if (total_power > 1e-6f) {
        results->spectral_entropy = 0.0f;
        for (uint32_t i = 0; i < analyzer->spectrum_size; i++) {
            float p = analyzer->power_spectrum[i] / total_power;
            if (p > 1e-10f) {
                results->spectral_entropy -= p * logf(p);
            }
        }
    }

    // Peak frequency
    results->peak_frequency = results->wave_power.dominant_freq;

    // Placeholder values for advanced metrics
    // TODO: Implement full PAC, coherence, synchrony computation
    results->theta_gamma_coupling = 0.0f;
    results->alpha_beta_coupling = 0.0f;
    results->synchrony = 0.5f;
    results->coherence = 0.5f;
    results->bandwidth = 5.0f;  // Placeholder

    return true;
}

//=============================================================================
// Export Functions
//=============================================================================

/**
 * @brief Get power spectrum
 */
bool brain_oscillation_get_spectrum(
    brain_oscillation_analyzer_t* analyzer,
    float** spectrum,
    uint32_t* num_bins)
{
    // Guard: Validate inputs
    if (!analyzer || !spectrum || !num_bins) {
        return false;
    }

    // Check if we have data
    if (analyzer->samples_recorded < analyzer->buffer_size) {
        return false;
    }

    *spectrum = analyzer->power_spectrum;
    *num_bins = analyzer->spectrum_size;

    return true;
}

/**
 * @brief Get activity buffer
 */
bool brain_oscillation_get_activity_buffer(
    brain_oscillation_analyzer_t* analyzer,
    const float** buffer,
    uint32_t* size)
{
    // Guard: Validate inputs
    if (!analyzer || !buffer || !size) {
        return false;
    }

    *buffer = analyzer->activity_buffer;
    *size = analyzer->buffer_size;

    return true;
}

//=============================================================================
// Advanced Analysis (Placeholder Implementations)
//=============================================================================

/**
 * @brief Compute phase-amplitude coupling
 *
 * NOTE: Placeholder - full implementation requires Hilbert transform
 */
float brain_oscillation_compute_pac(
    brain_oscillation_analyzer_t* analyzer,
    brain_wave_band_t phase_band,
    brain_wave_band_t amplitude_band)
{
    // Guard: Validate analyzer
    if (!analyzer) {
        return -1.0f;
    }

    (void)phase_band;
    (void)amplitude_band;

    // TODO: Implement full PAC using Hilbert transform
    return 0.0f;  // Placeholder
}

/**
 * @brief Compute network synchrony
 *
 * NOTE: Placeholder - full implementation requires multi-neuron tracking
 */
float brain_oscillation_compute_synchrony(brain_oscillation_analyzer_t* analyzer)
{
    // Guard: Validate analyzer
    if (!analyzer) {
        return -1.0f;
    }

    // TODO: Implement full synchrony computation
    return 0.5f;  // Placeholder
}
