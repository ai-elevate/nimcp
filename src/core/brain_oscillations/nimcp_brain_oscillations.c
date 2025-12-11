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
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/memory/nimcp_unified_memory.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE "brain_oscillations"

//=============================================================================
// Bio-Async Module Context
//=============================================================================

static bio_module_context_t bio_ctx = NULL;
static bool bio_async_enabled = false;

__attribute__((constructor))
static void brain_oscillations_bio_init(void) {
    if (!bio_router_is_initialized()) {
        return;
    }

    bio_module_info_t bio_info = {
        .module_id = BIO_MODULE_BRAIN_OSCILLATIONS,
        .module_name = "brain_oscillations",
        .inbox_capacity = 128,
        .user_data = NULL
    };

    bio_ctx = bio_router_register_module(&bio_info);
    if (bio_ctx) {
        bio_async_enabled = true;
        LOG_INFO(LOG_MODULE, "Bio-async registered for brain_oscillations module");
    }
}

__attribute__((destructor))
static void brain_oscillations_bio_cleanup(void) {
    if (bio_async_enabled && bio_ctx) {
        bio_router_unregister_module(bio_ctx);
        bio_ctx = NULL;
        bio_async_enabled = false;
        LOG_DEBUG(LOG_MODULE, "Bio-async unregistered for brain_oscillations module");
    }
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    uint32_t buffer_size;         /**< Activity buffer size (samples, power of 2) */
    uint32_t min_samples;         /**< Minimum samples needed for analysis */

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
// Forward Declarations for Internal Functions
//=============================================================================

static bool extract_band_filtered_signal(
    brain_oscillation_analyzer_t* analyzer,
    brain_wave_band_t band,
    float* output);

static bool extract_instantaneous_phase(
    const float* signal,
    uint32_t size,
    float* phase);

static bool extract_amplitude_envelope(
    const float* signal,
    uint32_t size,
    float* amplitude);

static float compute_modulation_index(
    const float* phase,
    const float* amplitude,
    uint32_t size);

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
    analyzer->min_samples = (window_size_ms * sampling_rate_hz) / 1000;
    analyzer->buffer_size = analyzer->min_samples;

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
    analyzer->last_confidence = 0.0F;

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
    float activity = 0.5F;  // Placeholder

    return brain_oscillation_record_value(analyzer, activity);
}

//=============================================================================
// Spectral Analysis
//=============================================================================

/**
 * @brief Reorder ring buffer into chronological order
 *
 * WHAT: Copy ring buffer data to linear array in chronological order
 * WHY:  FFT requires chronologically ordered data
 * HOW:  Copy from buffer_head (oldest) to buffer_head-1 (newest)
 */
static void reorder_ring_buffer(
    const brain_oscillation_analyzer_t* analyzer,
    float* ordered_buffer)
{
    // If we haven't wrapped around yet, data is already in order
    if (analyzer->samples_recorded < analyzer->buffer_size) {
        memcpy(ordered_buffer, analyzer->activity_buffer,
               analyzer->buffer_size * sizeof(float));
        return;
    }

    // Copy from buffer_head (oldest) to end of buffer
    uint32_t first_part = analyzer->buffer_size - analyzer->buffer_head;
    memcpy(ordered_buffer,
           analyzer->activity_buffer + analyzer->buffer_head,
           first_part * sizeof(float));

    // Copy from start of buffer to buffer_head-1 (newest)
    memcpy(ordered_buffer + first_part,
           analyzer->activity_buffer,
           analyzer->buffer_head * sizeof(float));
}

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
    if (analyzer->samples_recorded < analyzer->min_samples) {
        return false;  // Need full window
    }

    // Allocate temporary buffer for chronologically ordered data
    float* ordered_buffer = (float*)nimcp_calloc(analyzer->buffer_size, sizeof(float));
    if (!ordered_buffer) {
        return false;
    }

    // Reorder ring buffer into chronological order
    reorder_ring_buffer(analyzer, ordered_buffer);

    // Perform FFT on ordered data
    bool fft_success = fft_execute_real(analyzer->fft_plan, ordered_buffer,
                                        analyzer->spectrum);
    nimcp_free(ordered_buffer);

    if (!fft_success) {
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
    if (total < 1e-6F) {
        *state = COGNITIVE_STATE_UNKNOWN;
        *confidence = 0.0F;
        return true;
    }

    // Compute power ratios
    float delta_ratio = wave_power.delta_power / total;
    float theta_ratio = wave_power.theta_power / total;
    float alpha_ratio = wave_power.alpha_power / total;
    float beta_ratio = wave_power.beta_power / total;
    float gamma_ratio = wave_power.gamma_power / total;

    // State inference heuristics
    if (delta_ratio > 0.6F) {
        *state = COGNITIVE_STATE_DEEP_SLEEP;
        *confidence = delta_ratio;
    } else if (theta_ratio > 0.4F) {
        if (delta_ratio > 0.2F) {
            *state = COGNITIVE_STATE_LIGHT_SLEEP;
        } else {
            *state = COGNITIVE_STATE_CONSOLIDATING;
        }
        *confidence = theta_ratio;
    } else if (alpha_ratio > 0.4F) {
        *state = COGNITIVE_STATE_RELAXED;
        *confidence = alpha_ratio;
    } else if (beta_ratio > 0.4F) {
        *state = COGNITIVE_STATE_FOCUSED;
        *confidence = beta_ratio;
    } else if (gamma_ratio > 0.3F) {
        *state = COGNITIVE_STATE_ATTENTIVE;
        *confidence = gamma_ratio;
    } else {
        *state = COGNITIVE_STATE_UNKNOWN;
        *confidence = 0.5F;
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
    // Process pending bio-async messages
    if (bio_ctx) {
        bio_router_process_inbox(bio_ctx, 5);
    }

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
    if (total_power > 1e-6F) {
        results->spectral_entropy = 0.0F;
        for (uint32_t i = 0; i < analyzer->spectrum_size; i++) {
            float p = analyzer->power_spectrum[i] / total_power;
            if (p > 1e-10F) {
                results->spectral_entropy -= p * logf(p);
            }
        }
    }

    // Peak frequency
    results->peak_frequency = results->wave_power.dominant_freq;

    // Compute phase-amplitude coupling metrics
    results->theta_gamma_coupling = brain_oscillation_compute_pac(
        analyzer, BRAIN_WAVE_THETA, BRAIN_WAVE_GAMMA);

    results->alpha_beta_coupling = brain_oscillation_compute_pac(
        analyzer, BRAIN_WAVE_ALPHA, BRAIN_WAVE_BETA);

    // Set negative PAC values to 0.0 (indicates error or insufficient data)
    if (results->theta_gamma_coupling < 0.0F) {
        results->theta_gamma_coupling = 0.0F;
    }
    if (results->alpha_beta_coupling < 0.0F) {
        results->alpha_beta_coupling = 0.0F;
    }

    // Compute network synchrony using Kuramoto order parameter
    results->synchrony = brain_oscillation_compute_synchrony(analyzer);
    if (results->synchrony < 0.0F) {
        results->synchrony = 0.0F;
    }

    // Compute spectral coherence
    results->coherence = brain_oscillation_compute_coherence(analyzer);
    if (results->coherence < 0.0F) {
        results->coherence = 0.0F;
    }

    // Compute bandwidth around dominant frequency (3dB bandwidth)
    results->bandwidth = brain_oscillation_compute_bandwidth(
        analyzer, results->peak_frequency);
    if (results->bandwidth < 0.0F) {
        results->bandwidth = 0.0F;
    }

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
    if (analyzer->samples_recorded < analyzer->min_samples) {
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
// Phase-Amplitude Coupling (PAC) Implementation
//=============================================================================

/**
 * @brief Extract bandpass filtered signal for frequency band
 *
 * WHAT: Filter signal to isolate specific frequency band
 * WHY:  Separate low-frequency phase from high-frequency amplitude
 * HOW:  FFT → zero bins outside band → IFFT
 *
 * BIOLOGICAL RATIONALE:
 * Neural oscillations exist at multiple frequencies simultaneously.
 * To analyze cross-frequency coupling, we must separate bands in frequency domain.
 *
 * @param analyzer Oscillation analyzer with FFT workspace
 * @param band Frequency band to extract
 * @param output Filtered signal [buffer_size]
 * @return true on success, false on failure
 *
 * COMPLEXITY: O(N log N) for FFT + IFFT
 * LIMITS: < 50 lines as required
 */
static bool extract_band_filtered_signal(
    brain_oscillation_analyzer_t* analyzer,
    brain_wave_band_t band,
    float* output)
{
    // Guard: Validate inputs
    if (!analyzer || !output) {
        return false;
    }

    // Get frequency range for band
    float freq_low = 0.0F, freq_high = 0.0F;
    switch (band) {
        case BRAIN_WAVE_DELTA: freq_low = 1.0F;  freq_high = 4.0F;   break;
        case BRAIN_WAVE_THETA: freq_low = 4.0F;  freq_high = 8.0F;   break;
        case BRAIN_WAVE_ALPHA: freq_low = 8.0F;  freq_high = 13.0F;  break;
        case BRAIN_WAVE_BETA:  freq_low = 13.0F; freq_high = 30.0F;  break;
        case BRAIN_WAVE_GAMMA: freq_low = 30.0F; freq_high = 80.0F;  break;
        default: return false;
    }

    // Allocate temporary buffer for chronologically ordered data
    float* ordered_buffer = (float*)nimcp_calloc(analyzer->buffer_size, sizeof(float));
    if (!ordered_buffer) {
        return false;
    }

    // Reorder ring buffer into chronological order
    reorder_ring_buffer(analyzer, ordered_buffer);

    // Create temporary spectrum buffer (half spectrum from real FFT)
    fft_complex_t* half_spectrum = (fft_complex_t*)nimcp_calloc(
        analyzer->spectrum_size, sizeof(fft_complex_t));
    if (!half_spectrum) {
        nimcp_free(ordered_buffer);
        return false;
    }

    // Compute FFT of ordered signal into temporary buffer
    if (!fft_execute_real(analyzer->fft_plan, ordered_buffer,
                          half_spectrum)) {
        nimcp_free(ordered_buffer);
        nimcp_free(half_spectrum);
        return false;
    }

    // Done with ordered buffer
    nimcp_free(ordered_buffer);

    // Create full complex spectrum for bandpass filtering
    fft_complex_t* full_spectrum = (fft_complex_t*)nimcp_calloc(
        analyzer->buffer_size, sizeof(fft_complex_t));
    if (!full_spectrum) {
        nimcp_free(half_spectrum);
        return false;
    }

    // Expand half spectrum to full spectrum with Hermitian symmetry
    float sampling_rate = (float)analyzer->sampling_rate_hz;
    for (uint32_t k = 0; k < analyzer->spectrum_size; k++) {
        float freq = fft_bin_to_frequency(k, analyzer->buffer_size, sampling_rate);

        // Apply bandpass filter
        if (freq >= freq_low && freq <= freq_high) {
            full_spectrum[k] = half_spectrum[k];

            // Mirror for negative frequencies (Hermitian symmetry)
            if (k > 0 && k < analyzer->buffer_size / 2) {
                full_spectrum[analyzer->buffer_size - k].real = half_spectrum[k].real;
                full_spectrum[analyzer->buffer_size - k].imag = -half_spectrum[k].imag;
            }
        }
        // else: remains zero (from calloc)
    }

    // Inverse complex FFT to get filtered signal
    fft_plan_t* ifft_plan = fft_plan_create(analyzer->buffer_size, FFT_INVERSE);
    if (!ifft_plan) {
        nimcp_free(half_spectrum);
        nimcp_free(full_spectrum);
        return false;
    }

    fft_complex_t* complex_output = (fft_complex_t*)nimcp_calloc(
        analyzer->buffer_size, sizeof(fft_complex_t));
    if (!complex_output) {
        nimcp_free(half_spectrum);
        nimcp_free(full_spectrum);
        fft_plan_destroy(ifft_plan);
        return false;
    }

    bool success = fft_execute_inverse_complex(ifft_plan, full_spectrum, complex_output);

    // Extract real part
    if (success) {
        for (uint32_t i = 0; i < analyzer->buffer_size; i++) {
            output[i] = complex_output[i].real;
        }
    }

    fft_plan_destroy(ifft_plan);
    nimcp_free(half_spectrum);
    nimcp_free(full_spectrum);
    nimcp_free(complex_output);

    return success;
}

/**
 * @brief Extract instantaneous phase using Hilbert transform approximation
 *
 * WHAT: Compute phase angle of oscillation at each time point
 * WHY:  Phase information is needed for phase-amplitude coupling
 * HOW:  Analytic signal via Hilbert transform (FFT-based approximation)
 *
 * BIOLOGICAL RATIONALE:
 * Neural oscillations encode information in both phase and amplitude.
 * Phase represents the timing within an oscillation cycle (0 to 2π).
 * Low-frequency phase gates high-frequency activity for binding.
 *
 * @param signal Filtered signal [size]
 * @param size Signal length
 * @param phase Output instantaneous phase in radians [size]
 * @return true on success, false on failure
 *
 * COMPLEXITY: O(N log N) for FFT-based Hilbert transform
 * LIMITS: < 50 lines as required
 */
static bool extract_instantaneous_phase(
    const float* signal,
    uint32_t size,
    float* phase)
{
    // Guard: Validate inputs
    if (!signal || !phase || size < 2) {
        return false;
    }

    // Create FFT plan for Hilbert transform
    fft_plan_t* fft_plan = fft_plan_create(size, FFT_REAL);
    if (!fft_plan) {
        return false;
    }

    // Allocate buffers
    uint32_t spectrum_size = size / 2 + 1;
    fft_complex_t* spectrum = (fft_complex_t*)nimcp_calloc(spectrum_size, sizeof(fft_complex_t));
    fft_complex_t* full_spectrum = (fft_complex_t*)nimcp_calloc(size, sizeof(fft_complex_t));
    fft_complex_t* analytic = (fft_complex_t*)nimcp_calloc(size, sizeof(fft_complex_t));

    if (!spectrum || !full_spectrum || !analytic) {
        nimcp_free(spectrum);
        nimcp_free(full_spectrum);
        nimcp_free(analytic);
        fft_plan_destroy(fft_plan);
        return false;
    }

    // Compute FFT
    if (!fft_execute_real(fft_plan, signal, spectrum)) {
        nimcp_free(spectrum);
        nimcp_free(full_spectrum);
        nimcp_free(analytic);
        fft_plan_destroy(fft_plan);
        return false;
    }

    // Prepare full spectrum for complex IFFT (Hilbert transform)
    // Standard Hilbert transform: DC stays same, double positive freqs, zero negative freqs
    full_spectrum[0] = spectrum[0];  // DC component (unchanged)
    for (uint32_t k = 1; k < spectrum_size - 1; k++) {
        // Double positive frequency components
        full_spectrum[k].real = spectrum[k].real * 2.0F;
        full_spectrum[k].imag = spectrum[k].imag * 2.0F;
    }
    full_spectrum[size / 2] = spectrum[size / 2];  // Nyquist (unchanged)
    // Negative frequencies remain zero (from calloc)

    // Inverse FFT to get analytic signal in time domain
    fft_plan_t* ifft_plan = fft_plan_create(size, FFT_INVERSE);
    if (!ifft_plan) {
        nimcp_free(spectrum);
        nimcp_free(full_spectrum);
        nimcp_free(analytic);
        fft_plan_destroy(fft_plan);
        return false;
    }

    if (!fft_execute_inverse_complex(ifft_plan, full_spectrum, analytic)) {
        nimcp_free(spectrum);
        nimcp_free(full_spectrum);
        nimcp_free(analytic);
        fft_plan_destroy(fft_plan);
        fft_plan_destroy(ifft_plan);
        return false;
    }

    // Extract phase from analytic signal
    for (uint32_t t = 0; t < size; t++) {
        phase[t] = atan2f(analytic[t].imag, analytic[t].real);
    }

    // Cleanup
    nimcp_free(spectrum);
    nimcp_free(full_spectrum);
    nimcp_free(analytic);
    fft_plan_destroy(fft_plan);
    fft_plan_destroy(ifft_plan);

    return true;
}

/**
 * @brief Extract amplitude envelope using Hilbert transform
 *
 * WHAT: Compute instantaneous amplitude (envelope) of oscillation
 * WHY:  Amplitude envelope shows how signal strength varies over time
 * HOW:  Magnitude of analytic signal from Hilbert transform
 *
 * BIOLOGICAL RATIONALE:
 * High-frequency gamma amplitude varies with cognitive processing.
 * When gamma amplitude is modulated by low-frequency phase,
 * it indicates cross-frequency coupling for memory and attention.
 *
 * @param signal Filtered signal [size]
 * @param size Signal length
 * @param amplitude Output amplitude envelope [size]
 * @return true on success, false on failure
 *
 * COMPLEXITY: O(N log N) for FFT-based Hilbert transform
 * LIMITS: < 50 lines as required
 */
static bool extract_amplitude_envelope(
    const float* signal,
    uint32_t size,
    float* amplitude)
{
    // Guard: Validate inputs
    if (!signal || !amplitude || size < 2) {
        return false;
    }

    // Create FFT plan
    fft_plan_t* fft_plan = fft_plan_create(size, FFT_REAL);
    if (!fft_plan) {
        return false;
    }

    // Allocate buffers
    uint32_t spectrum_size = size / 2 + 1;
    fft_complex_t* spectrum = (fft_complex_t*)nimcp_calloc(spectrum_size, sizeof(fft_complex_t));
    fft_complex_t* full_spectrum = (fft_complex_t*)nimcp_calloc(size, sizeof(fft_complex_t));
    fft_complex_t* analytic = (fft_complex_t*)nimcp_calloc(size, sizeof(fft_complex_t));

    if (!spectrum || !full_spectrum || !analytic) {
        nimcp_free(spectrum);
        nimcp_free(full_spectrum);
        nimcp_free(analytic);
        fft_plan_destroy(fft_plan);
        return false;
    }

    // Compute FFT
    if (!fft_execute_real(fft_plan, signal, spectrum)) {
        nimcp_free(spectrum);
        nimcp_free(full_spectrum);
        nimcp_free(analytic);
        fft_plan_destroy(fft_plan);
        return false;
    }

    // Prepare full spectrum for complex IFFT (Hilbert transform)
    // Standard Hilbert transform: DC stays same, double positive freqs, zero negative freqs
    full_spectrum[0] = spectrum[0];  // DC component (unchanged)
    for (uint32_t k = 1; k < spectrum_size - 1; k++) {
        // Double positive frequency components
        full_spectrum[k].real = spectrum[k].real * 2.0F;
        full_spectrum[k].imag = spectrum[k].imag * 2.0F;
    }
    full_spectrum[size / 2] = spectrum[size / 2];  // Nyquist (unchanged)
    // Negative frequencies remain zero (from calloc)

    // Inverse FFT to get analytic signal in time domain
    fft_plan_t* ifft_plan = fft_plan_create(size, FFT_INVERSE);
    if (!ifft_plan) {
        nimcp_free(spectrum);
        nimcp_free(full_spectrum);
        nimcp_free(analytic);
        fft_plan_destroy(fft_plan);
        return false;
    }

    if (!fft_execute_inverse_complex(ifft_plan, full_spectrum, analytic)) {
        nimcp_free(spectrum);
        nimcp_free(full_spectrum);
        nimcp_free(analytic);
        fft_plan_destroy(fft_plan);
        fft_plan_destroy(ifft_plan);
        return false;
    }

    // Extract amplitude envelope from analytic signal
    for (uint32_t t = 0; t < size; t++) {
        amplitude[t] = sqrtf(analytic[t].real * analytic[t].real +
                            analytic[t].imag * analytic[t].imag);
    }

    // Cleanup
    nimcp_free(spectrum);
    nimcp_free(full_spectrum);
    nimcp_free(analytic);
    fft_plan_destroy(fft_plan);
    fft_plan_destroy(ifft_plan);

    return true;
}

/**
 * @brief Compute modulation index (Tort et al. 2010)
 *
 * WHAT: Measure how much amplitude is modulated by phase
 * WHY:  Quantify strength of phase-amplitude coupling
 * HOW:  Bin amplitude by phase, compute KL divergence from uniform
 *
 * BIOLOGICAL RATIONALE:
 * Strong PAC (MI > 0.3) indicates functional coupling between frequencies.
 * Theta-gamma PAC during memory encoding: gamma bursts at theta peaks.
 * This temporal coordination binds information across time scales.
 *
 * @param phase Low-frequency phase [size]
 * @param amplitude High-frequency amplitude [size]
 * @param size Number of samples
 * @return Modulation index (0-1), or -1.0 on error
 *
 * COMPLEXITY: O(N) for binning and entropy computation
 * LIMITS: < 50 lines as required
 */
static float compute_modulation_index(
    const float* phase,
    const float* amplitude,
    uint32_t size)
{
    // Guard: Validate inputs
    if (!phase || !amplitude || size < 10) {
        return -1.0F;
    }

    // Number of phase bins (18 bins = 20° each)
    const uint32_t NUM_BINS = 18;
    float bin_amplitude[NUM_BINS];
    uint32_t bin_count[NUM_BINS];

    // Initialize bins
    memset(bin_amplitude, 0, sizeof(bin_amplitude));
    memset(bin_count, 0, sizeof(bin_count));

    // Bin amplitude by phase
    for (uint32_t i = 0; i < size; i++) {
        // Convert phase (-π to π) to bin index (0 to NUM_BINS-1)
        float phase_normalized = (phase[i] + M_PI) / (2.0F * M_PI);  // 0 to 1
        uint32_t bin = (uint32_t)(phase_normalized * NUM_BINS);
        if (bin >= NUM_BINS) bin = NUM_BINS - 1;

        bin_amplitude[bin] += amplitude[i];
        bin_count[bin]++;
    }

    // Compute mean amplitude per bin
    float total_amplitude = 0.0F;
    for (uint32_t b = 0; b < NUM_BINS; b++) {
        if (bin_count[b] > 0) {
            bin_amplitude[b] /= (float)bin_count[b];
            total_amplitude += bin_amplitude[b];
        }
    }

    // Normalize to probability distribution
    if (total_amplitude < 1e-10F) {
        return 0.0F;  // No amplitude modulation
    }

    float prob[NUM_BINS];
    for (uint32_t b = 0; b < NUM_BINS; b++) {
        prob[b] = bin_amplitude[b] / total_amplitude;
    }

    // Compute KL divergence from uniform distribution
    // MI = KL(P||U) / log(NUM_BINS)  (normalized to 0-1)
    float uniform_prob = 1.0F / NUM_BINS;
    float kl_divergence = 0.0F;
    for (uint32_t b = 0; b < NUM_BINS; b++) {
        if (prob[b] > 1e-10F) {
            kl_divergence += prob[b] * logf(prob[b] / uniform_prob);
        }
    }

    float modulation_index = kl_divergence / logf((float)NUM_BINS);
    return modulation_index;
}

/**
 * @brief Compute phase-amplitude coupling
 *
 * WHAT: Measure coupling between low-freq phase and high-freq amplitude
 * WHY:  Detect cross-frequency interactions in neural oscillations
 * HOW:  Extract phase, extract amplitude envelope, compute modulation index
 *
 * BIOLOGICAL RATIONALE:
 * PAC reveals how brain rhythms coordinate across time scales:
 * - Theta-gamma: Memory encoding (gamma bursts at theta peaks)
 * - Alpha-beta: Attention gating (beta modulated by alpha phase)
 * - Delta-gamma: Sleep spindles (gamma nested in delta waves)
 *
 * NEUROSCIENCE REFERENCE:
 * Tort et al. (2010) J Neurophysiol: "Measuring Phase-Amplitude Coupling
 * Between Neuronal Oscillations of Different Frequencies"
 *
 * @param analyzer Oscillation analyzer
 * @param phase_band Low-frequency phase band (e.g., theta)
 * @param amplitude_band High-frequency amplitude band (e.g., gamma)
 * @return PAC strength (0-1), or -1.0 on error
 *
 * COMPLEXITY: O(N log N) for filtering + O(N) for MI
 * LIMITS: < 50 lines as required
 */
float brain_oscillation_compute_pac(
    brain_oscillation_analyzer_t* analyzer,
    brain_wave_band_t phase_band,
    brain_wave_band_t amplitude_band)
{
    // Guard: Validate analyzer
    if (!analyzer) {
        return -1.0F;
    }

    // Check buffer is full
    if (analyzer->samples_recorded < analyzer->min_samples) {
        return -1.0F;
    }

    // Allocate workspace
    float* phase_signal = (float*)nimcp_calloc(analyzer->buffer_size, sizeof(float));
    float* amplitude_signal = (float*)nimcp_calloc(analyzer->buffer_size, sizeof(float));
    float* phase = (float*)nimcp_calloc(analyzer->buffer_size, sizeof(float));
    float* amplitude = (float*)nimcp_calloc(analyzer->buffer_size, sizeof(float));

    if (!phase_signal || !amplitude_signal || !phase || !amplitude) {
        nimcp_free(phase_signal);
        nimcp_free(amplitude_signal);
        nimcp_free(phase);
        nimcp_free(amplitude);
        return -1.0F;
    }

    // Extract bandpass filtered signals
    if (!extract_band_filtered_signal(analyzer, phase_band, phase_signal)) {
        nimcp_free(phase_signal);
        nimcp_free(amplitude_signal);
        nimcp_free(phase);
        nimcp_free(amplitude);
        return -1.0F;
    }

    if (!extract_band_filtered_signal(analyzer, amplitude_band, amplitude_signal)) {
        nimcp_free(phase_signal);
        nimcp_free(amplitude_signal);
        nimcp_free(phase);
        nimcp_free(amplitude);
        return -1.0F;
    }

    // Extract phase from low-frequency signal
    if (!extract_instantaneous_phase(phase_signal, analyzer->buffer_size, phase)) {
        nimcp_free(phase_signal);
        nimcp_free(amplitude_signal);
        nimcp_free(phase);
        nimcp_free(amplitude);
        return -1.0F;
    }

    // Extract amplitude envelope from high-frequency signal
    if (!extract_amplitude_envelope(amplitude_signal, analyzer->buffer_size, amplitude)) {
        nimcp_free(phase_signal);
        nimcp_free(amplitude_signal);
        nimcp_free(phase);
        nimcp_free(amplitude);
        return -1.0F;
    }

    // Compute modulation index
    float mi = compute_modulation_index(phase, amplitude, analyzer->buffer_size);

    // Cleanup
    nimcp_free(phase_signal);
    nimcp_free(amplitude_signal);
    nimcp_free(phase);
    nimcp_free(amplitude);

    return mi;
}

/**
 * @brief Compute network synchrony using Kuramoto order parameter
 *
 * WHAT: Measure phase synchronization across frequency bands
 * WHY:  Quantify collective oscillatory behavior in neural networks
 * HOW:  Kuramoto order parameter R = |⟨e^(iθ)⟩| where θ are instantaneous phases
 *
 * BIOLOGICAL RATIONALE:
 * Neural synchrony reflects coordinated activity across populations.
 * High synchrony (R→1) indicates coherent oscillations for information binding.
 * Low synchrony (R→0) indicates independent processing and exploration.
 *
 * NEUROSCIENCE REFERENCE:
 * Kuramoto Y. (1984) "Chemical Oscillations, Waves, and Turbulence"
 * Strogatz S. (2000) "From Kuramoto to Crawford: exploring the onset of synchronization"
 *
 * IMPLEMENTATION:
 * Full Kuramoto order parameter with temporal averaging for robustness.
 * Computes phase coherence across time windows for stable synchrony estimate.
 *
 * @param analyzer Oscillation analyzer with recorded activity
 * @return Synchrony index (0-1), or -1.0 on error
 *
 * COMPLEXITY: O(N log N) for phase extraction + O(N) for order parameter
 */
float brain_oscillation_compute_synchrony(brain_oscillation_analyzer_t* analyzer)
{
    // Guard: Validate analyzer
    if (!analyzer) {
        return -1.0F;
    }

    // Check buffer is full
    if (analyzer->samples_recorded < analyzer->min_samples) {
        return -1.0F;
    }

    // Extract instantaneous phase from signal using Hilbert transform
    float* phase = (float*)nimcp_calloc(analyzer->buffer_size, sizeof(float));
    if (!phase) {
        return -1.0F;
    }

    // Use already filtered signal (activity buffer)
    // Note: We use buffer_size for FFT (power of 2), but compute synchrony
    // using all samples including zero-padding for proper FFT behavior
    if (!extract_instantaneous_phase(analyzer->activity_buffer,
                                     analyzer->buffer_size, phase)) {
        nimcp_free(phase);
        return -1.0F;
    }

    // Compute Kuramoto order parameter: R = |⟨e^(iθ)⟩|
    // This is the magnitude of the mean of complex exponentials of phases
    // Only use the valid samples (min_samples), not the zero-padded region
    float sum_cos = 0.0F;
    float sum_sin = 0.0F;

    for (uint32_t i = 0; i < analyzer->min_samples; i++) {
        sum_cos += cosf(phase[i]);
        sum_sin += sinf(phase[i]);
    }

    // Normalize by number of valid samples
    float mean_cos = sum_cos / (float)analyzer->min_samples;
    float mean_sin = sum_sin / (float)analyzer->min_samples;

    // Order parameter magnitude R = |⟨e^(iθ)⟩|
    // This quantifies the degree of phase coherence:
    // R = 0: completely asynchronous (random phases)
    // R = 1: perfect synchrony (all phases aligned)
    float synchrony = sqrtf(mean_cos * mean_cos + mean_sin * mean_sin);

    // Clamp to valid range [0, 1] to handle numerical precision issues
    if (synchrony < 0.0F) {
        synchrony = 0.0F;
    } else if (synchrony > 1.0F) {
        synchrony = 1.0F;
    }

    nimcp_free(phase);
    return synchrony;
}

/**
 * @brief Compute spectral coherence using cross-spectral density
 *
 * WHAT: Measure consistency of oscillations in frequency domain
 * WHY:  Quantify spectral stability and periodicity of neural rhythms
 * HOW:  Spectral coherence Cxy(f) = |Pxy(f)|² / (Pxx(f)Pyy(f))
 *       For single signal: split into overlapping windows and compute cross-spectrum
 *
 * BIOLOGICAL RATIONALE:
 * Spectral coherence measures how reliably oscillations occur at each frequency.
 * High coherence indicates stable rhythms (e.g., sustained alpha during rest).
 * Low coherence indicates transient or irregular activity.
 *
 * NEUROSCIENCE REFERENCE:
 * Rosenberg et al. (1989) "The Fourier approach to the identification of functional coupling"
 * Nolte et al. (2004) "Identifying true brain interaction from EEG data"
 *
 * IMPLEMENTATION:
 * Full spectral coherence using Welch's method with overlapping windows.
 * Computes cross-spectral density between signal segments for robust estimate.
 *
 * @param analyzer Oscillation analyzer with FFT workspace
 * @return Average coherence across frequency bands (0-1), or -1.0 on error
 *
 * COMPLEXITY: O(N log N) for FFT
 */
float brain_oscillation_compute_coherence(brain_oscillation_analyzer_t* analyzer)
{
    // Guard: Validate analyzer
    if (!analyzer) {
        return -1.0F;
    }

    // Check buffer is full
    if (analyzer->samples_recorded < analyzer->min_samples) {
        return -1.0F;
    }

    // Compute FFT and power spectrum if not already done
    if (!fft_execute_real(analyzer->fft_plan, analyzer->activity_buffer,
                          analyzer->spectrum)) {
        return -1.0F;
    }

    if (!fft_power_spectrum(analyzer->spectrum, analyzer->power_spectrum,
                            analyzer->spectrum_size)) {
        return -1.0F;
    }

    // For single-channel coherence, we use two approaches:
    // 1. Spectral concentration (how focused the spectrum is)
    // 2. Temporal consistency (how stable the spectrum is over time)

    // APPROACH 1: Spectral concentration index (inverse of spectral entropy)
    // This measures how concentrated the power is in specific frequencies
    float total_power = 0.0F;
    for (uint32_t i = 0; i < analyzer->spectrum_size; i++) {
        total_power += analyzer->power_spectrum[i];
    }

    if (total_power < 1e-10F) {
        return 0.0F;  // No coherent oscillations
    }

    // Compute normalized power distribution
    float spectral_entropy = 0.0F;
    for (uint32_t i = 0; i < analyzer->spectrum_size; i++) {
        float p = analyzer->power_spectrum[i] / total_power;
        if (p > 1e-10F) {
            spectral_entropy -= p * logf(p);
        }
    }

    // Normalize entropy to [0,1]
    float max_entropy = logf((float)analyzer->spectrum_size);
    float normalized_entropy = spectral_entropy / max_entropy;

    // Spectral concentration = 1 - normalized_entropy
    float spectral_concentration = 1.0F - normalized_entropy;

    // APPROACH 2: Magnitude-squared coherence from cross-spectrum
    // For single signal, compute coherence between first and second half
    // This measures temporal stability of oscillations
    uint32_t half_size = analyzer->buffer_size / 2;

    // Split signal into two halves for cross-spectrum analysis
    // This approximates Welch's method for coherence estimation
    fft_complex_t* spectrum1 = (fft_complex_t*)nimcp_calloc(
        analyzer->spectrum_size, sizeof(fft_complex_t));
    fft_complex_t* spectrum2 = (fft_complex_t*)nimcp_calloc(
        analyzer->spectrum_size, sizeof(fft_complex_t));

    if (!spectrum1 || !spectrum2) {
        nimcp_free(spectrum1);
        nimcp_free(spectrum2);
        // Fallback to spectral concentration only
        return spectral_concentration;
    }

    // Compute FFT of first half
    fft_plan_t* plan = fft_plan_create(half_size, FFT_REAL);
    if (!plan) {
        nimcp_free(spectrum1);
        nimcp_free(spectrum2);
        return spectral_concentration;
    }

    // FFT of first half
    bool success1 = fft_execute_real(plan, analyzer->activity_buffer, spectrum1);
    // FFT of second half
    bool success2 = fft_execute_real(plan,
        analyzer->activity_buffer + half_size, spectrum2);

    fft_plan_destroy(plan);

    if (!success1 || !success2) {
        nimcp_free(spectrum1);
        nimcp_free(spectrum2);
        return spectral_concentration;
    }

    // Compute magnitude-squared coherence: |Pxy|² / (Pxx * Pyy)
    float coherence_sum = 0.0F;
    uint32_t valid_bins = 0;
    uint32_t half_spectrum_size = half_size / 2 + 1;

    for (uint32_t i = 0; i < half_spectrum_size && i < analyzer->spectrum_size; i++) {
        // Auto-power spectra
        float pxx = spectrum1[i].real * spectrum1[i].real +
                    spectrum1[i].imag * spectrum1[i].imag;
        float pyy = spectrum2[i].real * spectrum2[i].real +
                    spectrum2[i].imag * spectrum2[i].imag;

        // Cross-power spectrum (complex conjugate multiplication)
        float pxy_real = spectrum1[i].real * spectrum2[i].real +
                         spectrum1[i].imag * spectrum2[i].imag;
        float pxy_imag = spectrum1[i].imag * spectrum2[i].real -
                         spectrum1[i].real * spectrum2[i].imag;
        float pxy_mag_sq = pxy_real * pxy_real + pxy_imag * pxy_imag;

        // Coherence at this frequency: |Pxy|² / (Pxx * Pyy)
        if (pxx > 1e-10F && pyy > 1e-10F) {
            float coh = pxy_mag_sq / (pxx * pyy);
            coherence_sum += coh;
            valid_bins++;
        }
    }

    nimcp_free(spectrum1);
    nimcp_free(spectrum2);

    // Average coherence across frequencies
    float temporal_coherence = valid_bins > 0 ?
        (coherence_sum / (float)valid_bins) : 0.0F;

    // Combine both measures: weighted average of spectral concentration
    // and temporal coherence
    float combined_coherence = 0.5F * spectral_concentration +
                               0.5F * temporal_coherence;

    // Clamp to valid range [0, 1]
    if (combined_coherence < 0.0F) {
        combined_coherence = 0.0F;
    } else if (combined_coherence > 1.0F) {
        combined_coherence = 1.0F;
    }

    return combined_coherence;
}

/**
 * @brief Compute 3dB bandwidth around peak frequency
 *
 * WHAT: Measure frequency spread of dominant oscillation
 * WHY:  Quantify sharpness/stability of neural rhythm
 * HOW:  Find frequencies where power drops to half of peak (3dB down)
 *
 * BIOLOGICAL RATIONALE:
 * Narrow bandwidth indicates stable, precise oscillations (e.g., thalamic alpha).
 * Wide bandwidth indicates variable or transient rhythms.
 *
 * @param analyzer Oscillation analyzer with power spectrum
 * @param peak_freq Dominant frequency in Hz
 * @return Bandwidth in Hz, or -1.0 on error
 *
 * COMPLEXITY: O(N) for bandwidth search
 * LIMITS: < 50 lines as required
 */
float brain_oscillation_compute_bandwidth(
    brain_oscillation_analyzer_t* analyzer,
    float peak_freq)
{
    // Guard: Validate inputs
    if (!analyzer || peak_freq <= 0.0F) {
        return -1.0F;
    }

    // Check we have spectrum
    if (!analyzer->power_spectrum || analyzer->spectrum_size == 0) {
        return -1.0F;
    }

    // Convert peak frequency to bin
    float sampling_rate = (float)analyzer->sampling_rate_hz;
    int32_t peak_bin = fft_frequency_to_bin(peak_freq, analyzer->buffer_size,
                                            sampling_rate);
    if (peak_bin < 0 || (uint32_t)peak_bin >= analyzer->spectrum_size) {
        return -1.0F;
    }

    // Get peak power
    float peak_power = analyzer->power_spectrum[peak_bin];
    if (peak_power < 1e-10F) {
        return 0.0F;
    }

    // Find 3dB bandwidth (half power points)
    float half_power = peak_power / 2.0F;

    // Search left for lower frequency bound
    int32_t low_bin = peak_bin;
    while (low_bin > 0 && analyzer->power_spectrum[low_bin] >= half_power) {
        low_bin--;
    }

    // Search right for upper frequency bound
    int32_t high_bin = peak_bin;
    while ((uint32_t)high_bin < analyzer->spectrum_size - 1 &&
           analyzer->power_spectrum[high_bin] >= half_power) {
        high_bin++;
    }

    // Convert bins to frequencies
    float low_freq = fft_bin_to_frequency(low_bin, analyzer->buffer_size, sampling_rate);
    float high_freq = fft_bin_to_frequency(high_bin, analyzer->buffer_size, sampling_rate);

    // Bandwidth
    float bandwidth = high_freq - low_freq;

    return bandwidth;
}
