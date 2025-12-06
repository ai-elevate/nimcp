/**
 * @file nimcp_brain_oscillations.h
 * @brief Brain oscillation analysis using spectral methods
 *
 * WHAT: Analyze neural oscillations and brain waves in NIMCP brains
 * WHY:  Brain oscillations encode cognitive states, attention, sleep, learning
 * HOW:  FFT-based spectral analysis of neural activity patterns
 *
 * NEUROSCIENCE BACKGROUND:
 * Brain oscillations are rhythmic patterns of neural activity that:
 * - Coordinate communication between brain regions
 * - Encode cognitive states (attention, memory consolidation, sleep)
 * - Gate information flow through brain networks
 * - Support temporal binding of distributed representations
 *
 * OSCILLATION TYPES:
 * - Delta (1-4 Hz): Deep sleep, unconscious processes
 * - Theta (4-8 Hz): Memory consolidation, drowsiness, meditation
 * - Alpha (8-13 Hz): Relaxed wakefulness, eyes closed, default mode
 * - Beta (13-30 Hz): Active thinking, focus, motor control
 * - Gamma (30-100 Hz): Feature binding, consciousness, attention
 *
 * USE CASES:
 * - Detect attention states from gamma power
 * - Monitor memory consolidation via theta activity
 * - Identify sleep-like states from delta dominance
 * - Measure cross-frequency coupling for binding
 * - Analyze network synchrony and coherence
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 2.6 (GPU P2P + Spectral Analysis)
 */

#ifndef NIMCP_BRAIN_OSCILLATIONS_H
#define NIMCP_BRAIN_OSCILLATIONS_H

#include "core/brain/nimcp_brain.h"
#include "utils/spectral/nimcp_fft.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Data Structures
//=============================================================================

/**
 * @brief Oscillation analyzer for brain activity
 *
 * WHAT: Tracks neural activity over time and performs spectral analysis
 * WHY:  Detect oscillations, measure power in frequency bands
 * HOW:  Ring buffer for activity history + FFT for frequency analysis
 */
typedef struct brain_oscillation_analyzer_struct brain_oscillation_analyzer_t;

/**
 * @brief Brain wave power measurements
 */
typedef struct {
    float delta_power;    /**< Delta band power (1-4 Hz) */
    float theta_power;    /**< Theta band power (4-8 Hz) */
    float alpha_power;    /**< Alpha band power (8-13 Hz) */
    float beta_power;     /**< Beta band power (13-30 Hz) */
    float gamma_power;    /**< Gamma band power (30-100 Hz) */

    float total_power;    /**< Total power across all frequencies */
    float dominant_freq;  /**< Dominant frequency (Hz) */
    brain_wave_band_t dominant_band;  /**< Dominant brain wave band */
} brain_wave_power_t;

/**
 * @brief Cognitive state inferred from oscillations
 */
typedef enum {
    COGNITIVE_STATE_UNKNOWN,        /**< Not enough data to determine */
    COGNITIVE_STATE_DEEP_SLEEP,     /**< Delta dominance */
    COGNITIVE_STATE_LIGHT_SLEEP,    /**< Theta + delta */
    COGNITIVE_STATE_RELAXED,        /**< Alpha dominance */
    COGNITIVE_STATE_FOCUSED,        /**< Beta dominance */
    COGNITIVE_STATE_ATTENTIVE,      /**< Gamma + beta */
    COGNITIVE_STATE_CONSOLIDATING   /**< Theta dominance (memory) */
} cognitive_state_t;

/**
 * @brief Oscillation analysis results
 */
typedef struct {
    brain_wave_power_t wave_power;   /**< Power in each brain wave band */
    cognitive_state_t state;          /**< Inferred cognitive state */
    float state_confidence;           /**< Confidence in state (0-1) */

    // Spectral metrics
    float spectral_entropy;           /**< Entropy of power distribution */
    float peak_frequency;             /**< Frequency of peak power */
    float bandwidth;                  /**< Bandwidth of dominant peak */

    // Cross-frequency coupling
    float theta_gamma_coupling;       /**< Theta-gamma PAC (phase-amplitude) */
    float alpha_beta_coupling;        /**< Alpha-beta coupling */

    // Synchrony
    float synchrony;                  /**< Network synchrony (0-1) */
    float coherence;                  /**< Inter-regional coherence */
} oscillation_analysis_t;

//=============================================================================
// Analyzer Creation and Management
//=============================================================================

/**
 * WHAT: Create brain oscillation analyzer
 * WHY:  Track neural activity for spectral analysis
 * HOW:  Allocate ring buffer, FFT plan, and tracking structures
 *
 * @param brain Brain to analyze
 * @param window_size_ms Analysis window size in milliseconds (100-1000ms typical)
 * @param sampling_rate_hz Sampling rate in Hz (100-1000 Hz typical)
 * @return Analyzer handle or NULL on failure
 *
 * COMPLEXITY: O(N) where N = window_size_ms * sampling_rate_hz / 1000
 * MEMORY: O(N) for activity buffer + FFT workspace
 *
 * RECOMMENDED SETTINGS:
 * - Short-term (attention): 250ms window, 500Hz sampling
 * - Medium-term (task): 500ms window, 250Hz sampling
 * - Long-term (sleep): 1000ms window, 100Hz sampling
 *
 * NOTE: Window size must allow capturing lowest frequency of interest
 * - Delta (1 Hz) requires >= 1000ms window
 * - Theta (4 Hz) requires >= 250ms window
 */
brain_oscillation_analyzer_t* brain_oscillation_create(
    brain_t brain,
    uint32_t window_size_ms,
    uint32_t sampling_rate_hz
);

/**
 * WHAT: Destroy oscillation analyzer
 * WHY:  Free resources
 * HOW:  Free buffer, FFT plan, and analyzer structure
 *
 * @param analyzer Analyzer to destroy
 */
void brain_oscillation_destroy(brain_oscillation_analyzer_t* analyzer);

//=============================================================================
// Activity Recording
//=============================================================================

/**
 * WHAT: Record current brain activity snapshot
 * WHY:  Build up temporal history for spectral analysis
 * HOW:  Sample average neuron activation, add to ring buffer
 *
 * @param analyzer Oscillation analyzer
 * @return true on success, false on failure
 *
 * COMPLEXITY: O(N) where N = number of neurons
 *
 * USAGE:
 * Call this regularly at sampling_rate_hz to build activity history.
 * For example, if sampling_rate = 250Hz, call every 4ms.
 */
bool brain_oscillation_record_activity(brain_oscillation_analyzer_t* analyzer);

/**
 * WHAT: Record custom activity value
 * WHY:  Inject pre-computed activity metric
 * HOW:  Add value to ring buffer
 *
 * @param analyzer Oscillation analyzer
 * @param activity Activity value (typically 0-1)
 * @return true on success, false on failure
 *
 * USE CASES:
 * - External sensors (EEG, MEG)
 * - Custom activity metrics
 * - Replay recorded data
 */
bool brain_oscillation_record_value(
    brain_oscillation_analyzer_t* analyzer,
    float activity
);

//=============================================================================
// Spectral Analysis
//=============================================================================

/**
 * WHAT: Analyze current oscillation patterns
 * WHY:  Extract brain wave power, cognitive state, and spectral metrics
 * HOW:  FFT on activity buffer, compute power in each band, infer state
 *
 * @param analyzer Oscillation analyzer
 * @param results Output analysis results
 * @return true on success, false on failure (e.g., insufficient data)
 *
 * COMPLEXITY: O(N log N) where N = buffer size (for FFT)
 *
 * REQUIRES:
 * - Activity buffer must be full (window_size samples recorded)
 * - At least one dominant oscillation present
 *
 * OUTPUT:
 * - Power in each brain wave band
 * - Inferred cognitive state
 * - Spectral metrics (entropy, peak frequency, etc.)
 */
bool brain_oscillation_analyze(
    brain_oscillation_analyzer_t* analyzer,
    oscillation_analysis_t* results
);

/**
 * WHAT: Get brain wave power in real-time
 * WHY:  Quick check of oscillatory state
 * HOW:  FFT + band power computation
 *
 * @param analyzer Oscillation analyzer
 * @param wave_power Output brain wave power
 * @return true on success, false on failure
 *
 * COMPLEXITY: O(N log N)
 *
 * NOTE: Lighter-weight than full analyze() - just returns power
 */
bool brain_oscillation_get_wave_power(
    brain_oscillation_analyzer_t* analyzer,
    brain_wave_power_t* wave_power
);

/**
 * WHAT: Get inferred cognitive state
 * WHY:  Quick state classification from oscillations
 * HOW:  Analyze dominant frequency band, compare power ratios
 *
 * @param analyzer Oscillation analyzer
 * @param state Output cognitive state
 * @param confidence Output confidence (0-1)
 * @return true on success, false on failure
 *
 * HEURISTICS:
 * - Delta > 60% total → DEEP_SLEEP
 * - Theta > 40% total → CONSOLIDATING or LIGHT_SLEEP
 * - Alpha > 40% total → RELAXED
 * - Beta > 40% total → FOCUSED
 * - Gamma > 30% total → ATTENTIVE
 */
bool brain_oscillation_get_state(
    brain_oscillation_analyzer_t* analyzer,
    cognitive_state_t* state,
    float* confidence
);

//=============================================================================
// Cross-Frequency Coupling
//=============================================================================

/**
 * WHAT: Compute phase-amplitude coupling (PAC)
 * WHY:  Measure how low-freq phase modulates high-freq amplitude
 * HOW:  Extract phase of low freq, amplitude of high freq, correlate
 *
 * @param analyzer Oscillation analyzer
 * @param phase_band Low-frequency phase band (e.g., theta)
 * @param amplitude_band High-frequency amplitude band (e.g., gamma)
 * @return PAC strength (0-1), or -1.0 on error
 *
 * COMPLEXITY: O(N log N) for FFT + O(N) for correlation
 *
 * NEUROSCIENCE SIGNIFICANCE:
 * - Theta-gamma PAC: Memory encoding/retrieval
 * - Alpha-beta PAC: Attention gating
 * - Delta-gamma PAC: Sleep spindles
 *
 * PAC VALUES:
 * - 0.0-0.2: Weak/no coupling
 * - 0.2-0.4: Moderate coupling
 * - 0.4-1.0: Strong coupling
 */
float brain_oscillation_compute_pac(
    brain_oscillation_analyzer_t* analyzer,
    brain_wave_band_t phase_band,
    brain_wave_band_t amplitude_band
);

//=============================================================================
// Network Synchrony
//=============================================================================

/**
 * WHAT: Compute network synchrony using Kuramoto order parameter
 * WHY:  Measure phase synchronization of neural oscillations
 * HOW:  Kuramoto order parameter R = |⟨e^(iθ)⟩| from instantaneous phases
 *
 * @param analyzer Oscillation analyzer
 * @return Synchrony (0-1), or -1.0 on error
 *
 * COMPLEXITY: O(N log N) where N = buffer size
 *
 * SYNCHRONY VALUES:
 * - 0.0-0.3: Asynchronous (independent oscillations)
 * - 0.3-0.6: Partially synchronized
 * - 0.6-1.0: Highly synchronized (coherent oscillations)
 *
 * INTERPRETATION:
 * - High synchrony: Strong phase locking, focused processing
 * - Low synchrony: Independent phases, exploration
 */
float brain_oscillation_compute_synchrony(brain_oscillation_analyzer_t* analyzer);

/**
 * WHAT: Compute spectral coherence
 * WHY:  Measure consistency of oscillations in frequency domain
 * HOW:  Spectral concentration index (inverse of spectral entropy)
 *
 * @param analyzer Oscillation analyzer
 * @return Coherence (0-1), or -1.0 on error
 *
 * COMPLEXITY: O(N) where N = spectrum size
 *
 * COHERENCE VALUES:
 * - 0.0-0.3: Broadband/noisy activity
 * - 0.3-0.7: Moderate spectral concentration
 * - 0.7-1.0: Highly coherent oscillations
 *
 * INTERPRETATION:
 * - High coherence: Stable, precise rhythms
 * - Low coherence: Transient or irregular activity
 */
float brain_oscillation_compute_coherence(brain_oscillation_analyzer_t* analyzer);

/**
 * WHAT: Compute 3dB bandwidth around peak frequency
 * WHY:  Quantify sharpness of dominant oscillation
 * HOW:  Find frequency range where power drops to half of peak
 *
 * @param analyzer Oscillation analyzer
 * @param peak_freq Dominant frequency in Hz
 * @return Bandwidth in Hz, or -1.0 on error
 *
 * COMPLEXITY: O(N) where N = spectrum size
 *
 * INTERPRETATION:
 * - Narrow bandwidth (< 2 Hz): Sharp, stable oscillation
 * - Moderate bandwidth (2-5 Hz): Normal variability
 * - Wide bandwidth (> 5 Hz): Transient or unstable rhythm
 */
float brain_oscillation_compute_bandwidth(
    brain_oscillation_analyzer_t* analyzer,
    float peak_freq);

//=============================================================================
// Visualization and Export
//=============================================================================

/**
 * WHAT: Get power spectrum for plotting
 * WHY:  Visualize frequency content
 * HOW:  Return FFT power spectrum
 *
 * @param analyzer Oscillation analyzer
 * @param spectrum Output power spectrum [num_bins]
 * @param num_bins Number of frequency bins (output)
 * @return true on success, false on failure
 *
 * FREQUENCY BINS:
 * - bin[k] corresponds to frequency k * (sampling_rate / window_size)
 * - Use fft_bin_to_frequency() to convert bin index to Hz
 */
bool brain_oscillation_get_spectrum(
    brain_oscillation_analyzer_t* analyzer,
    float** spectrum,
    uint32_t* num_bins
);

/**
 * WHAT: Get raw activity buffer for export
 * WHY:  Export raw data for external analysis or plotting
 * HOW:  Return pointer to ring buffer
 *
 * @param analyzer Oscillation analyzer
 * @param buffer Output buffer pointer [window_size samples]
 * @param size Number of samples (output)
 * @return true on success, false on failure
 *
 * WARNING: Buffer is read-only. Do not modify.
 */
bool brain_oscillation_get_activity_buffer(
    brain_oscillation_analyzer_t* analyzer,
    const float** buffer,
    uint32_t* size
);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * WHAT: Convert cognitive state to string
 * WHY:  Human-readable output
 * HOW:  Enum to string mapping
 *
 * @param state Cognitive state
 * @return String name
 */
const char* brain_oscillation_state_to_string(cognitive_state_t state);

/**
 * WHAT: Get recommended window size for frequency band
 * WHY:  Ensure window captures at least 3 cycles of lowest frequency
 * HOW:  window_ms >= 3000 / freq_low_hz
 *
 * @param band Brain wave band
 * @return Recommended window size in milliseconds
 *
 * EXAMPLES:
 * - Delta (1 Hz): 3000ms minimum
 * - Theta (4 Hz): 750ms minimum
 * - Alpha (8 Hz): 375ms minimum
 * - Beta (13 Hz): 230ms minimum
 * - Gamma (30 Hz): 100ms minimum
 */
uint32_t brain_oscillation_recommended_window(brain_wave_band_t band);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_BRAIN_OSCILLATIONS_H */
