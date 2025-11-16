/**
 * @file nimcp_audio_cortex.h
 * @brief Biologically-inspired auditory processing system
 *
 * WHAT: Audio cortex with cochlear processing and temporal analysis
 * WHY:  Enable auditory perception and memory in NIMCP
 * HOW:  FFT-based frequency analysis + temporal pattern recognition
 *
 * Mimics biological auditory pathway:
 * - Cochlea: Frequency decomposition (basilar membrane simulation)
 * - Primary Auditory Cortex (A1): Tonotopic feature extraction
 * - Higher-order areas: Temporal pattern recognition
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 2.6
 */

#ifndef NIMCP_AUDIO_CORTEX_H
#define NIMCP_AUDIO_CORTEX_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration for brain integration
typedef struct brain_struct* brain_t;

//=============================================================================
// Neuromodulation Structures
//=============================================================================

// NOTE: phasic_tonic_state_t and receptor_expression_t are defined in
// nimcp_visual_cortex.h to avoid duplicate definitions. Include that header
// first if you need these types in audio cortex.

//=============================================================================
// Configuration Constants
//=============================================================================

#define AUDIO_MAX_SAMPLE_RATE 48000
#define AUDIO_MIN_SAMPLE_RATE 8000
#define AUDIO_MAX_CHANNELS 2
#define AUDIO_MAX_MEMORIES 1000
#define AUDIO_MAX_FREQ_BINS 512

//=============================================================================
// Core Structures
//=============================================================================

/**
 * @brief Audio cortex configuration
 */
typedef struct {
    uint32_t sample_rate;        ///< Sample rate (Hz)
    uint32_t frame_size;         ///< Frame size (samples)
    uint32_t num_freq_bins;      ///< Number of frequency bins
    uint32_t num_mel_filters;    ///< Number of mel-scale filters
    uint32_t num_mfcc;           ///< Number of MFCC coefficients
    uint8_t num_channels;        ///< 1=mono, 2=stereo
    uint32_t feature_dim;        ///< Feature vector dimension
    bool enable_attention;       ///< Enable attention mechanisms
    bool enable_memory;          ///< Enable auditory memory

    // NIMCP 2.7 Phase 8.5: Fractal Topology Integration
    bool enable_fractal_topology;  /**< Enable scale-free topology within A1 */
    float hub_ratio;               /**< Fraction of hub neurons (0.1-0.2), default: 0.15 */
    float power_law_gamma;         /**< Power-law exponent (-2 to -3), default: -2.1 */
    uint32_t internal_neurons;     /**< Number of internal A1 neurons, default: num_mel_filters * 10 */
} audio_cortex_config_t;

/**
 * @brief Auditory memory entry
 */
typedef struct {
    float* features;             ///< Audio feature vector
    float salience;              ///< Memory salience [0,1]
    uint64_t timestamp;          ///< When memory was created
    char context[64];            ///< Context label
} auditory_memory_t;

/**
 * @brief Attention map for audio (frequency/time)
 */
typedef struct {
    float* values;               ///< Attention weights
    uint32_t num_freq;           ///< Frequency bins
    uint32_t num_time;           ///< Time frames
} audio_attention_map_t;

/**
 * @brief Audio cortex instance (opaque)
 */
typedef struct audio_cortex audio_cortex_t;

/**
 * @brief Audio cortex statistics
 */
typedef struct {
    uint64_t frames_processed;   ///< Total frames processed
    uint32_t memories_stored;    ///< Auditory memories stored
    float avg_processing_time;   ///< Avg processing time (ms)
} audio_cortex_stats_t;

//=============================================================================
// Core API
//=============================================================================

/**
 * @brief Create audio cortex instance
 * @param config Configuration parameters
 * @return Audio cortex instance or NULL on failure
 */
audio_cortex_t* audio_cortex_create(const audio_cortex_config_t* config);

/**
 * @brief Destroy audio cortex instance
 * @param cortex Audio cortex to destroy
 */
void audio_cortex_destroy(audio_cortex_t* cortex);

/**
 * @brief Process audio frame and extract features
 *
 * @param cortex Audio cortex instance
 * @param audio_data Raw audio samples (float32)
 * @param num_samples Number of samples
 * @param num_channels Number of channels (1 or 2)
 * @param features Output feature vector (must be pre-allocated)
 * @return true on success, false on failure
 */
bool audio_cortex_process(
    audio_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    uint8_t num_channels,
    float* features
);

/**
 * @brief Get audio cortex statistics
 * @param cortex Audio cortex instance
 * @param stats Output statistics structure
 * @return true on success, false on failure
 */
bool audio_cortex_get_stats(
    const audio_cortex_t* cortex,
    audio_cortex_stats_t* stats
);

//=============================================================================
// Cochlear Processing (Frequency Analysis)
//=============================================================================

/**
 * @brief Compute power spectrum using FFT
 *
 * @param cortex Audio cortex instance
 * @param audio_data Time-domain audio samples
 * @param num_samples Number of samples
 * @param spectrum Output power spectrum (must be pre-allocated)
 * @return true on success, false on failure
 */
bool audio_cortex_compute_spectrum(
    audio_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    float* spectrum
);

/**
 * @brief Apply mel-scale filterbank
 *
 * @param cortex Audio cortex instance
 * @param spectrum Power spectrum
 * @param num_bins Number of frequency bins
 * @param mel_features Output mel-scale features
 * @return true on success, false on failure
 */
bool audio_cortex_compute_mel_features(
    audio_cortex_t* cortex,
    const float* spectrum,
    uint32_t num_bins,
    float* mel_features
);

/**
 * @brief Compute MFCC features
 *
 * @param cortex Audio cortex instance
 * @param mel_features Mel-scale features
 * @param num_mel Number of mel filters
 * @param mfcc Output MFCC coefficients
 * @return true on success, false on failure
 */
bool audio_cortex_compute_mfcc(
    audio_cortex_t* cortex,
    const float* mel_features,
    uint32_t num_mel,
    float* mfcc
);

//=============================================================================
// Attention Mechanisms
//=============================================================================

/**
 * @brief Create audio attention map
 * @param num_freq Number of frequency bins
 * @param num_time Number of time frames
 * @return Attention map or NULL on failure
 */
audio_attention_map_t* audio_attention_map_create(
    uint32_t num_freq,
    uint32_t num_time
);

/**
 * @brief Destroy audio attention map
 * @param map Attention map to destroy
 */
void audio_attention_map_destroy(audio_attention_map_t* map);

/**
 * @brief Compute attention map from audio features
 *
 * @param cortex Audio cortex instance
 * @param audio_data Audio samples
 * @param num_samples Number of samples
 * @param attn_map Output attention map
 * @return true on success, false on failure
 */
bool audio_cortex_compute_attention(
    audio_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    audio_attention_map_t* attn_map
);

//=============================================================================
// Auditory Memory
//=============================================================================

/**
 * @brief Store auditory memory
 *
 * @param cortex Audio cortex instance
 * @param features Audio feature vector
 * @param salience Memory salience [0,1]
 * @return true on success, false on failure
 */
bool audio_cortex_store_memory(
    audio_cortex_t* cortex,
    const float* features,
    float salience
);

/**
 * @brief Recall similar auditory memories
 *
 * @param cortex Audio cortex instance
 * @param query_features Query feature vector
 * @param top_k Number of memories to retrieve
 * @param memories Output array of memory pointers
 * @param num_recalled Number of memories actually recalled
 * @return true on success, false on failure
 *
 * @note Caller must free the memories array with nimcp_free()
 */
bool audio_cortex_recall_memory(
    audio_cortex_t* cortex,
    const float* query_features,
    int top_k,
    auditory_memory_t*** memories,
    int* num_recalled
);

//=============================================================================
// Brain Integration Helpers
//=============================================================================

/**
 * @brief Compute novelty score for curiosity system
 *
 * Returns how novel/unfamiliar an audio pattern is based on
 * similarity to stored auditory memories.
 *
 * @param cortex Audio cortex instance
 * @param features Audio feature vector
 * @return Novelty score [0,1], 0=familiar, 1=novel
 */
float audio_cortex_compute_novelty(
    audio_cortex_t* cortex,
    const float* features
);

/**
 * @brief Get peak attention (most salient frequency/time)
 *
 * @param attn_map Attention map
 * @param max_freq Output: frequency bin with max attention
 * @param max_time Output: time frame with max attention
 * @param max_value Output: attention value at peak
 * @return true on success, false on failure
 */
bool audio_cortex_get_attention_peak(
    const audio_attention_map_t* attn_map,
    uint32_t* max_freq,
    uint32_t* max_time,
    float* max_value
);

/**
 * @brief Consolidate auditory memory
 *
 * Integrates with memory consolidation system (hippocampus, etc.)
 *
 * @param cortex Audio cortex instance
 * @param features Audio feature vector
 * @param salience Memory salience [0,1]
 * @param context Context label for memory
 * @return true on success, false on failure
 */
bool audio_cortex_consolidate_memory(
    audio_cortex_t* cortex,
    const float* features,
    float salience,
    const char* context
);

//=============================================================================
// Temporal Processing
//=============================================================================

/**
 * @brief Detect temporal patterns (onset, offset, rhythm)
 *
 * @param cortex Audio cortex instance
 * @param audio_data Audio samples
 * @param num_samples Number of samples
 * @param onset_detected Output: onset detected flag
 * @param offset_detected Output: offset detected flag
 * @return true on success, false on failure
 */
bool audio_cortex_detect_temporal_events(
    audio_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    bool* onset_detected,
    bool* offset_detected
);

/**
 * @brief Compute temporal envelope
 *
 * @param cortex Audio cortex instance
 * @param audio_data Audio samples
 * @param num_samples Number of samples
 * @param envelope Output envelope (must be pre-allocated)
 * @return true on success, false on failure
 */
bool audio_cortex_compute_envelope(
    audio_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    float* envelope
);

/**
 * @brief Associate brain with audio cortex for neuromodulation
 *
 * WHAT: Set brain reference for ACh + 5-HT modulation
 * WHY:  Enable neurochemical modulation of auditory processing
 * HOW:  Store brain pointer for neurotransmitter reading
 *
 * @param cortex Audio cortex instance
 * @param brain Brain instance (or NULL to clear)
 *
 * BIOLOGY:
 * - Acetylcholine enhances frequency selectivity (cocktail party effect)
 * - Serotonin gates auditory sensitivity (prevent sensory overload)
 *
 * CLINICAL EXAMPLES:
 * - Autism (low 5-HT): Sound sensitivity, overwhelmed by noise
 * - ADHD (low ACh): Poor auditory attention, can't filter background
 */
void audio_cortex_set_brain(audio_cortex_t* cortex, brain_t brain);

//=============================================================================
// Bidirectional Feedback Functions (Phase 10.11.3)
//=============================================================================

/**
 * @brief Get speech salience from audio features
 *
 * WHAT: Query how speech-like current audio is
 * WHY:  Speech cortex can prioritize speech processing
 * HOW:  Return energy concentration in speech frequencies (300-3400 Hz)
 *
 * BIOLOGY: Superior temporal gyrus (STG) receives speech-tuned signals from A1
 *
 * @param cortex Audio cortex instance
 * @param features Audio feature vector from recent processing
 * @param num_features Number of features
 * @return Speech salience [0, 1] (0=noise, 1=clear speech)
 */
float audio_cortex_get_speech_salience(audio_cortex_t* cortex,
                                        const float* features,
                                        uint32_t num_features);

/**
 * @brief Activate speech processing mode
 *
 * WHAT: Signal that speech detected, optimize for phoneme extraction
 * WHY:  Speech detection triggers specialized processing
 * HOW:  Prime frequency bands and temporal resolution for speech
 *
 * @param cortex Audio cortex instance
 */
void audio_cortex_activate_speech_mode(audio_cortex_t* cortex);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_AUDIO_CORTEX_H
