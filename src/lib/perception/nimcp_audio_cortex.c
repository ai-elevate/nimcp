/**
 * @file nimcp_audio_cortex.c
 * @brief Implementation of biologically-inspired auditory processing
 *
 * WHAT: Cochlear processing with FFT-based frequency analysis
 * WHY:  Enable auditory perception and memory in NIMCP
 * HOW:  Mel-scale filterbank + MFCC + temporal pattern recognition
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 2.6
 */

#include "include/perception/nimcp_audio_cortex.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Audio cortex internal structure
 */
struct audio_cortex {
    // Configuration
    audio_cortex_config_t config;

    // Cochlear processing
    float* fft_window;           ///< Hamming window
    float* fft_real;             ///< FFT real part buffer
    float* fft_imag;             ///< FFT imaginary part buffer
    float* mel_filterbank;       ///< Mel-scale filter weights

    // Temporal processing
    float* prev_frame;           ///< Previous frame for onset detection
    float prev_energy;           ///< Previous frame energy

    // Memory system
    auditory_memory_t** memories;
    uint32_t num_memories;
    uint32_t memory_capacity;

    // Statistics
    audio_cortex_stats_t stats;
};

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Compute Hamming window
 */
static void compute_hamming_window(float* window, uint32_t size)
{
    if (!nimcp_validate_pointer(window, "window")) return;

    for (uint32_t i = 0; i < size; i++) {
        window[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (size - 1));
    }
}

/**
 * @brief Convert frequency to mel scale
 */
static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

/**
 * @brief Convert mel to frequency
 */
static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/**
 * @brief Initialize mel filterbank
 */
static bool init_mel_filterbank(audio_cortex_t* cortex)
{
    if (!nimcp_validate_pointer(cortex, "cortex")) return false;

    uint32_t num_filters = cortex->config.num_mel_filters;
    uint32_t num_bins = cortex->config.num_freq_bins;
    float sample_rate = (float)cortex->config.sample_rate;

    // Allocate filterbank matrix
    cortex->mel_filterbank = (float*)nimcp_calloc(
        num_filters * num_bins, sizeof(float)
    );
    if (!nimcp_validate_pointer(cortex->mel_filterbank, "mel_filterbank")) {
        NIMCP_LOGGING_ERROR("Failed to allocate mel filterbank");
        return false;
    }

    // Compute mel-scale boundaries
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel(sample_rate / 2.0f);
    float mel_step = (mel_max - mel_min) / (num_filters + 1);

    // Create triangular filters
    for (uint32_t m = 0; m < num_filters; m++) {
        float f_left = mel_to_hz(mel_min + m * mel_step);
        float f_center = mel_to_hz(mel_min + (m + 1) * mel_step);
        float f_right = mel_to_hz(mel_min + (m + 2) * mel_step);

        // Convert frequencies to bin indices
        uint32_t bin_left = (uint32_t)(f_left * num_bins / (sample_rate / 2.0f));
        uint32_t bin_center = (uint32_t)(f_center * num_bins / (sample_rate / 2.0f));
        uint32_t bin_right = (uint32_t)(f_right * num_bins / (sample_rate / 2.0f));

        // Rising slope
        for (uint32_t k = bin_left; k < bin_center && k < num_bins; k++) {
            float weight = (float)(k - bin_left) / (bin_center - bin_left);
            cortex->mel_filterbank[m * num_bins + k] = weight;
        }

        // Falling slope
        for (uint32_t k = bin_center; k < bin_right && k < num_bins; k++) {
            float weight = (float)(bin_right - k) / (bin_right - bin_center);
            cortex->mel_filterbank[m * num_bins + k] = weight;
        }
    }

    return true;
}

/**
 * @brief Simple FFT implementation (Cooley-Tukey radix-2)
 */
static void fft(float* real, float* imag, uint32_t n, bool inverse)
{
    if (n <= 1) return;

    // Bit-reversal permutation
    uint32_t j = 0;
    for (uint32_t i = 0; i < n - 1; i++) {
        if (i < j) {
            float temp = real[i];
            real[i] = real[j];
            real[j] = temp;
            temp = imag[i];
            imag[i] = imag[j];
            imag[j] = temp;
        }
        uint32_t k = n / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    // Cooley-Tukey FFT
    float direction = inverse ? 1.0f : -1.0f;
    for (uint32_t s = 1; s <= (uint32_t)log2f((float)n); s++) {
        uint32_t m = 1 << s;
        uint32_t m2 = m / 2;
        float w_real = cosf(direction * 2.0f * M_PI / m);
        float w_imag = sinf(direction * 2.0f * M_PI / m);

        for (uint32_t k = 0; k < n; k += m) {
            float wr = 1.0f;
            float wi = 0.0f;

            for (uint32_t j = 0; j < m2; j++) {
                uint32_t t = k + j;
                uint32_t u = k + j + m2;

                float tr = real[u] * wr - imag[u] * wi;
                float ti = real[u] * wi + imag[u] * wr;

                real[u] = real[t] - tr;
                imag[u] = imag[t] - ti;
                real[t] = real[t] + tr;
                imag[t] = imag[t] + ti;

                float temp = wr;
                wr = temp * w_real - wi * w_imag;
                wi = temp * w_imag + wi * w_real;
            }
        }
    }

    // Normalize for inverse transform
    if (inverse) {
        for (uint32_t i = 0; i < n; i++) {
            real[i] /= n;
            imag[i] /= n;
        }
    }
}

/**
 * @brief Compute DCT for MFCC
 */
static void dct(const float* input, float* output, uint32_t n, uint32_t num_coeff)
{
    if (!nimcp_validate_pointer(input, "input") || !nimcp_validate_pointer(output, "output")) return;

    for (uint32_t k = 0; k < num_coeff; k++) {
        float sum = 0.0f;
        for (uint32_t n_idx = 0; n_idx < n; n_idx++) {
            sum += input[n_idx] * cosf(M_PI * k * (n_idx + 0.5f) / n);
        }
        output[k] = sum;
    }
}

//=============================================================================
// Core API Implementation
//=============================================================================

audio_cortex_t* audio_cortex_create(const audio_cortex_config_t* config)
{
    if (!nimcp_validate_pointer(config, "config")) {
        return NULL;
    }

    // Validate configuration
    if (config->sample_rate < AUDIO_MIN_SAMPLE_RATE ||
        config->sample_rate > AUDIO_MAX_SAMPLE_RATE) {
        NIMCP_LOGGING_ERROR("Invalid sample rate: %u", config->sample_rate);
        return NULL;
    }

    if (config->num_channels == 0 || config->num_channels > AUDIO_MAX_CHANNELS) {
        NIMCP_LOGGING_ERROR("Invalid number of channels: %u", config->num_channels);
        return NULL;
    }

    // Allocate cortex structure
    audio_cortex_t* cortex = (audio_cortex_t*)nimcp_calloc(1, sizeof(audio_cortex_t));
    if (!cortex) {
        NIMCP_LOGGING_ERROR("Failed to allocate audio cortex");
        return NULL;
    }

    // Copy configuration
    memcpy(&cortex->config, config, sizeof(audio_cortex_config_t));

    // Allocate FFT buffers
    cortex->fft_real = (float*)nimcp_calloc(config->frame_size, sizeof(float));
    cortex->fft_imag = (float*)nimcp_calloc(config->frame_size, sizeof(float));
    cortex->fft_window = (float*)nimcp_calloc(config->frame_size, sizeof(float));

    if (!nimcp_validate_pointer(cortex->fft_real, "fft_real") ||
        !nimcp_validate_pointer(cortex->fft_imag, "fft_imag") ||
        !nimcp_validate_pointer(cortex->fft_window, "fft_window")) {
        NIMCP_LOGGING_ERROR("Failed to allocate FFT buffers");
        audio_cortex_destroy(cortex);
        return NULL;
    }

    // Initialize Hamming window
    compute_hamming_window(cortex->fft_window, config->frame_size);

    // Initialize mel filterbank
    if (!init_mel_filterbank(cortex)) {
        audio_cortex_destroy(cortex);
        return NULL;
    }

    // Allocate temporal processing buffers
    cortex->prev_frame = (float*)nimcp_calloc(config->frame_size, sizeof(float));
    if (!nimcp_validate_pointer(cortex->prev_frame, "prev_frame")) {
        NIMCP_LOGGING_ERROR("Failed to allocate temporal processing buffer");
        audio_cortex_destroy(cortex);
        return NULL;
    }

    // Initialize memory system
    if (config->enable_memory) {
        cortex->memory_capacity = AUDIO_MAX_MEMORIES;
        cortex->memories = (auditory_memory_t**)nimcp_calloc(
            cortex->memory_capacity, sizeof(auditory_memory_t*)
        );
        if (!nimcp_validate_pointer(cortex->memories, "memories")) {
            NIMCP_LOGGING_ERROR("Failed to allocate auditory memory array");
            audio_cortex_destroy(cortex);
            return NULL;
        }
    }

    return cortex;
}

void audio_cortex_destroy(audio_cortex_t* cortex)
{
    if (!cortex) return;

    // Free FFT buffers
    nimcp_free(cortex->fft_real);
    nimcp_free(cortex->fft_imag);
    nimcp_free(cortex->fft_window);
    nimcp_free(cortex->mel_filterbank);

    // Free temporal buffers
    nimcp_free(cortex->prev_frame);

    // Free memories
    if (cortex->memories) {
        for (uint32_t i = 0; i < cortex->num_memories; i++) {
            if (cortex->memories[i]) {
                nimcp_free(cortex->memories[i]->features);
                nimcp_free(cortex->memories[i]);
            }
        }
        nimcp_free(cortex->memories);
    }

    nimcp_free(cortex);
}

bool audio_cortex_process(
    audio_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    uint8_t num_channels,
    float* features)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(audio_data, "audio_data") ||
        !nimcp_validate_pointer(features, "features")) {
        return false;
    }

    if (num_samples != cortex->config.frame_size) {
        NIMCP_LOGGING_ERROR("Invalid frame size: %u (expected %u)", num_samples, cortex->config.frame_size);
        return false;
    }

    if (num_channels != cortex->config.num_channels) {
        NIMCP_LOGGING_ERROR("Invalid number of channels: %u (expected %u)", num_channels, cortex->config.num_channels);
        return false;
    }

    // Convert to mono if stereo
    float* mono_audio = (float*)audio_data;
    float* temp_mono = NULL;

    if (num_channels == 2) {
        temp_mono = (float*)nimcp_calloc(num_samples, sizeof(float));
        if (!nimcp_validate_pointer(temp_mono, "temp_mono")) {
            NIMCP_LOGGING_ERROR("Failed to allocate mono conversion buffer");
            return false;
        }

        for (uint32_t i = 0; i < num_samples; i++) {
            temp_mono[i] = (audio_data[i * 2] + audio_data[i * 2 + 1]) * 0.5f;
        }
        mono_audio = temp_mono;
    }

    // Compute power spectrum
    float* spectrum = (float*)nimcp_calloc(cortex->config.num_freq_bins, sizeof(float));
    if (!nimcp_validate_pointer(spectrum, "spectrum")) {
        NIMCP_LOGGING_ERROR("Failed to allocate spectrum buffer");
        nimcp_free(temp_mono);
        return false;
    }

    bool success = audio_cortex_compute_spectrum(cortex, mono_audio, num_samples, spectrum);

    if (success) {
        // Compute mel features
        float* mel_features = (float*)nimcp_calloc(cortex->config.num_mel_filters, sizeof(float));
        if (mel_features) {
            success = audio_cortex_compute_mel_features(
                cortex, spectrum, cortex->config.num_freq_bins, mel_features
            );

            if (success) {
                // Compute MFCC
                success = audio_cortex_compute_mfcc(
                    cortex, mel_features, cortex->config.num_mel_filters, features
                );
            }

            nimcp_free(mel_features);
        } else {
            success = false;
        }
    }

    nimcp_free(spectrum);
    nimcp_free(temp_mono);

    if (success) {
        cortex->stats.frames_processed++;
    }

    return success;
}

bool audio_cortex_get_stats(
    const audio_cortex_t* cortex,
    audio_cortex_stats_t* stats)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(stats, "stats")) {
        return false;
    }

    memcpy(stats, &cortex->stats, sizeof(audio_cortex_stats_t));
    return true;
}

//=============================================================================
// Cochlear Processing Implementation
//=============================================================================

bool audio_cortex_compute_spectrum(
    audio_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    float* spectrum)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(audio_data, "audio_data") ||
        !nimcp_validate_pointer(spectrum, "spectrum")) {
        return false;
    }

    if (num_samples != cortex->config.frame_size) {
        NIMCP_LOGGING_ERROR("Invalid frame size: %u (expected %u)", num_samples, cortex->config.frame_size);
        return false;
    }

    // Apply window and prepare for FFT
    for (uint32_t i = 0; i < num_samples; i++) {
        cortex->fft_real[i] = audio_data[i] * cortex->fft_window[i];
        cortex->fft_imag[i] = 0.0f;
    }

    // Compute FFT
    fft(cortex->fft_real, cortex->fft_imag, num_samples, false);

    // Compute power spectrum (magnitude squared)
    uint32_t num_bins = cortex->config.num_freq_bins;
    for (uint32_t i = 0; i < num_bins; i++) {
        float re = cortex->fft_real[i];
        float im = cortex->fft_imag[i];
        spectrum[i] = re * re + im * im;
    }

    return true;
}

bool audio_cortex_compute_mel_features(
    audio_cortex_t* cortex,
    const float* spectrum,
    uint32_t num_bins,
    float* mel_features)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(spectrum, "spectrum") ||
        !nimcp_validate_pointer(mel_features, "mel_features")) {
        return false;
    }

    if (num_bins != cortex->config.num_freq_bins) {
        NIMCP_LOGGING_ERROR("Invalid number of bins: %u (expected %u)", num_bins, cortex->config.num_freq_bins);
        return false;
    }

    // Apply mel filterbank
    uint32_t num_filters = cortex->config.num_mel_filters;
    for (uint32_t m = 0; m < num_filters; m++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < num_bins; k++) {
            sum += spectrum[k] * cortex->mel_filterbank[m * num_bins + k];
        }

        // Convert to log scale (add small epsilon to avoid log(0))
        mel_features[m] = logf(sum + 1e-10f);
    }

    return true;
}

bool audio_cortex_compute_mfcc(
    audio_cortex_t* cortex,
    const float* mel_features,
    uint32_t num_mel,
    float* mfcc)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(mel_features, "mel_features") ||
        !nimcp_validate_pointer(mfcc, "mfcc")) {
        return false;
    }

    if (num_mel != cortex->config.num_mel_filters) {
        NIMCP_LOGGING_ERROR("Invalid number of mel filters: %u (expected %u)", num_mel, cortex->config.num_mel_filters);
        return false;
    }

    // Apply DCT to mel features
    dct(mel_features, mfcc, num_mel, cortex->config.num_mfcc);

    return true;
}

//=============================================================================
// Attention Mechanisms Implementation
//=============================================================================

audio_attention_map_t* audio_attention_map_create(
    uint32_t num_freq,
    uint32_t num_time)
{
    if (num_freq == 0 || num_time == 0) {
        NIMCP_LOGGING_ERROR("Invalid audio attention map dimensions: %u x %u", num_freq, num_time);
        return NULL;
    }

    audio_attention_map_t* map = (audio_attention_map_t*)nimcp_calloc(
        1, sizeof(audio_attention_map_t)
    );
    if (!nimcp_validate_pointer(map, "map")) {
        NIMCP_LOGGING_ERROR("Failed to allocate audio attention map");
        return NULL;
    }

    map->num_freq = num_freq;
    map->num_time = num_time;
    map->values = (float*)nimcp_calloc(num_freq * num_time, sizeof(float));

    if (!nimcp_validate_pointer(map->values, "map->values")) {
        NIMCP_LOGGING_ERROR("Failed to allocate audio attention map values");
        nimcp_free(map);
        return NULL;
    }

    return map;
}

void audio_attention_map_destroy(audio_attention_map_t* map)
{
    if (!map) return;

    nimcp_free(map->values);
    nimcp_free(map);
}

bool audio_cortex_compute_attention(
    audio_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    audio_attention_map_t* attn_map)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(audio_data, "audio_data") ||
        !nimcp_validate_pointer(attn_map, "attn_map")) {
        return false;
    }

    // Compute spectrum
    float* spectrum = (float*)nimcp_calloc(cortex->config.num_freq_bins, sizeof(float));
    if (!nimcp_validate_pointer(spectrum, "spectrum")) {
        NIMCP_LOGGING_ERROR("Failed to allocate spectrum buffer for attention");
        return false;
    }

    bool success = audio_cortex_compute_spectrum(cortex, audio_data, num_samples, spectrum);

    if (success) {
        // Normalize spectrum to attention weights
        float max_val = 0.0f;
        for (uint32_t i = 0; i < cortex->config.num_freq_bins; i++) {
            if (spectrum[i] > max_val) max_val = spectrum[i];
        }

        if (max_val > 0.0f) {
            for (uint32_t i = 0; i < cortex->config.num_freq_bins && i < attn_map->num_freq; i++) {
                // Simple attention: high energy frequencies get high attention
                attn_map->values[i] = spectrum[i] / max_val;
            }
        }
    }

    nimcp_free(spectrum);
    return success;
}

//=============================================================================
// Auditory Memory Implementation
//=============================================================================

bool audio_cortex_store_memory(
    audio_cortex_t* cortex,
    const float* features,
    float salience)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(features, "features")) {
        return false;
    }

    if (!cortex->config.enable_memory) {
        return false;
    }

    // Check capacity
    if (cortex->num_memories >= cortex->memory_capacity) {
        return false;  // Memory full
    }

    // Create memory entry
    auditory_memory_t* memory = (auditory_memory_t*)nimcp_calloc(
        1, sizeof(auditory_memory_t)
    );
    if (!nimcp_validate_pointer(memory, "memory")) {
        NIMCP_LOGGING_ERROR("Failed to allocate auditory memory entry");
        return false;
    }

    memory->features = (float*)nimcp_calloc(cortex->config.feature_dim, sizeof(float));
    if (!nimcp_validate_pointer(memory->features, "memory->features")) {
        NIMCP_LOGGING_ERROR("Failed to allocate auditory memory features");
        nimcp_free(memory);
        return false;
    }

    memcpy(memory->features, features, cortex->config.feature_dim * sizeof(float));
    memory->salience = salience;
    memory->timestamp = (uint64_t)time(NULL);
    memory->context[0] = '\0';

    cortex->memories[cortex->num_memories++] = memory;
    cortex->stats.memories_stored = cortex->num_memories;

    return true;
}

bool audio_cortex_recall_memory(
    audio_cortex_t* cortex,
    const float* query_features,
    int top_k,
    auditory_memory_t*** memories,
    int* num_recalled)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(query_features, "query_features") ||
        !nimcp_validate_pointer(memories, "memories") ||
        !nimcp_validate_pointer(num_recalled, "num_recalled")) {
        return false;
    }

    *memories = NULL;
    *num_recalled = 0;

    if (!cortex->config.enable_memory || cortex->num_memories == 0) {
        return true;  // No memories to recall
    }

    // Compute similarities
    typedef struct {
        auditory_memory_t* memory;
        float similarity;
    } memory_score_t;

    memory_score_t* scores = (memory_score_t*)nimcp_calloc(
        cortex->num_memories, sizeof(memory_score_t)
    );
    if (!nimcp_validate_pointer(scores, "scores")) {
        NIMCP_LOGGING_ERROR("Failed to allocate memory score buffer");
        return false;
    }

    for (uint32_t i = 0; i < cortex->num_memories; i++) {
        float dot_product = 0.0f;
        for (uint32_t j = 0; j < cortex->config.feature_dim; j++) {
            dot_product += query_features[j] * cortex->memories[i]->features[j];
        }
        scores[i].memory = cortex->memories[i];
        scores[i].similarity = dot_product;
    }

    // Sort by similarity (simple bubble sort for small arrays)
    for (uint32_t i = 0; i < cortex->num_memories - 1; i++) {
        for (uint32_t j = 0; j < cortex->num_memories - i - 1; j++) {
            if (scores[j].similarity < scores[j + 1].similarity) {
                memory_score_t temp = scores[j];
                scores[j] = scores[j + 1];
                scores[j + 1] = temp;
            }
        }
    }

    // Return top-k
    int k = (top_k < (int)cortex->num_memories) ? top_k : (int)cortex->num_memories;
    *memories = (auditory_memory_t**)nimcp_calloc(k, sizeof(auditory_memory_t*));
    if (!nimcp_validate_pointer(*memories, "result_memories")) {
        NIMCP_LOGGING_ERROR("Failed to allocate memory results array");
        nimcp_free(scores);
        return false;
    }

    for (int i = 0; i < k; i++) {
        (*memories)[i] = scores[i].memory;
    }
    *num_recalled = k;

    nimcp_free(scores);
    return true;
}

//=============================================================================
// Brain Integration Helpers Implementation
//=============================================================================

float audio_cortex_compute_novelty(
    audio_cortex_t* cortex,
    const float* features)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(features, "features")) {
        return 0.0f;
    }

    if (!cortex->config.enable_memory || cortex->num_memories == 0) {
        return 1.0f;  // Everything is novel with no memories
    }

    // Find maximum similarity to any stored memory
    float max_similarity = 0.0f;
    for (uint32_t i = 0; i < cortex->num_memories; i++) {
        float dot_product = 0.0f;
        for (uint32_t j = 0; j < cortex->config.feature_dim; j++) {
            dot_product += features[j] * cortex->memories[i]->features[j];
        }

        if (dot_product > max_similarity) {
            max_similarity = dot_product;
        }
    }

    // Novelty = 1 - max_similarity
    float novelty = 1.0f - max_similarity;
    if (novelty < 0.0f) novelty = 0.0f;
    if (novelty > 1.0f) novelty = 1.0f;

    return novelty;
}

bool audio_cortex_get_attention_peak(
    const audio_attention_map_t* attn_map,
    uint32_t* max_freq,
    uint32_t* max_time,
    float* max_value)
{
    if (!nimcp_validate_pointer(attn_map, "attn_map") ||
        !nimcp_validate_pointer(max_freq, "max_freq") ||
        !nimcp_validate_pointer(max_time, "max_time") ||
        !nimcp_validate_pointer(max_value, "max_value")) {
        return false;
    }

    if (!nimcp_validate_pointer(attn_map->values, "attn_map->values")) {
        NIMCP_LOGGING_ERROR("Audio attention map has no values");
        return false;
    }

    *max_value = -INFINITY;
    *max_freq = 0;
    *max_time = 0;

    for (uint32_t t = 0; t < attn_map->num_time; t++) {
        for (uint32_t f = 0; f < attn_map->num_freq; f++) {
            float value = attn_map->values[t * attn_map->num_freq + f];
            if (value > *max_value) {
                *max_value = value;
                *max_freq = f;
                *max_time = t;
            }
        }
    }

    return true;
}

bool audio_cortex_consolidate_memory(
    audio_cortex_t* cortex,
    const float* features,
    float salience,
    const char* context)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(features, "features")) {
        return false;
    }

    // Store memory
    if (!audio_cortex_store_memory(cortex, features, salience)) {
        return false;
    }

    // Add context if provided
    if (context && cortex->num_memories > 0) {
        auditory_memory_t* last_memory = cortex->memories[cortex->num_memories - 1];
        strncpy(last_memory->context, context, sizeof(last_memory->context) - 1);
        last_memory->context[sizeof(last_memory->context) - 1] = '\0';
    }

    // TODO: Future hippocampus integration
    // TODO: Future knowledge graph integration

    return true;
}

//=============================================================================
// Temporal Processing Implementation
//=============================================================================

bool audio_cortex_detect_temporal_events(
    audio_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    bool* onset_detected,
    bool* offset_detected)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(audio_data, "audio_data") ||
        !nimcp_validate_pointer(onset_detected, "onset_detected") ||
        !nimcp_validate_pointer(offset_detected, "offset_detected")) {
        return false;
    }

    *onset_detected = false;
    *offset_detected = false;

    // Compute current frame energy
    float energy = 0.0f;
    for (uint32_t i = 0; i < num_samples; i++) {
        energy += audio_data[i] * audio_data[i];
    }
    energy /= num_samples;

    // Onset: sudden increase in energy
    float onset_threshold = 2.0f;
    if (energy > cortex->prev_energy * onset_threshold && cortex->prev_energy > 1e-6f) {
        *onset_detected = true;
    }

    // Offset: sudden decrease in energy
    float offset_threshold = 0.5f;
    if (energy < cortex->prev_energy * offset_threshold && cortex->prev_energy > 1e-6f) {
        *offset_detected = true;
    }

    // Update state
    cortex->prev_energy = energy;
    memcpy(cortex->prev_frame, audio_data, num_samples * sizeof(float));

    return true;
}

bool audio_cortex_compute_envelope(
    audio_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    float* envelope)
{
    if (!nimcp_validate_pointer(cortex, "cortex") ||
        !nimcp_validate_pointer(audio_data, "audio_data") ||
        !nimcp_validate_pointer(envelope, "envelope")) {
        return false;
    }

    // Simple envelope: rectify and smooth
    float alpha = 0.1f;  // Smoothing factor
    float smoothed = 0.0f;

    for (uint32_t i = 0; i < num_samples; i++) {
        float rectified = fabsf(audio_data[i]);
        smoothed = alpha * rectified + (1.0f - alpha) * smoothed;
        envelope[i] = smoothed;
    }

    return true;
}
