//=============================================================================
// nimcp_feature_extractor.c - Neural Feature Extraction Implementation
//=============================================================================

#include "middleware/features/nimcp_feature_extractor.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * WHAT: Feature extractor internal state
 * WHY:  Maintain configuration and working buffers
 * HOW:  Store config, allocate workspace for computations
 */
struct feature_extractor_struct {
    feature_extractor_config_t config;

    // Thread safety
    nimcp_platform_mutex_t mutex;

    // Working buffers (avoid repeated allocation)
    float* rate_buffer;          // Per-neuron rates
    float* isi_buffer;           // ISI values
    uint32_t* count_buffer;      // Spike counts
    uint32_t buffer_capacity;    // Current buffer size

    // Statistics
    uint64_t total_extractions;
    uint64_t last_update_time;
};

//=============================================================================
// Helper Function Declarations
//=============================================================================

static bool ensure_buffer_capacity(feature_extractor_t extractor, uint32_t neurons);
static float compute_mean(const float* data, uint32_t count);
static float compute_std(const float* data, uint32_t count, float mean);
static float compute_cv(const float* isi_values, uint32_t count);
static bool extract_isi_for_neuron(const uint64_t* spikes, uint32_t count,
                                    float* isi_out, uint32_t* isi_count);
static bool extract_basic_features(feature_extractor_t extractor,
                                    const spike_data_t* spike_data,
                                    middleware_features_t* features);
static bool extract_optional_features(feature_extractor_t extractor,
                                       const spike_data_t* spike_data,
                                       middleware_features_t* features);
static uint32_t count_spike_coincidences(const spike_data_t* spike_data,
                                          uint64_t spike_time, uint32_t neuron_idx,
                                          float window_ms);
static bool build_rate_signal(const spike_data_t* spike_data, float* rate_signal,
                               uint32_t num_bins);
static void compute_band_power_autocorr(const float* rate_signal, uint32_t num_bins,
                                        float min_freq, float max_freq, float* power);
static void normalize_band_powers(const float* band_powers, float* delta, float* theta,
                                   float* alpha, float* beta, float* gamma);

//=============================================================================
// Lifecycle Functions
//=============================================================================

feature_extractor_config_t feature_extractor_default_config(void) {
    feature_extractor_config_t config = {0};
    config.window_ms = 100.0f;
    config.synchrony_window_ms = 5.0f;
    config.burst_isi_threshold_ms = 10.0f;
    config.min_burst_spikes = 3;
    config.entropy_bins = 20;
    config.compute_oscillations = true;
    config.compute_entropy = true;
    config.compute_synchrony = true;
    return config;
}

feature_extractor_t feature_extractor_create(const feature_extractor_config_t* config) {
    if (config && (config->window_ms < FEATURE_EXTRACTOR_MIN_WINDOW_MS ||
                   config->window_ms > FEATURE_EXTRACTOR_MAX_WINDOW_MS)) {
        return NULL;
    }

    feature_extractor_t extractor = nimcp_calloc(1, sizeof(struct feature_extractor_struct));
    if (!extractor) {
        return NULL;
    }

    extractor->config = config ? *config : feature_extractor_default_config();
    extractor->buffer_capacity = 1024;

    extractor->rate_buffer = nimcp_calloc(extractor->buffer_capacity, sizeof(float));
    extractor->isi_buffer = nimcp_calloc(extractor->buffer_capacity * 10, sizeof(float));
    extractor->count_buffer = nimcp_calloc(extractor->buffer_capacity, sizeof(uint32_t));

    if (!extractor->rate_buffer || !extractor->isi_buffer || !extractor->count_buffer) {
        feature_extractor_destroy(extractor);
        return NULL;
    }

    if (nimcp_platform_mutex_init(&extractor->mutex, false) != 0) {
        feature_extractor_destroy(extractor);
        return NULL;
    }

    return extractor;
}

void feature_extractor_destroy(feature_extractor_t extractor) {
    if (!extractor) {
        return;
    }

    nimcp_platform_mutex_destroy(&extractor->mutex);
    nimcp_free(extractor->rate_buffer);
    nimcp_free(extractor->isi_buffer);
    nimcp_free(extractor->count_buffer);
    nimcp_free(extractor);
}

//=============================================================================
// Main Extraction Function
//=============================================================================

bool feature_extractor_update(
    feature_extractor_t extractor,
    const spike_data_t* spike_data,
    middleware_features_t* features_out
) {
    if (!extractor || !spike_data || !features_out) {
        return false;
    }

    if (spike_data->num_neurons == 0 || spike_data->num_neurons > FEATURE_EXTRACTOR_MAX_NEURONS) {
        return false;
    }

    // WHY: Check if all spike counts are zero (empty data)
    // HOW: Sum spike counts and fail if zero
    uint32_t total_spikes = 0;
    for (uint32_t i = 0; i < spike_data->num_neurons; i++) {
        total_spikes += spike_data->spike_counts[i];
    }
    if (total_spikes == 0) {
        return false;
    }

    nimcp_platform_mutex_lock(&extractor->mutex);

    memset(features_out, 0, sizeof(middleware_features_t));
    features_out->valid = false;

    if (!ensure_buffer_capacity(extractor, spike_data->num_neurons)) {
        nimcp_platform_mutex_unlock(&extractor->mutex);
        return false;
    }

    bool success = extract_basic_features(extractor, spike_data, features_out);
    success &= extract_optional_features(extractor, spike_data, features_out);

    features_out->timestamp = spike_data->end_time;
    features_out->valid = success;

    extractor->total_extractions++;
    extractor->last_update_time = spike_data->end_time;

    nimcp_platform_mutex_unlock(&extractor->mutex);

    return success;
}

//=============================================================================
// Individual Feature Computation
//=============================================================================

bool feature_extractor_compute_mean_firing_rate(
    feature_extractor_t extractor,
    const spike_data_t* spike_data,
    float* rate_out
) {
    if (!extractor || !spike_data || !rate_out || spike_data->num_neurons == 0) {
        return false;
    }

    float window_duration_s = (float)(spike_data->end_time - spike_data->start_time) / 1000.0f;
    if (window_duration_s <= 0.0f) {
        *rate_out = 0.0f;
        return true;
    }

    float total_rate = 0.0f;
    float rate_sum_sq = 0.0f;

    for (uint32_t i = 0; i < spike_data->num_neurons; i++) {
        float neuron_rate = (float)spike_data->spike_counts[i] / window_duration_s;
        extractor->rate_buffer[i] = neuron_rate;
        total_rate += neuron_rate;
        rate_sum_sq += neuron_rate * neuron_rate;
    }

    *rate_out = total_rate / (float)spike_data->num_neurons;
    return true;
}

bool feature_extractor_compute_population_cv(
    feature_extractor_t extractor,
    const spike_data_t* spike_data,
    float* cv_out
) {
    if (!extractor || !spike_data || !cv_out || spike_data->num_neurons == 0) {
        return false;
    }

    float cv_sum = 0.0f;
    uint32_t valid_neurons = 0;

    for (uint32_t i = 0; i < spike_data->num_neurons; i++) {
        if (spike_data->spike_counts[i] < 2) {
            continue;
        }

        uint32_t isi_count = 0;
        if (!extract_isi_for_neuron(spike_data->spike_times[i],
                                     spike_data->spike_counts[i],
                                     extractor->isi_buffer,
                                     &isi_count)) {
            continue;
        }

        if (isi_count > 0) {
            float cv = compute_cv(extractor->isi_buffer, isi_count);
            cv_sum += cv;
            valid_neurons++;
        }
    }

    // WHY: Fail if no valid neurons (all have < 2 spikes)
    // HOW: Return false for degenerate case
    if (valid_neurons == 0) {
        return false;
    }

    *cv_out = cv_sum / (float)valid_neurons;
    return true;
}

bool feature_extractor_compute_fano_factor(
    feature_extractor_t extractor,
    const spike_data_t* spike_data,
    float* fano_out
) {
    if (!extractor || !spike_data || !fano_out || spike_data->num_neurons == 0) {
        return false;
    }

    float mean_count = 0.0f;
    for (uint32_t i = 0; i < spike_data->num_neurons; i++) {
        mean_count += (float)spike_data->spike_counts[i];
    }
    mean_count /= (float)spike_data->num_neurons;

    // WHY: Fail if no spikes (all counts zero)
    // HOW: Return false for degenerate case
    if (mean_count < 0.001f) {
        return false;
    }

    float variance = 0.0f;
    for (uint32_t i = 0; i < spike_data->num_neurons; i++) {
        float diff = (float)spike_data->spike_counts[i] - mean_count;
        variance += diff * diff;
    }
    variance /= (float)spike_data->num_neurons;

    *fano_out = variance / mean_count;
    return true;
}

bool feature_extractor_compute_burst_index(
    feature_extractor_t extractor,
    const spike_data_t* spike_data,
    float* burst_index_out
) {
    if (!extractor || !spike_data || !burst_index_out || spike_data->num_neurons == 0) {
        return false;
    }

    uint32_t total_spikes = 0;
    uint32_t burst_spikes = 0;

    for (uint32_t neuron = 0; neuron < spike_data->num_neurons; neuron++) {
        uint32_t count = spike_data->spike_counts[neuron];
        const uint64_t* spikes = spike_data->spike_times[neuron];

        if (count < 2) {
            total_spikes += count;
            continue;
        }

        total_spikes += count;

        uint32_t current_burst_length = 1;
        for (uint32_t i = 1; i < count; i++) {
            float isi = (float)(spikes[i] - spikes[i-1]);

            if (isi < extractor->config.burst_isi_threshold_ms) {
                current_burst_length++;
            } else {
                if (current_burst_length >= extractor->config.min_burst_spikes) {
                    burst_spikes += current_burst_length;
                }
                current_burst_length = 1;
            }
        }

        if (current_burst_length >= extractor->config.min_burst_spikes) {
            burst_spikes += current_burst_length;
        }
    }

    if (total_spikes == 0) {
        *burst_index_out = 0.0f;
        return true;
    }

    *burst_index_out = (float)burst_spikes / (float)total_spikes;
    return true;
}

bool feature_extractor_compute_synchrony_index(
    feature_extractor_t extractor,
    const spike_data_t* spike_data,
    float* synchrony_out
) {
    if (!extractor || !spike_data || !synchrony_out || spike_data->num_neurons == 0) {
        return false;
    }

    if (spike_data->num_neurons < 2) {
        *synchrony_out = 0.0f;
        return true;
    }

    uint32_t total_coincidences = 0;
    uint32_t total_spikes = 0;
    float sync_window_ms = extractor->config.synchrony_window_ms;

    for (uint32_t i = 0; i < spike_data->num_neurons; i++) {
        for (uint32_t j = 0; j < spike_data->spike_counts[i]; j++) {
            uint64_t spike_time = spike_data->spike_times[i][j];
            total_spikes++;

            uint32_t coincident = count_spike_coincidences(spike_data, spike_time, i, sync_window_ms);
            total_coincidences += coincident;
        }
    }

    if (total_spikes == 0) {
        *synchrony_out = 0.0f;
        return true;
    }

    float max_possible = (float)(spike_data->num_neurons - 1);
    if (max_possible < 1.0f) {
        *synchrony_out = 0.0f;
        return true;
    }

    float avg_coincidences = (float)total_coincidences / (float)total_spikes;
    *synchrony_out = fminf(avg_coincidences / max_possible, 1.0f);

    return true;
}

bool feature_extractor_compute_oscillation_power(
    feature_extractor_t extractor,
    const spike_data_t* spike_data,
    float* delta_power,
    float* theta_power,
    float* alpha_power,
    float* beta_power,
    float* gamma_power
) {
    if (!extractor || !spike_data || !delta_power || !theta_power ||
        !alpha_power || !beta_power || !gamma_power) {
        return false;
    }

    *delta_power = *theta_power = *alpha_power = *beta_power = *gamma_power = 0.0f;

    if (spike_data->num_neurons == 0) {
        return true;
    }

    // WHY: Check if all spike counts are zero (no activity)
    // HOW: Sum spike counts and fail if zero
    uint32_t total_spikes = 0;
    for (uint32_t i = 0; i < spike_data->num_neurons; i++) {
        total_spikes += spike_data->spike_counts[i];
    }
    if (total_spikes == 0) {
        return false;
    }

    float window_duration_ms = (float)(spike_data->end_time - spike_data->start_time);
    uint32_t num_bins = (uint32_t)(window_duration_ms / 10.0f);

    if (window_duration_ms < 100.0f || num_bins < 10 || num_bins > 1000) {
        return true;
    }

    float* rate_signal = nimcp_calloc(num_bins, sizeof(float));
    if (!rate_signal) {
        return false;
    }

    if (!build_rate_signal(spike_data, rate_signal, num_bins)) {
        nimcp_free(rate_signal);
        return false;
    }

    float band_powers[FEATURE_EXTRACTOR_NUM_BANDS] = {0};
    compute_band_power_autocorr(rate_signal, num_bins, FEATURE_DELTA_MIN_HZ, FEATURE_DELTA_MAX_HZ, &band_powers[0]);
    compute_band_power_autocorr(rate_signal, num_bins, FEATURE_THETA_MIN_HZ, FEATURE_THETA_MAX_HZ, &band_powers[1]);
    compute_band_power_autocorr(rate_signal, num_bins, FEATURE_ALPHA_MIN_HZ, FEATURE_ALPHA_MAX_HZ, &band_powers[2]);
    compute_band_power_autocorr(rate_signal, num_bins, FEATURE_BETA_MIN_HZ, FEATURE_BETA_MAX_HZ, &band_powers[3]);
    compute_band_power_autocorr(rate_signal, num_bins, FEATURE_GAMMA_MIN_HZ, FEATURE_GAMMA_MAX_HZ, &band_powers[4]);

    normalize_band_powers(band_powers, delta_power, theta_power, alpha_power, beta_power, gamma_power);

    nimcp_free(rate_signal);
    return true;
}

bool feature_extractor_compute_spike_entropy(
    feature_extractor_t extractor,
    const spike_data_t* spike_data,
    float* entropy_out
) {
    if (!extractor || !spike_data || !entropy_out || spike_data->num_neurons == 0) {
        return false;
    }

    // WHY: Compute total spike count across all neurons
    // HOW: Sum individual counts to get distribution denominator
    uint32_t total_spikes = 0;
    for (uint32_t i = 0; i < spike_data->num_neurons; i++) {
        total_spikes += spike_data->spike_counts[i];
    }

    // WHY: Fail if no spikes (cannot compute entropy from empty distribution)
    // HOW: Return false for degenerate case
    if (total_spikes == 0) {
        return false;
    }

    // WHY: Compute Shannon entropy directly from spike count distribution
    // HOW: -Σ p_i * log2(p_i) where p_i = count_i / total
    float entropy = 0.0f;
    for (uint32_t i = 0; i < spike_data->num_neurons; i++) {
        if (spike_data->spike_counts[i] > 0) {
            float prob = (float)spike_data->spike_counts[i] / (float)total_spikes;
            entropy -= prob * log2f(prob);
        }
    }

    *entropy_out = entropy;
    return true;
}

//=============================================================================
// Utility Functions
//=============================================================================

middleware_features_t* middleware_features_create(void) {
    middleware_features_t* features = nimcp_calloc(1, sizeof(middleware_features_t));
    if (features) {
        features->valid = false;
    }
    return features;
}

void middleware_features_destroy(middleware_features_t* features) {
    nimcp_free(features);
}

spike_data_t* spike_data_create(uint32_t num_neurons) {
    if (num_neurons == 0 || num_neurons > FEATURE_EXTRACTOR_MAX_NEURONS) {
        return NULL;
    }

    spike_data_t* data = nimcp_calloc(1, sizeof(spike_data_t));
    if (!data) {
        return NULL;
    }

    data->spike_times = nimcp_calloc(num_neurons, sizeof(uint64_t*));
    data->spike_counts = nimcp_calloc(num_neurons, sizeof(uint32_t));

    if (!data->spike_times || !data->spike_counts) {
        spike_data_destroy(data);
        return NULL;
    }

    data->num_neurons = num_neurons;
    return data;
}

void spike_data_destroy(spike_data_t* data) {
    if (!data) {
        return;
    }

    if (data->spike_times) {
        for (uint32_t i = 0; i < data->num_neurons; i++) {
            nimcp_free(data->spike_times[i]);
        }
        nimcp_free(data->spike_times);
    }

    nimcp_free(data->spike_counts);
    nimcp_free(data);
}

void middleware_features_reset(middleware_features_t* features) {
    if (!features) {
        return;
    }
    features->valid = false;
}

//=============================================================================
// Helper Functions
//=============================================================================

static bool ensure_buffer_capacity(feature_extractor_t extractor, uint32_t neurons) {
    if (!extractor || neurons == 0) {
        return false;
    }

    if (neurons <= extractor->buffer_capacity) {
        return true;
    }

    uint32_t new_capacity = neurons * 2;

    float* new_rate = nimcp_realloc(extractor->rate_buffer,
                                    new_capacity * sizeof(float));
    if (!new_rate) {
        return false;
    }
    extractor->rate_buffer = new_rate;

    float* new_isi = nimcp_realloc(extractor->isi_buffer,
                                   new_capacity * 10 * sizeof(float));
    if (!new_isi) {
        return false;
    }
    extractor->isi_buffer = new_isi;

    uint32_t* new_count = nimcp_realloc(extractor->count_buffer,
                                        new_capacity * sizeof(uint32_t));
    if (!new_count) {
        return false;
    }
    extractor->count_buffer = new_count;

    extractor->buffer_capacity = new_capacity;
    return true;
}

static float compute_mean(const float* data, uint32_t count) {
    if (!data || count == 0) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        sum += data[i];
    }
    return sum / (float)count;
}

static float compute_std(const float* data, uint32_t count, float mean) {
    if (!data || count < 2) {
        return 0.0f;
    }

    float sum_sq_diff = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sqrtf(sum_sq_diff / (float)(count - 1));
}

static float compute_cv(const float* isi_values, uint32_t count) {
    if (!isi_values || count < 2) {
        return 0.0f;
    }

    float mean = compute_mean(isi_values, count);
    if (mean < 0.001f) {
        return 0.0f;
    }

    float std = compute_std(isi_values, count, mean);
    return std / mean;
}

static bool extract_isi_for_neuron(
    const uint64_t* spikes,
    uint32_t count,
    float* isi_out,
    uint32_t* isi_count
) {
    if (!spikes || !isi_out || !isi_count || count < 2) {
        if (isi_count) {
            *isi_count = 0;
        }
        return false;
    }

    for (uint32_t i = 0; i < count - 1; i++) {
        isi_out[i] = (float)(spikes[i + 1] - spikes[i]);
    }

    *isi_count = count - 1;
    return true;
}

static bool extract_basic_features(
    feature_extractor_t extractor,
    const spike_data_t* spike_data,
    middleware_features_t* features
) {
    if (!extractor || !spike_data || !features) {
        return false;
    }

    bool success = true;

    success &= feature_extractor_compute_mean_firing_rate(extractor, spike_data,
                                                           &features->mean_firing_rate);
    success &= feature_extractor_compute_fano_factor(extractor, spike_data,
                                                      &features->fano_factor);
    success &= feature_extractor_compute_burst_index(extractor, spike_data,
                                                      &features->burst_index);

    // WHY: Compute CV and mean ISI in single pass (optimization)
    // HOW: Extract ISIs once and compute both metrics together
    float cv_sum = 0.0f;
    float total_isi_sum = 0.0f;
    uint32_t total_isi_count = 0;
    uint32_t valid_neurons = 0;

    for (uint32_t i = 0; i < spike_data->num_neurons; i++) {
        if (spike_data->spike_counts[i] < 2) {
            continue;
        }
        uint32_t isi_count = 0;
        if (extract_isi_for_neuron(spike_data->spike_times[i],
                                    spike_data->spike_counts[i],
                                    extractor->isi_buffer,
                                    &isi_count) && isi_count > 0) {
            float cv = compute_cv(extractor->isi_buffer, isi_count);
            cv_sum += cv;
            valid_neurons++;

            for (uint32_t j = 0; j < isi_count; j++) {
                total_isi_sum += extractor->isi_buffer[j];
            }
            total_isi_count += isi_count;
        }
    }

    features->isi_cv = (valid_neurons > 0) ? (cv_sum / (float)valid_neurons) : 0.0f;
    features->mean_isi = (total_isi_count > 0) ? (total_isi_sum / (float)total_isi_count) : 0.0f;

    if (valid_neurons == 0) {
        success = false;
    }

    // WHY: Compute population rate standard deviation
    // HOW: Collect per-neuron rates and compute std
    float duration_sec = (float)(spike_data->end_time - spike_data->start_time) / 1000.0f;
    if (duration_sec > 0.0f) {
        for (uint32_t i = 0; i < spike_data->num_neurons; i++) {
            extractor->isi_buffer[i] = ((float)spike_data->spike_counts[i] / duration_sec);
        }
        float mean_rate = features->mean_firing_rate;
        features->population_rate_std = compute_std(extractor->isi_buffer, spike_data->num_neurons, mean_rate);
    } else {
        features->population_rate_std = 0.0f;
    }

    return success;
}

static bool extract_optional_features(
    feature_extractor_t extractor,
    const spike_data_t* spike_data,
    middleware_features_t* features
) {
    if (!extractor || !spike_data || !features) {
        return false;
    }

    if (extractor->config.compute_synchrony) {
        feature_extractor_compute_synchrony_index(extractor, spike_data,
                                                   &features->synchrony_index);
    }

    // WHY: Skip oscillation power for large populations (performance optimization)
    // HOW: Oscillation computation is O(n) with expensive autocorrelation - skip for n>500
    if (extractor->config.compute_oscillations && spike_data->num_neurons <= 500) {
        feature_extractor_compute_oscillation_power(
            extractor, spike_data,
            &features->delta_power,
            &features->theta_power,
            &features->alpha_power,
            &features->beta_power,
            &features->gamma_power
        );
    }

    if (extractor->config.compute_entropy) {
        feature_extractor_compute_spike_entropy(extractor, spike_data,
                                                 &features->spike_entropy);
    }

    return true;
}

static uint32_t count_spike_coincidences(
    const spike_data_t* spike_data,
    uint64_t spike_time,
    uint32_t neuron_idx,
    float window_ms
) {
    if (!spike_data) {
        return 0;
    }

    uint32_t coincident_neurons = 0;

    for (uint32_t k = 0; k < spike_data->num_neurons; k++) {
        if (k == neuron_idx) {
            continue;
        }

        for (uint32_t m = 0; m < spike_data->spike_counts[k]; m++) {
            uint64_t other_time = spike_data->spike_times[k][m];
            float time_diff = fabs((float)((int64_t)spike_time - (int64_t)other_time));

            if (time_diff <= window_ms) {
                coincident_neurons++;
                break;
            }
        }
    }

    return coincident_neurons;
}

static bool build_rate_signal(
    const spike_data_t* spike_data,
    float* rate_signal,
    uint32_t num_bins
) {
    if (!spike_data || !rate_signal || num_bins == 0) {
        return false;
    }

    for (uint32_t i = 0; i < spike_data->num_neurons; i++) {
        for (uint32_t j = 0; j < spike_data->spike_counts[i]; j++) {
            uint64_t spike_time = spike_data->spike_times[i][j];
            uint64_t relative_time = spike_time - spike_data->start_time;
            uint32_t bin = (uint32_t)(relative_time / 10);

            if (bin < num_bins) {
                rate_signal[bin] += 1.0f;
            }
        }
    }

    for (uint32_t i = 0; i < num_bins; i++) {
        rate_signal[i] *= 100.0f;
    }

    float mean_rate = compute_mean(rate_signal, num_bins);
    for (uint32_t i = 0; i < num_bins; i++) {
        rate_signal[i] -= mean_rate;
    }

    return true;
}

static void compute_band_power_autocorr(
    const float* rate_signal,
    uint32_t num_bins,
    float min_freq,
    float max_freq,
    float* power
) {
    if (!rate_signal || !power || num_bins == 0) {
        if (power) *power = 0.0f;
        return;
    }

    float min_period_bins = 100.0f / max_freq;
    float max_period_bins = 100.0f / min_freq;

    float total_power = 0.0f;
    uint32_t valid_windows = 0;

    for (uint32_t period_idx = (uint32_t)min_period_bins;
         period_idx <= (uint32_t)max_period_bins && period_idx > 0;
         period_idx++) {

        // Ensure start + period_idx stays within bounds
        if (period_idx >= num_bins) {
            continue;
        }

        for (uint32_t start = 0; start + period_idx < num_bins; start++) {
            float autocorr = 0.0f;
            // Fix buffer overflow: ensure start + i + period_idx < num_bins
            uint32_t max_offset = (num_bins > start + period_idx) ?
                                  (num_bins - start - period_idx) : 0;
            uint32_t max_i = (max_offset < 50) ? max_offset : 50;

            for (uint32_t i = 0; i < max_i; i++) {
                autocorr += rate_signal[start + i] * rate_signal[start + i + period_idx];
            }

            total_power += fabs(autocorr);
            valid_windows++;
        }
    }

    *power = (valid_windows > 0) ? (total_power / (float)valid_windows) : 0.0f;
}

static void normalize_band_powers(
    const float* band_powers,
    float* delta,
    float* theta,
    float* alpha,
    float* beta,
    float* gamma
) {
    if (!band_powers || !delta || !theta || !alpha || !beta || !gamma) {
        return;
    }

    float total = band_powers[0] + band_powers[1] + band_powers[2] + band_powers[3] + band_powers[4];

    if (total > 0.0f) {
        *delta = band_powers[0] / total;
        *theta = band_powers[1] / total;
        *alpha = band_powers[2] / total;
        *beta = band_powers[3] / total;
        *gamma = band_powers[4] / total;
    }
}
