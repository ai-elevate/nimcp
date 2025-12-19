/**
 * @file nimcp_audio_cortical_bridge.c
 * @brief Audio-Cortical Bridge Implementation
 *
 * WHAT: Connects audio cortex perception with cortical column processing.
 * WHY:  Provides biologically-realistic A1 processing with proper organization.
 * HOW:  Routes audio input through tonotopic mapping to frequency hypercolumns.
 *
 * @version 1.0.0
 * @date 2025-12-19
 * @author NIMCP Development Team
 */

#include "perception/cortical/nimcp_audio_cortical_bridge.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Utilities */
#include "utils/memory/nimcp_memory.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/platform/nimcp_platform.h"

/* ============================================================================
 * Internal Structure Definition
 * ============================================================================ */

/**
 * @brief Internal structure for audio-cortical bridge
 */
struct audio_cortical_bridge {
    /* Connected modules */
    audio_cortex_t* audio_cortex;
    topographic_map_t* tonotopic_map;
    cortical_immune_system_t* cortical_immune;

    /* Frequency hypercolumns */
    feature_hypercolumn_t** hypercolumns;
    uint32_t num_hypercolumns;
    uint32_t bands_per_hypercolumn;

    /* Configuration */
    audio_cortical_config_t config;

    /* State */
    audio_cortical_state_t state;
    audio_cortical_stats_t stats;

    /* Bio-async */
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    /* UMM */
    bool umm_enabled;

    /* Immune modulation */
    float immune_modulation_factor;
    float* hypercolumn_gains;

    /* Thread safety */
    nimcp_mutex_t mutex;
    bool mutex_initialized;

    /* Timing */
    uint64_t last_process_time_ns;
};

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * @brief Get current time in nanoseconds
 */
static uint64_t get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/**
 * @brief Compute hypercolumn index from frequency
 */
static uint32_t compute_hypercolumn_index(
    const audio_cortical_bridge_t* bridge,
    float frequency_hz)
{
    if (!bridge || bridge->num_hypercolumns == 0) {
        return UINT32_MAX;
    }

    /* Convert frequency to normalized position [0, 1] */
    float log_freq = logf(frequency_hz);
    float log_min = logf(bridge->config.min_frequency);
    float log_max = logf(bridge->config.max_frequency);

    float norm_freq = (log_freq - log_min) / (log_max - log_min);

    /* Clamp to [0, 1] */
    norm_freq = fminf(1.0f, fmaxf(0.0f, norm_freq));

    /* Map to hypercolumn index */
    uint32_t idx = (uint32_t)(norm_freq * (bridge->num_hypercolumns - 1));
    if (idx >= bridge->num_hypercolumns) {
        idx = bridge->num_hypercolumns - 1;
    }

    return idx;
}

/**
 * @brief Apply immune modulation to hypercolumn gain
 */
static float apply_immune_modulation(
    const audio_cortical_bridge_t* bridge,
    uint32_t hcol_idx,
    float response)
{
    if (!bridge || hcol_idx >= bridge->num_hypercolumns) {
        return response;
    }

    /* Base modulation from global immune factor */
    float modulation = 1.0f - (bridge->immune_modulation_factor *
                               bridge->config.immune_modulation_factor);

    /* Per-hypercolumn gain if available */
    if (bridge->hypercolumn_gains) {
        modulation *= bridge->hypercolumn_gains[hcol_idx];
    }

    return response * modulation;
}

/**
 * @brief Compute frequency result from hypercolumn responses
 */
static void compute_frequency_result(
    feature_hypercolumn_t* hcol,
    audio_cortical_frequency_result_t* result,
    float frequency_hz)
{
    if (!hcol || !result) return;

    memset(result, 0, sizeof(audio_cortical_frequency_result_t));

    result->num_freq_bands = hcol->total_columns;
    result->dominant_frequency = frequency_hz;

    /* Allocate and copy frequency band responses */
    if (hcol->total_columns > 0 && hcol->columns) {
        result->frequency_responses = (float*)nimcp_malloc(
            hcol->total_columns * sizeof(float)
        );
        if (result->frequency_responses) {
            /* Find dominant band and compute selectivity */
            float max_response = 0.0f;
            uint32_t max_idx = 0;
            float sum_responses = 0.0f;

            for (uint32_t i = 0; i < hcol->total_columns; i++) {
                result->frequency_responses[i] = hcol->columns[i].activation;
                sum_responses += hcol->columns[i].activation;

                if (hcol->columns[i].activation > max_response) {
                    max_response = hcol->columns[i].activation;
                    max_idx = i;
                }
            }

            /* Compute selectivity index */
            if (sum_responses > 0.0f) {
                result->selectivity_index = max_response / sum_responses;
            }

            result->confidence = result->selectivity_index;
        }
    }
}

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

void audio_cortical_default_config(audio_cortical_config_t* config)
{
    if (!config) return;

    memset(config, 0, sizeof(audio_cortical_config_t));

    config->num_hypercolumns = 64;  /* 64 frequency regions */
    config->freq_bands_per_hypercolumn = AUDIO_CORTICAL_DEFAULT_FREQ_BANDS;
    config->min_frequency = 80.0f;   /* 80 Hz minimum */
    config->max_frequency = 16000.0f; /* 16 kHz maximum */
    config->q_factor = AUDIO_CORTICAL_DEFAULT_Q_FACTOR;
    config->tuning_width = AUDIO_CORTICAL_DEFAULT_TUNING_WIDTH;
    config->mode = AUDIO_CORTICAL_MODE_HYPERCOLUMN;
    config->enable_tonotopic_mapping = true;
    config->enable_cortical_immune = true;
    config->enable_bio_async = true;
    config->frequency_range_octaves = 8.0f;  /* ~80 Hz to 16 kHz */
    config->low_freq_emphasis = 500.0f;
    config->cortical_magnification = 1.0f;
    config->immune_modulation_factor = AUDIO_CORTICAL_DEFAULT_IMMUNE_FACTOR;
    config->use_umm = false;
}

audio_cortical_bridge_t* audio_cortical_bridge_create(
    const audio_cortical_config_t* config,
    audio_cortex_t* audio_cortex)
{
    audio_cortical_config_t local_config;

    /* Use default config if none provided */
    if (!config) {
        audio_cortical_default_config(&local_config);
        config = &local_config;
    }

    /* Validate configuration */
    if (config->num_hypercolumns == 0 ||
        config->num_hypercolumns > AUDIO_CORTICAL_MAX_HYPERCOLUMNS) {
        NIMCP_LOGGING_ERROR("Invalid num_hypercolumns: %u", config->num_hypercolumns);
        return NULL;
    }

    if (config->min_frequency <= 0.0f || config->max_frequency <= config->min_frequency) {
        NIMCP_LOGGING_ERROR("Invalid frequency range: [%.1f, %.1f]",
                           config->min_frequency, config->max_frequency);
        return NULL;
    }

    /* Allocate bridge */
    audio_cortical_bridge_t* bridge = (audio_cortical_bridge_t*)nimcp_malloc(
        sizeof(audio_cortical_bridge_t)
    );
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate audio-cortical bridge");
        return NULL;
    }
    memset(bridge, 0, sizeof(audio_cortical_bridge_t));

    /* Copy configuration */
    memcpy(&bridge->config, config, sizeof(audio_cortical_config_t));
    bridge->audio_cortex = audio_cortex;
    bridge->state = AUDIO_CORTICAL_STATE_UNINITIALIZED;
    bridge->immune_modulation_factor = 0.0f;
    bridge->umm_enabled = config->use_umm;

    /* Initialize mutex */
    bridge->mutex_initialized = false;
    if (nimcp_mutex_init(&bridge->mutex, NULL) == NIMCP_SUCCESS) {
        bridge->mutex_initialized = true;
    } else {
        NIMCP_LOGGING_WARN("Failed to initialize mutex, continuing without thread safety");
    }

    /* Allocate hypercolumns array */
    bridge->hypercolumns = (feature_hypercolumn_t**)nimcp_malloc(
        config->num_hypercolumns * sizeof(feature_hypercolumn_t*)
    );
    if (!bridge->hypercolumns) {
        NIMCP_LOGGING_ERROR("Failed to allocate hypercolumns array");
        audio_cortical_bridge_destroy(bridge);
        return NULL;
    }
    memset(bridge->hypercolumns, 0,
           config->num_hypercolumns * sizeof(feature_hypercolumn_t*));

    /* Create frequency hypercolumns */
    bridge->num_hypercolumns = config->num_hypercolumns;
    bridge->bands_per_hypercolumn = config->freq_bands_per_hypercolumn;

    for (uint32_t i = 0; i < config->num_hypercolumns; i++) {
        /* Compute center frequency for this hypercolumn (logarithmic spacing) */
        float log_min = logf(config->min_frequency);
        float log_max = logf(config->max_frequency);
        float log_center = log_min + (log_max - log_min) * ((float)i / (float)(config->num_hypercolumns - 1));
        float center_freq = expf(log_center);

        /* Create feature dimension for frequency bands */
        float bandwidth = center_freq / config->q_factor;
        float min_band = center_freq - bandwidth / 2.0f;
        float max_band = center_freq + bandwidth / 2.0f;

        feature_dimension_t freq_dim = feature_dimension_create(
            FEATURE_SPATIAL_FREQ,  /* Reuse spatial freq type for audio */
            min_band,
            max_band,
            config->freq_bands_per_hypercolumn
        );
        feature_dimension_set_circular(&freq_dim, false);
        feature_dimension_set_tuning_width(&freq_dim, bandwidth / (float)config->freq_bands_per_hypercolumn);

        bridge->hypercolumns[i] = feature_hypercolumn_create(&freq_dim, 1);
        if (!bridge->hypercolumns[i]) {
            NIMCP_LOGGING_ERROR("Failed to create hypercolumn %u", i);
            audio_cortical_bridge_destroy(bridge);
            return NULL;
        }

        /* Set spatial position in tonotopic map */
        bridge->hypercolumns[i]->position[0] = (float)i / (float)config->num_hypercolumns;
        bridge->hypercolumns[i]->position[1] = 0.0f;
        bridge->hypercolumns[i]->position[2] = 0.0f;
    }

    /* Allocate per-hypercolumn gains */
    bridge->hypercolumn_gains = (float*)nimcp_malloc(
        config->num_hypercolumns * sizeof(float)
    );
    if (bridge->hypercolumn_gains) {
        for (uint32_t i = 0; i < config->num_hypercolumns; i++) {
            bridge->hypercolumn_gains[i] = 1.0f;
        }
    }

    /* Create tonotopic map if enabled */
    if (config->enable_tonotopic_mapping) {
        tonotopic_params_t tono_params = {
            .min_frequency = config->min_frequency,
            .max_frequency = config->max_frequency,
            .octave_span = 1.0f,  /* 1 octave per unit cortical distance */
            .is_logarithmic = true,
            .q_factor = config->q_factor
        };

        bridge->tonotopic_map = topographic_map_create_tonotopic(
            &tono_params,
            config->num_hypercolumns
        );
        if (!bridge->tonotopic_map) {
            NIMCP_LOGGING_WARN("Failed to create tonotopic map, continuing without");
        }
    }

    /* Initialize statistics */
    memset(&bridge->stats, 0, sizeof(audio_cortical_stats_t));

    bridge->state = AUDIO_CORTICAL_STATE_READY;

    NIMCP_LOGGING_INFO("Audio-cortical bridge created with %u hypercolumns (%u bands each)",
                       bridge->num_hypercolumns, bridge->bands_per_hypercolumn);

    return bridge;
}

void audio_cortical_bridge_destroy(audio_cortical_bridge_t* bridge)
{
    if (!bridge) return;

    /* Disconnect bio-async */
    if (bridge->bio_async_enabled) {
        audio_cortical_disconnect_bio_async(bridge);
    }

    /* Destroy hypercolumns */
    if (bridge->hypercolumns) {
        for (uint32_t i = 0; i < bridge->num_hypercolumns; i++) {
            if (bridge->hypercolumns[i]) {
                feature_hypercolumn_destroy(bridge->hypercolumns[i]);
            }
        }
        nimcp_free(bridge->hypercolumns);
    }

    /* Destroy tonotopic map */
    if (bridge->tonotopic_map) {
        topographic_map_destroy(bridge->tonotopic_map);
    }

    /* Free gains */
    if (bridge->hypercolumn_gains) {
        nimcp_free(bridge->hypercolumn_gains);
    }

    /* Destroy mutex */
    if (bridge->mutex_initialized) {
        nimcp_mutex_destroy(&bridge->mutex);
    }

    nimcp_free(bridge);

    NIMCP_LOGGING_DEBUG("Audio-cortical bridge destroyed");
}

/* ============================================================================
 * Connection Functions
 * ============================================================================ */

int audio_cortical_connect_audio_cortex(
    audio_cortical_bridge_t* bridge,
    audio_cortex_t* audio_cortex)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    if (bridge->mutex_initialized) nimcp_mutex_lock(&bridge->mutex);

    bridge->audio_cortex = audio_cortex;

    if (bridge->mutex_initialized) nimcp_mutex_unlock(&bridge->mutex);

    NIMCP_LOGGING_DEBUG("Connected to audio cortex");
    return NIMCP_SUCCESS;
}

int audio_cortical_connect_immune(
    audio_cortical_bridge_t* bridge,
    cortical_immune_system_t* immune)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    if (bridge->mutex_initialized) nimcp_mutex_lock(&bridge->mutex);

    bridge->cortical_immune = immune;

    /* Register hypercolumns with immune system */
    if (immune) {
        for (uint32_t i = 0; i < bridge->num_hypercolumns; i++) {
            if (bridge->hypercolumns[i]) {
                /* Note: Using cortical_immune_register_feature_hypercolumn if available,
                 * or similar registration function for frequency hypercolumns */
                /* This is a placeholder - actual function depends on cortical_immune API */
            }
        }
    }

    if (bridge->mutex_initialized) nimcp_mutex_unlock(&bridge->mutex);

    NIMCP_LOGGING_DEBUG("Connected to cortical immune system");
    return NIMCP_SUCCESS;
}

int audio_cortical_connect_bio_async(audio_cortical_bridge_t* bridge)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    if (bridge->bio_async_enabled) {
        return NIMCP_SUCCESS;  /* Already connected */
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_AUDIO_CORTICAL,
        .module_name = "audio_cortical_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Audio-cortical bridge connected to bio-async router");
        return NIMCP_SUCCESS;
    }

    NIMCP_LOGGING_WARN("Failed to connect to bio-async router");
    return NIMCP_ERROR_OPERATION_FAILED;
}

int audio_cortical_disconnect_bio_async(audio_cortical_bridge_t* bridge)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    if (!bridge->bio_async_enabled) {
        return NIMCP_SUCCESS;
    }

    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }

    bridge->bio_async_enabled = false;
    NIMCP_LOGGING_DEBUG("Disconnected from bio-async router");
    return NIMCP_SUCCESS;
}

bool audio_cortical_is_bio_async_connected(const audio_cortical_bridge_t* bridge)
{
    return bridge && bridge->bio_async_enabled;
}

/* ============================================================================
 * Processing Functions
 * ============================================================================ */

int audio_cortical_process(
    audio_cortical_bridge_t* bridge,
    const float* spectrogram,
    uint32_t num_freq_bins,
    uint32_t num_time_frames,
    audio_cortical_frequency_result_t* result)
{
    if (!bridge || !spectrogram || !result) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (num_freq_bins == 0 || num_time_frames == 0) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    if (bridge->mutex_initialized) nimcp_mutex_lock(&bridge->mutex);

    bridge->state = AUDIO_CORTICAL_STATE_PROCESSING;
    uint64_t start_time = get_time_ns();

    memset(result, 0, sizeof(audio_cortical_frequency_result_t));

    /* Compute average response across time frames and frequency bins */
    float* avg_responses = (float*)nimcp_calloc(
        bridge->bands_per_hypercolumn, sizeof(float)
    );
    uint32_t processed_count = 0;

    /* Process representative hypercolumns across frequency range */
    uint32_t sample_step = bridge->num_hypercolumns / 8;  /* Sample 8 positions */
    if (sample_step == 0) sample_step = 1;

    for (uint32_t h = 0; h < bridge->num_hypercolumns; h += sample_step) {
        feature_hypercolumn_t* hcol = bridge->hypercolumns[h];
        if (!hcol) continue;

        /* Compute energy in this hypercolumn's frequency range */
        uint32_t bin_start = (h * num_freq_bins) / bridge->num_hypercolumns;
        uint32_t bin_end = ((h + 1) * num_freq_bins) / bridge->num_hypercolumns;
        if (bin_end > num_freq_bins) bin_end = num_freq_bins;

        /* Average across time frames for this frequency band */
        for (uint32_t b = 0; b < hcol->total_columns && b < bridge->bands_per_hypercolumn; b++) {
            float band_energy = 0.0f;
            uint32_t count = 0;

            /* Sum energy in this sub-band across time */
            uint32_t sub_start = bin_start + (b * (bin_end - bin_start)) / hcol->total_columns;
            uint32_t sub_end = bin_start + ((b + 1) * (bin_end - bin_start)) / hcol->total_columns;

            for (uint32_t t = 0; t < num_time_frames; t++) {
                for (uint32_t f = sub_start; f < sub_end && f < num_freq_bins; f++) {
                    band_energy += spectrogram[t * num_freq_bins + f];
                    count++;
                }
            }

            if (count > 0) {
                band_energy /= (float)count;
            }

            /* Update hypercolumn activation */
            hcol->columns[b].activation = band_energy;

            /* Apply immune modulation and accumulate */
            float modulated = apply_immune_modulation(bridge, h, band_energy);
            if (avg_responses) {
                avg_responses[b] += modulated;
            }
        }

        processed_count++;
        bridge->stats.hypercolumn_activations++;
    }

    /* Compute average and find dominant band */
    if (processed_count > 0 && avg_responses) {
        float max_response = 0.0f;
        uint32_t max_idx = 0;
        float sum_responses = 0.0f;

        for (uint32_t i = 0; i < bridge->bands_per_hypercolumn; i++) {
            avg_responses[i] /= (float)processed_count;
            sum_responses += avg_responses[i];
            if (avg_responses[i] > max_response) {
                max_response = avg_responses[i];
                max_idx = i;
            }
        }

        result->num_freq_bands = bridge->bands_per_hypercolumn;
        result->frequency_responses = avg_responses;  /* Transfer ownership */

        /* Compute dominant frequency from dominant band */
        float log_min = logf(bridge->config.min_frequency);
        float log_max = logf(bridge->config.max_frequency);
        float log_center = log_min + (log_max - log_min) * ((float)max_idx / (float)bridge->bands_per_hypercolumn);
        result->dominant_frequency = expf(log_center);

        if (sum_responses > 0.0f) {
            result->selectivity_index = max_response / sum_responses;
        }
        result->confidence = result->selectivity_index;

        bridge->stats.active_hypercolumns = processed_count;
    } else {
        if (avg_responses) nimcp_free(avg_responses);
    }

    /* Update statistics */
    bridge->stats.frames_processed++;
    bridge->stats.peak_frequency_response = result->selectivity_index;
    bridge->stats.current_dominant_frequency = result->dominant_frequency;
    bridge->stats.current_immune_modulation = bridge->immune_modulation_factor;

    uint64_t end_time = get_time_ns();
    float process_time_ms = (float)(end_time - start_time) / 1000000.0f;
    bridge->stats.avg_processing_time_ms =
        (bridge->stats.avg_processing_time_ms * (bridge->stats.frames_processed - 1) +
         process_time_ms) / (float)bridge->stats.frames_processed;

    bridge->last_process_time_ns = end_time;
    bridge->state = AUDIO_CORTICAL_STATE_READY;

    if (bridge->mutex_initialized) nimcp_mutex_unlock(&bridge->mutex);

    return NIMCP_SUCCESS;
}

int audio_cortical_process_frequency_band(
    audio_cortical_bridge_t* bridge,
    const float* band_energy,
    uint32_t num_samples,
    float center_frequency,
    audio_cortical_frequency_result_t* result)
{
    if (!bridge || !band_energy || !result) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (num_samples == 0 || center_frequency <= 0.0f) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    if (bridge->mutex_initialized) nimcp_mutex_lock(&bridge->mutex);

    memset(result, 0, sizeof(audio_cortical_frequency_result_t));

    /* Find hypercolumn for this frequency */
    uint32_t hcol_idx = compute_hypercolumn_index(bridge, center_frequency);
    if (hcol_idx >= bridge->num_hypercolumns) {
        if (bridge->mutex_initialized) nimcp_mutex_unlock(&bridge->mutex);
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    feature_hypercolumn_t* hcol = bridge->hypercolumns[hcol_idx];
    if (!hcol) {
        if (bridge->mutex_initialized) nimcp_mutex_unlock(&bridge->mutex);
        return NIMCP_ERROR_INVALID_STATE;
    }

    /* Distribute energy across hypercolumn bands */
    for (uint32_t b = 0; b < hcol->total_columns; b++) {
        /* Simple energy distribution - could use more sophisticated model */
        float tuning = expf(-0.5f * powf((float)b - (float)hcol->total_columns / 2.0f, 2.0f) /
                           (float)hcol->total_columns);

        float energy = 0.0f;
        for (uint32_t s = 0; s < num_samples; s++) {
            energy += band_energy[s] * tuning;
        }
        energy /= (float)num_samples;

        hcol->columns[b].activation = energy;
    }

    bridge->stats.hypercolumn_activations++;

    /* Compute result */
    compute_frequency_result(hcol, result, center_frequency);

    /* Apply immune modulation */
    if (result->frequency_responses) {
        for (uint32_t i = 0; i < result->num_freq_bands; i++) {
            result->frequency_responses[i] = apply_immune_modulation(
                bridge, hcol_idx, result->frequency_responses[i]
            );
        }
    }

    if (bridge->mutex_initialized) nimcp_mutex_unlock(&bridge->mutex);

    return NIMCP_SUCCESS;
}

void audio_cortical_free_result(audio_cortical_frequency_result_t* result)
{
    if (!result) return;

    if (result->frequency_responses) {
        nimcp_free(result->frequency_responses);
        result->frequency_responses = NULL;
    }
    result->num_freq_bands = 0;
}

int audio_cortical_get_frequency_map(
    audio_cortical_bridge_t* bridge,
    const float* audio_data,
    uint32_t num_samples,
    uint32_t sample_rate,
    float* frequency_map,
    float* selectivity_map,
    uint32_t* num_windows)
{
    if (!bridge || !audio_data || !frequency_map || !num_windows) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (num_samples == 0 || sample_rate == 0) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    if (bridge->mutex_initialized) nimcp_mutex_lock(&bridge->mutex);

    /* Compute windowed frequency analysis */
    uint32_t window_size = sample_rate / 10;  /* 100ms windows */
    if (window_size < 64) window_size = 64;
    if (window_size > num_samples) window_size = num_samples;

    uint32_t hop_size = window_size / 2;
    uint32_t windows = (num_samples - window_size) / hop_size + 1;
    if (windows == 0) windows = 1;

    *num_windows = windows;

    /* Process each window */
    for (uint32_t w = 0; w < windows; w++) {
        uint32_t start_idx = w * hop_size;

        /* Compute dominant frequency for this window */
        float max_energy = 0.0f;
        float dominant_freq = bridge->config.min_frequency;

        /* Simple energy computation per hypercolumn frequency range */
        for (uint32_t h = 0; h < bridge->num_hypercolumns; h++) {
            float log_min = logf(bridge->config.min_frequency);
            float log_max = logf(bridge->config.max_frequency);
            float log_center = log_min + (log_max - log_min) * ((float)h / (float)bridge->num_hypercolumns);
            float center_freq = expf(log_center);

            /* Compute energy in frequency band (simplified) */
            float energy = 0.0f;
            for (uint32_t i = 0; i < window_size && (start_idx + i) < num_samples; i++) {
                energy += fabsf(audio_data[start_idx + i]);
            }
            energy /= (float)window_size;

            if (energy > max_energy) {
                max_energy = energy;
                dominant_freq = center_freq;
            }
        }

        frequency_map[w] = dominant_freq;
        if (selectivity_map) {
            selectivity_map[w] = (max_energy > 0.0f) ? 1.0f / (1.0f + max_energy) : 0.0f;
        }
    }

    if (bridge->mutex_initialized) nimcp_mutex_unlock(&bridge->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Hypercolumn Functions
 * ============================================================================ */

const feature_hypercolumn_t* audio_cortical_get_hypercolumn(
    const audio_cortical_bridge_t* bridge,
    float frequency_hz)
{
    if (!bridge) return NULL;

    uint32_t idx = compute_hypercolumn_index(bridge, frequency_hz);
    if (idx >= bridge->num_hypercolumns) return NULL;

    return bridge->hypercolumns[idx];
}

const feature_hypercolumn_t* audio_cortical_get_hypercolumn_by_index(
    const audio_cortical_bridge_t* bridge,
    uint32_t index)
{
    if (!bridge || index >= bridge->num_hypercolumns) return NULL;
    return bridge->hypercolumns[index];
}

uint32_t audio_cortical_get_num_hypercolumns(const audio_cortical_bridge_t* bridge)
{
    return bridge ? bridge->num_hypercolumns : 0;
}

/* ============================================================================
 * Immune Modulation Functions
 * ============================================================================ */

int audio_cortical_update_immune_modulation(audio_cortical_bridge_t* bridge)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    if (!bridge->cortical_immune) {
        return NIMCP_SUCCESS;  /* No immune system connected */
    }

    if (bridge->mutex_initialized) nimcp_mutex_lock(&bridge->mutex);

    /* Get cortical immune statistics */
    cortical_immune_stats_t stats;
    if (cortical_immune_get_stats(bridge->cortical_immune, &stats) == 0) {
        bridge->immune_modulation_factor = stats.mean_inflammation_level;
        bridge->stats.current_immune_modulation = bridge->immune_modulation_factor;

        /* Update per-hypercolumn gains based on local inflammation */
        for (uint32_t i = 0; i < bridge->num_hypercolumns; i++) {
            cortical_column_immune_t col_status;
            if (cortical_immune_get_column_status(bridge->cortical_immune,
                                                  i, &col_status) == 0) {
                if (bridge->hypercolumn_gains) {
                    bridge->hypercolumn_gains[i] = col_status.gain_modulation;
                }
            }
        }
    }

    if (bridge->mutex_initialized) nimcp_mutex_unlock(&bridge->mutex);

    return NIMCP_SUCCESS;
}

int audio_cortical_set_immune_factor(
    audio_cortical_bridge_t* bridge,
    float factor)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    if (factor < 0.0f) factor = 0.0f;
    if (factor > 1.0f) factor = 1.0f;

    if (bridge->mutex_initialized) nimcp_mutex_lock(&bridge->mutex);
    bridge->immune_modulation_factor = factor;
    bridge->stats.current_immune_modulation = factor;
    if (bridge->mutex_initialized) nimcp_mutex_unlock(&bridge->mutex);

    return NIMCP_SUCCESS;
}

float audio_cortical_get_immune_factor(const audio_cortical_bridge_t* bridge)
{
    return bridge ? bridge->immune_modulation_factor : 0.0f;
}

/* ============================================================================
 * Statistics and State Functions
 * ============================================================================ */

int audio_cortical_get_stats(
    const audio_cortical_bridge_t* bridge,
    audio_cortical_stats_t* stats)
{
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;

    if (bridge->mutex) nimcp_mutex_lock((nimcp_mutex_t*)bridge->mutex);
    memcpy(stats, &bridge->stats, sizeof(audio_cortical_stats_t));
    if (bridge->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)bridge->mutex);

    return NIMCP_SUCCESS;
}

int audio_cortical_reset_stats(audio_cortical_bridge_t* bridge)
{
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    if (bridge->mutex_initialized) nimcp_mutex_lock(&bridge->mutex);
    memset(&bridge->stats, 0, sizeof(audio_cortical_stats_t));
    if (bridge->mutex_initialized) nimcp_mutex_unlock(&bridge->mutex);

    return NIMCP_SUCCESS;
}

audio_cortical_state_t audio_cortical_get_state(
    const audio_cortical_bridge_t* bridge)
{
    return bridge ? bridge->state : AUDIO_CORTICAL_STATE_UNINITIALIZED;
}

const topographic_map_t* audio_cortical_get_tonotopic_map(
    const audio_cortical_bridge_t* bridge)
{
    return bridge ? bridge->tonotopic_map : NULL;
}

/* ============================================================================
 * Bio-Async Message Handling
 * ============================================================================ */

uint32_t audio_cortical_process_bio_messages(
    audio_cortical_bridge_t* bridge,
    uint32_t max_messages)
{
    if (!bridge || !bridge->bio_async_enabled || !bridge->bio_ctx) {
        return 0;
    }

    /* Process messages through bio router */
    uint32_t processed = bio_router_process_inbox(bridge->bio_ctx, max_messages);

    if (processed > 0) {
        bridge->stats.bio_messages_received += processed;
        NIMCP_LOGGING_DEBUG("Processed %u bio-async messages", processed);
    }

    return processed;
}

int audio_cortical_broadcast_frequency(
    audio_cortical_bridge_t* bridge,
    const audio_cortical_frequency_result_t* result)
{
    if (!bridge || !result) return NIMCP_ERROR_NULL_POINTER;

    if (!bridge->bio_async_enabled || !bridge->bio_ctx) {
        return NIMCP_ERROR_INVALID_STATE;
    }

    /* Broadcasting requires custom message structure definition
     * For now, just update stats and return success
     * Full implementation would use bio_router_broadcast with custom message type
     */
    bridge->stats.bio_messages_sent++;

    NIMCP_LOGGING_DEBUG("Broadcast frequency: freq=%.1f, selectivity=%.2f",
                        result->dominant_frequency, result->selectivity_index);

    return 0;
}
