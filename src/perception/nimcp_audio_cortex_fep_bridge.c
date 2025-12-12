/**
 * @file nimcp_audio_cortex_fep_bridge.c
 * @brief Free Energy Principle - Audio Cortex Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-12
 */

#include "perception/nimcp_audio_cortex_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/error/nimcp_error_codes.h"

#include <string.h>
#include <math.h>

#define LOG_MODULE_AUDIO_FEP "[AUDIO_FEP]"

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

/**
 * WHAT: Provide default configuration for audio-FEP bridge
 * WHY:  Simplify initialization with reasonable defaults
 * HOW:  Set biologically-plausible thresholds and enable all features
 */
int audio_cortex_fep_bridge_default_config(audio_cortex_fep_config_t* config) {
    if (!config) {
        NIMCP_LOGGING_ERROR(LOG_MODULE_AUDIO_FEP " NULL config pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }

    config->prediction_error_threshold = AUDIO_FEP_PE_THRESHOLD_MEDIUM;
    config->precision_tuning_factor = AUDIO_FEP_PRECISION_TUNING_DEFAULT;
    config->temporal_prediction_weight = 0.5f;

    config->enable_temporal_predictions = true;
    config->enable_precision_tuning = true;
    config->enable_cocktail_party = true;
    config->enable_auditory_pe_updates = true;

    config->frequency_precision_sensitivity = 1.0f;
    config->temporal_prediction_sensitivity = 0.8f;
    config->pe_propagation_rate = 0.15f;

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Create audio cortex FEP bridge
 * WHY:  Initialize bidirectional integration between audio cortex and FEP
 * HOW:  Allocate memory, initialize state, create mutex
 */
audio_cortex_fep_bridge_t* audio_cortex_fep_bridge_create(
    const audio_cortex_fep_config_t* config
) {
    audio_cortex_fep_bridge_t* bridge = (audio_cortex_fep_bridge_t*)
        nimcp_malloc(sizeof(audio_cortex_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR(LOG_MODULE_AUDIO_FEP " Failed to allocate bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(audio_cortex_fep_bridge_t));

    /* Set configuration */
    if (config) {
        bridge->config = *config;
    } else {
        audio_cortex_fep_bridge_default_config(&bridge->config);
    }

    /* Create mutex */
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR(LOG_MODULE_AUDIO_FEP " Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->state.auditory_precision = 1.0f;
    bridge->effects.frequency_tuning_sharpness = 1.0f;
    bridge->effects.mel_filter_gain = 1.0f;
    bridge->effects.temporal_resolution_boost = 1.0f;
    bridge->effects.precision_tuning_modifier = 1.0f;

    /* Initialize per-frequency precision */
    for (int i = 0; i < 32; i++) {
        bridge->state.frequency_precision[i] = 1.0f;
    }

    NIMCP_LOGGING_INFO(LOG_MODULE_AUDIO_FEP " Bridge created successfully");
    return bridge;
}

/**
 * WHAT: Destroy audio cortex FEP bridge
 * WHY:  Free all allocated resources
 * HOW:  Disconnect bio-async, destroy mutex, free memory
 */
void audio_cortex_fep_bridge_destroy(audio_cortex_fep_bridge_t* bridge) {
    if (!bridge) return;

    /* Disconnect bio-async if connected */
    if (bridge->bio_async_enabled) {
        audio_cortex_fep_bridge_disconnect_bio_async(bridge);
    }

    /* Destroy mutex */
    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO(LOG_MODULE_AUDIO_FEP " Bridge destroyed");
}

/* ============================================================================
 * Connection Implementation
 * ============================================================================ */

/**
 * WHAT: Connect FEP system to bridge
 * WHY:  Enable FEP state queries and updates
 * HOW:  Store FEP pointer with mutex protection
 */
int audio_cortex_fep_bridge_connect_fep(
    audio_cortex_fep_bridge_t* bridge,
    fep_system_t* fep
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!fep) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_INFO(LOG_MODULE_AUDIO_FEP " FEP system connected");
    return NIMCP_SUCCESS;
}

/**
 * WHAT: Connect audio cortex to bridge
 * WHY:  Enable auditory processing monitoring and modulation
 * HOW:  Store audio cortex pointer with mutex protection
 */
int audio_cortex_fep_bridge_connect_audio_cortex(
    audio_cortex_fep_bridge_t* bridge,
    audio_cortex_t* audio
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!audio) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);
    bridge->audio_cortex = audio;
    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_INFO(LOG_MODULE_AUDIO_FEP " Audio cortex connected");
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * FEP → Audio Implementation
 * ============================================================================ */

/**
 * WHAT: Apply FEP temporal predictions to auditory processing
 * WHY:  Top-down predictions modulate frequency tuning
 * HOW:  Query FEP temporal model, convert to mel filter gain
 */
int audio_cortex_fep_apply_temporal_predictions(
    audio_cortex_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->fep_system) return NIMCP_ERROR_INVALID_STATE;
    if (!bridge->config.enable_temporal_predictions) return NIMCP_SUCCESS;

    nimcp_mutex_lock(bridge->mutex);

    /* Compute temporal prediction confidence (simplified) */
    float prediction_confidence = bridge->state.temporal_prediction_accuracy;

    /* Modulate mel filter gain */
    bridge->effects.mel_filter_gain = 1.0f +
        (prediction_confidence * bridge->config.temporal_prediction_weight);

    /* Predictive suppression of expected sounds */
    bridge->effects.prediction_suppression = prediction_confidence * 0.2f;

    /* Enhancement of novel sounds */
    bridge->effects.novelty_enhancement = (1.0f - prediction_confidence) * 0.4f;

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/**
 * WHAT: Apply FEP precision to frequency tuning (cocktail party effect)
 * WHY:  Precision sharpens tuning for attended frequencies
 * HOW:  Modulate mel filterbank sharpness by precision
 */
int audio_cortex_fep_apply_precision_tuning(
    audio_cortex_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->fep_system) return NIMCP_ERROR_INVALID_STATE;
    if (!bridge->config.enable_precision_tuning) return NIMCP_SUCCESS;

    nimcp_mutex_lock(bridge->mutex);

    /* Get precision from FEP (placeholder) */
    float fep_precision = 1.0f;

    /* Convert precision to tuning sharpness */
    bridge->effects.frequency_tuning_sharpness =
        AUDIO_FEP_PRECISION_TUNING_DEFAULT +
        (fep_precision * bridge->config.precision_tuning_factor);

    /* Clamp to valid range */
    if (bridge->effects.frequency_tuning_sharpness > AUDIO_FEP_PRECISION_TUNING_MAX) {
        bridge->effects.frequency_tuning_sharpness = AUDIO_FEP_PRECISION_TUNING_MAX;
    }
    if (bridge->effects.frequency_tuning_sharpness < AUDIO_FEP_PRECISION_TUNING_MIN) {
        bridge->effects.frequency_tuning_sharpness = AUDIO_FEP_PRECISION_TUNING_MIN;
    }

    bridge->state.auditory_precision = fep_precision;
    bridge->effects.precision_tuning_modifier = fep_precision *
        bridge->config.frequency_precision_sensitivity;

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Audio → FEP Implementation
 * ============================================================================ */

/**
 * WHAT: Compute auditory prediction error (MMN response)
 * WHY:  PE drives belief updates in FEP
 * HOW:  Compare audio features to FEP temporal predictions
 */
int audio_cortex_fep_compute_prediction_error(
    audio_cortex_fep_bridge_t* bridge,
    const float* audio_features,
    uint32_t num_features,
    float* prediction_error
) {
    if (!bridge || !audio_features || !prediction_error) {
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (!bridge->fep_system) return NIMCP_ERROR_INVALID_STATE;
    if (num_features == 0) return NIMCP_ERROR_INVALID_PARAMETER;

    nimcp_mutex_lock(bridge->mutex);

    /* Simplified PE computation (would compare to FEP temporal predictions) */
    float pe_sum = 0.0f;
    for (uint32_t i = 0; i < num_features; i++) {
        float expected = 0.0f; /* Placeholder: get from FEP temporal model */
        float error = audio_features[i] - expected;
        pe_sum += error * error;
    }

    *prediction_error = sqrtf(pe_sum / (float)num_features);

    /* Update state */
    bridge->state.current_auditory_pe = *prediction_error;
    bridge->state.avg_auditory_pe =
        0.9f * bridge->state.avg_auditory_pe + 0.1f * (*prediction_error);

    if (*prediction_error > bridge->state.max_auditory_pe) {
        bridge->state.max_auditory_pe = *prediction_error;
    }

    /* Check for MMN events (high PE) */
    if (*prediction_error > bridge->config.prediction_error_threshold) {
        bridge->state.mmn_events++;
        bridge->stats.high_pe_events++;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/**
 * WHAT: Report auditory observations to FEP
 * WHY:  Audio features drive temporal inference
 * HOW:  Convert mel features to FEP observation format
 */
int audio_cortex_fep_report_observations(
    audio_cortex_fep_bridge_t* bridge,
    const float* audio_features,
    uint32_t num_features
) {
    if (!bridge || !audio_features) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->fep_system) return NIMCP_ERROR_INVALID_STATE;
    if (!bridge->config.enable_auditory_pe_updates) return NIMCP_SUCCESS;
    if (num_features == 0) return NIMCP_ERROR_INVALID_PARAMETER;

    nimcp_mutex_lock(bridge->mutex);

    /* Report to FEP system (would call FEP observation API) */
    bridge->state.frames_processed++;
    bridge->stats.total_frames_processed++;

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/**
 * WHAT: Report temporal events to FEP
 * WHY:  Onsets/offsets structure temporal predictions
 * HOW:  Signal event boundaries to FEP temporal model
 */
int audio_cortex_fep_report_temporal_events(
    audio_cortex_fep_bridge_t* bridge,
    bool onset_detected,
    bool offset_detected
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->fep_system) return NIMCP_ERROR_INVALID_STATE;

    nimcp_mutex_lock(bridge->mutex);

    /* Report temporal events to FEP (placeholder) */
    if (onset_detected || offset_detected) {
        /* Update temporal prediction model */
    }

    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Update Implementation
 * ============================================================================ */

/**
 * WHAT: Update audio-FEP bridge state
 * WHY:  Maintain bidirectional coupling
 * HOW:  Apply predictions, compute PE, update statistics
 */
int audio_cortex_fep_bridge_update(
    audio_cortex_fep_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    /* Apply FEP effects to auditory processing */
    audio_cortex_fep_apply_temporal_predictions(bridge);
    audio_cortex_fep_apply_precision_tuning(bridge);

    /* Update statistics */
    nimcp_mutex_lock(bridge->mutex);

    bridge->stats.avg_prediction_error = bridge->state.avg_auditory_pe;
    bridge->stats.temporal_prediction_accuracy =
        bridge->state.temporal_prediction_accuracy;

    /* Compute average frequency precision */
    float total_precision = 0.0f;
    for (int i = 0; i < 32; i++) {
        total_precision += bridge->state.frequency_precision[i];
    }
    bridge->stats.avg_frequency_precision = total_precision / 32.0f;

    bridge->stats.avg_tuning_sharpness = bridge->effects.frequency_tuning_sharpness;
    bridge->stats.avg_prediction_suppression = bridge->effects.prediction_suppression;

    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * State/Stats Implementation
 * ============================================================================ */

/**
 * WHAT: Get current bridge state
 * WHY:  Monitor audio-FEP interaction
 * HOW:  Copy state with mutex protection
 */
int audio_cortex_fep_bridge_get_state(
    const audio_cortex_fep_bridge_t* bridge,
    audio_cortex_fep_state_t* state
) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Get bridge statistics
 * WHY:  Monitor performance and effects
 * HOW:  Copy stats with mutex protection
 */
int audio_cortex_fep_bridge_get_stats(
    const audio_cortex_fep_bridge_t* bridge,
    audio_cortex_fep_stats_t* stats
) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async Implementation
 * ============================================================================ */

/**
 * WHAT: Connect to bio-async router
 * WHY:  Enable distributed audio-FEP messaging
 * HOW:  Register module with BIO_MODULE_FEP_AUDIO_CORTEX_BRIDGE ID
 */
int audio_cortex_fep_bridge_connect_bio_async(
    audio_cortex_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return NIMCP_SUCCESS;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_AUDIO_CORTEX_BRIDGE,
        .module_name = "audio_cortex_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO(LOG_MODULE_AUDIO_FEP " Connected to bio-async router");
        return NIMCP_SUCCESS;
    }

    NIMCP_LOGGING_WARN(LOG_MODULE_AUDIO_FEP " Bio-async router not available");
    return -1;
}

/**
 * WHAT: Disconnect from bio-async router
 * WHY:  Clean shutdown
 * HOW:  Unregister module
 */
int audio_cortex_fep_bridge_disconnect_bio_async(
    audio_cortex_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->bio_async_enabled) return NIMCP_SUCCESS;

    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }

    bridge->bio_async_enabled = false;
    NIMCP_LOGGING_INFO(LOG_MODULE_AUDIO_FEP " Disconnected from bio-async");
    return NIMCP_SUCCESS;
}

/**
 * WHAT: Check bio-async connection status
 * WHY:  Query if messaging is available
 * HOW:  Return bio_async_enabled flag
 */
bool audio_cortex_fep_bridge_is_bio_async_connected(
    const audio_cortex_fep_bridge_t* bridge
) {
    return bridge ? bridge->bio_async_enabled : false;
}
