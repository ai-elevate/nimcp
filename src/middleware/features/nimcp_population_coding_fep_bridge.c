/**
 * @file nimcp_population_coding_fep_bridge.c
 * @brief Free Energy Principle - Population Coding Integration Bridge
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Implementation of bidirectional population coding-FEP integration
 * WHY:  Population codes encode observations; FEP precision modulates tuning
 * HOW:  Precision → tuning width; synchrony → confidence; sparsity → efficiency
 */

#include "middleware/features/nimcp_population_coding_fep_bridge.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static inline float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static inline float map_precision_to_tuning(float precision) {
    /* High precision → narrow tuning */
    if (precision > 0.7f) return FEP_PRECISION_HIGH_TUNING_WIDTH;
    if (precision > 0.4f) return FEP_PRECISION_MED_TUNING_WIDTH;
    return FEP_PRECISION_LOW_TUNING_WIDTH;
}

static inline float map_synchrony_to_confidence(float synchrony) {
    /* High synchrony → high confidence */
    if (synchrony > SYNCHRONY_HIGH_CONFIDENCE_THRESHOLD) return 0.9f;
    if (synchrony > SYNCHRONY_MED_CONFIDENCE_THRESHOLD) return 0.6f;
    if (synchrony > SYNCHRONY_LOW_CONFIDENCE_THRESHOLD) return 0.3f;
    return 0.1f;
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int population_coding_fep_bridge_default_config(
    population_coding_fep_config_t* config
) {
    if (!config) return -1;

    config->enable_precision_tuning = true;
    config->enable_prediction_baseline = true;
    config->enable_synchrony_confidence = true;
    config->enable_sparsity_optimization = true;

    config->tuning_sensitivity = 1.0f;
    config->baseline_sensitivity = 1.0f;
    config->synchrony_sensitivity = 1.0f;
    config->sparsity_weight = 0.5f;

    config->min_tuning_width = 0.1f;
    config->max_tuning_width = 2.0f;
    config->min_synchrony_threshold = 0.1f;

    return 0;
}

population_coding_fep_bridge_t* population_coding_fep_bridge_create(
    const population_coding_fep_config_t* config
) {
    population_coding_fep_bridge_t* bridge = (population_coding_fep_bridge_t*)
        nimcp_calloc(1, sizeof(population_coding_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate population-FEP bridge");
        return NULL;
    }

    /* Apply configuration */
    population_coding_fep_config_t default_cfg;
    if (!config) {
        population_coding_fep_bridge_default_config(&default_cfg);
        config = &default_cfg;
    }
    bridge->config = *config;

    /* Initialize state */
    bridge->state.current_precision = 0.5f;
    bridge->state.current_prediction = 0.0f;
    bridge->state.prediction_error = 0.0f;
    bridge->state.population_synchrony = 0.0f;
    bridge->state.population_sparsity = OPTIMAL_SPARSITY_TARGET;
    bridge->state.effective_tuning_width = FEP_PRECISION_MED_TUNING_WIDTH;
    bridge->state.effective_baseline = 0.0f;

    /* Initialize effects */
    bridge->effects.tuning_width_modifier = 1.0f;
    bridge->effects.baseline_activation = 0.0f;
    bridge->effects.synchrony_threshold = config->min_synchrony_threshold;
    bridge->effects.precision_gain = 1.0f;
    bridge->effects.noise_correlation_reduction = 0.0f;

    /* Create mutex */
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        population_coding_fep_bridge_destroy(bridge);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Population-FEP bridge created");
    return bridge;
}

void population_coding_fep_bridge_destroy(
    population_coding_fep_bridge_t* bridge
) {
    if (!bridge) return;

    if (bridge->bio_async_enabled) {
        population_coding_fep_bridge_disconnect_bio_async(bridge);
    }

    if (bridge->mutex) {
        nimcp_platform_mutex_destroy(bridge->mutex);
        nimcp_free(bridge->mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Population-FEP bridge destroyed");
}

/* ============================================================================
 * Connection Implementation
 * ============================================================================ */

int population_coding_fep_bridge_connect_encoder(
    population_coding_fep_bridge_t* bridge,
    population_coding_encoder_t encoder
) {
    if (!bridge || !encoder) return -1;

    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->population_encoder = encoder;
    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_INFO("Population encoder connected to FEP bridge");
    return 0;
}

int population_coding_fep_bridge_connect_fep(
    population_coding_fep_bridge_t* bridge,
    fep_system_t* fep
) {
    if (!bridge || !fep) return -1;

    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_INFO("FEP system connected to population bridge");
    return 0;
}

int population_coding_fep_bridge_disconnect(
    population_coding_fep_bridge_t* bridge
) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->population_encoder = NULL;
    bridge->fep_system = NULL;
    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_INFO("Population-FEP bridge disconnected");
    return 0;
}

/* ============================================================================
 * FEP → Population Coding Implementation
 * ============================================================================ */

int population_coding_fep_apply_precision_tuning(
    population_coding_fep_bridge_t* bridge,
    float precision
) {
    if (!bridge) return -1;
    if (!bridge->config.enable_precision_tuning) return 0;

    nimcp_platform_mutex_lock(bridge->mutex);

    /* Map precision to tuning width */
    float base_tuning = map_precision_to_tuning(precision);
    float tuning_width = base_tuning * bridge->config.tuning_sensitivity;
    tuning_width = clamp_f(tuning_width,
                           bridge->config.min_tuning_width,
                           bridge->config.max_tuning_width);

    /* Update state */
    bridge->state.current_precision = precision;
    bridge->state.effective_tuning_width = tuning_width;
    bridge->effects.tuning_width_modifier = tuning_width / FEP_PRECISION_MED_TUNING_WIDTH;

    /* Precision also affects gain and noise correlation */
    bridge->effects.precision_gain = 0.5f + precision * 0.5f;
    bridge->effects.noise_correlation_reduction = precision * 0.3f;

    /* Update stats */
    bridge->stats.tuning_adjustments++;
    bridge->stats.avg_tuning_width =
        (bridge->stats.avg_tuning_width * (bridge->stats.tuning_adjustments - 1) +
         tuning_width) / bridge->stats.tuning_adjustments;
    bridge->stats.avg_precision =
        (bridge->stats.avg_precision * (bridge->stats.tuning_adjustments - 1) +
         precision) / bridge->stats.tuning_adjustments;

    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Precision tuning applied: %.3f → width %.3f",
                        precision, tuning_width);
    return 0;
}

int population_coding_fep_set_baseline(
    population_coding_fep_bridge_t* bridge,
    float prediction
) {
    if (!bridge) return -1;
    if (!bridge->config.enable_prediction_baseline) return 0;

    nimcp_platform_mutex_lock(bridge->mutex);

    /* Scale prediction to baseline activation */
    float baseline = prediction * bridge->config.baseline_sensitivity;
    baseline = clamp_f(baseline, 0.0f, MAX_BASELINE_ACTIVATION);

    /* Update state */
    bridge->state.current_prediction = prediction;
    bridge->state.effective_baseline = baseline;
    bridge->effects.baseline_activation = baseline;

    /* Update stats */
    if (baseline > 0.0f) {
        bridge->stats.baseline_activations++;
        bridge->stats.avg_baseline_level =
            (bridge->stats.avg_baseline_level * (bridge->stats.baseline_activations - 1) +
             baseline) / bridge->stats.baseline_activations;
    }

    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Baseline set from prediction: %.3f → %.3f",
                        prediction, baseline);
    return 0;
}

int population_coding_fep_adjust_synchrony_threshold(
    population_coding_fep_bridge_t* bridge,
    float prediction_error
) {
    if (!bridge) return -1;
    if (!bridge->config.enable_synchrony_confidence) return 0;

    nimcp_platform_mutex_lock(bridge->mutex);

    /* High PE increases threshold (require stronger synchrony) */
    float threshold = bridge->config.min_synchrony_threshold +
                     prediction_error * 0.3f;
    threshold = clamp_f(threshold, 0.1f, 0.9f);

    /* Update state */
    bridge->state.prediction_error = prediction_error;
    bridge->effects.synchrony_threshold = threshold;

    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Synchrony threshold adjusted: PE=%.3f → θ=%.3f",
                        prediction_error, threshold);
    return 0;
}

/* ============================================================================
 * Population Coding → FEP Implementation
 * ============================================================================ */

int population_coding_fep_report_observation(
    population_coding_fep_bridge_t* bridge,
    const vector3d_t* vector
) {
    if (!bridge || !vector) return -1;
    if (!bridge->fep_system) return -1;

    nimcp_platform_mutex_lock(bridge->mutex);

    /* Store population vector as observation */
    bridge->state.population_vector = *vector;

    /* TODO: Convert vector to FEP observation format and report */
    /* This would require FEP system API for observation updates */

    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Population vector reported: (%.3f, %.3f, %.3f)",
                        vector->x, vector->y, vector->z);
    return 0;
}

int population_coding_fep_update_precision_from_synchrony(
    population_coding_fep_bridge_t* bridge,
    float synchrony
) {
    if (!bridge) return -1;
    if (!bridge->config.enable_synchrony_confidence) return 0;
    if (!bridge->fep_system) return -1;

    nimcp_platform_mutex_lock(bridge->mutex);

    /* Map synchrony to confidence/precision */
    float confidence = map_synchrony_to_confidence(synchrony);
    confidence *= bridge->config.synchrony_sensitivity;
    confidence = clamp_f(confidence, 0.0f, 1.0f);

    /* Update state */
    bridge->state.population_synchrony = synchrony;

    /* Update stats */
    bridge->stats.synchrony_updates++;
    bridge->stats.avg_synchrony =
        (bridge->stats.avg_synchrony * (bridge->stats.synchrony_updates - 1) +
         synchrony) / bridge->stats.synchrony_updates;
    bridge->stats.avg_confidence =
        (bridge->stats.avg_confidence * (bridge->stats.synchrony_updates - 1) +
         confidence) / bridge->stats.synchrony_updates;

    /* TODO: Update FEP precision (requires FEP system API) */

    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Precision updated from synchrony: %.3f → %.3f",
                        synchrony, confidence);
    return 0;
}

int population_coding_fep_report_sparsity(
    population_coding_fep_bridge_t* bridge,
    float sparsity
) {
    if (!bridge) return -1;
    if (!bridge->config.enable_sparsity_optimization) return 0;

    nimcp_platform_mutex_lock(bridge->mutex);

    /* Calculate free energy bonus for sparse codes */
    float sparsity_bonus = 0.0f;
    if (sparsity <= OPTIMAL_SPARSITY_TARGET * 1.2f &&
        sparsity >= OPTIMAL_SPARSITY_TARGET * 0.8f) {
        sparsity_bonus = SPARSITY_FREE_ENERGY_BONUS * bridge->config.sparsity_weight;
    }

    /* Update state */
    bridge->state.population_sparsity = sparsity;

    /* Update stats */
    bridge->stats.avg_sparsity =
        (bridge->stats.avg_sparsity * bridge->stats.synchrony_updates + sparsity) /
        (bridge->stats.synchrony_updates + 1);
    bridge->stats.free_energy_savings += sparsity_bonus;

    /* TODO: Report sparsity bonus to FEP system */

    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Sparsity reported: %.3f (bonus: %.3f)",
                        sparsity, sparsity_bonus);
    return 0;
}

/* ============================================================================
 * Update Cycle Implementation
 * ============================================================================ */

int population_coding_fep_bridge_update(
    population_coding_fep_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) return -1;

    /* Update precision tuning if FEP system available */
    if (bridge->fep_system && bridge->config.enable_precision_tuning) {
        /* TODO: Query FEP precision and apply tuning */
    }

    /* Update baseline from predictions */
    if (bridge->fep_system && bridge->config.enable_prediction_baseline) {
        /* TODO: Query FEP predictions and set baseline */
    }

    /* Update synchrony-based confidence */
    if (bridge->population_encoder && bridge->config.enable_synchrony_confidence) {
        /* TODO: Query population synchrony and update precision */
    }

    return 0;
}

/* ============================================================================
 * State/Stats Implementation
 * ============================================================================ */

int population_coding_fep_bridge_get_state(
    const population_coding_fep_bridge_t* bridge,
    population_coding_fep_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_platform_mutex_lock((void*)bridge->mutex);
    *state = bridge->state;
    nimcp_platform_mutex_unlock((void*)bridge->mutex);

    return 0;
}

int population_coding_fep_bridge_get_stats(
    const population_coding_fep_bridge_t* bridge,
    population_coding_fep_stats_t* stats
) {
    if (!bridge || !stats) return -1;

    nimcp_platform_mutex_lock((void*)bridge->mutex);
    *stats = bridge->stats;
    nimcp_platform_mutex_unlock((void*)bridge->mutex);

    return 0;
}

int population_coding_fep_get_tuning_width(
    const population_coding_fep_bridge_t* bridge,
    float* tuning_width
) {
    if (!bridge || !tuning_width) return -1;

    nimcp_platform_mutex_lock((void*)bridge->mutex);
    *tuning_width = bridge->state.effective_tuning_width;
    nimcp_platform_mutex_unlock((void*)bridge->mutex);

    return 0;
}

int population_coding_fep_get_baseline(
    const population_coding_fep_bridge_t* bridge,
    float* baseline
) {
    if (!bridge || !baseline) return -1;

    nimcp_platform_mutex_lock((void*)bridge->mutex);
    *baseline = bridge->state.effective_baseline;
    nimcp_platform_mutex_unlock((void*)bridge->mutex);

    return 0;
}

/* ============================================================================
 * Bio-Async Implementation
 * ============================================================================ */

int population_coding_fep_bridge_connect_bio_async(
    population_coding_fep_bridge_t* bridge
) {
    if (!bridge) return -1;
    if (bridge->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_POPULATION_CODING_BRIDGE,
        .module_name = "population_coding_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
        return 0;
    }

    NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    return 0;
}

int population_coding_fep_bridge_disconnect_bio_async(
    population_coding_fep_bridge_t* bridge
) {
    if (!bridge) return -1;
    if (!bridge->bio_async_enabled) return 0;

    bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_async_enabled = false;

    NIMCP_LOGGING_INFO("Disconnected from bio-async router");
    return 0;
}

bool population_coding_fep_bridge_is_bio_async_connected(
    const population_coding_fep_bridge_t* bridge
) {
    if (!bridge) return false;
    return bridge->bio_async_enabled;
}
