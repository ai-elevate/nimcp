/**
 * @file nimcp_introspection_substrate_bridge.c
 * @brief Bridge between neural substrate and introspection module
 *
 * WHAT: Bidirectional integration between neural substrate metabolic state
 *       and introspection capabilities (consciousness metrics, metacognition,
 *       self-awareness, uncertainty estimation)
 *
 * WHY: Introspection is metabolically expensive, requiring sustained prefrontal-
 *      medial cortex engagement. ATP depletion impairs self-awareness depth,
 *      metacognitive accuracy, and internal state monitoring.
 *
 * HOW: Monitors substrate ATP levels, fatigue state, and metabolic stress to
 *      compute effects on introspection parameters. Provides modulation of
 *      self-awareness depth, metacognitive accuracy, monitoring capacity, and
 *      uncertainty estimation based on metabolic availability.
 *
 * BIOLOGICAL BASIS:
 * - Introspection requires sustained activity in dorsomedial prefrontal cortex
 *   (dmPFC), anterior cingulate (ACC), posterior cingulate (PCC), and insula
 * - ATP depletion progressively impairs metacognitive precision and self-awareness
 * - Fatigue reduces monitoring capacity and uncertainty estimation quality
 *
 * @author NIMCP Development Team
 * @date 2024-12
 */

#include "cognitive/introspection/nimcp_introspection_substrate_bridge.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/validation/nimcp_common.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/memory/nimcp_memory.h"

#include <string.h>
#include <math.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * Clamp value to range [min, max]
 *
 * WHAT: Constrains value to specified bounds
 * WHY: Ensure biological effects stay within valid ranges
 * HOW: Return min if below, max if above, value if within
 */
static inline float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * Compute self-awareness depth from substrate state
 *
 * WHAT: Calculate depth of self-reflective processing capacity
 * WHY: Self-awareness requires sustained medial prefrontal engagement
 * HOW: Metabolic capacity * ATP level, clamped to [0.2, 1.0]
 *
 * BIOLOGICAL BASIS: dmPFC and PCC activity degrades with ATP depletion,
 *                   reducing introspective depth. Minimum 0.2 represents
 *                   baseline awareness even in depleted state.
 */
static float compute_self_awareness_depth(
    const substrate_metabolic_state_t* metabolic,
    float sensitivity
) {
    /* Base depth from metabolic capacity and ATP */
    float base_depth = metabolic->metabolic_capacity * metabolic->atp_level;

    /* Apply sensitivity factor */
    float depth = base_depth * sensitivity;

    /* Clamp to biological range [0.2, 1.0] */
    return clamp(depth, 0.2f, 1.0f);
}

/**
 * Compute metacognitive accuracy from substrate state
 *
 * WHAT: Calculate accuracy of self-assessment and performance monitoring
 * WHY: Accurate metacognition requires optimal metabolic and physical conditions
 * HOW: Average of metabolic and physical capacity, clamped to [0.3, 1.0]
 *
 * BIOLOGICAL BASIS: ACC (performance monitoring) is highly sensitive to
 *                   metabolic depletion and physical stress, impairing
 *                   metacognitive accuracy early in fatigue progression.
 */
static float compute_metacognitive_accuracy(
    const substrate_metabolic_state_t* metabolic,
    const substrate_physical_state_t* physical,
    float sensitivity
) {
    /* Combined metabolic and physical factors */
    float combined = (metabolic->metabolic_capacity + physical->physical_capacity) / 2.0f;

    /* Apply sensitivity factor */
    float accuracy = combined * sensitivity;

    /* Clamp to biological range [0.3, 1.0] */
    return clamp(accuracy, 0.3f, 1.0f);
}

/**
 * Compute monitoring capacity from ATP level
 *
 * WHAT: Calculate capacity for continuous internal state monitoring
 * WHY: Monitoring is metabolically demanding, directly dependent on ATP
 * HOW: ATP level with sensitivity factor, clamped to [0.3, 1.0]
 *
 * BIOLOGICAL BASIS: Insula (interoceptive awareness) and ACC require sustained
 *                   metabolic support. Capacity degrades proportionally with
 *                   ATP depletion, minimum 0.3 for basic awareness.
 */
static float compute_monitoring_capacity(
    const substrate_metabolic_state_t* metabolic,
    float sensitivity
) {
    /* Direct ATP-based capacity */
    float capacity = metabolic->atp_level * sensitivity;

    /* Clamp to biological range [0.3, 1.0] */
    return clamp(capacity, 0.3f, 1.0f);
}

/**
 * Compute uncertainty estimation quality
 *
 * WHAT: Calculate quality of epistemic uncertainty tracking
 * WHY: Uncertainty estimation requires cognitive resources for calibration
 * HOW: Metabolic capacity * 0.9, clamped to [0.3, 1.0]
 *
 * BIOLOGICAL BASIS: Fatigue impairs calibration of confidence judgments,
 *                   leading to overconfidence (low ATP) or underconfidence.
 *                   Uses 0.9 factor to model higher sensitivity to depletion.
 */
static float compute_uncertainty_estimation(
    const substrate_metabolic_state_t* metabolic,
    float sensitivity
) {
    /* Metabolic capacity with degradation factor */
    float estimation = metabolic->metabolic_capacity * 0.9f * sensitivity;

    /* Clamp to biological range [0.3, 1.0] */
    return clamp(estimation, 0.3f, 1.0f);
}

/**
 * Update running average
 *
 * WHAT: Compute exponential moving average
 * WHY: Smooth statistics over time for meaningful tracking
 * HOW: avg = alpha * new_value + (1 - alpha) * avg
 */
static float update_running_avg(float current_avg, float new_value, float alpha) {
    return alpha * new_value + (1.0f - alpha) * current_avg;
}

/* ============================================================================
 * API Implementation
 * ============================================================================ */

void introspection_substrate_default_config(introspection_substrate_config_t* config) {
    /* Guard: validate config pointer */
    if (!config) {
        NIMCP_LOGGING_ERROR("Cannot initialize default config: NULL pointer");
        return;
    }

    /* Feature enables */
    config->enable_atp_modulation = true;
    config->enable_fatigue_effects = true;
    config->enable_metabolic_monitoring = true;
    config->enable_bio_async = true;

    /* Sensitivity factors [0-1] */
    config->atp_sensitivity = 1.0f;
    config->fatigue_sensitivity = 0.8f;
    config->recovery_rate = 0.1f;

    /* Thresholds */
    config->impairment_threshold = 0.5f;
    config->critical_threshold = 0.3f;

    NIMCP_LOGGING_DEBUG("Initialized default introspection substrate config");
}

introspection_substrate_bridge_t* introspection_substrate_bridge_create(
    const introspection_substrate_config_t* config,
    neural_substrate_t* substrate,
    nimcp_introspection_t* introspection
) {
    /* Guard: validate component pointers */
    if (!substrate) {
        NIMCP_LOGGING_ERROR("Cannot create bridge: NULL substrate");
        return NULL;
    }

    if (!introspection) {
        NIMCP_LOGGING_ERROR("Cannot create bridge: NULL introspection module");
        return NULL;
    }

    /* Allocate bridge structure */
    introspection_substrate_bridge_t* bridge =
        (introspection_substrate_bridge_t*)nimcp_malloc(sizeof(introspection_substrate_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate introspection substrate bridge");
        return NULL;
    }

    /* Zero-initialize structure */
    memset(bridge, 0, sizeof(introspection_substrate_bridge_t));

    /* Set component pointers */
    bridge->substrate = substrate;
    bridge->introspection = introspection;

    /* Initialize configuration */
    if (config) {
        memcpy(&bridge->config, config, sizeof(introspection_substrate_config_t));
    } else {
        introspection_substrate_default_config(&bridge->config);
    }

    /* Initialize effects to baseline */
    bridge->effects.self_awareness_depth = 1.0f;
    bridge->effects.metacognitive_accuracy = 1.0f;
    bridge->effects.monitoring_capacity = 1.0f;
    bridge->effects.uncertainty_estimation = 1.0f;
    bridge->effects.is_impaired = false;

    /* Initialize statistics */
    bridge->stats.update_count = 0;
    bridge->stats.impairment_count = 0;
    bridge->stats.critical_count = 0;
    bridge->stats.min_atp_observed = 1.0f;
    bridge->stats.max_atp_observed = 0.0f;
    bridge->stats.avg_atp = 0.0f;
    bridge->stats.min_self_awareness = 1.0f;
    bridge->stats.max_self_awareness = 0.0f;
    bridge->stats.avg_metacognitive_accuracy = 0.0f;
    bridge->stats.recovery_count = 0;
    bridge->stats.avg_recovery_time = 0.0f;

    /* Initialize bio-async state */
    bridge->bio_async_enabled = false;
    bridge->bio_ctx = NULL;

    /* Create mutex for thread safety */
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex for introspection substrate bridge");
        nimcp_free(bridge);
        return NULL;
    }

    /* Connect to bio-async if enabled */
    if (bridge->config.enable_bio_async) {
        introspection_substrate_connect_bio_async(bridge);
    }

    NIMCP_LOGGING_INFO("Created introspection substrate bridge");
    return bridge;
}

void introspection_substrate_bridge_destroy(introspection_substrate_bridge_t* bridge) {
    /* Guard: NULL-safe */
    if (!bridge) {
        return;
    }

    /* Disconnect bio-async if connected */
    if (bridge->bio_async_enabled) {
        introspection_substrate_disconnect_bio_async(bridge);
    }

    /* Destroy mutex */
    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
        bridge->mutex = NULL;
    }

    /* Free bridge structure */
    nimcp_free(bridge);

    NIMCP_LOGGING_DEBUG("Destroyed introspection substrate bridge");
}

int introspection_substrate_connect_bio_async(introspection_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot connect bio-async: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Guard: check if already connected */
    if (bridge->bio_async_enabled) {
        NIMCP_LOGGING_DEBUG("Bio-async already connected");
        return NIMCP_SUCCESS;
    }

    /* Create module info for registration */
    bio_module_info_t info = {
        .module_id = BIO_MODULE_SUBSTRATE_INTROSPECTION,
        .module_name = "introspection_substrate_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    /* Register with bio-async router */
    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected introspection substrate bridge to bio-async router");
        return NIMCP_SUCCESS;
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
        return NIMCP_ERROR_NOT_FOUND;
    }
}

int introspection_substrate_disconnect_bio_async(introspection_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot disconnect bio-async: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Guard: check if connected */
    if (!bridge->bio_async_enabled) {
        return NIMCP_SUCCESS;
    }

    /* Unregister from bio-async router */
    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }

    bridge->bio_async_enabled = false;
    NIMCP_LOGGING_DEBUG("Disconnected introspection substrate bridge from bio-async");
    return NIMCP_SUCCESS;
}

bool introspection_substrate_is_bio_async_connected(
    const introspection_substrate_bridge_t* bridge
) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        return false;
    }

    return bridge->bio_async_enabled;
}

int introspection_substrate_update(introspection_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot update substrate effects: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Lock mutex for thread safety */
    nimcp_mutex_lock(bridge->mutex);

    /* Query substrate metabolic state */
    substrate_metabolic_state_t metabolic;
    int ret = substrate_get_metabolic_state(bridge->substrate, &metabolic);
    if (ret != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR("Failed to get substrate metabolic state");
        nimcp_mutex_unlock(bridge->mutex);
        return ret;
    }

    /* Query substrate physical state */
    substrate_physical_state_t physical;
    ret = substrate_get_physical_state(bridge->substrate, &physical);
    if (ret != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR("Failed to get substrate physical state");
        nimcp_mutex_unlock(bridge->mutex);
        return ret;
    }

    /* Store previous state for recovery tracking */
    bool was_impaired = bridge->effects.is_impaired;

    /* Compute substrate effects on introspection parameters */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.self_awareness_depth = compute_self_awareness_depth(
            &metabolic,
            bridge->config.atp_sensitivity
        );

        bridge->effects.monitoring_capacity = compute_monitoring_capacity(
            &metabolic,
            bridge->config.atp_sensitivity
        );
    }

    if (bridge->config.enable_fatigue_effects) {
        bridge->effects.metacognitive_accuracy = compute_metacognitive_accuracy(
            &metabolic,
            &physical,
            bridge->config.fatigue_sensitivity
        );
    }

    if (bridge->config.enable_metabolic_monitoring) {
        bridge->effects.uncertainty_estimation = compute_uncertainty_estimation(
            &metabolic,
            bridge->config.atp_sensitivity
        );
    }

    /* Determine impairment status */
    bridge->effects.is_impaired =
        (bridge->effects.self_awareness_depth < bridge->config.impairment_threshold) ||
        (bridge->effects.metacognitive_accuracy < 0.6f);

    /* Update statistics */
    bridge->stats.update_count++;

    /* Track ATP levels */
    if (metabolic.atp_level < bridge->stats.min_atp_observed) {
        bridge->stats.min_atp_observed = metabolic.atp_level;
    }
    if (metabolic.atp_level > bridge->stats.max_atp_observed) {
        bridge->stats.max_atp_observed = metabolic.atp_level;
    }
    bridge->stats.avg_atp = update_running_avg(
        bridge->stats.avg_atp,
        metabolic.atp_level,
        0.1f
    );

    /* Track self-awareness depth */
    if (bridge->effects.self_awareness_depth < bridge->stats.min_self_awareness) {
        bridge->stats.min_self_awareness = bridge->effects.self_awareness_depth;
    }
    if (bridge->effects.self_awareness_depth > bridge->stats.max_self_awareness) {
        bridge->stats.max_self_awareness = bridge->effects.self_awareness_depth;
    }

    /* Track metacognitive accuracy */
    bridge->stats.avg_metacognitive_accuracy = update_running_avg(
        bridge->stats.avg_metacognitive_accuracy,
        bridge->effects.metacognitive_accuracy,
        0.1f
    );

    /* Track impairment events */
    if (bridge->effects.is_impaired) {
        bridge->stats.impairment_count++;

        /* Check for critical state */
        if (bridge->effects.self_awareness_depth < bridge->config.critical_threshold ||
            bridge->effects.metacognitive_accuracy < bridge->config.critical_threshold) {
            bridge->stats.critical_count++;
        }
    }

    /* Track recovery events */
    if (was_impaired && !bridge->effects.is_impaired) {
        bridge->stats.recovery_count++;
    }

    /* Unlock mutex */
    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Updated introspection substrate effects: "
                       "self_awareness=%.3f, metacognitive=%.3f, monitoring=%.3f, "
                       "uncertainty=%.3f, impaired=%d",
                       bridge->effects.self_awareness_depth,
                       bridge->effects.metacognitive_accuracy,
                       bridge->effects.monitoring_capacity,
                       bridge->effects.uncertainty_estimation,
                       bridge->effects.is_impaired);

    return NIMCP_SUCCESS;
}

float introspection_substrate_get_self_awareness_depth(
    const introspection_substrate_bridge_t* bridge
) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get self-awareness depth: NULL bridge");
        return 0.0f;
    }

    /* Lock mutex for thread safety */
    nimcp_mutex_lock(bridge->mutex);
    float depth = bridge->effects.self_awareness_depth;
    nimcp_mutex_unlock(bridge->mutex);

    return depth;
}

float introspection_substrate_get_metacognitive_accuracy(
    const introspection_substrate_bridge_t* bridge
) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get metacognitive accuracy: NULL bridge");
        return 0.0f;
    }

    /* Lock mutex for thread safety */
    nimcp_mutex_lock(bridge->mutex);
    float accuracy = bridge->effects.metacognitive_accuracy;
    nimcp_mutex_unlock(bridge->mutex);

    return accuracy;
}

float introspection_substrate_get_monitoring_capacity(
    const introspection_substrate_bridge_t* bridge
) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get monitoring capacity: NULL bridge");
        return 0.0f;
    }

    /* Lock mutex for thread safety */
    nimcp_mutex_lock(bridge->mutex);
    float capacity = bridge->effects.monitoring_capacity;
    nimcp_mutex_unlock(bridge->mutex);

    return capacity;
}

float introspection_substrate_get_uncertainty_estimation(
    const introspection_substrate_bridge_t* bridge
) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get uncertainty estimation: NULL bridge");
        return 0.0f;
    }

    /* Lock mutex for thread safety */
    nimcp_mutex_lock(bridge->mutex);
    float estimation = bridge->effects.uncertainty_estimation;
    nimcp_mutex_unlock(bridge->mutex);

    return estimation;
}

int introspection_substrate_get_effects(
    const introspection_substrate_bridge_t* bridge,
    introspection_substrate_effects_t* effects
) {
    /* Guard: validate pointers */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get substrate effects: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!effects) {
        NIMCP_LOGGING_ERROR("Cannot get substrate effects: NULL effects buffer");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Lock mutex for thread safety */
    nimcp_mutex_lock(bridge->mutex);

    /* Copy effects structure */
    memcpy(effects, &bridge->effects, sizeof(introspection_substrate_effects_t));

    /* Unlock mutex */
    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}

bool introspection_substrate_is_impaired(const introspection_substrate_bridge_t* bridge) {
    /* Guard: validate bridge pointer */
    if (!bridge) {
        return false;
    }

    /* Lock mutex for thread safety */
    nimcp_mutex_lock(bridge->mutex);
    bool impaired = bridge->effects.is_impaired;
    nimcp_mutex_unlock(bridge->mutex);

    return impaired;
}

int introspection_substrate_get_stats(
    const introspection_substrate_bridge_t* bridge,
    introspection_substrate_stats_t* stats
) {
    /* Guard: validate pointers */
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Cannot get bridge statistics: NULL bridge");
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!stats) {
        NIMCP_LOGGING_ERROR("Cannot get bridge statistics: NULL stats buffer");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Lock mutex for thread safety */
    nimcp_mutex_lock(bridge->mutex);

    /* Copy statistics structure */
    memcpy(stats, &bridge->stats, sizeof(introspection_substrate_stats_t));

    /* Unlock mutex */
    nimcp_mutex_unlock(bridge->mutex);

    return NIMCP_SUCCESS;
}
