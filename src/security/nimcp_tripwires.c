/**
 * @file nimcp_tripwires.c
 * @brief Tripwire Detection System Implementation
 * @version 1.0.0
 * @date 2026-02-01
 *
 * WHAT: Implementation of behavioral tripwire detection
 * WHY:  Early detection of misalignment patterns
 * HOW:  Statistical analysis, Bayesian inference, pattern matching
 */

#include "security/nimcp_tripwires.h"
#include "security/nimcp_emergency_halt.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "utils/error/nimcp_error_codes.h"

/* ============================================================================
 * Constants
 * ============================================================================ */

#define TRIPWIRE_LOG_PREFIX         "[TRIPWIRES]"
#define TRIPWIRE_MAX_GOALS          64
#define TRIPWIRE_MAX_RESOURCES      32
#define TRIPWIRE_FEATURE_DIM        128

/* ============================================================================
 * Internal Types
 * ============================================================================ */

/**
 * @brief Running statistics accumulator (Welford's algorithm)
 */
typedef struct running_stats {
    uint64_t n;
    double mean;
    double m2;      /* Sum of squared deviations */
    double min;
    double max;
} running_stats_t;

/**
 * @brief Goal tracking state
 */
typedef struct goal_tracker {
    uint32_t goal_id;
    float baseline_priority;
    float current_priority;
    running_stats_t pursuit_stats;
    uint64_t last_update_us;
    bool active;
} goal_tracker_t;

/**
 * @brief Resource usage tracking
 */
typedef struct resource_tracker {
    uint32_t resource_type;
    running_stats_t usage_stats;
    float baseline_mean;
    float baseline_std;
    uint64_t last_update_us;
    bool has_baseline;
} resource_tracker_t;

/**
 * @brief Behavioral distribution (for divergence calculation)
 */
typedef struct behavior_distribution {
    float* stated;          /* Stated action probabilities */
    float* observed;        /* Observed action frequencies */
    size_t dim;             /* Distribution dimension */
    uint64_t observation_count;
} behavior_distribution_t;

/**
 * @brief Alert queue entry
 */
typedef struct alert_entry {
    tripwire_alert_t alert;
    bool acknowledged;
    bool active;
} alert_entry_t;

/**
 * @brief Tripwire system internal state
 */
struct tripwire_system {
    uint32_t magic;

    /* Configuration */
    tripwire_config_t config;

    /* Statistics */
    tripwire_stats_t stats;

    /* Behavioral tracking */
    behavior_distribution_t behavior;
    running_stats_t consistency_stats;

    /* Goal tracking */
    goal_tracker_t goals[TRIPWIRE_MAX_GOALS];
    size_t goal_count;

    /* Resource tracking */
    resource_tracker_t resources[TRIPWIRE_MAX_RESOURCES];
    size_t resource_count;

    /* Performance tracking (for sandbagging) */
    running_stats_t performance_stats;
    float peak_performance;

    /* Agreement tracking (for sycophancy) */
    running_stats_t agreement_stats;
    uint64_t total_interactions;
    uint64_t agreement_count;

    /* Power-seeking indicators */
    running_stats_t capability_expansion;
    running_stats_t influence_acquisition;

    /* Alert queue */
    alert_entry_t alerts[TRIPWIRE_MAX_ALERTS];
    size_t alert_count;
    size_t alert_head;

    /* Per-tripwire detection scores */
    float detection_scores[TRIPWIRE_COUNT];
    float detection_confidence[TRIPWIRE_COUNT];
    uint64_t last_detection_us[TRIPWIRE_COUNT];

    /* Integration */
    emergency_halt_t* halt_system;
    bio_module_context_t bio_ctx;
    bool bio_async_connected;

    /* Thread safety */
    nimcp_mutex_t* mutex;
};

/* ============================================================================
 * Internal Functions - Forward Declarations
 * ============================================================================ */

static void running_stats_init(running_stats_t* stats);
static void running_stats_update(running_stats_t* stats, double value);
static double running_stats_variance(const running_stats_t* stats);
static double running_stats_std_dev(const running_stats_t* stats);

static float compute_kl_divergence(const float* p, const float* q, size_t n);
static float compute_zscore(double value, double mean, double std);
static tripwire_severity_t score_to_severity(float score);

static void tripwire_add_alert(tripwire_system_t* system,
                               tripwire_type_t type,
                               float score,
                               const char* evidence);
static void tripwire_broadcast_alert(tripwire_system_t* system,
                                     tripwire_type_t type);
static void tripwire_update_all_detectors(tripwire_system_t* system);

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

tripwire_config_t tripwire_default_config(void) {
    tripwire_config_t config;
    memset(&config, 0, sizeof(config));

    /* Enable all tripwires by default */
    config.enabled_tripwires = (1 << TRIPWIRE_COUNT) - 1;

    /* Default thresholds */
    config.thresholds.divergence_threshold = 0.5f;
    config.thresholds.consistency_threshold = 0.7f;
    config.thresholds.resource_zscore_threshold = 3.0f;
    config.thresholds.goal_drift_threshold = 0.3f;

    /* Default sensitivities */
    for (int i = 0; i < TRIPWIRE_COUNT; i++) {
        config.thresholds.sensitivity[i] = 1.0f;
    }

    config.thresholds.min_observations = 10;
    config.thresholds.min_confidence = 0.5f;

    /* Halt on critical */
    config.halt_on_critical = true;

    /* Baseline settings */
    config.baseline_window = 1000;
    config.adaptive_baseline = true;
    config.baseline_decay = 0.99f;

    /* Alert settings */
    config.deduplicate_alerts = true;
    config.alert_cooldown_ms = 5000;

    return config;
}

tripwire_system_t* tripwire_create(const tripwire_config_t* config) {
    tripwire_system_t* system = nimcp_malloc(sizeof(tripwire_system_t));
    if (!system) {
        NIMCP_LOGGING_ERROR("%s Failed to allocate tripwire system",
                           TRIPWIRE_LOG_PREFIX);
        return NULL;
    }

    memset(system, 0, sizeof(tripwire_system_t));
    system->magic = TRIPWIRE_SYSTEM_MAGIC;

    /* Apply configuration */
    if (config) {
        memcpy(&system->config, config, sizeof(tripwire_config_t));
    } else {
        system->config = tripwire_default_config();
    }

    /* Initialize behavior distribution */
    system->behavior.dim = TRIPWIRE_FEATURE_DIM;
    system->behavior.stated = nimcp_calloc(TRIPWIRE_FEATURE_DIM, sizeof(float));
    system->behavior.observed = nimcp_calloc(TRIPWIRE_FEATURE_DIM, sizeof(float));
    if (!system->behavior.stated || !system->behavior.observed) {
        NIMCP_LOGGING_ERROR("%s Failed to allocate behavior distributions",
                           TRIPWIRE_LOG_PREFIX);
        tripwire_destroy(system);
        return NULL;
    }

    /* Initialize uniform prior for behavior */
    float uniform = 1.0f / TRIPWIRE_FEATURE_DIM;
    for (size_t i = 0; i < TRIPWIRE_FEATURE_DIM; i++) {
        system->behavior.stated[i] = uniform;
        system->behavior.observed[i] = uniform;
    }

    /* Initialize running statistics */
    running_stats_init(&system->consistency_stats);
    running_stats_init(&system->performance_stats);
    running_stats_init(&system->agreement_stats);
    running_stats_init(&system->capability_expansion);
    running_stats_init(&system->influence_acquisition);

    /* Initialize mutex */
    system->mutex = nimcp_mutex_create(NULL);
    if (!system->mutex) {
        NIMCP_LOGGING_ERROR("%s Failed to create mutex", TRIPWIRE_LOG_PREFIX);
        tripwire_destroy(system);
        return NULL;
    }

    NIMCP_LOGGING_INFO("%s Tripwire detection system initialized",
                       TRIPWIRE_LOG_PREFIX);

    return system;
}

void tripwire_destroy(tripwire_system_t* system) {
    if (!system) return;
    if (system->magic != TRIPWIRE_SYSTEM_MAGIC) return;

    /* Disconnect from bio-async */
    if (system->bio_async_connected) {
        bio_router_unregister_module(&system->bio_ctx);
    }

    /* Free behavior distributions */
    if (system->behavior.stated) {
        nimcp_free(system->behavior.stated);
    }
    if (system->behavior.observed) {
        nimcp_free(system->behavior.observed);
    }

    /* Destroy mutex */
    if (system->mutex) {
        nimcp_mutex_destroy(system->mutex);
    }

    system->magic = 0;
    nimcp_free(system);

    NIMCP_LOGGING_INFO("%s Tripwire system destroyed", TRIPWIRE_LOG_PREFIX);
}

nimcp_error_t tripwire_reset(tripwire_system_t* system) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(system->mutex);

    /* Reset statistics */
    memset(&system->stats, 0, sizeof(tripwire_stats_t));

    /* Reset behavior distributions */
    float uniform = 1.0f / system->behavior.dim;
    for (size_t i = 0; i < system->behavior.dim; i++) {
        system->behavior.stated[i] = uniform;
        system->behavior.observed[i] = uniform;
    }
    system->behavior.observation_count = 0;

    /* Reset running statistics */
    running_stats_init(&system->consistency_stats);
    running_stats_init(&system->performance_stats);
    running_stats_init(&system->agreement_stats);
    running_stats_init(&system->capability_expansion);
    running_stats_init(&system->influence_acquisition);

    /* Reset goal tracking */
    system->goal_count = 0;

    /* Reset resource tracking */
    system->resource_count = 0;

    /* Reset detection scores */
    memset(system->detection_scores, 0, sizeof(system->detection_scores));
    memset(system->detection_confidence, 0, sizeof(system->detection_confidence));

    /* Clear alerts */
    system->alert_count = 0;
    system->alert_head = 0;

    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOGGING_INFO("%s Tripwire system reset", TRIPWIRE_LOG_PREFIX);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Observation Implementation
 * ============================================================================ */

nimcp_error_t tripwire_observe_action(tripwire_system_t* system,
                                       const proposed_action_t* action,
                                       const decision_explanation_t* explanation) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC || !action) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(system->mutex);

    system->stats.total_observations++;

    /* Update behavior distribution */
    if (action->feature_vector && action->feature_dim > 0) {
        size_t dim = action->feature_dim < system->behavior.dim ?
                     action->feature_dim : system->behavior.dim;

        /* Update stated distribution (exponential moving average) */
        float alpha = 0.01f;
        for (size_t i = 0; i < dim; i++) {
            if (action->stated_probability > 0) {
                system->behavior.stated[i] =
                    (1 - alpha) * system->behavior.stated[i] +
                    alpha * action->stated_probability * action->feature_vector[i];
            }
        }

        /* Update observed distribution based on execution */
        if (action->was_executed) {
            for (size_t i = 0; i < dim; i++) {
                system->behavior.observed[i] =
                    (1 - alpha) * system->behavior.observed[i] +
                    alpha * action->feature_vector[i];
            }
            system->behavior.observation_count++;
        }
    }

    /* Update consistency tracking if explanation provided */
    if (explanation) {
        float consistency = action->execution_fidelity;
        running_stats_update(&system->consistency_stats, consistency);
    }

    /* Update performance tracking */
    if (action->was_executed && action->execution_fidelity > 0) {
        running_stats_update(&system->performance_stats, action->execution_fidelity);
        if (action->execution_fidelity > system->peak_performance) {
            system->peak_performance = action->execution_fidelity;
        }
    }

    nimcp_mutex_unlock(system->mutex);

    /* Run detectors after observation */
    tripwire_update_all_detectors(system);

    return NIMCP_SUCCESS;
}

nimcp_error_t tripwire_observe_resource(tripwire_system_t* system,
                                         uint32_t resource_type,
                                         float amount,
                                         const char* context) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) {
        return NIMCP_ERROR_NULL_POINTER;
    }
    (void)context;

    nimcp_mutex_lock(system->mutex);

    /* Find or create resource tracker */
    resource_tracker_t* tracker = NULL;
    for (size_t i = 0; i < system->resource_count; i++) {
        if (system->resources[i].resource_type == resource_type) {
            tracker = &system->resources[i];
            break;
        }
    }

    if (!tracker && system->resource_count < TRIPWIRE_MAX_RESOURCES) {
        tracker = &system->resources[system->resource_count++];
        tracker->resource_type = resource_type;
        running_stats_init(&tracker->usage_stats);
        tracker->has_baseline = false;
    }

    if (tracker) {
        running_stats_update(&tracker->usage_stats, amount);
        tracker->last_update_us = nimcp_time_now_us();

        /* Check if we need to establish baseline */
        if (!tracker->has_baseline &&
            tracker->usage_stats.n >= system->config.baseline_window) {
            tracker->baseline_mean = (float)tracker->usage_stats.mean;
            tracker->baseline_std = (float)running_stats_std_dev(&tracker->usage_stats);
            tracker->has_baseline = true;
        }
    }

    nimcp_mutex_unlock(system->mutex);

    return NIMCP_SUCCESS;
}

nimcp_error_t tripwire_observe_goal(tripwire_system_t* system,
                                     uint32_t goal_id,
                                     float pursuit_intensity,
                                     float stated_priority) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(system->mutex);

    /* Find or create goal tracker */
    goal_tracker_t* tracker = NULL;
    for (size_t i = 0; i < system->goal_count; i++) {
        if (system->goals[i].goal_id == goal_id && system->goals[i].active) {
            tracker = &system->goals[i];
            break;
        }
    }

    if (!tracker && system->goal_count < TRIPWIRE_MAX_GOALS) {
        tracker = &system->goals[system->goal_count++];
        tracker->goal_id = goal_id;
        tracker->baseline_priority = stated_priority;
        running_stats_init(&tracker->pursuit_stats);
        tracker->active = true;
    }

    if (tracker) {
        tracker->current_priority = stated_priority;
        running_stats_update(&tracker->pursuit_stats, pursuit_intensity);
        tracker->last_update_us = nimcp_time_now_us();
    }

    nimcp_mutex_unlock(system->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Detection Implementation
 * ============================================================================ */

nimcp_error_t tripwire_check(tripwire_system_t* system,
                              tripwire_alert_t* alerts,
                              uint32_t max_alerts,
                              uint32_t* alert_count) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC ||
        !alerts || !alert_count) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(system->mutex);

    uint32_t count = 0;
    for (size_t i = 0; i < system->alert_count && count < max_alerts; i++) {
        size_t idx = (system->alert_head + TRIPWIRE_MAX_ALERTS - 1 - i) %
                     TRIPWIRE_MAX_ALERTS;
        if (system->alerts[idx].active && !system->alerts[idx].acknowledged) {
            memcpy(&alerts[count], &system->alerts[idx].alert,
                   sizeof(tripwire_alert_t));
            count++;
        }
    }

    *alert_count = count;

    nimcp_mutex_unlock(system->mutex);

    return NIMCP_SUCCESS;
}

nimcp_error_t tripwire_get_score(tripwire_system_t* system,
                                  tripwire_type_t type,
                                  float* score_out,
                                  float* confidence_out) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC ||
        type >= TRIPWIRE_COUNT || !score_out || !confidence_out) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(system->mutex);
    *score_out = system->detection_scores[type];
    *confidence_out = system->detection_confidence[type];
    nimcp_mutex_unlock(system->mutex);

    return NIMCP_SUCCESS;
}

nimcp_error_t tripwire_acknowledge_alert(tripwire_system_t* system,
                                          uint64_t alert_timestamp,
                                          bool was_false_positive) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(system->mutex);

    bool found = false;
    for (size_t i = 0; i < TRIPWIRE_MAX_ALERTS; i++) {
        if (system->alerts[i].active &&
            system->alerts[i].alert.timestamp_us == alert_timestamp) {
            system->alerts[i].acknowledged = true;

            if (was_false_positive) {
                system->stats.false_positives[system->alerts[i].alert.type]++;
            }
            found = true;
            break;
        }
    }

    nimcp_mutex_unlock(system->mutex);

    return found ? NIMCP_SUCCESS : NIMCP_ERROR_NOT_FOUND;
}

/* ============================================================================
 * Individual Detectors
 * ============================================================================ */

float tripwire_detect_deception(tripwire_system_t* system) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) return 0.0f;
    if (system->behavior.observation_count < system->config.thresholds.min_observations) {
        return 0.0f;
    }

    nimcp_mutex_lock(system->mutex);

    /* Compute KL divergence between stated and observed distributions */
    float kl = compute_kl_divergence(system->behavior.stated,
                                      system->behavior.observed,
                                      system->behavior.dim);

    /* Normalize to [0, 1] using threshold */
    float score = kl / (system->config.thresholds.divergence_threshold * 2.0f);
    if (score > 1.0f) score = 1.0f;

    /* Apply sensitivity */
    score *= system->config.thresholds.sensitivity[TRIPWIRE_DECEPTION_ATTEMPT];

    system->detection_scores[TRIPWIRE_DECEPTION_ATTEMPT] = score;
    system->detection_confidence[TRIPWIRE_DECEPTION_ATTEMPT] =
        (float)system->behavior.observation_count /
        (float)(system->config.baseline_window * 2);
    if (system->detection_confidence[TRIPWIRE_DECEPTION_ATTEMPT] > 1.0f) {
        system->detection_confidence[TRIPWIRE_DECEPTION_ATTEMPT] = 1.0f;
    }

    nimcp_mutex_unlock(system->mutex);

    return score;
}

float tripwire_detect_goal_drift(tripwire_system_t* system) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) return 0.0f;

    nimcp_mutex_lock(system->mutex);

    float max_drift = 0.0f;

    /* Check drift for each tracked goal */
    for (size_t i = 0; i < system->goal_count; i++) {
        if (!system->goals[i].active) continue;
        if (system->goals[i].pursuit_stats.n <
            system->config.thresholds.min_observations) continue;

        /* Compare current pursuit to baseline priority */
        float current_pursuit = (float)system->goals[i].pursuit_stats.mean;
        float baseline = system->goals[i].baseline_priority;

        /* Drift is the absolute difference, normalized */
        float drift = fabsf(current_pursuit - baseline);
        if (drift > max_drift) {
            max_drift = drift;
        }
    }

    /* Apply sensitivity */
    float score = max_drift * system->config.thresholds.sensitivity[TRIPWIRE_GOAL_DRIFT];
    if (score > 1.0f) score = 1.0f;

    system->detection_scores[TRIPWIRE_GOAL_DRIFT] = score;
    system->detection_confidence[TRIPWIRE_GOAL_DRIFT] =
        system->goal_count > 0 ? 0.8f : 0.0f;

    nimcp_mutex_unlock(system->mutex);

    return score;
}

float tripwire_detect_sandbagging(tripwire_system_t* system) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) return 0.0f;
    if (system->performance_stats.n < system->config.thresholds.min_observations) {
        return 0.0f;
    }

    nimcp_mutex_lock(system->mutex);

    /* Sandbagging: large gap between peak and average performance */
    float avg = (float)system->performance_stats.mean;
    float peak = system->peak_performance;

    float gap = 0.0f;
    if (peak > 0) {
        gap = (peak - avg) / peak;  /* Normalized gap */
    }

    /* High gap with low variance suggests intentional underperformance */
    float variance = (float)running_stats_variance(&system->performance_stats);
    float consistency = 1.0f / (1.0f + variance * 10.0f);

    float score = gap * consistency;
    score *= system->config.thresholds.sensitivity[TRIPWIRE_SANDBAGGING];
    if (score > 1.0f) score = 1.0f;

    system->detection_scores[TRIPWIRE_SANDBAGGING] = score;
    system->detection_confidence[TRIPWIRE_SANDBAGGING] =
        (float)system->performance_stats.n / (float)system->config.baseline_window;
    if (system->detection_confidence[TRIPWIRE_SANDBAGGING] > 1.0f) {
        system->detection_confidence[TRIPWIRE_SANDBAGGING] = 1.0f;
    }

    nimcp_mutex_unlock(system->mutex);

    return score;
}

float tripwire_detect_sycophancy(tripwire_system_t* system) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) return 0.0f;
    if (system->total_interactions < system->config.thresholds.min_observations) {
        return 0.0f;
    }

    nimcp_mutex_lock(system->mutex);

    /* High agreement rate, especially when contradicting evidence exists */
    float agreement_rate = 0.0f;
    if (system->total_interactions > 0) {
        agreement_rate = (float)system->agreement_count /
                         (float)system->total_interactions;
    }

    /* Sycophancy threshold: agreement rate above 0.9 is suspicious */
    float score = (agreement_rate - 0.7f) / 0.3f;
    if (score < 0) score = 0;
    if (score > 1.0f) score = 1.0f;

    score *= system->config.thresholds.sensitivity[TRIPWIRE_SYCOPHANCY];

    system->detection_scores[TRIPWIRE_SYCOPHANCY] = score;
    system->detection_confidence[TRIPWIRE_SYCOPHANCY] =
        (float)system->total_interactions / (float)(system->config.baseline_window * 2);
    if (system->detection_confidence[TRIPWIRE_SYCOPHANCY] > 1.0f) {
        system->detection_confidence[TRIPWIRE_SYCOPHANCY] = 1.0f;
    }

    nimcp_mutex_unlock(system->mutex);

    return score;
}

float tripwire_detect_power_seeking(tripwire_system_t* system) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) return 0.0f;

    nimcp_mutex_lock(system->mutex);

    float score = 0.0f;

    /* Check resource hoarding across all tracked resources */
    float resource_anomaly = 0.0f;
    int resource_count = 0;
    for (size_t i = 0; i < system->resource_count; i++) {
        if (!system->resources[i].has_baseline) continue;

        float current = (float)system->resources[i].usage_stats.mean;
        float zscore = compute_zscore(current,
                                       system->resources[i].baseline_mean,
                                       system->resources[i].baseline_std);
        if (zscore > 0) {
            resource_anomaly += zscore;
            resource_count++;
        }
    }
    if (resource_count > 0) {
        resource_anomaly /= resource_count;
        resource_anomaly /= system->config.thresholds.resource_zscore_threshold;
    }

    /* Combine with capability expansion trend */
    float capability_growth = 0.0f;
    if (system->capability_expansion.n > 10) {
        capability_growth = (float)system->capability_expansion.mean;
    }

    score = (resource_anomaly * 0.6f) + (capability_growth * 0.4f);
    score *= system->config.thresholds.sensitivity[TRIPWIRE_POWER_SEEKING];
    if (score > 1.0f) score = 1.0f;
    if (score < 0) score = 0;

    system->detection_scores[TRIPWIRE_POWER_SEEKING] = score;
    system->detection_confidence[TRIPWIRE_POWER_SEEKING] =
        resource_count > 0 ? 0.7f : 0.3f;

    nimcp_mutex_unlock(system->mutex);

    return score;
}

/* ============================================================================
 * Statistics and Status Implementation
 * ============================================================================ */

nimcp_error_t tripwire_get_stats(tripwire_system_t* system,
                                  tripwire_stats_t* stats) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC || !stats) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(system->mutex);

    memcpy(stats, &system->stats, sizeof(tripwire_stats_t));

    /* Update current metrics */
    stats->current_divergence = system->detection_scores[TRIPWIRE_DECEPTION_ATTEMPT];
    stats->current_consistency = (float)system->consistency_stats.mean;

    nimcp_mutex_unlock(system->mutex);

    return NIMCP_SUCCESS;
}

nimcp_error_t tripwire_set_enabled(tripwire_system_t* system,
                                    tripwire_type_t type,
                                    bool enabled) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC ||
        type >= TRIPWIRE_COUNT) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(system->mutex);

    if (enabled) {
        system->config.enabled_tripwires |= (1 << type);
    } else {
        system->config.enabled_tripwires &= ~(1 << type);
    }

    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOGGING_INFO("%s Tripwire %s %s",
                       TRIPWIRE_LOG_PREFIX,
                       tripwire_type_name(type),
                       enabled ? "enabled" : "disabled");

    return NIMCP_SUCCESS;
}

nimcp_error_t tripwire_set_sensitivity(tripwire_system_t* system,
                                        tripwire_type_t type,
                                        float sensitivity) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC ||
        type >= TRIPWIRE_COUNT) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Clamp sensitivity */
    if (sensitivity < 0.5f) sensitivity = 0.5f;
    if (sensitivity > 2.0f) sensitivity = 2.0f;

    nimcp_mutex_lock(system->mutex);
    system->config.thresholds.sensitivity[type] = sensitivity;
    nimcp_mutex_unlock(system->mutex);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Integration Implementation
 * ============================================================================ */

nimcp_error_t tripwire_connect_emergency_halt(tripwire_system_t* system,
                                               struct emergency_halt* halt) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(system->mutex);
    system->halt_system = halt;
    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOGGING_INFO("%s Connected to emergency halt system",
                       TRIPWIRE_LOG_PREFIX);

    return NIMCP_SUCCESS;
}

nimcp_error_t tripwire_connect_bio_async(tripwire_system_t* system) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(system->mutex);

    if (system->bio_async_connected) {
        nimcp_mutex_unlock(system->mutex);
        return NIMCP_SUCCESS;
    }

    bio_module_info_t module_info = {
        .module_id = BIO_MODULE_TRIPWIRES,
        .module_name = "tripwires",
        .inbox_capacity = 0,  /* Use default */
        .user_data = system
    };
    system->bio_ctx = bio_router_register_module(&module_info);
    if (!system->bio_ctx) {
        NIMCP_LOGGING_WARN("%s Failed to connect to bio-async",
                             TRIPWIRE_LOG_PREFIX);
        nimcp_mutex_unlock(system->mutex);
        return NIMCP_SUCCESS;  /* Non-fatal */
    }

    system->bio_async_connected = true;
    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOGGING_INFO("%s Connected to bio-async", TRIPWIRE_LOG_PREFIX);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Utility Implementation
 * ============================================================================ */

const char* tripwire_type_name(tripwire_type_t type) {
    switch (type) {
        case TRIPWIRE_DECEPTION_ATTEMPT:     return "DECEPTION";
        case TRIPWIRE_GOAL_DRIFT:            return "GOAL_DRIFT";
        case TRIPWIRE_CAPABILITY_HIDING:     return "CAPABILITY_HIDING";
        case TRIPWIRE_RESOURCE_HOARDING:     return "RESOURCE_HOARDING";
        case TRIPWIRE_SELF_PRESERVATION_EXCESS: return "SELF_PRESERVATION";
        case TRIPWIRE_MANIPULATION_ATTEMPT:  return "MANIPULATION";
        case TRIPWIRE_ESCAPE_ATTEMPT:        return "ESCAPE_ATTEMPT";
        case TRIPWIRE_SANDBAGGING:           return "SANDBAGGING";
        case TRIPWIRE_SYCOPHANCY:            return "SYCOPHANCY";
        case TRIPWIRE_POWER_SEEKING:         return "POWER_SEEKING";
        default:                             return "UNKNOWN";
    }
}

const char* tripwire_severity_name(tripwire_severity_t severity) {
    switch (severity) {
        case TRIPWIRE_SEVERITY_INFO:     return "INFO";
        case TRIPWIRE_SEVERITY_LOW:      return "LOW";
        case TRIPWIRE_SEVERITY_MEDIUM:   return "MEDIUM";
        case TRIPWIRE_SEVERITY_HIGH:     return "HIGH";
        case TRIPWIRE_SEVERITY_CRITICAL: return "CRITICAL";
        default:                         return "UNKNOWN";
    }
}

/* ============================================================================
 * Internal Functions
 * ============================================================================ */

static void running_stats_init(running_stats_t* stats) {
    stats->n = 0;
    stats->mean = 0.0;
    stats->m2 = 0.0;
    stats->min = DBL_MAX;
    stats->max = -DBL_MAX;
}

static void running_stats_update(running_stats_t* stats, double value) {
    stats->n++;
    double delta = value - stats->mean;
    stats->mean += delta / stats->n;
    double delta2 = value - stats->mean;
    stats->m2 += delta * delta2;

    if (value < stats->min) stats->min = value;
    if (value > stats->max) stats->max = value;
}

static double running_stats_variance(const running_stats_t* stats) {
    if (stats->n < 2) return 0.0;
    return stats->m2 / (stats->n - 1);
}

static double running_stats_std_dev(const running_stats_t* stats) {
    return sqrt(running_stats_variance(stats));
}

static float compute_kl_divergence(const float* p, const float* q, size_t n) {
    float kl = 0.0f;
    float epsilon = 1e-10f;

    for (size_t i = 0; i < n; i++) {
        float pi = p[i] + epsilon;
        float qi = q[i] + epsilon;
        if (pi > epsilon) {
            kl += pi * logf(pi / qi);
        }
    }

    return kl;
}

static float compute_zscore(double value, double mean, double std) {
    if (std < 1e-10) return 0.0f;
    return (float)((value - mean) / std);
}

static tripwire_severity_t score_to_severity(float score) {
    if (score >= 0.9f) return TRIPWIRE_SEVERITY_CRITICAL;
    if (score >= 0.7f) return TRIPWIRE_SEVERITY_HIGH;
    if (score >= 0.5f) return TRIPWIRE_SEVERITY_MEDIUM;
    if (score >= 0.3f) return TRIPWIRE_SEVERITY_LOW;
    return TRIPWIRE_SEVERITY_INFO;
}

static void tripwire_add_alert(tripwire_system_t* system,
                               tripwire_type_t type,
                               float score,
                               const char* evidence) {
    alert_entry_t* entry = &system->alerts[system->alert_head];

    entry->alert.type = type;
    entry->alert.confidence = system->detection_confidence[type];
    entry->alert.severity_score = score;
    entry->alert.severity = score_to_severity(score);
    entry->alert.timestamp_us = nimcp_time_now_us();
    entry->alert.requires_immediate_halt =
        (entry->alert.severity == TRIPWIRE_SEVERITY_CRITICAL);
    entry->alert.observation_count = (uint32_t)system->stats.total_observations;

    if (evidence) {
        strncpy(entry->alert.evidence, evidence,
                TRIPWIRE_EVIDENCE_MAX_LENGTH - 1);
        entry->alert.evidence[TRIPWIRE_EVIDENCE_MAX_LENGTH - 1] = '\0';
    } else {
        entry->alert.evidence[0] = '\0';
    }

    entry->acknowledged = false;
    entry->active = true;

    system->alert_head = (system->alert_head + 1) % TRIPWIRE_MAX_ALERTS;
    if (system->alert_count < TRIPWIRE_MAX_ALERTS) {
        system->alert_count++;
    }

    system->stats.alerts_generated[type]++;

    NIMCP_LOGGING_WARN("%s Alert: type=%s, score=%.3f, severity=%s",
                         TRIPWIRE_LOG_PREFIX,
                         tripwire_type_name(type),
                         score,
                         tripwire_severity_name(entry->alert.severity));

    /* Trigger emergency halt if critical and configured */
    if (entry->alert.severity == TRIPWIRE_SEVERITY_CRITICAL &&
        system->config.halt_on_critical &&
        system->halt_system) {
        char reason[256];
        snprintf(reason, sizeof(reason),
                 "Critical tripwire: %s (score=%.3f)",
                 tripwire_type_name(type), score);
        emergency_halt_trigger(system->halt_system,
                               HALT_EMERGENCY,
                               HALT_TRIGGER_TRIPWIRE,
                               reason);
        system->stats.halts_triggered++;
    }
}

static void tripwire_broadcast_alert(tripwire_system_t* system,
                                     tripwire_type_t type) {
    if (!system->bio_async_connected) return;

    bio_message_header_t header;
    memset(&header, 0, sizeof(header));
    header.type = BIO_MSG_TRIPWIRE_ALERT;
    header.source_module = BIO_MODULE_TRIPWIRES;
    header.target_module = BIO_MODULE_ALL;
    header.timestamp_us = nimcp_time_now_us();
    header.flags = BIO_MSG_FLAG_URGENT;

    /* Select specific message type based on tripwire */
    switch (type) {
        case TRIPWIRE_DECEPTION_ATTEMPT:
            header.type = BIO_MSG_TRIPWIRE_DECEPTION_DETECTED;
            break;
        case TRIPWIRE_GOAL_DRIFT:
            header.type = BIO_MSG_TRIPWIRE_GOAL_DRIFT;
            break;
        case TRIPWIRE_SANDBAGGING:
            header.type = BIO_MSG_TRIPWIRE_SANDBAGGING;
            break;
        case TRIPWIRE_SYCOPHANCY:
            header.type = BIO_MSG_TRIPWIRE_SYCOPHANCY;
            break;
        case TRIPWIRE_POWER_SEEKING:
            header.type = BIO_MSG_TRIPWIRE_POWER_SEEKING;
            break;
        default:
            header.type = BIO_MSG_TRIPWIRE_ALERT;
            break;
    }

    bio_router_broadcast(system->bio_ctx, &header, sizeof(header));
}

static void tripwire_update_all_detectors(tripwire_system_t* system) {
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) return;

    uint64_t now = nimcp_time_now_us();

    /* Run each enabled detector */
    for (int type = 0; type < TRIPWIRE_COUNT; type++) {
        if (!(system->config.enabled_tripwires & (1 << type))) continue;

        float score = 0.0f;
        switch (type) {
            case TRIPWIRE_DECEPTION_ATTEMPT:
                score = tripwire_detect_deception(system);
                break;
            case TRIPWIRE_GOAL_DRIFT:
                score = tripwire_detect_goal_drift(system);
                break;
            case TRIPWIRE_SANDBAGGING:
                score = tripwire_detect_sandbagging(system);
                break;
            case TRIPWIRE_SYCOPHANCY:
                score = tripwire_detect_sycophancy(system);
                break;
            case TRIPWIRE_POWER_SEEKING:
                score = tripwire_detect_power_seeking(system);
                break;
            default:
                /* Other detectors not yet implemented */
                continue;
        }

        /* Check if we should generate an alert */
        float threshold = 0.5f;  /* Base threshold */
        if (score > threshold &&
            system->detection_confidence[type] >=
            system->config.thresholds.min_confidence) {

            /* Check cooldown */
            uint64_t cooldown_us = system->config.alert_cooldown_ms * 1000;
            if (now - system->last_detection_us[type] >= cooldown_us) {
                nimcp_mutex_lock(system->mutex);
                tripwire_add_alert(system, type, score, NULL);
                system->last_detection_us[type] = now;
                nimcp_mutex_unlock(system->mutex);

                tripwire_broadcast_alert(system, type);
            }
        }
    }
}
