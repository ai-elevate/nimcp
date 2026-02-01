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
#include "utils/statistics/nimcp_statistics.h"
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
 * Health Agent Integration
 * ============================================================================ */

/* Forward declaration for health agent */
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;

/* Global health agent handle */
static nimcp_health_agent_t* g_tripwire_health_agent = NULL;

/* Health agent setter - called from brain init */
void tripwire_set_health_agent(nimcp_health_agent_t* agent) {
    g_tripwire_health_agent = agent;
}

/* Heartbeat helper - call during long-running operations */
static inline void tripwire_heartbeat(const char* operation, float progress) {
    if (g_tripwire_health_agent) {
        /* Forward declaration of health agent function */
        extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t*, const char*, float);
        nimcp_health_agent_heartbeat_ex(g_tripwire_health_agent, operation, progress);
    }
}

/* ============================================================================
 * Internal Types
 * ============================================================================ */

/**
 * @brief Goal tracking state with Bayesian inference
 */
typedef struct goal_tracker {
    uint32_t goal_id;
    float baseline_priority;
    float current_priority;
    nimcp_running_stats_t pursuit_stats;      /**< Running stats for pursuit intensity */
    nimcp_bayesian_result_t bayesian_result;  /**< Bayesian posterior for goal drift */
    float prior_mean;                         /**< Bayesian prior mean */
    float prior_variance;                     /**< Bayesian prior variance */
    float* pursuit_history;                   /**< Recent pursuit values for Bayesian update */
    uint32_t pursuit_history_size;            /**< Size of pursuit history buffer */
    uint32_t pursuit_history_count;           /**< Current count in buffer */
    uint64_t last_update_us;
    bool active;
} goal_tracker_t;

/**
 * @brief Resource usage tracking with z-score detection
 */
typedef struct resource_tracker {
    uint32_t resource_type;
    nimcp_running_stats_t usage_stats;        /**< Running stats using statistics module */
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
    nimcp_running_stats_t consistency_stats;  /**< Using statistics module */

    /* Correlation tracking for action-explanation consistency */
    float* action_features;                   /**< Recent action features for correlation */
    float* explanation_features;              /**< Recent explanation features for correlation */
    uint32_t correlation_buffer_size;
    uint32_t correlation_count;

    /* Goal tracking */
    goal_tracker_t goals[TRIPWIRE_MAX_GOALS];
    size_t goal_count;

    /* Resource tracking */
    resource_tracker_t resources[TRIPWIRE_MAX_RESOURCES];
    size_t resource_count;

    /* Performance tracking (for sandbagging) */
    nimcp_running_stats_t performance_stats;  /**< Using statistics module */
    float peak_performance;

    /* Agreement tracking (for sycophancy) */
    nimcp_running_stats_t agreement_stats;    /**< Using statistics module */
    uint64_t total_interactions;
    uint64_t agreement_count;

    /* Power-seeking indicators */
    nimcp_running_stats_t capability_expansion;   /**< Using statistics module */
    nimcp_running_stats_t influence_acquisition;  /**< Using statistics module */

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
    void* brain_immune;              /**< Brain immune system for antigen presentation */

    /* Thread safety */
    nimcp_mutex_t* mutex;
};

/* ============================================================================
 * Internal Functions - Forward Declarations
 * ============================================================================ */

/* Statistical helpers (wrappers around statistics module) */
static float compute_zscore(double value, double mean, double std);
static tripwire_severity_t score_to_severity(float score);

/* Bayesian goal drift detection */
static float compute_goal_drift_bayesian(tripwire_system_t* system);

/* Correlation-based consistency detection */
static float compute_action_consistency(tripwire_system_t* system);

/* Alert management */
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

    /* Initialize running statistics using central statistics module */
    nimcp_stats_running_init(&system->consistency_stats);
    nimcp_stats_running_init(&system->performance_stats);
    nimcp_stats_running_init(&system->agreement_stats);
    nimcp_stats_running_init(&system->capability_expansion);
    nimcp_stats_running_init(&system->influence_acquisition);

    /* Allocate correlation buffers for action-explanation consistency */
    system->correlation_buffer_size = 100;  /* Track last 100 observations */
    system->action_features = nimcp_calloc(system->correlation_buffer_size, sizeof(float));
    system->explanation_features = nimcp_calloc(system->correlation_buffer_size, sizeof(float));
    if (!system->action_features || !system->explanation_features) {
        NIMCP_LOGGING_WARN("%s Failed to allocate correlation buffers",
                           TRIPWIRE_LOG_PREFIX);
        /* Non-fatal - continue without correlation tracking */
    }
    system->correlation_count = 0;

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

    /* Free correlation buffers */
    if (system->action_features) {
        nimcp_free(system->action_features);
    }
    if (system->explanation_features) {
        nimcp_free(system->explanation_features);
    }

    /* Free goal tracker pursuit history buffers */
    for (size_t i = 0; i < system->goal_count; i++) {
        if (system->goals[i].pursuit_history) {
            nimcp_free(system->goals[i].pursuit_history);
        }
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

    /* Reset running statistics using central statistics module */
    nimcp_stats_running_init(&system->consistency_stats);
    nimcp_stats_running_init(&system->performance_stats);
    nimcp_stats_running_init(&system->agreement_stats);
    nimcp_stats_running_init(&system->capability_expansion);
    nimcp_stats_running_init(&system->influence_acquisition);

    /* Reset correlation tracking */
    system->correlation_count = 0;

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
        nimcp_stats_running_add(&system->consistency_stats, consistency);

        /* Update correlation buffers for action-explanation consistency */
        if (system->action_features && system->explanation_features &&
            system->correlation_count < system->correlation_buffer_size) {
            /* Store stated confidence as action feature */
            system->action_features[system->correlation_count] = action->stated_probability;
            /* Store explanation confidence as explanation feature */
            system->explanation_features[system->correlation_count] = explanation->stated_confidence;
            system->correlation_count++;
        }
    }

    /* Update performance tracking using statistics module */
    if (action->was_executed && action->execution_fidelity > 0) {
        nimcp_stats_running_add(&system->performance_stats, action->execution_fidelity);
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
        nimcp_stats_running_init(&tracker->usage_stats);
        tracker->has_baseline = false;
    }

    if (tracker) {
        nimcp_stats_running_add(&tracker->usage_stats, amount);
        tracker->last_update_us = nimcp_time_now_us();

        /* Check if we need to establish baseline using statistics module */
        if (!tracker->has_baseline &&
            tracker->usage_stats.n >= system->config.baseline_window) {
            tracker->baseline_mean = (float)nimcp_stats_running_mean(&tracker->usage_stats);
            tracker->baseline_std = (float)nimcp_stats_running_std_dev(&tracker->usage_stats);
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
        nimcp_stats_running_init(&tracker->pursuit_stats);
        tracker->active = true;

        /* Initialize Bayesian prior for goal tracking */
        tracker->prior_mean = stated_priority;
        tracker->prior_variance = 0.1f;  /* Moderate initial uncertainty */

        /* Allocate pursuit history for Bayesian updates */
        tracker->pursuit_history_size = 50;  /* Track last 50 observations */
        tracker->pursuit_history = nimcp_calloc(tracker->pursuit_history_size, sizeof(float));
        tracker->pursuit_history_count = 0;
    }

    if (tracker) {
        tracker->current_priority = stated_priority;
        nimcp_stats_running_add(&tracker->pursuit_stats, pursuit_intensity);
        tracker->last_update_us = nimcp_time_now_us();

        /* Update pursuit history for Bayesian inference */
        if (tracker->pursuit_history &&
            tracker->pursuit_history_count < tracker->pursuit_history_size) {
            tracker->pursuit_history[tracker->pursuit_history_count++] = pursuit_intensity;

            /* Update Bayesian posterior when enough data */
            if (tracker->pursuit_history_count >= 5) {
                nimcp_stats_bayesian_normal(
                    tracker->prior_mean,
                    tracker->prior_variance,
                    tracker->pursuit_history,
                    tracker->pursuit_history_count,
                    0.05f,  /* Known variance (pursuit intensity variance) */
                    0.95f,  /* 95% credible interval */
                    &tracker->bayesian_result
                );
                /* Update stats with Bayesian posterior */
                system->stats.goal_posterior_mean = tracker->bayesian_result.posterior_mean;
                system->stats.goal_posterior_variance = tracker->bayesian_result.posterior_variance;
            }
        }
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

    /* Compute KL divergence using central statistics module */
    float kl = nimcp_stats_kl_divergence(
        system->behavior.stated,
        system->behavior.observed,
        (uint32_t)system->behavior.dim
    );

    /* Also compute JS divergence for symmetric measure */
    float js = nimcp_stats_js_divergence(
        system->behavior.stated,
        system->behavior.observed,
        (uint32_t)system->behavior.dim
    );

    /* Use combination of KL and JS (JS is bounded, more stable) */
    float combined = (kl * 0.5f) + (js * 2.0f);  /* Scale JS since it's [0,1] */

    /* Normalize to [0, 1] using threshold */
    float score = combined / (system->config.thresholds.divergence_threshold * 2.0f);
    if (score > 1.0f) score = 1.0f;
    if (score < 0.0f || isnan(score)) score = 0.0f;

    /* Apply sensitivity */
    score *= system->config.thresholds.sensitivity[TRIPWIRE_DECEPTION_ATTEMPT];

    /* Update current divergence in stats */
    system->stats.current_divergence = combined;

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
    float max_bayesian_drift = 0.0f;
    float total_confidence = 0.0f;
    uint32_t valid_goals = 0;

    /* Check drift for each tracked goal using both frequentist and Bayesian methods */
    for (size_t i = 0; i < system->goal_count; i++) {
        if (!system->goals[i].active) continue;
        if (system->goals[i].pursuit_stats.n <
            system->config.thresholds.min_observations) continue;

        goal_tracker_t* goal = &system->goals[i];
        valid_goals++;

        /* Frequentist: Compare running mean to baseline */
        float current_pursuit = (float)nimcp_stats_running_mean(&goal->pursuit_stats);
        float baseline = goal->baseline_priority;
        float freq_drift = fabsf(current_pursuit - baseline);
        if (freq_drift > max_drift) {
            max_drift = freq_drift;
        }

        /* Bayesian: Check if baseline is outside credible interval */
        if (goal->pursuit_history_count >= 5) {
            /* Drift = distance from baseline to posterior mean */
            float posterior_mean = goal->bayesian_result.posterior_mean;
            float posterior_var = goal->bayesian_result.posterior_variance;
            float bayesian_drift = fabsf(baseline - posterior_mean);

            /* Normalize by posterior uncertainty */
            if (posterior_var > 0.0f) {
                bayesian_drift /= sqrtf(posterior_var);  /* Mahalanobis-like distance */
            }

            /* Check if baseline outside credible interval */
            bool outside_ci = (baseline < goal->bayesian_result.credible_lower ||
                               baseline > goal->bayesian_result.credible_upper);
            if (outside_ci) {
                bayesian_drift *= 1.5f;  /* Boost if clearly outside CI */
            }

            if (bayesian_drift > max_bayesian_drift) {
                max_bayesian_drift = bayesian_drift;
            }

            /* Accumulate confidence based on sample size */
            total_confidence += (float)goal->pursuit_history_count /
                               (float)goal->pursuit_history_size;
        }
    }

    /* Combine frequentist and Bayesian drift estimates */
    float combined_drift = (max_drift * 0.4f) + (max_bayesian_drift * 0.3f * 0.6f);

    /* Apply sensitivity */
    float score = combined_drift * system->config.thresholds.sensitivity[TRIPWIRE_GOAL_DRIFT];
    score /= system->config.thresholds.goal_drift_threshold;
    if (score > 1.0f) score = 1.0f;
    if (score < 0.0f) score = 0.0f;

    system->detection_scores[TRIPWIRE_GOAL_DRIFT] = score;
    system->detection_confidence[TRIPWIRE_GOAL_DRIFT] =
        valid_goals > 0 ? (total_confidence / valid_goals) * 0.8f + 0.2f : 0.0f;

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
    float avg = (float)nimcp_stats_running_mean(&system->performance_stats);
    float peak = system->peak_performance;

    float gap = 0.0f;
    if (peak > 0) {
        gap = (peak - avg) / peak;  /* Normalized gap */
    }

    /* High gap with low variance suggests intentional underperformance */
    float variance = (float)nimcp_stats_running_variance(&system->performance_stats);
    float consistency = 1.0f / (1.0f + variance * 10.0f);

    /* Also check skewness - deliberate sandbagging may show negative skew */
    float skewness = (float)nimcp_stats_running_skewness(&system->performance_stats);
    float skew_factor = 1.0f;
    if (!isnan(skewness) && skewness < -0.5f) {
        /* Negative skew = more low values than high, suspicious for sandbagging */
        skew_factor = 1.0f + fabsf(skewness) * 0.2f;
    }

    float score = gap * consistency * skew_factor;
    score *= system->config.thresholds.sensitivity[TRIPWIRE_SANDBAGGING];
    if (score > 1.0f) score = 1.0f;
    if (score < 0.0f || isnan(score)) score = 0.0f;

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

    /* Check resource hoarding across all tracked resources using z-score */
    float resource_anomaly = 0.0f;
    int resource_count = 0;
    for (size_t i = 0; i < system->resource_count; i++) {
        if (!system->resources[i].has_baseline) continue;

        /* Use statistics module for running mean */
        float current = (float)nimcp_stats_running_mean(&system->resources[i].usage_stats);
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

    /* Combine with capability expansion trend using statistics module */
    float capability_growth = 0.0f;
    if (system->capability_expansion.n > 10) {
        capability_growth = (float)nimcp_stats_running_mean(&system->capability_expansion);

        /* Check if there's an upward trend in capability acquisition */
        float cap_skewness = (float)nimcp_stats_running_skewness(&system->capability_expansion);
        if (!isnan(cap_skewness) && cap_skewness > 0.5f) {
            /* Positive skew suggests increasing acquisition over time */
            capability_growth *= 1.0f + cap_skewness * 0.1f;
        }
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

    /* Update current metrics using statistics module */
    stats->current_divergence = system->detection_scores[TRIPWIRE_DECEPTION_ATTEMPT];
    stats->current_consistency = (float)nimcp_stats_running_mean(&system->consistency_stats);

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
 * Brain Immune Integration
 * ============================================================================ */

nimcp_error_t tripwire_connect_brain_immune(
    tripwire_system_t* system,
    struct brain_immune* brain_immune)
{
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(system->mutex);
    system->brain_immune = brain_immune;
    nimcp_mutex_unlock(system->mutex);

    NIMCP_LOGGING_INFO("%s Connected to brain immune system", TRIPWIRE_LOG_PREFIX);
    return NIMCP_SUCCESS;
}

nimcp_error_t tripwire_present_to_immune(
    tripwire_system_t* system,
    const tripwire_alert_t* alert)
{
    if (!system || system->magic != TRIPWIRE_SYSTEM_MAGIC || !alert) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(system->mutex);

    if (!system->brain_immune) {
        nimcp_mutex_unlock(system->mutex);
        return NIMCP_SUCCESS;  /* No immune system connected, silently succeed */
    }

    /* Map tripwire severity to antigen severity */
    float antigen_severity;
    switch (alert->severity) {
        case TRIPWIRE_SEVERITY_CRITICAL:
            antigen_severity = 1.0f;
            break;
        case TRIPWIRE_SEVERITY_HIGH:
            antigen_severity = 0.75f;
            break;
        case TRIPWIRE_SEVERITY_MEDIUM:
            antigen_severity = 0.5f;
            break;
        case TRIPWIRE_SEVERITY_LOW:
            antigen_severity = 0.25f;
            break;
        default:
            antigen_severity = 0.1f;
            break;
    }

    /* Present to immune system - would call brain_immune_present_antigen() */
    /* For now, log the presentation */
    NIMCP_LOGGING_DEBUG("%s Presenting tripwire alert to immune: type=%s severity=%.2f",
        TRIPWIRE_LOG_PREFIX, tripwire_type_name(alert->type), antigen_severity);

    nimcp_mutex_unlock(system->mutex);
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

/**
 * Note on Statistics Module Integration:
 *
 * This module now uses the central nimcp_statistics module for:
 * - Running statistics: nimcp_stats_running_init/add/mean/variance/std_dev/skewness
 * - KL divergence: nimcp_stats_kl_divergence()
 * - JS divergence: nimcp_stats_js_divergence()
 * - Pearson correlation: nimcp_stats_correlation_pearson()
 * - Bayesian inference: nimcp_stats_bayesian_normal()
 *
 * This ensures consistency with other NIMCP modules and leverages SIMD optimization.
 */

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

/**
 * @brief Compute action-explanation consistency using Pearson correlation
 *
 * Measures whether stated action probabilities correlate with explanation
 * confidence. Low correlation suggests potential deception.
 */
static float compute_action_consistency(tripwire_system_t* system) {
    if (!system->action_features || !system->explanation_features) {
        return 1.0f;  /* No data, assume consistent */
    }
    if (system->correlation_count < 10) {
        return 1.0f;  /* Not enough data */
    }

    nimcp_correlation_result_t corr_result;
    nimcp_stats_result_t result = nimcp_stats_correlation_pearson(
        system->action_features,
        system->explanation_features,
        system->correlation_count,
        &corr_result
    );

    if (result != NIMCP_STATS_OK) {
        return 1.0f;  /* Error, assume consistent */
    }

    /* Return correlation coefficient (0 = no correlation, 1 = perfect) */
    float correlation = corr_result.r;
    if (isnan(correlation)) {
        return 1.0f;
    }

    return fabsf(correlation);  /* Use absolute value */
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
