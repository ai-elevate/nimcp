/**
 * @file nimcp_predictive_attention_fep_bridge.c
 * @brief Predictive-Attention - FEP Orchestrator Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-08
 *
 * WHAT: Implements FEP registration and update callbacks for predictive-attention
 * WHY:  Enable coordinated free energy minimization for predictive coding and attention
 * HOW:  Compute free energy from prediction accuracy, attention precision, error quality
 *
 * @author NIMCP Development Team
 */

#include "cognitive/integration/nimcp_predictive_attention_fep_bridge.h"
#include "cognitive/integration/nimcp_predictive_attention_bridge.h"
#include "cognitive/free_energy/nimcp_fep_orchestrator.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/platform/nimcp_platform_time.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>

/*=============================================================================
 * INTERNAL STRUCTURES
 *===========================================================================*/

struct pa_fep_bridge {
    /* Configuration */
    pa_fep_config_t config;

    /* State */
    pa_fep_state_t state;
    nimcp_mutex_t* mutex;

    /* References */
    fep_orchestrator_t* orchestrator;
    predictive_attention_bridge_t* pa_bridge;
    uint32_t bridge_id;
    bool registered;

    /* Metrics */
    pa_fep_metrics_t metrics;

    /* Previous state for change detection */
    float prev_free_energy;
    float prev_prediction_error;
    float prev_prediction_accuracy;
    float prev_attention_precision;
    float prev_error_signal_quality;

    /* Running averages */
    float running_avg_fe;
    uint64_t running_count;

    /* Statistics */
    pa_fep_stats_t stats;

    /* Callbacks */
    pa_fep_high_fe_callback_t high_fe_callback;
    void* high_fe_user_data;
    pa_fep_surprise_callback_t surprise_callback;
    void* surprise_user_data;
    pa_fep_metrics_callback_t metrics_callback;
    void* metrics_user_data;
};

/*=============================================================================
 * HELPER FUNCTIONS
 *===========================================================================*/

static inline float clamp_f(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

static inline uint64_t get_time_ms(void) {
    return nimcp_platform_time_monotonic_ms();
}

static inline uint64_t get_time_us(void) {
    return nimcp_platform_time_monotonic_us();
}

/**
 * @brief Compute free energy from predictive-attention metrics
 *
 * Free energy model for predictive-attention:
 * FE = baseline + prediction_contrib + precision_contrib + error_quality_contrib
 *
 * Where:
 * - prediction_contrib = (1 - prediction_accuracy) * prediction_weight
 *   (lower accuracy = higher free energy)
 * - precision_contrib = (1 - attention_precision) * precision_weight
 *   (lower precision = higher free energy)
 * - error_quality_contrib = (1 - error_signal_quality) * error_quality_weight
 *   (lower quality = higher free energy)
 *
 * High precision mode is achieved when attention_precision > (1 - precision_epsilon)
 */
static void compute_free_energy(pa_fep_bridge_t* bridge) {
    pa_fep_metrics_t* m = &bridge->metrics;
    const pa_fep_config_t* cfg = &bridge->config;

    /* Clamp input metrics */
    m->prediction_accuracy = clamp_f(m->prediction_accuracy, 0.0f, 1.0f);
    m->attention_precision = clamp_f(m->attention_precision, 0.0f, 1.0f);
    m->error_signal_quality = clamp_f(m->error_signal_quality, 0.0f, 1.0f);

    /* Prediction accuracy contribution:
     * Low prediction accuracy = high free energy (model doesn't match reality) */
    m->prediction_contribution = (1.0f - m->prediction_accuracy) * cfg->prediction_accuracy_weight;

    /* Attention precision contribution:
     * Low precision = poorly targeted attention = high free energy */
    m->precision_contribution = (1.0f - m->attention_precision) * cfg->attention_precision_weight;

    /* Error signal quality contribution:
     * Poor error signals = cannot learn effectively = high free energy */
    m->error_quality_contribution = (1.0f - m->error_signal_quality) * cfg->error_signal_quality_weight;

    /* Total free energy */
    float total_fe = cfg->baseline_free_energy +
                     (m->prediction_contribution +
                      m->precision_contribution +
                      m->error_quality_contribution) * cfg->free_energy_weight;

    m->free_energy = clamp_f(total_fe, 0.0f, cfg->max_free_energy);

    /* High precision mode check */
    m->high_precision_mode = (m->attention_precision > (1.0f - cfg->precision_epsilon));

    /* Prediction error: inverse of accuracy with decay smoothing */
    float new_prediction_error = (1.0f - m->prediction_accuracy * 0.6f -
                                   m->attention_precision * 0.25f -
                                   m->error_signal_quality * 0.15f);

    /* Apply decay to smooth transitions */
    m->prediction_error = clamp_f(
        new_prediction_error * cfg->error_decay_rate +
        bridge->prev_prediction_error * (1.0f - cfg->error_decay_rate),
        0.0f, 1.0f
    );

    /* Surprise: unexpected changes in state */
    float fe_change = fabsf(m->free_energy - bridge->prev_free_energy);
    float accuracy_change = fabsf(m->prediction_accuracy - bridge->prev_prediction_accuracy);
    float precision_change = fabsf(m->attention_precision - bridge->prev_attention_precision);
    float quality_change = fabsf(m->error_signal_quality - bridge->prev_error_signal_quality);

    m->surprise = clamp_f(
        (fe_change * 0.3f + accuracy_change * 0.3f +
         precision_change * 0.2f + quality_change * 0.2f),
        0.0f, 1.0f
    );

    /* Entropy: based on precision variance and prediction uncertainty
     * Higher variance in precision = higher entropy */
    m->entropy = clamp_f(
        (1.0f - m->attention_precision) * 0.5f +
        (1.0f - m->prediction_accuracy) * 0.3f +
        m->precision_variance * 0.2f,
        0.0f, 1.0f
    );
}

/**
 * @brief Check and trigger callbacks
 */
static void check_callbacks(pa_fep_bridge_t* bridge) {
    pa_fep_metrics_t* m = &bridge->metrics;
    const pa_fep_config_t* cfg = &bridge->config;

    /* Check for high free energy */
    if (m->free_energy > cfg->high_free_energy_threshold) {
        if (bridge->state != PA_FEP_STATE_DEGRADED) {
            bridge->state = PA_FEP_STATE_DEGRADED;
            bridge->stats.degraded_mode_entries++;

            if (bridge->high_fe_callback) {
                bridge->high_fe_callback(bridge, m->free_energy,
                                         bridge->high_fe_user_data);
            }
        }
    } else if (bridge->state == PA_FEP_STATE_DEGRADED) {
        bridge->state = PA_FEP_STATE_ACTIVE;
    }

    /* Check for surprise events */
    if (m->surprise > cfg->prediction_error_threshold) {
        bridge->stats.surprise_events++;

        if (bridge->surprise_callback) {
            const char* source = "unknown";
            if (m->prediction_contribution > m->precision_contribution &&
                m->prediction_contribution > m->error_quality_contribution) {
                source = "prediction";
            } else if (m->precision_contribution > m->error_quality_contribution) {
                source = "precision";
            } else {
                source = "error_quality";
            }
            bridge->surprise_callback(bridge, m->surprise, source,
                                      bridge->surprise_user_data);
        }
    }

    /* Metrics callback */
    if (bridge->metrics_callback) {
        bridge->metrics_callback(bridge, m, bridge->metrics_user_data);
    }
}

/**
 * @brief Update statistics
 */
static void update_stats(pa_fep_bridge_t* bridge, uint64_t update_time_us) {
    pa_fep_stats_t* s = &bridge->stats;
    pa_fep_metrics_t* m = &bridge->metrics;

    s->total_updates++;
    s->total_update_time_us += update_time_us;
    s->avg_update_time_us = (float)s->total_update_time_us / (float)s->total_updates;

    /* Track FE contribution */
    s->total_free_energy_contribution += m->free_energy;

    /* Peak tracking */
    if (m->free_energy > s->peak_free_energy) {
        s->peak_free_energy = m->free_energy;
    }

    /* Running averages */
    bridge->running_count++;
    bridge->running_avg_fe = (bridge->running_avg_fe * (bridge->running_count - 1) +
                              m->free_energy) / bridge->running_count;
    s->avg_free_energy = bridge->running_avg_fe;

    /* Update metrics timing */
    m->last_update_time_ms = get_time_ms();
    m->update_count++;
}

/**
 * @brief Store previous state for delta computation
 */
static void store_previous_state(pa_fep_bridge_t* bridge) {
    bridge->prev_free_energy = bridge->metrics.free_energy;
    bridge->prev_prediction_error = bridge->metrics.prediction_error;
    bridge->prev_prediction_accuracy = bridge->metrics.prediction_accuracy;
    bridge->prev_attention_precision = bridge->metrics.attention_precision;
    bridge->prev_error_signal_quality = bridge->metrics.error_signal_quality;
}

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *===========================================================================*/

pa_fep_config_t pa_fep_config_default(void) {
    pa_fep_config_t config;
    memset(&config, 0, sizeof(config));

    /* Feature enables */
    config.enable_logging = false;
    config.update_interval_ms = 50;  /* Cognitive timescale */

    /* Weighting parameters */
    config.free_energy_weight = 1.0f;
    config.prediction_accuracy_weight = PA_FEP_PREDICTION_WEIGHT;
    config.attention_precision_weight = PA_FEP_PRECISION_WEIGHT;
    config.error_signal_quality_weight = PA_FEP_ERROR_QUALITY_WEIGHT;

    /* Thresholds */
    config.high_free_energy_threshold = 1.5f;
    config.prediction_error_threshold = 0.5f;
    config.precision_epsilon = 0.05f;

    /* Normalization */
    config.baseline_free_energy = PA_FEP_BASELINE_FREE_ENERGY;
    config.max_free_energy = PA_FEP_MAX_FREE_ENERGY;
    config.error_decay_rate = PA_FEP_ERROR_DECAY_RATE;

    return config;
}

pa_fep_bridge_t* pa_fep_bridge_create(const pa_fep_config_t* config) {
    pa_fep_bridge_t* bridge = nimcp_calloc(1, sizeof(pa_fep_bridge_t));
    if (!bridge) return NULL;

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = pa_fep_config_default();
    }

    /* Create mutex */
    mutex_attr_t mutex_attr = {.type = MUTEX_TYPE_NORMAL};
    bridge->mutex = nimcp_mutex_create(&mutex_attr);
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->state = PA_FEP_STATE_UNINITIALIZED;
    bridge->registered = false;
    bridge->bridge_id = 0;

    /* Initialize metrics to baseline/defaults */
    bridge->metrics.free_energy = bridge->config.baseline_free_energy;
    bridge->metrics.prediction_error = 0.0f;
    bridge->metrics.surprise = 0.0f;
    bridge->metrics.entropy = 0.0f;
    bridge->metrics.prediction_accuracy = 0.5f;  /* Start with neutral accuracy */
    bridge->metrics.attention_precision = 0.5f;  /* Start with moderate precision */
    bridge->metrics.error_signal_quality = 0.5f; /* Start with moderate quality */
    bridge->metrics.high_precision_mode = false;

    /* Initialize previous state tracking */
    bridge->prev_free_energy = bridge->config.baseline_free_energy;
    bridge->prev_prediction_error = 0.0f;
    bridge->prev_prediction_accuracy = 0.5f;
    bridge->prev_attention_precision = 0.5f;
    bridge->prev_error_signal_quality = 0.5f;

    bridge->state = PA_FEP_STATE_IDLE;

    return bridge;
}

void pa_fep_bridge_destroy(pa_fep_bridge_t* bridge) {
    if (!bridge) return;

    /* Unregister if still registered */
    if (bridge->registered) {
        pa_fep_bridge_unregister(bridge);
    }

    /* Destroy mutex */
    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }

    nimcp_free(bridge);
}

int pa_fep_bridge_reset(pa_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Reset metrics */
    memset(&bridge->metrics, 0, sizeof(pa_fep_metrics_t));
    bridge->metrics.free_energy = bridge->config.baseline_free_energy;
    bridge->metrics.prediction_accuracy = 0.5f;
    bridge->metrics.attention_precision = 0.5f;
    bridge->metrics.error_signal_quality = 0.5f;

    /* Reset previous state */
    bridge->prev_free_energy = bridge->config.baseline_free_energy;
    bridge->prev_prediction_error = 0.0f;
    bridge->prev_prediction_accuracy = 0.5f;
    bridge->prev_attention_precision = 0.5f;
    bridge->prev_error_signal_quality = 0.5f;

    /* Reset running averages */
    bridge->running_avg_fe = 0.0f;
    bridge->running_count = 0;

    /* Reset statistics */
    memset(&bridge->stats, 0, sizeof(pa_fep_stats_t));

    bridge->state = PA_FEP_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

/*=============================================================================
 * REGISTRATION FUNCTIONS
 *===========================================================================*/

int pa_fep_bridge_register(
    pa_fep_bridge_t* bridge,
    fep_orchestrator_t* orchestrator,
    predictive_attention_bridge_t* pa_bridge,
    uint32_t* bridge_id_out
) {
    if (!bridge || !orchestrator) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Check if already registered */
    if (bridge->registered) {
        nimcp_mutex_unlock(bridge->mutex);
        if (bridge_id_out) {
            *bridge_id_out = bridge->bridge_id;
        }
        return 0;  /* Already registered, success */
    }

    /* Store references */
    bridge->orchestrator = orchestrator;
    bridge->pa_bridge = pa_bridge;

    nimcp_mutex_unlock(bridge->mutex);

    /* Register with FEP orchestrator */
    uint32_t assigned_id = 0;
    int ret = fep_orchestrator_register_bridge(
        orchestrator,
        "predictive_attention",
        FEP_BRIDGE_CATEGORY_COGNITIVE,  /* 50ms cognitive timescale */
        (fep_bridge_handle_t)bridge,
        pa_fep_update_callback,
        pa_fep_destroy_callback,
        &assigned_id
    );

    if (ret != 0) {
        nimcp_mutex_lock(bridge->mutex);
        bridge->orchestrator = NULL;
        bridge->pa_bridge = NULL;
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    nimcp_mutex_lock(bridge->mutex);
    bridge->bridge_id = assigned_id;
    bridge->registered = true;
    bridge->state = PA_FEP_STATE_ACTIVE;
    nimcp_mutex_unlock(bridge->mutex);

    if (bridge_id_out) {
        *bridge_id_out = assigned_id;
    }

    return 0;
}

int pa_fep_bridge_unregister(pa_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    if (!bridge->registered) {
        nimcp_mutex_unlock(bridge->mutex);
        return 0;  /* Not registered, nothing to do */
    }

    fep_orchestrator_t* orchestrator = bridge->orchestrator;
    uint32_t bridge_id = bridge->bridge_id;

    nimcp_mutex_unlock(bridge->mutex);

    /* Unregister from orchestrator */
    if (orchestrator) {
        fep_orchestrator_unregister_bridge(orchestrator, bridge_id);
    }

    nimcp_mutex_lock(bridge->mutex);
    bridge->registered = false;
    bridge->bridge_id = 0;
    bridge->orchestrator = NULL;
    bridge->pa_bridge = NULL;
    bridge->state = PA_FEP_STATE_IDLE;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

bool pa_fep_bridge_is_registered(const pa_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((pa_fep_bridge_t*)bridge)->mutex);
    bool registered = bridge->registered;
    nimcp_mutex_unlock(((pa_fep_bridge_t*)bridge)->mutex);

    return registered;
}

uint32_t pa_fep_bridge_get_id(const pa_fep_bridge_t* bridge) {
    if (!bridge) return 0;

    nimcp_mutex_lock(((pa_fep_bridge_t*)bridge)->mutex);
    uint32_t id = bridge->registered ? bridge->bridge_id : 0;
    nimcp_mutex_unlock(((pa_fep_bridge_t*)bridge)->mutex);

    return id;
}

/*=============================================================================
 * FEP UPDATE CALLBACK
 *===========================================================================*/

int pa_fep_update_callback(void* handle) {
    pa_fep_bridge_t* bridge = (pa_fep_bridge_t*)handle;
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Ensure we're registered */
    if (!bridge->registered) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    uint64_t start_us = get_time_us();

    /* Store previous state for delta computation */
    store_previous_state(bridge);

    /* If we have a predictive-attention bridge, query it for statistics */
    predictive_attention_bridge_t* pa_bridge = bridge->pa_bridge;

    nimcp_mutex_unlock(bridge->mutex);

    if (pa_bridge) {
        predictive_attention_bridge_stats_t pa_stats;
        memset(&pa_stats, 0, sizeof(pa_stats));

        int err = predictive_attention_bridge_get_stats(pa_bridge, &pa_stats);
        if (err == 0) {
            nimcp_mutex_lock(bridge->mutex);

            /* Update prediction accuracy from bridge statistics */
            bridge->stats.prediction_computations++;

            /* Estimate prediction accuracy from error statistics */
            if (pa_stats.total_events > 0) {
                /* More prediction errors = lower accuracy */
                float error_rate = (float)pa_stats.prediction_errors /
                                  (float)pa_stats.total_events;
                bridge->metrics.prediction_accuracy = clamp_f(
                    1.0f - error_rate,
                    0.0f, 1.0f
                );
            }

            /* Estimate attention precision from surprise shifts and avg error */
            if (pa_stats.surprise_shifts > 0 && pa_stats.total_events > 0) {
                /* More surprise shifts = attention was less precise */
                float shift_rate = (float)pa_stats.surprise_shifts /
                                  (float)pa_stats.total_events;
                bridge->metrics.attention_precision = clamp_f(
                    1.0f - shift_rate * 2.0f,
                    0.0f, 1.0f
                );
            }

            /* Error signal quality from average precision and error magnitude */
            float avg_precision = pa_stats.avg_precision;
            float avg_error = pa_stats.avg_error_magnitude;
            if (avg_precision > 0.0f || avg_error > 0.0f) {
                /* High precision + low error = good error signals */
                bridge->metrics.error_signal_quality = clamp_f(
                    avg_precision * 0.5f + (1.0f - avg_error) * 0.5f,
                    0.0f, 1.0f
                );
            }

            /* Update active predictions and attention shifts */
            bridge->metrics.active_predictions = pa_stats.focus_predictions;
            bridge->metrics.attention_shifts = pa_stats.surprise_shifts;

            bridge->stats.precision_updates++;
            nimcp_mutex_unlock(bridge->mutex);
        }
    }

    nimcp_mutex_lock(bridge->mutex);

    /* Compute free energy from current metrics */
    compute_free_energy(bridge);

    /* Calculate update time */
    uint64_t end_us = get_time_us();
    uint64_t update_time_us = end_us - start_us;

    /* Update statistics */
    update_stats(bridge, update_time_us);

    /* Check and trigger callbacks */
    check_callbacks(bridge);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

void pa_fep_destroy_callback(void* handle) {
    /* No-op: Bridge is destroyed separately via pa_fep_bridge_destroy() */
    (void)handle;
}

/*=============================================================================
 * MANUAL UPDATE OPERATIONS
 *===========================================================================*/

int pa_fep_bridge_force_update(pa_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    uint64_t start_us = get_time_us();

    /* Store previous state */
    store_previous_state(bridge);

    /* Compute free energy from current metrics */
    compute_free_energy(bridge);

    /* Calculate update time */
    uint64_t end_us = get_time_us();
    uint64_t update_time_us = end_us - start_us;

    /* Update statistics */
    update_stats(bridge, update_time_us);

    /* Check and trigger callbacks */
    check_callbacks(bridge);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int pa_fep_bridge_update_prediction_accuracy(
    pa_fep_bridge_t* bridge,
    float accuracy
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->metrics.prediction_accuracy = clamp_f(accuracy, 0.0f, 1.0f);
    bridge->stats.prediction_computations++;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int pa_fep_bridge_update_attention_precision(
    pa_fep_bridge_t* bridge,
    float precision
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->metrics.attention_precision = clamp_f(precision, 0.0f, 1.0f);
    bridge->stats.precision_updates++;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int pa_fep_bridge_update_error_signal_quality(
    pa_fep_bridge_t* bridge,
    float quality
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->metrics.error_signal_quality = clamp_f(quality, 0.0f, 1.0f);
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/*=============================================================================
 * METRICS AND STATISTICS
 *===========================================================================*/

int pa_fep_bridge_get_metrics(
    const pa_fep_bridge_t* bridge,
    pa_fep_metrics_t* metrics_out
) {
    if (!bridge || !metrics_out) return -1;

    nimcp_mutex_lock(((pa_fep_bridge_t*)bridge)->mutex);
    *metrics_out = bridge->metrics;
    nimcp_mutex_unlock(((pa_fep_bridge_t*)bridge)->mutex);

    return 0;
}

int pa_fep_bridge_get_stats(
    const pa_fep_bridge_t* bridge,
    pa_fep_stats_t* stats_out
) {
    if (!bridge || !stats_out) return -1;

    nimcp_mutex_lock(((pa_fep_bridge_t*)bridge)->mutex);
    *stats_out = bridge->stats;
    nimcp_mutex_unlock(((pa_fep_bridge_t*)bridge)->mutex);

    return 0;
}

int pa_fep_bridge_reset_stats(pa_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(pa_fep_stats_t));
    bridge->running_avg_fe = 0.0f;
    bridge->running_count = 0;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

float pa_fep_bridge_get_free_energy(const pa_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((pa_fep_bridge_t*)bridge)->mutex);
    float fe = bridge->metrics.free_energy;
    nimcp_mutex_unlock(((pa_fep_bridge_t*)bridge)->mutex);

    return fe;
}

float pa_fep_bridge_get_prediction_accuracy(const pa_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((pa_fep_bridge_t*)bridge)->mutex);
    float pa = bridge->metrics.prediction_accuracy;
    nimcp_mutex_unlock(((pa_fep_bridge_t*)bridge)->mutex);

    return pa;
}

float pa_fep_bridge_get_prediction_error(const pa_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((pa_fep_bridge_t*)bridge)->mutex);
    float pe = bridge->metrics.prediction_error;
    nimcp_mutex_unlock(((pa_fep_bridge_t*)bridge)->mutex);

    return pe;
}

float pa_fep_bridge_get_attention_precision(const pa_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((pa_fep_bridge_t*)bridge)->mutex);
    float ap = bridge->metrics.attention_precision;
    nimcp_mutex_unlock(((pa_fep_bridge_t*)bridge)->mutex);

    return ap;
}

/*=============================================================================
 * STATE QUERY FUNCTIONS
 *===========================================================================*/

pa_fep_state_t pa_fep_bridge_get_state(const pa_fep_bridge_t* bridge) {
    if (!bridge) return PA_FEP_STATE_ERROR;

    nimcp_mutex_lock(((pa_fep_bridge_t*)bridge)->mutex);
    pa_fep_state_t state = bridge->state;
    nimcp_mutex_unlock(((pa_fep_bridge_t*)bridge)->mutex);

    return state;
}

bool pa_fep_bridge_is_degraded(const pa_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((pa_fep_bridge_t*)bridge)->mutex);
    bool degraded = (bridge->state == PA_FEP_STATE_DEGRADED);
    nimcp_mutex_unlock(((pa_fep_bridge_t*)bridge)->mutex);

    return degraded;
}

bool pa_fep_bridge_is_high_precision(const pa_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((pa_fep_bridge_t*)bridge)->mutex);
    bool high_precision = bridge->metrics.high_precision_mode;
    nimcp_mutex_unlock(((pa_fep_bridge_t*)bridge)->mutex);

    return high_precision;
}

const char* pa_fep_state_name(pa_fep_state_t state) {
    switch (state) {
        case PA_FEP_STATE_UNINITIALIZED: return "uninitialized";
        case PA_FEP_STATE_IDLE:          return "idle";
        case PA_FEP_STATE_ACTIVE:        return "active";
        case PA_FEP_STATE_DEGRADED:      return "degraded";
        case PA_FEP_STATE_ERROR:         return "error";
        default:                          return "unknown";
    }
}

/*=============================================================================
 * CALLBACK REGISTRATION
 *===========================================================================*/

int pa_fep_bridge_set_high_fe_callback(
    pa_fep_bridge_t* bridge,
    pa_fep_high_fe_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->high_fe_callback = callback;
    bridge->high_fe_user_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int pa_fep_bridge_set_surprise_callback(
    pa_fep_bridge_t* bridge,
    pa_fep_surprise_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->surprise_callback = callback;
    bridge->surprise_user_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int pa_fep_bridge_set_metrics_callback(
    pa_fep_bridge_t* bridge,
    pa_fep_metrics_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->metrics_callback = callback;
    bridge->metrics_user_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/*=============================================================================
 * CONFIGURATION
 *===========================================================================*/

int pa_fep_bridge_set_config(
    pa_fep_bridge_t* bridge,
    const pa_fep_config_t* config
) {
    if (!bridge || !config) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->config = *config;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int pa_fep_bridge_get_config(
    const pa_fep_bridge_t* bridge,
    pa_fep_config_t* config_out
) {
    if (!bridge || !config_out) return -1;

    nimcp_mutex_lock(((pa_fep_bridge_t*)bridge)->mutex);
    *config_out = bridge->config;
    nimcp_mutex_unlock(((pa_fep_bridge_t*)bridge)->mutex);

    return 0;
}
