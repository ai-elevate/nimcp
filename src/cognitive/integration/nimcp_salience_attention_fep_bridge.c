/**
 * @file nimcp_salience_attention_fep_bridge.c
 * @brief Salience-Attention - FEP Orchestrator Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-08
 *
 * WHAT: Implements FEP registration and update callbacks for salience-attention
 * WHY:  Enable coordinated free energy minimization for attention allocation
 * HOW:  Compute free energy from salience prediction, attention allocation,
 *       and priority estimation errors
 *
 * @author NIMCP Development Team
 */

#include "cognitive/integration/nimcp_salience_attention_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/integration/nimcp_salience_attention_bridge.h"
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

struct sa_fep_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */

    /* Configuration */
    sa_fep_config_t config;

    /* State */
    sa_fep_state_t state;

    /* References */
    fep_orchestrator_t* orchestrator;
    salience_attention_bridge_t* sa_bridge;
    uint32_t bridge_id;
    bool registered;

    /* Metrics */
    sa_fep_metrics_t metrics;

    /* Previous state for change detection */
    float prev_free_energy;
    float prev_prediction_error;
    float prev_salience_error;
    float prev_attention_error;
    float prev_priority_error;

    /* Running averages */
    float running_avg_fe;
    uint64_t running_count;

    /* Statistics */
    sa_fep_stats_t stats;

    /* Callbacks */
    sa_fep_high_fe_callback_t high_fe_callback;
    void* high_fe_user_data;
    sa_fep_surprise_callback_t surprise_callback;
    void* surprise_user_data;
    sa_fep_metrics_callback_t metrics_callback;
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
 * @brief Compute free energy from salience-attention metrics
 *
 * Free energy model for attention allocation:
 * FE = baseline + salience_contrib + attention_contrib + priority_contrib
 *
 * Where:
 * - salience_contrib = salience_prediction_error * salience_weight
 *   (higher mismatch in salience prediction = higher FE)
 * - attention_contrib = attention_allocation_error * attention_weight
 *   (worse attention allocation = higher FE)
 * - priority_contrib = priority_estimation_error * priority_weight
 *   (worse priority estimation = higher FE)
 *
 * Efficient attention represents minimum free energy state where
 * salience is accurately predicted and attention is optimally allocated.
 */
static void compute_free_energy(sa_fep_bridge_t* bridge) {
    sa_fep_metrics_t* m = &bridge->metrics;
    const sa_fep_config_t* cfg = &bridge->config;

    /* Clamp input metrics */
    m->salience_prediction_error = clamp_f(m->salience_prediction_error, 0.0f, 1.0f);
    m->attention_allocation_error = clamp_f(m->attention_allocation_error, 0.0f, 1.0f);
    m->priority_estimation_error = clamp_f(m->priority_estimation_error, 0.0f, 1.0f);
    m->attention_efficiency = clamp_f(m->attention_efficiency, 0.0f, 1.0f);

    /* Salience prediction contribution:
     * High error in predicting salient stimuli = prediction error */
    m->salience_contribution = m->salience_prediction_error * cfg->salience_prediction_weight;

    /* Attention allocation contribution:
     * Poor attention allocation = prediction error */
    m->attention_contribution = m->attention_allocation_error * cfg->attention_allocation_weight;

    /* Priority estimation contribution:
     * Error in estimating priorities = prediction error
     * Good priority estimation means accurate predictions of importance */
    m->priority_contribution = m->priority_estimation_error * cfg->priority_estimation_weight;

    /* Total free energy */
    float total_fe = cfg->baseline_free_energy +
                     (m->salience_contribution +
                      m->attention_contribution +
                      m->priority_contribution) * cfg->free_energy_weight;

    m->free_energy = clamp_f(total_fe, 0.0f, cfg->max_free_energy);

    /* Attention captured check - attention is captured when salience is high
     * and attention successfully shifted */
    m->attention_captured = (m->attention_efficiency > cfg->attention_efficiency_threshold &&
                             m->salience_prediction_error < 0.3f);

    /* Prediction error: weighted combination of all uncertainty sources */
    float new_prediction_error = (m->salience_prediction_error * 0.4f +
                                   m->attention_allocation_error * 0.4f +
                                   m->priority_estimation_error * 0.2f);

    /* Apply decay to smooth transitions */
    m->prediction_error = clamp_f(
        new_prediction_error * cfg->error_decay_rate +
        bridge->prev_prediction_error * (1.0f - cfg->error_decay_rate),
        0.0f, 1.0f
    );

    /* Surprise: unexpected changes in attention state */
    float fe_change = fabsf(m->free_energy - bridge->prev_free_energy);
    float salience_change = fabsf(m->salience_prediction_error - bridge->prev_salience_error);
    float attention_change = fabsf(m->attention_allocation_error - bridge->prev_attention_error);
    float priority_change = fabsf(m->priority_estimation_error - bridge->prev_priority_error);

    m->surprise = clamp_f(
        (fe_change * 0.3f + salience_change * 0.3f +
         attention_change * 0.2f + priority_change * 0.2f),
        0.0f, 1.0f
    );

    /* Entropy: based on attention distribution
     * More spread attention = higher entropy = higher free energy
     * Approximate based on allocation error */
    m->entropy = clamp_f(
        m->attention_allocation_error * 0.7f + m->salience_prediction_error * 0.3f,
        0.0f, 1.0f
    );
}

/**
 * @brief Check and trigger callbacks
 */
static void check_callbacks(sa_fep_bridge_t* bridge) {
    sa_fep_metrics_t* m = &bridge->metrics;
    const sa_fep_config_t* cfg = &bridge->config;

    /* Check for high free energy */
    if (m->free_energy > cfg->high_free_energy_threshold) {
        if (bridge->state != SA_FEP_STATE_DEGRADED) {
            bridge->state = SA_FEP_STATE_DEGRADED;
            bridge->stats.degraded_mode_entries++;

            if (bridge->high_fe_callback) {
                bridge->high_fe_callback(bridge, m->free_energy,
                                         bridge->high_fe_user_data);
            }
        }
    } else if (bridge->state == SA_FEP_STATE_DEGRADED) {
        bridge->state = SA_FEP_STATE_ACTIVE;
    }

    /* Check for surprise events */
    if (m->surprise > cfg->prediction_error_threshold) {
        bridge->stats.surprise_events++;

        if (bridge->surprise_callback) {
            const char* source = "unknown";
            if (m->salience_contribution > m->attention_contribution &&
                m->salience_contribution > m->priority_contribution) {
                source = "salience";
            } else if (m->attention_contribution > m->priority_contribution) {
                source = "attention";
            } else {
                source = "priority";
            }
            bridge->surprise_callback(bridge, m->surprise, source,
                                      bridge->surprise_user_data);
        }
    }

    /* Track attention captures */
    if (m->attention_captured) {
        bridge->stats.attention_captures++;
    }

    /* Metrics callback */
    if (bridge->metrics_callback) {
        bridge->metrics_callback(bridge, m, bridge->metrics_user_data);
    }
}

/**
 * @brief Update statistics
 */
static void update_stats(sa_fep_bridge_t* bridge, uint64_t update_time_us) {
    sa_fep_stats_t* s = &bridge->stats;
    sa_fep_metrics_t* m = &bridge->metrics;

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
static void store_previous_state(sa_fep_bridge_t* bridge) {
    bridge->prev_free_energy = bridge->metrics.free_energy;
    bridge->prev_prediction_error = bridge->metrics.prediction_error;
    bridge->prev_salience_error = bridge->metrics.salience_prediction_error;
    bridge->prev_attention_error = bridge->metrics.attention_allocation_error;
    bridge->prev_priority_error = bridge->metrics.priority_estimation_error;
}

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *===========================================================================*/

sa_fep_config_t sa_fep_config_default(void) {
    sa_fep_config_t config;
    memset(&config, 0, sizeof(config));

    /* Feature enables */
    config.enable_logging = false;
    config.update_interval_ms = 50;  /* Cognitive timescale */

    /* Weighting parameters */
    config.free_energy_weight = 1.0f;
    config.salience_prediction_weight = SA_FEP_SALIENCE_WEIGHT;
    config.attention_allocation_weight = SA_FEP_ATTENTION_WEIGHT;
    config.priority_estimation_weight = SA_FEP_PRIORITY_WEIGHT;

    /* Thresholds */
    config.high_free_energy_threshold = 1.5f;
    config.prediction_error_threshold = 0.5f;
    config.attention_efficiency_threshold = 0.7f;

    /* Normalization */
    config.baseline_free_energy = SA_FEP_BASELINE_FREE_ENERGY;
    config.max_free_energy = SA_FEP_MAX_FREE_ENERGY;
    config.error_decay_rate = SA_FEP_ERROR_DECAY_RATE;

    return config;
}

sa_fep_bridge_t* sa_fep_bridge_create(const sa_fep_config_t* config) {
    sa_fep_bridge_t* bridge = nimcp_calloc(1, sizeof(sa_fep_bridge_t));
    if (!bridge) return NULL;

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = sa_fep_config_default();
    }

    /* Initialize base bridge infrastructure */
    if (bridge_base_init(&bridge->base, 0, "salience_attention_fep") != 0) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->state = SA_FEP_STATE_UNINITIALIZED;
    bridge->registered = false;
    bridge->bridge_id = 0;

    /* Initialize metrics to baseline */
    bridge->metrics.free_energy = bridge->config.baseline_free_energy;
    bridge->metrics.prediction_error = 0.0f;
    bridge->metrics.surprise = 0.0f;
    bridge->metrics.entropy = 0.0f;
    bridge->metrics.salience_prediction_error = 0.0f;
    bridge->metrics.attention_allocation_error = 0.5f;  /* Start at neutral */
    bridge->metrics.priority_estimation_error = 0.5f;   /* Start at neutral */
    bridge->metrics.attention_efficiency = 0.5f;        /* Start at neutral */
    bridge->metrics.attention_captured = false;

    /* Initialize previous state tracking */
    bridge->prev_free_energy = bridge->config.baseline_free_energy;
    bridge->prev_prediction_error = 0.0f;
    bridge->prev_salience_error = 0.0f;
    bridge->prev_attention_error = 0.5f;
    bridge->prev_priority_error = 0.5f;

    bridge->state = SA_FEP_STATE_IDLE;

    return bridge;
}

void sa_fep_bridge_destroy(sa_fep_bridge_t* bridge) {
    if (!bridge) return;

    /* Unregister if still registered */
    if (bridge->registered) {
        sa_fep_bridge_unregister(bridge);
    }

    /* Cleanup base bridge infrastructure */
    bridge_base_cleanup(&bridge->base);

    nimcp_free(bridge);
}

int sa_fep_bridge_reset(sa_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Reset metrics */
    memset(&bridge->metrics, 0, sizeof(sa_fep_metrics_t));
    bridge->metrics.free_energy = bridge->config.baseline_free_energy;
    bridge->metrics.attention_allocation_error = 0.5f;
    bridge->metrics.priority_estimation_error = 0.5f;
    bridge->metrics.attention_efficiency = 0.5f;

    /* Reset previous state */
    bridge->prev_free_energy = bridge->config.baseline_free_energy;
    bridge->prev_prediction_error = 0.0f;
    bridge->prev_salience_error = 0.0f;
    bridge->prev_attention_error = 0.5f;
    bridge->prev_priority_error = 0.5f;

    /* Reset running averages */
    bridge->running_avg_fe = 0.0f;
    bridge->running_count = 0;

    /* Reset statistics */
    memset(&bridge->stats, 0, sizeof(sa_fep_stats_t));

    bridge->state = SA_FEP_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/*=============================================================================
 * REGISTRATION FUNCTIONS
 *===========================================================================*/

int sa_fep_bridge_register(
    sa_fep_bridge_t* bridge,
    fep_orchestrator_t* orchestrator,
    salience_attention_bridge_t* sa_bridge,
    uint32_t* bridge_id_out
) {
    if (!bridge || !orchestrator) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Check if already registered */
    if (bridge->registered) {
        nimcp_mutex_unlock(bridge->base.mutex);
        if (bridge_id_out) {
            *bridge_id_out = bridge->bridge_id;
        }
        return 0;  /* Already registered, success */
    }

    /* Store references */
    bridge->orchestrator = orchestrator;
    bridge->sa_bridge = sa_bridge;

    nimcp_mutex_unlock(bridge->base.mutex);

    /* Register with FEP orchestrator */
    uint32_t assigned_id = 0;
    int ret = fep_orchestrator_register_bridge(
        orchestrator,
        "salience_attention",
        FEP_BRIDGE_CATEGORY_COGNITIVE,  /* 50ms cognitive timescale */
        (fep_bridge_handle_t)bridge,
        sa_fep_update_callback,
        sa_fep_destroy_callback,
        &assigned_id
    );

    if (ret != 0) {
        nimcp_mutex_lock(bridge->base.mutex);
        bridge->orchestrator = NULL;
        bridge->sa_bridge = NULL;
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->bridge_id = assigned_id;
    bridge->registered = true;
    bridge->state = SA_FEP_STATE_ACTIVE;
    nimcp_mutex_unlock(bridge->base.mutex);

    if (bridge_id_out) {
        *bridge_id_out = assigned_id;
    }

    return 0;
}

int sa_fep_bridge_unregister(sa_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->registered) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;  /* Not registered, nothing to do */
    }

    /* Unregister from orchestrator */
    if (bridge->orchestrator) {
        fep_orchestrator_unregister_bridge(bridge->orchestrator, bridge->bridge_id);
    }

    bridge->registered = false;
    bridge->bridge_id = 0;
    bridge->orchestrator = NULL;
    bridge->sa_bridge = NULL;
    bridge->state = SA_FEP_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

bool sa_fep_bridge_is_registered(const sa_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((sa_fep_bridge_t*)bridge)->base.mutex);
    bool registered = bridge->registered;
    nimcp_mutex_unlock(((sa_fep_bridge_t*)bridge)->base.mutex);

    return registered;
}

uint32_t sa_fep_bridge_get_id(const sa_fep_bridge_t* bridge) {
    if (!bridge) return 0;

    nimcp_mutex_lock(((sa_fep_bridge_t*)bridge)->base.mutex);
    uint32_t id = bridge->registered ? bridge->bridge_id : 0;
    nimcp_mutex_unlock(((sa_fep_bridge_t*)bridge)->base.mutex);

    return id;
}

/*=============================================================================
 * FEP UPDATE CALLBACK
 *===========================================================================*/

int sa_fep_update_callback(void* handle) {
    sa_fep_bridge_t* bridge = (sa_fep_bridge_t*)handle;
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Ensure we're registered */
    if (!bridge->registered) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    uint64_t start_us = get_time_us();

    /* Store previous state for delta computation */
    store_previous_state(bridge);

    /* If we have a salience-attention bridge, query it for statistics */
    if (bridge->sa_bridge) {
        salience_attention_stats_t sa_stats;
        memset(&sa_stats, 0, sizeof(sa_stats));

        int err = salience_attention_bridge_get_stats(bridge->sa_bridge, &sa_stats);
        if (err == 0) {
            /* Update metrics from salience-attention statistics */
            bridge->stats.salience_computations++;

            /* Estimate salience prediction error from detection accuracy
             * Use average salience score as a proxy - low average means
             * many false alarms or missed detections */
            if (sa_stats.salience_detections > 0) {
                /* Lower average salience = higher prediction error
                 * (detecting non-salient items or missing salient ones) */
                bridge->metrics.salience_prediction_error = clamp_f(
                    1.0f - sa_stats.avg_salience_score,
                    0.0f, 1.0f
                );
            }

            /* Estimate attention allocation error from attention shifts
             * More shifts might indicate inefficient allocation */
            if (sa_stats.attention_shifts > 0 && sa_stats.total_events > 0) {
                float shift_rate = (float)sa_stats.attention_shifts /
                                   (float)sa_stats.total_events;
                /* Higher shift rate = less stable = higher error */
                bridge->metrics.attention_allocation_error = clamp_f(
                    shift_rate,
                    0.0f, 1.0f
                );
                bridge->stats.attention_computations++;
            }

            /* Estimate attention efficiency from average attention strength
             * and focus notifications */
            if (sa_stats.focus_notifications > 0) {
                bridge->metrics.attention_efficiency = clamp_f(
                    sa_stats.avg_attention_strength,
                    0.0f, 1.0f
                );
            }

            /* Priority estimation error from priority update frequency
             * Frequent priority updates suggest estimation difficulty */
            if (sa_stats.total_events > 0) {
                float priority_update_rate = (float)sa_stats.priority_updates /
                                             (float)sa_stats.total_events;
                /* Higher update rate = more volatile = higher estimation error */
                bridge->metrics.priority_estimation_error = clamp_f(
                    priority_update_rate * 2.0f,  /* Scale factor */
                    0.0f, 1.0f
                );
                bridge->stats.priority_computations++;
            }

            /* Track active targets */
            bridge->metrics.salience_detections = (uint32_t)sa_stats.salience_detections;
            bridge->metrics.active_targets = (uint32_t)sa_stats.attention_shifts;
        }
    }

    /* Compute free energy from current metrics */
    compute_free_energy(bridge);

    /* Calculate update time */
    uint64_t end_us = get_time_us();
    uint64_t update_time_us = end_us - start_us;

    /* Update statistics */
    update_stats(bridge, update_time_us);

    /* Check and trigger callbacks */
    check_callbacks(bridge);

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

void sa_fep_destroy_callback(void* handle) {
    /* No-op: Bridge is destroyed separately via sa_fep_bridge_destroy() */
    (void)handle;
}

/*=============================================================================
 * MANUAL UPDATE OPERATIONS
 *===========================================================================*/

int sa_fep_bridge_force_update(sa_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

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

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int sa_fep_bridge_update_salience_error(
    sa_fep_bridge_t* bridge,
    float error
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->metrics.salience_prediction_error = clamp_f(error, 0.0f, 1.0f);
    bridge->stats.salience_computations++;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int sa_fep_bridge_update_attention_error(
    sa_fep_bridge_t* bridge,
    float error
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->metrics.attention_allocation_error = clamp_f(error, 0.0f, 1.0f);
    bridge->stats.attention_computations++;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int sa_fep_bridge_update_priority_error(
    sa_fep_bridge_t* bridge,
    float error
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->metrics.priority_estimation_error = clamp_f(error, 0.0f, 1.0f);
    bridge->stats.priority_computations++;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int sa_fep_bridge_update_attention_efficiency(
    sa_fep_bridge_t* bridge,
    float efficiency
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->metrics.attention_efficiency = clamp_f(efficiency, 0.0f, 1.0f);
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/*=============================================================================
 * METRICS AND STATISTICS
 *===========================================================================*/

int sa_fep_bridge_get_metrics(
    const sa_fep_bridge_t* bridge,
    sa_fep_metrics_t* metrics_out
) {
    if (!bridge || !metrics_out) return -1;

    nimcp_mutex_lock(((sa_fep_bridge_t*)bridge)->base.mutex);
    *metrics_out = bridge->metrics;
    nimcp_mutex_unlock(((sa_fep_bridge_t*)bridge)->base.mutex);

    return 0;
}

int sa_fep_bridge_get_stats(
    const sa_fep_bridge_t* bridge,
    sa_fep_stats_t* stats_out
) {
    if (!bridge || !stats_out) return -1;

    nimcp_mutex_lock(((sa_fep_bridge_t*)bridge)->base.mutex);
    *stats_out = bridge->stats;
    nimcp_mutex_unlock(((sa_fep_bridge_t*)bridge)->base.mutex);

    return 0;
}

int sa_fep_bridge_reset_stats(sa_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(sa_fep_stats_t));
    bridge->running_avg_fe = 0.0f;
    bridge->running_count = 0;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float sa_fep_bridge_get_free_energy(const sa_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((sa_fep_bridge_t*)bridge)->base.mutex);
    float fe = bridge->metrics.free_energy;
    nimcp_mutex_unlock(((sa_fep_bridge_t*)bridge)->base.mutex);

    return fe;
}

float sa_fep_bridge_get_salience_error(const sa_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((sa_fep_bridge_t*)bridge)->base.mutex);
    float se = bridge->metrics.salience_prediction_error;
    nimcp_mutex_unlock(((sa_fep_bridge_t*)bridge)->base.mutex);

    return se;
}

float sa_fep_bridge_get_prediction_error(const sa_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((sa_fep_bridge_t*)bridge)->base.mutex);
    float pe = bridge->metrics.prediction_error;
    nimcp_mutex_unlock(((sa_fep_bridge_t*)bridge)->base.mutex);

    return pe;
}

float sa_fep_bridge_get_attention_efficiency(const sa_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((sa_fep_bridge_t*)bridge)->base.mutex);
    float eff = bridge->metrics.attention_efficiency;
    nimcp_mutex_unlock(((sa_fep_bridge_t*)bridge)->base.mutex);

    return eff;
}

/*=============================================================================
 * STATE QUERY FUNCTIONS
 *===========================================================================*/

sa_fep_state_t sa_fep_bridge_get_state(const sa_fep_bridge_t* bridge) {
    if (!bridge) return SA_FEP_STATE_ERROR;

    nimcp_mutex_lock(((sa_fep_bridge_t*)bridge)->base.mutex);
    sa_fep_state_t state = bridge->state;
    nimcp_mutex_unlock(((sa_fep_bridge_t*)bridge)->base.mutex);

    return state;
}

bool sa_fep_bridge_is_degraded(const sa_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((sa_fep_bridge_t*)bridge)->base.mutex);
    bool degraded = (bridge->state == SA_FEP_STATE_DEGRADED);
    nimcp_mutex_unlock(((sa_fep_bridge_t*)bridge)->base.mutex);

    return degraded;
}

bool sa_fep_bridge_is_efficient(const sa_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((sa_fep_bridge_t*)bridge)->base.mutex);
    bool efficient = (bridge->metrics.attention_efficiency >=
                      bridge->config.attention_efficiency_threshold);
    nimcp_mutex_unlock(((sa_fep_bridge_t*)bridge)->base.mutex);

    return efficient;
}

const char* sa_fep_state_name(sa_fep_state_t state) {
    switch (state) {
        case SA_FEP_STATE_UNINITIALIZED: return "uninitialized";
        case SA_FEP_STATE_IDLE:          return "idle";
        case SA_FEP_STATE_ACTIVE:        return "active";
        case SA_FEP_STATE_DEGRADED:      return "degraded";
        case SA_FEP_STATE_ERROR:         return "error";
        default:                          return "unknown";
    }
}

/*=============================================================================
 * CALLBACK REGISTRATION
 *===========================================================================*/

int sa_fep_bridge_set_high_fe_callback(
    sa_fep_bridge_t* bridge,
    sa_fep_high_fe_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->high_fe_callback = callback;
    bridge->high_fe_user_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int sa_fep_bridge_set_surprise_callback(
    sa_fep_bridge_t* bridge,
    sa_fep_surprise_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->surprise_callback = callback;
    bridge->surprise_user_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int sa_fep_bridge_set_metrics_callback(
    sa_fep_bridge_t* bridge,
    sa_fep_metrics_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->metrics_callback = callback;
    bridge->metrics_user_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/*=============================================================================
 * CONFIGURATION
 *===========================================================================*/

int sa_fep_bridge_set_config(
    sa_fep_bridge_t* bridge,
    const sa_fep_config_t* config
) {
    if (!bridge || !config) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->config = *config;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int sa_fep_bridge_get_config(
    const sa_fep_bridge_t* bridge,
    sa_fep_config_t* config_out
) {
    if (!bridge || !config_out) return -1;

    nimcp_mutex_lock(((sa_fep_bridge_t*)bridge)->base.mutex);
    *config_out = bridge->config;
    nimcp_mutex_unlock(((sa_fep_bridge_t*)bridge)->base.mutex);

    return 0;
}
