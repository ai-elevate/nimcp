/**
 * @file nimcp_mirror_empathy_fep_bridge.c
 * @brief Mirror-Empathy - FEP Orchestrator Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-08
 *
 * WHAT: Implements FEP registration and update callbacks for mirror-empathy
 * WHY:  Enable coordinated free energy minimization for social cognition
 * HOW:  Compute free energy from mirroring error, empathy prediction, resonance
 *
 * @author NIMCP Development Team
 */

#include "cognitive/integration/nimcp_mirror_empathy_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/integration/nimcp_mirror_empathy_bridge.h"
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

struct me_fep_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */

    /* Configuration */
    me_fep_config_t config;

    /* State */
    me_fep_state_t state;

    /* References */
    fep_orchestrator_t* orchestrator;
    mirror_empathy_bridge_t* me_bridge;
    uint32_t bridge_id;
    bool registered;

    /* Metrics */
    me_fep_metrics_t metrics;

    /* Previous state for change detection */
    float prev_free_energy;
    float prev_prediction_error;
    float prev_mirroring_error;
    float prev_empathy_error;
    float prev_resonance_deficit;

    /* Running averages */
    float running_avg_fe;
    uint64_t running_count;

    /* Statistics */
    me_fep_stats_t stats;

    /* Callbacks */
    me_fep_high_fe_callback_t high_fe_callback;
    void* high_fe_user_data;
    me_fep_surprise_callback_t surprise_callback;
    void* surprise_user_data;
    me_fep_metrics_callback_t metrics_callback;
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
 * @brief Compute free energy from mirror-empathy metrics
 *
 * Free energy model for social cognition:
 * FE = baseline + mirroring_contrib + empathy_contrib + resonance_contrib
 *
 * Where:
 * - mirroring_contrib = mirroring_error * mirroring_weight
 *   (higher action mirroring error = higher FE)
 * - empathy_contrib = empathy_prediction_error * empathy_weight
 *   (worse empathy prediction = higher FE)
 * - resonance_contrib = resonance_deficit * resonance_weight
 *   (lower emotional resonance = higher FE)
 *
 * High resonance represents minimum free energy state where social
 * predictions are accurate and emotional connection is strong.
 */
static void compute_free_energy(me_fep_bridge_t* bridge) {
    me_fep_metrics_t* m = &bridge->metrics;
    const me_fep_config_t* cfg = &bridge->config;

    /* Clamp input metrics */
    m->mirroring_error = clamp_f(m->mirroring_error, 0.0f, 1.0f);
    m->empathy_prediction_error = clamp_f(m->empathy_prediction_error, 0.0f, 1.0f);
    m->resonance_deficit = clamp_f(m->resonance_deficit, 0.0f, 1.0f);

    /* Mirroring accuracy contribution:
     * High error in understanding observed actions = prediction error */
    m->mirroring_contribution = m->mirroring_error * cfg->mirroring_accuracy_weight;

    /* Empathy prediction contribution:
     * Poor predictions of others' emotional states = prediction error */
    m->empathy_contribution = m->empathy_prediction_error * cfg->empathy_prediction_weight;

    /* Emotional resonance contribution:
     * Deficit in resonance represents failure to achieve shared affect
     * High resonance = successful social prediction = low free energy */
    m->resonance_contribution = m->resonance_deficit * cfg->emotional_resonance_weight;

    /* Total free energy */
    float total_fe = cfg->baseline_free_energy +
                     (m->mirroring_contribution +
                      m->empathy_contribution +
                      m->resonance_contribution) * cfg->free_energy_weight;

    m->free_energy = clamp_f(total_fe, 0.0f, cfg->max_free_energy);

    /* High resonance state check (low deficit = high resonance) */
    m->high_resonance_state = (m->resonance_deficit < cfg->resonance_epsilon);

    /* Prediction error: weighted combination of all uncertainty sources */
    float new_prediction_error = (m->mirroring_error * 0.35f +
                                   m->empathy_prediction_error * 0.35f +
                                   m->resonance_deficit * 0.30f);

    /* Apply decay to smooth transitions */
    m->prediction_error = clamp_f(
        new_prediction_error * cfg->error_decay_rate +
        bridge->prev_prediction_error * (1.0f - cfg->error_decay_rate),
        0.0f, 1.0f
    );

    /* Surprise: unexpected changes in social state */
    float fe_change = fabsf(m->free_energy - bridge->prev_free_energy);
    float mirroring_change = fabsf(m->mirroring_error - bridge->prev_mirroring_error);
    float empathy_change = fabsf(m->empathy_prediction_error - bridge->prev_empathy_error);
    float resonance_change = fabsf(m->resonance_deficit - bridge->prev_resonance_deficit);

    m->surprise = clamp_f(
        (fe_change * 0.3f + mirroring_change * 0.25f +
         empathy_change * 0.25f + resonance_change * 0.2f),
        0.0f, 1.0f
    );

    /* Entropy: based on prediction uncertainty
     * Higher uncertainty about social states = higher entropy */
    m->entropy = clamp_f(
        m->mirroring_error * 0.4f + m->empathy_prediction_error * 0.4f +
        m->resonance_deficit * 0.2f,
        0.0f, 1.0f
    );

    /* Intention uncertainty approximation from social prediction errors */
    m->intention_uncertainty = m->mirroring_error * m->empathy_prediction_error;
}

/**
 * @brief Check and trigger callbacks
 */
static void check_callbacks(me_fep_bridge_t* bridge) {
    me_fep_metrics_t* m = &bridge->metrics;
    const me_fep_config_t* cfg = &bridge->config;

    /* Check for high free energy */
    if (m->free_energy > cfg->high_free_energy_threshold) {
        if (bridge->state != ME_FEP_STATE_DEGRADED) {
            bridge->state = ME_FEP_STATE_DEGRADED;
            bridge->stats.degraded_mode_entries++;

            if (bridge->high_fe_callback) {
                bridge->high_fe_callback(bridge, m->free_energy,
                                         bridge->high_fe_user_data);
            }
        }
    } else if (bridge->state == ME_FEP_STATE_DEGRADED) {
        bridge->state = ME_FEP_STATE_ACTIVE;
    }

    /* Check for surprise events */
    if (m->surprise > cfg->prediction_error_threshold) {
        bridge->stats.surprise_events++;

        if (bridge->surprise_callback) {
            const char* source = "unknown";
            if (m->mirroring_contribution > m->empathy_contribution &&
                m->mirroring_contribution > m->resonance_contribution) {
                source = "mirroring";
            } else if (m->empathy_contribution > m->resonance_contribution) {
                source = "empathy";
            } else {
                source = "resonance";
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
static void update_stats(me_fep_bridge_t* bridge, uint64_t update_time_us) {
    me_fep_stats_t* s = &bridge->stats;
    me_fep_metrics_t* m = &bridge->metrics;

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
static void store_previous_state(me_fep_bridge_t* bridge) {
    bridge->prev_free_energy = bridge->metrics.free_energy;
    bridge->prev_prediction_error = bridge->metrics.prediction_error;
    bridge->prev_mirroring_error = bridge->metrics.mirroring_error;
    bridge->prev_empathy_error = bridge->metrics.empathy_prediction_error;
    bridge->prev_resonance_deficit = bridge->metrics.resonance_deficit;
}

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *===========================================================================*/

me_fep_config_t me_fep_config_default(void) {
    me_fep_config_t config;
    memset(&config, 0, sizeof(config));

    /* Feature enables */
    config.enable_logging = false;
    config.update_interval_ms = 50;  /* Cognitive timescale */

    /* Weighting parameters - weight must be high enough for max errors to exceed threshold */
    config.free_energy_weight = 2.0f;
    config.mirroring_accuracy_weight = ME_FEP_MIRRORING_WEIGHT;
    config.empathy_prediction_weight = ME_FEP_EMPATHY_WEIGHT;
    config.emotional_resonance_weight = ME_FEP_RESONANCE_WEIGHT;

    /* Thresholds */
    config.high_free_energy_threshold = 1.5f;
    config.prediction_error_threshold = 0.5f;
    config.resonance_epsilon = 0.1f;

    /* Normalization */
    config.baseline_free_energy = ME_FEP_BASELINE_FREE_ENERGY;
    config.max_free_energy = ME_FEP_MAX_FREE_ENERGY;
    config.error_decay_rate = ME_FEP_ERROR_DECAY_RATE;

    return config;
}

me_fep_bridge_t* me_fep_bridge_create(const me_fep_config_t* config) {
    me_fep_bridge_t* bridge = nimcp_calloc(1, sizeof(me_fep_bridge_t));
    if (!bridge) return NULL;

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = me_fep_config_default();
    }

    /* Initialize base bridge infrastructure */
    if (bridge_base_init(&bridge->base, 0, "mirror_empathy_fep") != 0) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->state = ME_FEP_STATE_UNINITIALIZED;
    bridge->registered = false;
    bridge->bridge_id = 0;

    /* Initialize metrics to baseline */
    bridge->metrics.free_energy = bridge->config.baseline_free_energy;
    bridge->metrics.prediction_error = 0.0f;
    bridge->metrics.surprise = 0.0f;
    bridge->metrics.entropy = 0.0f;
    bridge->metrics.mirroring_error = 0.0f;
    bridge->metrics.empathy_prediction_error = 0.0f;
    bridge->metrics.resonance_deficit = 1.0f;  /* Start with low resonance */
    bridge->metrics.high_resonance_state = false;

    /* Initialize previous state tracking */
    bridge->prev_free_energy = bridge->config.baseline_free_energy;
    bridge->prev_prediction_error = 0.0f;
    bridge->prev_mirroring_error = 0.0f;
    bridge->prev_empathy_error = 0.0f;
    bridge->prev_resonance_deficit = 1.0f;

    bridge->state = ME_FEP_STATE_IDLE;

    return bridge;
}

void me_fep_bridge_destroy(me_fep_bridge_t* bridge) {
    if (!bridge) return;

    /* Unregister if still registered */
    if (bridge->registered) {
        me_fep_bridge_unregister(bridge);
    }

    /* Cleanup base bridge infrastructure */
    bridge_base_cleanup(&bridge->base);

    nimcp_free(bridge);
}

int me_fep_bridge_reset(me_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Reset metrics */
    memset(&bridge->metrics, 0, sizeof(me_fep_metrics_t));
    bridge->metrics.free_energy = bridge->config.baseline_free_energy;
    bridge->metrics.resonance_deficit = 1.0f;

    /* Reset previous state */
    bridge->prev_free_energy = bridge->config.baseline_free_energy;
    bridge->prev_prediction_error = 0.0f;
    bridge->prev_mirroring_error = 0.0f;
    bridge->prev_empathy_error = 0.0f;
    bridge->prev_resonance_deficit = 1.0f;

    /* Reset running averages */
    bridge->running_avg_fe = 0.0f;
    bridge->running_count = 0;

    /* Reset statistics */
    memset(&bridge->stats, 0, sizeof(me_fep_stats_t));

    bridge->state = ME_FEP_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/*=============================================================================
 * REGISTRATION FUNCTIONS
 *===========================================================================*/

int me_fep_bridge_register(
    me_fep_bridge_t* bridge,
    fep_orchestrator_t* orchestrator,
    mirror_empathy_bridge_t* me_bridge,
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
    bridge->me_bridge = me_bridge;

    /* Register with FEP orchestrator */
    uint32_t assigned_id = 0;
    int ret = fep_orchestrator_register_bridge(
        orchestrator,
        "mirror_empathy",
        FEP_BRIDGE_CATEGORY_COGNITIVE,  /* 50ms cognitive timescale */
        (fep_bridge_handle_t)bridge,
        me_fep_update_callback,
        me_fep_destroy_callback,
        &assigned_id
    );

    if (ret != 0) {
        bridge->orchestrator = NULL;
        bridge->me_bridge = NULL;
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    bridge->bridge_id = assigned_id;
    bridge->registered = true;
    bridge->state = ME_FEP_STATE_ACTIVE;

    if (bridge_id_out) {
        *bridge_id_out = assigned_id;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int me_fep_bridge_unregister(me_fep_bridge_t* bridge) {
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
    bridge->me_bridge = NULL;
    bridge->state = ME_FEP_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

bool me_fep_bridge_is_registered(const me_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((me_fep_bridge_t*)bridge)->base.mutex);
    bool registered = bridge->registered;
    nimcp_mutex_unlock(((me_fep_bridge_t*)bridge)->base.mutex);

    return registered;
}

uint32_t me_fep_bridge_get_id(const me_fep_bridge_t* bridge) {
    if (!bridge) return 0;

    nimcp_mutex_lock(((me_fep_bridge_t*)bridge)->base.mutex);
    uint32_t id = bridge->registered ? bridge->bridge_id : 0;
    nimcp_mutex_unlock(((me_fep_bridge_t*)bridge)->base.mutex);

    return id;
}

/*=============================================================================
 * FEP UPDATE CALLBACK
 *===========================================================================*/

int me_fep_update_callback(void* handle) {
    me_fep_bridge_t* bridge = (me_fep_bridge_t*)handle;
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

    /* If we have a mirror-empathy bridge, query it for statistics */
    if (bridge->me_bridge) {
        mirror_empathy_stats_t me_stats;
        memset(&me_stats, 0, sizeof(me_stats));

        int err = mirror_empathy_bridge_get_stats(bridge->me_bridge, &me_stats);
        if (err == 0) {
            /* Update metrics from mirror-empathy statistics */
            bridge->stats.mirroring_computations++;

            /* Estimate mirroring error from action understanding rate */
            if (me_stats.total_events > 0) {
                float action_success_rate = (float)me_stats.actions_mirrored /
                                           (float)me_stats.total_events;
                /* Lower success rate = higher mirroring error */
                bridge->metrics.mirroring_error = clamp_f(
                    1.0f - action_success_rate * 2.0f,  /* Scale up */
                    0.0f, 1.0f
                );
            }

            /* Estimate empathy prediction from response rate */
            if (me_stats.events_received > 0) {
                float response_rate = (float)me_stats.empathetic_responses /
                                     (float)me_stats.events_received;
                /* Lower response rate = higher prediction error */
                bridge->metrics.empathy_prediction_error = clamp_f(
                    1.0f - response_rate,
                    0.0f, 1.0f
                );
            }

            /* Resonance deficit from average resonance strength */
            bridge->metrics.resonance_deficit = clamp_f(
                1.0f - me_stats.avg_resonance_strength,
                0.0f, 1.0f
            );

            /* Update active interactions count */
            bridge->metrics.active_interactions = (uint32_t)me_stats.empathetic_responses;
            bridge->metrics.successful_predictions = (uint32_t)me_stats.intentions_predicted;

            bridge->stats.empathy_computations++;
            bridge->stats.resonance_computations++;
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

void me_fep_destroy_callback(void* handle) {
    /* No-op: Bridge is destroyed separately via me_fep_bridge_destroy() */
    (void)handle;
}

/*=============================================================================
 * MANUAL UPDATE OPERATIONS
 *===========================================================================*/

int me_fep_bridge_force_update(me_fep_bridge_t* bridge) {
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

int me_fep_bridge_update_mirroring_error(
    me_fep_bridge_t* bridge,
    float error
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->metrics.mirroring_error = clamp_f(error, 0.0f, 1.0f);
    bridge->stats.mirroring_computations++;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int me_fep_bridge_update_empathy_error(
    me_fep_bridge_t* bridge,
    float error
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->metrics.empathy_prediction_error = clamp_f(error, 0.0f, 1.0f);
    bridge->stats.empathy_computations++;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int me_fep_bridge_update_resonance_deficit(
    me_fep_bridge_t* bridge,
    float deficit
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->metrics.resonance_deficit = clamp_f(deficit, 0.0f, 1.0f);
    bridge->stats.resonance_computations++;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/*=============================================================================
 * METRICS AND STATISTICS
 *===========================================================================*/

int me_fep_bridge_get_metrics(
    const me_fep_bridge_t* bridge,
    me_fep_metrics_t* metrics_out
) {
    if (!bridge || !metrics_out) return -1;

    nimcp_mutex_lock(((me_fep_bridge_t*)bridge)->base.mutex);
    *metrics_out = bridge->metrics;
    nimcp_mutex_unlock(((me_fep_bridge_t*)bridge)->base.mutex);

    return 0;
}

int me_fep_bridge_get_stats(
    const me_fep_bridge_t* bridge,
    me_fep_stats_t* stats_out
) {
    if (!bridge || !stats_out) return -1;

    nimcp_mutex_lock(((me_fep_bridge_t*)bridge)->base.mutex);
    *stats_out = bridge->stats;
    nimcp_mutex_unlock(((me_fep_bridge_t*)bridge)->base.mutex);

    return 0;
}

int me_fep_bridge_reset_stats(me_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(me_fep_stats_t));
    bridge->running_avg_fe = 0.0f;
    bridge->running_count = 0;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float me_fep_bridge_get_free_energy(const me_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((me_fep_bridge_t*)bridge)->base.mutex);
    float fe = bridge->metrics.free_energy;
    nimcp_mutex_unlock(((me_fep_bridge_t*)bridge)->base.mutex);

    return fe;
}

float me_fep_bridge_get_mirroring_error(const me_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((me_fep_bridge_t*)bridge)->base.mutex);
    float me = bridge->metrics.mirroring_error;
    nimcp_mutex_unlock(((me_fep_bridge_t*)bridge)->base.mutex);

    return me;
}

float me_fep_bridge_get_prediction_error(const me_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((me_fep_bridge_t*)bridge)->base.mutex);
    float pe = bridge->metrics.prediction_error;
    nimcp_mutex_unlock(((me_fep_bridge_t*)bridge)->base.mutex);

    return pe;
}

/*=============================================================================
 * STATE QUERY FUNCTIONS
 *===========================================================================*/

me_fep_state_t me_fep_bridge_get_state(const me_fep_bridge_t* bridge) {
    if (!bridge) return ME_FEP_STATE_ERROR;

    nimcp_mutex_lock(((me_fep_bridge_t*)bridge)->base.mutex);
    me_fep_state_t state = bridge->state;
    nimcp_mutex_unlock(((me_fep_bridge_t*)bridge)->base.mutex);

    return state;
}

bool me_fep_bridge_is_degraded(const me_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((me_fep_bridge_t*)bridge)->base.mutex);
    bool degraded = (bridge->state == ME_FEP_STATE_DEGRADED);
    nimcp_mutex_unlock(((me_fep_bridge_t*)bridge)->base.mutex);

    return degraded;
}

bool me_fep_bridge_is_high_resonance(const me_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((me_fep_bridge_t*)bridge)->base.mutex);
    bool high_res = bridge->metrics.high_resonance_state;
    nimcp_mutex_unlock(((me_fep_bridge_t*)bridge)->base.mutex);

    return high_res;
}

const char* me_fep_state_name(me_fep_state_t state) {
    switch (state) {
        case ME_FEP_STATE_UNINITIALIZED: return "uninitialized";
        case ME_FEP_STATE_IDLE:          return "idle";
        case ME_FEP_STATE_ACTIVE:        return "active";
        case ME_FEP_STATE_DEGRADED:      return "degraded";
        case ME_FEP_STATE_ERROR:         return "error";
        default:                          return "unknown";
    }
}

/*=============================================================================
 * CALLBACK REGISTRATION
 *===========================================================================*/

int me_fep_bridge_set_high_fe_callback(
    me_fep_bridge_t* bridge,
    me_fep_high_fe_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->high_fe_callback = callback;
    bridge->high_fe_user_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int me_fep_bridge_set_surprise_callback(
    me_fep_bridge_t* bridge,
    me_fep_surprise_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->surprise_callback = callback;
    bridge->surprise_user_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int me_fep_bridge_set_metrics_callback(
    me_fep_bridge_t* bridge,
    me_fep_metrics_callback_t callback,
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

int me_fep_bridge_set_config(
    me_fep_bridge_t* bridge,
    const me_fep_config_t* config
) {
    if (!bridge || !config) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->config = *config;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int me_fep_bridge_get_config(
    const me_fep_bridge_t* bridge,
    me_fep_config_t* config_out
) {
    if (!bridge || !config_out) return -1;

    nimcp_mutex_lock(((me_fep_bridge_t*)bridge)->base.mutex);
    *config_out = bridge->config;
    nimcp_mutex_unlock(((me_fep_bridge_t*)bridge)->base.mutex);

    return 0;
}
