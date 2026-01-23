/**
 * @file nimcp_parietal_fep_bridge.c
 * @brief Parietal Lobe - FEP Orchestrator Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-08
 *
 * WHAT: Implements FEP registration and update callbacks for parietal lobe
 * WHY: Enable coordinated free energy minimization across spatial and mathematical processing
 * HOW: Compute free energy from spatial uncertainty, body schema errors, and math prediction errors
 *
 * FEP MODEL FOR PARIETAL CORTEX:
 * - Spatial uncertainty contributes to free energy (uncertain object positions)
 * - Body schema errors contribute to free energy (mismatched proprioception)
 * - Mathematical prediction errors contribute to free energy (wrong estimates)
 * - Accurate spatial predictions minimize free energy
 *
 * @author NIMCP Development Team
 */

#include "cognitive/parietal/nimcp_parietal_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/parietal/nimcp_parietal.h"
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

struct parietal_fep_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */

    /* Configuration */
    parietal_fep_config_t config;

    /* State */
    parietal_fep_state_t state;

    /* References */
    fep_orchestrator_t* orchestrator;
    parietal_lobe_t* parietal;
    uint32_t bridge_id;
    bool registered;

    /* Metrics */
    parietal_fep_metrics_t metrics;

    /* Previous state for change detection */
    float prev_free_energy;
    float prev_prediction_error;
    float prev_spatial_uncertainty;
    float prev_body_schema_error;

    /* Running averages */
    float running_avg_fe;
    float running_avg_pe;
    uint64_t running_count;

    /* Statistics */
    parietal_fep_stats_t stats;

    /* Callbacks */
    parietal_fep_high_fe_callback_t high_fe_callback;
    void* high_fe_user_data;
    parietal_fep_surprise_callback_t surprise_callback;
    void* surprise_user_data;
    parietal_fep_metrics_callback_t metrics_callback;
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
 * @brief Compute free energy from parietal metrics
 *
 * Free energy model:
 * FE = baseline + spatial_contrib + body_schema_contrib + math_contrib
 *
 * Where:
 * - spatial_contrib = spatial_uncertainty * spatial_weight
 * - body_schema_contrib = body_schema_error * body_schema_weight
 * - math_contrib = math_prediction_error * math_weight
 */
static void compute_free_energy(
    parietal_fep_bridge_t* bridge,
    const parietal_stats_t* parietal_stats
) {
    parietal_fep_metrics_t* m = &bridge->metrics;
    const parietal_fep_config_t* cfg = &bridge->config;

    /* Extract spatial uncertainty from parietal statistics */
    /* Total spatial operations = rotations + transforms + queries */
    uint64_t spatial_total = parietal_stats->spatial.rotations_performed +
                             parietal_stats->spatial.transforms_performed +
                             parietal_stats->spatial.spatial_queries;

    /* Use failed_requests as proxy for spatial failures (normalized by total) */
    if (spatial_total > 0) {
        /* Higher failure rate = higher uncertainty */
        float failure_rate = (parietal_stats->total_requests > 0) ?
            (float)parietal_stats->failed_requests / (float)parietal_stats->total_requests :
            0.0f;
        m->spatial_uncertainty = clamp_f(failure_rate, 0.0f, 1.0f);
    } else {
        /* No operations yet - assume low uncertainty */
        m->spatial_uncertainty = 0.1f;
    }

    /* Body schema error from transformation failures and low confidence */
    /* High inflammation or fatigue indicates body schema degradation */
    float inflammation = parietal_stats->current_inflammation;
    float fatigue = parietal_stats->current_fatigue;
    float body_schema_factor = (inflammation + fatigue) / 2.0f;

    /* Also consider rotation/transform operations */
    m->body_schema_error = clamp_f(
        body_schema_factor * 0.5f + (1.0f - parietal_stats->avg_confidence) * 0.5f,
        0.0f, 1.0f
    );

    /* Mathematical prediction error from number sense and equation results */
    /* Sum individual operation counts from submodules */
    uint64_t math_ops = parietal_stats->number_sense.estimates_performed +
                        parietal_stats->number_sense.comparisons_performed +
                        parietal_stats->number_sense.arithmetic_operations +
                        parietal_stats->math_intuition.patterns_detected +
                        parietal_stats->math_intuition.symmetries_detected +
                        parietal_stats->math_intuition.analogies_solved +
                        parietal_stats->equation.expressions_parsed +
                        parietal_stats->equation.simplifications +
                        parietal_stats->equation.differentiations +
                        parietal_stats->equation.evaluations +
                        parietal_stats->equation.equations_solved;
    uint64_t math_failed = parietal_stats->failed_requests;

    if (math_ops > 0) {
        m->math_prediction_error = clamp_f(
            (float)math_failed / (float)math_ops,
            0.0f, 1.0f
        );
    } else {
        m->math_prediction_error = 0.0f;
    }

    /* Compute free energy contributions */
    m->spatial_contribution = m->spatial_uncertainty * cfg->spatial_uncertainty_weight;
    m->body_schema_contribution = m->body_schema_error * cfg->body_schema_error_weight;
    m->math_contribution = m->math_prediction_error * cfg->math_error_weight;

    /* Total free energy */
    float total_fe = cfg->baseline_free_energy +
                     m->spatial_contribution +
                     m->body_schema_contribution +
                     m->math_contribution;

    m->free_energy = clamp_f(total_fe, 0.0f, cfg->max_free_energy);

    /* Prediction error: EMA with decay */
    float current_error = (m->spatial_uncertainty + m->body_schema_error +
                          m->math_prediction_error) / 3.0f;

    m->prediction_error = clamp_f(
        current_error * cfg->error_decay_rate +
        bridge->prev_prediction_error * (1.0f - cfg->error_decay_rate),
        0.0f, 1.0f
    );

    /* Surprise: unexpected changes in state */
    float fe_change = fabsf(m->free_energy - bridge->prev_free_energy);
    float spatial_change = fabsf(m->spatial_uncertainty - bridge->prev_spatial_uncertainty);
    float body_change = fabsf(m->body_schema_error - bridge->prev_body_schema_error);

    m->surprise = clamp_f(
        (fe_change * 0.4f + spatial_change * 0.3f + body_change * 0.3f),
        0.0f, 1.0f
    );

    /* Entropy: state uncertainty based on current processing */
    float uncertainty_sum = m->spatial_uncertainty + m->body_schema_error +
                           m->math_prediction_error;
    m->entropy = clamp_f(uncertainty_sum / 3.0f, 0.0f, 1.0f);

    /* Update operation counts */
    m->spatial_computations = (uint32_t)(parietal_stats->spatial.rotations_performed +
                                          parietal_stats->spatial.transforms_performed +
                                          parietal_stats->spatial.spatial_queries);
    m->body_schema_updates = (uint32_t)(parietal_stats->spatial.transforms_performed);
    m->math_operations = (uint32_t)math_ops;
}

/**
 * @brief Check and trigger callbacks
 */
static void check_callbacks(parietal_fep_bridge_t* bridge) {
    parietal_fep_metrics_t* m = &bridge->metrics;
    const parietal_fep_config_t* cfg = &bridge->config;

    /* Check for high free energy */
    if (cfg->enable_degraded_mode && m->free_energy > cfg->high_free_energy_threshold) {
        if (bridge->state != PARIETAL_FEP_STATE_DEGRADED) {
            bridge->state = PARIETAL_FEP_STATE_DEGRADED;
            bridge->stats.degraded_mode_entries++;

            if (cfg->enable_callbacks && bridge->high_fe_callback) {
                bridge->high_fe_callback(bridge, m->free_energy,
                                         bridge->high_fe_user_data);
            }
        }
    } else if (bridge->state == PARIETAL_FEP_STATE_DEGRADED) {
        bridge->state = PARIETAL_FEP_STATE_ACTIVE;
    }

    /* Check for surprise events */
    if (cfg->enable_callbacks &&
        m->surprise > cfg->prediction_error_threshold) {
        bridge->stats.surprise_events++;

        if (bridge->surprise_callback) {
            const char* source = "unknown";
            if (m->spatial_contribution > m->body_schema_contribution &&
                m->spatial_contribution > m->math_contribution) {
                source = "spatial";
            } else if (m->body_schema_contribution > m->math_contribution) {
                source = "body_schema";
            } else {
                source = "math";
            }
            bridge->surprise_callback(bridge, m->surprise, source,
                                      bridge->surprise_user_data);
        }
    }

    /* Metrics callback */
    if (cfg->enable_callbacks && bridge->metrics_callback) {
        bridge->metrics_callback(bridge, m, bridge->metrics_user_data);
    }
}

/**
 * @brief Update statistics
 */
static void update_stats(parietal_fep_bridge_t* bridge, uint64_t update_time_us) {
    parietal_fep_stats_t* s = &bridge->stats;
    parietal_fep_metrics_t* m = &bridge->metrics;

    s->total_updates++;
    s->total_update_time_us += update_time_us;
    s->spatial_computations += m->spatial_computations;
    s->body_schema_updates += m->body_schema_updates;
    s->math_operations += m->math_operations;
    s->total_free_energy_contribution += m->free_energy;

    /* Peak tracking */
    if (m->free_energy > s->peak_free_energy) {
        s->peak_free_energy = m->free_energy;
    }

    /* Running averages */
    bridge->running_count++;
    bridge->running_avg_fe = (bridge->running_avg_fe * (bridge->running_count - 1) +
                              m->free_energy) / bridge->running_count;
    bridge->running_avg_pe = (bridge->running_avg_pe * (bridge->running_count - 1) +
                              m->prediction_error) / bridge->running_count;

    s->avg_free_energy = bridge->running_avg_fe;
    s->avg_prediction_error = bridge->running_avg_pe;
    s->avg_update_time_us = (float)s->total_update_time_us / (float)s->total_updates;

    /* Update metrics timing */
    m->last_update_time_ms = get_time_ms();
    m->update_count++;
    m->avg_update_time_us = s->avg_update_time_us;
}

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *===========================================================================*/

parietal_fep_config_t parietal_fep_config_default(void) {
    parietal_fep_config_t config;
    memset(&config, 0, sizeof(config));

    /* Logging and timing */
    config.enable_logging = false;
    config.update_interval_ms = PARIETAL_FEP_DEFAULT_UPDATE_MS;

    /* Weighting parameters */
    config.free_energy_weight = 1.0f;
    config.spatial_uncertainty_weight = PARIETAL_FEP_SPATIAL_WEIGHT;
    config.body_schema_error_weight = PARIETAL_FEP_BODY_SCHEMA_WEIGHT;
    config.math_error_weight = PARIETAL_FEP_MATH_WEIGHT;

    /* Thresholds */
    config.high_free_energy_threshold = 1.5f;
    config.prediction_error_threshold = 0.5f;

    /* Normalization */
    config.baseline_free_energy = PARIETAL_FEP_BASELINE_FREE_ENERGY;
    config.max_free_energy = PARIETAL_FEP_MAX_FREE_ENERGY;

    /* Behavior */
    config.enable_adaptive_weights = true;
    config.enable_degraded_mode = true;
    config.enable_callbacks = true;
    config.error_decay_rate = PARIETAL_FEP_ERROR_DECAY_RATE;

    return config;
}

parietal_fep_bridge_t* parietal_fep_bridge_create(const parietal_fep_config_t* config) {
    parietal_fep_bridge_t* bridge = nimcp_calloc(1, sizeof(parietal_fep_bridge_t));
    if (!bridge) return NULL;

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = parietal_fep_config_default();
    }

    /* Initialize bridge base infrastructure (includes mutex) */
    if (bridge_base_init(&bridge->base, 0, "parietal_fep") != 0) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->state = PARIETAL_FEP_STATE_UNINITIALIZED;
    bridge->registered = false;
    bridge->bridge_id = 0;

    /* Initialize metrics to baseline */
    bridge->metrics.free_energy = bridge->config.baseline_free_energy;
    bridge->metrics.prediction_error = 0.0f;
    bridge->metrics.surprise = 0.0f;
    bridge->metrics.entropy = 0.0f;
    bridge->metrics.spatial_uncertainty = 0.1f;
    bridge->metrics.body_schema_error = 0.0f;
    bridge->metrics.math_prediction_error = 0.0f;

    /* Initialize previous state tracking */
    bridge->prev_free_energy = bridge->config.baseline_free_energy;
    bridge->prev_prediction_error = 0.0f;
    bridge->prev_spatial_uncertainty = 0.1f;
    bridge->prev_body_schema_error = 0.0f;

    bridge->state = PARIETAL_FEP_STATE_IDLE;

    return bridge;
}

void parietal_fep_bridge_destroy(parietal_fep_bridge_t* bridge) {
    if (!bridge) return;

    /* Unregister if still registered */
    if (bridge->registered) {
        parietal_fep_bridge_unregister(bridge);
    }

    /* Cleanup base bridge infrastructure */
    bridge_base_cleanup(&bridge->base);

    nimcp_free(bridge);
}

int parietal_fep_bridge_reset(parietal_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Reset metrics */
    memset(&bridge->metrics, 0, sizeof(parietal_fep_metrics_t));
    bridge->metrics.free_energy = bridge->config.baseline_free_energy;
    bridge->metrics.spatial_uncertainty = 0.1f;

    /* Reset previous state */
    bridge->prev_free_energy = bridge->config.baseline_free_energy;
    bridge->prev_prediction_error = 0.0f;
    bridge->prev_spatial_uncertainty = 0.1f;
    bridge->prev_body_schema_error = 0.0f;

    /* Reset running averages */
    bridge->running_avg_fe = 0.0f;
    bridge->running_avg_pe = 0.0f;
    bridge->running_count = 0;

    /* Reset statistics */
    memset(&bridge->stats, 0, sizeof(parietal_fep_stats_t));

    bridge->state = PARIETAL_FEP_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/*=============================================================================
 * REGISTRATION FUNCTIONS
 *===========================================================================*/

int parietal_fep_bridge_register(
    parietal_fep_bridge_t* bridge,
    fep_orchestrator_t* orchestrator,
    parietal_lobe_t* parietal,
    uint32_t* bridge_id_out
) {
    if (!bridge || !orchestrator) return -1;  /* parietal can be NULL for standalone testing */

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
    bridge->parietal = parietal;

    /* Register with FEP orchestrator */
    uint32_t assigned_id = 0;
    int ret = fep_orchestrator_register_bridge(
        orchestrator,
        "parietal_lobe",
        FEP_BRIDGE_CATEGORY_COGNITIVE,  /* 50ms cognitive timescale */
        (fep_bridge_handle_t)bridge,
        parietal_fep_update_callback,
        parietal_fep_destroy_callback,
        &assigned_id
    );

    if (ret != 0) {
        bridge->orchestrator = NULL;
        bridge->parietal = NULL;
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    bridge->bridge_id = assigned_id;
    bridge->registered = true;
    bridge->state = PARIETAL_FEP_STATE_ACTIVE;

    if (bridge_id_out) {
        *bridge_id_out = assigned_id;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int parietal_fep_bridge_unregister(parietal_fep_bridge_t* bridge) {
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
    bridge->parietal = NULL;
    bridge->state = PARIETAL_FEP_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

bool parietal_fep_bridge_is_registered(const parietal_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((parietal_fep_bridge_t*)bridge)->base.mutex);
    bool registered = bridge->registered;
    nimcp_mutex_unlock(((parietal_fep_bridge_t*)bridge)->base.mutex);

    return registered;
}

uint32_t parietal_fep_bridge_get_id(const parietal_fep_bridge_t* bridge) {
    if (!bridge) return 0;

    nimcp_mutex_lock(((parietal_fep_bridge_t*)bridge)->base.mutex);
    uint32_t id = bridge->registered ? bridge->bridge_id : 0;
    nimcp_mutex_unlock(((parietal_fep_bridge_t*)bridge)->base.mutex);

    return id;
}

/*=============================================================================
 * FEP UPDATE CALLBACK
 *===========================================================================*/

int parietal_fep_update_callback(void* handle) {
    parietal_fep_bridge_t* bridge = (parietal_fep_bridge_t*)handle;
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Ensure we're registered and have a parietal module */
    if (!bridge->registered || !bridge->parietal) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    uint64_t start_us = get_time_us();

    /* Get parietal statistics */
    parietal_stats_t parietal_stats;
    memset(&parietal_stats, 0, sizeof(parietal_stats));

    int ret = parietal_get_stats(bridge->parietal, &parietal_stats);
    if (ret != 0) {
        /* Stats not available, perform minimal update */
        bridge->metrics.last_update_time_ms = get_time_ms();
        bridge->metrics.update_count++;
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;  /* Non-critical failure */
    }

    /* Store previous state for delta computation */
    bridge->prev_free_energy = bridge->metrics.free_energy;
    bridge->prev_prediction_error = bridge->metrics.prediction_error;
    bridge->prev_spatial_uncertainty = bridge->metrics.spatial_uncertainty;
    bridge->prev_body_schema_error = bridge->metrics.body_schema_error;

    /* Compute free energy from parietal stats */
    compute_free_energy(bridge, &parietal_stats);

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

void parietal_fep_destroy_callback(void* handle) {
    /* No-op: Bridge is destroyed separately via parietal_fep_bridge_destroy() */
    (void)handle;
}

/*=============================================================================
 * OPERATIONS
 *===========================================================================*/

int parietal_fep_bridge_update(parietal_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    /* If registered with orchestrator and has parietal module, do full update */
    if (bridge->registered && bridge->parietal) {
        return parietal_fep_update_callback(bridge);
    }

    return -1;  /* Cannot update without registration */
}

int parietal_fep_bridge_force_update(parietal_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    /* If registered with orchestrator and has parietal module, do full update */
    if (bridge->registered && bridge->parietal) {
        return parietal_fep_update_callback(bridge);
    }

    /* For unit testing: perform minimal update with callback invocation */
    nimcp_mutex_lock(bridge->base.mutex);

    uint64_t start_us = get_time_us();

    /* Store previous state */
    bridge->prev_free_energy = bridge->metrics.free_energy;
    bridge->prev_prediction_error = bridge->metrics.prediction_error;

    /* Simulate minimal metrics update */
    bridge->metrics.last_update_time_ms = get_time_ms();
    bridge->metrics.update_count++;

    /* Calculate update time */
    uint64_t end_us = get_time_us();
    uint64_t update_time_us = end_us - start_us;

    /* Update stats */
    bridge->stats.total_updates++;
    bridge->stats.total_update_time_us += update_time_us;

    /* Invoke callbacks even without full parietal data */
    check_callbacks(bridge);

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/*=============================================================================
 * METRICS AND STATISTICS
 *===========================================================================*/

int parietal_fep_bridge_get_metrics(
    const parietal_fep_bridge_t* bridge,
    parietal_fep_metrics_t* metrics_out
) {
    if (!bridge || !metrics_out) return -1;

    nimcp_mutex_lock(((parietal_fep_bridge_t*)bridge)->base.mutex);
    *metrics_out = bridge->metrics;
    nimcp_mutex_unlock(((parietal_fep_bridge_t*)bridge)->base.mutex);

    return 0;
}

int parietal_fep_bridge_get_stats(
    const parietal_fep_bridge_t* bridge,
    parietal_fep_stats_t* stats_out
) {
    if (!bridge || !stats_out) return -1;

    nimcp_mutex_lock(((parietal_fep_bridge_t*)bridge)->base.mutex);
    *stats_out = bridge->stats;
    nimcp_mutex_unlock(((parietal_fep_bridge_t*)bridge)->base.mutex);

    return 0;
}

int parietal_fep_bridge_reset_stats(parietal_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(parietal_fep_stats_t));
    bridge->running_avg_fe = 0.0f;
    bridge->running_avg_pe = 0.0f;
    bridge->running_count = 0;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/*=============================================================================
 * ACCESSORS
 *===========================================================================*/

float parietal_fep_bridge_get_free_energy_contribution(const parietal_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((parietal_fep_bridge_t*)bridge)->base.mutex);
    float fe = bridge->metrics.free_energy;
    nimcp_mutex_unlock(((parietal_fep_bridge_t*)bridge)->base.mutex);

    return fe;
}

float parietal_fep_bridge_get_spatial_uncertainty(const parietal_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((parietal_fep_bridge_t*)bridge)->base.mutex);
    float uncertainty = bridge->metrics.spatial_uncertainty;
    nimcp_mutex_unlock(((parietal_fep_bridge_t*)bridge)->base.mutex);

    return uncertainty;
}

float parietal_fep_bridge_get_body_schema_error(const parietal_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((parietal_fep_bridge_t*)bridge)->base.mutex);
    float error = bridge->metrics.body_schema_error;
    nimcp_mutex_unlock(((parietal_fep_bridge_t*)bridge)->base.mutex);

    return error;
}

float parietal_fep_bridge_get_prediction_error(const parietal_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((parietal_fep_bridge_t*)bridge)->base.mutex);
    float pe = bridge->metrics.prediction_error;
    nimcp_mutex_unlock(((parietal_fep_bridge_t*)bridge)->base.mutex);

    return pe;
}

parietal_fep_state_t parietal_fep_bridge_get_state(const parietal_fep_bridge_t* bridge) {
    if (!bridge) return PARIETAL_FEP_STATE_ERROR;

    nimcp_mutex_lock(((parietal_fep_bridge_t*)bridge)->base.mutex);
    parietal_fep_state_t state = bridge->state;
    nimcp_mutex_unlock(((parietal_fep_bridge_t*)bridge)->base.mutex);

    return state;
}

bool parietal_fep_bridge_is_degraded(const parietal_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((parietal_fep_bridge_t*)bridge)->base.mutex);
    bool degraded = (bridge->state == PARIETAL_FEP_STATE_DEGRADED);
    nimcp_mutex_unlock(((parietal_fep_bridge_t*)bridge)->base.mutex);

    return degraded;
}

/*=============================================================================
 * CALLBACK REGISTRATION
 *===========================================================================*/

int parietal_fep_bridge_set_high_fe_callback(
    parietal_fep_bridge_t* bridge,
    parietal_fep_high_fe_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->high_fe_callback = callback;
    bridge->high_fe_user_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int parietal_fep_bridge_set_surprise_callback(
    parietal_fep_bridge_t* bridge,
    parietal_fep_surprise_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->surprise_callback = callback;
    bridge->surprise_user_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int parietal_fep_bridge_set_metrics_callback(
    parietal_fep_bridge_t* bridge,
    parietal_fep_metrics_callback_t callback,
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

int parietal_fep_bridge_set_config(
    parietal_fep_bridge_t* bridge,
    const parietal_fep_config_t* config
) {
    if (!bridge || !config) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->config = *config;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int parietal_fep_bridge_get_config(
    const parietal_fep_bridge_t* bridge,
    parietal_fep_config_t* config_out
) {
    if (!bridge || !config_out) return -1;

    nimcp_mutex_lock(((parietal_fep_bridge_t*)bridge)->base.mutex);
    *config_out = bridge->config;
    nimcp_mutex_unlock(((parietal_fep_bridge_t*)bridge)->base.mutex);

    return 0;
}

/*=============================================================================
 * UTILITY FUNCTIONS
 *===========================================================================*/

const char* parietal_fep_state_name(parietal_fep_state_t state) {
    switch (state) {
        case PARIETAL_FEP_STATE_UNINITIALIZED: return "uninitialized";
        case PARIETAL_FEP_STATE_IDLE:          return "idle";
        case PARIETAL_FEP_STATE_ACTIVE:        return "active";
        case PARIETAL_FEP_STATE_DEGRADED:      return "degraded";
        case PARIETAL_FEP_STATE_ERROR:         return "error";
        default:                                return "unknown";
    }
}
