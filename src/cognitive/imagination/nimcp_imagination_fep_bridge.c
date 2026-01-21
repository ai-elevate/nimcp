/**
 * @file nimcp_imagination_fep_bridge.c
 * @brief Imagination - FEP Orchestrator Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-08
 *
 * WHAT: Implements FEP registration and update callbacks for imagination engine
 * WHY:  Enable coordinated free energy minimization across mental simulation
 * HOW:  Compute free energy from simulation divergence, counterfactual costs,
 *       coherence penalties, and prediction accuracy bonuses
 *
 * @author NIMCP Development Team
 */

#include "cognitive/imagination/nimcp_imagination_fep_bridge.h"
#include "cognitive/imagination/nimcp_imagination_engine.h"
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

struct imagination_fep_bridge {
    /* Configuration */
    imagination_fep_config_t config;

    /* State */
    imagination_fep_state_t state;
    nimcp_mutex_t* mutex;

    /* References */
    fep_orchestrator_t* orchestrator;
    imagination_engine_t* engine;
    uint32_t bridge_id;
    bool registered;

    /* Current metrics */
    float free_energy;
    float simulation_divergence;
    float prediction_error;
    float coherence;
    float counterfactual_cost_accumulated;

    /* Previous state for delta computation */
    float prev_free_energy;
    float prev_divergence;
    float prev_coherence;

    /* Running averages */
    float running_avg_fe;
    float running_avg_divergence;
    float running_avg_counterfactual;
    uint64_t running_count;

    /* Statistics */
    imagination_fep_stats_t stats;

    /* Timing */
    uint64_t last_update_time_ms;
    uint64_t total_update_time_us;

    /* Callbacks */
    imagination_fep_high_fe_callback_t high_fe_callback;
    void* high_fe_user_data;
    imagination_fep_divergence_callback_t divergence_callback;
    void* divergence_user_data;
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
 * @brief Compute free energy from imagination metrics
 *
 * Free energy model for imagination:
 * FE = baseline + divergence_contrib + counterfactual_cost - prediction_bonus + coherence_penalty
 *
 * Where:
 * - divergence_contrib = simulation_divergence * divergence_weight (reality mismatch)
 * - counterfactual_cost = counterfactuals_evaluated * cost_per_counterfactual
 * - prediction_bonus = prediction_accuracy * prediction_weight (reduces FE)
 * - coherence_penalty = (1 - coherence) * coherence_weight (incoherence increases FE)
 */
static void compute_free_energy(
    imagination_fep_bridge_t* bridge,
    const imagination_stats_t* engine_stats
) {
    const imagination_fep_config_t* cfg = &bridge->config;

    /* Compute simulation divergence from reality */
    /* Use inverse of coherence as proxy for divergence when direct measure not available */
    float divergence = 0.0f;
    if (engine_stats->avg_coherence > 0.0f) {
        /* Lower coherence = higher divergence from expected reality model */
        divergence = 1.0f - engine_stats->avg_coherence;
    }
    bridge->simulation_divergence = clamp_f(divergence, 0.0f, 1.0f);

    /* Coherence from engine stats */
    bridge->coherence = clamp_f(engine_stats->avg_coherence, 0.0f, 1.0f);

    /* Counterfactual cost: scenarios that required hypothetical reasoning */
    /* Use scenarios created as proxy for counterfactual load */
    float counterfactual_load = 0.0f;
    if (engine_stats->scenarios_created > 0) {
        /* More scenarios = more counterfactual reasoning = more metabolic cost */
        counterfactual_load = (float)engine_stats->scenarios_created * cfg->counterfactual_cost;
        /* Normalize to reasonable range */
        counterfactual_load = clamp_f(counterfactual_load / 100.0f, 0.0f, 1.0f);
    }
    bridge->counterfactual_cost_accumulated = counterfactual_load;

    /* Prediction accuracy - vividness as proxy for clear predictions */
    float prediction_accuracy = engine_stats->avg_vividness;

    /* Compute individual contributions */
    float divergence_contrib = bridge->simulation_divergence * cfg->simulation_divergence_weight;
    float counterfactual_contrib = counterfactual_load * cfg->counterfactual_cost;
    float coherence_penalty = (1.0f - bridge->coherence) * cfg->coherence_weight;
    float prediction_bonus = prediction_accuracy * cfg->prediction_accuracy_weight;

    /* Total free energy */
    float total_fe = cfg->baseline_free_energy +
                     divergence_contrib +
                     counterfactual_contrib +
                     coherence_penalty -
                     prediction_bonus;  /* Bonus reduces FE */

    bridge->free_energy = clamp_f(total_fe, 0.0f, cfg->max_free_energy);

    /* Prediction error: unexpected changes in imagination state */
    float divergence_delta = fabsf(bridge->simulation_divergence - bridge->prev_divergence);
    float coherence_delta = fabsf(bridge->coherence - bridge->prev_coherence);
    float fe_delta = fabsf(bridge->free_energy - bridge->prev_free_energy);

    bridge->prediction_error = clamp_f(
        (divergence_delta * 0.4f + coherence_delta * 0.3f + fe_delta * 0.3f),
        0.0f, 1.0f
    );

    /* Update stats */
    bridge->stats.simulations_run = engine_stats->scenarios_created;
    bridge->stats.counterfactuals_evaluated = engine_stats->scenarios_created;
}

/**
 * @brief Check and trigger callbacks
 */
static void check_callbacks(imagination_fep_bridge_t* bridge) {
    const imagination_fep_config_t* cfg = &bridge->config;

    /* Check for high free energy */
    if (bridge->free_energy > cfg->high_free_energy_threshold) {
        if (bridge->state != IMAGINATION_FEP_STATE_DEGRADED) {
            bridge->state = IMAGINATION_FEP_STATE_DEGRADED;
            bridge->stats.degraded_mode_entries++;

            if (bridge->high_fe_callback && cfg->enable_callbacks) {
                bridge->high_fe_callback(bridge, bridge->free_energy,
                                         bridge->high_fe_user_data);
            }
        }
    } else if (bridge->state == IMAGINATION_FEP_STATE_DEGRADED) {
        bridge->state = IMAGINATION_FEP_STATE_ACTIVE;
    }

    /* Check for high divergence */
    if (cfg->enable_callbacks &&
        bridge->simulation_divergence > cfg->divergence_threshold) {
        if (bridge->divergence_callback) {
            bridge->divergence_callback(bridge, bridge->simulation_divergence,
                                        bridge->divergence_user_data);
        }
    }
}

/**
 * @brief Update statistics
 */
static void update_stats(imagination_fep_bridge_t* bridge, uint64_t update_time_us) {
    imagination_fep_stats_t* s = &bridge->stats;

    s->total_updates++;
    bridge->total_update_time_us += update_time_us;

    /* Peak tracking */
    if (bridge->free_energy > s->peak_free_energy) {
        s->peak_free_energy = bridge->free_energy;
    }

    /* Total contribution */
    s->total_free_energy_contribution += bridge->free_energy;

    /* Running averages */
    bridge->running_count++;
    bridge->running_avg_fe = (bridge->running_avg_fe * (bridge->running_count - 1) +
                              bridge->free_energy) / bridge->running_count;
    bridge->running_avg_divergence = (bridge->running_avg_divergence * (bridge->running_count - 1) +
                                      bridge->simulation_divergence) / bridge->running_count;
    bridge->running_avg_counterfactual = (bridge->running_avg_counterfactual * (bridge->running_count - 1) +
                                          bridge->counterfactual_cost_accumulated) / bridge->running_count;

    s->avg_free_energy = bridge->running_avg_fe;
    s->avg_simulation_divergence = bridge->running_avg_divergence;
    s->avg_counterfactual_cost = bridge->running_avg_counterfactual;
    s->avg_update_time_us = (float)bridge->total_update_time_us / (float)s->total_updates;
}

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *===========================================================================*/

imagination_fep_config_t imagination_fep_config_default(void) {
    imagination_fep_config_t config;
    memset(&config, 0, sizeof(config));

    /* Feature flags */
    config.enable_logging = false;
    config.enable_degraded_mode = true;
    config.enable_callbacks = true;

    /* Timing */
    config.update_interval_ms = 50;  /* Cognitive timescale */

    /* Weighting parameters */
    config.free_energy_weight = 1.0f;
    config.simulation_divergence_weight = IMAGINATION_FEP_DIVERGENCE_WEIGHT;
    config.counterfactual_cost = IMAGINATION_FEP_COUNTERFACTUAL_WEIGHT;
    config.coherence_weight = IMAGINATION_FEP_COHERENCE_WEIGHT;
    config.prediction_accuracy_weight = IMAGINATION_FEP_PREDICTION_WEIGHT;

    /* Thresholds */
    config.high_free_energy_threshold = 1.5f;
    config.divergence_threshold = 0.7f;
    config.coherence_threshold = 0.3f;

    /* Normalization */
    config.baseline_free_energy = IMAGINATION_FEP_BASELINE_FREE_ENERGY;
    config.max_free_energy = IMAGINATION_FEP_MAX_FREE_ENERGY;

    return config;
}

imagination_fep_bridge_t* imagination_fep_bridge_create(
    const imagination_fep_config_t* config
) {
    imagination_fep_bridge_t* bridge = nimcp_calloc(1, sizeof(imagination_fep_bridge_t));
    if (!bridge) return NULL;

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = imagination_fep_config_default();
    }

    /* Create mutex */
    mutex_attr_t mutex_attr = {.type = MUTEX_TYPE_NORMAL};
    bridge->mutex = nimcp_mutex_create(&mutex_attr);
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->state = IMAGINATION_FEP_STATE_UNINITIALIZED;
    bridge->registered = false;
    bridge->bridge_id = 0;

    /* Initialize metrics to baseline */
    bridge->free_energy = bridge->config.baseline_free_energy;
    bridge->simulation_divergence = 0.0f;
    bridge->prediction_error = 0.0f;
    bridge->coherence = 1.0f;
    bridge->counterfactual_cost_accumulated = 0.0f;

    /* Initialize previous state tracking */
    bridge->prev_free_energy = bridge->config.baseline_free_energy;
    bridge->prev_divergence = 0.0f;
    bridge->prev_coherence = 1.0f;

    bridge->state = IMAGINATION_FEP_STATE_IDLE;

    return bridge;
}

void imagination_fep_bridge_destroy(imagination_fep_bridge_t* bridge) {
    if (!bridge) return;

    /* Unregister if still registered */
    if (bridge->registered) {
        imagination_fep_bridge_unregister(bridge);
    }

    /* Destroy mutex */
    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }

    nimcp_free(bridge);
}

int imagination_fep_bridge_reset(imagination_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Reset metrics */
    bridge->free_energy = bridge->config.baseline_free_energy;
    bridge->simulation_divergence = 0.0f;
    bridge->prediction_error = 0.0f;
    bridge->coherence = 1.0f;
    bridge->counterfactual_cost_accumulated = 0.0f;

    /* Reset previous state */
    bridge->prev_free_energy = bridge->config.baseline_free_energy;
    bridge->prev_divergence = 0.0f;
    bridge->prev_coherence = 1.0f;

    /* Reset running averages */
    bridge->running_avg_fe = 0.0f;
    bridge->running_avg_divergence = 0.0f;
    bridge->running_avg_counterfactual = 0.0f;
    bridge->running_count = 0;

    /* Reset statistics */
    memset(&bridge->stats, 0, sizeof(imagination_fep_stats_t));

    /* Reset timing */
    bridge->last_update_time_ms = 0;
    bridge->total_update_time_us = 0;

    bridge->state = IMAGINATION_FEP_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

/*=============================================================================
 * REGISTRATION FUNCTIONS
 *===========================================================================*/

int imagination_fep_bridge_register(
    imagination_fep_bridge_t* bridge,
    fep_orchestrator_t* orchestrator,
    imagination_engine_t* engine,
    uint32_t* bridge_id_out
) {
    if (!bridge || !orchestrator) return -1;  /* engine can be NULL for standalone testing */

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
    bridge->engine = engine;

    /* Register with FEP orchestrator */
    uint32_t assigned_id = 0;
    int ret = fep_orchestrator_register_bridge(
        orchestrator,
        "imagination",
        FEP_BRIDGE_CATEGORY_COGNITIVE,  /* 50ms cognitive timescale */
        (fep_bridge_handle_t)bridge,
        imagination_fep_update_callback,
        imagination_fep_destroy_callback,
        &assigned_id
    );

    if (ret != 0) {
        bridge->orchestrator = NULL;
        bridge->engine = NULL;
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    bridge->bridge_id = assigned_id;
    bridge->registered = true;
    bridge->state = IMAGINATION_FEP_STATE_ACTIVE;

    if (bridge_id_out) {
        *bridge_id_out = assigned_id;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int imagination_fep_bridge_unregister(imagination_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    if (!bridge->registered) {
        nimcp_mutex_unlock(bridge->mutex);
        return 0;  /* Not registered, nothing to do */
    }

    /* Unregister from orchestrator */
    if (bridge->orchestrator) {
        fep_orchestrator_unregister_bridge(bridge->orchestrator, bridge->bridge_id);
    }

    bridge->registered = false;
    bridge->bridge_id = 0;
    bridge->orchestrator = NULL;
    bridge->engine = NULL;
    bridge->state = IMAGINATION_FEP_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

bool imagination_fep_bridge_is_registered(const imagination_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((imagination_fep_bridge_t*)bridge)->mutex);
    bool registered = bridge->registered;
    nimcp_mutex_unlock(((imagination_fep_bridge_t*)bridge)->mutex);

    return registered;
}

uint32_t imagination_fep_bridge_get_id(const imagination_fep_bridge_t* bridge) {
    if (!bridge) return 0;

    nimcp_mutex_lock(((imagination_fep_bridge_t*)bridge)->mutex);
    uint32_t id = bridge->registered ? bridge->bridge_id : 0;
    nimcp_mutex_unlock(((imagination_fep_bridge_t*)bridge)->mutex);

    return id;
}

/*=============================================================================
 * FEP UPDATE CALLBACK
 *===========================================================================*/

int imagination_fep_update_callback(void* handle) {
    imagination_fep_bridge_t* bridge = (imagination_fep_bridge_t*)handle;
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Ensure we're registered and have an engine */
    if (!bridge->registered || !bridge->engine) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    uint64_t start_us = get_time_us();

    /* Get engine statistics */
    imagination_stats_t engine_stats;
    memset(&engine_stats, 0, sizeof(engine_stats));

    int ret = imagination_get_stats(bridge->engine, &engine_stats);
    if (ret != 0) {
        /* Engine stats not available, use minimal update */
        bridge->last_update_time_ms = get_time_ms();
        bridge->stats.total_updates++;
        nimcp_mutex_unlock(bridge->mutex);
        return 0;  /* Non-critical failure */
    }

    /* Store previous state for delta computation */
    bridge->prev_free_energy = bridge->free_energy;
    bridge->prev_divergence = bridge->simulation_divergence;
    bridge->prev_coherence = bridge->coherence;

    /* Compute free energy from engine stats */
    compute_free_energy(bridge, &engine_stats);

    /* Calculate update time */
    uint64_t end_us = get_time_us();
    uint64_t update_time_us = end_us - start_us;

    /* Update timing */
    bridge->last_update_time_ms = get_time_ms();

    /* Update statistics */
    update_stats(bridge, update_time_us);

    /* Check and trigger callbacks */
    check_callbacks(bridge);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

void imagination_fep_destroy_callback(void* handle) {
    /* No-op: Bridge is destroyed separately via imagination_fep_bridge_destroy() */
    (void)handle;
}

/*=============================================================================
 * OPERATIONS
 *===========================================================================*/

int imagination_fep_bridge_update(imagination_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    /* If registered with orchestrator and has engine, do full update */
    if (bridge->registered && bridge->engine) {
        return imagination_fep_update_callback(bridge);
    }

    return -1;  /* Cannot update without engine */
}

int imagination_fep_bridge_force_update(imagination_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    /* If registered with orchestrator and has engine, do full update */
    if (bridge->registered && bridge->engine) {
        return imagination_fep_update_callback(bridge);
    }

    /* For unit testing: perform minimal update with callback invocation */
    nimcp_mutex_lock(bridge->mutex);

    uint64_t start_us = get_time_us();

    /* Store previous state */
    bridge->prev_free_energy = bridge->free_energy;
    bridge->prev_divergence = bridge->simulation_divergence;
    bridge->prev_coherence = bridge->coherence;

    /* Simulate minimal metrics update */
    bridge->last_update_time_ms = get_time_ms();

    /* Calculate update time */
    uint64_t end_us = get_time_us();
    uint64_t update_time_us = end_us - start_us;

    /* Update stats */
    bridge->stats.total_updates++;
    bridge->total_update_time_us += update_time_us;

    /* Invoke callbacks even without full engine data */
    check_callbacks(bridge);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

/*=============================================================================
 * STATISTICS AND METRICS
 *===========================================================================*/

int imagination_fep_bridge_get_stats(
    const imagination_fep_bridge_t* bridge,
    imagination_fep_stats_t* stats_out
) {
    if (!bridge || !stats_out) return -1;

    nimcp_mutex_lock(((imagination_fep_bridge_t*)bridge)->mutex);
    *stats_out = bridge->stats;
    nimcp_mutex_unlock(((imagination_fep_bridge_t*)bridge)->mutex);

    return 0;
}

int imagination_fep_bridge_reset_stats(imagination_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(imagination_fep_stats_t));
    bridge->running_avg_fe = 0.0f;
    bridge->running_avg_divergence = 0.0f;
    bridge->running_avg_counterfactual = 0.0f;
    bridge->running_count = 0;
    bridge->total_update_time_us = 0;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/*=============================================================================
 * ACCESSORS
 *===========================================================================*/

float imagination_fep_bridge_get_free_energy(const imagination_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((imagination_fep_bridge_t*)bridge)->mutex);
    float fe = bridge->free_energy;
    nimcp_mutex_unlock(((imagination_fep_bridge_t*)bridge)->mutex);

    return fe;
}

float imagination_fep_bridge_get_simulation_divergence(
    const imagination_fep_bridge_t* bridge
) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((imagination_fep_bridge_t*)bridge)->mutex);
    float divergence = bridge->simulation_divergence;
    nimcp_mutex_unlock(((imagination_fep_bridge_t*)bridge)->mutex);

    return divergence;
}

float imagination_fep_bridge_get_prediction_error(
    const imagination_fep_bridge_t* bridge
) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((imagination_fep_bridge_t*)bridge)->mutex);
    float pe = bridge->prediction_error;
    nimcp_mutex_unlock(((imagination_fep_bridge_t*)bridge)->mutex);

    return pe;
}

imagination_fep_state_t imagination_fep_bridge_get_state(
    const imagination_fep_bridge_t* bridge
) {
    if (!bridge) return IMAGINATION_FEP_STATE_ERROR;

    nimcp_mutex_lock(((imagination_fep_bridge_t*)bridge)->mutex);
    imagination_fep_state_t state = bridge->state;
    nimcp_mutex_unlock(((imagination_fep_bridge_t*)bridge)->mutex);

    return state;
}

bool imagination_fep_bridge_is_degraded(const imagination_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((imagination_fep_bridge_t*)bridge)->mutex);
    bool degraded = (bridge->state == IMAGINATION_FEP_STATE_DEGRADED);
    nimcp_mutex_unlock(((imagination_fep_bridge_t*)bridge)->mutex);

    return degraded;
}

/*=============================================================================
 * CALLBACK REGISTRATION
 *===========================================================================*/

int imagination_fep_bridge_set_high_fe_callback(
    imagination_fep_bridge_t* bridge,
    imagination_fep_high_fe_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->high_fe_callback = callback;
    bridge->high_fe_user_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int imagination_fep_bridge_set_divergence_callback(
    imagination_fep_bridge_t* bridge,
    imagination_fep_divergence_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->divergence_callback = callback;
    bridge->divergence_user_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/*=============================================================================
 * CONFIGURATION
 *===========================================================================*/

int imagination_fep_bridge_set_config(
    imagination_fep_bridge_t* bridge,
    const imagination_fep_config_t* config
) {
    if (!bridge || !config) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->config = *config;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int imagination_fep_bridge_get_config(
    const imagination_fep_bridge_t* bridge,
    imagination_fep_config_t* config_out
) {
    if (!bridge || !config_out) return -1;

    nimcp_mutex_lock(((imagination_fep_bridge_t*)bridge)->mutex);
    *config_out = bridge->config;
    nimcp_mutex_unlock(((imagination_fep_bridge_t*)bridge)->mutex);

    return 0;
}

/*=============================================================================
 * UTILITY FUNCTIONS
 *===========================================================================*/

const char* imagination_fep_state_name(imagination_fep_state_t state) {
    switch (state) {
        case IMAGINATION_FEP_STATE_UNINITIALIZED: return "uninitialized";
        case IMAGINATION_FEP_STATE_IDLE:          return "idle";
        case IMAGINATION_FEP_STATE_ACTIVE:        return "active";
        case IMAGINATION_FEP_STATE_DEGRADED:      return "degraded";
        case IMAGINATION_FEP_STATE_ERROR:         return "error";
        default:                                   return "unknown";
    }
}
