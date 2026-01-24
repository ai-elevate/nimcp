/**
 * @file nimcp_game_theory_executive_fep_bridge.c
 * @brief Game Theory-Executive Bridge - FEP Orchestrator Integration Implementation
 * @version 1.0.0
 * @date 2026-01-08
 *
 * WHAT: Implements FEP registration and update callbacks for game theory-executive bridge
 * WHY:  Enable coordinated free energy minimization for strategic decision-making
 * HOW:  Compute free energy from decision quality, executive alignment, action coherence
 *
 * @author NIMCP Development Team
 */

#include "cognitive/integration/nimcp_game_theory_executive_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/integration/nimcp_game_theory_executive_bridge.h"
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

struct gt_exec_fep_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */

    /* Configuration */
    gt_exec_fep_config_t config;

    /* State */
    gt_exec_fep_state_t state;

    /* References */
    fep_orchestrator_t* orchestrator;
    game_theory_executive_bridge_t* gt_exec_bridge;
    uint32_t bridge_id;
    bool registered;

    /* Metrics */
    gt_exec_fep_metrics_t metrics;

    /* Previous state for change detection */
    float prev_free_energy;
    float prev_prediction_error;
    float prev_decision_quality;
    float prev_executive_alignment;
    float prev_action_coherence;

    /* Running averages */
    float running_avg_fe;
    uint64_t running_count;

    /* Statistics */
    gt_exec_fep_stats_t stats;

    /* Callbacks */
    gt_exec_fep_high_fe_callback_t high_fe_callback;
    void* high_fe_user_data;
    gt_exec_fep_surprise_callback_t surprise_callback;
    void* surprise_user_data;
    gt_exec_fep_metrics_callback_t metrics_callback;
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
 * @brief Compute free energy from game theory-executive metrics
 *
 * Free energy model for strategic decision-making:
 * FE = baseline + decision_contrib + alignment_contrib + coherence_contrib
 *
 * Where:
 * - decision_contrib = (1 - decision_quality) * decision_weight
 *   (lower decision quality = higher FE)
 * - alignment_contrib = (1 - executive_alignment) * alignment_weight
 *   (misalignment between exec and strategy = higher FE)
 * - coherence_contrib = (1 - action_coherence) * coherence_weight
 *   (incoherent action selection = higher FE)
 *
 * High alignment represents minimum free energy state where executive
 * decisions consistently follow optimal strategic recommendations.
 */
static void compute_free_energy(gt_exec_fep_bridge_t* bridge) {
    gt_exec_fep_metrics_t* m = &bridge->metrics;
    const gt_exec_fep_config_t* cfg = &bridge->config;

    /* Clamp input metrics */
    m->decision_quality = clamp_f(m->decision_quality, 0.0f, 1.0f);
    m->executive_alignment = clamp_f(m->executive_alignment, 0.0f, 1.0f);
    m->action_coherence = clamp_f(m->action_coherence, 0.0f, 1.0f);

    /* Decision quality contribution:
     * Low quality decisions = high prediction error about optimal action */
    m->decision_contribution = (1.0f - m->decision_quality) * cfg->decision_quality_weight;

    /* Executive alignment contribution:
     * Misalignment between executive and strategy = prediction error */
    m->alignment_contribution = (1.0f - m->executive_alignment) * cfg->executive_alignment_weight;

    /* Action coherence contribution:
     * Incoherent action selection = system-level prediction error */
    m->coherence_contribution = (1.0f - m->action_coherence) * cfg->action_coherence_weight;

    /* Total free energy */
    float total_fe = cfg->baseline_free_energy +
                     (m->decision_contribution +
                      m->alignment_contribution +
                      m->coherence_contribution) * cfg->free_energy_weight;

    m->free_energy = clamp_f(total_fe, 0.0f, cfg->max_free_energy);

    /* Executive aligned check */
    m->exec_aligned = (m->executive_alignment >= (1.0f - cfg->alignment_epsilon));

    /* Prediction error: weighted combination of all uncertainty sources */
    float new_prediction_error = ((1.0f - m->decision_quality) * 0.4f +
                                   (1.0f - m->executive_alignment) * 0.4f +
                                   (1.0f - m->action_coherence) * 0.2f);

    /* Apply decay to smooth transitions */
    m->prediction_error = clamp_f(
        new_prediction_error * cfg->error_decay_rate +
        bridge->prev_prediction_error * (1.0f - cfg->error_decay_rate),
        0.0f, 1.0f
    );

    /* Surprise: unexpected changes in decision state */
    float fe_change = fabsf(m->free_energy - bridge->prev_free_energy);
    float decision_change = fabsf(m->decision_quality - bridge->prev_decision_quality);
    float alignment_change = fabsf(m->executive_alignment - bridge->prev_executive_alignment);
    float coherence_change = fabsf(m->action_coherence - bridge->prev_action_coherence);

    m->surprise = clamp_f(
        (fe_change * 0.3f + decision_change * 0.3f +
         alignment_change * 0.2f + coherence_change * 0.2f),
        0.0f, 1.0f
    );

    /* Entropy: based on decision uncertainty */
    m->entropy = clamp_f(
        (1.0f - m->decision_quality) * 0.6f + (1.0f - m->action_coherence) * 0.4f,
        0.0f, 1.0f
    );

    /* Recommendation accuracy from alignment tracking */
    if (bridge->stats.recommendations_followed + bridge->stats.recommendations_overridden > 0) {
        m->recommendation_accuracy = (float)bridge->stats.recommendations_followed /
            (float)(bridge->stats.recommendations_followed + bridge->stats.recommendations_overridden);
    }
}

/**
 * @brief Check and trigger callbacks
 */
static void check_callbacks(gt_exec_fep_bridge_t* bridge) {
    gt_exec_fep_metrics_t* m = &bridge->metrics;
    const gt_exec_fep_config_t* cfg = &bridge->config;

    /* Check for high free energy */
    if (m->free_energy > cfg->high_free_energy_threshold) {
        if (bridge->state != GT_EXEC_FEP_STATE_DEGRADED) {
            bridge->state = GT_EXEC_FEP_STATE_DEGRADED;
            bridge->stats.degraded_mode_entries++;

            if (bridge->high_fe_callback) {
                bridge->high_fe_callback(bridge, m->free_energy,
                                         bridge->high_fe_user_data);
            }
        }
    } else if (bridge->state == GT_EXEC_FEP_STATE_DEGRADED) {
        bridge->state = GT_EXEC_FEP_STATE_ACTIVE;
    }

    /* Check for surprise events */
    if (m->surprise > cfg->prediction_error_threshold) {
        bridge->stats.surprise_events++;

        if (bridge->surprise_callback) {
            const char* source = "unknown";
            if (m->decision_contribution > m->alignment_contribution &&
                m->decision_contribution > m->coherence_contribution) {
                source = "decision";
            } else if (m->alignment_contribution > m->coherence_contribution) {
                source = "alignment";
            } else {
                source = "coherence";
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
static void update_stats(gt_exec_fep_bridge_t* bridge, uint64_t update_time_us) {
    gt_exec_fep_stats_t* s = &bridge->stats;
    gt_exec_fep_metrics_t* m = &bridge->metrics;

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
static void store_previous_state(gt_exec_fep_bridge_t* bridge) {
    bridge->prev_free_energy = bridge->metrics.free_energy;
    bridge->prev_prediction_error = bridge->metrics.prediction_error;
    bridge->prev_decision_quality = bridge->metrics.decision_quality;
    bridge->prev_executive_alignment = bridge->metrics.executive_alignment;
    bridge->prev_action_coherence = bridge->metrics.action_coherence;
}

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *===========================================================================*/

gt_exec_fep_config_t gt_exec_fep_config_default(void) {
    gt_exec_fep_config_t config;
    memset(&config, 0, sizeof(config));

    /* Feature enables */
    config.enable_logging = false;
    config.update_interval_ms = 50;  /* Cognitive timescale */

    /* Weighting parameters */
    config.free_energy_weight = 1.0f;
    config.decision_quality_weight = GT_EXEC_FEP_DECISION_QUALITY_WEIGHT;
    config.executive_alignment_weight = GT_EXEC_FEP_EXEC_ALIGNMENT_WEIGHT;
    config.action_coherence_weight = GT_EXEC_FEP_ACTION_COHERENCE_WEIGHT;

    /* Thresholds */
    config.high_free_energy_threshold = 1.5f;
    config.prediction_error_threshold = 0.5f;
    config.alignment_epsilon = 0.01f;

    /* Normalization */
    config.baseline_free_energy = GT_EXEC_FEP_BASELINE_FREE_ENERGY;
    config.max_free_energy = GT_EXEC_FEP_MAX_FREE_ENERGY;
    config.error_decay_rate = GT_EXEC_FEP_ERROR_DECAY_RATE;

    return config;
}

gt_exec_fep_bridge_t* gt_exec_fep_bridge_create(const gt_exec_fep_config_t* config) {
    gt_exec_fep_bridge_t* bridge = nimcp_calloc(1, sizeof(gt_exec_fep_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = gt_exec_fep_config_default();
    }

    /* Initialize base bridge infrastructure */
    if (bridge_base_init(&bridge->base, 0, "game_theory_executive_fep") != 0) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->state = GT_EXEC_FEP_STATE_UNINITIALIZED;
    bridge->registered = false;
    bridge->bridge_id = 0;

    /* Initialize metrics to baseline */
    bridge->metrics.free_energy = bridge->config.baseline_free_energy;
    bridge->metrics.prediction_error = 0.0f;
    bridge->metrics.surprise = 0.0f;
    bridge->metrics.entropy = 0.0f;
    bridge->metrics.decision_quality = 1.0f;      /* Start with good quality */
    bridge->metrics.executive_alignment = 1.0f;   /* Start aligned */
    bridge->metrics.action_coherence = 1.0f;      /* Start coherent */
    bridge->metrics.exec_aligned = true;

    /* Initialize previous state tracking */
    bridge->prev_free_energy = bridge->config.baseline_free_energy;
    bridge->prev_prediction_error = 0.0f;
    bridge->prev_decision_quality = 1.0f;
    bridge->prev_executive_alignment = 1.0f;
    bridge->prev_action_coherence = 1.0f;

    bridge->state = GT_EXEC_FEP_STATE_IDLE;

    return bridge;
}

void gt_exec_fep_bridge_destroy(gt_exec_fep_bridge_t* bridge) {
    if (!bridge) return;

    /* Unregister if still registered */
    if (bridge->registered) {
        gt_exec_fep_bridge_unregister(bridge);
    }

    /* Cleanup base bridge infrastructure */
    bridge_base_cleanup(&bridge->base);

    nimcp_free(bridge);
}

int gt_exec_fep_bridge_reset(gt_exec_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Reset metrics */
    memset(&bridge->metrics, 0, sizeof(gt_exec_fep_metrics_t));
    bridge->metrics.free_energy = bridge->config.baseline_free_energy;
    bridge->metrics.decision_quality = 1.0f;
    bridge->metrics.executive_alignment = 1.0f;
    bridge->metrics.action_coherence = 1.0f;
    bridge->metrics.exec_aligned = true;

    /* Reset previous state */
    bridge->prev_free_energy = bridge->config.baseline_free_energy;
    bridge->prev_prediction_error = 0.0f;
    bridge->prev_decision_quality = 1.0f;
    bridge->prev_executive_alignment = 1.0f;
    bridge->prev_action_coherence = 1.0f;

    /* Reset running averages */
    bridge->running_avg_fe = 0.0f;
    bridge->running_count = 0;

    /* Reset statistics */
    memset(&bridge->stats, 0, sizeof(gt_exec_fep_stats_t));

    bridge->state = GT_EXEC_FEP_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/*=============================================================================
 * REGISTRATION FUNCTIONS
 *===========================================================================*/

int gt_exec_fep_bridge_register(
    gt_exec_fep_bridge_t* bridge,
    fep_orchestrator_t* orchestrator,
    game_theory_executive_bridge_t* gt_exec_bridge,
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
    bridge->gt_exec_bridge = gt_exec_bridge;

    /* Register with FEP orchestrator */
    uint32_t assigned_id = 0;
    int ret = fep_orchestrator_register_bridge(
        orchestrator,
        "game_theory_executive",
        FEP_BRIDGE_CATEGORY_COGNITIVE,  /* 50ms cognitive timescale */
        (fep_bridge_handle_t)bridge,
        gt_exec_fep_update_callback,
        gt_exec_fep_destroy_callback,
        &assigned_id
    );

    if (ret != 0) {
        bridge->orchestrator = NULL;
        bridge->gt_exec_bridge = NULL;
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    bridge->bridge_id = assigned_id;
    bridge->registered = true;
    bridge->state = GT_EXEC_FEP_STATE_ACTIVE;

    if (bridge_id_out) {
        *bridge_id_out = assigned_id;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int gt_exec_fep_bridge_unregister(gt_exec_fep_bridge_t* bridge) {
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
    bridge->gt_exec_bridge = NULL;
    bridge->state = GT_EXEC_FEP_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

bool gt_exec_fep_bridge_is_registered(const gt_exec_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);
    bool registered = bridge->registered;
    nimcp_mutex_unlock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);

    return registered;
}

uint32_t gt_exec_fep_bridge_get_id(const gt_exec_fep_bridge_t* bridge) {
    if (!bridge) return 0;

    nimcp_mutex_lock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);
    uint32_t id = bridge->registered ? bridge->bridge_id : 0;
    nimcp_mutex_unlock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);

    return id;
}

/*=============================================================================
 * FEP UPDATE CALLBACK
 *===========================================================================*/

int gt_exec_fep_update_callback(void* handle) {
    gt_exec_fep_bridge_t* bridge = (gt_exec_fep_bridge_t*)handle;
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

    /* If we have a game theory-executive bridge, query it for statistics */
    if (bridge->gt_exec_bridge) {
        game_theory_executive_stats_t gt_exec_stats;
        memset(&gt_exec_stats, 0, sizeof(gt_exec_stats));

        int err = game_theory_executive_bridge_get_stats(bridge->gt_exec_bridge, &gt_exec_stats);
        if (err == 0) {
            /* Update metrics from bridge statistics */
            bridge->stats.decision_computations++;

            /* Estimate decision quality from recommendation accuracy */
            if (gt_exec_stats.recommendations_made > 0) {
                float follow_rate = (float)gt_exec_stats.recommendations_followed /
                                   (float)gt_exec_stats.recommendations_made;
                /* Higher follow rate with good outcomes = higher quality */
                bridge->metrics.decision_quality = clamp_f(
                    gt_exec_stats.recommendation_accuracy * 0.6f + follow_rate * 0.4f,
                    0.0f, 1.0f
                );
            }

            /* Executive alignment from override rate */
            if (gt_exec_stats.strategic_decisions > 0) {
                float override_rate = (float)gt_exec_stats.executive_overrides /
                                     (float)gt_exec_stats.strategic_decisions;
                /* Lower override rate = higher alignment */
                bridge->metrics.executive_alignment = clamp_f(
                    1.0f - override_rate,
                    0.0f, 1.0f
                );
            }

            /* Action coherence from utility realization */
            if (gt_exec_stats.avg_expected_utility > 0.0f) {
                float utility_ratio = gt_exec_stats.avg_realized_utility /
                                     gt_exec_stats.avg_expected_utility;
                /* Realized close to expected = high coherence */
                bridge->metrics.action_coherence = clamp_f(
                    utility_ratio,
                    0.0f, 1.0f
                );
            }

            /* Track follow/override stats */
            bridge->stats.recommendations_followed = gt_exec_stats.recommendations_followed;
            bridge->stats.recommendations_overridden = gt_exec_stats.executive_overrides;

            bridge->stats.alignment_checks++;
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

void gt_exec_fep_destroy_callback(void* handle) {
    /* No-op: Bridge is destroyed separately via gt_exec_fep_bridge_destroy() */
    (void)handle;
}

/*=============================================================================
 * MANUAL UPDATE OPERATIONS
 *===========================================================================*/

int gt_exec_fep_bridge_force_update(gt_exec_fep_bridge_t* bridge) {
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

int gt_exec_fep_bridge_update_decision_quality(
    gt_exec_fep_bridge_t* bridge,
    float quality
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->metrics.decision_quality = clamp_f(quality, 0.0f, 1.0f);
    bridge->stats.decision_computations++;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int gt_exec_fep_bridge_update_executive_alignment(
    gt_exec_fep_bridge_t* bridge,
    float alignment
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->metrics.executive_alignment = clamp_f(alignment, 0.0f, 1.0f);
    bridge->stats.alignment_checks++;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int gt_exec_fep_bridge_update_action_coherence(
    gt_exec_fep_bridge_t* bridge,
    float coherence
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->metrics.action_coherence = clamp_f(coherence, 0.0f, 1.0f);
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int gt_exec_fep_bridge_notify_recommendation_result(
    gt_exec_fep_bridge_t* bridge,
    bool followed,
    float outcome_utility
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    if (followed) {
        bridge->stats.recommendations_followed++;
    } else {
        bridge->stats.recommendations_overridden++;
    }

    /* Update recommendation accuracy */
    float total = (float)(bridge->stats.recommendations_followed +
                          bridge->stats.recommendations_overridden);
    if (total > 0) {
        bridge->metrics.recommendation_accuracy =
            (float)bridge->stats.recommendations_followed / total;
    }

    /* Adjust decision quality based on outcome */
    float quality_delta = outcome_utility - 0.5f;  /* Centered around 0.5 */
    bridge->metrics.decision_quality = clamp_f(
        bridge->metrics.decision_quality + quality_delta * 0.1f,
        0.0f, 1.0f
    );

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/*=============================================================================
 * METRICS AND STATISTICS
 *===========================================================================*/

int gt_exec_fep_bridge_get_metrics(
    const gt_exec_fep_bridge_t* bridge,
    gt_exec_fep_metrics_t* metrics_out
) {
    if (!bridge || !metrics_out) return -1;

    nimcp_mutex_lock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);
    *metrics_out = bridge->metrics;
    nimcp_mutex_unlock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);

    return 0;
}

int gt_exec_fep_bridge_get_stats(
    const gt_exec_fep_bridge_t* bridge,
    gt_exec_fep_stats_t* stats_out
) {
    if (!bridge || !stats_out) return -1;

    nimcp_mutex_lock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);
    *stats_out = bridge->stats;
    nimcp_mutex_unlock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);

    return 0;
}

int gt_exec_fep_bridge_reset_stats(gt_exec_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(gt_exec_fep_stats_t));
    bridge->running_avg_fe = 0.0f;
    bridge->running_count = 0;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float gt_exec_fep_bridge_get_free_energy(const gt_exec_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);
    float fe = bridge->metrics.free_energy;
    nimcp_mutex_unlock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);

    return fe;
}

float gt_exec_fep_bridge_get_decision_quality(const gt_exec_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);
    float dq = bridge->metrics.decision_quality;
    nimcp_mutex_unlock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);

    return dq;
}

float gt_exec_fep_bridge_get_prediction_error(const gt_exec_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);
    float pe = bridge->metrics.prediction_error;
    nimcp_mutex_unlock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);

    return pe;
}

float gt_exec_fep_bridge_get_executive_alignment(const gt_exec_fep_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);
    float ea = bridge->metrics.executive_alignment;
    nimcp_mutex_unlock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);

    return ea;
}

/*=============================================================================
 * STATE QUERY FUNCTIONS
 *===========================================================================*/

gt_exec_fep_state_t gt_exec_fep_bridge_get_state(const gt_exec_fep_bridge_t* bridge) {
    if (!bridge) return GT_EXEC_FEP_STATE_ERROR;

    nimcp_mutex_lock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);
    gt_exec_fep_state_t state = bridge->state;
    nimcp_mutex_unlock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);

    return state;
}

bool gt_exec_fep_bridge_is_degraded(const gt_exec_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);
    bool degraded = (bridge->state == GT_EXEC_FEP_STATE_DEGRADED);
    nimcp_mutex_unlock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);

    return degraded;
}

bool gt_exec_fep_bridge_is_exec_aligned(const gt_exec_fep_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);
    bool aligned = bridge->metrics.exec_aligned;
    nimcp_mutex_unlock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);

    return aligned;
}

const char* gt_exec_fep_state_name(gt_exec_fep_state_t state) {
    switch (state) {
        case GT_EXEC_FEP_STATE_UNINITIALIZED: return "uninitialized";
        case GT_EXEC_FEP_STATE_IDLE:          return "idle";
        case GT_EXEC_FEP_STATE_ACTIVE:        return "active";
        case GT_EXEC_FEP_STATE_DEGRADED:      return "degraded";
        case GT_EXEC_FEP_STATE_ERROR:         return "error";
        default:                               return "unknown";
    }
}

/*=============================================================================
 * CALLBACK REGISTRATION
 *===========================================================================*/

int gt_exec_fep_bridge_set_high_fe_callback(
    gt_exec_fep_bridge_t* bridge,
    gt_exec_fep_high_fe_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->high_fe_callback = callback;
    bridge->high_fe_user_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int gt_exec_fep_bridge_set_surprise_callback(
    gt_exec_fep_bridge_t* bridge,
    gt_exec_fep_surprise_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->surprise_callback = callback;
    bridge->surprise_user_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int gt_exec_fep_bridge_set_metrics_callback(
    gt_exec_fep_bridge_t* bridge,
    gt_exec_fep_metrics_callback_t callback,
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

int gt_exec_fep_bridge_set_config(
    gt_exec_fep_bridge_t* bridge,
    const gt_exec_fep_config_t* config
) {
    if (!bridge || !config) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->config = *config;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int gt_exec_fep_bridge_get_config(
    const gt_exec_fep_bridge_t* bridge,
    gt_exec_fep_config_t* config_out
) {
    if (!bridge || !config_out) return -1;

    nimcp_mutex_lock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);
    *config_out = bridge->config;
    nimcp_mutex_unlock(((gt_exec_fep_bridge_t*)bridge)->base.mutex);

    return 0;
}
