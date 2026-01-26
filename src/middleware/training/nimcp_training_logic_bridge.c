#include <stddef.h>  /* for NULL */
//=============================================================================
// nimcp_training_logic_bridge.c - Training-Logic Bridge Implementation
//=============================================================================

#include "middleware/training/nimcp_training_logic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "middleware/training/nimcp_perception_training_bridge.h"
#include "middleware/training/nimcp_cortical_training_bridge.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "core/neuron_types/nimcp_neural_logic.h"
#include "api/nimcp_api_exception.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <math.h>

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for training_logic_bridge module */
static nimcp_health_agent_t* g_training_logic_bridge_health_agent = NULL;

/**
 * @brief Set health agent for training_logic_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void training_logic_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_training_logic_bridge_health_agent = agent;
}

/** @brief Send heartbeat from training_logic_bridge module */
static inline void training_logic_bridge_heartbeat(const char* operation, float progress) {
    if (g_training_logic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_training_logic_bridge_health_agent, operation, progress);
    }
}


/*=============================================================================
 * TIME HELPER
 *============================================================================*/

/**
 * @brief Get current time in microseconds
 *
 * WHAT: Returns monotonic clock time in microseconds
 * WHY:  Used for timing gate evaluations
 * HOW:  Uses clock_gettime with CLOCK_MONOTONIC
 */
static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/**
 * @brief Get current time in milliseconds
 *
 * WHAT: Returns monotonic clock time in milliseconds
 * WHY:  Used for checkpoint tracking
 * HOW:  Uses clock_gettime with CLOCK_MONOTONIC
 */
static uint64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

/*=============================================================================
 * CONSTANTS
 *============================================================================*/

/* BIO_MODULE_TRAINING_LOGIC should be 0x0520 */
#ifndef BIO_MODULE_TRAINING_LOGIC
#define BIO_MODULE_TRAINING_LOGIC 0x0520
#endif

/* Default metric thresholds for condition computation */
#define DEFAULT_LOSS_STABILITY_THRESHOLD    0.1f    /**< Max loss variance for stability */
#define DEFAULT_GRAD_NORM_MIN               1e-7f   /**< Min grad norm (vanishing threshold) */
#define DEFAULT_GRAD_NORM_MAX               10.0f   /**< Max grad norm (explosion threshold) */
#define DEFAULT_LR_MIN                      1e-8f   /**< Minimum reasonable LR */
#define DEFAULT_LR_MAX                      1.0f    /**< Maximum reasonable LR */
#define DEFAULT_MEMORY_THRESHOLD            0.85f   /**< Max memory usage */
#define DEFAULT_THROUGHPUT_MIN              1.0f    /**< Min batches/sec */
#define DEFAULT_LOSS_TREND_WINDOW           10      /**< Steps for trend computation */

/*=============================================================================
 * DATA STRUCTURES
 *============================================================================*/

/**
 * @brief History entry for metric tracking
 */
typedef struct {
    float loss;
    float grad_norm;
    uint64_t timestamp_ms;
} training_metric_history_entry_t;

/**
 * @brief Main bridge structure
 *
 * WHAT: Internal state for training-logic bridge
 * WHY:  Encapsulates all bridge data and integrations
 * HOW:  Single struct with all subsystems
 */
struct training_logic_bridge {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    /* Neural logic network */
    neural_logic_network_t logic_network;

    /* Pre-built decision gates */
    uint32_t stability_check_gate;      /**< AND gate: loss_stable AND grad_stable AND lr_reasonable */
    uint32_t intervention_gate;         /**< OR gate: grad_exploding OR loss_nan OR diverging */
    uint32_t lr_increase_gate;          /**< IMPLIES gate: stable_n_steps -> immune_ok AND resource_ok */
    uint32_t batch_size_gate;           /**< AND gate: memory_ok AND throughput_ok */
    uint32_t checkpoint_gate;           /**< AND gate: memory_ok AND not_mid_batch AND sufficient_progress */

    /* Training conditions */
    training_logic_conditions_t conditions;

    /* Configuration */
    training_logic_config_t config;

    /* Integration pointers */
    nimcp_brain_training_ctx_t* training_ctx;
    training_immune_system_t* immune_system;
    portia_logic_bridge_t* portia_logic;
    swarm_logic_bridge_t* swarm_logic;
    portia_swarm_logic_bridge_t* unified_bridge;
    perception_training_bridge_t* perception_training;
    cortical_training_bridge_t* cortical_training;

    /* Statistics */
    training_logic_stats_t stats;

    /* Custom gates tracking */
    uint32_t next_custom_gate_id;

    /* Metric history buffer */
    training_metric_history_entry_t* history;
    uint32_t history_head;
    uint32_t history_count;

    /* Last LR modulation */
    float last_lr_factor;
    uint32_t last_batch_size;
};

/*=============================================================================
 * HELPER FUNCTIONS
 *============================================================================*/

/**
 * @brief Update training conditions from current metrics
 *
 * WHAT: Converts continuous metrics to boolean conditions
 * WHY:  Logic gates operate on discrete inputs
 * HOW:  Applies thresholds to each metric
 */
static int update_conditions_internal(training_logic_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    training_logic_conditions_t* cond = &bridge->conditions;

    /* Update loss stability based on history variance */
    if (bridge->history_count >= 2) {
        float mean = 0.0f;
        float variance = 0.0f;
        uint32_t count = (bridge->history_count < DEFAULT_LOSS_TREND_WINDOW)
                         ? bridge->history_count : DEFAULT_LOSS_TREND_WINDOW;

        /* Compute mean */
        for (uint32_t i = 0; i < count; i++) {
            uint32_t idx = (bridge->history_head + bridge->history_count - 1 - i)
                           % bridge->config.history_size;
            mean += bridge->history[idx].loss;
        }
        mean /= count;

        /* Compute variance */
        for (uint32_t i = 0; i < count; i++) {
            uint32_t idx = (bridge->history_head + bridge->history_count - 1 - i)
                           % bridge->config.history_size;
            float diff = bridge->history[idx].loss - mean;
            variance += diff * diff;
        }
        variance /= count;

        cond->loss_stable = (variance < DEFAULT_LOSS_STABILITY_THRESHOLD);
    } else {
        cond->loss_stable = true;  /* Assume stable with insufficient history */
    }

    /* Update gradient stability */
    cond->grad_stable = (cond->grad_norm >= DEFAULT_GRAD_NORM_MIN &&
                         cond->grad_norm <= DEFAULT_GRAD_NORM_MAX);

    /* Update learning rate reasonableness */
    cond->lr_reasonable = (cond->learning_rate >= DEFAULT_LR_MIN &&
                           cond->learning_rate <= DEFAULT_LR_MAX);

    /* Update memory condition */
    cond->memory_ok = (cond->memory_usage < DEFAULT_MEMORY_THRESHOLD);

    /* Update throughput condition */
    cond->throughput_ok = (cond->throughput >= DEFAULT_THROUGHPUT_MIN);

    /* Update gradient explosion detection - use OR to preserve manual signals */
    cond->grad_exploding = cond->grad_exploding ||
                           (cond->grad_norm > DEFAULT_GRAD_NORM_MAX);

    /* Update loss NaN/Inf detection - use OR to preserve manual signals */
    cond->loss_nan = cond->loss_nan ||
                     (isnan(cond->loss_current) || isinf(cond->loss_current));

    /* Update divergence detection (loss increasing significantly) - use OR */
    if (bridge->history_count >= 2) {
        uint32_t latest_idx = (bridge->history_head + bridge->history_count - 1)
                              % bridge->config.history_size;
        uint32_t prev_idx = (bridge->history_head + bridge->history_count - 2)
                            % bridge->config.history_size;
        float loss_ratio = bridge->history[latest_idx].loss /
                           (bridge->history[prev_idx].loss + 1e-10f);
        cond->diverging = cond->diverging || (loss_ratio > 10.0f);
    }
    /* Note: diverging is NOT reset to false - once signaled, stays until cleared */

    /* Update stable step count */
    if (cond->loss_stable && cond->grad_stable && cond->lr_reasonable) {
        cond->stable_step_count++;
    } else {
        cond->stable_step_count = 0;
    }
    cond->stable_for_n_steps = (cond->stable_step_count >=
                                bridge->config.stable_steps_required);

    /* Update immune condition (query immune system if connected)
     * Only auto-update if system is actually connected - preserves manual test settings */
    if (bridge->immune_system && bridge->config.enable_immune_integration) {
        /* Would query immune inflammation level here */
        /* For now, assume OK - actual integration would check inflammation < SYSTEMIC */
        cond->immune_ok = true;
    }
    /* If integration enabled but no system connected, don't overwrite manual settings */

    /* Update resource condition (query Portia if connected) */
    if (bridge->portia_logic && bridge->config.enable_portia_integration) {
        /* Would query Portia resource state here */
        cond->resource_ok = true;
    }

    /* Update swarm consensus (query swarm if connected) */
    if (bridge->swarm_logic && bridge->config.enable_swarm_integration) {
        /* Would query swarm consensus here */
        cond->swarm_consensus = true;
    }

    /*=========================================================================
     * CROSS-BRIDGE INTEGRATION: Perception-Training → Logic
     *
     * Queries perception bridge for quality condition:
     * - lr_factor > 0.8 → perception_quality (good perception → safe to learn)
     *========================================================================*/
    if (bridge->perception_training) {
        perception_training_effects_t perception_effects;
        if (perception_training_get_effects(bridge->perception_training,
                                            &perception_effects) == 0 &&
            perception_effects.valid) {
            /* Perception quality based on lr_factor (already computed) */
            cond->perception_quality = (perception_effects.lr_factor > 0.8f);

            NIMCP_LOGGING_DEBUG("Perception → Logic: quality=%d (lr_factor=%.2f)",
                               cond->perception_quality, perception_effects.lr_factor);
        }
    }

    /*=========================================================================
     * CROSS-BRIDGE INTEGRATION: Cortical-Training → Logic
     *
     * Queries cortical bridge for stability conditions:
     * - predictions_stable → cortical_stable (converged predictive model)
     * - burst_rate > 0.5 → predictions_ok (dendrites confirming predictions)
     *========================================================================*/
    if (bridge->cortical_training) {
        cortical_training_effects_t cortical_effects;
        if (cortical_training_get_effects(bridge->cortical_training,
                                          &cortical_effects) == 0 &&
            cortical_effects.valid) {
            /* Cortical stability from predictions */
            cond->cortical_stable = cortical_effects.predictions_stable;

            /* Predictions OK based on burst rate */
            cond->predictions_ok = (cortical_effects.burst_rate > 0.5f);

            NIMCP_LOGGING_DEBUG("Cortical → Logic: stable=%d predictions_ok=%d (burst=%.2f)",
                               cond->cortical_stable, cond->predictions_ok,
                               cortical_effects.burst_rate);
        }
    }

    /* Update checkpoint tracking */
    cond->steps_since_checkpoint++;
    cond->sufficient_progress = (cond->steps_since_checkpoint >=
                                 bridge->config.checkpoint_interval);

    /* Update not_mid_batch (simplified - assume true) */
    cond->not_mid_batch = true;

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * INTERNAL DECISION HELPERS (NO MUTEX)
 * These functions assume the mutex is already held by the caller
 *============================================================================*/

/**
 * @brief Check stability (internal, no mutex)
 */
static bool check_stability_internal(training_logic_bridge_t* bridge) {
    return bridge->conditions.loss_stable &&
           bridge->conditions.grad_stable &&
           bridge->conditions.lr_reasonable;
}

/**
 * @brief Check if intervention needed (internal, no mutex)
 */
static bool needs_intervention_internal(training_logic_bridge_t* bridge) {
    return bridge->conditions.grad_exploding ||
           bridge->conditions.loss_nan ||
           bridge->conditions.diverging;
}

/**
 * @brief Check if safe to increase LR (internal, no mutex)
 */
static bool can_increase_lr_internal(training_logic_bridge_t* bridge) {
    return bridge->conditions.stable_for_n_steps &&
           bridge->conditions.immune_ok &&
           bridge->conditions.resource_ok;
}

/**
 * @brief Check batch adjustment (internal, no mutex)
 */
static bool should_adjust_batch_internal(training_logic_bridge_t* bridge, bool* increase_batch) {
    bool can_increase = bridge->conditions.memory_ok &&
                        bridge->conditions.throughput_ok;
    *increase_batch = can_increase;
    return true;  /* Always return true - there's always a recommendation */
}

/**
 * @brief Check checkpoint (internal, no mutex)
 */
static bool should_checkpoint_internal(training_logic_bridge_t* bridge) {
    return bridge->conditions.memory_ok &&
           bridge->conditions.not_mid_batch &&
           bridge->conditions.sufficient_progress;
}

/**
 * @brief Initialize pre-built decision gates
 *
 * WHAT: Creates standard decision gates for training logic
 * WHY:  Provide out-of-box decision logic
 * HOW:  Creates AND/OR/IMPLIES gates for common scenarios
 */
static int init_decision_gates(training_logic_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    NIMCP_CHECK_THROW(bridge->logic_network != NULL, NIMCP_ERROR_NULL_POINTER, "logic_network is NULL");

    /* STABILITY_CHECK gate: loss_stable AND grad_stable AND lr_reasonable
     * Threshold 2.9 requires all 3 inputs active (sum > 2.9) */
    bridge->stability_check_gate = neural_logic_create_gate(
        bridge->logic_network,
        LOGIC_GATE_AND,
        2.9f
    );
    if (bridge->stability_check_gate == UINT32_MAX) {
        NIMCP_LOGGING_ERROR("Failed to create stability check gate");
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    /* NEED_INTERVENTION gate: grad_exploding OR loss_nan OR diverging
     * Threshold 0.5 fires if any input active */
    bridge->intervention_gate = neural_logic_create_gate(
        bridge->logic_network,
        LOGIC_GATE_OR,
        0.5f
    );
    if (bridge->intervention_gate == UINT32_MAX) {
        NIMCP_LOGGING_ERROR("Failed to create intervention gate");
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    /* SAFE_TO_INCREASE_LR gate: stable_for_n_steps IMPLIES (immune_ok AND resource_ok)
     * IMPLIES logic: A -> B */
    bridge->lr_increase_gate = neural_logic_create_gate(
        bridge->logic_network,
        LOGIC_GATE_IMPLIES,
        0.7f
    );
    if (bridge->lr_increase_gate == UINT32_MAX) {
        NIMCP_LOGGING_ERROR("Failed to create LR increase gate");
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    /* BATCH_SIZE_DECISION gate: memory_ok AND throughput_ok
     * Threshold 1.5 requires both inputs active */
    bridge->batch_size_gate = neural_logic_create_gate(
        bridge->logic_network,
        LOGIC_GATE_AND,
        1.5f
    );
    if (bridge->batch_size_gate == UINT32_MAX) {
        NIMCP_LOGGING_ERROR("Failed to create batch size gate");
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    /* CHECKPOINT_DECISION gate: memory_ok AND not_mid_batch AND sufficient_progress
     * Threshold 2.5 requires all 3 inputs active */
    bridge->checkpoint_gate = neural_logic_create_gate(
        bridge->logic_network,
        LOGIC_GATE_AND,
        2.5f
    );
    if (bridge->checkpoint_gate == UINT32_MAX) {
        NIMCP_LOGGING_ERROR("Failed to create checkpoint gate");
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    NIMCP_LOGGING_INFO("Initialized 5 pre-built decision gates");

    return NIMCP_SUCCESS;
}

/**
 * @brief Add metric to history buffer
 *
 * WHAT: Circular buffer for metric tracking
 * WHY:  Needed for trend analysis and stability detection
 * HOW:  Ring buffer with head/count tracking
 */
static void add_metric_to_history(
    training_logic_bridge_t* bridge,
    float loss,
    float grad_norm)
{
    if (!bridge || !bridge->history) {
        return;
    }

    uint32_t idx = (bridge->history_head + bridge->history_count)
                   % bridge->config.history_size;

    bridge->history[idx].loss = loss;
    bridge->history[idx].grad_norm = grad_norm;
    bridge->history[idx].timestamp_ms = get_time_ms();

    if (bridge->history_count < bridge->config.history_size) {
        bridge->history_count++;
    } else {
        bridge->history_head = (bridge->history_head + 1) % bridge->config.history_size;
    }
}

/*=============================================================================
 * PUBLIC API - LIFECYCLE
 *============================================================================*/

void training_logic_default_config(training_logic_config_t* config) {
    if (!config) {
        return;
    }

    memset(config, 0, sizeof(training_logic_config_t));

    /* Mode */
    config->mode = TRAINING_LOGIC_MODE_ADVISORY;

    /* Gate thresholds */
    config->stability_threshold = 0.7f;
    config->intervention_threshold = 0.5f;
    config->lr_increase_threshold = 0.7f;
    config->confidence_threshold = TRAINING_LOGIC_DEFAULT_CONFIDENCE_THRESHOLD;

    /* Modulation parameters */
    config->lr_increase_factor = TRAINING_LOGIC_DEFAULT_LR_INCREASE_FACTOR;
    config->lr_decrease_factor = TRAINING_LOGIC_DEFAULT_LR_DECREASE_FACTOR;
    config->batch_scale_factor = TRAINING_LOGIC_DEFAULT_BATCH_SCALE_FACTOR;
    config->stable_steps_required = TRAINING_LOGIC_DEFAULT_STABLE_STEPS;
    config->checkpoint_interval = TRAINING_LOGIC_DEFAULT_CHECKPOINT_INTERVAL;

    /* Integration flags */
    config->enable_immune_integration = false;
    config->enable_portia_integration = false;
    config->enable_swarm_integration = false;
    config->enable_bio_async = true;

    /* Safety settings */
    config->min_learning_rate = 1e-8f;
    config->max_learning_rate = 1.0f;
    config->min_batch_size = 1;
    config->max_batch_size = 1024;

    /* Consensus settings */
    config->consensus_timeout_ms = TRAINING_LOGIC_DEFAULT_CONSENSUS_TIMEOUT_MS;
    config->consensus_threshold = 0.7f;

    /* History */
    config->history_size = TRAINING_LOGIC_MAX_HISTORY_SIZE;

    /* Testing */
    config->disable_auto_update = false;
}

training_logic_bridge_t* training_logic_create(
    const training_logic_config_t* config)
{
    /* Use default config if not provided */
    training_logic_config_t default_config;
    if (!config) {
        training_logic_default_config(&default_config);
        config = &default_config;
    }

    /* Allocate bridge */
    training_logic_bridge_t* bridge = (training_logic_bridge_t*)nimcp_malloc(
        sizeof(training_logic_bridge_t)
    );
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    memset(bridge, 0, sizeof(training_logic_bridge_t));

    /* Store config */
    memcpy(&bridge->config, config, sizeof(training_logic_config_t));

    /* Create mutex for thread safety */
    if (bridge_base_init(&bridge->base, 0, "training_logic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate history buffer */
    bridge->history = (training_metric_history_entry_t*)nimcp_malloc(
        sizeof(training_metric_history_entry_t) * config->history_size
    );
    if (!bridge->history) {
        NIMCP_LOGGING_ERROR("Failed to allocate history buffer");
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        return NULL;
    }
    memset(bridge->history, 0,
           sizeof(training_metric_history_entry_t) * config->history_size);

    /* Create neural logic network */
    neural_logic_config_t logic_config = neural_logic_default_config(
        TRAINING_LOGIC_MAX_CUSTOM_GATES + 10
    );
    logic_config.enable_bio_async = config->enable_bio_async;
    logic_config.use_gpu = false;  /* CPU-only for training decisions */

    bridge->logic_network = neural_logic_create(&logic_config);
    if (!bridge->logic_network) {
        NIMCP_LOGGING_ERROR("Failed to create neural logic network");
        nimcp_free(bridge->history);
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize decision gates */
    if (init_decision_gates(bridge) != NIMCP_SUCCESS) {
        NIMCP_LOGGING_ERROR("Failed to initialize decision gates");
        neural_logic_destroy(bridge->logic_network);
        nimcp_free(bridge->history);
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize custom gate tracking */
    bridge->next_custom_gate_id = TRAINING_LOGIC_GATE_CUSTOM_START;

    /* Initialize LR modulation */
    bridge->last_lr_factor = 1.0f;

    /* Initialize stats mode */
    bridge->stats.current_mode = config->mode;

    NIMCP_LOGGING_INFO("Created Training-Logic bridge");

    return bridge;
}

void training_logic_destroy(training_logic_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    /* Disconnect bio-async if connected */
    if (bridge->base.bio_async_enabled) {
        training_logic_disconnect_bio_async(bridge);
    }

    /* Destroy neural logic network */
    if (bridge->logic_network) {
        neural_logic_destroy(bridge->logic_network);
    }

    /* Free history buffer */
    if (bridge->history) {
        nimcp_free(bridge->history);
    }

    /* Destroy mutex */
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }

    /* Free bridge */
    nimcp_free(bridge);

    NIMCP_LOGGING_INFO("Destroyed Training-Logic bridge");
}

int training_logic_start(training_logic_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    /* Connect to bio-async if enabled */
    if (bridge->config.enable_bio_async && !bridge->base.bio_async_enabled) {
        int result = training_logic_connect_bio_async(bridge);
        if (result != NIMCP_SUCCESS) {
            NIMCP_LOGGING_WARN("Bio-async connection failed, continuing without it");
        }
    }

    /* Update initial conditions (unless disabled for testing) */
    if (!bridge->config.disable_auto_update) {
        update_conditions_internal(bridge);
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Started Training-Logic bridge");

    return NIMCP_SUCCESS;
}

int training_logic_stop(training_logic_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    /* Disconnect bio-async */
    if (bridge->base.bio_async_enabled) {
        training_logic_disconnect_bio_async(bridge);
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Stopped Training-Logic bridge");

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - INTEGRATION
 *============================================================================*/

int training_logic_connect_brain_training(
    training_logic_bridge_t* bridge,
    nimcp_brain_training_ctx_t* training_ctx)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->training_ctx = training_ctx;
    if (training_ctx) {
        NIMCP_LOGGING_INFO("Connected brain training context to Training-Logic bridge");
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int training_logic_connect_training_immune(
    training_logic_bridge_t* bridge,
    training_immune_system_t* immune_system)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->immune_system = immune_system;
    if (immune_system) {
        NIMCP_LOGGING_INFO("Connected training immune system to Training-Logic bridge");
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int training_logic_connect_portia_logic(
    training_logic_bridge_t* bridge,
    portia_logic_bridge_t* portia_logic)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->portia_logic = portia_logic;
    if (portia_logic) {
        NIMCP_LOGGING_INFO("Connected Portia-logic bridge to Training-Logic bridge");
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int training_logic_connect_swarm_logic(
    training_logic_bridge_t* bridge,
    swarm_logic_bridge_t* swarm_logic)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->swarm_logic = swarm_logic;
    if (swarm_logic) {
        NIMCP_LOGGING_INFO("Connected swarm-logic bridge to Training-Logic bridge");
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int training_logic_connect_unified(
    training_logic_bridge_t* bridge,
    portia_swarm_logic_bridge_t* unified_bridge)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->unified_bridge = unified_bridge;
    if (unified_bridge) {
        NIMCP_LOGGING_INFO("Connected unified bridge to Training-Logic bridge");
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int training_logic_connect_perception_training(
    training_logic_bridge_t* bridge,
    perception_training_bridge_t* perception_training)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->perception_training = perception_training;

    if (perception_training) {
        NIMCP_LOGGING_INFO("Connected perception-training bridge to Training-Logic bridge");
    } else {
        NIMCP_LOGGING_INFO("Disconnected perception-training bridge from Training-Logic bridge");
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int training_logic_connect_cortical_training(
    training_logic_bridge_t* bridge,
    cortical_training_bridge_t* cortical_training)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->cortical_training = cortical_training;

    if (cortical_training) {
        NIMCP_LOGGING_INFO("Connected cortical-training bridge to Training-Logic bridge");
    } else {
        NIMCP_LOGGING_INFO("Disconnected cortical-training bridge from Training-Logic bridge");
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - METRIC UPDATES
 *============================================================================*/

int training_logic_update_metrics(
    training_logic_bridge_t* bridge,
    float loss,
    float grad_norm,
    float learning_rate,
    uint64_t step)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update numeric conditions */
    bridge->conditions.loss_current = loss;
    bridge->conditions.grad_norm = grad_norm;
    bridge->conditions.learning_rate = learning_rate;
    bridge->conditions.current_step = step;

    /* Safety-critical NaN/Inf detection - always runs regardless of disable_auto_update */
    bridge->conditions.loss_nan = isnan(loss) || isinf(loss);
    bridge->conditions.grad_exploding = isnan(grad_norm) || isinf(grad_norm) ||
                                        grad_norm > bridge->config.intervention_threshold;

    /* Add to history */
    add_metric_to_history(bridge, loss, grad_norm);

    /* Update boolean conditions from metrics */
    if (!bridge->config.disable_auto_update) {
        update_conditions_internal(bridge);
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int training_logic_update_batch_metrics(
    training_logic_bridge_t* bridge,
    uint32_t batch_size,
    float throughput,
    float memory_usage)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update numeric conditions */
    bridge->conditions.memory_usage = memory_usage;
    bridge->conditions.throughput = throughput;
    bridge->last_batch_size = batch_size;

    /* Update boolean conditions */
    if (!bridge->config.disable_auto_update) {
        update_conditions_internal(bridge);
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int training_logic_signal_instability(
    training_logic_bridge_t* bridge,
    training_logic_instability_t instability_type,
    uint32_t severity)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    NIMCP_CHECK_THROW(instability_type < LOGIC_INSTABILITY_COUNT, NIMCP_ERROR_INVALID_PARAM, "instability_type out of range");

    nimcp_mutex_lock(bridge->base.mutex);

    /* Map instability type to conditions */
    switch (instability_type) {
        case LOGIC_INSTABILITY_LOSS_NAN:
        case LOGIC_INSTABILITY_LOSS_INF:
            bridge->conditions.loss_nan = true;
            break;

        case LOGIC_INSTABILITY_LOSS_EXPLOSION:
            bridge->conditions.diverging = true;
            break;

        case LOGIC_INSTABILITY_GRAD_EXPLOSION:
            bridge->conditions.grad_exploding = true;
            break;

        case LOGIC_INSTABILITY_GRAD_VANISHING:
            bridge->conditions.grad_stable = false;
            break;

        case LOGIC_INSTABILITY_LOSS_PLATEAU:
            bridge->conditions.loss_stable = false;
            break;

        case LOGIC_INSTABILITY_OSCILLATION:
            bridge->conditions.loss_stable = false;
            bridge->conditions.grad_stable = false;
            bridge->conditions.diverging = true;  /* Oscillation is a form of divergence */
            break;

        default:
            break;
    }

    /* Update stats */
    bridge->stats.intervention_triggers++;

    NIMCP_LOGGING_WARN("Training instability signaled: type=%u severity=%u",
                       instability_type, severity);

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - DECISION EVALUATION
 *============================================================================*/

bool training_logic_check_stability(training_logic_bridge_t* bridge) {
    if (!bridge || !bridge->logic_network) {
        return false;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update conditions (unless disabled for testing) */
    if (!bridge->config.disable_auto_update) {
        update_conditions_internal(bridge);
    }

    /* Use internal function */
    bool stable = check_stability_internal(bridge);

    /* Update stats */
    bridge->stats.stability_checks++;
    if (stable) {
        bridge->stats.stability_passed++;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return stable;
}

bool training_logic_needs_intervention(training_logic_bridge_t* bridge) {
    if (!bridge || !bridge->logic_network) {
        return false;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update conditions (unless disabled for testing) */
    if (!bridge->config.disable_auto_update) {
        update_conditions_internal(bridge);
    }

    /* Use internal function */
    bool needs_intervention = needs_intervention_internal(bridge);

    /* Update stats */
    if (needs_intervention) {
        bridge->stats.intervention_triggers++;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return needs_intervention;
}

bool training_logic_can_increase_lr(training_logic_bridge_t* bridge) {
    if (!bridge || !bridge->logic_network) {
        return false;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update conditions (unless disabled for testing) */
    if (!bridge->config.disable_auto_update) {
        update_conditions_internal(bridge);
    }

    /* Use internal function */
    bool can_increase = can_increase_lr_internal(bridge);

    /* Update stats */
    if (can_increase) {
        bridge->stats.lr_increase_allowed++;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return can_increase;
}

bool training_logic_should_adjust_batch(
    training_logic_bridge_t* bridge,
    bool* increase_batch)
{
    if (!bridge || !bridge->logic_network || !increase_batch) {
        return false;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update conditions (unless disabled for testing) */
    if (!bridge->config.disable_auto_update) {
        update_conditions_internal(bridge);
    }

    /* Use internal function */
    bool result = should_adjust_batch_internal(bridge, increase_batch);

    /* Update stats */
    bridge->stats.batch_adjustments++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

bool training_logic_should_checkpoint(training_logic_bridge_t* bridge) {
    if (!bridge || !bridge->logic_network) {
        return false;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update conditions (unless disabled for testing) */
    if (!bridge->config.disable_auto_update) {
        update_conditions_internal(bridge);
    }

    /* Use internal function */
    bool result = should_checkpoint_internal(bridge);

    /* Update stats */
    if (result) {
        bridge->stats.checkpoints_triggered++;
        /* Reset checkpoint tracking */
        bridge->conditions.steps_since_checkpoint = 0;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

int training_logic_get_decision(
    training_logic_bridge_t* bridge,
    training_logic_decision_t* decision)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    NIMCP_CHECK_THROW(decision != NULL, NIMCP_ERROR_NULL_POINTER, "decision is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    uint64_t start_time = get_time_us();

    /* Update conditions */
    if (!bridge->config.disable_auto_update) {
        update_conditions_internal(bridge);
    }

    memset(decision, 0, sizeof(training_logic_decision_t));

    /* Evaluate all gates using internal functions (mutex already held) */
    decision->stability_check_passed = check_stability_internal(bridge);
    decision->intervention_needed = needs_intervention_internal(bridge);
    decision->safe_to_increase_lr = can_increase_lr_internal(bridge);
    decision->checkpoint_needed = should_checkpoint_internal(bridge);

    bool increase_batch = false;
    decision->batch_size_ok = should_adjust_batch_internal(bridge, &increase_batch);

    /* Update stats for gate evaluations */
    bridge->stats.stability_checks++;
    if (decision->stability_check_passed) {
        bridge->stats.stability_passed++;
    }
    if (decision->intervention_needed) {
        bridge->stats.intervention_triggers++;
    }

    /* Determine decision priority (highest priority wins) */
    if (decision->intervention_needed) {
        /* Prioritize by severity: NaN > diverging > grad_exploding */
        if (bridge->conditions.loss_nan) {
            /* NaN is unrecoverable without rollback/pause */
            decision->type = TRAINING_DECISION_PAUSE;
            decision->confidence = 0.99f;
            snprintf(decision->reason, TRAINING_LOGIC_MAX_REASON_LENGTH,
                     "PAUSE: Loss is NaN/Inf, training must stop");
        } else if (bridge->conditions.diverging) {
            /* Divergence needs pause to prevent further damage */
            decision->type = TRAINING_DECISION_PAUSE;
            decision->confidence = 0.95f;
            snprintf(decision->reason, TRAINING_LOGIC_MAX_REASON_LENGTH,
                     "PAUSE: Training diverging, intervention required");
        } else {
            /* Gradient explosion can be recovered with LR decrease */
            decision->type = TRAINING_DECISION_DECREASE_LR;
            decision->confidence = 0.90f;
            snprintf(decision->reason, TRAINING_LOGIC_MAX_REASON_LENGTH,
                     "Decrease LR: grad_exploding=%d",
                     bridge->conditions.grad_exploding);
            bridge->stats.lr_decrease_triggered++;
        }
        decision->approved = true;
        decision->modulation_factor = bridge->config.lr_decrease_factor;
    } else if (decision->checkpoint_needed) {
        decision->type = TRAINING_DECISION_CHECKPOINT;
        decision->approved = true;
        decision->confidence = 0.90f;
        decision->modulation_factor = 1.0f;
        snprintf(decision->reason, TRAINING_LOGIC_MAX_REASON_LENGTH,
                 "Checkpoint recommended: %u steps since last",
                 bridge->conditions.steps_since_checkpoint);
    } else if (decision->safe_to_increase_lr) {
        decision->type = TRAINING_DECISION_INCREASE_LR;
        decision->approved = true;
        decision->confidence = 0.85f;
        decision->modulation_factor = bridge->config.lr_increase_factor;
        snprintf(decision->reason, TRAINING_LOGIC_MAX_REASON_LENGTH,
                 "Safe to increase LR: stable for %u steps",
                 bridge->conditions.stable_step_count);
        bridge->stats.lr_increases++;
    } else if (decision->batch_size_ok && !increase_batch) {
        /* Only recommend batch DECREASE when conditions require it.
         * When increase_batch is true (can increase), we don't force it -
         * let training continue normally if stable. */
        decision->type = TRAINING_DECISION_DECREASE_BATCH;
        decision->modulation_factor = bridge->config.batch_scale_factor;
        bridge->stats.batch_decreases++;
        decision->approved = true;
        decision->confidence = 0.75f;
        snprintf(decision->reason, TRAINING_LOGIC_MAX_REASON_LENGTH,
                 "Batch size decrease: memory=%.2f throughput=%.2f",
                 bridge->conditions.memory_usage,
                 bridge->conditions.throughput);
    } else if (decision->stability_check_passed) {
        decision->type = TRAINING_DECISION_CONTINUE;
        decision->approved = true;
        decision->confidence = 0.80f;
        decision->modulation_factor = 1.0f;
        snprintf(decision->reason, TRAINING_LOGIC_MAX_REASON_LENGTH,
                 "Training stable, continue normally");
    } else {
        decision->type = TRAINING_DECISION_CONTINUE;
        decision->approved = true;
        decision->confidence = 0.60f;
        decision->modulation_factor = 1.0f;
        snprintf(decision->reason, TRAINING_LOGIC_MAX_REASON_LENGTH,
                 "No strong decision, continue monitoring");
    }

    uint64_t end_time = get_time_us();
    decision->evaluation_time_us = end_time - start_time;

    /* Update stats */
    bridge->stats.total_decisions++;
    bridge->stats.decisions_by_type[decision->type]++;
    bridge->stats.total_decision_time_us += decision->evaluation_time_us;
    bridge->stats.last_decision_time_ms = get_time_ms();

    /* Update average timing */
    if (bridge->stats.total_decisions > 0) {
        bridge->stats.avg_decision_time_us =
            (float)bridge->stats.total_decision_time_us / bridge->stats.total_decisions;
    }
    if (decision->evaluation_time_us > (uint64_t)bridge->stats.max_decision_time_us) {
        bridge->stats.max_decision_time_us = (float)decision->evaluation_time_us;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - MODULATION
 *============================================================================*/

float training_logic_get_lr_modulation(
    const training_logic_bridge_t* bridge,
    float base_lr)
{
    if (!bridge) {
        return base_lr;
    }

    float modulated_lr = base_lr * bridge->last_lr_factor;

    /* Clamp to safety bounds */
    if (modulated_lr < bridge->config.min_learning_rate) {
        modulated_lr = bridge->config.min_learning_rate;
    }
    if (modulated_lr > bridge->config.max_learning_rate) {
        modulated_lr = bridge->config.max_learning_rate;
    }

    return modulated_lr;
}

uint32_t training_logic_get_batch_size_modulation(
    const training_logic_bridge_t* bridge,
    uint32_t base_batch_size)
{
    if (!bridge) {
        return base_batch_size;
    }

    /* Use last batch size if available, otherwise base */
    uint32_t modulated = (bridge->last_batch_size > 0)
                         ? bridge->last_batch_size : base_batch_size;

    /* Clamp to safety bounds */
    if (modulated < bridge->config.min_batch_size) {
        modulated = bridge->config.min_batch_size;
    }
    if (modulated > bridge->config.max_batch_size) {
        modulated = bridge->config.max_batch_size;
    }

    return modulated;
}

int training_logic_apply_decision(
    training_logic_bridge_t* bridge,
    const training_logic_decision_t* decision)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    NIMCP_CHECK_THROW(decision != NULL, NIMCP_ERROR_NULL_POINTER, "decision is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    /* Only apply if in AUTOMATIC mode */
    if (bridge->config.mode != TRAINING_LOGIC_MODE_AUTOMATIC) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return NIMCP_SUCCESS;  /* Advisory mode - don't apply */
    }

    /* Apply modulation factor based on decision type */
    switch (decision->type) {
        case TRAINING_DECISION_INCREASE_LR:
        case TRAINING_DECISION_DECREASE_LR:
            bridge->last_lr_factor = decision->modulation_factor;
            break;

        case TRAINING_DECISION_INCREASE_BATCH:
        case TRAINING_DECISION_DECREASE_BATCH:
            bridge->last_batch_size = (uint32_t)(bridge->last_batch_size *
                                                 decision->modulation_factor);
            break;

        case TRAINING_DECISION_PAUSE:
            bridge->stats.currently_paused = true;
            break;

        case TRAINING_DECISION_RESUME:
            bridge->stats.currently_paused = false;
            break;

        default:
            break;
    }

    NIMCP_LOGGING_INFO("Applied decision: type=%s factor=%.3f",
                       training_logic_decision_type_to_string(decision->type),
                       decision->modulation_factor);

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - CONDITION MANAGEMENT
 *============================================================================*/

int training_logic_update_conditions(training_logic_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    int result = update_conditions_internal(bridge);
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

int training_logic_get_conditions(
    const training_logic_bridge_t* bridge,
    training_logic_conditions_t* conditions)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    NIMCP_CHECK_THROW(conditions != NULL, NIMCP_ERROR_NULL_POINTER, "conditions is NULL");

    memcpy(conditions, &bridge->conditions, sizeof(training_logic_conditions_t));

    return NIMCP_SUCCESS;
}

int training_logic_set_condition(
    training_logic_bridge_t* bridge,
    training_logic_condition_t condition,
    bool value)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    NIMCP_CHECK_THROW(condition < TRAINING_COND_COUNT, NIMCP_ERROR_INVALID_PARAM, "condition out of range");

    nimcp_mutex_lock(bridge->base.mutex);

    switch (condition) {
        case TRAINING_COND_LOSS_STABLE:
            bridge->conditions.loss_stable = value;
            break;
        case TRAINING_COND_GRAD_STABLE:
            bridge->conditions.grad_stable = value;
            break;
        case TRAINING_COND_LR_REASONABLE:
            bridge->conditions.lr_reasonable = value;
            break;
        case TRAINING_COND_MEMORY_OK:
            bridge->conditions.memory_ok = value;
            break;
        case TRAINING_COND_THROUGHPUT_OK:
            bridge->conditions.throughput_ok = value;
            break;
        case TRAINING_COND_NOT_MID_BATCH:
            bridge->conditions.not_mid_batch = value;
            break;
        case TRAINING_COND_SUFFICIENT_PROGRESS:
            bridge->conditions.sufficient_progress = value;
            break;
        case TRAINING_COND_GRAD_EXPLODING:
            bridge->conditions.grad_exploding = value;
            break;
        case TRAINING_COND_LOSS_NAN:
            bridge->conditions.loss_nan = value;
            break;
        case TRAINING_COND_DIVERGING:
            bridge->conditions.diverging = value;
            break;
        case TRAINING_COND_STABLE_FOR_N_STEPS:
            bridge->conditions.stable_for_n_steps = value;
            break;
        case TRAINING_COND_IMMUNE_OK:
            bridge->conditions.immune_ok = value;
            break;
        case TRAINING_COND_RESOURCE_OK:
            bridge->conditions.resource_ok = value;
            break;
        case TRAINING_COND_SWARM_CONSENSUS:
            bridge->conditions.swarm_consensus = value;
            break;
        case TRAINING_COND_PERCEPTION_QUALITY:
            bridge->conditions.perception_quality = value;
            break;
        case TRAINING_COND_CORTICAL_STABLE:
            bridge->conditions.cortical_stable = value;
            break;
        case TRAINING_COND_PREDICTIONS_OK:
            bridge->conditions.predictions_ok = value;
            break;
        default:
            nimcp_mutex_unlock(bridge->base.mutex);
            return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int training_logic_set_numeric_condition(
    training_logic_bridge_t* bridge,
    const char* name,
    float value)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    NIMCP_CHECK_THROW(name != NULL, NIMCP_ERROR_NULL_POINTER, "name is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    if (strcmp(name, "loss") == 0) {
        bridge->conditions.loss_current = value;
    } else if (strcmp(name, "grad_norm") == 0) {
        bridge->conditions.grad_norm = value;
    } else if (strcmp(name, "learning_rate") == 0) {
        bridge->conditions.learning_rate = value;
    } else if (strcmp(name, "memory_usage") == 0) {
        bridge->conditions.memory_usage = value;
    } else if (strcmp(name, "throughput") == 0) {
        bridge->conditions.throughput = value;
    } else if (strcmp(name, "loss_trend") == 0) {
        bridge->conditions.loss_trend = value;
    } else {
        NIMCP_LOGGING_ERROR("Unknown numeric condition: %s", name);
        nimcp_mutex_unlock(bridge->base.mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - CUSTOM GATES
 *============================================================================*/

int training_logic_add_custom_gate(
    training_logic_bridge_t* bridge,
    const char* expression,
    uint32_t* gate_id)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    NIMCP_CHECK_THROW(expression != NULL, NIMCP_ERROR_NULL_POINTER, "expression is NULL");
    NIMCP_CHECK_THROW(gate_id != NULL, NIMCP_ERROR_NULL_POINTER, "gate_id is NULL");

    if (bridge->stats.custom_gate_count >= TRAINING_LOGIC_MAX_CUSTOM_GATES) {
        NIMCP_LOGGING_ERROR("Maximum custom gates reached");
        return NIMCP_ERROR_INVALID_STATE;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Simple expression parsing - support basic patterns */
    logic_gate_type_t gate_type = LOGIC_GATE_AND;
    if (strstr(expression, "AND")) {
        gate_type = LOGIC_GATE_AND;
    } else if (strstr(expression, "OR")) {
        gate_type = LOGIC_GATE_OR;
    } else if (strstr(expression, "NOT")) {
        gate_type = LOGIC_GATE_NOT;
    } else if (strstr(expression, "XOR")) {
        gate_type = LOGIC_GATE_XOR;
    } else if (strstr(expression, "IMPLIES")) {
        gate_type = LOGIC_GATE_IMPLIES;
    } else {
        NIMCP_LOGGING_ERROR("Unsupported gate expression: %s", expression);
        nimcp_mutex_unlock(bridge->base.mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Create gate */
    uint32_t new_gate_id = neural_logic_create_gate(
        bridge->logic_network,
        gate_type,
        0.7f  /* Default threshold */
    );

    if (new_gate_id == UINT32_MAX) {
        NIMCP_LOGGING_ERROR("Failed to create custom gate");
        nimcp_mutex_unlock(bridge->base.mutex);
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    /* Use next_custom_gate_id for external ID, starting at TRAINING_LOGIC_GATE_CUSTOM_START */
    *gate_id = bridge->next_custom_gate_id++;
    bridge->stats.custom_gate_count++;

    NIMCP_LOGGING_INFO("Created custom gate %u (internal: %u): %s",
                       *gate_id, new_gate_id, expression);

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

bool training_logic_evaluate_gate(
    training_logic_bridge_t* bridge,
    uint32_t gate_id)
{
    if (!bridge || !bridge->logic_network) {
        return false;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update conditions */
    if (!bridge->config.disable_auto_update) {
        update_conditions_internal(bridge);
    }

    /* Prepare generic inputs based on current conditions */
    float inputs[8] = {
        bridge->conditions.loss_stable ? 1.0f : 0.0f,
        bridge->conditions.grad_stable ? 1.0f : 0.0f,
        bridge->conditions.lr_reasonable ? 1.0f : 0.0f,
        bridge->conditions.memory_ok ? 1.0f : 0.0f,
        bridge->conditions.grad_exploding ? 1.0f : 0.0f,
        bridge->conditions.loss_nan ? 1.0f : 0.0f,
        bridge->conditions.diverging ? 1.0f : 0.0f,
        bridge->conditions.immune_ok ? 1.0f : 0.0f
    };

    float output = 0.0f;
    bool success = neural_logic_evaluate(
        bridge->logic_network,
        gate_id,
        inputs,
        8,
        &output
    );

    nimcp_mutex_unlock(bridge->base.mutex);

    return success && (output >= bridge->config.confidence_threshold);
}

int training_logic_get_gate_decision(
    training_logic_bridge_t* bridge,
    uint32_t gate_id,
    training_logic_decision_t* decision)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    NIMCP_CHECK_THROW(decision != NULL, NIMCP_ERROR_NULL_POINTER, "decision is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update conditions */
    if (!bridge->config.disable_auto_update) {
        update_conditions_internal(bridge);
    }

    /* Evaluate gate */
    float inputs[8] = {
        bridge->conditions.loss_stable ? 1.0f : 0.0f,
        bridge->conditions.grad_stable ? 1.0f : 0.0f,
        bridge->conditions.lr_reasonable ? 1.0f : 0.0f,
        bridge->conditions.memory_ok ? 1.0f : 0.0f,
        bridge->conditions.grad_exploding ? 1.0f : 0.0f,
        bridge->conditions.loss_nan ? 1.0f : 0.0f,
        bridge->conditions.diverging ? 1.0f : 0.0f,
        bridge->conditions.immune_ok ? 1.0f : 0.0f
    };

    float output = 0.0f;
    uint64_t start_time = get_time_us();
    bool success = neural_logic_evaluate(
        bridge->logic_network,
        gate_id,
        inputs,
        8,
        &output
    );
    uint64_t end_time = get_time_us();

    /* Populate decision */
    memset(decision, 0, sizeof(training_logic_decision_t));
    decision->type = TRAINING_DECISION_CONTINUE;
    decision->approved = success && (output >= bridge->config.confidence_threshold);
    decision->confidence = output;
    decision->modulation_factor = 1.0f;
    decision->evaluation_time_us = end_time - start_time;
    snprintf(decision->reason, TRAINING_LOGIC_MAX_REASON_LENGTH,
             "Gate %u evaluated to %.2f (threshold %.2f)",
             gate_id, output, bridge->config.confidence_threshold);

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - BIO-ASYNC
 *============================================================================*/

int training_logic_connect_bio_async(training_logic_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    if (bridge->base.bio_async_enabled) {
        return NIMCP_SUCCESS;  /* Already connected */
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_TRAINING_LOGIC,
        .module_name = "training_logic_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
        return NIMCP_SUCCESS;
    }

    NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    return NIMCP_ERROR_OPERATION_FAILED;
}

int training_logic_disconnect_bio_async(training_logic_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    if (!bridge->base.bio_async_enabled) {
        return NIMCP_SUCCESS;  /* Already disconnected */
    }

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected from bio-async router");

    return NIMCP_SUCCESS;
}

bool training_logic_is_bio_async_connected(const training_logic_bridge_t* bridge) {
    if (!bridge) {
        return false;
    }

    return bridge->base.bio_async_enabled;
}

int training_logic_process_inbox(training_logic_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    if (!bridge->base.bio_async_enabled || !bridge->base.bio_ctx) {
        return 0;  /* No messages to process */
    }

    return bio_router_process_inbox(bridge->base.bio_ctx, 10);
}

int training_logic_broadcast_decision(
    training_logic_bridge_t* bridge,
    const training_logic_decision_t* decision)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    NIMCP_CHECK_THROW(decision != NULL, NIMCP_ERROR_NULL_POINTER, "decision is NULL");

    if (!bridge->base.bio_async_enabled || !bridge->base.bio_ctx) {
        return NIMCP_ERROR_INVALID_STATE;  /* Bio-async not connected */
    }

    /* Create bio-async message with proper header */
    bio_msg_logic_gate_result_t msg = {0};
    bio_msg_init_header(&msg.header, BIO_MSG_LOGIC_GATE_RESULT,
                        bio_module_context_get_id(bridge->base.bio_ctx), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;

    /* Map decision to gate result */
    msg.gate_id = 0;  /* Generic decision */
    msg.gate_type = 0;
    msg.output = decision->approved ? 1.0f : 0.0f;
    msg.spiked = decision->approved;
    msg.spike_time_us = decision->evaluation_time_us;
    msg.threshold_used = decision->confidence;

    /* Broadcast to subscribers */
    bio_router_broadcast(bridge->base.bio_ctx, &msg, sizeof(msg));

    NIMCP_LOGGING_DEBUG("Broadcast decision: type=%s approved=%d",
                        training_logic_decision_type_to_string(decision->type),
                        decision->approved);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - STATISTICS
 *============================================================================*/

int training_logic_get_stats(
    const training_logic_bridge_t* bridge,
    training_logic_stats_t* stats)
{
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    NIMCP_CHECK_THROW(stats != NULL, NIMCP_ERROR_NULL_POINTER, "stats is NULL");

    memcpy(stats, &bridge->stats, sizeof(training_logic_stats_t));

    return NIMCP_SUCCESS;
}

int training_logic_reset_stats(training_logic_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge != NULL, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    /* Preserve mode and custom gate count */
    training_logic_mode_t mode = bridge->stats.current_mode;
    uint32_t custom_gates = bridge->stats.custom_gate_count;

    memset(&bridge->stats, 0, sizeof(training_logic_stats_t));

    bridge->stats.current_mode = mode;
    bridge->stats.custom_gate_count = custom_gates;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Reset Training-Logic statistics");

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - UTILITIES
 *============================================================================*/

const char* training_logic_condition_to_string(training_logic_condition_t condition) {
    switch (condition) {
        case TRAINING_COND_LOSS_STABLE: return "loss_stable";
        case TRAINING_COND_GRAD_STABLE: return "grad_stable";
        case TRAINING_COND_LR_REASONABLE: return "lr_reasonable";
        case TRAINING_COND_MEMORY_OK: return "memory_ok";
        case TRAINING_COND_THROUGHPUT_OK: return "throughput_ok";
        case TRAINING_COND_NOT_MID_BATCH: return "not_mid_batch";
        case TRAINING_COND_SUFFICIENT_PROGRESS: return "sufficient_progress";
        case TRAINING_COND_GRAD_EXPLODING: return "grad_exploding";
        case TRAINING_COND_LOSS_NAN: return "loss_nan";
        case TRAINING_COND_DIVERGING: return "diverging";
        case TRAINING_COND_STABLE_FOR_N_STEPS: return "stable_for_n_steps";
        case TRAINING_COND_IMMUNE_OK: return "immune_ok";
        case TRAINING_COND_RESOURCE_OK: return "resource_ok";
        case TRAINING_COND_SWARM_CONSENSUS: return "swarm_consensus";
        case TRAINING_COND_PERCEPTION_QUALITY: return "perception_quality";
        case TRAINING_COND_CORTICAL_STABLE: return "cortical_stable";
        case TRAINING_COND_PREDICTIONS_OK: return "predictions_ok";
        default: return "unknown";
    }
}

const char* training_logic_decision_type_to_string(training_logic_decision_type_t type) {
    switch (type) {
        case TRAINING_DECISION_CONTINUE: return "continue";
        case TRAINING_DECISION_PAUSE: return "pause";
        case TRAINING_DECISION_RESUME: return "resume";
        case TRAINING_DECISION_INCREASE_LR: return "increase_lr";
        case TRAINING_DECISION_DECREASE_LR: return "decrease_lr";
        case TRAINING_DECISION_INCREASE_BATCH: return "increase_batch";
        case TRAINING_DECISION_DECREASE_BATCH: return "decrease_batch";
        case TRAINING_DECISION_CHECKPOINT: return "checkpoint";
        case TRAINING_DECISION_ROLLBACK: return "rollback";
        case TRAINING_DECISION_TERMINATE: return "terminate";
        default: return "unknown";
    }
}

const char* training_logic_mode_to_string(training_logic_mode_t mode) {
    switch (mode) {
        case TRAINING_LOGIC_MODE_DISABLED: return "disabled";
        case TRAINING_LOGIC_MODE_MONITOR_ONLY: return "monitor_only";
        case TRAINING_LOGIC_MODE_ADVISORY: return "advisory";
        case TRAINING_LOGIC_MODE_AUTOMATIC: return "automatic";
        case TRAINING_LOGIC_MODE_CONSENSUS_REQUIRED: return "consensus_required";
        default: return "unknown";
    }
}

void training_logic_dump_state(const training_logic_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    NIMCP_LOGGING_INFO("=== Training-Logic Bridge State ===");
    NIMCP_LOGGING_INFO("Mode: %s", training_logic_mode_to_string(bridge->stats.current_mode));
    NIMCP_LOGGING_INFO("Conditions:");
    NIMCP_LOGGING_INFO("  loss_stable=%d grad_stable=%d lr_reasonable=%d",
                       bridge->conditions.loss_stable,
                       bridge->conditions.grad_stable,
                       bridge->conditions.lr_reasonable);
    NIMCP_LOGGING_INFO("  memory_ok=%d throughput_ok=%d",
                       bridge->conditions.memory_ok,
                       bridge->conditions.throughput_ok);
    NIMCP_LOGGING_INFO("  grad_exploding=%d loss_nan=%d diverging=%d",
                       bridge->conditions.grad_exploding,
                       bridge->conditions.loss_nan,
                       bridge->conditions.diverging);
    NIMCP_LOGGING_INFO("  stable_for_n_steps=%d (count=%u)",
                       bridge->conditions.stable_for_n_steps,
                       bridge->conditions.stable_step_count);
    NIMCP_LOGGING_INFO("Metrics:");
    NIMCP_LOGGING_INFO("  loss=%.6f grad_norm=%.6f lr=%.6e",
                       bridge->conditions.loss_current,
                       bridge->conditions.grad_norm,
                       bridge->conditions.learning_rate);
    NIMCP_LOGGING_INFO("  memory=%.2f throughput=%.2f",
                       bridge->conditions.memory_usage,
                       bridge->conditions.throughput);
    NIMCP_LOGGING_INFO("  step=%lu steps_since_checkpoint=%u",
                       bridge->conditions.current_step,
                       bridge->conditions.steps_since_checkpoint);
    NIMCP_LOGGING_INFO("Statistics:");
    NIMCP_LOGGING_INFO("  total_decisions=%lu stability_checks=%lu",
                       bridge->stats.total_decisions,
                       bridge->stats.stability_checks);
    NIMCP_LOGGING_INFO("  lr_increases=%lu lr_decreases=%lu",
                       bridge->stats.lr_increases,
                       bridge->stats.lr_decreases);
    NIMCP_LOGGING_INFO("  checkpoints=%lu interventions=%lu",
                       bridge->stats.checkpoints_triggered,
                       bridge->stats.intervention_triggers);
    NIMCP_LOGGING_INFO("  custom_gates=%u", bridge->stats.custom_gate_count);
    NIMCP_LOGGING_INFO("====================================");
}
