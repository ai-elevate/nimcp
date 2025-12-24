//=============================================================================
// nimcp_cortical_training_bridge.c - Cortical-Training Bridge Implementation
//=============================================================================
//
// WHAT: Bidirectional bridge integrating cortical modules with training
//       pipeline for cortically-aware learning rate and gradient modulation
//
// WHY:  Models how cortical dynamics (predictive coding, dendritic bursts,
//       column competition) affect learning. Free energy guides learning
//       intensity, burst rate indicates stability, winner confidence shows
//       representation quality.
//
// HOW:  Cortical → Training: Modulate LR and gradient confidence
//       Training → Cortical: Consolidate predictions, adjust PE gains
//       Integrates with cognitive-training, logic-training, immune-training
//
// BIOLOGICAL BASIS:
// - Free energy: High FE → learning opportunity (boost LR)
// - Dendritic bursts: High burst rate → stable predictions (maintain LR)
// - Column competition: High winner confidence → good representations (stable LR)
// - Precision weights: Attention-like modulation of prediction errors
//
//=============================================================================

#include "middleware/training/nimcp_cortical_training_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "middleware/training/nimcp_cognitive_training_bridge.h"
#include "middleware/training/nimcp_training_logic_bridge.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/error/nimcp_error_codes.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <math.h>

/*=============================================================================
 * TIME HELPERS
 *============================================================================*/

/**
 * @brief Get current time in milliseconds
 *
 * WHAT: Returns monotonic clock time in milliseconds
 * WHY:  Used for tracking update intervals
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

/* BIO_MODULE_CORTICAL_TRAINING should be 0x0524 */
#ifndef BIO_MODULE_CORTICAL_TRAINING
#define BIO_MODULE_CORTICAL_TRAINING 0x0524
#endif

/* Clamping limits */
#define CORTICAL_LR_FACTOR_MIN  0.3f    /**< Minimum LR factor */
#define CORTICAL_LR_FACTOR_MAX  1.15f   /**< Maximum LR factor */
#define CORTICAL_GRAD_CONF_MIN  0.5f    /**< Minimum gradient confidence */
#define CORTICAL_GRAD_CONF_MAX  1.0f    /**< Maximum gradient confidence */

/* History size for trend analysis */
#define CORTICAL_HISTORY_SIZE  100  /**< Number of FE values to track */

/*=============================================================================
 * DATA STRUCTURES
 *============================================================================*/

/**
 * @brief Main bridge structure
 *
 * WHAT: Internal state for cortical-training bridge
 * WHY:  Encapsulates all bridge data and integrations
 * HOW:  Single struct with all subsystems
 */
struct cortical_training_bridge {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */

    cortical_training_config_t config;

    /* Connected cortical modules (may be NULL) */
    predictive_coding_context_t* predictive_coding;
    dendritic_compartment_t* dendritic;
    hypercolumn_t* columns;

    /* Connected training components */
    cognitive_training_bridge_t* cognitive_training;
    training_logic_bridge_t* training_logic;
    training_immune_system_t* training_immune;
    training_plasticity_bridge_t* training_plasticity;

    /* Current effects */
    cortical_training_effects_t cortical_effects;
    training_cortical_effects_t training_effects;

    /* Free energy history for trend analysis */
    float* fe_history;
    uint32_t history_head;
    uint32_t history_count;

    /* Statistics */
    cortical_training_stats_t stats;

    /* State */
    bool running;
    uint64_t last_update_ms;
};

/*=============================================================================
 * HELPER FUNCTIONS
 *============================================================================*/

/**
 * @brief Clamp float value to range
 *
 * WHAT: Limits value between min and max
 * WHY:  Prevent extreme modulation factors
 * HOW:  Returns min if below, max if above, value otherwise
 */
static inline float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief Extract cortical state from modules
 *
 * WHAT: Queries all connected cortical modules for current state
 * WHY:  Aggregates cortical metrics for modulation computation
 * HOW:  Calls getter functions on each module, handles NULL safely
 */
static int extract_cortical_state(cortical_training_bridge_t* bridge) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    cortical_training_effects_t* effects = &bridge->cortical_effects;

    /* TODO: When cortical module APIs are implemented, query them here.
     * For now, use default/placeholder values that produce neutral modulation.
     * This allows the bridge to be tested and integrated without requiring
     * all cortical modules to be fully implemented.
     */

    /* Predictive Coding: Default to moderate free energy */
    if (bridge->predictive_coding && bridge->config.enable_predictive_coding) {
        effects->free_energy = 5.0f;               /* Moderate FE */
        effects->prediction_error_mag = 1.0f;      /* Moderate PE */
        effects->convergence_rate = 0.5f;          /* Moderate convergence */

        /* Default precision weights (all equal) */
        if (effects->precision_weights && effects->num_layers > 0) {
            for (uint32_t i = 0; i < effects->num_layers; i++) {
                effects->precision_weights[i] = 1.0f;
            }
        }
    }

    /* Dendritic: Default to moderate burst rate */
    if (bridge->dendritic && bridge->config.enable_dendritic) {
        effects->burst_rate = 0.5f;           /* Moderate burst rate */
        effects->bac_success_rate = 0.5f;     /* Moderate BAC success */
        effects->calcium_spikes = 5.0f;       /* Moderate Ca spikes */
    }

    /* Cortical Columns: Default to moderate winner confidence */
    if (bridge->columns && bridge->config.enable_columns) {
        effects->winner_confidence = 0.6f;    /* Moderate confidence */
        effects->population_entropy = 1.0f;   /* Moderate entropy */
        effects->inhibition_strength = 0.5f;  /* Moderate inhibition */
    }

    effects->valid = true;

    return NIMCP_SUCCESS;
}

/**
 * @brief Compute learning rate modulation factor
 *
 * WHAT: Calculates multiplicative LR factor from cortical states
 * WHY:  Adjusts learning rate based on cortical dynamics
 * HOW:  Combines effects from predictive coding, dendritic, and columns
 *
 * FORMULAS:
 * - High burst rate → stable predictions → maintain/increase LR (factor 1.0-1.15)
 * - Low burst rate → unstable → reduce LR (factor 0.7-1.0)
 * - High free energy → learning opportunity → boost LR (factor 1.1)
 * - Free energy explosion → danger → drastically reduce LR (factor 0.3)
 * - High winner confidence → good representations → stable LR
 * - High entropy → uncertain → conservative LR (factor 0.8)
 * Final factor clamped to [0.3, 1.15]
 */
static float compute_lr_modulation(
    const cortical_training_bridge_t* bridge,
    const cortical_training_effects_t* effects)
{
    float lr_factor = 1.0f;

    /* Predictive Coding: Free energy modulation */
    if (bridge->config.enable_predictive_coding) {
        float fe = effects->free_energy;

        if (fe > bridge->config.fe_explosion_threshold) {
            /* Free energy explosion → emergency reduction */
            lr_factor *= 0.3f;
        } else if (fe > bridge->config.fe_high_threshold) {
            /* High free energy → boost learning (surprise = opportunity) */
            float boost = 1.0f + (fe - bridge->config.fe_high_threshold) * 0.01f;
            boost = clamp_f(boost, 1.0f, 1.1f);
            lr_factor *= boost * bridge->config.predictive_strength;
        } else {
            /* Normal free energy → maintain LR */
            lr_factor *= 1.0f;
        }
    }

    /* Dendritic: Burst rate modulation */
    if (bridge->config.enable_dendritic) {
        float burst_rate = effects->burst_rate;

        if (burst_rate < bridge->config.burst_collapse_threshold) {
            /* Burst collapse → instability → reduce LR */
            lr_factor *= 0.7f;
        } else if (burst_rate > CORTICAL_TRAINING_BURST_RATE_HIGH_THRESHOLD) {
            /* High burst rate → stable predictions → boost LR slightly */
            float boost = 1.0f + (burst_rate - CORTICAL_TRAINING_BURST_RATE_HIGH_THRESHOLD) * 0.5f;
            boost = clamp_f(boost, 1.0f, 1.15f);
            lr_factor *= boost * bridge->config.dendritic_strength;
        } else if (burst_rate < CORTICAL_TRAINING_BURST_RATE_LOW_THRESHOLD) {
            /* Low burst rate → unstable → reduce LR */
            float reduction = 1.0f - (CORTICAL_TRAINING_BURST_RATE_LOW_THRESHOLD - burst_rate) * 0.5f;
            reduction = clamp_f(reduction, 0.7f, 1.0f);
            lr_factor *= reduction;
        }
    }

    /* Cortical Columns: Winner confidence and entropy modulation */
    if (bridge->config.enable_columns) {
        float winner_conf = effects->winner_confidence;
        float entropy = effects->population_entropy;

        /* High winner confidence → stable representations → maintain LR */
        if (winner_conf > 0.8f) {
            lr_factor *= 1.0f;  /* Neutral */
        }

        /* High entropy → uncertain → conservative LR */
        if (entropy > 1.5f) {
            lr_factor *= 0.8f * bridge->config.columns_strength;
        }
    }

    /* Clamp to safety bounds */
    return clamp_f(lr_factor, CORTICAL_LR_FACTOR_MIN, CORTICAL_LR_FACTOR_MAX);
}

/**
 * @brief Compute gradient confidence factor
 *
 * WHAT: Calculates confidence in gradient direction from cortical states
 * WHY:  Scale gradient updates by confidence
 * HOW:  Combines burst rate and winner confidence
 *
 * FORMULAS:
 * - High burst rate → stable predictions → high confidence (0.9-1.0)
 * - High winner confidence → good representations → high confidence (0.9-1.0)
 * - Low burst rate → unstable → low confidence (0.5-0.7)
 * - Low winner confidence → poor representations → low confidence (0.5-0.7)
 * Final confidence clamped to [0.5, 1.0]
 */
static float compute_gradient_confidence(
    const cortical_training_bridge_t* bridge,
    const cortical_training_effects_t* effects)
{
    float confidence = 0.75f;  /* Start at moderate confidence */

    /* Dendritic: Burst rate contribution */
    if (bridge->config.enable_dendritic) {
        float burst_contrib = effects->burst_rate * bridge->config.confidence_burst_weight;
        confidence += (burst_contrib - 0.5f) * 0.3f;  /* ±0.15 range */
    }

    /* Cortical Columns: Winner confidence contribution */
    if (bridge->config.enable_columns) {
        float winner_contrib = effects->winner_confidence * bridge->config.confidence_winner_weight;
        confidence += (winner_contrib - 0.5f) * 0.3f;  /* ±0.15 range */
    }

    /* Clamp to configured bounds */
    return clamp_f(confidence,
                   bridge->config.gradient_min_confidence,
                   bridge->config.gradient_max_confidence);
}

/**
 * @brief Check if predictions are stable
 *
 * WHAT: Determines if cortical predictions have converged
 * WHY:  Trigger consolidation or reduce exploration
 * HOW:  Combines free energy, convergence rate, and burst rate
 */
static bool check_predictions_stable(
    const cortical_training_bridge_t* bridge,
    const cortical_training_effects_t* effects)
{
    /* Low free energy indicates stable predictions */
    bool low_fe = effects->free_energy < bridge->config.consolidation_fe_threshold;

    /* High convergence rate indicates predictions improving */
    bool good_convergence = effects->convergence_rate > bridge->config.consolidation_convergence_threshold;

    /* High burst rate indicates stable representations */
    bool high_bursts = effects->burst_rate > bridge->config.consolidation_burst_threshold;

    /* All conditions must be met for stable predictions */
    return low_fe && good_convergence && high_bursts;
}

/**
 * @brief Add free energy to history buffer
 *
 * WHAT: Circular buffer for FE tracking
 * WHY:  Needed for trend analysis and feedback
 * HOW:  Ring buffer with head/count tracking
 */
static void add_fe_to_history(
    cortical_training_bridge_t* bridge,
    float free_energy)
{
    if (!bridge || !bridge->fe_history) {
        return;
    }

    uint32_t idx = (bridge->history_head + bridge->history_count)
                   % CORTICAL_HISTORY_SIZE;

    bridge->fe_history[idx] = free_energy;

    if (bridge->history_count < CORTICAL_HISTORY_SIZE) {
        bridge->history_count++;
    } else {
        bridge->history_head = (bridge->history_head + 1) % CORTICAL_HISTORY_SIZE;
    }
}

/**
 * @brief Update cognitive-training bridge with cortical state
 *
 * WHAT: Syncs cortical state to cognitive-training for coordination
 * WHY:  Combine cortical and cognitive signals for unified modulation
 * HOW:  Updates epistemic uncertainty from free energy
 */
static int sync_to_cognitive(cortical_training_bridge_t* bridge) {
    if (!bridge->cognitive_training || !bridge->config.enable_cognitive_training) {
        return NIMCP_SUCCESS;
    }

    /* Map free energy to epistemic uncertainty (normalized to [0,1]) */
    float fe_normalized = bridge->cortical_effects.free_energy / 20.0f;  /* Assume max FE ~20 */
    fe_normalized = clamp_f(fe_normalized, 0.0f, 1.0f);

    /* TODO: Call cognitive_training API to set epistemic_uncertainty when available */
    (void)fe_normalized;  /* Unused for now */

    return NIMCP_SUCCESS;
}

/**
 * @brief Update training-logic bridge with cortical conditions
 *
 * WHAT: Syncs cortical state to training-logic for rule-based decisions
 * WHY:  Allows logic gates to incorporate cortical conditions
 * HOW:  Sets numeric and boolean conditions
 */
static int sync_to_logic(cortical_training_bridge_t* bridge) {
    if (!bridge->training_logic || !bridge->config.enable_training_logic) {
        return NIMCP_SUCCESS;
    }

    /* Set numeric conditions */
    training_logic_set_numeric_condition(
        bridge->training_logic,
        "free_energy",
        bridge->cortical_effects.free_energy
    );

    training_logic_set_numeric_condition(
        bridge->training_logic,
        "burst_rate",
        bridge->cortical_effects.burst_rate
    );

    /* Set boolean conditions */
    bool cortical_stable = bridge->cortical_effects.predictions_stable;
    training_logic_set_numeric_condition(
        bridge->training_logic,
        "cortical_stable",
        cortical_stable ? 1.0f : 0.0f
    );

    bool predictions_ok = bridge->cortical_effects.free_energy <
                         bridge->config.fe_high_threshold;
    training_logic_set_numeric_condition(
        bridge->training_logic,
        "predictions_ok",
        predictions_ok ? 1.0f : 0.0f
    );

    return NIMCP_SUCCESS;
}

/**
 * @brief Update training-immune system with cortical anomalies
 *
 * WHAT: Reports cortical anomalies as antigens to immune system
 * WHY:  Free energy explosion and burst collapse are threats
 * HOW:  Presents antigens when thresholds exceeded
 */
static int sync_to_immune(cortical_training_bridge_t* bridge) {
    if (!bridge->training_immune || !bridge->config.enable_training_immune) {
        return NIMCP_SUCCESS;
    }

    /* Check for free energy explosion */
    if (bridge->cortical_effects.free_energy > bridge->config.fe_explosion_threshold) {
        /* TODO: Report to training_immune when API is available */
        NIMCP_LOGGING_WARN("Free energy explosion detected: FE=%.2f",
                          bridge->cortical_effects.free_energy);
    }

    /* Check for burst rate collapse */
    if (bridge->cortical_effects.burst_rate < bridge->config.burst_collapse_threshold) {
        /* TODO: Report to training_immune when API is available */
        NIMCP_LOGGING_WARN("Burst rate collapse detected: burst_rate=%.2f",
                          bridge->cortical_effects.burst_rate);
    }

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - LIFECYCLE
 *============================================================================*/

void cortical_training_default_config(cortical_training_config_t* config) {
    if (!config) {
        return;
    }

    memset(config, 0, sizeof(cortical_training_config_t));

    /* Mode */
    config->mode = CORTICAL_TRAINING_MODE_AUTOMATIC;

    /* Enable all modules */
    config->enable_predictive_coding = true;
    config->enable_dendritic = true;
    config->enable_columns = true;

    /* Modulation strengths (0-1) */
    config->predictive_strength = 0.7f;
    config->dendritic_strength = 0.6f;
    config->columns_strength = 0.5f;

    /* LR modulation limits */
    config->lr_min_factor = CORTICAL_TRAINING_DEFAULT_LR_MIN_FACTOR;
    config->lr_max_factor = CORTICAL_TRAINING_DEFAULT_LR_MAX_FACTOR;
    config->lr_fe_scale = 0.1f;
    config->lr_burst_scale = 0.2f;

    /* Gradient confidence limits */
    config->gradient_min_confidence = CORTICAL_TRAINING_DEFAULT_GRADIENT_MIN_CONF;
    config->gradient_max_confidence = CORTICAL_TRAINING_DEFAULT_GRADIENT_MAX_CONF;
    config->confidence_burst_weight = 0.6f;
    config->confidence_winner_weight = 0.4f;

    /* Precision modulation */
    config->enable_precision_modulation = true;
    config->precision_min_weight = 0.5f;
    config->precision_max_weight = 1.5f;

    /* Consolidation thresholds */
    config->consolidation_burst_threshold = 0.7f;
    config->consolidation_fe_threshold = 3.0f;
    config->consolidation_convergence_threshold = 0.6f;

    /* Free energy thresholds */
    config->fe_explosion_threshold = CORTICAL_TRAINING_FE_EXPLOSION_THRESHOLD;
    config->fe_high_threshold = CORTICAL_TRAINING_FE_HIGH_THRESHOLD;
    config->burst_collapse_threshold = CORTICAL_TRAINING_BURST_RATE_COLLAPSE_THRESHOLD;

    /* Integration */
    config->enable_cognitive_training = true;
    config->enable_training_logic = true;
    config->enable_training_immune = true;
    config->enable_bio_async = true;

    /* Update settings */
    config->update_interval_ms = CORTICAL_TRAINING_DEFAULT_UPDATE_INTERVAL_MS;
    config->disable_auto_update = false;

    /* Safety */
    config->max_modulation_change_per_step = 0.1f;
    config->enable_emergency_override = true;
}

cortical_training_bridge_t* cortical_training_create(
    const cortical_training_config_t* config)
{
    /* Use default config if not provided */
    cortical_training_config_t default_config;
    if (!config) {
        cortical_training_default_config(&default_config);
        config = &default_config;
    }

    /* Allocate bridge */
    cortical_training_bridge_t* bridge = (cortical_training_bridge_t*)nimcp_malloc(
        sizeof(cortical_training_bridge_t)
    );
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(cortical_training_bridge_t));

    /* Store config */
    memcpy(&bridge->config, config, sizeof(cortical_training_config_t));

    /* Create mutex for thread safety */
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate FE history buffer */
    bridge->fe_history = (float*)nimcp_malloc(
        sizeof(float) * CORTICAL_HISTORY_SIZE
    );
    if (!bridge->fe_history) {
        NIMCP_LOGGING_ERROR("Failed to allocate history buffer");
        nimcp_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge);
        return NULL;
    }
    memset(bridge->fe_history, 0, sizeof(float) * CORTICAL_HISTORY_SIZE);

    /* Allocate precision weights (default max layers) */
    bridge->cortical_effects.precision_weights = (float*)nimcp_malloc(
        sizeof(float) * CORTICAL_TRAINING_MAX_LAYERS
    );
    if (!bridge->cortical_effects.precision_weights) {
        NIMCP_LOGGING_ERROR("Failed to allocate precision weights");
        nimcp_free(bridge->fe_history);
        nimcp_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge);
        return NULL;
    }
    bridge->cortical_effects.num_layers = CORTICAL_TRAINING_MAX_LAYERS;
    for (uint32_t i = 0; i < CORTICAL_TRAINING_MAX_LAYERS; i++) {
        bridge->cortical_effects.precision_weights[i] = 1.0f;  /* Default: equal precision */
    }

    /* Initialize default modulation factors to 1.0 (no modulation) */
    bridge->cortical_effects.lr_factor = 1.0f;
    bridge->cortical_effects.gradient_confidence = 0.75f;

    NIMCP_LOGGING_INFO("Created Cortical-Training bridge");

    return bridge;
}

void cortical_training_destroy(cortical_training_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    /* Disconnect bio-async if connected */
    if (bridge->base.bio_async_enabled) {
        cortical_training_disconnect_bio_async(bridge);
    }

    /* Free precision weights */
    if (bridge->cortical_effects.precision_weights) {
        nimcp_free(bridge->cortical_effects.precision_weights);
    }

    /* Free FE history */
    if (bridge->fe_history) {
        nimcp_free(bridge->fe_history);
    }

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_mutex_destroy(bridge->base.mutex);
    }

    /* Free bridge */
    nimcp_free(bridge);

    NIMCP_LOGGING_INFO("Destroyed Cortical-Training bridge");
}

int cortical_training_start(cortical_training_bridge_t* bridge) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Connect to bio-async if enabled */
    if (bridge->config.enable_bio_async && !bridge->base.bio_async_enabled) {
        int result = cortical_training_connect_bio_async(bridge);
        if (result != NIMCP_SUCCESS) {
            NIMCP_LOGGING_WARN("Bio-async connection failed, continuing without it");
        }
    }

    /* Extract initial cortical state */
    if (!bridge->config.disable_auto_update) {
        extract_cortical_state(bridge);
    }

    bridge->running = true;
    bridge->last_update_ms = get_time_ms();

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Started Cortical-Training bridge");

    return NIMCP_SUCCESS;
}

int cortical_training_stop(cortical_training_bridge_t* bridge) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->running = false;

    /* Disconnect bio-async */
    if (bridge->base.bio_async_enabled) {
        cortical_training_disconnect_bio_async(bridge);
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Stopped Cortical-Training bridge");

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - CORTICAL MODULE CONNECTIONS
 *============================================================================*/

int cortical_training_connect_predictive_coding(
    cortical_training_bridge_t* bridge,
    predictive_coding_context_t* predictive_coding)
{
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->predictive_coding = predictive_coding;
    if (predictive_coding) {
        NIMCP_LOGGING_INFO("Connected predictive coding to Cortical-Training bridge");
        bridge->stats.predictive_coding_connected = true;
    } else {
        bridge->stats.predictive_coding_connected = false;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int cortical_training_connect_dendritic(
    cortical_training_bridge_t* bridge,
    dendritic_compartment_t* dendritic)
{
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->dendritic = dendritic;
    if (dendritic) {
        NIMCP_LOGGING_INFO("Connected dendritic to Cortical-Training bridge");
        bridge->stats.dendritic_connected = true;
    } else {
        bridge->stats.dendritic_connected = false;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int cortical_training_connect_columns(
    cortical_training_bridge_t* bridge,
    hypercolumn_t* columns)
{
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->columns = columns;
    if (columns) {
        NIMCP_LOGGING_INFO("Connected cortical columns to Cortical-Training bridge");
        bridge->stats.columns_connected = true;
    } else {
        bridge->stats.columns_connected = false;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - TRAINING COMPONENT CONNECTIONS
 *============================================================================*/

int cortical_training_connect_cognitive_training(
    cortical_training_bridge_t* bridge,
    cognitive_training_bridge_t* cognitive_training)
{
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->cognitive_training = cognitive_training;
    if (cognitive_training) {
        NIMCP_LOGGING_INFO("Connected cognitive-training to Cortical-Training bridge");
        bridge->stats.cognitive_training_connected = true;
    } else {
        bridge->stats.cognitive_training_connected = false;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int cortical_training_connect_training_logic(
    cortical_training_bridge_t* bridge,
    training_logic_bridge_t* training_logic)
{
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->training_logic = training_logic;
    if (training_logic) {
        NIMCP_LOGGING_INFO("Connected training-logic to Cortical-Training bridge");
        bridge->stats.training_logic_connected = true;
    } else {
        bridge->stats.training_logic_connected = false;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int cortical_training_connect_training_immune(
    cortical_training_bridge_t* bridge,
    training_immune_system_t* training_immune)
{
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->training_immune = training_immune;
    if (training_immune) {
        NIMCP_LOGGING_INFO("Connected training-immune to Cortical-Training bridge");
        bridge->stats.training_immune_connected = true;
    } else {
        bridge->stats.training_immune_connected = false;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int cortical_training_connect_training_plasticity(
    cortical_training_bridge_t* bridge,
    training_plasticity_bridge_t* training_plasticity)
{
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->training_plasticity = training_plasticity;
    if (training_plasticity) {
        NIMCP_LOGGING_INFO("Connected training-plasticity to Cortical-Training bridge");
        bridge->stats.training_plasticity_connected = true;
    } else {
        bridge->stats.training_plasticity_connected = false;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - CORTICAL → TRAINING
 *============================================================================*/

int cortical_training_update(cortical_training_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }
    (void)delta_ms;  /* Currently unused, for future time-based updates */

    nimcp_mutex_lock(bridge->base.mutex);

    /* Extract current cortical state */
    int result = extract_cortical_state(bridge);
    if (result != NIMCP_SUCCESS) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return result;
    }

    /* Compute modulation factors */
    float lr_factor = compute_lr_modulation(bridge, &bridge->cortical_effects);
    float grad_conf = compute_gradient_confidence(bridge, &bridge->cortical_effects);
    bool predictions_stable = check_predictions_stable(bridge, &bridge->cortical_effects);

    /* Store in cortical effects */
    bridge->cortical_effects.lr_factor = lr_factor;
    bridge->cortical_effects.gradient_confidence = grad_conf;
    bridge->cortical_effects.predictions_stable = predictions_stable;
    bridge->cortical_effects.should_consolidate = predictions_stable;

    /* Add FE to history */
    add_fe_to_history(bridge, bridge->cortical_effects.free_energy);

    /* Sync to other bridges */
    sync_to_cognitive(bridge);
    sync_to_logic(bridge);
    sync_to_immune(bridge);

    /* Update stats */
    bridge->stats.total_update_calls++;
    bridge->stats.avg_free_energy =
        (bridge->stats.avg_free_energy * 0.9f) + (bridge->cortical_effects.free_energy * 0.1f);
    bridge->stats.avg_burst_rate =
        (bridge->stats.avg_burst_rate * 0.9f) + (bridge->cortical_effects.burst_rate * 0.1f);
    bridge->stats.avg_winner_confidence =
        (bridge->stats.avg_winner_confidence * 0.9f) + (bridge->cortical_effects.winner_confidence * 0.1f);
    bridge->stats.avg_prediction_error =
        (bridge->stats.avg_prediction_error * 0.9f) + (bridge->cortical_effects.prediction_error_mag * 0.1f);
    bridge->last_update_ms = get_time_ms();

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int cortical_training_get_effects(
    const cortical_training_bridge_t* bridge,
    cortical_training_effects_t* effects)
{
    if (!bridge || !effects) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    memcpy(effects, &bridge->cortical_effects, sizeof(cortical_training_effects_t));

    return NIMCP_SUCCESS;
}

float cortical_training_get_modulated_lr(
    const cortical_training_bridge_t* bridge,
    float base_lr)
{
    if (!bridge) {
        return base_lr;
    }

    return base_lr * bridge->cortical_effects.lr_factor;
}

float cortical_training_get_gradient_confidence(
    const cortical_training_bridge_t* bridge)
{
    if (!bridge) {
        return 0.75f;  /* Default moderate confidence */
    }

    return bridge->cortical_effects.gradient_confidence;
}

bool cortical_training_are_predictions_stable(
    const cortical_training_bridge_t* bridge)
{
    if (!bridge) {
        return false;
    }

    return bridge->cortical_effects.predictions_stable;
}

int cortical_training_get_precision_weights(
    const cortical_training_bridge_t* bridge,
    float* weights,
    uint32_t num_layers)
{
    if (!bridge || !weights) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (num_layers == 0) {
        return NIMCP_SUCCESS;  /* Nothing to copy */
    }

    nimcp_mutex_lock(bridge->base.mutex);

    uint32_t copy_count = (num_layers < bridge->cortical_effects.num_layers)
                         ? num_layers
                         : bridge->cortical_effects.num_layers;

    if (bridge->cortical_effects.precision_weights) {
        memcpy(weights, bridge->cortical_effects.precision_weights,
               copy_count * sizeof(float));
    } else {
        /* No precision weights, use default 1.0 */
        for (uint32_t i = 0; i < num_layers; i++) {
            weights[i] = 1.0f;
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - TRAINING → CORTICAL
 *============================================================================*/

int cortical_training_update_metrics(
    cortical_training_bridge_t* bridge,
    float loss,
    float grad_norm,
    float lr,
    uint64_t step)
{
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update training effects */
    bridge->training_effects.loss_current = loss;
    bridge->training_effects.gradient_norm = grad_norm;
    (void)lr;   /* Not currently stored in training_effects */
    (void)step; /* Step tracking handled separately if needed */

    /* Compute loss delta (requires history) */
    if (bridge->history_count > 0) {
        uint32_t prev_idx = (bridge->history_head + bridge->history_count - 1)
                           % CORTICAL_HISTORY_SIZE;
        float prev_fe = bridge->fe_history[prev_idx];
        bridge->training_effects.loss_delta = bridge->cortical_effects.free_energy - prev_fe;
    }

    /* Update stats */
    bridge->stats.total_modulations++;
    bridge->stats.current_training_step = step;

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

int cortical_training_signal_event(
    cortical_training_bridge_t* bridge,
    cortical_training_feedback_t event,
    float magnitude)
{
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (event < 0 || event >= CORTICAL_TRAINING_FEEDBACK_COUNT) {
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update stats */
    bridge->stats.total_feedback_events++;
    bridge->stats.feedback_by_type[event]++;

    /* Process event */
    switch (event) {
        case CORTICAL_TRAINING_FEEDBACK_STRENGTHEN_PREDICTIONS:
            /* Loss improved → strengthen predictions (reduce FE) */
            /* TODO: Call predictive_coding API when available */
            NIMCP_LOGGING_DEBUG("Strengthening predictions (magnitude=%.2f)", magnitude);
            break;

        case CORTICAL_TRAINING_FEEDBACK_INCREASE_PE_GAIN:
            /* Loss plateaued → increase prediction error gain */
            /* TODO: Call predictive_coding API when available */
            NIMCP_LOGGING_DEBUG("Increasing PE gain (magnitude=%.2f)", magnitude);
            break;

        case CORTICAL_TRAINING_FEEDBACK_RESET_PRECISION:
            /* Divergence → reset precision weights */
            if (bridge->cortical_effects.precision_weights) {
                for (uint32_t i = 0; i < bridge->cortical_effects.num_layers; i++) {
                    bridge->cortical_effects.precision_weights[i] = 1.0f;
                }
            }
            NIMCP_LOGGING_INFO("Reset precision weights due to divergence");
            break;

        case CORTICAL_TRAINING_FEEDBACK_CONSOLIDATE:
            /* Good convergence → consolidate representations */
            bridge->cortical_effects.should_consolidate = true;
            bridge->stats.consolidations_triggered++;
            NIMCP_LOGGING_INFO("Consolidation triggered (magnitude=%.2f)", magnitude);
            break;

        default:
            break;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - BIO-ASYNC
 *============================================================================*/

int cortical_training_connect_bio_async(cortical_training_bridge_t* bridge) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (bridge->base.bio_async_enabled) {
        return NIMCP_SUCCESS;  /* Already connected */
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_CORTICAL_TRAINING,
        .module_name = "cortical_training_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        bridge->stats.bio_async_connected = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
        return NIMCP_SUCCESS;
    }

    NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    return NIMCP_ERROR_OPERATION_FAILED;
}

int cortical_training_disconnect_bio_async(cortical_training_bridge_t* bridge) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (!bridge->base.bio_async_enabled) {
        return NIMCP_SUCCESS;  /* Already disconnected */
    }

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    bridge->stats.bio_async_connected = false;
    NIMCP_LOGGING_INFO("Disconnected from bio-async router");

    return NIMCP_SUCCESS;
}

bool cortical_training_is_bio_async_connected(
    const cortical_training_bridge_t* bridge)
{
    if (!bridge) {
        return false;
    }

    return bridge->base.bio_async_enabled;
}

/*=============================================================================
 * PUBLIC API - STATISTICS
 *============================================================================*/

int cortical_training_get_stats(
    const cortical_training_bridge_t* bridge,
    cortical_training_stats_t* stats)
{
    if (!bridge || !stats) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    memcpy(stats, &bridge->stats, sizeof(cortical_training_stats_t));

    return NIMCP_SUCCESS;
}

int cortical_training_reset_stats(cortical_training_bridge_t* bridge) {
    if (!bridge) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Preserve connection status */
    bool predictive_connected = bridge->stats.predictive_coding_connected;
    bool dendritic_connected = bridge->stats.dendritic_connected;
    bool columns_connected = bridge->stats.columns_connected;
    bool cognitive_connected = bridge->stats.cognitive_training_connected;
    bool logic_connected = bridge->stats.training_logic_connected;
    bool immune_connected = bridge->stats.training_immune_connected;
    bool plasticity_connected = bridge->stats.training_plasticity_connected;
    bool bio_connected = bridge->stats.bio_async_connected;
    cortical_training_mode_t mode = bridge->stats.current_mode;

    memset(&bridge->stats, 0, sizeof(cortical_training_stats_t));

    /* Restore connection status */
    bridge->stats.predictive_coding_connected = predictive_connected;
    bridge->stats.dendritic_connected = dendritic_connected;
    bridge->stats.columns_connected = columns_connected;
    bridge->stats.cognitive_training_connected = cognitive_connected;
    bridge->stats.training_logic_connected = logic_connected;
    bridge->stats.training_immune_connected = immune_connected;
    bridge->stats.training_plasticity_connected = plasticity_connected;
    bridge->stats.bio_async_connected = bio_connected;
    bridge->stats.current_mode = mode;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Reset Cortical-Training statistics");

    return NIMCP_SUCCESS;
}

/*=============================================================================
 * PUBLIC API - UTILITIES
 *============================================================================*/

const char* cortical_training_modulation_to_string(
    cortical_training_modulation_t modulation)
{
    switch (modulation) {
        case CORTICAL_TRAINING_MODULATION_LR:
            return "LEARNING_RATE";
        case CORTICAL_TRAINING_MODULATION_GRADIENT_CONFIDENCE:
            return "GRADIENT_CONFIDENCE";
        case CORTICAL_TRAINING_MODULATION_PRECISION:
            return "PRECISION";
        case CORTICAL_TRAINING_MODULATION_CONSOLIDATION:
            return "CONSOLIDATION";
        default:
            return "UNKNOWN";
    }
}

const char* cortical_training_feedback_to_string(
    cortical_training_feedback_t event)
{
    switch (event) {
        case CORTICAL_TRAINING_FEEDBACK_STRENGTHEN_PREDICTIONS:
            return "STRENGTHEN_PREDICTIONS";
        case CORTICAL_TRAINING_FEEDBACK_INCREASE_PE_GAIN:
            return "INCREASE_PE_GAIN";
        case CORTICAL_TRAINING_FEEDBACK_RESET_PRECISION:
            return "RESET_PRECISION";
        case CORTICAL_TRAINING_FEEDBACK_CONSOLIDATE:
            return "CONSOLIDATE";
        default:
            return "UNKNOWN";
    }
}

const char* cortical_training_mode_to_string(
    cortical_training_mode_t mode)
{
    switch (mode) {
        case CORTICAL_TRAINING_MODE_DISABLED:
            return "disabled";
        case CORTICAL_TRAINING_MODE_MONITOR_ONLY:
            return "monitor_only";
        case CORTICAL_TRAINING_MODE_ADVISORY:
            return "advisory";
        case CORTICAL_TRAINING_MODE_AUTOMATIC:
            return "automatic";
        case CORTICAL_TRAINING_MODE_COORDINATED:
            return "coordinated";
        default:
            return "unknown";
    }
}

void cortical_training_dump_state(const cortical_training_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    NIMCP_LOGGING_INFO("=== Cortical-Training Bridge State ===");
    NIMCP_LOGGING_INFO("Mode: %s",
                       cortical_training_mode_to_string(bridge->config.mode));
    NIMCP_LOGGING_INFO("Running: %s", bridge->running ? "true" : "false");

    NIMCP_LOGGING_INFO("Cortical Effects:");
    NIMCP_LOGGING_INFO("  free_energy=%.3f prediction_error=%.3f convergence_rate=%.3f",
                       bridge->cortical_effects.free_energy,
                       bridge->cortical_effects.prediction_error_mag,
                       bridge->cortical_effects.convergence_rate);
    NIMCP_LOGGING_INFO("  burst_rate=%.3f bac_success=%.3f calcium_spikes=%.3f",
                       bridge->cortical_effects.burst_rate,
                       bridge->cortical_effects.bac_success_rate,
                       bridge->cortical_effects.calcium_spikes);
    NIMCP_LOGGING_INFO("  winner_confidence=%.3f entropy=%.3f inhibition=%.3f",
                       bridge->cortical_effects.winner_confidence,
                       bridge->cortical_effects.population_entropy,
                       bridge->cortical_effects.inhibition_strength);

    NIMCP_LOGGING_INFO("Modulation:");
    NIMCP_LOGGING_INFO("  lr_factor=%.3f gradient_confidence=%.3f predictions_stable=%s",
                       bridge->cortical_effects.lr_factor,
                       bridge->cortical_effects.gradient_confidence,
                       bridge->cortical_effects.predictions_stable ? "yes" : "no");

    NIMCP_LOGGING_INFO("Training Effects:");
    NIMCP_LOGGING_INFO("  loss=%.6f gradient_norm=%.6f",
                       bridge->training_effects.loss_current,
                       bridge->training_effects.gradient_norm);

    NIMCP_LOGGING_INFO("Statistics:");
    NIMCP_LOGGING_INFO("  update_calls=%lu modulations=%lu consolidations=%lu",
                       bridge->stats.total_update_calls,
                       bridge->stats.total_modulations,
                       bridge->stats.consolidations_triggered);
    NIMCP_LOGGING_INFO("  avg_free_energy=%.3f avg_burst_rate=%.3f",
                       bridge->stats.avg_free_energy,
                       bridge->stats.avg_burst_rate);

    NIMCP_LOGGING_INFO("======================================");
}

//=============================================================================
// Test API
//=============================================================================

int cortical_training_set_effects_for_testing(
    cortical_training_bridge_t* bridge,
    const cortical_training_effects_t* effects)
{
    if (!bridge || !effects) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Preserve precision_weights pointer */
    float* existing_precision_weights = bridge->cortical_effects.precision_weights;
    uint32_t existing_num_layers = bridge->cortical_effects.num_layers;

    memcpy(&bridge->cortical_effects, effects, sizeof(cortical_training_effects_t));

    /* If new effects don't have precision_weights, restore the existing one */
    if (!effects->precision_weights) {
        bridge->cortical_effects.precision_weights = existing_precision_weights;
        bridge->cortical_effects.num_layers = existing_num_layers;
    }

    bridge->cortical_effects.valid = true;

    nimcp_mutex_unlock(bridge->base.mutex);

    return NIMCP_SUCCESS;
}
