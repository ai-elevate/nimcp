/**
 * @file nimcp_energy_consistency_fep_bridge.c
 * @brief Energy Consistency - FEP Bridge Implementation
 *
 * Maps logical consistency energy to FEP prediction error.
 *
 * @author NIMCP Team
 * @version 2.6.3
 */

#include "cognitive/neuro_symbolic/bridges/nimcp_energy_consistency_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "ENERGY_FEP_BRIDGE"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for energy_consistency_fep_bridge module */
static nimcp_health_agent_t* g_energy_consistency_fep_bridge_health_agent = NULL;

/**
 * @brief Set health agent for energy_consistency_fep_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void energy_consistency_fep_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_energy_consistency_fep_bridge_health_agent = agent;
}

/** @brief Send heartbeat from energy_consistency_fep_bridge module */
static inline void energy_consistency_fep_bridge_heartbeat(const char* operation, float progress) {
    if (g_energy_consistency_fep_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_energy_consistency_fep_bridge_health_agent, operation, progress);
    }
}

#include <math.h>
#include <string.h>

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

NIMCP_API energy_fep_bridge_t* energy_fep_bridge_create(void) {
    energy_fep_bridge_config_t config;
    energy_fep_bridge_get_default_config(&config);
    return energy_fep_bridge_create_with_config(&config);
}

NIMCP_API energy_fep_bridge_t* energy_fep_bridge_create_with_config(
    const energy_fep_bridge_config_t* config) {

    if (!config) {


        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config is NULL");


        return NULL;


    }

    energy_fep_bridge_t* bridge = nimcp_calloc(1, sizeof(energy_fep_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    /* Initialize base - order is (base, module_id, module_name) */
    if (bridge_base_init(&bridge->base, BIO_MODULE_ENERGY_CONSISTENCY_FEP_BRIDGE,
                         "energy_fep_bridge") != NIMCP_SUCCESS) {
        nimcp_free(bridge);
        return NULL;
    }

    memcpy(&bridge->config, config, sizeof(energy_fep_bridge_config_t));

    /* Initialize state */
    bridge->current_energy = 0.0f;
    bridge->smoothed_energy = 0.0f;
    bridge->energy_derivative = 0.0f;
    bridge->estimated_precision = bridge->config.base_precision;

    NIMCP_LOG_DEBUG("Created energy-FEP bridge");
    return bridge;
}

NIMCP_API void energy_fep_bridge_destroy(energy_fep_bridge_t* bridge) {
    if (!bridge) return;

    bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge);

    NIMCP_LOG_DEBUG("Destroyed energy-FEP bridge");
}

NIMCP_API nimcp_error_t energy_fep_bridge_reset(energy_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    bridge->current_energy = 0.0f;
    bridge->smoothed_energy = 0.0f;
    bridge->energy_derivative = 0.0f;
    bridge->estimated_precision = bridge->config.base_precision;
    bridge->total_updates = 0;
    bridge->error_signals_sent = 0;
    bridge->actions_triggered = 0;
    bridge->avg_prediction_error = 0.0f;

    memset(bridge->precision_history, 0, sizeof(bridge->precision_history));
    bridge->precision_history_idx = 0;

    memset(&bridge->effects, 0, sizeof(energy_fep_bridge_effects_t));

    return NIMCP_SUCCESS;
}

NIMCP_API nimcp_error_t energy_fep_bridge_get_default_config(
    energy_fep_bridge_config_t* config) {

    if (!config) return NIMCP_ERROR_INVALID_PARAM;

    memset(config, 0, sizeof(energy_fep_bridge_config_t));

    config->enable_modulation = true;
    config->sensitivity = 1.0f;
    config->energy_to_error_scale = 1.0f;
    config->error_threshold = 0.5f;
    config->base_precision = 1.0f;
    config->precision_learning_rate = 0.01f;
    config->enable_active_inference = true;
    config->enable_bio_async = true;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Connection Functions
 * ============================================================================ */

NIMCP_API nimcp_error_t energy_fep_bridge_connect_checker(
    energy_fep_bridge_t* bridge,
    energy_consistency_checker_t* checker) {

    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    bridge->base.system_a = checker;
    bridge->base.system_a_connected = (checker != NULL);
    bridge->base.bridge_active = (checker != NULL && bridge->base.system_b != NULL);

    return NIMCP_SUCCESS;
}

NIMCP_API nimcp_error_t energy_fep_bridge_connect_fep(
    energy_fep_bridge_t* bridge,
    void* fep) {

    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    bridge->base.system_b = fep;
    bridge->base.system_b_connected = (fep != NULL);
    bridge->base.bridge_active = (bridge->base.system_a != NULL && fep != NULL);

    return NIMCP_SUCCESS;
}

NIMCP_API nimcp_error_t energy_fep_bridge_disconnect_checker(
    energy_fep_bridge_t* bridge) {

    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    bridge->base.system_a = NULL;
    bridge->base.system_a_connected = false;
    bridge->base.bridge_active = false;

    return NIMCP_SUCCESS;
}

NIMCP_API nimcp_error_t energy_fep_bridge_disconnect_fep(
    energy_fep_bridge_t* bridge) {

    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    bridge->base.system_b = NULL;
    bridge->base.system_b_connected = false;
    bridge->base.bridge_active = false;

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Update Functions
 * ============================================================================ */

NIMCP_API nimcp_error_t energy_fep_bridge_update(
    energy_fep_bridge_t* bridge,
    float consistency_energy) {

    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    /* Compute derivative */
    bridge->energy_derivative = consistency_energy - bridge->current_energy;
    bridge->current_energy = consistency_energy;

    /* Exponential smoothing */
    float alpha = 0.1f;
    bridge->smoothed_energy = alpha * consistency_energy +
                               (1.0f - alpha) * bridge->smoothed_energy;

    /* Map to prediction error */
    bridge->effects.prediction_error = consistency_energy *
                                        bridge->config.energy_to_error_scale *
                                        bridge->config.sensitivity;

    /* Update precision estimate */
    float error_variance = fabsf(bridge->energy_derivative);
    float new_precision = 1.0f / (error_variance + 0.1f);

    bridge->estimated_precision = (1.0f - bridge->config.precision_learning_rate) *
                                   bridge->estimated_precision +
                                   bridge->config.precision_learning_rate * new_precision;

    bridge->effects.precision = bridge->estimated_precision;

    /* Store in history */
    bridge->precision_history[bridge->precision_history_idx] = bridge->estimated_precision;
    bridge->precision_history_idx = (bridge->precision_history_idx + 1) % 64;

    /* Compute belief update rate */
    bridge->effects.belief_update_rate = bridge->estimated_precision *
                                          bridge->effects.prediction_error;

    /* Compute action urgency */
    if (consistency_energy > bridge->config.error_threshold) {
        bridge->effects.action_urgency = (consistency_energy - bridge->config.error_threshold) /
                                          bridge->config.error_threshold;
        if (bridge->effects.action_urgency > 1.0f) {
            bridge->effects.action_urgency = 1.0f;
        }
        bridge->actions_triggered++;
    } else {
        bridge->effects.action_urgency = 0.0f;
    }

    /* Compute surprise */
    bridge->effects.surprise = -logf(1.0f / (1.0f + consistency_energy));

    /* Update statistics */
    bridge->total_updates++;
    bridge->avg_prediction_error = (bridge->avg_prediction_error *
                                     (bridge->total_updates - 1) +
                                     bridge->effects.prediction_error) /
                                    bridge->total_updates;

    if (bridge->effects.prediction_error > 0.01f) {
        bridge->error_signals_sent++;
    }

    return NIMCP_SUCCESS;
}

NIMCP_API nimcp_error_t energy_fep_bridge_update_from_result(
    energy_fep_bridge_t* bridge,
    const energy_consistency_result_t* result) {

    if (!bridge || !result) return NIMCP_ERROR_INVALID_PARAM;

    return energy_fep_bridge_update(bridge, result->total_energy);
}

NIMCP_API float energy_fep_bridge_get_precision_weighted_energy(
    const energy_fep_bridge_t* bridge,
    const consistency_violation_t* violation) {

    if (!bridge || !violation) return 0.0f;

    /* Weight violation energy by precision */
    return violation->energy_cost * bridge->estimated_precision;
}

/* ============================================================================
 * Effects Query
 * ============================================================================ */

NIMCP_API nimcp_error_t energy_fep_bridge_get_effects(
    const energy_fep_bridge_t* bridge,
    energy_fep_bridge_effects_t* effects) {

    if (!bridge || !effects) return NIMCP_ERROR_INVALID_PARAM;

    memcpy(effects, &bridge->effects, sizeof(energy_fep_bridge_effects_t));
    return NIMCP_SUCCESS;
}

NIMCP_API float energy_fep_bridge_get_prediction_error(
    const energy_fep_bridge_t* bridge) {

    if (!bridge) return 0.0f;
    return bridge->effects.prediction_error;
}

NIMCP_API float energy_fep_bridge_get_precision(
    const energy_fep_bridge_t* bridge) {

    if (!bridge) return 0.0f;
    return bridge->estimated_precision;
}

/* ============================================================================
 * Active Inference
 * ============================================================================ */

NIMCP_API nimcp_error_t energy_fep_bridge_get_repair_action(
    const energy_fep_bridge_t* bridge,
    const consistency_violation_t* violation,
    int* action) {

    if (!bridge || !violation || !action) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* Select action based on violation type and expected free energy reduction */
    switch (violation->type) {
        case CONSISTENCY_CONTRADICTION:
            *action = 0; /* Resolve contradiction */
            break;
        case CONSISTENCY_UNSATISFIED_RULE:
            *action = 1; /* Apply missing rule */
            break;
        case CONSISTENCY_CIRCULARITY:
            *action = 2; /* Break cycle */
            break;
        default:
            *action = 0;
            break;
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

NIMCP_API nimcp_error_t energy_fep_bridge_register_bio_async(
    energy_fep_bridge_t* bridge) {

    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    /* Bio-async registration handled by bridge_base */
    bridge->base.bio_async_enabled = true;
    return NIMCP_SUCCESS;
}

NIMCP_API nimcp_error_t energy_fep_bridge_unregister_bio_async(
    energy_fep_bridge_t* bridge) {

    if (!bridge) return NIMCP_ERROR_INVALID_PARAM;

    bridge->base.bio_async_enabled = false;
    return NIMCP_SUCCESS;
}
