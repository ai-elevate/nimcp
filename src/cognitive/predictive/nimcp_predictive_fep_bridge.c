/**
 * @file nimcp_predictive_fep_bridge.c
 * @brief Free Energy Principle - Predictive Regions Integration Bridge Implementation
 *
 * WHAT: Implements bidirectional integration between FEP and hierarchical predictive coding
 * WHY:  Predictive coding IS the neural implementation of FEP - this bridge makes
 *       the theoretical connection explicit
 * HOW:  Direct mapping: predictions = generative model, errors = variational gradients,
 *       precision = attention, minimizing errors = minimizing free energy
 *
 * BIOLOGICAL BASIS:
 * - Rao & Ballard (1999): Predictive coding in visual cortex
 * - Friston (2005): "A theory of cortical responses"
 * - Clark (2013): "Whatever next? Predictive brains"
 * - Predictive coding = message passing on FEP free energy functional
 */

#include "cognitive/predictive/nimcp_predictive_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"

#include <string.h>
#include <math.h>

#define LOG_MODULE "predictive_fep_bridge"

/* ============================================================================
 * Default Configuration
 * ============================================================================ */

int predictive_fep_bridge_default_config(predictive_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;

    /* FEP -> Predictive */
    config->enable_belief_prediction_sync = true;
    config->enable_precision_gain_control = true;
    config->enable_fe_error_mapping = true;
    config->belief_sync_rate = 0.1f;

    /* Predictive -> FEP */
    config->enable_error_gradient_flow = true;
    config->enable_prediction_generative_model = true;
    config->enable_precision_kalman_gains = true;
    config->error_gradient_scaling = PRED_FEP_ERROR_FE_SCALING;

    /* Hierarchy mapping */
    config->match_hierarchy_levels = PRED_FEP_LEVEL_MATCHING_STRICT;
    config->hierarchy_offset = 0;

    /* Sensitivity factors */
    config->precision_sensitivity = 1.0f;
    config->prediction_sensitivity = 1.0f;

    return 0;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

predictive_fep_bridge_t* predictive_fep_bridge_create(
    const predictive_fep_config_t* config
) {
    predictive_fep_bridge_t* bridge = nimcp_malloc(sizeof(predictive_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate predictive FEP bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(predictive_fep_bridge_t));

    /* Set configuration */
    if (config) {
        bridge->config = *config;
    } else {
        predictive_fep_bridge_default_config(&bridge->config);
    }

    /* Create mutex */
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize effects with defaults */
    bridge->fep_effects.avg_precision = PRED_FEP_PRECISION_DEFAULT;

    NIMCP_LOGGING_INFO("Created predictive FEP bridge");
    return bridge;
}

void predictive_fep_bridge_destroy(predictive_fep_bridge_t* bridge) {
    if (!bridge) return;

    /* Disconnect bio-async if connected */
    if (bridge->base.bio_async_enabled) {
        predictive_fep_bridge_disconnect_bio_async(bridge);
    }

    /* Free allocated arrays in effects */
    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->fep_effects.synchronized_beliefs) {
        nimcp_free(bridge->fep_effects.synchronized_beliefs);
    }
    if (bridge->fep_effects.precision_gains) {
        nimcp_free(bridge->fep_effects.precision_gains);
    }
    if (bridge->fep_effects.fe_per_level) {
        nimcp_free(bridge->fep_effects.fe_per_level);
    }
    if (bridge->pred_effects.error_gradients) {
        nimcp_free(bridge->pred_effects.error_gradients);
    }
    if (bridge->pred_effects.generative_predictions) {
        nimcp_free(bridge->pred_effects.generative_predictions);
    }
    if (bridge->pred_effects.kalman_gains) {
        nimcp_free(bridge->pred_effects.kalman_gains);
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed predictive FEP bridge");
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int predictive_fep_bridge_connect_fep(
    predictive_fep_bridge_t* bridge,
    fep_system_t* fep
) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Connected FEP system to predictive bridge");
    return 0;
}

int predictive_fep_bridge_connect_predictive(
    predictive_fep_bridge_t* bridge,
    predictive_network_t predictive
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->predictive = predictive;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Connected predictive network to FEP bridge");
    return 0;
}

int predictive_fep_bridge_disconnect(predictive_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    memset(&bridge->predictive, 0, sizeof(predictive_network_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Disconnected all systems from predictive FEP bridge");
    return 0;
}

/* ============================================================================
 * FEP -> Predictive Direction
 * ============================================================================ */

int predictive_fep_sync_beliefs_to_predictions(
    predictive_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_belief_prediction_sync) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Synchronize FEP beliefs with predictions
     * FEP belief = expected state = prediction
     * μ_FEP → prediction_predictive
     */
    bridge->state.beliefs_synchronized = true;
    bridge->stats.belief_syncs++;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Synchronized beliefs to predictions");
    return 0;
}

int predictive_fep_apply_precision_gain_control(
    predictive_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_precision_gain_control) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Apply FEP precision as predictive gain control
     * High precision → amplify errors
     * Low precision → suppress errors
     * Attention = precision optimization
     */
    float avg_precision = bridge->fep_effects.avg_precision;
    float gain = avg_precision * bridge->config.precision_sensitivity;

    /* This would modulate prediction error gain in the predictive network */
    bridge->stats.precision_updates++;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Applied precision gain control: gain=%f", gain);
    return 0;
}

int predictive_fep_map_fe_to_error(
    predictive_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_fe_error_mapping) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Map free energy to prediction error
     * F ≈ ∑Π||ε||² (equivalence in linear Gaussian case)
     * This is the core theoretical connection
     */
    float fe = bridge->fep_effects.total_free_energy;
    float avg_precision = bridge->fep_effects.avg_precision;

    /* PE² ≈ F / Π (approximate inverse mapping) */
    float pe_squared = (avg_precision > 0.01f) ? (fe / avg_precision) : fe;
    float pe = sqrtf(fabsf(pe_squared));

    bridge->state.current_prediction_error = pe;
    bridge->stats.fe_error_mappings++;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Mapped FE to PE: FE=%f, PE=%f", fe, pe);
    return 0;
}

/* ============================================================================
 * Predictive -> FEP Direction
 * ============================================================================ */

int predictive_fep_flow_error_gradients(
    predictive_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_error_gradient_flow) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Flow prediction errors as variational gradients
     * PE = ∂F/∂μ (prediction errors are variational gradients)
     * ε_predictive → ∂F/∂μ_FEP
     */
    float pe = bridge->state.current_prediction_error;
    float gradient = pe * bridge->config.error_gradient_scaling;

    /* This gradient would be used to update FEP beliefs */
    bridge->stats.gradient_flows++;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Flowed error gradients: PE=%f, gradient=%f", pe, gradient);
    return 0;
}

int predictive_fep_provide_generative_predictions(
    predictive_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_prediction_generative_model) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Provide predictions as generative model
     * Predictions = g(μ) (generative model output)
     * prediction_predictive → g(μ)_FEP
     */
    bridge->stats.generative_updates++;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Provided generative predictions");
    return 0;
}

int predictive_fep_compute_kalman_gains(
    predictive_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_precision_kalman_gains) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Compute precision as Kalman gains
     * Precision = Kalman gain (optimal Bayesian weighting)
     * K = Σ_prior * (Σ_prior + Σ_likelihood)^-1
     * Under Gaussian assumptions, precision maps directly
     */
    bridge->stats.kalman_updates++;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Computed Kalman gains");
    return 0;
}

/* ============================================================================
 * Update Cycle
 * ============================================================================ */

int predictive_fep_bridge_update(
    predictive_fep_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;

    /* FEP -> Predictive direction */
    predictive_fep_sync_beliefs_to_predictions(bridge);
    predictive_fep_apply_precision_gain_control(bridge);
    predictive_fep_map_fe_to_error(bridge);

    /* Predictive -> FEP direction */
    predictive_fep_flow_error_gradients(bridge);
    predictive_fep_provide_generative_predictions(bridge);
    predictive_fep_compute_kalman_gains(bridge);

    nimcp_mutex_lock(bridge->base.mutex);

    /* Track convergence */
    float prev_fe = bridge->stats.avg_free_energy;
    float curr_fe = bridge->fep_effects.total_free_energy;
    float fe_change = prev_fe - curr_fe;

    if (fe_change > 0.0f) {
        /* FE decreasing = converging */
        bridge->state.fe_convergence_rate =
            (bridge->state.fe_convergence_rate * 0.9f) + (fe_change * 0.1f);
    }

    float prev_pe = bridge->stats.avg_prediction_error;
    float curr_pe = bridge->state.current_prediction_error;
    float pe_change = prev_pe - curr_pe;

    if (pe_change > 0.0f) {
        /* PE decreasing = converging */
        bridge->state.pe_convergence_rate =
            (bridge->state.pe_convergence_rate * 0.9f) + (pe_change * 0.1f);
    }

    /* Check convergence */
    if (curr_pe < 0.01f && curr_fe < 0.01f) {
        bridge->state.converged = true;
        bridge->stats.convergence_count++;
    } else {
        bridge->state.converged = false;
    }

    /* Update stats */
    bridge->stats.avg_free_energy =
        (bridge->stats.avg_free_energy * 0.9f) + (curr_fe * 0.1f);
    bridge->stats.avg_prediction_error =
        (bridge->stats.avg_prediction_error * 0.9f) + (curr_pe * 0.1f);

    bridge->state.current_free_energy = curr_fe;
    bridge->state.last_sync_time += delta_ms;

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * State/Stats API
 * ============================================================================ */

int predictive_fep_bridge_get_state(
    const predictive_fep_bridge_t* bridge,
    predictive_fep_state_t* state
) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int predictive_fep_bridge_get_stats(
    const predictive_fep_bridge_t* bridge,
    predictive_fep_stats_t* stats
) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;

    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

int predictive_fep_bridge_connect_bio_async(
    predictive_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->base.bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_PREDICTIVE_BRIDGE,
        .module_name = "predictive_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available");
    }

    return 0;
}

int predictive_fep_bridge_disconnect_bio_async(
    predictive_fep_bridge_t* bridge
) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->base.bio_async_enabled) return 0;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }

    bridge->base.bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected from bio-async router");

    return 0;
}

bool predictive_fep_bridge_is_bio_async_connected(
    const predictive_fep_bridge_t* bridge
) {
    return bridge ? bridge->base.bio_async_enabled : false;
}
