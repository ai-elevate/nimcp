/**
 * @file nimcp_neuron_substrate_bridge.c
 * @brief Neuron Model-Substrate Integration Implementation
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Bidirectional neuron-substrate coupling
 * WHY:  Neurons depend on metabolic substrate; neuron activity consumes resources
 * HOW:  Apply substrate effects to neuron dynamics; consume ATP per spike
 */

#include "core/neuron_models/nimcp_neuron_substrate_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "utils/math/nimcp_math_helpers.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(neuron_substrate_bridge)

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Compute Q10 temperature scaling factor
 *
 * WHAT: Calculate rate scaling based on temperature difference
 * WHY:  Biological processes follow Q10 rule (rate change per 10°C)
 * HOW:  Formula: factor = Q10^((T - Tref) / 10)
 *
 * Biological Basis:
 * - Most neural processes have Q10 between 2-3
 * - Firing rate increases ~2.5x per 10°C
 * - Reference: Hodgkin & Huxley (1952)
 *
 * @param current_temp Current temperature (°C)
 * @param reference_temp Reference temperature (°C)
 * @param q10 Q10 coefficient
 * @return Scaling factor
 */
static float compute_q10_factor(float current_temp, float reference_temp, float q10) {
    float temp_diff = current_temp - reference_temp;
    float exponent = temp_diff / 10.0f;
    return powf(q10, exponent);
}

/**
 * @brief Compute ATP-dependent excitability modulation
 *
 * WHAT: Calculate excitability scaling based on ATP level
 * WHY:  ATP powers Na+/K+-ATPase which maintains resting potential
 * HOW:  Sigmoid-like function: normal above threshold, drops below
 *
 * Biological Basis:
 * - Na+/K+-ATPase maintains -70mV resting potential
 * - ATP depletion → pump failure → depolarization
 * - At 20% ATP, neurons are nearly silent
 *
 * @param atp_level ATP level [0-1]
 * @return Excitability modulation [0-1.2]
 */
static float compute_atp_excitability(float atp_level) {
    if (atp_level >= ATP_EXCITABILITY_THRESHOLD) {
        /* Above threshold: normal or slightly enhanced */
        return 1.0f + 0.2f * (atp_level - ATP_EXCITABILITY_THRESHOLD) /
                             (1.0f - ATP_EXCITABILITY_THRESHOLD);
    } else if (atp_level >= ATP_CRITICAL_THRESHOLD) {
        /* Between critical and normal: linear decline */
        return (atp_level - ATP_CRITICAL_THRESHOLD) /
               (ATP_EXCITABILITY_THRESHOLD - ATP_CRITICAL_THRESHOLD);
    } else {
        /* Below critical: severe impairment */
        return 0.1f * (atp_level / ATP_CRITICAL_THRESHOLD);
    }
}

/**
 * @brief Compute oxygen-dependent transmission modulation
 *
 * WHAT: Calculate synaptic transmission scaling based on O2 level
 * WHY:  O2 required for ATP synthesis to support neurotransmitter release
 * HOW:  Linear scaling below threshold
 *
 * @param o2_level O2 saturation [0-1]
 * @return Transmission modulation [0-1]
 */
static float compute_o2_transmission(float o2_level) {
    if (o2_level >= O2_TRANSMISSION_THRESHOLD) {
        return 1.0f;
    } else if (o2_level >= O2_CRITICAL_THRESHOLD) {
        return (o2_level - O2_CRITICAL_THRESHOLD) /
               (O2_TRANSMISSION_THRESHOLD - O2_CRITICAL_THRESHOLD);
    } else {
        return 0.1f * (o2_level / O2_CRITICAL_THRESHOLD);
    }
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int neuron_substrate_default_config(neuron_substrate_config_t* config) {
    if (!config) {
        NIMCP_LOGGING_ERROR("NULL config pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* All features enabled */
    config->enable_temperature_effects = true;
    config->enable_atp_modulation = true;
    config->enable_o2_modulation = true;
    config->enable_ion_effects = true;
    config->enable_spike_consumption = true;
    config->enable_bio_async = true;

    /* Energy parameters */
    config->atp_cost_per_spike = NEURON_SPIKE_ATP_COST_DEFAULT;
    config->baseline_metabolic_cost = 0.00001f;  /* Very small baseline drain */

    /* Modulation parameters */
    config->modulation_mode = MODULATION_MODE_MULTIPLICATIVE;
    config->temperature_sensitivity = 1.0f;
    config->atp_sensitivity = 1.0f;
    config->o2_sensitivity = 1.0f;

    /* Safety limits */
    config->max_firing_rate_mod = 3.0f;  /* Max 3x firing rate increase */
    config->min_excitability = 0.01f;    /* Always maintain 1% excitability */

    return NIMCP_SUCCESS;
}

neuron_substrate_bridge_t* neuron_substrate_bridge_create(
    const neuron_substrate_config_t* config,
    neuron_model_state_t neuron_model,
    neural_substrate_t* substrate
) {
    /* Guard: require neuron model and substrate */
    if (!neuron_model) {
        NIMCP_LOGGING_ERROR("Cannot create bridge without neuron model");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "neuron_model is NULL");

        return NULL;
    }
    if (!substrate) {
        NIMCP_LOGGING_ERROR("Cannot create bridge without substrate");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;
    }

    neuron_substrate_bridge_t* bridge = (neuron_substrate_bridge_t*)
        nimcp_calloc(1, sizeof(neuron_substrate_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Bridge allocation failed");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    /* Apply configuration */
    neuron_substrate_config_t default_cfg;
    if (!config) {
        neuron_substrate_default_config(&default_cfg);
        config = &default_cfg;
    }
    bridge->config = *config;

    /* Link systems */
    bridge->neuron_model = neuron_model;
    bridge->substrate = substrate;

    /* Initialize effects to baseline (no modulation) */
    bridge->substrate_effects.q10_firing_rate_mod = 1.0f;
    bridge->substrate_effects.q10_tau_mod = 1.0f;
    bridge->substrate_effects.q10_threshold_mod = 1.0f;
    bridge->substrate_effects.atp_excitability_mod = 1.0f;
    bridge->substrate_effects.atp_threshold_shift = 0.0f;
    bridge->substrate_effects.o2_transmission_mod = 1.0f;
    bridge->substrate_effects.o2_recovery_mod = 1.0f;
    bridge->substrate_effects.ion_resting_potential_shift = 0.0f;
    bridge->substrate_effects.firing_rate_mod = 1.0f;
    bridge->substrate_effects.excitability_mod = 1.0f;
    bridge->substrate_effects.input_scaling = 1.0f;

    /* Initialize energy tracking */
    bridge->energy_tracking.atp_per_spike = config->atp_cost_per_spike;

    /* Create mutex */
    if (bridge_base_init(&bridge->base, 0, "neuron_substrate") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Mutex allocation failed");
        neuron_substrate_bridge_destroy(bridge);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "neuron_substrate_bridge_create: bridge->base is NULL");
        return NULL;
    }

    /* Try bio-async connection */
    bridge->base.bio_async_enabled = false;
    if (config->enable_bio_async) {
        neuron_substrate_connect_bio_async(bridge);
    }

    NIMCP_LOGGING_INFO("Neuron substrate bridge created");
    return bridge;
}

void neuron_substrate_bridge_destroy(neuron_substrate_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->base.bio_async_enabled) {
        neuron_substrate_disconnect_bio_async(bridge);
    }

    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Neuron substrate bridge destroyed");
}

/* ============================================================================
 * Bio-async Implementation
 * ============================================================================ */

int neuron_substrate_connect_bio_async(neuron_substrate_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (bridge->base.bio_async_enabled) return NIMCP_SUCCESS;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_NEURON_SUBSTRATE,
        .module_name = "neuron_substrate_bridge",
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
    return NIMCP_ERROR_NOT_FOUND;
}

int neuron_substrate_disconnect_bio_async(neuron_substrate_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (!bridge->base.bio_async_enabled) return NIMCP_SUCCESS;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }
    bridge->base.bio_async_enabled = false;

    return NIMCP_SUCCESS;
}

bool neuron_substrate_is_bio_async_connected(const neuron_substrate_bridge_t* bridge) {
    if (!bridge) {
        return false;
    }
    return bridge->base.bio_async_enabled;
}

/* ============================================================================
 * Substrate → Neuron Implementation
 * ============================================================================ */

int neuron_substrate_update_effects(neuron_substrate_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (!bridge->substrate) {
        NIMCP_LOGGING_ERROR("NULL substrate pointer");
        return NIMCP_ERROR_INVALID_STATE;
    }

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get substrate state */
    substrate_metabolic_state_t metabolic;
    substrate_physical_state_t physical;
    substrate_get_metabolic_state(bridge->substrate, &metabolic);
    substrate_get_physical_state(bridge->substrate, &physical);

    /* Temperature effects (Q10 scaling) */
    if (bridge->config.enable_temperature_effects) {
        float sens = bridge->config.temperature_sensitivity;
        bridge->substrate_effects.q10_firing_rate_mod = compute_q10_factor(
            physical.temperature,
            NEURON_REFERENCE_TEMPERATURE,
            NEURON_Q10_FIRING_RATE
        );
        bridge->substrate_effects.q10_tau_mod = compute_q10_factor(
            physical.temperature,
            NEURON_REFERENCE_TEMPERATURE,
            NEURON_Q10_MEMBRANE_TIME_CONSTANT
        );
        bridge->substrate_effects.q10_threshold_mod = compute_q10_factor(
            physical.temperature,
            NEURON_REFERENCE_TEMPERATURE,
            NEURON_Q10_THRESHOLD
        );

        /* Apply sensitivity */
        bridge->substrate_effects.q10_firing_rate_mod =
            1.0f + (bridge->substrate_effects.q10_firing_rate_mod - 1.0f) * sens;
        bridge->substrate_effects.q10_tau_mod =
            1.0f + (bridge->substrate_effects.q10_tau_mod - 1.0f) * sens;
        bridge->substrate_effects.q10_threshold_mod =
            1.0f + (bridge->substrate_effects.q10_threshold_mod - 1.0f) * sens;
    } else {
        bridge->substrate_effects.q10_firing_rate_mod = 1.0f;
        bridge->substrate_effects.q10_tau_mod = 1.0f;
        bridge->substrate_effects.q10_threshold_mod = 1.0f;
    }

    /* ATP-dependent effects */
    if (bridge->config.enable_atp_modulation) {
        float sens = bridge->config.atp_sensitivity;
        bridge->substrate_effects.atp_excitability_mod =
            compute_atp_excitability(metabolic.atp_level) * sens +
            (1.0f - sens);  /* Blend with baseline */

        /* Threshold shift: ATP depletion depolarizes neuron */
        float atp_deficit = 1.0f - metabolic.atp_level;
        bridge->substrate_effects.atp_threshold_shift =
            atp_deficit * 10.0f * sens;  /* Up to +10mV shift */
    } else {
        bridge->substrate_effects.atp_excitability_mod = 1.0f;
        bridge->substrate_effects.atp_threshold_shift = 0.0f;
    }

    /* Oxygen-dependent effects */
    if (bridge->config.enable_o2_modulation) {
        float sens = bridge->config.o2_sensitivity;
        bridge->substrate_effects.o2_transmission_mod =
            compute_o2_transmission(metabolic.oxygen_saturation) * sens +
            (1.0f - sens);
        bridge->substrate_effects.o2_recovery_mod =
            nimcp_clampf(metabolic.oxygen_saturation, 0.1f, 1.0f);
    } else {
        bridge->substrate_effects.o2_transmission_mod = 1.0f;
        bridge->substrate_effects.o2_recovery_mod = 1.0f;
    }

    /* Ion balance effects */
    if (bridge->config.enable_ion_effects) {
        /* Ion imbalance shifts resting potential */
        float ion_deficit = 1.0f - physical.ion_balance;
        bridge->substrate_effects.ion_resting_potential_shift =
            ion_deficit * 15.0f;  /* Up to +15mV depolarization */
    } else {
        bridge->substrate_effects.ion_resting_potential_shift = 0.0f;
    }

    /* Compute composite modulation factors */
    bridge->substrate_effects.firing_rate_mod =
        bridge->substrate_effects.q10_firing_rate_mod *
        bridge->substrate_effects.atp_excitability_mod;
    bridge->substrate_effects.firing_rate_mod = nimcp_clampf(
        bridge->substrate_effects.firing_rate_mod,
        0.0f,
        bridge->config.max_firing_rate_mod
    );

    bridge->substrate_effects.excitability_mod =
        bridge->substrate_effects.atp_excitability_mod *
        bridge->substrate_effects.o2_transmission_mod;
    bridge->substrate_effects.excitability_mod = nimcp_clampf(
        bridge->substrate_effects.excitability_mod,
        bridge->config.min_excitability,
        1.5f
    );

    /* Overall input scaling based on substrate capacity */
    float substrate_capacity = substrate_get_capacity(bridge->substrate);
    bridge->substrate_effects.input_scaling =
        substrate_capacity * bridge->substrate_effects.excitability_mod;

    /* Track statistics */
    if (bridge->substrate_effects.firing_rate_mod > bridge->stats.max_firing_rate_mod) {
        bridge->stats.max_firing_rate_mod = bridge->substrate_effects.firing_rate_mod;
    }
    if (bridge->substrate_effects.excitability_mod < bridge->stats.min_excitability_mod ||
        bridge->stats.min_excitability_mod == 0.0f) {
        bridge->stats.min_excitability_mod = bridge->substrate_effects.excitability_mod;
    }
    if (substrate_capacity < 0.3f) {
        bridge->stats.substrate_critical_events++;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int neuron_substrate_apply_modulation(neuron_substrate_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (!bridge->neuron_model) {
        NIMCP_LOGGING_ERROR("NULL neuron model pointer");
        return NIMCP_ERROR_INVALID_STATE;
    }

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Apply voltage shift from ion imbalance and ATP depletion */
    if (bridge->config.enable_ion_effects || bridge->config.enable_atp_modulation) {
        float current_v = neuron_model_get_voltage(bridge->neuron_model);
        float voltage_shift = bridge->substrate_effects.ion_resting_potential_shift +
                            bridge->substrate_effects.atp_threshold_shift;

        /* Apply shift (depolarization from substrate stress) */
        if (fabs(voltage_shift) > 0.1f) {
            neuron_model_set_voltage(bridge->neuron_model, current_v + voltage_shift * 0.1f);
        }
    }

    bridge->stats.modulation_applications++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

float neuron_substrate_get_modulated_input(
    const neuron_substrate_bridge_t* bridge,
    float base_input
) {
    if (!bridge) return base_input;
    return base_input * bridge->substrate_effects.input_scaling;
}

float neuron_substrate_get_modulated_firing_rate(
    const neuron_substrate_bridge_t* bridge,
    float base_rate
) {
    if (!bridge) return base_rate;
    return base_rate * bridge->substrate_effects.firing_rate_mod;
}

/* ============================================================================
 * Neuron → Substrate Implementation
 * ============================================================================ */

int neuron_substrate_consume_spike(neuron_substrate_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (!bridge->config.enable_spike_consumption) return NIMCP_SUCCESS;
    if (!bridge->substrate) {
        NIMCP_LOGGING_ERROR("NULL substrate pointer");
        return NIMCP_ERROR_INVALID_STATE;
    }

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Get current ATP level */
    substrate_metabolic_state_t metabolic;
    substrate_get_metabolic_state(bridge->substrate, &metabolic);

    /* Consume ATP for spike */
    float atp_cost = bridge->config.atp_cost_per_spike;
    float new_atp = metabolic.atp_level - atp_cost;
    substrate_set_atp(bridge->substrate, new_atp);

    /* Update tracking */
    bridge->energy_tracking.total_spikes++;
    bridge->energy_tracking.total_atp_consumed += atp_cost;
    bridge->stats.spikes_consumed++;
    bridge->stats.total_atp_depleted += atp_cost;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

int neuron_substrate_update_metabolic_rate(
    neuron_substrate_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (delta_ms == 0) return NIMCP_SUCCESS;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    /* Calculate firing rate (spikes per second) */
    float delta_s = delta_ms / 1000.0f;
    float firing_rate = (float)bridge->energy_tracking.total_spikes /
                       (delta_s > 0.0f ? delta_s : 1.0f);
    bridge->energy_tracking.avg_firing_rate = firing_rate;

    /* Calculate metabolic rate (ATP per second) */
    float metabolic_rate = firing_rate * bridge->config.atp_cost_per_spike +
                          bridge->config.baseline_metabolic_cost;
    bridge->energy_tracking.current_metabolic_rate = metabolic_rate;

    /* Track peak metabolic rate */
    if (metabolic_rate > bridge->energy_tracking.peak_metabolic_rate) {
        bridge->energy_tracking.peak_metabolic_rate = metabolic_rate;
    }

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Bidirectional Update Implementation
 * ============================================================================ */

int neuron_substrate_bridge_update(
    neuron_substrate_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Update substrate effects on neuron */
    int result = neuron_substrate_update_effects(bridge);
    if (result != NIMCP_SUCCESS) return result;

    /* Apply modulation to neuron model */
    result = neuron_substrate_apply_modulation(bridge);
    if (result != NIMCP_SUCCESS) return result;

    /* Update metabolic tracking */
    result = neuron_substrate_update_metabolic_rate(bridge, delta_ms);
    if (result != NIMCP_SUCCESS) return result;

    bridge->stats.total_updates++;
    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int neuron_substrate_get_effects(
    const neuron_substrate_bridge_t* bridge,
    neuron_substrate_effects_t* effects
) {
    if (!bridge || !effects) {
        NIMCP_LOGGING_ERROR("NULL pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }
    *effects = bridge->substrate_effects;
    return NIMCP_SUCCESS;
}

int neuron_substrate_get_energy_tracking(
    const neuron_substrate_bridge_t* bridge,
    neuron_energy_tracking_t* tracking
) {
    if (!bridge || !tracking) {
        NIMCP_LOGGING_ERROR("NULL pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }
    *tracking = bridge->energy_tracking;
    return NIMCP_SUCCESS;
}

bool neuron_substrate_is_modulated(const neuron_substrate_bridge_t* bridge) {
    if (!bridge) {
        return false;
    }

    /* Check if any modulation factor deviates >5% from baseline (1.0) */
    return fabs(bridge->substrate_effects.firing_rate_mod - 1.0f) > 0.05f ||
           fabs(bridge->substrate_effects.excitability_mod - 1.0f) > 0.05f ||
           fabs(bridge->substrate_effects.input_scaling - 1.0f) > 0.05f;
}

float neuron_substrate_get_excitability(const neuron_substrate_bridge_t* bridge) {
    if (!bridge) return 1.0f;
    return bridge->substrate_effects.excitability_mod;
}

int neuron_substrate_get_stats(
    const neuron_substrate_bridge_t* bridge,
    neuron_substrate_stats_t* stats
) {
    if (!bridge || !stats) {
        NIMCP_LOGGING_ERROR("NULL pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }
    *stats = bridge->stats;
    return NIMCP_SUCCESS;
}
