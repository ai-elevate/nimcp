/**
 * @file nimcp_neural_substrate.c
 * @brief Neural Substrate Module Implementation
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Low-level computational substrate modeling
 * WHY:  Physical/metabolic constraints affect neural computation
 * HOW:  Track energy, temperature, ions; compute modulation factors
 */

#include "core/neural_substrate/nimcp_neural_substrate.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Clamp value to range
 */
static inline float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief Compute Q10 temperature effect
 *
 * WHAT: Calculate temperature scaling factor
 * WHY:  Biological processes have Q10 temperature dependence
 * HOW:  Q10 formula: factor = Q10^((T - T_ref) / 10)
 */
static float compute_q10_factor(float temperature, float q10, float t_ref) {
    float delta = (temperature - t_ref) / 10.0f;
    return powf(q10, delta);
}

/**
 * @brief Update metabolic capacity
 */
static void update_metabolic_capacity(substrate_metabolic_state_t* metabolic) {
    if (!metabolic) return;

    /* Weighted average of metabolic components */
    metabolic->metabolic_capacity =
        metabolic->atp_level * 0.5f +
        metabolic->oxygen_saturation * 0.3f +
        metabolic->glucose_level * 0.2f;
}

/**
 * @brief Update physical capacity
 */
static void update_physical_capacity(substrate_physical_state_t* physical) {
    if (!physical) return;

    /* Temperature factor (optimal at 37°C) */
    float temp_factor = 1.0f;
    if (physical->temperature < SUBSTRATE_HYPOTHERMIA_THRESHOLD) {
        temp_factor = 0.5f + (physical->temperature - 28.0f) /
            (SUBSTRATE_HYPOTHERMIA_THRESHOLD - 28.0f) * 0.5f;
    } else if (physical->temperature > SUBSTRATE_HYPERTHERMIA_THRESHOLD) {
        temp_factor = 1.0f - (physical->temperature - SUBSTRATE_HYPERTHERMIA_THRESHOLD) / 5.0f;
    }
    temp_factor = clamp_f(temp_factor, 0.1f, 1.0f);

    /* Weighted average including temperature */
    physical->physical_capacity =
        physical->membrane_integrity * 0.35f +
        physical->ion_balance * 0.35f +
        temp_factor * 0.3f;
}

/**
 * @brief Compute modulation factors from state
 */
static void compute_modulation(neural_substrate_t* substrate) {
    if (!substrate) return;

    float metabolic = substrate->metabolic.metabolic_capacity;
    float physical = substrate->physical.physical_capacity;
    float temp = substrate->physical.temperature;

    /* Firing rate modulation */
    float temp_q10_firing = compute_q10_factor(temp, SUBSTRATE_Q10_FIRING,
                                                SUBSTRATE_NORMAL_TEMPERATURE);
    substrate->modulation.firing_rate_mod =
        metabolic * physical * clamp_f(temp_q10_firing, 0.5f, 1.5f);

    /* Transmission efficiency */
    float temp_q10_trans = compute_q10_factor(temp, SUBSTRATE_Q10_TRANSMISSION,
                                               SUBSTRATE_NORMAL_TEMPERATURE);
    substrate->modulation.transmission_efficiency =
        metabolic * substrate->physical.membrane_integrity *
        clamp_f(temp_q10_trans, 0.5f, 1.2f);

    /* Conduction velocity */
    float temp_q10_cond = compute_q10_factor(temp, SUBSTRATE_Q10_CONDUCTION,
                                              SUBSTRATE_NORMAL_TEMPERATURE);
    substrate->modulation.conduction_velocity =
        physical * clamp_f(temp_q10_cond, 0.6f, 1.4f);

    /* Plasticity capacity */
    float temp_q10_plast = compute_q10_factor(temp, SUBSTRATE_Q10_PLASTICITY,
                                               SUBSTRATE_NORMAL_TEMPERATURE);
    substrate->modulation.plasticity_capacity =
        metabolic * physical * clamp_f(temp_q10_plast, 0.5f, 1.3f);

    /* Overall capacity */
    substrate->modulation.overall_capacity = (metabolic + physical) / 2.0f;

    /* Clamp all values */
    substrate->modulation.firing_rate_mod =
        clamp_f(substrate->modulation.firing_rate_mod, 0.0f, 1.5f);
    substrate->modulation.transmission_efficiency =
        clamp_f(substrate->modulation.transmission_efficiency, 0.0f, 1.0f);
    substrate->modulation.conduction_velocity =
        clamp_f(substrate->modulation.conduction_velocity, 0.5f, 1.5f);
    substrate->modulation.plasticity_capacity =
        clamp_f(substrate->modulation.plasticity_capacity, 0.0f, 1.0f);
    substrate->modulation.overall_capacity =
        clamp_f(substrate->modulation.overall_capacity, 0.0f, 1.0f);
}

/**
 * @brief Update health level from state
 */
static void update_health_level(neural_substrate_t* substrate) {
    if (!substrate) return;

    float overall = substrate->modulation.overall_capacity;

    if (overall >= 0.9f) {
        substrate->health_level = SUBSTRATE_HEALTH_OPTIMAL;
    } else if (overall >= 0.7f) {
        substrate->health_level = SUBSTRATE_HEALTH_STRESSED;
    } else if (overall >= 0.5f) {
        substrate->health_level = SUBSTRATE_HEALTH_COMPROMISED;
    } else if (overall >= 0.3f) {
        substrate->health_level = SUBSTRATE_HEALTH_CRITICAL;
    } else {
        substrate->health_level = SUBSTRATE_HEALTH_FAILING;
    }
}

/**
 * @brief Check and generate alerts
 */
static void check_alerts(neural_substrate_t* substrate) {
    if (!substrate || !substrate->config.enable_alerts) return;

    substrate->alert_count = 0;

    if (substrate->metabolic.atp_level < SUBSTRATE_CRITICAL_ATP) {
        substrate->active_alerts[substrate->alert_count++] = SUBSTRATE_ALERT_LOW_ATP;
    }
    if (substrate->metabolic.oxygen_saturation < SUBSTRATE_CRITICAL_O2) {
        substrate->active_alerts[substrate->alert_count++] = SUBSTRATE_ALERT_HYPOXIA;
    }
    if (substrate->metabolic.glucose_level < SUBSTRATE_CRITICAL_GLUCOSE) {
        substrate->active_alerts[substrate->alert_count++] = SUBSTRATE_ALERT_HYPOGLYCEMIA;
    }
    if (substrate->physical.temperature > SUBSTRATE_HYPERTHERMIA_THRESHOLD) {
        substrate->active_alerts[substrate->alert_count++] = SUBSTRATE_ALERT_HYPERTHERMIA;
    }
    if (substrate->physical.temperature < SUBSTRATE_HYPOTHERMIA_THRESHOLD) {
        substrate->active_alerts[substrate->alert_count++] = SUBSTRATE_ALERT_HYPOTHERMIA;
    }
    if (substrate->physical.ion_balance < SUBSTRATE_CRITICAL_ION_IMBALANCE) {
        substrate->active_alerts[substrate->alert_count++] = SUBSTRATE_ALERT_ION_IMBALANCE;
    }
    if (substrate->physical.membrane_integrity < SUBSTRATE_CRITICAL_MEMBRANE) {
        substrate->active_alerts[substrate->alert_count++] = SUBSTRATE_ALERT_MEMBRANE_DAMAGE;
    }

    if (substrate->alert_count > 0) {
        substrate->stats.alerts_generated += substrate->alert_count;
    }
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int substrate_default_config(substrate_config_t* config) {
    if (!config) return -1;

    /* Initial values - normal healthy state */
    config->initial_atp = SUBSTRATE_NORMAL_ATP;
    config->initial_o2 = SUBSTRATE_NORMAL_O2_SAT;
    config->initial_glucose = SUBSTRATE_NORMAL_GLUCOSE;
    config->initial_temperature = SUBSTRATE_NORMAL_TEMPERATURE;
    config->initial_membrane = SUBSTRATE_NORMAL_MEMBRANE;
    config->initial_ion_balance = SUBSTRATE_NORMAL_ION_BALANCE;

    /* Recovery parameters */
    config->atp_recovery_rate = SUBSTRATE_ATP_RECOVERY_RATE;
    config->ion_recovery_rate = SUBSTRATE_ION_RECOVERY_RATE;
    config->membrane_repair_rate = SUBSTRATE_MEMBRANE_REPAIR_RATE;

    /* Energy costs */
    config->cost_per_spike = SUBSTRATE_COST_PER_SPIKE;
    config->cost_per_transmission = SUBSTRATE_COST_PER_TRANSMISSION;
    config->baseline_cost = SUBSTRATE_COST_BASELINE;

    /* All features enabled */
    config->enable_metabolic_model = true;
    config->enable_temperature_effects = true;
    config->enable_ion_dynamics = true;
    config->enable_alerts = true;

    return 0;
}

neural_substrate_t* substrate_create(const substrate_config_t* config) {
    neural_substrate_t* substrate = (neural_substrate_t*)
        nimcp_calloc(1, sizeof(neural_substrate_t));
    if (!substrate) {
        NIMCP_LOGGING_ERROR("Substrate allocation failed");
        return NULL;
    }

    /* Apply configuration */
    substrate_config_t default_cfg;
    if (!config) {
        substrate_default_config(&default_cfg);
        config = &default_cfg;
    }
    substrate->config = *config;

    /* Initialize metabolic state */
    substrate->metabolic.atp_level = config->initial_atp;
    substrate->metabolic.oxygen_saturation = config->initial_o2;
    substrate->metabolic.glucose_level = config->initial_glucose;
    substrate->metabolic.metabolic_rate = config->baseline_cost;
    substrate->metabolic.recovery_rate = config->atp_recovery_rate;
    update_metabolic_capacity(&substrate->metabolic);

    /* Initialize physical state */
    substrate->physical.temperature = config->initial_temperature;
    substrate->physical.membrane_integrity = config->initial_membrane;
    substrate->physical.ion_balance = config->initial_ion_balance;
    substrate->physical.na_k_pump_activity = 0.95f;
    substrate->physical.ca_homeostasis = 0.95f;
    update_physical_capacity(&substrate->physical);

    /* Create mutex */
    substrate->mutex = nimcp_platform_mutex_create();
    if (!substrate->mutex) {
        NIMCP_LOGGING_ERROR("Mutex allocation failed");
        substrate_destroy(substrate);
        return NULL;
    }

    /* Compute initial modulation */
    compute_modulation(substrate);
    update_health_level(substrate);

    NIMCP_LOGGING_INFO("Neural substrate created");
    return substrate;
}

void substrate_destroy(neural_substrate_t* substrate) {
    if (!substrate) return;

    if (substrate->mutex) {
        nimcp_platform_mutex_destroy(substrate->mutex);
        nimcp_free(substrate->mutex);
    }

    nimcp_free(substrate);
    NIMCP_LOGGING_INFO("Neural substrate destroyed");
}

int substrate_reset(neural_substrate_t* substrate) {
    if (!substrate) return -1;

    nimcp_platform_mutex_lock(substrate->mutex);

    /* Reset to initial values */
    substrate->metabolic.atp_level = substrate->config.initial_atp;
    substrate->metabolic.oxygen_saturation = substrate->config.initial_o2;
    substrate->metabolic.glucose_level = substrate->config.initial_glucose;
    substrate->metabolic.metabolic_rate = substrate->config.baseline_cost;
    update_metabolic_capacity(&substrate->metabolic);

    substrate->physical.temperature = substrate->config.initial_temperature;
    substrate->physical.membrane_integrity = substrate->config.initial_membrane;
    substrate->physical.ion_balance = substrate->config.initial_ion_balance;
    substrate->physical.na_k_pump_activity = 0.95f;
    substrate->physical.ca_homeostasis = 0.95f;
    update_physical_capacity(&substrate->physical);

    substrate->alert_count = 0;
    memset(&substrate->stats, 0, sizeof(substrate_stats_t));

    compute_modulation(substrate);
    update_health_level(substrate);

    nimcp_platform_mutex_unlock(substrate->mutex);
    return 0;
}

/* ============================================================================
 * Update Implementation
 * ============================================================================ */

int substrate_update(neural_substrate_t* substrate, uint64_t delta_ms) {
    if (!substrate) return -1;

    nimcp_platform_mutex_lock(substrate->mutex);

    float delta_sec = delta_ms / 1000.0f;

    /* Metabolic recovery */
    if (substrate->config.enable_metabolic_model) {
        /* Baseline consumption */
        substrate->metabolic.atp_level -=
            substrate->config.baseline_cost * delta_sec;

        /* ATP recovery (depends on O2 and glucose) */
        float recovery = substrate->config.atp_recovery_rate * delta_sec;
        recovery *= substrate->metabolic.oxygen_saturation;
        recovery *= substrate->metabolic.glucose_level;
        substrate->metabolic.atp_level += recovery;

        /* Clamp */
        substrate->metabolic.atp_level =
            clamp_f(substrate->metabolic.atp_level, 0.0f, 1.0f);

        /* Slow glucose/O2 recovery toward normal */
        substrate->metabolic.glucose_level +=
            (SUBSTRATE_NORMAL_GLUCOSE - substrate->metabolic.glucose_level) * 0.001f * delta_sec;
        substrate->metabolic.oxygen_saturation +=
            (SUBSTRATE_NORMAL_O2_SAT - substrate->metabolic.oxygen_saturation) * 0.005f * delta_sec;

        update_metabolic_capacity(&substrate->metabolic);
    }

    /* Ion dynamics */
    if (substrate->config.enable_ion_dynamics) {
        /* Ion recovery */
        float ion_recovery = substrate->config.ion_recovery_rate * delta_sec;
        ion_recovery *= substrate->metabolic.atp_level;  /* Na/K pump needs ATP */
        substrate->physical.ion_balance +=
            (SUBSTRATE_NORMAL_ION_BALANCE - substrate->physical.ion_balance) * ion_recovery;
        substrate->physical.ion_balance =
            clamp_f(substrate->physical.ion_balance, 0.0f, 1.0f);

        /* Update pump activity based on ATP */
        substrate->physical.na_k_pump_activity = substrate->metabolic.atp_level * 0.95f;
    }

    /* Membrane repair */
    float membrane_repair = substrate->config.membrane_repair_rate * delta_sec;
    substrate->physical.membrane_integrity +=
        (SUBSTRATE_NORMAL_MEMBRANE - substrate->physical.membrane_integrity) * membrane_repair;
    substrate->physical.membrane_integrity =
        clamp_f(substrate->physical.membrane_integrity, 0.0f, 1.0f);

    /* Temperature drift toward normal */
    if (substrate->config.enable_temperature_effects) {
        substrate->physical.temperature +=
            (SUBSTRATE_NORMAL_TEMPERATURE - substrate->physical.temperature) * 0.001f * delta_sec;
    }

    update_physical_capacity(&substrate->physical);

    /* Compute modulation and health */
    compute_modulation(substrate);
    update_health_level(substrate);
    check_alerts(substrate);

    /* Track critical events */
    if (substrate->health_level >= SUBSTRATE_HEALTH_CRITICAL) {
        substrate->stats.critical_events++;
    }

    substrate->stats.total_updates++;
    substrate->last_update_ms += delta_ms;

    /* Update average health */
    float health_score = substrate->modulation.overall_capacity;
    substrate->stats.avg_health_score =
        (substrate->stats.avg_health_score * (substrate->stats.total_updates - 1) + health_score) /
        substrate->stats.total_updates;

    nimcp_platform_mutex_unlock(substrate->mutex);
    return 0;
}

int substrate_record_spikes(neural_substrate_t* substrate, uint32_t neuron_count) {
    if (!substrate || !substrate->config.enable_metabolic_model) return -1;

    nimcp_platform_mutex_lock(substrate->mutex);

    float cost = substrate->config.cost_per_spike * neuron_count;
    substrate->metabolic.atp_level -= cost;
    substrate->metabolic.atp_level = clamp_f(substrate->metabolic.atp_level, 0.0f, 1.0f);

    substrate->stats.spikes_processed += neuron_count;
    substrate->stats.total_atp_consumed += cost;

    if (substrate->metabolic.metabolic_rate < cost) {
        substrate->metabolic.metabolic_rate = cost;
    }
    if (cost > substrate->stats.peak_metabolic_rate) {
        substrate->stats.peak_metabolic_rate = cost;
    }

    update_metabolic_capacity(&substrate->metabolic);
    compute_modulation(substrate);

    nimcp_platform_mutex_unlock(substrate->mutex);
    return 0;
}

int substrate_record_transmissions(neural_substrate_t* substrate, uint32_t transmission_count) {
    if (!substrate || !substrate->config.enable_metabolic_model) return -1;

    nimcp_platform_mutex_lock(substrate->mutex);

    float cost = substrate->config.cost_per_transmission * transmission_count;
    substrate->metabolic.atp_level -= cost;
    substrate->metabolic.atp_level = clamp_f(substrate->metabolic.atp_level, 0.0f, 1.0f);

    substrate->stats.transmissions_processed += transmission_count;
    substrate->stats.total_atp_consumed += cost;

    update_metabolic_capacity(&substrate->metabolic);
    compute_modulation(substrate);

    nimcp_platform_mutex_unlock(substrate->mutex);
    return 0;
}

/* ============================================================================
 * Setter Implementation
 * ============================================================================ */

int substrate_set_atp(neural_substrate_t* substrate, float atp_level) {
    if (!substrate) return -1;

    nimcp_platform_mutex_lock(substrate->mutex);
    substrate->metabolic.atp_level = clamp_f(atp_level, 0.0f, 1.0f);
    update_metabolic_capacity(&substrate->metabolic);
    compute_modulation(substrate);
    nimcp_platform_mutex_unlock(substrate->mutex);

    return 0;
}

int substrate_set_oxygen(neural_substrate_t* substrate, float o2_sat) {
    if (!substrate) return -1;

    nimcp_platform_mutex_lock(substrate->mutex);
    substrate->metabolic.oxygen_saturation = clamp_f(o2_sat, 0.0f, 1.0f);
    update_metabolic_capacity(&substrate->metabolic);
    compute_modulation(substrate);
    nimcp_platform_mutex_unlock(substrate->mutex);

    return 0;
}

int substrate_set_glucose(neural_substrate_t* substrate, float glucose) {
    if (!substrate) return -1;

    nimcp_platform_mutex_lock(substrate->mutex);
    substrate->metabolic.glucose_level = clamp_f(glucose, 0.0f, 1.0f);
    update_metabolic_capacity(&substrate->metabolic);
    compute_modulation(substrate);
    nimcp_platform_mutex_unlock(substrate->mutex);

    return 0;
}

int substrate_set_temperature(neural_substrate_t* substrate, float temperature) {
    if (!substrate) return -1;

    nimcp_platform_mutex_lock(substrate->mutex);
    substrate->physical.temperature = clamp_f(temperature, 20.0f, 45.0f);
    update_physical_capacity(&substrate->physical);
    compute_modulation(substrate);
    nimcp_platform_mutex_unlock(substrate->mutex);

    return 0;
}

int substrate_set_membrane_integrity(neural_substrate_t* substrate, float integrity) {
    if (!substrate) return -1;

    nimcp_platform_mutex_lock(substrate->mutex);
    substrate->physical.membrane_integrity = clamp_f(integrity, 0.0f, 1.0f);
    update_physical_capacity(&substrate->physical);
    compute_modulation(substrate);
    nimcp_platform_mutex_unlock(substrate->mutex);

    return 0;
}

int substrate_set_ion_balance(neural_substrate_t* substrate, float balance) {
    if (!substrate) return -1;

    nimcp_platform_mutex_lock(substrate->mutex);
    substrate->physical.ion_balance = clamp_f(balance, 0.0f, 1.0f);
    update_physical_capacity(&substrate->physical);
    compute_modulation(substrate);
    nimcp_platform_mutex_unlock(substrate->mutex);

    return 0;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int substrate_get_metabolic_state(
    const neural_substrate_t* substrate,
    substrate_metabolic_state_t* state
) {
    if (!substrate || !state) return -1;
    *state = substrate->metabolic;
    return 0;
}

int substrate_get_physical_state(
    const neural_substrate_t* substrate,
    substrate_physical_state_t* state
) {
    if (!substrate || !state) return -1;
    *state = substrate->physical;
    return 0;
}

int substrate_get_modulation(
    const neural_substrate_t* substrate,
    substrate_modulation_t* mod
) {
    if (!substrate || !mod) return -1;
    *mod = substrate->modulation;
    return 0;
}

substrate_health_level_t substrate_get_health_level(const neural_substrate_t* substrate) {
    if (!substrate) return SUBSTRATE_HEALTH_FAILING;
    return substrate->health_level;
}

float substrate_get_capacity(const neural_substrate_t* substrate) {
    if (!substrate) return 0.0f;
    return substrate->modulation.overall_capacity;
}

float substrate_get_firing_modulation(const neural_substrate_t* substrate) {
    if (!substrate) return 1.0f;
    return substrate->modulation.firing_rate_mod;
}

float substrate_get_transmission_efficiency(const neural_substrate_t* substrate) {
    if (!substrate) return 1.0f;
    return substrate->modulation.transmission_efficiency;
}

int substrate_get_alerts(
    const neural_substrate_t* substrate,
    substrate_alert_type_t* alerts,
    uint32_t* count
) {
    if (!substrate || !alerts || !count) return -1;

    *count = substrate->alert_count;
    for (uint32_t i = 0; i < substrate->alert_count && i < 8; i++) {
        alerts[i] = substrate->active_alerts[i];
    }

    return 0;
}

int substrate_get_stats(
    const neural_substrate_t* substrate,
    substrate_stats_t* stats
) {
    if (!substrate || !stats) return -1;
    *stats = substrate->stats;
    return 0;
}

/* ============================================================================
 * String Conversion Implementation
 * ============================================================================ */

const char* substrate_health_level_to_string(substrate_health_level_t level) {
    switch (level) {
        case SUBSTRATE_HEALTH_OPTIMAL:     return "OPTIMAL";
        case SUBSTRATE_HEALTH_STRESSED:    return "STRESSED";
        case SUBSTRATE_HEALTH_COMPROMISED: return "COMPROMISED";
        case SUBSTRATE_HEALTH_CRITICAL:    return "CRITICAL";
        case SUBSTRATE_HEALTH_FAILING:     return "FAILING";
        default:                           return "UNKNOWN";
    }
}

const char* substrate_alert_type_to_string(substrate_alert_type_t alert) {
    switch (alert) {
        case SUBSTRATE_ALERT_NONE:           return "NONE";
        case SUBSTRATE_ALERT_LOW_ATP:        return "LOW_ATP";
        case SUBSTRATE_ALERT_HYPOXIA:        return "HYPOXIA";
        case SUBSTRATE_ALERT_HYPOGLYCEMIA:   return "HYPOGLYCEMIA";
        case SUBSTRATE_ALERT_HYPERTHERMIA:   return "HYPERTHERMIA";
        case SUBSTRATE_ALERT_HYPOTHERMIA:    return "HYPOTHERMIA";
        case SUBSTRATE_ALERT_ION_IMBALANCE:  return "ION_IMBALANCE";
        case SUBSTRATE_ALERT_MEMBRANE_DAMAGE:return "MEMBRANE_DAMAGE";
        default:                             return "UNKNOWN";
    }
}
