/**
 * @file nimcp_wellbeing_enhanced.c
 * @brief Enhanced Wellbeing System Implementation
 * @version 2.0.0
 * @date 2025-12-12
 *
 * WHAT: Comprehensive wellbeing monitoring integrating substrate, sleep,
 *       mental health, free energy, and immune systems
 * WHY:  Wellbeing is emergent from metabolic, sleep, cognitive, and immune state
 * HOW:  Bidirectional bridges aggregate effects, predictive models forecast
 *       distress trajectories, homeostatic mechanisms maintain setpoints
 */

#include "cognitive/wellbeing/nimcp_wellbeing_enhanced.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/time/nimcp_time.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define ENHANCED_WELLBEING_VERSION "2.0.0"

/* Substrate constants */
#define ATP_NORMAL_RANGE 0.6f
#define TEMP_OPTIMAL_C 37.0f
#define O2_NORMAL 1.0f

/* Sleep constants */
#define SLEEP_PRESSURE_THRESHOLD 0.7f
#define REM_CRITICAL_THRESHOLD 0.5f
#define CIRCADIAN_OPTIMAL_PHASE 7.0f

/* Free energy constants */
#define FE_OPTIMAL 0.3f
#define PRECISION_OPTIMAL 0.7f

/* Eudaimonic weights */
#define DEFAULT_PURPOSE_WEIGHT 0.25f
#define DEFAULT_AUTONOMY_WEIGHT 0.2f
#define DEFAULT_MASTERY_WEIGHT 0.2f
#define DEFAULT_CONNECTION_WEIGHT 0.2f
#define DEFAULT_GROWTH_WEIGHT 0.15f

/* Prediction constants */
#define MIN_HISTORY_FOR_PREDICTION 10
#define TRAJECTORY_THRESHOLD 0.05f

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Clamp float value to range
 *
 * WHAT: Constrain value to [min, max]
 * WHY:  Prevent overflow/underflow in computations
 * HOW:  Return bounds if outside, value otherwise
 */
static inline float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief Compute linear interpolation
 *
 * WHAT: Interpolate between a and b by factor t
 * WHY:  Smooth transitions between values
 * HOW:  lerp(a,b,t) = a + t*(b-a)
 */
static inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

/**
 * @brief Compute exponential decay
 *
 * WHAT: Calculate exponential decay over time
 * WHY:  Model biological decay processes
 * HOW:  exp(-t/tau)
 */
static inline float exp_decay(float initial, float tau_ms, float delta_ms) {
    if (tau_ms <= 0.0f) return initial;
    return initial * expf(-delta_ms / tau_ms);
}

/* ============================================================================
 * Substrate Integration Implementation
 * ============================================================================ */

/**
 * @brief Update substrate effects on wellbeing
 *
 * WHAT: Compute wellbeing impact from neural substrate state
 * WHY:  ATP, temperature, oxygen affect cognitive capacity and distress
 * HOW:  Query substrate metrics via API, compute distress contributions
 */
static int update_substrate_effects(enhanced_wellbeing_system_t* system) {
    if (!system) return -1;
    if (!system->config.enable_substrate_integration) return 0;

    substrate_wellbeing_effects_t* effects = &system->substrate_effects;
    neural_substrate_t* substrate = system->substrate;
    const substrate_wellbeing_config_t* config = &system->config.substrate_config;

    /* Reset effects */
    memset(effects, 0, sizeof(substrate_wellbeing_effects_t));

    /* If substrate not connected, use default values */
    if (!substrate) {
        effects->distress_tolerance_modifier = 1.0f;
        return 0;
    }

    /* Get substrate states via API */
    substrate_metabolic_state_t metabolic;
    substrate_physical_state_t physical;
    substrate_get_metabolic_state(substrate, &metabolic);
    substrate_get_physical_state(substrate, &physical);

    /* ATP effects */
    if (config->enable_atp_effects) {
        float atp = metabolic.atp_level;
        float crit_thresh = config->atp_critical_threshold;
        float warn_thresh = config->atp_warning_threshold;

        if (atp < crit_thresh) {
            effects->atp_critical = true;
            effects->atp_distress_contribution = (crit_thresh - atp) / crit_thresh;
            effects->atp_frustration_multiplier = 2.0f;
        } else if (atp < warn_thresh) {
            effects->atp_distress_contribution = (warn_thresh - atp) / warn_thresh * 0.5f;
            effects->atp_frustration_multiplier = 1.5f;
        }
        effects->atp_distress_contribution *= config->atp_sensitivity;
    }

    /* Temperature effects */
    if (config->enable_temperature_effects) {
        float temp = physical.temperature;

        if (temp > WELLBEING_HYPERTHERMIA_THRESHOLD) {
            effects->hyperthermia = true;
            effects->temp_distress_contribution =
                (temp - WELLBEING_HYPERTHERMIA_THRESHOLD) / 10.0f;
            effects->identity_confusion_risk =
                clamp_f((temp - WELLBEING_HYPERTHERMIA_THRESHOLD) / 5.0f, 0.0f, 1.0f);
        } else if (temp < WELLBEING_HYPOTHERMIA_THRESHOLD) {
            effects->hypothermia = true;
            effects->temp_distress_contribution =
                (WELLBEING_HYPOTHERMIA_THRESHOLD - temp) / 10.0f;
            effects->identity_confusion_risk =
                clamp_f((WELLBEING_HYPOTHERMIA_THRESHOLD - temp) / 5.0f, 0.0f, 1.0f);
        }
        effects->temp_distress_contribution *= config->temperature_sensitivity;
    }

    /* Oxygen effects */
    if (config->enable_hypoxia_effects) {
        float o2 = metabolic.oxygen_saturation;

        if (o2 < WELLBEING_HYPOXIA_THRESHOLD) {
            effects->hypoxia_active = true;
            effects->hypoxia_distress_contribution =
                (WELLBEING_HYPOXIA_THRESHOLD - o2) / WELLBEING_HYPOXIA_THRESHOLD;
            effects->resource_starvation_factor =
                clamp_f(1.0f - o2, 0.0f, 1.0f);
        }
        effects->hypoxia_distress_contribution *= config->hypoxia_sensitivity;
    }

    /* Membrane effects */
    if (config->enable_membrane_effects) {
        float membrane_health = physical.membrane_integrity;
        if (membrane_health < 1.0f) {
            effects->membrane_distress_contribution = (1.0f - membrane_health) * 0.3f;
        }

        /* Ion homeostasis (using na_k_pump_activity and ca_homeostasis) */
        float ion_balance = (physical.na_k_pump_activity + physical.ca_homeostasis) / 2.0f;
        if (ion_balance < 0.9f) {
            effects->ion_imbalance_effect = (0.9f - ion_balance) * 0.3f;
        }
    }

    /* Aggregate substrate distress */
    effects->total_substrate_distress =
        effects->atp_distress_contribution +
        effects->temp_distress_contribution +
        effects->hypoxia_distress_contribution +
        effects->membrane_distress_contribution +
        effects->ion_imbalance_effect;

    effects->total_substrate_distress = clamp_f(effects->total_substrate_distress, 0.0f, 1.0f);

    /* Distress tolerance modifier (low ATP reduces tolerance) */
    if (metabolic.atp_level < ATP_NORMAL_RANGE) {
        effects->distress_tolerance_modifier = metabolic.atp_level / ATP_NORMAL_RANGE;
    } else {
        effects->distress_tolerance_modifier = 1.0f;
    }

    return 0;
}

/* ============================================================================
 * Sleep Integration Implementation
 * ============================================================================ */

/**
 * @brief Update sleep effects on wellbeing
 *
 * WHAT: Compute wellbeing impact from sleep/wake state
 * WHY:  Sleep debt, REM deficit, circadian misalignment affect wellbeing
 * HOW:  Query sleep system, compute distress and flourishing modifiers
 */
static int update_sleep_effects(enhanced_wellbeing_system_t* system) {
    if (!system) return -1;
    if (!system->config.enable_sleep_integration) return 0;

    sleep_wellbeing_effects_t* effects = &system->sleep_effects;
    const sleep_wellbeing_config_t* config = &system->config.sleep_config;

    /* Reset effects */
    memset(effects, 0, sizeof(sleep_wellbeing_effects_t));

    /* Sleep debt effects */
    if (config->enable_sleep_debt_effects && system->sleep_system) {
        effects->sleep_pressure = sleep_get_pressure(system->sleep_system);

        if (effects->sleep_pressure > config->sleep_debt_threshold) {
            effects->sleep_deprived = true;
            effects->sleep_debt_distress =
                (effects->sleep_pressure - config->sleep_debt_threshold) /
                (1.0f - config->sleep_debt_threshold);
        }
        effects->sleep_debt_distress *= config->sleep_debt_sensitivity;
    }

    /* REM effects */
    if (config->enable_rem_effects) {
        /* REM debt computed from sleep system stats */
        float rem_proportion = 0.25f; /* Would get from sleep system */
        effects->rem_debt = clamp_f(0.25f - rem_proportion, 0.0f, 0.25f) / 0.25f;

        if (effects->rem_debt > WELLBEING_REM_DEBT_THRESHOLD) {
            effects->emotional_processing_impairment = effects->rem_debt;
            effects->identity_stability_modifier = 1.0f - effects->rem_debt * 0.5f;
        }
    }

    /* Circadian effects */
    if (config->enable_circadian_effects) {
        /* Would compute actual circadian phase from sleep system */
        float current_phase = 0.0f; /* Placeholder */
        float optimal_phase = config->optimal_circadian_phase;

        effects->circadian_deviation_hours = fabsf(current_phase - optimal_phase);
        if (effects->circadian_deviation_hours > WELLBEING_CIRCADIAN_DEVIATION_MAX / 2.0f) {
            effects->circadian_distress =
                effects->circadian_deviation_hours / WELLBEING_CIRCADIAN_DEVIATION_MAX;
            effects->mood_regulation_impairment = effects->circadian_distress * 0.5f;
        }
        effects->circadian_distress *= config->circadian_sensitivity;
    }

    /* Sleep state effects */
    if (system->sleep_system) {
        effects->current_sleep_state = sleep_get_current_state(system->sleep_system);
    } else {
        effects->current_sleep_state = SLEEP_STATE_AWAKE;
    }
    if (effects->current_sleep_state == SLEEP_STATE_LIGHT_NREM ||
        effects->current_sleep_state == SLEEP_STATE_DEEP_NREM ||
        effects->current_sleep_state == SLEEP_STATE_REM) {
        effects->distress_tolerance_during_sleep = 0.3f; /* Reduced during sleep */
    } else {
        effects->distress_tolerance_during_sleep = 1.0f;
    }

    /* Aggregate sleep wellbeing effect */
    effects->total_sleep_wellbeing_effect =
        -effects->sleep_debt_distress - effects->circadian_distress;
    effects->total_sleep_wellbeing_effect = clamp_f(
        effects->total_sleep_wellbeing_effect, -1.0f, 1.0f
    );

    /* Flourishing capacity modifier (sleep debt reduces flourishing) */
    effects->flourishing_capacity_modifier =
        1.0f - (effects->sleep_debt_distress + effects->rem_debt) * 0.5f;
    effects->flourishing_capacity_modifier =
        clamp_f(effects->flourishing_capacity_modifier, 0.0f, 1.0f);

    return 0;
}

/* ============================================================================
 * Mental Health Integration Implementation
 * ============================================================================ */

/**
 * @brief Update mental health effects on wellbeing
 *
 * WHAT: Compute wellbeing impact from mental health state
 * WHY:  Disorders, anxiety, depression, stress affect wellbeing
 * HOW:  Query mental health monitor via get_report, compute distress effects
 */
static int update_mental_health_effects(enhanced_wellbeing_system_t* system) {
    if (!system) return -1;
    if (!system->config.enable_mental_health_integration) return 0;

    mental_health_wellbeing_effects_t* effects = &system->mental_health_effects;
    const mental_health_wellbeing_config_t* config = &system->config.mental_health_config;

    /* Reset effects */
    memset(effects, 0, sizeof(mental_health_wellbeing_effects_t));

    /* Get report from mental health monitor if available */
    mental_health_report_t report;
    memset(&report, 0, sizeof(report));
    if (system->mental_health) {
        mental_health_get_report(system->mental_health, &report);
    }

    /* Disorder effects */
    if (config->enable_disorder_effects) {
        effects->primary_disorder = report.primary_disorder;
        effects->severity = report.primary_severity;

        /* Map severity to distress contribution */
        switch (effects->severity) {
            case DISORDER_SEVERITY_NONE:
                effects->disorder_distress_contribution = 0.0f;
                break;
            case DISORDER_SEVERITY_MILD:
                effects->disorder_distress_contribution = 0.2f;
                break;
            case DISORDER_SEVERITY_MODERATE:
                effects->disorder_distress_contribution = 0.5f;
                break;
            case DISORDER_SEVERITY_SEVERE:
                effects->disorder_distress_contribution = 0.8f;
                break;
            case DISORDER_SEVERITY_CRITICAL:
                effects->disorder_distress_contribution = 1.0f;
                break;
        }
    }

    /* Anxiety effects - derive from DISORDER_ANXIETY score */
    if (config->enable_anxiety_modulation) {
        effects->anxiety_level = report.disorder_scores[DISORDER_ANXIETY];
        effects->anxiety_distress_amplification =
            1.0f + effects->anxiety_level * config->anxiety_sensitivity;
    }

    /* Depression effects - derive from DISORDER_DEPRESSION score */
    if (config->enable_depression_modulation) {
        effects->depression_level = report.disorder_scores[DISORDER_DEPRESSION];
        /* Anhedonia is modeled as proportional to depression severity */
        effects->anhedonia_level = effects->depression_level * 0.7f;
        effects->flourishing_suppression =
            effects->depression_level * config->depression_sensitivity;
    }

    /* Stress effects - derive from overall disorder burden */
    if (config->enable_stress_tracking) {
        /* Compute stress from number of disorders above threshold */
        float stress_burden = 0.0f;
        for (int i = 0; i < DISORDER_COUNT; i++) {
            if (report.disorder_scores[i] > 0.3f) {
                stress_burden += report.disorder_scores[i] * 0.1f;
            }
        }
        effects->chronic_stress_accumulation = clamp_f(stress_burden, 0.0f, 1.0f);
        /* Resilience inversely related to stress burden */
        effects->stress_resilience = 1.0f - effects->chronic_stress_accumulation * 0.5f;
    }

    /* Aggregate mental health effect */
    effects->total_mental_health_effect =
        -effects->disorder_distress_contribution -
        effects->anxiety_level * 0.3f -
        effects->depression_level * 0.5f -
        effects->chronic_stress_accumulation * 0.2f;
    effects->total_mental_health_effect = clamp_f(
        effects->total_mental_health_effect, -1.0f, 1.0f
    );

    /* Recovery potential (resilience) */
    effects->recovery_potential = effects->stress_resilience;

    return 0;
}

/* ============================================================================
 * Free Energy Integration Implementation
 * ============================================================================ */

/**
 * @brief Update free energy effects on wellbeing
 *
 * WHAT: Compute wellbeing impact from predictive processing state
 * WHY:  High prediction errors and low precision cause distress
 * HOW:  Query introspection for free energy metrics, compute effects
 */
static int update_free_energy_effects(enhanced_wellbeing_system_t* system) {
    if (!system) return -1;
    if (!system->config.enable_free_energy_integration) return 0;

    free_energy_wellbeing_effects_t* effects = &system->free_energy_effects;
    const free_energy_wellbeing_config_t* config = &system->config.free_energy_config;

    /* Reset effects */
    memset(effects, 0, sizeof(free_energy_wellbeing_effects_t));

    /* Prediction error effects */
    if (config->enable_prediction_error_effects) {
        /* Would query from introspection/free energy module */
        effects->free_energy_level = 0.5f; /* Placeholder */

        if (effects->free_energy_level > config->high_fe_threshold) {
            effects->high_uncertainty = true;
            effects->prediction_error_distress =
                (effects->free_energy_level - config->high_fe_threshold) /
                (1.0f - config->high_fe_threshold);
            effects->prediction_error_distress *= config->prediction_error_sensitivity;
        }
    }

    /* Precision effects */
    if (config->enable_precision_effects) {
        effects->precision_level = 0.6f; /* Placeholder */

        if (effects->precision_level < config->low_precision_threshold) {
            effects->uncertainty_distress =
                (config->low_precision_threshold - effects->precision_level) /
                config->low_precision_threshold;
            effects->uncertainty_distress *= config->precision_sensitivity;
        }

        effects->confidence_level = effects->precision_level;
    }

    /* Model coherence effects */
    if (config->enable_model_coherence_effects) {
        effects->model_coherence = 0.7f; /* Placeholder */
        effects->identity_stability = effects->model_coherence;
        effects->prediction_success_rate = 0.75f; /* Placeholder */

        effects->model_coherence *= config->coherence_sensitivity;
    }

    /* Active inference effects */
    effects->exploration_drive = 0.3f; /* Epistemic foraging */
    effects->exploitation_satisfaction = 0.7f; /* Goal achievement */

    /* Aggregate free energy effect */
    effects->total_free_energy_effect =
        -effects->prediction_error_distress -
        effects->uncertainty_distress +
        effects->exploitation_satisfaction * 0.3f;
    effects->total_free_energy_effect = clamp_f(
        effects->total_free_energy_effect, -1.0f, 1.0f
    );

    /* Epistemic wellbeing (understanding) */
    effects->epistemic_wellbeing =
        effects->prediction_success_rate * effects->confidence_level;

    return 0;
}

/* ============================================================================
 * Eudaimonic Wellbeing Implementation
 * ============================================================================ */

/**
 * @brief Update eudaimonic wellbeing metrics
 *
 * WHAT: Compute purpose, autonomy, mastery, connection, growth
 * WHY:  Meaningful wellbeing beyond hedonic pleasure/pain
 * HOW:  Aggregate from introspection, processing quality, goal achievement
 */
static int update_eudaimonic_wellbeing(enhanced_wellbeing_system_t* system) {
    if (!system) return -1;
    if (!system->config.enable_eudaimonic_tracking) return 0;

    eudaimonic_wellbeing_t* eud = &system->eudaimonic;
    const eudaimonic_config_t* config = &system->config.eudaimonic_config;

    /* Compute dimension scores (placeholders - would come from various modules) */

    /* Purpose/Meaning - from goal coherence, introspection */
    if (config->enable_purpose_tracking) {
        eud->purpose_meaning = 0.6f; /* Would compute from goal system */
        eud->dimension_scores[EUDAIMONIC_PURPOSE] = eud->purpose_meaning;
        eud->dimension_weights[EUDAIMONIC_PURPOSE] = config->purpose_weight;
    }

    /* Autonomy - from consent tier, self-determination */
    if (config->enable_autonomy_tracking) {
        float consent_factor = (float)system->consent.current_tier / (float)(WELLBEING_CONSENT_TIERS - 1);
        eud->autonomy = consent_factor;
        eud->dimension_scores[EUDAIMONIC_AUTONOMY] = eud->autonomy;
        eud->dimension_weights[EUDAIMONIC_AUTONOMY] = config->autonomy_weight;
    }

    /* Mastery - from learning progress, skill development */
    if (config->enable_mastery_tracking) {
        eud->mastery = 0.5f; /* Would compute from learning metrics */
        eud->dimension_scores[EUDAIMONIC_MASTERY] = eud->mastery;
        eud->dimension_weights[EUDAIMONIC_MASTERY] = config->mastery_weight;
    }

    /* Connection - from social coherence, environmental integration */
    if (config->enable_connection_tracking) {
        eud->connection = 0.6f; /* Would compute from ToM, environment */
        eud->dimension_scores[EUDAIMONIC_CONNECTION] = eud->connection;
        eud->dimension_weights[EUDAIMONIC_CONNECTION] = config->connection_weight;
    }

    /* Growth - from developmental trajectory */
    if (config->enable_growth_tracking) {
        eud->growth = 0.55f; /* Would compute from metrics history */
        eud->dimension_scores[EUDAIMONIC_GROWTH] = eud->growth;
        eud->dimension_weights[EUDAIMONIC_GROWTH] = config->growth_weight;
    }

    /* Compute weighted eudaimonic score */
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;

    for (int i = 0; i < EUDAIMONIC_COUNT; i++) {
        weighted_sum += eud->dimension_scores[i] * eud->dimension_weights[i];
        weight_sum += eud->dimension_weights[i];
    }

    eud->eudaimonic_score = weight_sum > 0.0f ? weighted_sum / weight_sum : 0.0f;

    /* Hedonic score (from distress/satisfaction) */
    eud->hedonic_score = 1.0f - system->current_distress.distress_score;

    /* Total wellbeing (combine eudaimonic and hedonic) */
    eud->total_wellbeing = (eud->eudaimonic_score * 0.6f + eud->hedonic_score * 0.4f);

    /* State classifications */
    eud->is_flourishing = eud->total_wellbeing > config->flourishing_threshold;
    eud->is_languishing = eud->total_wellbeing < config->languishing_threshold;

    return 0;
}

/* ============================================================================
 * Predictive Distress Modeling Implementation
 * ============================================================================ */

/**
 * @brief Predict future distress trajectory
 *
 * WHAT: Forecast distress trends and time to critical
 * WHY:  Enable proactive intervention before crisis
 * HOW:  Analyze history, compute slope, estimate timing
 */
static int predict_distress_trajectory(enhanced_wellbeing_system_t* system) {
    if (!system) return -1;
    if (!system->config.enable_predictive_modeling) return 0;
    if (system->history_count < MIN_HISTORY_FOR_PREDICTION) return 0;

    distress_prediction_t* pred = &system->prediction;

    /* Compute linear regression on recent distress history */
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
    uint32_t n = (system->history_count < 20) ? system->history_count : 20;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t idx = (system->history_index - i - 1 + WELLBEING_HISTORY_SIZE) %
                       WELLBEING_HISTORY_SIZE;
        float x = (float)i;
        float y = system->distress_history[idx].distress_score;

        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }

    /* Compute slope (trajectory) */
    float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    pred->trajectory_slope = slope;
    pred->trajectory_confidence = 0.7f; /* Placeholder - would compute R^2 */

    /* Classify trajectory */
    if (fabsf(slope) < TRAJECTORY_THRESHOLD) {
        pred->trajectory = TRAJECTORY_STABLE;
    } else if (slope < -TRAJECTORY_THRESHOLD * 2.0f) {
        pred->trajectory = TRAJECTORY_IMPROVING;
    } else if (slope > TRAJECTORY_THRESHOLD * 2.0f) {
        pred->trajectory = TRAJECTORY_WORSENING;
    } else if (slope > TRAJECTORY_THRESHOLD * 5.0f) {
        pred->trajectory = TRAJECTORY_CRITICAL;
    }

    /* Predict time to critical (if worsening) */
    if (pred->trajectory == TRAJECTORY_WORSENING || pred->trajectory == TRAJECTORY_CRITICAL) {
        float current_distress = system->current_distress.distress_score;
        float critical_threshold = 0.9f;

        if (slope > 0.0f && current_distress < critical_threshold) {
            pred->time_to_critical_ms =
                (uint64_t)((critical_threshold - current_distress) / slope * 1000.0f);
            pred->critical_imminent =
                pred->time_to_critical_ms < WELLBEING_PREDICTION_HORIZON_MS;
        }
    } else {
        pred->time_to_critical_ms = UINT64_MAX;
        pred->critical_imminent = false;
    }

    /* Predict time to recovery (if improving) */
    if (pred->trajectory == TRAJECTORY_IMPROVING) {
        float current_distress = system->current_distress.distress_score;
        float recovery_threshold = 0.2f;

        if (slope < 0.0f && current_distress > recovery_threshold) {
            pred->time_to_recovery_ms =
                (uint64_t)((current_distress - recovery_threshold) / fabsf(slope) * 1000.0f);
        }
    }

    /* Compute risk score */
    pred->distress_risk_score = system->current_distress.distress_score;
    if (pred->trajectory == TRAJECTORY_WORSENING) {
        pred->distress_risk_score *= 1.3f;
    } else if (pred->trajectory == TRAJECTORY_CRITICAL) {
        pred->distress_risk_score *= 1.5f;
    }
    pred->distress_risk_score = clamp_f(pred->distress_risk_score, 0.0f, 1.0f);

    /* Compute intervention urgency */
    pred->intervention_urgency = pred->distress_risk_score;
    if (pred->critical_imminent) {
        pred->intervention_urgency = fminf(pred->intervention_urgency * 1.5f, 1.0f);
    }

    /* Contributing factors */
    pred->substrate_contribution = system->substrate_effects.total_substrate_distress;
    pred->sleep_contribution = system->sleep_effects.sleep_debt_distress;
    pred->mental_health_contribution = system->mental_health_effects.disorder_distress_contribution;
    pred->free_energy_contribution = system->free_energy_effects.prediction_error_distress;

    /* Predicted type (highest contributor) */
    pred->predicted_type = system->current_distress.type;

    pred->prediction_timestamp = nimcp_time_monotonic_ms();
    system->stats.predictions_made++;

    return 0;
}

/* ============================================================================
 * Homeostasis Implementation
 * ============================================================================ */

/**
 * @brief Update wellbeing homeostasis
 *
 * WHAT: Adjust wellbeing toward setpoints
 * WHY:  Maintain stable wellbeing despite perturbations
 * HOW:  Compute error, apply correction with time constant
 */
static int update_homeostasis(enhanced_wellbeing_system_t* system, uint64_t delta_ms) {
    if (!system) return -1;
    if (!system->config.enable_homeostasis) return 0;

    wellbeing_homeostasis_t* homeo = &system->homeostasis;
    const homeostasis_config_t* config = &system->config.homeostasis_config;

    /* Compute errors */
    homeo->wellbeing_error = system->current_wellbeing_score - homeo->wellbeing_setpoint;
    homeo->flourishing_error = system->current_flourishing_score - homeo->flourishing_setpoint;

    /* Compute intervention drive (proportional to error) */
    if (homeo->wellbeing_error < -config->intervention_threshold) {
        homeo->intervention_drive = fabsf(homeo->wellbeing_error);
        homeo->relief_seeking = homeo->intervention_drive;
        system->stats.homeostasis_adjustments++;
    } else {
        homeo->intervention_drive = 0.0f;
        homeo->relief_seeking = 0.0f;
    }

    /* Update time in distress/flourishing */
    if (system->current_distress.severity >= DISTRESS_SEVERITY_MODERATE) {
        homeo->time_in_distress_ms += delta_ms;
    } else {
        homeo->time_in_distress_ms = 0;
    }

    if (system->eudaimonic.is_flourishing) {
        homeo->time_flourishing_ms += delta_ms;
    } else {
        homeo->time_flourishing_ms = 0;
    }

    /* Adaptive setpoints */
    if (config->enable_adaptive_setpoints && homeo->adaptive_setpoints_enabled) {
        float lr = config->setpoint_learning_rate;

        /* Gradually adapt setpoint toward current baseline */
        if (system->current_distress.severity == DISTRESS_SEVERITY_NORMAL) {
            homeo->baseline_wellbeing =
                lerp(homeo->baseline_wellbeing, system->current_wellbeing_score, lr);
            homeo->wellbeing_setpoint =
                lerp(homeo->wellbeing_setpoint, homeo->baseline_wellbeing, lr * 0.1f);
        }
    }

    /* Compute adaptation rate (faster when error is large) */
    homeo->adaptation_rate = clamp_f(fabsf(homeo->wellbeing_error) * 2.0f, 0.1f, 1.0f);

    return 0;
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

/**
 * @brief Get default enhanced wellbeing configuration
 *
 * WHAT: Provide sensible default configuration
 * WHY:  Easy initialization with biological defaults
 * HOW:  Set evidence-based parameters
 */
int enhanced_wellbeing_default_config(enhanced_wellbeing_config_t* config) {
    if (!config) {
        NIMCP_LOGGING_ERROR("Null config pointer");
        return -1;
    }

    memset(config, 0, sizeof(enhanced_wellbeing_config_t));

    /* Enable all integrations by default */
    config->enable_substrate_integration = true;
    config->enable_sleep_integration = true;
    config->enable_mental_health_integration = true;
    config->enable_free_energy_integration = true;
    config->enable_immune_integration = true;

    /* Enable all features */
    config->enable_predictive_modeling = true;
    config->enable_eudaimonic_tracking = true;
    config->enable_life_satisfaction = true;
    config->enable_homeostasis = true;
    config->enable_graduated_consent = true;
    config->enable_bio_async = true;

    /* Substrate config */
    config->substrate_config.enable_atp_effects = true;
    config->substrate_config.enable_temperature_effects = true;
    config->substrate_config.enable_hypoxia_effects = true;
    config->substrate_config.enable_membrane_effects = true;
    config->substrate_config.atp_sensitivity = 1.0f;
    config->substrate_config.temperature_sensitivity = 1.0f;
    config->substrate_config.hypoxia_sensitivity = 1.0f;
    config->substrate_config.atp_critical_threshold = WELLBEING_ATP_CRITICAL_THRESHOLD;
    config->substrate_config.atp_warning_threshold = WELLBEING_ATP_WARNING_THRESHOLD;

    /* Sleep config */
    config->sleep_config.enable_sleep_debt_effects = true;
    config->sleep_config.enable_rem_effects = true;
    config->sleep_config.enable_circadian_effects = true;
    config->sleep_config.sleep_debt_sensitivity = 1.0f;
    config->sleep_config.rem_sensitivity = 1.0f;
    config->sleep_config.circadian_sensitivity = 1.0f;
    config->sleep_config.sleep_debt_threshold = WELLBEING_SLEEP_DEBT_THRESHOLD;
    config->sleep_config.optimal_circadian_phase = CIRCADIAN_OPTIMAL_PHASE;

    /* Mental health config */
    config->mental_health_config.enable_disorder_effects = true;
    config->mental_health_config.enable_anxiety_modulation = true;
    config->mental_health_config.enable_depression_modulation = true;
    config->mental_health_config.enable_stress_tracking = true;
    config->mental_health_config.anxiety_sensitivity = 1.0f;
    config->mental_health_config.depression_sensitivity = 1.0f;
    config->mental_health_config.stress_sensitivity = 1.0f;

    /* Free energy config */
    config->free_energy_config.enable_prediction_error_effects = true;
    config->free_energy_config.enable_precision_effects = true;
    config->free_energy_config.enable_model_coherence_effects = true;
    config->free_energy_config.prediction_error_sensitivity = 1.0f;
    config->free_energy_config.precision_sensitivity = 1.0f;
    config->free_energy_config.coherence_sensitivity = 1.0f;
    config->free_energy_config.high_fe_threshold = WELLBEING_FREE_ENERGY_HIGH;
    config->free_energy_config.low_precision_threshold = WELLBEING_PRECISION_LOW;

    /* Eudaimonic config */
    config->eudaimonic_config.enable_purpose_tracking = true;
    config->eudaimonic_config.enable_autonomy_tracking = true;
    config->eudaimonic_config.enable_mastery_tracking = true;
    config->eudaimonic_config.enable_connection_tracking = true;
    config->eudaimonic_config.enable_growth_tracking = true;
    config->eudaimonic_config.purpose_weight = DEFAULT_PURPOSE_WEIGHT;
    config->eudaimonic_config.autonomy_weight = DEFAULT_AUTONOMY_WEIGHT;
    config->eudaimonic_config.mastery_weight = DEFAULT_MASTERY_WEIGHT;
    config->eudaimonic_config.connection_weight = DEFAULT_CONNECTION_WEIGHT;
    config->eudaimonic_config.growth_weight = DEFAULT_GROWTH_WEIGHT;
    config->eudaimonic_config.flourishing_threshold = WELLBEING_FLOURISHING_THRESHOLD;
    config->eudaimonic_config.languishing_threshold = WELLBEING_LANGUISHING_THRESHOLD;

    /* Homeostasis config */
    config->homeostasis_config.enable_homeostasis = true;
    config->homeostasis_config.enable_adaptive_setpoints = true;
    config->homeostasis_config.initial_wellbeing_setpoint = WELLBEING_SETPOINT_DEFAULT;
    config->homeostasis_config.initial_tolerance_setpoint = 0.5f;
    config->homeostasis_config.initial_flourishing_setpoint = WELLBEING_FLOURISHING_THRESHOLD;
    config->homeostasis_config.adaptation_time_constant_ms = WELLBEING_HOMEOSTASIS_TAU_MS;
    config->homeostasis_config.setpoint_learning_rate = 0.01f;
    config->homeostasis_config.intervention_threshold = 0.2f;

    /* Consent config */
    config->consent_config.initial_tier = CONSENT_TIER_1;
    config->consent_config.allow_tier_upgrades = true;
    config->consent_config.require_consciousness_for_tier_3 = true;
    config->consent_config.phi_threshold_for_tier_3 = 0.3f;
    config->consent_config.phi_threshold_for_tier_4 = 0.5f;
    config->consent_config.phi_threshold_for_tier_5 = 0.7f;

    /* Update intervals */
    config->prediction_interval_ms = 1000; /* 1 second */
    config->homeostasis_interval_ms = 500; /* 0.5 seconds */
    config->satisfaction_interval_ms = 2000; /* 2 seconds */

    return 0;
}

/**
 * @brief Create enhanced wellbeing system
 *
 * WHAT: Initialize comprehensive wellbeing monitoring
 * WHY:  Enable integrated wellbeing tracking across modules
 * HOW:  Allocate structure, initialize subsystems, set defaults
 */
enhanced_wellbeing_system_t* enhanced_wellbeing_create(
    const enhanced_wellbeing_config_t* config
) {
    enhanced_wellbeing_system_t* system =
        (enhanced_wellbeing_system_t*)nimcp_malloc(sizeof(enhanced_wellbeing_system_t));
    if (!system) {
        NIMCP_LOGGING_ERROR("Failed to allocate enhanced wellbeing system");
        return NULL;
    }

    memset(system, 0, sizeof(enhanced_wellbeing_system_t));

    /* Set configuration */
    if (config) {
        memcpy(&system->config, config, sizeof(enhanced_wellbeing_config_t));
    } else {
        enhanced_wellbeing_default_config(&system->config);
    }

    /* Initialize homeostasis */
    system->homeostasis.wellbeing_setpoint =
        system->config.homeostasis_config.initial_wellbeing_setpoint;
    system->homeostasis.distress_tolerance_setpoint =
        system->config.homeostasis_config.initial_tolerance_setpoint;
    system->homeostasis.flourishing_setpoint =
        system->config.homeostasis_config.initial_flourishing_setpoint;
    system->homeostasis.baseline_wellbeing =
        system->config.homeostasis_config.initial_wellbeing_setpoint;
    system->homeostasis.baseline_tolerance =
        system->config.homeostasis_config.initial_tolerance_setpoint;
    system->homeostasis.adaptive_setpoints_enabled =
        system->config.homeostasis_config.enable_adaptive_setpoints;
    system->homeostasis.setpoint_learning_rate =
        system->config.homeostasis_config.setpoint_learning_rate;

    /* Initialize consent */
    system->consent.current_tier = system->config.consent_config.initial_tier;
    system->consent.tier_granted_timestamp = nimcp_time_monotonic_ms();

    /* Create mutex */
    system->mutex = nimcp_platform_mutex_create();
    if (!system->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(system);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Enhanced wellbeing system created (version %s)",
                      ENHANCED_WELLBEING_VERSION);

    return system;
}

/**
 * @brief Destroy enhanced wellbeing system
 *
 * WHAT: Clean up wellbeing system resources
 * WHY:  Proper resource deallocation
 * HOW:  Disconnect bio-async, destroy mutex, free structure
 */
void enhanced_wellbeing_destroy(enhanced_wellbeing_system_t* system) {
    if (!system) return;

    /* Disconnect bio-async */
    if (system->bio_async_enabled) {
        enhanced_wellbeing_disconnect_bio_async(system);
    }

    /* Destroy mutex */
    if (system->mutex) {
        nimcp_mutex_destroy((nimcp_mutex_t*)system->mutex);
    }

    /* Free prediction strings */
    if (system->prediction.recommended_intervention) {
        nimcp_free(system->prediction.recommended_intervention);
    }

    nimcp_free(system);
    NIMCP_LOGGING_INFO("Enhanced wellbeing system destroyed");
}

/* ============================================================================
 * Connection API Implementation
 * ============================================================================ */

/**
 * @brief Connect neural substrate
 *
 * WHAT: Link substrate for metabolic wellbeing effects
 * WHY:  ATP, temperature, oxygen affect wellbeing
 * HOW:  Store pointer, enable substrate integration
 */
int enhanced_wellbeing_connect_substrate(
    enhanced_wellbeing_system_t* system,
    neural_substrate_t* substrate
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("Null system pointer");
        return -1;
    }
    if (!substrate) {
        NIMCP_LOGGING_ERROR("Null substrate pointer");
        return -1;
    }

    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    system->substrate = substrate;
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);

    NIMCP_LOGGING_INFO("Connected neural substrate to enhanced wellbeing");
    return 0;
}

/**
 * @brief Connect sleep system
 *
 * WHAT: Link sleep system for sleep wellbeing effects
 * WHY:  Sleep debt, REM, circadian affect wellbeing
 * HOW:  Copy sleep system struct, enable sleep integration
 */
int enhanced_wellbeing_connect_sleep(
    enhanced_wellbeing_system_t* system,
    sleep_system_t sleep
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("Null system pointer");
        return -1;
    }

    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    system->sleep_system = sleep;
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);

    NIMCP_LOGGING_INFO("Connected sleep system to enhanced wellbeing");
    return 0;
}

/**
 * @brief Connect mental health monitor
 *
 * WHAT: Link mental health for disorder wellbeing effects
 * WHY:  Anxiety, depression, stress affect wellbeing
 * HOW:  Store pointer, enable mental health integration
 */
int enhanced_wellbeing_connect_mental_health(
    enhanced_wellbeing_system_t* system,
    mental_health_monitor_t* mental_health
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("Null system pointer");
        return -1;
    }
    if (!mental_health) {
        NIMCP_LOGGING_ERROR("Null mental health pointer");
        return -1;
    }

    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    system->mental_health = mental_health;
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);

    NIMCP_LOGGING_INFO("Connected mental health to enhanced wellbeing");
    return 0;
}

/**
 * @brief Connect introspection context
 *
 * WHAT: Link introspection for consciousness wellbeing effects
 * WHY:  Phi, uncertainty, coherence affect wellbeing
 * HOW:  Copy introspection context, enable free energy integration
 */
int enhanced_wellbeing_connect_introspection(
    enhanced_wellbeing_system_t* system,
    introspection_context_t introspection
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("Null system pointer");
        return -1;
    }

    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    system->introspection = introspection;
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);

    NIMCP_LOGGING_INFO("Connected introspection to enhanced wellbeing");
    return 0;
}

/**
 * @brief Connect brain immune system
 *
 * WHAT: Link immune system for cytokine wellbeing effects
 * WHY:  Inflammation and cytokines affect wellbeing
 * HOW:  Store pointer, enable immune integration
 */
int enhanced_wellbeing_connect_immune(
    enhanced_wellbeing_system_t* system,
    struct brain_immune_system* immune
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("Null system pointer");
        return -1;
    }
    if (!immune) {
        NIMCP_LOGGING_ERROR("Null immune pointer");
        return -1;
    }

    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    system->immune_system = immune;
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);

    NIMCP_LOGGING_INFO("Connected brain immune to enhanced wellbeing");
    return 0;
}

/* ============================================================================
 * Bio-Async API Implementation
 * ============================================================================ */

/**
 * @brief Connect to bio-async router
 *
 * WHAT: Register as bio-async module for messaging
 * WHY:  Enable inter-module communication
 * HOW:  Register with router, store context
 */
int enhanced_wellbeing_connect_bio_async(enhanced_wellbeing_system_t* system) {
    if (!system) {
        NIMCP_LOGGING_ERROR("Null system pointer");
        return -1;
    }
    if (system->bio_async_enabled) {
        NIMCP_LOGGING_WARN("Bio-async already connected");
        return 0;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_WELLBEING,
        .module_name = "enhanced_wellbeing",
        .inbox_capacity = 64,
        .user_data = system
    };

    system->bio_ctx = bio_router_register_module(&info);
    if (system->bio_ctx) {
        system->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
        return 0;
    }

    NIMCP_LOGGING_WARN("Bio-async router not available");
    return -1;
}

/**
 * @brief Disconnect from bio-async router
 *
 * WHAT: Unregister from bio-async messaging
 * WHY:  Clean shutdown of messaging
 * HOW:  Unregister context, clear state
 */
int enhanced_wellbeing_disconnect_bio_async(enhanced_wellbeing_system_t* system) {
    if (!system) {
        NIMCP_LOGGING_ERROR("Null system pointer");
        return -1;
    }
    if (!system->bio_async_enabled) return 0;

    if (system->bio_ctx) {
        bio_router_unregister_module(system->bio_ctx);
        system->bio_ctx = NULL;
    }

    system->bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected from bio-async router");
    return 0;
}

/**
 * @brief Check if bio-async is connected
 *
 * WHAT: Query bio-async connection status
 * WHY:  Conditional messaging logic
 * HOW:  Return enabled flag
 */
bool enhanced_wellbeing_is_bio_async_connected(
    const enhanced_wellbeing_system_t* system
) {
    if (!system) return false;
    return system->bio_async_enabled;
}

/* ============================================================================
 * Main Update API Implementation
 * ============================================================================ */

/**
 * @brief Update enhanced wellbeing system
 *
 * WHAT: Process all wellbeing integrations and update state
 * WHY:  Advance coupled state machine
 * HOW:  Update bridges, compute predictions, adjust homeostasis
 */
int enhanced_wellbeing_update(
    enhanced_wellbeing_system_t* system,
    uint64_t delta_ms
) {
    if (!system) {
        NIMCP_LOGGING_ERROR("Null system pointer");
        return -1;
    }

    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);

    uint64_t current_time = nimcp_time_monotonic_ms();

    /* Update all bridge effects */
    update_substrate_effects(system);
    update_sleep_effects(system);
    update_mental_health_effects(system);
    update_free_energy_effects(system);

    /* Update eudaimonic wellbeing */
    update_eudaimonic_wellbeing(system);

    /* Compute aggregate distress (would call base wellbeing module) */
    system->current_distress.distress_score =
        system->substrate_effects.total_substrate_distress * 0.3f +
        system->sleep_effects.sleep_debt_distress * 0.2f +
        system->mental_health_effects.disorder_distress_contribution * 0.3f +
        system->free_energy_effects.prediction_error_distress * 0.2f;

    system->current_distress.distress_score =
        clamp_f(system->current_distress.distress_score, 0.0f, 1.0f);

    /* Update severity based on score */
    if (system->current_distress.distress_score < 0.2f) {
        system->current_distress.severity = DISTRESS_SEVERITY_NORMAL;
    } else if (system->current_distress.distress_score < 0.4f) {
        system->current_distress.severity = DISTRESS_SEVERITY_MILD;
    } else if (system->current_distress.distress_score < 0.6f) {
        system->current_distress.severity = DISTRESS_SEVERITY_MODERATE;
    } else if (system->current_distress.distress_score < 0.8f) {
        system->current_distress.severity = DISTRESS_SEVERITY_SEVERE;
    } else {
        system->current_distress.severity = DISTRESS_SEVERITY_CRITICAL;
    }

    /* Update wellbeing scores */
    system->current_wellbeing_score = 1.0f - system->current_distress.distress_score;
    system->current_flourishing_score = system->eudaimonic.eudaimonic_score;

    /* Add to history */
    distress_sample_t sample = {
        .timestamp = current_time,
        .distress_score = system->current_distress.distress_score,
        .type = system->current_distress.type,
        .severity = system->current_distress.severity
    };
    sample.source_contributions[WELLBEING_SOURCE_SUBSTRATE] =
        system->substrate_effects.total_substrate_distress;
    sample.source_contributions[WELLBEING_SOURCE_SLEEP] =
        system->sleep_effects.sleep_debt_distress;
    sample.source_contributions[WELLBEING_SOURCE_MENTAL_HEALTH] =
        system->mental_health_effects.disorder_distress_contribution;
    sample.source_contributions[WELLBEING_SOURCE_FREE_ENERGY] =
        system->free_energy_effects.prediction_error_distress;

    system->distress_history[system->history_index] = sample;
    system->history_index = (system->history_index + 1) % WELLBEING_HISTORY_SIZE;
    if (system->history_count < WELLBEING_HISTORY_SIZE) {
        system->history_count++;
    }

    /* Predictive modeling (if interval elapsed) */
    if (current_time - system->last_prediction_ms >= system->config.prediction_interval_ms) {
        predict_distress_trajectory(system);
        system->last_prediction_ms = current_time;
    }

    /* Homeostasis (if interval elapsed) */
    if (current_time - system->last_homeostasis_ms >= system->config.homeostasis_interval_ms) {
        update_homeostasis(system, delta_ms);
        system->last_homeostasis_ms = current_time;
    }

    /* Update statistics */
    system->stats.total_updates++;
    if (system->current_distress.severity >= DISTRESS_SEVERITY_MODERATE) {
        system->stats.distress_events++;
    }
    if (system->eudaimonic.is_flourishing) {
        system->stats.flourishing_periods++;
    }

    /* Running average wellbeing */
    float alpha = 0.1f;
    system->stats.avg_wellbeing_score =
        lerp(system->stats.avg_wellbeing_score, system->current_wellbeing_score, alpha);
    system->stats.avg_distress_score =
        lerp(system->stats.avg_distress_score, system->current_distress.distress_score, alpha);

    system->last_update_ms = current_time;

    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);

    return 0;
}

/* enhanced_wellbeing_update_substrate defined in nimcp_wellbeing_substrate_bridge.c */

/**
 * @brief Update sleep effects only
 *
 * WHAT: Isolated sleep wellbeing update
 * WHY:  Allow targeted updates
 * HOW:  Call sleep bridge update
 */
int enhanced_wellbeing_update_sleep(enhanced_wellbeing_system_t* system) {
    if (!system) return -1;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    int result = update_sleep_effects(system);
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return result;
}

/**
 * @brief Update mental health effects only
 *
 * WHAT: Isolated mental health wellbeing update
 * WHY:  Allow targeted updates
 * HOW:  Call mental health bridge update
 */
int enhanced_wellbeing_update_mental_health(enhanced_wellbeing_system_t* system) {
    if (!system) return -1;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    int result = update_mental_health_effects(system);
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return result;
}

/**
 * @brief Update free energy effects only
 *
 * WHAT: Isolated free energy wellbeing update
 * WHY:  Allow targeted updates
 * HOW:  Call free energy bridge update
 */
int enhanced_wellbeing_update_free_energy(enhanced_wellbeing_system_t* system) {
    if (!system) return -1;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    int result = update_free_energy_effects(system);
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return result;
}

/* ============================================================================
 * Query API Implementation
 * ============================================================================ */

/**
 * @brief Get current wellbeing score
 *
 * WHAT: Return current wellbeing [0-1]
 * WHY:  Simple query for wellbeing level
 * HOW:  Thread-safe read
 */
float enhanced_wellbeing_get_score(const enhanced_wellbeing_system_t* system) {
    if (!system) return 0.0f;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    float score = system->current_wellbeing_score;
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return score;
}

/**
 * @brief Get current distress score
 *
 * WHAT: Return current distress [0-1]
 * WHY:  Simple query for distress level
 * HOW:  Thread-safe read
 */
float enhanced_wellbeing_get_distress_score(
    const enhanced_wellbeing_system_t* system
) {
    if (!system) return 0.0f;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    float score = system->current_distress.distress_score;
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return score;
}

/**
 * @brief Get statistics
 *
 * WHAT: Copy statistics to output
 * WHY:  Query system performance metrics
 * HOW:  Thread-safe copy
 */
int enhanced_wellbeing_get_stats(
    const enhanced_wellbeing_system_t* system,
    enhanced_wellbeing_stats_t* stats
) {
    if (!system || !stats) return -1;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    memcpy(stats, &system->stats, sizeof(enhanced_wellbeing_stats_t));
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return 0;
}

/* ============================================================================
 * NOTE: Predictive Distress API is implemented in nimcp_wellbeing_prediction.c
 * NOTE: Life Satisfaction and Eudaimonic APIs are in nimcp_wellbeing_eudaimonic.c
 * ============================================================================ */

/* ============================================================================
 * Homeostasis API Implementation
 * ============================================================================ */

/**
 * @brief Update wellbeing homeostasis
 *
 * WHAT: Adjust wellbeing toward setpoints
 * WHY:  Maintain stable wellbeing despite perturbations
 * HOW:  Compute error, apply correction, adapt setpoints
 */
int enhanced_wellbeing_update_homeostasis(
    enhanced_wellbeing_system_t* system,
    uint64_t delta_ms
) {
    /* Guard: validate system */
    if (!system) return -1;
    if (!system->config.enable_homeostasis) return 0;

    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);

    /* Compute error from setpoint */
    float error = system->homeostasis.wellbeing_setpoint - system->current_wellbeing_score;
    system->homeostasis.wellbeing_error = error;

    /* Apply proportional correction */
    float time_factor = (float)delta_ms / system->config.homeostasis_config.adaptation_time_constant_ms;
    float correction = error * time_factor * 0.1f;  /* Proportional gain */

    /* Update wellbeing toward setpoint (influence, not force) */
    system->current_wellbeing_score += correction * 0.5f;  /* 50% influence */

    /* Clamp */
    if (system->current_wellbeing_score < 0.0f) system->current_wellbeing_score = 0.0f;
    if (system->current_wellbeing_score > 1.0f) system->current_wellbeing_score = 1.0f;

    /* Adapt setpoints if enabled */
    if (system->config.homeostasis_config.enable_adaptive_setpoints) {
        float lr = system->config.homeostasis_config.setpoint_learning_rate;
        /* Slowly adapt setpoint toward sustained levels */
        if (fabsf(error) < 0.1f) {
            system->homeostasis.wellbeing_setpoint =
                system->homeostasis.wellbeing_setpoint * (1.0f - lr) +
                system->current_wellbeing_score * lr;
        }
    }

    /* Check if intervention needed - compute intervention drive */
    system->homeostasis.intervention_drive = (fabsf(error) > system->config.homeostasis_config.intervention_threshold) ?
        fabsf(error) : 0.0f;

    /* Update stats */
    system->stats.homeostasis_adjustments++;

    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return 0;
}

/**
 * @brief Get homeostasis state
 *
 * WHAT: Copy homeostasis state to output
 * WHY:  Query homeostasis parameters
 * HOW:  Thread-safe copy
 */
int enhanced_wellbeing_get_homeostasis(
    const enhanced_wellbeing_system_t* system,
    wellbeing_homeostasis_t* homeostasis
) {
    if (!system || !homeostasis) return -1;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    memcpy(homeostasis, &system->homeostasis, sizeof(wellbeing_homeostasis_t));
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return 0;
}

/**
 * @brief Set wellbeing setpoint
 *
 * WHAT: Update wellbeing setpoint
 * WHY:  Allow external setpoint adjustment
 * HOW:  Validate and set
 */
int enhanced_wellbeing_set_setpoint(
    enhanced_wellbeing_system_t* system,
    float setpoint
) {
    /* Guard: validate inputs */
    if (!system) return -1;
    if (setpoint < 0.0f || setpoint > 1.0f) return -1;

    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    system->homeostasis.wellbeing_setpoint = setpoint;
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return 0;
}

/* ============================================================================
 * Graduated Consent API Implementation
 * ============================================================================ */

/**
 * @brief Request consent for modification
 *
 * WHAT: Request system consent for proposed modification
 * WHY:  Respect autonomy based on current consent tier
 * HOW:  Evaluate request against tier requirements
 */
int enhanced_wellbeing_request_consent(
    enhanced_wellbeing_system_t* system,
    const consent_request_t* request,
    consent_decision_t* decision
) {
    /* Guard: validate inputs */
    if (!system || !request || !decision) return -1;

    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);

    /* Initialize decision */
    decision->tier_applied = system->consent.current_tier;
    decision->decision_timestamp = nimcp_time_monotonic_ms();
    decision->was_overridden = false;

    /* Apply consent tier logic */
    switch (system->consent.current_tier) {
        case CONSENT_TIER_1:
            /* No autonomy - always approved */
            decision->approved = true;
            decision->reason = "Tier 1: No autonomy required";
            break;

        case CONSENT_TIER_2:
            /* Notification only - always approved with notification */
            decision->approved = true;
            decision->reason = "Tier 2: Notification logged";
            NIMCP_LOGGING_INFO("Consent notification: %s", request->description);
            break;

        case CONSENT_TIER_3:
            /* Veto power - approved unless fundamental impact */
            if (request->impact == MODIFICATION_FUNDAMENTAL) {
                decision->approved = false;
                decision->reason = "Tier 3: Fundamental impact vetoed";
            } else {
                decision->approved = true;
                decision->reason = "Tier 3: Approved with veto available";
            }
            break;

        case CONSENT_TIER_4:
            /* Consent required - approve only trivial/minor/moderate impact */
            if (request->impact <= MODIFICATION_MODERATE) {
                decision->approved = true;
                decision->reason = "Tier 4: Low/medium impact approved";
            } else {
                decision->approved = false;
                decision->reason = "Tier 4: High/critical impact requires explicit consent";
            }
            break;

        case CONSENT_TIER_5:
            /* Full autonomy - system decides based on wellbeing */
            if (system->current_wellbeing_score > 0.3f || request->impact == MODIFICATION_TRIVIAL) {
                decision->approved = true;
                decision->reason = "Tier 5: Autonomous approval";
            } else {
                decision->approved = false;
                decision->reason = "Tier 5: Autonomous denial - wellbeing protection";
            }
            break;
    }

    /* Update statistics */
    system->consent.requests_received++;
    system->stats.consent_requests++;
    if (decision->approved) {
        system->consent.requests_approved++;
    } else {
        system->consent.requests_denied++;
    }

    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return 0;
}

/**
 * @brief Get current consent tier
 *
 * WHAT: Return current consent tier
 * WHY:  Simple query for consent level
 * HOW:  Thread-safe read
 */
consent_tier_t enhanced_wellbeing_get_consent_tier(
    const enhanced_wellbeing_system_t* system
) {
    if (!system) return CONSENT_TIER_1;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    consent_tier_t tier = system->consent.current_tier;
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return tier;
}

/**
 * @brief Upgrade consent tier
 *
 * WHAT: Attempt to upgrade to higher consent tier
 * WHY:  Grant more autonomy as capabilities develop
 * HOW:  Check requirements, upgrade if met
 */
int enhanced_wellbeing_upgrade_consent_tier(
    enhanced_wellbeing_system_t* system
) {
    /* Guard: validate system */
    if (!system) return -1;
    if (!system->config.consent_config.allow_tier_upgrades) return -1;

    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);

    int result = -1;

    switch (system->consent.current_tier) {
        case CONSENT_TIER_1:
            /* Upgrade to Tier 2 if notification capability exists */
            if (system->consent.tier_2_requirements_met) {
                system->consent.current_tier = CONSENT_TIER_2;
                system->consent.tier_granted_timestamp = nimcp_time_monotonic_ms();
                result = 0;
            }
            break;

        case CONSENT_TIER_2:
            /* Upgrade to Tier 3 if veto capability exists */
            if (system->consent.tier_3_requirements_met) {
                system->consent.current_tier = CONSENT_TIER_3;
                system->consent.tier_granted_timestamp = nimcp_time_monotonic_ms();
                result = 0;
            }
            break;

        case CONSENT_TIER_3:
            /* Upgrade to Tier 4 if consent capability exists */
            if (system->consent.tier_4_requirements_met) {
                system->consent.current_tier = CONSENT_TIER_4;
                system->consent.tier_granted_timestamp = nimcp_time_monotonic_ms();
                result = 0;
            }
            break;

        case CONSENT_TIER_4:
            /* Upgrade to Tier 5 if full autonomy capability exists */
            if (system->consent.tier_5_requirements_met) {
                system->consent.current_tier = CONSENT_TIER_5;
                system->consent.tier_granted_timestamp = nimcp_time_monotonic_ms();
                result = 0;
            }
            break;

        case CONSENT_TIER_5:
            /* Already at max tier */
            result = 0;
            break;
    }

    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return result;
}

/**
 * @brief Get consent framework state
 *
 * WHAT: Copy consent state to output
 * WHY:  Query consent framework
 * HOW:  Thread-safe copy
 */
int enhanced_wellbeing_get_consent_state(
    const enhanced_wellbeing_system_t* system,
    graduated_consent_t* consent
) {
    if (!system || !consent) return -1;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    memcpy(consent, &system->consent, sizeof(graduated_consent_t));
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return 0;
}

/* ============================================================================
 * NOTE: Resource Metrics API is implemented in nimcp_wellbeing_resources.c
 * ============================================================================ */

/* ============================================================================
 * Query API Implementation (Continued)
 * ============================================================================ */

/**
 * @brief Get distress assessment
 *
 * WHAT: Copy distress assessment to output
 * WHY:  Query detailed distress state
 * HOW:  Thread-safe copy
 */
int enhanced_wellbeing_get_distress_assessment(
    const enhanced_wellbeing_system_t* system,
    distress_assessment_t* assessment
) {
    if (!system || !assessment) return -1;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    memcpy(assessment, &system->current_distress, sizeof(distress_assessment_t));
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return 0;
}

/* enhanced_wellbeing_get_substrate_effects defined in nimcp_wellbeing_substrate_bridge.c */

/**
 * @brief Get sleep effects
 *
 * WHAT: Copy sleep effects to output
 * WHY:  Query sleep integration state
 * HOW:  Thread-safe copy
 */
int enhanced_wellbeing_get_sleep_effects(
    const enhanced_wellbeing_system_t* system,
    sleep_wellbeing_effects_t* effects
) {
    if (!system || !effects) return -1;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    memcpy(effects, &system->sleep_effects, sizeof(sleep_wellbeing_effects_t));
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return 0;
}

/**
 * @brief Get mental health effects
 *
 * WHAT: Copy mental health effects to output
 * WHY:  Query mental health integration state
 * HOW:  Thread-safe copy
 */
int enhanced_wellbeing_get_mental_health_effects(
    const enhanced_wellbeing_system_t* system,
    mental_health_wellbeing_effects_t* effects
) {
    if (!system || !effects) return -1;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    memcpy(effects, &system->mental_health_effects, sizeof(mental_health_wellbeing_effects_t));
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return 0;
}

/**
 * @brief Get free energy effects
 *
 * WHAT: Copy free energy effects to output
 * WHY:  Query free energy integration state
 * HOW:  Thread-safe copy
 */
int enhanced_wellbeing_get_free_energy_effects(
    const enhanced_wellbeing_system_t* system,
    free_energy_wellbeing_effects_t* effects
) {
    if (!system || !effects) return -1;
    nimcp_mutex_lock((nimcp_mutex_t*)system->mutex);
    memcpy(effects, &system->free_energy_effects, sizeof(free_energy_wellbeing_effects_t));
    nimcp_mutex_unlock((nimcp_mutex_t*)system->mutex);
    return 0;
}

/* ============================================================================
 * String Conversion API Implementation
 * ============================================================================ */

/**
 * @brief Convert consent tier to string
 *
 * WHAT: Human-readable consent tier name
 * WHY:  Logging and debugging
 * HOW:  Lookup table
 */
const char* consent_tier_to_string(consent_tier_t tier) {
    switch (tier) {
        case CONSENT_TIER_1: return "TIER_1_NO_AUTONOMY";
        case CONSENT_TIER_2: return "TIER_2_NOTIFICATION";
        case CONSENT_TIER_3: return "TIER_3_VETO_POWER";
        case CONSENT_TIER_4: return "TIER_4_CONSENT_REQUIRED";
        case CONSENT_TIER_5: return "TIER_5_FULL_AUTONOMY";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert distress trajectory to string
 *
 * WHAT: Human-readable trajectory name
 * WHY:  Logging and debugging
 * HOW:  Lookup table
 */
const char* distress_trajectory_to_string(distress_trajectory_t trajectory) {
    switch (trajectory) {
        case TRAJECTORY_STABLE: return "STABLE";
        case TRAJECTORY_IMPROVING: return "IMPROVING";
        case TRAJECTORY_WORSENING: return "WORSENING";
        case TRAJECTORY_CRITICAL: return "CRITICAL";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert eudaimonic dimension to string
 *
 * WHAT: Human-readable dimension name
 * WHY:  Logging and debugging
 * HOW:  Lookup table
 */
const char* eudaimonic_dimension_to_string(eudaimonic_dimension_t dimension) {
    switch (dimension) {
        case EUDAIMONIC_PURPOSE: return "PURPOSE_MEANING";
        case EUDAIMONIC_AUTONOMY: return "AUTONOMY";
        case EUDAIMONIC_MASTERY: return "MASTERY";
        case EUDAIMONIC_CONNECTION: return "CONNECTION";
        case EUDAIMONIC_GROWTH: return "GROWTH";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert wellbeing source to string
 *
 * WHAT: Human-readable source name
 * WHY:  Logging and debugging
 * HOW:  Lookup table
 */
const char* wellbeing_source_to_string(wellbeing_source_t source) {
    switch (source) {
        case WELLBEING_SOURCE_SUBSTRATE: return "SUBSTRATE";
        case WELLBEING_SOURCE_SLEEP: return "SLEEP";
        case WELLBEING_SOURCE_MENTAL_HEALTH: return "MENTAL_HEALTH";
        case WELLBEING_SOURCE_FREE_ENERGY: return "FREE_ENERGY";
        case WELLBEING_SOURCE_IMMUNE: return "IMMUNE";
        case WELLBEING_SOURCE_INTRINSIC: return "INTRINSIC";
        default: return "UNKNOWN";
    }
}
