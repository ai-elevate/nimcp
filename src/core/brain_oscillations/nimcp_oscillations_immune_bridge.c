/**
 * @file nimcp_oscillations_immune_bridge.c
 * @brief Brain Oscillations-Immune System Integration Implementation
 * @version 1.0.0
 * @date 2025-12-11
 *
 * WHAT: Bidirectional coupling between brain immune and oscillation systems
 * WHY:  Biological realism - cytokines alter brain waves, abnormal oscillations trigger immune
 * HOW:  Monitor cytokine/inflammation to modulate oscillations, monitor oscillations to trigger immune
 */

#include "core/brain_oscillations/nimcp_oscillations_immune_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>
#include <pthread.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Clamp value to range
 *
 * WHAT: Constrain value to [min, max]
 * WHY:  Prevent overflow/underflow
 * HOW:  Return min if below, max if above, value otherwise
 */
static inline float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief Get cytokine concentration from immune system
 *
 * WHAT: Query cytokine level from brain immune system
 * WHY:  Need current cytokine state to modulate oscillations
 * HOW:  Iterate through cytokines array, find matching type
 */
static float get_cytokine_concentration(
    const brain_immune_system_t* immune,
    brain_cytokine_type_t type
) {
    if (!immune) return 0.0f;

    /* Query immune system cytokines */
    float total_concentration = 0.0f;
    for (size_t i = 0; i < immune->cytokine_count; i++) {
        if (immune->cytokines[i].type == type && !immune->cytokines[i].delivered) {
            total_concentration += immune->cytokines[i].concentration;
        }
    }

    return clamp_f(total_concentration, 0.0f, 1.0f);
}

/**
 * @brief Get max inflammation level
 *
 * WHAT: Get highest inflammation level in system
 * WHY:  Max inflammation determines oscillation impact
 * HOW:  Query immune system inflammation sites
 */
static brain_inflammation_level_t get_max_inflammation_level(
    const brain_immune_system_t* immune
) {
    if (!immune) return INFLAMMATION_NONE;

    brain_inflammation_level_t max_level = INFLAMMATION_NONE;
    for (size_t i = 0; i < immune->inflammation_count; i++) {
        if (immune->inflammation_sites[i].level > max_level) {
            max_level = immune->inflammation_sites[i].level;
        }
    }

    return max_level;
}

/**
 * @brief Get inflammation duration
 *
 * WHAT: Calculate how long inflammation has persisted
 * WHY:  Chronic inflammation has stronger oscillation effects
 * HOW:  Find oldest active inflammation site
 */
static float get_inflammation_duration_sec(
    const brain_immune_system_t* immune,
    uint64_t current_time
) {
    if (!immune || immune->inflammation_count == 0) return 0.0f;

    uint64_t oldest_start = current_time;
    for (size_t i = 0; i < immune->inflammation_count; i++) {
        if (immune->inflammation_sites[i].start_time < oldest_start) {
            oldest_start = immune->inflammation_sites[i].start_time;
        }
    }

    return (current_time - oldest_start) / 1000.0f;
}

/**
 * @brief Create epitope from oscillation signature
 *
 * WHAT: Generate antigen signature from abnormal oscillation pattern
 * WHY:  Need threat signature for immune system
 * HOW:  Pack power ratios and coherence into byte array
 */
static size_t create_oscillation_epitope(
    const brain_wave_power_t* power,
    float coherence,
    float synchrony,
    uint8_t* epitope_out,
    size_t max_len
) {
    if (!power || !epitope_out || max_len < 32) return 0;

    /* Pack oscillation signature into epitope */
    uint32_t* data = (uint32_t*)epitope_out;
    size_t idx = 0;

    /* Encode power ratios as fixed-point */
    data[idx++] = (uint32_t)(power->delta_power * 1000.0f);
    data[idx++] = (uint32_t)(power->theta_power * 1000.0f);
    data[idx++] = (uint32_t)(power->alpha_power * 1000.0f);
    data[idx++] = (uint32_t)(power->beta_power * 1000.0f);
    data[idx++] = (uint32_t)(power->gamma_power * 1000.0f);
    data[idx++] = (uint32_t)(coherence * 1000.0f);
    data[idx++] = (uint32_t)(synchrony * 1000.0f);
    data[idx++] = (uint32_t)(power->dominant_freq * 100.0f);

    return idx * sizeof(uint32_t);
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int oscillations_immune_default_config(oscillations_immune_config_t* config) {
    if (!config) return -1;

    /* All features enabled by default */
    config->enable_cytokine_oscillation_modulation = true;
    config->enable_inflammation_power_shift = true;
    config->enable_oscillation_immune_trigger = true;
    config->enable_abnormality_surveillance = true;
    config->enable_il10_restoration = true;

    /* Biologically-based default sensitivities */
    config->cytokine_sensitivity = 1.0f;
    config->inflammation_sensitivity = 1.0f;
    config->abnormality_sensitivity = 1.0f;

    /* Evidence-based thresholds */
    config->excessive_delta_threshold = INFLAMMATION_DELTA_THRESHOLD;
    config->suppressed_gamma_threshold = INFLAMMATION_GAMMA_THRESHOLD;
    config->persistence_threshold = ABNORMALITY_PERSISTENCE_THRESHOLD;

    return 0;
}

oscillations_immune_bridge_t* oscillations_immune_bridge_create(
    const oscillations_immune_config_t* config,
    brain_oscillation_analyzer_t* oscillation_analyzer,
    brain_immune_system_t* immune_system
) {
    /* Guard: require both systems */
    if (!oscillation_analyzer || !immune_system) {
        nimcp_log(NIMCP_LOG_ERROR, "oscillations_immune_bridge",
                  "Cannot create bridge without oscillation analyzer and immune system");
        return NULL;
    }

    /* Allocate bridge */
    oscillations_immune_bridge_t* bridge = (oscillations_immune_bridge_t*)
        nimcp_malloc(sizeof(oscillations_immune_bridge_t));
    if (!bridge) {
        nimcp_log(NIMCP_LOG_ERROR, "oscillations_immune_bridge", "Allocation failed");
        return NULL;
    }

    /* Initialize to zero */
    memset(bridge, 0, sizeof(oscillations_immune_bridge_t));

    /* Link systems */
    bridge->oscillation_analyzer = oscillation_analyzer;
    bridge->immune_system = immune_system;

    /* Apply configuration */
    oscillations_immune_config_t default_cfg;
    if (!config) {
        oscillations_immune_default_config(&default_cfg);
        config = &default_cfg;
    }

    bridge->enable_cytokine_oscillation_modulation = config->enable_cytokine_oscillation_modulation;
    bridge->enable_inflammation_power_shift = config->enable_inflammation_power_shift;
    bridge->enable_oscillation_immune_trigger = config->enable_oscillation_immune_trigger;
    bridge->enable_abnormality_surveillance = config->enable_abnormality_surveillance;
    bridge->enable_il10_restoration = config->enable_il10_restoration;

    bridge->cytokine_sensitivity = config->cytokine_sensitivity;
    bridge->inflammation_sensitivity = config->inflammation_sensitivity;
    bridge->abnormality_sensitivity = config->abnormality_sensitivity;

    bridge->excessive_delta_threshold = config->excessive_delta_threshold;
    bridge->suppressed_gamma_threshold = config->suppressed_gamma_threshold;
    bridge->persistence_threshold = config->persistence_threshold;

    /* Create mutex */
    bridge->mutex = nimcp_malloc(sizeof(pthread_mutex_t));
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    pthread_mutex_init((pthread_mutex_t*)bridge->mutex, NULL);

    nimcp_log(NIMCP_LOG_INFO, "oscillations_immune_bridge", "Bridge created successfully");
    return bridge;
}

void oscillations_immune_bridge_destroy(oscillations_immune_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->mutex) {
        pthread_mutex_destroy((pthread_mutex_t*)bridge->mutex);
        nimcp_free(bridge->mutex);
    }

    /* Free bridge (don't destroy linked systems - we don't own them) */
    nimcp_free(bridge);
    nimcp_log(NIMCP_LOG_INFO, "oscillations_immune_bridge", "Bridge destroyed");
}

int oscillations_immune_establish_baseline(oscillations_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->oscillation_analyzer) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Get current oscillation state */
    oscillation_analysis_t analysis;
    if (!brain_oscillation_analyze(bridge->oscillation_analyzer, &analysis)) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return -1;
    }

    /* Store as baseline */
    bridge->baseline.baseline_power = analysis.wave_power;
    bridge->baseline.baseline_state = analysis.state;
    bridge->baseline.baseline_coherence = analysis.coherence;
    bridge->baseline.baseline_synchrony = analysis.synchrony;
    bridge->baseline.baseline_theta_gamma_pac = analysis.theta_gamma_coupling;
    bridge->baseline.baseline_established = true;
    bridge->baseline.baseline_timestamp = 0; /* Would use actual timestamp */

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    nimcp_log(NIMCP_LOG_INFO, "oscillations_immune_bridge", "Baseline established");
    return 0;
}

/* ============================================================================
 * Immune → Oscillations Implementation
 * ============================================================================ */

int oscillations_immune_apply_cytokine_effects(oscillations_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_cytokine_oscillation_modulation) return 0;
    if (!bridge->immune_system || !bridge->oscillation_analyzer) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    cytokine_oscillation_effects_t* effects = &bridge->cytokine_effects;

    /* Query cytokine concentrations */
    float il1_conc = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_IL1);
    float il6_conc = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_IL6);
    float tnf_conc = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_TNF);
    float ifn_gamma_conc = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_IFN_GAMMA);
    float il10_conc = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_IL10);

    /* Apply sensitivity */
    il1_conc *= bridge->cytokine_sensitivity;
    il6_conc *= bridge->cytokine_sensitivity;
    tnf_conc *= bridge->cytokine_sensitivity;
    ifn_gamma_conc *= bridge->cytokine_sensitivity;

    /* Pro-inflammatory effects */
    effects->il1_delta_amplification = 1.0f + (il1_conc * (CYTOKINE_IL1_DELTA_AMPLIFICATION - 1.0f));
    effects->il1_gamma_suppression = 1.0f - (il1_conc * (1.0f - CYTOKINE_IL1_GAMMA_SUPPRESSION));
    effects->il6_delta_amplification = 1.0f + (il6_conc * (CYTOKINE_IL6_DELTA_AMPLIFICATION - 1.0f));
    effects->il6_beta_suppression = 1.0f - (il6_conc * (1.0f - CYTOKINE_IL6_BETA_SUPPRESSION));
    effects->tnf_delta_amplification = 1.0f + (tnf_conc * (CYTOKINE_TNF_DELTA_AMPLIFICATION - 1.0f));
    effects->tnf_gamma_suppression = 1.0f - (tnf_conc * (1.0f - CYTOKINE_TNF_GAMMA_SUPPRESSION));
    effects->ifn_gamma_theta_suppression = 1.0f - (ifn_gamma_conc * (1.0f - CYTOKINE_IFN_GAMMA_THETA_SUPPRESSION));

    /* Anti-inflammatory restoration */
    effects->il10_restoration = il10_conc * CYTOKINE_IL10_RESTORATION_RATE;

    /* Aggregate effects */
    effects->total_delta_amplification = fmaxf(fmaxf(effects->il1_delta_amplification,
                                                      effects->il6_delta_amplification),
                                                effects->tnf_delta_amplification);

    effects->total_gamma_suppression = fminf(effects->il1_gamma_suppression,
                                              effects->tnf_gamma_suppression);

    effects->total_beta_suppression = effects->il6_beta_suppression;

    effects->total_theta_suppression = effects->ifn_gamma_theta_suppression;

    /* Network disruption scales with pro-inflammatory burden */
    float proinflam_total = (il1_conc + il6_conc + tnf_conc) / 3.0f;
    effects->coherence_disruption = clamp_f(proinflam_total * 0.6f, 0.0f, 0.8f);
    effects->synchrony_disruption = clamp_f(proinflam_total * 0.5f, 0.0f, 0.7f);
    effects->theta_gamma_decoupling = clamp_f(proinflam_total * 0.7f, 0.0f, 0.9f);

    /* Apply to oscillation analyzer */
    immune_oscillation_effects_t osc_effects = {
        .delta_amplification = effects->total_delta_amplification,
        .theta_suppression = effects->total_theta_suppression,
        .gamma_suppression = effects->total_gamma_suppression,
        .beta_suppression = effects->total_beta_suppression,
        .coherence_disruption = effects->coherence_disruption,
        .synchrony_disruption = effects->synchrony_disruption
    };

    brain_oscillation_apply_immune_effects(bridge->oscillation_analyzer, &osc_effects);

    bridge->cytokine_modulations++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int oscillations_immune_apply_inflammation_effects(oscillations_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_inflammation_power_shift) return 0;
    if (!bridge->immune_system || !bridge->oscillation_analyzer) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    inflammation_oscillation_state_t* state = &bridge->inflammation_state;

    /* Get inflammation state */
    state->current_level = get_max_inflammation_level(bridge->immune_system);
    state->inflammation_intensity = (float)state->current_level / (float)INFLAMMATION_STORM;
    state->inflammation_intensity *= bridge->inflammation_sensitivity;
    state->inflammation_intensity = clamp_f(state->inflammation_intensity, 0.0f, 1.0f);

    uint64_t current_time = 0; /* Would use actual time */
    state->inflammation_duration_sec = get_inflammation_duration_sec(
        bridge->immune_system, current_time);

    /* Power spectrum shifts based on inflammation level */
    switch (state->current_level) {
        case INFLAMMATION_NONE:
            state->delta_power_shift = 1.0f;
            state->theta_power_shift = 1.0f;
            state->alpha_power_shift = 1.0f;
            state->beta_power_shift = 1.0f;
            state->gamma_power_shift = 1.0f;
            state->coherence_reduction = 0.0f;
            state->synchrony_reduction = 0.0f;
            state->expected_state = COGNITIVE_STATE_UNKNOWN;
            break;

        case INFLAMMATION_LOCAL:
            state->delta_power_shift = 1.2f;
            state->theta_power_shift = 0.95f;
            state->alpha_power_shift = 0.9f;
            state->beta_power_shift = 0.9f;
            state->gamma_power_shift = 0.85f;
            state->coherence_reduction = 0.1f;
            state->synchrony_reduction = 0.1f;
            state->expected_state = COGNITIVE_STATE_RELAXED;
            break;

        case INFLAMMATION_REGIONAL:
            state->delta_power_shift = 1.5f;
            state->theta_power_shift = 0.85f;
            state->alpha_power_shift = 0.8f;
            state->beta_power_shift = 0.75f;
            state->gamma_power_shift = 0.7f;
            state->coherence_reduction = 0.3f;
            state->synchrony_reduction = 0.25f;
            state->expected_state = COGNITIVE_STATE_LIGHT_SLEEP;
            break;

        case INFLAMMATION_SYSTEMIC:
            state->delta_power_shift = 2.0f;
            state->theta_power_shift = 0.7f;
            state->alpha_power_shift = 0.6f;
            state->beta_power_shift = 0.6f;
            state->gamma_power_shift = 0.5f;
            state->coherence_reduction = 0.5f;
            state->synchrony_reduction = 0.4f;
            state->expected_state = COGNITIVE_STATE_DEEP_SLEEP;
            break;

        case INFLAMMATION_STORM:
            state->delta_power_shift = 3.0f;
            state->theta_power_shift = 0.6f;
            state->alpha_power_shift = 0.5f;
            state->beta_power_shift = 0.5f;
            state->gamma_power_shift = 0.3f;
            state->coherence_reduction = 0.7f;
            state->synchrony_reduction = 0.6f;
            state->expected_state = COGNITIVE_STATE_DEEP_SLEEP;
            break;
    }

    state->state_shift_severity = state->inflammation_intensity;

    bridge->power_spectrum_shifts++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int oscillations_immune_restore_with_il10(
    oscillations_immune_bridge_t* bridge,
    float il10_concentration
) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_il10_restoration) return 0;
    if (!bridge->baseline.baseline_established) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* IL-10 gradually restores oscillations toward baseline */
    float restoration_rate = il10_concentration * CYTOKINE_IL10_RESTORATION_RATE;

    /* Would apply restoration to oscillation analyzer */
    /* This would interpolate current state toward baseline state */

    bridge->il10_restorations++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

cognitive_state_t oscillations_immune_compute_state_shift(
    const oscillations_immune_bridge_t* bridge
) {
    if (!bridge) return COGNITIVE_STATE_UNKNOWN;

    return bridge->inflammation_state.expected_state;
}

/* ============================================================================
 * Oscillations → Immune Implementation
 * ============================================================================ */

bool oscillations_immune_detect_abnormality(oscillations_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return false;
    if (!bridge->enable_abnormality_surveillance) return false;
    if (!bridge->oscillation_analyzer) return false;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    oscillation_immune_trigger_t* trigger = &bridge->immune_trigger;

    /* Get current oscillation analysis */
    oscillation_analysis_t analysis;
    if (!brain_oscillation_analyze(bridge->oscillation_analyzer, &analysis)) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return false;
    }

    /* Check for abnormalities */
    float delta_ratio = analysis.wave_power.delta_power / analysis.wave_power.total_power;
    float gamma_ratio = analysis.wave_power.gamma_power / analysis.wave_power.total_power;

    trigger->excessive_delta = (delta_ratio > bridge->excessive_delta_threshold);
    trigger->suppressed_gamma = (gamma_ratio < bridge->suppressed_gamma_threshold);
    trigger->low_coherence = (analysis.coherence < INFLAMMATION_COHERENCE_THRESHOLD);
    trigger->low_synchrony = (analysis.synchrony < INFLAMMATION_SYNCHRONY_THRESHOLD);

    /* Update persistence counter */
    bool any_abnormal = trigger->excessive_delta || trigger->suppressed_gamma ||
                        trigger->low_coherence || trigger->low_synchrony;

    if (any_abnormal) {
        trigger->consecutive_abnormal++;
    } else {
        trigger->consecutive_abnormal = 0;
    }

    /* Compute abnormality score */
    trigger->abnormality_score = 0.0f;
    if (trigger->excessive_delta) {
        trigger->abnormality_score += ABNORMALITY_WEIGHT_EXCESSIVE_DELTA;
    }
    if (trigger->suppressed_gamma) {
        trigger->abnormality_score += ABNORMALITY_WEIGHT_SUPPRESSED_GAMMA;
    }
    if (trigger->low_coherence) {
        trigger->abnormality_score += ABNORMALITY_WEIGHT_LOW_COHERENCE;
    }
    if (trigger->low_synchrony) {
        trigger->abnormality_score += ABNORMALITY_WEIGHT_LOW_SYNCHRONY;
    }

    /* Apply sensitivity */
    trigger->abnormality_score *= bridge->abnormality_sensitivity;
    trigger->abnormality_score = clamp_f(trigger->abnormality_score, 0.0f, 1.0f);

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return any_abnormal;
}

int oscillations_immune_trigger_from_abnormality(oscillations_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_oscillation_immune_trigger) return 0;
    if (!bridge->immune_system || !bridge->oscillation_analyzer) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    oscillation_immune_trigger_t* trigger = &bridge->immune_trigger;

    /* Only trigger if abnormality persists */
    if (trigger->consecutive_abnormal < bridge->persistence_threshold) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return 0;
    }

    /* Prevent duplicate triggers */
    if (trigger->antigen_presented) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return 0;
    }

    /* Get current oscillation state for epitope */
    oscillation_analysis_t analysis;
    if (!brain_oscillation_analyze(bridge->oscillation_analyzer, &analysis)) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return -1;
    }

    /* Create epitope from abnormal pattern */
    uint8_t epitope[BRAIN_IMMUNE_EPITOPE_SIZE];
    size_t epitope_len = create_oscillation_epitope(
        &analysis.wave_power,
        analysis.coherence,
        analysis.synchrony,
        epitope,
        BRAIN_IMMUNE_EPITOPE_SIZE
    );

    if (epitope_len == 0) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return -1;
    }

    /* Map abnormality score to immune severity [1-10] */
    trigger->immune_severity = (uint32_t)(trigger->abnormality_score * 9.0f) + 1;
    trigger->immune_severity = (trigger->immune_severity < 1) ? 1 :
                               (trigger->immune_severity > 10) ? 10 : trigger->immune_severity;

    /* Present antigen to immune system */
    int result = brain_immune_present_antigen(
        bridge->immune_system,
        ANTIGEN_SOURCE_ANOMALY,
        epitope,
        epitope_len,
        trigger->immune_severity,
        0, /* Source node - would use actual ID */
        &trigger->antigen_id
    );

    if (result == 0) {
        trigger->immune_surveillance_triggered = true;
        trigger->antigen_presented = true;
        bridge->immune_triggers++;
        bridge->antigens_presented++;

        nimcp_log(LOG_LEVEL_WARN, "oscillations_immune_bridge",
                  "Abnormal oscillations triggered immune response (severity=%u, score=%.2f)",
                  trigger->immune_severity, trigger->abnormality_score);
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return result;
}

float oscillations_immune_compute_abnormality_score(
    const oscillations_immune_bridge_t* bridge
) {
    if (!bridge) return 0.0f;

    return bridge->immune_trigger.abnormality_score;
}

/* ============================================================================
 * Bidirectional Update Implementation
 * ============================================================================ */

int oscillations_immune_bridge_update(
    oscillations_immune_bridge_t* bridge,
    uint64_t delta_ms
) {
    /* Guard clauses */
    if (!bridge) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Immune → Oscillations pathway */
    if (bridge->enable_cytokine_oscillation_modulation) {
        oscillations_immune_apply_cytokine_effects(bridge);
    }

    if (bridge->enable_inflammation_power_shift) {
        oscillations_immune_apply_inflammation_effects(bridge);
    }

    /* IL-10 restoration */
    if (bridge->enable_il10_restoration) {
        float il10_conc = get_cytokine_concentration(
            bridge->immune_system, BRAIN_CYTOKINE_IL10);
        if (il10_conc > 0.0f) {
            oscillations_immune_restore_with_il10(bridge, il10_conc);
        }
    }

    /* Oscillations → Immune pathway */
    if (bridge->enable_abnormality_surveillance) {
        bool abnormal = oscillations_immune_detect_abnormality(bridge);

        if (abnormal && bridge->enable_oscillation_immune_trigger) {
            oscillations_immune_trigger_from_abnormality(bridge);
        }
    }

    bridge->total_updates++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int oscillations_immune_get_cytokine_effects(
    const oscillations_immune_bridge_t* bridge,
    cytokine_oscillation_effects_t* effects
) {
    if (!bridge || !effects) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    *effects = bridge->cytokine_effects;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int oscillations_immune_get_inflammation_state(
    const oscillations_immune_bridge_t* bridge,
    inflammation_oscillation_state_t* state
) {
    if (!bridge || !state) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    *state = bridge->inflammation_state;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int oscillations_immune_get_trigger_state(
    const oscillations_immune_bridge_t* bridge,
    oscillation_immune_trigger_t* trigger
) {
    if (!bridge || !trigger) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    *trigger = bridge->immune_trigger;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

bool oscillations_immune_is_modulated(const oscillations_immune_bridge_t* bridge) {
    if (!bridge) return false;

    /* Check if any cytokine effects are active */
    return (bridge->cytokine_effects.total_delta_amplification > 1.1f) ||
           (bridge->cytokine_effects.total_gamma_suppression < 0.9f) ||
           (bridge->inflammation_state.current_level > INFLAMMATION_NONE);
}

float oscillations_immune_get_delta_amplification(
    const oscillations_immune_bridge_t* bridge
) {
    if (!bridge) return 1.0f;

    return bridge->cytokine_effects.total_delta_amplification;
}

float oscillations_immune_get_gamma_suppression(
    const oscillations_immune_bridge_t* bridge
) {
    if (!bridge) return 1.0f;

    return bridge->cytokine_effects.total_gamma_suppression;
}
