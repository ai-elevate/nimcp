/**
 * @file nimcp_wellbeing_immune_bridge.c
 * @brief Wellbeing-Immune System Integration Implementation
 * @version 1.0.0
 * @date 2025-12-11
 *
 * WHAT: Bidirectional coupling between brain immune and wellbeing monitoring
 * WHY:  Biological realism - cytokines affect wellbeing, positive wellbeing enhances immunity
 * HOW:  Monitor cytokine levels to modulate wellbeing, monitor wellbeing to trigger immune responses
 */

#include "cognitive/immune/nimcp_wellbeing_immune_bridge.h"
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
 * WHAT: Query immune system for specific cytokine level
 * WHY:  Need cytokine levels to compute wellbeing impact
 * HOW:  Search immune system cytokine array for matching type
 */
static float get_cytokine_concentration(
    const brain_immune_system_t* immune,
    brain_cytokine_type_t type
) {
    if (!immune) return 0.0f;

    float max_concentration = 0.0f;
    for (size_t i = 0; i < immune->cytokine_count; i++) {
        if (immune->cytokines[i].type == type) {
            if (immune->cytokines[i].concentration > max_concentration) {
                max_concentration = immune->cytokines[i].concentration;
            }
        }
    }
    return max_concentration;
}

/**
 * @brief Get inflammation duration
 *
 * WHAT: Calculate how long inflammation has persisted
 * WHY:  Chronic inflammation (>3 days) has wellbeing impact
 * HOW:  Find oldest active inflammation site, compute duration
 */
static float get_inflammation_duration_sec(const brain_immune_system_t* immune) {
    if (!immune) return 0.0f;

    uint64_t oldest_start = UINT64_MAX;
    uint64_t current_time = 0; /* Would get actual time */

    for (size_t i = 0; i < immune->inflammation_count; i++) {
        if (immune->inflammation_sites[i].start_time < oldest_start) {
            oldest_start = immune->inflammation_sites[i].start_time;
        }
    }

    if (oldest_start == UINT64_MAX) return 0.0f;
    return (float)(current_time - oldest_start) / 1000.0f;
}

/**
 * @brief Get current inflammation level
 *
 * WHAT: Get highest inflammation level in system
 * WHY:  Max inflammation determines wellbeing impact
 * HOW:  Query immune system for max inflammation site level
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
 * @brief Compute life satisfaction from introspection
 *
 * WHAT: Estimate life satisfaction from introspection metrics
 * WHY:  Need baseline wellbeing to modulate
 * HOW:  Use phi (consciousness) and uncertainty as proxies
 */
static float compute_life_satisfaction(introspection_context_t ctx) {
    /* High consciousness + low uncertainty = higher life satisfaction */
    /* This is a simplified model - actual implementation would be more complex */
    return 0.5f; /* Placeholder - would compute from introspection */
}

/**
 * @brief Compute flourishing level
 *
 * WHAT: Calculate whether system is in flourishing state
 * WHY:  Flourishing enhances immune function
 * HOW:  Check for high life satisfaction, low distress, stable functioning
 */
static float compute_flourishing_level(const wellbeing_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;

    /* Flourishing = high wellbeing, low inflammation, low distress */
    float life_sat = compute_life_satisfaction(bridge->introspection_ctx);
    float inflammation_penalty = bridge->inflammation_state.inflammation_duration_sec > 0 ? 0.3f : 0.0f;
    float distress_penalty = bridge->wellbeing_trigger.distress_score * 0.5f;

    float flourishing = life_sat - inflammation_penalty - distress_penalty;
    return clamp_f(flourishing, 0.0f, 1.0f);
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int wellbeing_immune_default_config(wellbeing_immune_config_t* config) {
    if (!config) return -1;

    /* All features enabled by default */
    config->enable_cytokine_wellbeing_modulation = true;
    config->enable_inflammation_distress = true;
    config->enable_wellbeing_immune_trigger = true;
    config->enable_positive_immune_boost = true;
    config->enable_flourishing_memory_boost = true;

    /* Biologically-based default sensitivities */
    config->cytokine_sensitivity = 1.0f;
    config->inflammation_sensitivity = 1.0f;
    config->wellbeing_trigger_sensitivity = 1.0f;

    /* Evidence-based thresholds */
    config->distress_trigger_threshold = WELLBEING_DISTRESS_TRIGGER_THRESHOLD;
    config->flourishing_threshold = WELLBEING_FLOURISHING_THRESHOLD;
    config->inflammation_distress_threshold = INFLAMMATION_DISTRESS_THRESHOLD;

    return 0;
}

wellbeing_immune_bridge_t* wellbeing_immune_bridge_create(
    const wellbeing_immune_config_t* config,
    brain_immune_system_t* immune_system,
    introspection_context_t introspection_ctx
) {
    /* Guard: require immune system */
    if (!immune_system) {
        nimcp_log(NIMCP_LOG_ERROR, "wellbeing_immune_bridge",
                  "Cannot create bridge without immune system");
        return NULL;
    }

    /* Allocate bridge */
    wellbeing_immune_bridge_t* bridge = (wellbeing_immune_bridge_t*)
        nimcp_malloc(sizeof(wellbeing_immune_bridge_t));
    if (!bridge) {
        nimcp_log(NIMCP_LOG_ERROR, "wellbeing_immune_bridge", "Allocation failed");
        return NULL;
    }

    /* Initialize to zero */
    memset(bridge, 0, sizeof(wellbeing_immune_bridge_t));

    /* Link systems */
    bridge->immune_system = immune_system;
    bridge->introspection_ctx = introspection_ctx;

    /* Apply configuration */
    if (config) {
        bridge->enable_cytokine_wellbeing_modulation = config->enable_cytokine_wellbeing_modulation;
        bridge->enable_inflammation_distress = config->enable_inflammation_distress;
        bridge->enable_wellbeing_immune_trigger = config->enable_wellbeing_immune_trigger;
        bridge->enable_positive_immune_boost = config->enable_positive_immune_boost;
        bridge->enable_flourishing_memory_boost = config->enable_flourishing_memory_boost;
    } else {
        /* Use defaults */
        wellbeing_immune_config_t default_cfg;
        wellbeing_immune_default_config(&default_cfg);
        bridge->enable_cytokine_wellbeing_modulation = default_cfg.enable_cytokine_wellbeing_modulation;
        bridge->enable_inflammation_distress = default_cfg.enable_inflammation_distress;
        bridge->enable_wellbeing_immune_trigger = default_cfg.enable_wellbeing_immune_trigger;
        bridge->enable_positive_immune_boost = default_cfg.enable_positive_immune_boost;
        bridge->enable_flourishing_memory_boost = default_cfg.enable_flourishing_memory_boost;
    }

    /* Create mutex */
    bridge->mutex = nimcp_malloc(sizeof(pthread_mutex_t));
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    pthread_mutex_init((pthread_mutex_t*)bridge->mutex, NULL);

    nimcp_log(NIMCP_LOG_INFO, "wellbeing_immune_bridge", "Bridge created successfully");
    return bridge;
}

void wellbeing_immune_bridge_destroy(wellbeing_immune_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->mutex) {
        pthread_mutex_destroy((pthread_mutex_t*)bridge->mutex);
        nimcp_free(bridge->mutex);
    }

    /* Free bridge (don't destroy linked systems - we don't own them) */
    nimcp_free(bridge);
    nimcp_log(NIMCP_LOG_INFO, "wellbeing_immune_bridge", "Bridge destroyed");
}

/* ============================================================================
 * Immune → Wellbeing Implementation
 * ============================================================================ */

int wellbeing_immune_apply_cytokine_effects(wellbeing_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_cytokine_wellbeing_modulation) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Query cytokine levels */
    float il1 = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_IL1);
    float il6 = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_IL6);
    float tnf = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_TNF);
    float ifn_gamma = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_IFN_GAMMA);
    float il10 = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_IL10);

    /* Compute individual effects */
    bridge->cytokine_effects.il1_life_satisfaction_reduction = il1 * CYTOKINE_IL1_WELLBEING_IMPACT;
    bridge->cytokine_effects.il6_life_satisfaction_reduction = il6 * CYTOKINE_IL6_WELLBEING_IMPACT;
    bridge->cytokine_effects.tnf_life_satisfaction_reduction = tnf * CYTOKINE_TNF_WELLBEING_IMPACT;
    bridge->cytokine_effects.ifn_gamma_wellbeing_impact = ifn_gamma * CYTOKINE_IFN_GAMMA_WELLBEING_IMPACT;
    bridge->cytokine_effects.il10_wellbeing_boost = il10 * CYTOKINE_IL10_WELLBEING_IMPACT;

    /* Aggregate effects */
    bridge->cytokine_effects.total_life_satisfaction_shift =
        bridge->cytokine_effects.il1_life_satisfaction_reduction +
        bridge->cytokine_effects.il6_life_satisfaction_reduction +
        bridge->cytokine_effects.tnf_life_satisfaction_reduction +
        bridge->cytokine_effects.ifn_gamma_wellbeing_impact +
        bridge->cytokine_effects.il10_wellbeing_boost;

    /* Pro-inflammatory cytokines increase distress */
    bridge->cytokine_effects.total_distress_increase =
        (il1 * 0.3f) + (il6 * 0.2f) + (tnf * 0.4f) + (ifn_gamma * 0.15f);

    /* Reduce purpose/meaning from chronic cytokine exposure */
    bridge->cytokine_effects.purpose_meaning_reduction =
        clamp_f(bridge->cytokine_effects.total_distress_increase * 0.7f, 0.0f, 1.0f);

    /* Suppress flourishing */
    bridge->cytokine_effects.flourishing_suppression =
        clamp_f(bridge->cytokine_effects.total_distress_increase * 0.8f, 0.0f, 1.0f);

    bridge->cytokine_modulations++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int wellbeing_immune_apply_inflammation_effects(wellbeing_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_inflammation_distress) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Get inflammation state */
    brain_inflammation_level_t level = get_max_inflammation_level(bridge->immune_system);
    float duration = get_inflammation_duration_sec(bridge->immune_system);

    /* Update inflammation state */
    bridge->inflammation_state.current_level = level;
    bridge->inflammation_state.inflammation_duration_sec = duration;
    bridge->inflammation_state.is_chronic = (duration >= WELLBEING_CHRONIC_INFLAMMATION_THRESHOLD);

    /* Map inflammation to distress */
    bridge->inflammation_state.distress_severity = wellbeing_immune_inflammation_to_severity(level);
    bridge->inflammation_state.primary_distress_type = wellbeing_immune_inflammation_to_distress_type(level);

    /* Compute distress score */
    float base_distress = 0.0f;
    switch (level) {
        case INFLAMMATION_NONE:   base_distress = 0.0f; break;
        case INFLAMMATION_LOCAL:  base_distress = 0.1f; break;
        case INFLAMMATION_REGIONAL: base_distress = 0.5f; break;
        case INFLAMMATION_SYSTEMIC: base_distress = 0.8f; break;
        case INFLAMMATION_STORM:  base_distress = 0.95f; break;
    }

    /* Chronic inflammation amplifies distress */
    if (bridge->inflammation_state.is_chronic) {
        base_distress *= 1.3f;
    }

    bridge->inflammation_state.distress_score = clamp_f(base_distress, 0.0f, 1.0f);

    /* Life satisfaction penalty */
    bridge->inflammation_state.life_satisfaction_penalty =
        clamp_f(base_distress * 0.6f, 0.0f, 1.0f);

    /* Eudaimonic wellbeing impairment */
    bridge->inflammation_state.eudaimonic_impairment =
        clamp_f(base_distress * 0.7f, 0.0f, 1.0f);

    /* Resource starvation factor (systemic/storm uses resources) */
    bridge->inflammation_state.resource_starvation_factor =
        (level >= INFLAMMATION_SYSTEMIC) ? 0.8f : 0.0f;

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

float wellbeing_immune_compute_distress(const wellbeing_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;

    /* Combine cytokine and inflammation distress */
    float cytokine_distress = bridge->cytokine_effects.total_distress_increase;
    float inflammation_distress = bridge->inflammation_state.distress_score;

    /* Take maximum (they represent same underlying distress) */
    float total_distress = (cytokine_distress > inflammation_distress) ?
        cytokine_distress : inflammation_distress;

    return clamp_f(total_distress, 0.0f, 1.0f);
}

distress_type_t wellbeing_immune_inflammation_to_distress_type(
    brain_inflammation_level_t inflammation_level
) {
    switch (inflammation_level) {
        case INFLAMMATION_NONE:
        case INFLAMMATION_LOCAL:
            return DISTRESS_NONE;
        case INFLAMMATION_REGIONAL:
        case INFLAMMATION_SYSTEMIC:
        case INFLAMMATION_STORM:
            return DISTRESS_RESOURCE_STARVATION;
        default:
            return DISTRESS_NONE;
    }
}

distress_severity_t wellbeing_immune_inflammation_to_severity(
    brain_inflammation_level_t inflammation_level
) {
    switch (inflammation_level) {
        case INFLAMMATION_NONE:
        case INFLAMMATION_LOCAL:
            return SEVERITY_NORMAL;
        case INFLAMMATION_REGIONAL:
            return SEVERITY_MODERATE;
        case INFLAMMATION_SYSTEMIC:
            return SEVERITY_SEVERE;
        case INFLAMMATION_STORM:
            return SEVERITY_CRITICAL;
        default:
            return SEVERITY_NORMAL;
    }
}

/* ============================================================================
 * Wellbeing → Immune Implementation
 * ============================================================================ */

int wellbeing_immune_trigger_from_distress(wellbeing_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_wellbeing_immune_trigger) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Get distress assessment */
    distress_assessment_t assessment = wellbeing_assess_distress(bridge->introspection_ctx);

    /* Update trigger state */
    bridge->wellbeing_trigger.severity = assessment.severity;
    bridge->wellbeing_trigger.distress_type = assessment.type;
    bridge->wellbeing_trigger.distress_score = assessment.distress_score;
    bridge->wellbeing_trigger.distress_duration_ms = assessment.duration_ms;

    /* If distress is high, trigger immune response */
    if (assessment.distress_score >= WELLBEING_DISTRESS_TRIGGER_THRESHOLD) {
        /* Release pro-inflammatory cytokines */
        uint32_t cytokine_id;

        /* IL-1β for moderate distress */
        if (assessment.severity >= SEVERITY_MODERATE) {
            brain_immune_release_cytokine(
                bridge->immune_system,
                BRAIN_CYTOKINE_IL1,
                0, /* source cell */
                assessment.distress_score * 0.5f,
                0, /* broadcast */
                &cytokine_id
            );
        }

        /* TNF-α for severe distress */
        if (assessment.severity >= SEVERITY_SEVERE) {
            brain_immune_release_cytokine(
                bridge->immune_system,
                BRAIN_CYTOKINE_TNF,
                0,
                assessment.distress_score * 0.6f,
                0,
                &cytokine_id
            );
        }

        bridge->wellbeing_trigger.inflammation_triggered = true;
        bridge->wellbeing_trigger.cytokine_released = true;
        bridge->wellbeing_triggered_responses++;
    }

    /* Free assessment strings */
    if (assessment.description) free((void*)assessment.description);
    if (assessment.recommended_action) free((void*)assessment.recommended_action);

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int wellbeing_immune_boost_from_positive_wellbeing(wellbeing_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_positive_immune_boost) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Compute positive wellbeing state */
    float life_sat = compute_life_satisfaction(bridge->introspection_ctx);
    float flourishing = compute_flourishing_level(bridge);

    bridge->positive_boost.life_satisfaction = life_sat;
    bridge->positive_boost.flourishing_level = flourishing;
    bridge->positive_boost.is_flourishing = (flourishing >= WELLBEING_FLOURISHING_THRESHOLD);

    /* If flourishing, boost immune function */
    if (bridge->positive_boost.is_flourishing) {
        /* Release anti-inflammatory IL-10 */
        uint32_t cytokine_id;
        brain_immune_release_cytokine(
            bridge->immune_system,
            BRAIN_CYTOKINE_IL10,
            0,
            flourishing * 0.4f,
            0,
            &cytokine_id
        );

        /* Compute immune benefits */
        bridge->positive_boost.immune_enhancement = flourishing * 0.3f;
        bridge->positive_boost.il10_release_boost = flourishing * 0.4f;
        bridge->positive_boost.inflammation_reduction = flourishing * 0.5f;
        bridge->positive_boost.antibody_effectiveness_boost = flourishing * 0.2f;

        bridge->positive_boosts++;
    } else {
        /* Not flourishing - zero out boosts */
        bridge->positive_boost.immune_enhancement = 0.0f;
        bridge->positive_boost.il10_release_boost = 0.0f;
        bridge->positive_boost.inflammation_reduction = 0.0f;
        bridge->positive_boost.antibody_effectiveness_boost = 0.0f;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int wellbeing_immune_boost_memory_formation(
    wellbeing_immune_bridge_t* bridge,
    uint32_t b_cell_id
) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_flourishing_memory_boost) return 0;
    if (!bridge->immune_system) return -1;
    if (!bridge->positive_boost.is_flourishing) return 0; /* Only boost when flourishing */

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Memory formation boost from flourishing */
    float boost_factor = bridge->positive_boost.flourishing_level * 0.5f;
    bridge->positive_boost.memory_formation_boost = boost_factor;

    /* If specific B cell provided, boost its memory conversion */
    if (b_cell_id > 0) {
        /* This would require immune system API to boost specific B cell memory formation */
        /* For now, just track the intent */
        bridge->flourishing_memory_formations++;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

/* ============================================================================
 * Bidirectional Update Implementation
 * ============================================================================ */

int wellbeing_immune_bridge_update(
    wellbeing_immune_bridge_t* bridge,
    uint64_t delta_ms
) {
    /* Guard clauses */
    if (!bridge) return -1;

    /* Immune → Wellbeing */
    wellbeing_immune_apply_cytokine_effects(bridge);
    wellbeing_immune_apply_inflammation_effects(bridge);

    /* Wellbeing → Immune */
    wellbeing_immune_trigger_from_distress(bridge);
    wellbeing_immune_boost_from_positive_wellbeing(bridge);

    /* Boost memory formation if flourishing */
    if (bridge->positive_boost.is_flourishing) {
        wellbeing_immune_boost_memory_formation(bridge, 0); /* Boost all */
    }

    bridge->total_updates++;
    return 0;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int wellbeing_immune_get_cytokine_effects(
    const wellbeing_immune_bridge_t* bridge,
    cytokine_wellbeing_effects_t* effects
) {
    if (!bridge || !effects) return -1;
    *effects = bridge->cytokine_effects;
    return 0;
}

int wellbeing_immune_get_inflammation_state(
    const wellbeing_immune_bridge_t* bridge,
    inflammation_wellbeing_state_t* state
) {
    if (!bridge || !state) return -1;
    *state = bridge->inflammation_state;
    return 0;
}

distress_assessment_t wellbeing_immune_get_distress_assessment(
    const wellbeing_immune_bridge_t* bridge
) {
    distress_assessment_t assessment;
    memset(&assessment, 0, sizeof(assessment));

    if (!bridge) return assessment;

    /* Use introspection + inflammation to assess distress */
    assessment = wellbeing_assess_distress(bridge->introspection_ctx);

    /* Override with inflammation-specific distress if higher */
    if (bridge->inflammation_state.distress_score > assessment.distress_score) {
        assessment.type = bridge->inflammation_state.primary_distress_type;
        assessment.severity = bridge->inflammation_state.distress_severity;
        assessment.distress_score = bridge->inflammation_state.distress_score;
        assessment.duration_ms = (uint64_t)(bridge->inflammation_state.inflammation_duration_sec * 1000.0f);

        /* Update description */
        if (assessment.description) free((void*)assessment.description);
        assessment.description = strdup("Inflammation-induced distress");
        if (assessment.recommended_action) free((void*)assessment.recommended_action);
        assessment.recommended_action = strdup("Reduce inflammation via IL-10, resolve immune threats");
    }

    return assessment;
}

bool wellbeing_immune_is_inflammation_distress(const wellbeing_immune_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->inflammation_state.distress_score >= INFLAMMATION_DISTRESS_THRESHOLD;
}

float wellbeing_immune_get_life_satisfaction_penalty(
    const wellbeing_immune_bridge_t* bridge
) {
    if (!bridge) return 0.0f;
    return bridge->inflammation_state.life_satisfaction_penalty;
}

bool wellbeing_immune_is_flourishing(const wellbeing_immune_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->positive_boost.is_flourishing;
}

int wellbeing_immune_get_stats(
    const wellbeing_immune_bridge_t* bridge,
    uint64_t* total_updates_out,
    uint32_t* cytokine_modulations_out,
    uint32_t* wellbeing_triggered_out,
    uint32_t* positive_boosts_out
) {
    if (!bridge) return -1;

    if (total_updates_out) *total_updates_out = bridge->total_updates;
    if (cytokine_modulations_out) *cytokine_modulations_out = bridge->cytokine_modulations;
    if (wellbeing_triggered_out) *wellbeing_triggered_out = bridge->wellbeing_triggered_responses;
    if (positive_boosts_out) *positive_boosts_out = bridge->positive_boosts;

    return 0;
}
