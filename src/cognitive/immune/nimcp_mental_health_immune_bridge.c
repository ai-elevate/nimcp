/**
 * @file nimcp_mental_health_immune_bridge.c
 * @brief Mental Health-Immune System Integration Implementation
 * @version 1.0.0
 * @date 2025-12-11
 *
 * WHAT: Bidirectional coupling between brain immune and mental health systems
 * WHY:  Biological realism - cytokines affect mental health, disorders affect immunity
 * HOW:  Monitor cytokine levels to modulate disorder risk, monitor disorders to trigger immune responses
 */

#include "cognitive/immune/nimcp_mental_health_immune_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
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
 * @brief Get inflammation duration
 *
 * WHAT: Calculate how long inflammation has persisted
 * WHY:  Chronic inflammation (>14 days) has different mental health effects
 * HOW:  Find oldest active inflammation site, compute duration
 */
static float get_inflammation_duration_sec(const brain_immune_system_t* immune) {
    if (!immune) return 0.0f;

    /* Query immune system for inflammation sites */
    brain_immune_stats_t stats;
    if (brain_immune_get_stats((brain_immune_system_t*)immune, &stats) != 0) {
        return 0.0f;
    }

    /* If no inflammation, duration is zero */
    if (stats.inflammation_sites == 0) {
        return 0.0f;
    }

    /* Approximate duration - would need actual timestamp tracking */
    /* For now, assume sites persist for a while if count is high */
    return (float)(stats.inflammation_sites * 3600.0f); /* 1 hour per site */
}

/**
 * @brief Get current inflammation level
 *
 * WHAT: Get highest inflammation level in system
 * WHY:  Max inflammation determines mental health impact
 * HOW:  Query immune system for inflammation state
 */
static brain_inflammation_level_t get_max_inflammation_level(
    const brain_immune_system_t* immune
) {
    if (!immune) return INFLAMMATION_NONE;

    brain_immune_stats_t stats;
    if (brain_immune_get_stats((brain_immune_system_t*)immune, &stats) != 0) {
        return INFLAMMATION_NONE;
    }

    /* Map site count to inflammation level */
    if (stats.inflammation_sites == 0) return INFLAMMATION_NONE;
    if (stats.inflammation_sites < 2) return INFLAMMATION_LOCAL;
    if (stats.inflammation_sites < 4) return INFLAMMATION_REGIONAL;
    if (stats.inflammation_sites < 6) return INFLAMMATION_SYSTEMIC;
    return INFLAMMATION_STORM;
}

/**
 * @brief Get cytokine level estimate
 *
 * WHAT: Estimate cytokine concentration from immune state
 * WHY:  Cytokines not directly accessible, estimate from inflammation
 * HOW:  Map inflammation level to cytokine concentration
 */
static float estimate_cytokine_level(
    const brain_immune_system_t* immune,
    brain_cytokine_type_t type
) {
    if (!immune) return 0.0f;

    brain_inflammation_level_t level = get_max_inflammation_level(immune);
    float base_level = (float)level / (float)INFLAMMATION_STORM;

    /* Different cytokines scale differently */
    switch (type) {
        case BRAIN_CYTOKINE_IL1:
            return base_level * 0.8f; /* IL-1β peaks early */
        case BRAIN_CYTOKINE_IL6:
            return base_level * 1.0f; /* IL-6 tracks inflammation closely */
        case BRAIN_CYTOKINE_TNF:
            return base_level * 0.9f; /* TNF-α strong but variable */
        case BRAIN_CYTOKINE_IFN_GAMMA:
            return base_level * 0.6f; /* IFN-γ lower levels */
        case BRAIN_CYTOKINE_IL10:
            /* IL-10 is anti-inflammatory, inversely related */
            return (1.0f - base_level) * 0.5f;
        default:
            return 0.0f;
    }
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int mental_health_immune_default_config(mental_health_immune_config_t* config) {
    if (!config) return -1;

    /* All features enabled by default */
    config->enable_cytokine_disorder_modulation = true;
    config->enable_inflammation_depression = true;
    config->enable_inflammation_anxiety = true;
    config->enable_disorder_immune_trigger = true;
    config->enable_recovery_immune_boost = true;
    config->enable_neurotransmitter_modulation = true;

    /* Biologically-based default sensitivities */
    config->cytokine_sensitivity = 1.0f;
    config->inflammation_sensitivity = 1.0f;
    config->disorder_trigger_sensitivity = 1.0f;

    /* Evidence-based thresholds */
    config->depression_trigger_threshold = DEPRESSION_IMMUNE_TRIGGER_THRESHOLD;
    config->anxiety_trigger_threshold = ANXIETY_IMMUNE_TRIGGER_THRESHOLD;
    config->inflammation_depression_threshold = INFLAMMATION_DEPRESSION_THRESHOLD;
    config->inflammation_anxiety_threshold = INFLAMMATION_ANXIETY_THRESHOLD;

    return 0;
}

mental_health_immune_bridge_t* mental_health_immune_bridge_create(
    const mental_health_immune_config_t* config,
    brain_immune_system_t* immune_system,
    mental_health_monitor_t* mental_health_monitor
) {
    /* Guard: require immune and mental health systems */
    if (!immune_system || !mental_health_monitor) {
        LOG_MODULE_ERROR("mental_health_immune_bridge",
                  "Cannot create bridge without immune and mental health systems");
        return NULL;
    }

    /* Allocate bridge */
    mental_health_immune_bridge_t* bridge = (mental_health_immune_bridge_t*)
        nimcp_malloc(sizeof(mental_health_immune_bridge_t));
    if (!bridge) {
        LOG_MODULE_ERROR("mental_health_immune_bridge", "Allocation failed");
        return NULL;
    }

    /* Initialize to zero */
    memset(bridge, 0, sizeof(mental_health_immune_bridge_t));

    /* Link systems */
    bridge->immune_system = immune_system;
    bridge->mental_health_monitor = mental_health_monitor;

    /* Apply configuration */
    if (config) {
        bridge->enable_cytokine_disorder_modulation = config->enable_cytokine_disorder_modulation;
        bridge->enable_inflammation_depression = config->enable_inflammation_depression;
        bridge->enable_inflammation_anxiety = config->enable_inflammation_anxiety;
        bridge->enable_disorder_immune_trigger = config->enable_disorder_immune_trigger;
        bridge->enable_recovery_immune_boost = config->enable_recovery_immune_boost;
        bridge->enable_neurotransmitter_modulation = config->enable_neurotransmitter_modulation;
    } else {
        /* Use defaults */
        mental_health_immune_config_t default_cfg;
        mental_health_immune_default_config(&default_cfg);
        bridge->enable_cytokine_disorder_modulation = default_cfg.enable_cytokine_disorder_modulation;
        bridge->enable_inflammation_depression = default_cfg.enable_inflammation_depression;
        bridge->enable_inflammation_anxiety = default_cfg.enable_inflammation_anxiety;
        bridge->enable_disorder_immune_trigger = default_cfg.enable_disorder_immune_trigger;
        bridge->enable_recovery_immune_boost = default_cfg.enable_recovery_immune_boost;
        bridge->enable_neurotransmitter_modulation = default_cfg.enable_neurotransmitter_modulation;
    }

    /* Create mutex */
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        nimcp_free(bridge);    return NULL;
    }

    LOG_MODULE_INFO("mental_health_immune_bridge", "Bridge created successfully");
    return bridge;
}

void mental_health_immune_bridge_destroy(mental_health_immune_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge->base.mutex);
    }

    /* Free bridge (don't destroy linked systems - we don't own them) */
    nimcp_free(bridge);
    LOG_MODULE_INFO("mental_health_immune_bridge", "Bridge destroyed");
}

/* ============================================================================
 * Immune → Mental Health Implementation
 * ============================================================================ */

int mental_health_immune_apply_cytokine_effects(mental_health_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_cytokine_disorder_modulation) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->base.mutex);

    /* Compute cytokine effects */
    cytokine_mental_health_effects_t* effects = &bridge->cytokine_effects;

    /* Estimate cytokine levels from immune state */
    float il1_level = estimate_cytokine_level(bridge->immune_system, BRAIN_CYTOKINE_IL1);
    float il6_level = estimate_cytokine_level(bridge->immune_system, BRAIN_CYTOKINE_IL6);
    float tnf_level = estimate_cytokine_level(bridge->immune_system, BRAIN_CYTOKINE_TNF);
    float ifn_gamma_level = estimate_cytokine_level(bridge->immune_system, BRAIN_CYTOKINE_IFN_GAMMA);
    float il10_level = estimate_cytokine_level(bridge->immune_system, BRAIN_CYTOKINE_IL10);

    /* Pro-inflammatory cytokines → depression risk */
    effects->il1_depression_risk = il1_level * CYTOKINE_IL1_DEPRESSION_RISK;
    effects->il6_depression_risk = il6_level * CYTOKINE_IL6_DEPRESSION_RISK;
    effects->tnf_depression_risk = tnf_level * CYTOKINE_TNF_DEPRESSION_RISK;
    effects->ifn_gamma_depression_risk = ifn_gamma_level * CYTOKINE_IFN_GAMMA_DEPRESSION_RISK;

    /* Pro-inflammatory cytokines → anxiety risk */
    effects->il1_anxiety_risk = il1_level * CYTOKINE_IL1_ANXIETY_RISK;
    effects->il6_anxiety_risk = il6_level * CYTOKINE_IL6_ANXIETY_RISK;
    effects->tnf_anxiety_risk = tnf_level * CYTOKINE_TNF_ANXIETY_RISK;

    /* Anti-inflammatory cytokines → recovery */
    effects->il10_recovery_benefit = il10_level * CYTOKINE_IL10_RECOVERY_BENEFIT;

    /* Aggregate effects */
    effects->total_depression_risk_shift =
        effects->il1_depression_risk +
        effects->il6_depression_risk +
        effects->tnf_depression_risk +
        effects->ifn_gamma_depression_risk +
        effects->il10_recovery_benefit;

    effects->total_anxiety_risk_shift =
        effects->il1_anxiety_risk +
        effects->il6_anxiety_risk +
        effects->tnf_anxiety_risk +
        effects->il10_recovery_benefit;

    /* Neurotransmitter suppression */
    float proinflam_total = il1_level + il6_level + tnf_level;
    effects->neurotransmitter_suppression = clamp_f(proinflam_total * 0.3f, 0.0f, 1.0f);

    bridge->cytokine_modulations++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->base.mutex);
    return 0;
}

int mental_health_immune_apply_inflammation_effects(mental_health_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->base.mutex);

    inflammation_mental_health_state_t* state = &bridge->inflammation_state;

    /* Get inflammation state */
    state->current_level = get_max_inflammation_level(bridge->immune_system);
    state->inflammation_duration_sec = get_inflammation_duration_sec(bridge->immune_system);
    state->is_chronic = (state->inflammation_duration_sec >= CHRONIC_MH_INFLAMMATION_THRESHOLD);

    /* Map inflammation level to intensity */
    float inflammation_intensity = (float)state->current_level / (float)INFLAMMATION_STORM;

    /* Depression risk multiplier (chronic inflammation is major risk factor) */
    if (bridge->enable_inflammation_depression) {
        if (state->is_chronic && inflammation_intensity >= INFLAMMATION_DEPRESSION_THRESHOLD) {
            /* Chronic inflammation → high depression risk */
            float duration_factor = clamp_f(
                state->inflammation_duration_sec / (CHRONIC_MH_INFLAMMATION_THRESHOLD * 2.0f),
                0.0f, 1.0f
            );
            state->depression_risk_multiplier = 1.0f + (inflammation_intensity * 2.0f * duration_factor);
        } else if (inflammation_intensity >= INFLAMMATION_DEPRESSION_THRESHOLD) {
            /* Acute inflammation → moderate depression risk */
            state->depression_risk_multiplier = 1.0f + (inflammation_intensity * 1.0f);
        } else {
            state->depression_risk_multiplier = 1.0f;
        }
    } else {
        state->depression_risk_multiplier = 1.0f;
    }

    /* Anxiety risk multiplier */
    if (bridge->enable_inflammation_anxiety) {
        if (inflammation_intensity >= INFLAMMATION_ANXIETY_THRESHOLD) {
            state->anxiety_risk_multiplier = 1.0f + (inflammation_intensity * 1.5f);
        } else {
            state->anxiety_risk_multiplier = 1.0f;
        }
    } else {
        state->anxiety_risk_multiplier = 1.0f;
    }

    /* Psychosis risk from cytokine storm */
    if (state->current_level == INFLAMMATION_STORM) {
        state->psychosis_risk = clamp_f(inflammation_intensity * 0.7f, 0.0f, 1.0f);
    } else {
        state->psychosis_risk = 0.0f;
    }

    /* Cognitive impairment */
    state->cognitive_impairment = clamp_f(inflammation_intensity * 0.6f, 0.0f, 1.0f);

    /* Neurotransmitter suppression */
    state->serotonin_suppression = clamp_f(inflammation_intensity * 0.5f, 0.0f, 1.0f);
    state->dopamine_suppression = clamp_f(inflammation_intensity * 0.4f, 0.0f, 1.0f);

    pthread_mutex_unlock((pthread_mutex_t*)bridge->base.mutex);
    return 0;
}

int mental_health_immune_modulate_neurotransmitters(mental_health_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_neurotransmitter_modulation) return 0;

    /* Neurotransmitter modulation would integrate with neuromodulator system */
    /* For now, just track the suppression levels in our state */

    float total_suppression = bridge->cytokine_effects.neurotransmitter_suppression;
    if (bridge->inflammation_state.serotonin_suppression > total_suppression) {
        total_suppression = bridge->inflammation_state.serotonin_suppression;
    }

    /* Would apply suppression to actual neuromodulator levels here */
    /* This would require integration with nimcp_neuromodulators.h */

    return 0;
}

float mental_health_immune_compute_depression_risk(const mental_health_immune_bridge_t* bridge) {
    if (!bridge) return 1.0f;

    return bridge->inflammation_state.depression_risk_multiplier;
}

float mental_health_immune_compute_anxiety_risk(const mental_health_immune_bridge_t* bridge) {
    if (!bridge) return 1.0f;

    return bridge->inflammation_state.anxiety_risk_multiplier;
}

/* ============================================================================
 * Mental Health → Immune Implementation
 * ============================================================================ */

int mental_health_immune_trigger_from_depression(mental_health_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_disorder_immune_trigger) return 0;
    if (!bridge->mental_health_monitor || !bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->base.mutex);

    mental_disorder_immune_trigger_t* trigger = &bridge->disorder_trigger;

    /* Get depression severity from mental health monitor */
    /* Note: Would need actual API to query disorder scores */
    /* For now, use placeholder */
    trigger->depression_severity = 0.0f; /* Would query mental_health_check_specific() */

    /* High depression triggers immune response */
    if (trigger->depression_severity >= DEPRESSION_IMMUNE_TRIGGER_THRESHOLD) {
        trigger->depression_triggered = true;

        /* Depression → increased pro-inflammatory cytokines */
        /* Release IL-6 and TNF-α */
        uint32_t cytokine_id;
        brain_immune_release_cytokine(bridge->immune_system,
                                     BRAIN_CYTOKINE_IL6,
                                     0, /* source cell */
                                     trigger->depression_severity * 0.6f,
                                     0, /* broadcast */
                                     &cytokine_id);

        brain_immune_release_cytokine(bridge->immune_system,
                                     BRAIN_CYTOKINE_TNF,
                                     0,
                                     trigger->depression_severity * 0.5f,
                                     0,
                                     &cytokine_id);

        bridge->depression_triggers++;
        bridge->disorder_triggered_responses++;
    } else {
        trigger->depression_triggered = false;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->base.mutex);
    return 0;
}

int mental_health_immune_trigger_from_anxiety(mental_health_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_disorder_immune_trigger) return 0;
    if (!bridge->mental_health_monitor || !bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->base.mutex);

    mental_disorder_immune_trigger_t* trigger = &bridge->disorder_trigger;

    /* Get anxiety severity */
    trigger->anxiety_severity = 0.0f; /* Would query mental_health_check_specific() */

    /* High anxiety triggers HPA axis and immune response */
    if (trigger->anxiety_severity >= ANXIETY_IMMUNE_TRIGGER_THRESHOLD) {
        trigger->anxiety_triggered = true;

        /* Anxiety → HPA axis activation → cortisol → initial immune suppression */
        trigger->cortisol_activation = clamp_f((trigger->anxiety_severity - 0.5f) * 2.0f, 0.0f, 1.0f);
        trigger->immune_suppression = trigger->cortisol_activation * 0.4f;

        /* Followed by inflammatory rebound */
        trigger->inflammatory_rebound = trigger->anxiety_severity * 0.5f;

        /* Release inflammatory cytokines (rebound effect) */
        uint32_t cytokine_id;
        brain_immune_release_cytokine(bridge->immune_system,
                                     BRAIN_CYTOKINE_IL6,
                                     0,
                                     trigger->inflammatory_rebound,
                                     0,
                                     &cytokine_id);

        bridge->anxiety_triggers++;
        bridge->disorder_triggered_responses++;
    } else {
        trigger->anxiety_triggered = false;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->base.mutex);
    return 0;
}

int mental_health_immune_trigger_from_ptsd(mental_health_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_disorder_immune_trigger) return 0;
    if (!bridge->mental_health_monitor || !bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->base.mutex);

    mental_disorder_immune_trigger_t* trigger = &bridge->disorder_trigger;

    /* Get PTSD severity */
    trigger->ptsd_severity = 0.0f; /* Would query mental_health_check_specific() */

    /* High PTSD triggers chronic inflammation */
    if (trigger->ptsd_severity >= PTSD_IMMUNE_TRIGGER_THRESHOLD) {
        trigger->ptsd_triggered = true;

        /* PTSD → chronic inflammatory state */
        trigger->chronic_inflammation_level = trigger->ptsd_severity * 0.7f;

        /* Release multiple pro-inflammatory cytokines */
        uint32_t cytokine_id;
        brain_immune_release_cytokine(bridge->immune_system,
                                     BRAIN_CYTOKINE_IL1,
                                     0,
                                     trigger->chronic_inflammation_level,
                                     0,
                                     &cytokine_id);

        brain_immune_release_cytokine(bridge->immune_system,
                                     BRAIN_CYTOKINE_IL6,
                                     0,
                                     trigger->chronic_inflammation_level,
                                     0,
                                     &cytokine_id);

        bridge->ptsd_triggers++;
        bridge->disorder_triggered_responses++;
    } else {
        trigger->ptsd_triggered = false;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->base.mutex);
    return 0;
}

int mental_health_immune_boost_from_recovery(mental_health_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_recovery_immune_boost) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->base.mutex);

    recovery_immune_enhancement_t* boost = &bridge->recovery_boost;

    /* Check for recent intervention */
    /* Would need to query mental health monitor for intervention history */
    /* For now, assume no recent intervention */
    boost->recent_intervention = false;

    if (boost->recent_intervention && boost->intervention_effectiveness > 0.5f) {
        /* Successful intervention → immune normalization */
        boost->immune_normalization = boost->intervention_effectiveness * 0.6f;

        /* Release IL-10 (anti-inflammatory) */
        boost->il10_release_boost = RECOVERY_IL10_BOOST * boost->intervention_effectiveness;

        uint32_t cytokine_id;
        brain_immune_release_cytokine(bridge->immune_system,
                                     BRAIN_CYTOKINE_IL10,
                                     0,
                                     boost->il10_release_boost,
                                     0,
                                     &cytokine_id);

        /* Reduce inflammation */
        boost->inflammation_reduction = boost->intervention_effectiveness * 0.5f;

        /* Reduce stress response */
        boost->stress_response_reduction = boost->intervention_effectiveness * 0.4f;

        bridge->recovery_boosts++;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Update Implementation
 * ============================================================================ */

int mental_health_immune_bridge_update(
    mental_health_immune_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) return -1;

    (void)delta_ms; /* Unused for now */

    /* Apply all bidirectional effects */

    /* Immune → Mental Health */
    mental_health_immune_apply_cytokine_effects(bridge);
    mental_health_immune_apply_inflammation_effects(bridge);
    mental_health_immune_modulate_neurotransmitters(bridge);

    /* Mental Health → Immune */
    mental_health_immune_trigger_from_depression(bridge);
    mental_health_immune_trigger_from_anxiety(bridge);
    mental_health_immune_trigger_from_ptsd(bridge);
    mental_health_immune_boost_from_recovery(bridge);

    bridge->total_updates++;
    return 0;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int mental_health_immune_get_cytokine_effects(
    const mental_health_immune_bridge_t* bridge,
    cytokine_mental_health_effects_t* effects
) {
    if (!bridge || !effects) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->base.mutex);
    memcpy(effects, &bridge->cytokine_effects, sizeof(cytokine_mental_health_effects_t));
    pthread_mutex_unlock((pthread_mutex_t*)bridge->base.mutex);

    return 0;
}

int mental_health_immune_get_inflammation_state(
    const mental_health_immune_bridge_t* bridge,
    inflammation_mental_health_state_t* state
) {
    if (!bridge || !state) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->base.mutex);
    memcpy(state, &bridge->inflammation_state, sizeof(inflammation_mental_health_state_t));
    pthread_mutex_unlock((pthread_mutex_t*)bridge->base.mutex);

    return 0;
}

bool mental_health_immune_is_cytokine_depression(const mental_health_immune_bridge_t* bridge) {
    if (!bridge) return false;

    /* Cytokine-induced depression threshold */
    return bridge->cytokine_effects.total_depression_risk_shift >= 0.3f;
}

float mental_health_immune_get_neurotransmitter_suppression(const mental_health_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->cytokine_effects.neurotransmitter_suppression;
}

int mental_health_immune_get_stats(
    const mental_health_immune_bridge_t* bridge,
    uint64_t* total_updates,
    uint32_t* depression_triggers,
    uint32_t* anxiety_triggers
) {
    if (!bridge) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->base.mutex);

    if (total_updates) *total_updates = bridge->total_updates;
    if (depression_triggers) *depression_triggers = bridge->depression_triggers;
    if (anxiety_triggers) *anxiety_triggers = bridge->anxiety_triggers;

    pthread_mutex_unlock((pthread_mutex_t*)bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Bio-Async Integration Implementation
 * ============================================================================ */

#define MENTAL_HEALTH_IMMUNE_MODULE_NAME "mental_health_immune_bridge"

/**
 * @brief Connect bridge to bio-async router
 */
int mental_health_immune_connect_bio_async(mental_health_immune_bridge_t* bridge) {
    if (!bridge) return -1;
    if (bridge->base.bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_IMMUNE_MENTAL_HEALTH,
        .module_name = MENTAL_HEALTH_IMMUNE_MODULE_NAME,
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("mental_health_immune_bridge connected to bio-async router");
    } else {
        NIMCP_LOGGING_INFO("Bio-async router not available, skipping registration");
    }

    return 0;
}

/**
 * @brief Disconnect from bio-async router
 */
int mental_health_immune_disconnect_bio_async(mental_health_immune_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->base.bio_async_enabled) return 0;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
        bridge->base.bio_ctx = NULL;
    }
    bridge->base.bio_async_enabled = false;

    NIMCP_LOGGING_DEBUG("mental_health_immune_bridge disconnected from bio-async router");
    return 0;
}

/**
 * @brief Check if bio-async is connected
 */
bool mental_health_immune_is_bio_async_connected(const mental_health_immune_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->base.bio_async_enabled;
}
