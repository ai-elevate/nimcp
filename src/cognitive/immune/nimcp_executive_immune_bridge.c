/**
 * @file nimcp_executive_immune_bridge.c
 * @brief Executive Function-Immune System Integration Implementation
 * @version 1.0.0
 * @date 2025-12-11
 *
 * WHAT: Bidirectional coupling between brain immune and executive functions
 * WHY:  Biological realism - cytokines impair executive control, stress affects immunity
 * HOW:  Monitor cytokine levels to modulate executive capacity, monitor load to trigger immune responses
 */

#include "cognitive/immune/nimcp_executive_immune_bridge.h"
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
 * @brief Compute cognitive fog level from cytokines
 *
 * WHAT: Calculate overall cognitive fog intensity
 * WHY:  Cognitive fog is distinct syndrome from pro-inflammatory cytokines
 * HOW:  Weighted sum of IL-1, IL-6, TNF-α concentrations
 */
static float compute_cognitive_fog(const brain_immune_system_t* immune) {
    if (!immune) return 0.0f;

    /* Query cytokine concentrations from immune system */
    float il1_level = 0.0f;
    float il6_level = 0.0f;
    float tnf_level = 0.0f;

    /* Count active cytokines as proxy for concentration */
    if (immune->cytokines && immune->cytokine_count > 0) {
        uint32_t il1_count = 0, il6_count = 0, tnf_count = 0;
        for (size_t i = 0; i < immune->cytokine_count; i++) {
            if (!immune->cytokines[i].delivered) continue;

            switch (immune->cytokines[i].type) {
                case BRAIN_CYTOKINE_IL1:
                    il1_count++;
                    il1_level += immune->cytokines[i].concentration;
                    break;
                case BRAIN_CYTOKINE_IL6:
                    il6_count++;
                    il6_level += immune->cytokines[i].concentration;
                    break;
                case BRAIN_CYTOKINE_TNF:
                    tnf_count++;
                    tnf_level += immune->cytokines[i].concentration;
                    break;
                default:
                    break;
            }
        }
        /* Normalize by max count */
        il1_level = clamp_f(il1_level / 10.0f, 0.0f, 1.0f);
        il6_level = clamp_f(il6_level / 10.0f, 0.0f, 1.0f);
        tnf_level = clamp_f(tnf_level / 10.0f, 0.0f, 1.0f);
    }

    /* Cognitive fog is weighted combination (IL-6 strongest effect) */
    float fog = (il1_level * 0.3f) + (il6_level * 0.5f) + (tnf_level * 0.2f);
    return clamp_f(fog, 0.0f, 1.0f);
}

/**
 * @brief Get inflammation duration
 *
 * WHAT: Calculate how long inflammation has persisted
 * WHY:  Chronic inflammation (>4 hours) has different executive effects
 * HOW:  Find oldest active inflammation site, compute duration
 */
static float get_inflammation_duration_sec(const brain_immune_system_t* immune) {
    if (!immune || !immune->inflammation_sites || immune->inflammation_count == 0) {
        return 0.0f;
    }

    /* Find oldest inflammation site */
    uint64_t oldest_start = UINT64_MAX;
    for (size_t i = 0; i < immune->inflammation_count; i++) {
        if (immune->inflammation_sites[i].start_time < oldest_start) {
            oldest_start = immune->inflammation_sites[i].start_time;
        }
    }

    if (oldest_start == UINT64_MAX) return 0.0f;

    /* Calculate duration (simplified - would use actual time API) */
    uint64_t current_time = immune->start_time; /* Placeholder */
    return (float)(current_time - oldest_start) / 1000.0f; /* ms to sec */
}

/**
 * @brief Get current inflammation level
 *
 * WHAT: Get highest inflammation level in system
 * WHY:  Max inflammation determines executive impact
 * HOW:  Query immune system for max inflammation site level
 */
static brain_inflammation_level_t get_max_inflammation_level(
    const brain_immune_system_t* immune
) {
    if (!immune || !immune->inflammation_sites || immune->inflammation_count == 0) {
        return INFLAMMATION_NONE;
    }

    brain_inflammation_level_t max_level = INFLAMMATION_NONE;
    for (size_t i = 0; i < immune->inflammation_count; i++) {
        if (immune->inflammation_sites[i].level > max_level) {
            max_level = immune->inflammation_sites[i].level;
        }
    }

    return max_level;
}

/**
 * @brief Get inflammation level as normalized float
 *
 * WHAT: Convert inflammation enum to [0-1] scale
 * WHY:  Easier computation for modulation
 * HOW:  Map INFLAMMATION_NONE=0.0 to INFLAMMATION_STORM=1.0
 */
static float inflammation_level_to_float(brain_inflammation_level_t level) {
    return (float)level / (float)INFLAMMATION_STORM;
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int executive_immune_default_config(executive_immune_config_t* config) {
    if (!config) return -1;

    /* All features enabled by default */
    config->enable_cytokine_executive_modulation = true;
    config->enable_inflammation_impairment = true;
    config->enable_executive_immune_trigger = true;
    config->enable_success_immune_boost = true;
    config->enable_overload_monitoring = true;

    /* Biologically-based default sensitivities */
    config->cytokine_sensitivity = 1.0f;
    config->inflammation_sensitivity = 1.0f;
    config->overload_trigger_sensitivity = 1.0f;

    /* Evidence-based thresholds */
    config->overload_trigger_threshold = EXECUTIVE_OVERLOAD_THRESHOLD;
    config->burnout_threshold = EXECUTIVE_BURNOUT_THRESHOLD;
    config->capacity_floor = INFLAMMATION_CAPACITY_FLOOR;

    return 0;
}

executive_immune_bridge_t* executive_immune_bridge_create(
    const executive_immune_config_t* config,
    brain_immune_system_t* immune_system,
    executive_controller_t* executive_controller
) {
    /* Guard: require immune and executive systems */
    if (!immune_system || !executive_controller) {
        nimcp_log(NIMCP_LOG_ERROR, "executive_immune_bridge",
                  "Cannot create bridge without immune and executive systems");
        return NULL;
    }

    /* Allocate bridge */
    executive_immune_bridge_t* bridge = (executive_immune_bridge_t*)
        nimcp_malloc(sizeof(executive_immune_bridge_t));
    if (!bridge) {
        nimcp_log(NIMCP_LOG_ERROR, "executive_immune_bridge", "Allocation failed");
        return NULL;
    }

    /* Initialize to zero */
    memset(bridge, 0, sizeof(executive_immune_bridge_t));

    /* Link systems */
    bridge->immune_system = immune_system;
    bridge->executive_controller = executive_controller;

    /* Apply configuration */
    if (config) {
        bridge->enable_cytokine_executive_modulation = config->enable_cytokine_executive_modulation;
        bridge->enable_inflammation_impairment = config->enable_inflammation_impairment;
        bridge->enable_executive_immune_trigger = config->enable_executive_immune_trigger;
        bridge->enable_success_immune_boost = config->enable_success_immune_boost;
        bridge->enable_overload_monitoring = config->enable_overload_monitoring;
    } else {
        /* Use defaults */
        executive_immune_config_t default_cfg;
        executive_immune_default_config(&default_cfg);
        bridge->enable_cytokine_executive_modulation = default_cfg.enable_cytokine_executive_modulation;
        bridge->enable_inflammation_impairment = default_cfg.enable_inflammation_impairment;
        bridge->enable_executive_immune_trigger = default_cfg.enable_executive_immune_trigger;
        bridge->enable_success_immune_boost = default_cfg.enable_success_immune_boost;
        bridge->enable_overload_monitoring = default_cfg.enable_overload_monitoring;
    }

    /* Create mutex */
    bridge->mutex = nimcp_malloc(sizeof(pthread_mutex_t));
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    pthread_mutex_init((pthread_mutex_t*)bridge->mutex, NULL);

    nimcp_log(NIMCP_LOG_INFO, "executive_immune_bridge", "Bridge created successfully");
    return bridge;
}

void executive_immune_bridge_destroy(executive_immune_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->mutex) {
        pthread_mutex_destroy((pthread_mutex_t*)bridge->mutex);
        nimcp_free(bridge->mutex);
    }

    /* Free bridge (don't destroy linked systems - we don't own them) */
    nimcp_free(bridge);
    nimcp_log(NIMCP_LOG_INFO, "executive_immune_bridge", "Bridge destroyed");
}

/* ============================================================================
 * Immune → Executive Implementation
 * ============================================================================ */

int executive_immune_apply_cytokine_effects(executive_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_cytokine_executive_modulation) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Compute cytokine effects */
    cytokine_executive_effects_t* effects = &bridge->cytokine_effects;

    /* Pro-inflammatory cytokines → capacity reduction */
    /* Query actual cytokine levels from immune system */
    float il1_level = 0.0f;
    float il6_level = 0.0f;
    float tnf_level = 0.0f;
    float ifn_gamma_level = 0.0f;
    float il10_level = 0.0f;

    if (bridge->immune_system->cytokines) {
        for (size_t i = 0; i < bridge->immune_system->cytokine_count; i++) {
            if (!bridge->immune_system->cytokines[i].delivered) continue;

            float conc = bridge->immune_system->cytokines[i].concentration;
            switch (bridge->immune_system->cytokines[i].type) {
                case BRAIN_CYTOKINE_IL1:
                    il1_level += conc;
                    break;
                case BRAIN_CYTOKINE_IL6:
                    il6_level += conc;
                    break;
                case BRAIN_CYTOKINE_TNF:
                    tnf_level += conc;
                    break;
                case BRAIN_CYTOKINE_IFN_GAMMA:
                    ifn_gamma_level += conc;
                    break;
                case BRAIN_CYTOKINE_IL10:
                    il10_level += conc;
                    break;
                default:
                    break;
            }
        }
        /* Normalize */
        il1_level = clamp_f(il1_level / 5.0f, 0.0f, 1.0f);
        il6_level = clamp_f(il6_level / 5.0f, 0.0f, 1.0f);
        tnf_level = clamp_f(tnf_level / 5.0f, 0.0f, 1.0f);
        ifn_gamma_level = clamp_f(ifn_gamma_level / 5.0f, 0.0f, 1.0f);
        il10_level = clamp_f(il10_level / 5.0f, 0.0f, 1.0f);
    }

    effects->il1_capacity_reduction = il1_level * CYTOKINE_IL1_CAPACITY_IMPACT;
    effects->il6_capacity_reduction = il6_level * CYTOKINE_IL6_CAPACITY_IMPACT;
    effects->tnf_capacity_reduction = tnf_level * CYTOKINE_TNF_CAPACITY_IMPACT;
    effects->ifn_gamma_capacity_reduction = ifn_gamma_level * CYTOKINE_IFN_GAMMA_CAPACITY_IMPACT;

    /* Anti-inflammatory cytokines → recovery */
    effects->il10_capacity_recovery = il10_level * CYTOKINE_IL10_CAPACITY_IMPACT;

    /* Aggregate effects */
    effects->total_capacity_impact =
        effects->il1_capacity_reduction +
        effects->il6_capacity_reduction +
        effects->tnf_capacity_reduction +
        effects->ifn_gamma_capacity_reduction +
        effects->il10_capacity_recovery;

    /* Cognitive fog */
    effects->cognitive_fog_level = compute_cognitive_fog(bridge->immune_system);

    /* Processing slowdown (IL-6 strongest effect) */
    effects->processing_slowdown = clamp_f(il6_level * 0.8f, 0.0f, 1.0f);

    /* Flexibility impairment (set-shifting) */
    float proinflam_total = fabs(effects->il1_capacity_reduction) +
                           fabs(effects->il6_capacity_reduction) +
                           fabs(effects->tnf_capacity_reduction);
    effects->flexibility_impairment = clamp_f(proinflam_total * 0.7f, 0.0f, 1.0f);

    bridge->cytokine_modulations++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int executive_immune_apply_inflammation_effects(executive_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_inflammation_impairment) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    inflammation_executive_state_t* state = &bridge->inflammation_state;

    /* Get inflammation state */
    state->current_level = get_max_inflammation_level(bridge->immune_system);
    state->inflammation_duration_sec = get_inflammation_duration_sec(bridge->immune_system);
    state->is_chronic = (state->inflammation_duration_sec >= CHRONIC_OVERLOAD_THRESHOLD);

    /* Inflammation intensity [0-1] */
    float inflammation_intensity = inflammation_level_to_float(state->current_level);

    /* Capacity reduction scales with inflammation */
    state->capacity_reduction = clamp_f(inflammation_intensity * 0.9f, 0.0f, 0.9f);

    /* Switch cost increases (perseveration) */
    state->switch_cost_increase = 1.0f + (inflammation_intensity * INFLAMMATION_SWITCH_COST_MULT);

    /* Inhibition impairment */
    state->inhibition_impairment = inflammation_intensity * INFLAMMATION_INHIBITION_PENALTY;

    /* Planning depth reduction */
    state->planning_depth_reduction = inflammation_intensity * INFLAMMATION_PLANNING_REDUCTION;

    /* Working memory impairment (7±2 → 3-4 items) */
    state->working_memory_impairment = clamp_f(inflammation_intensity * 0.6f, 0.0f, 0.6f);

    /* Perseveration increase */
    state->perseveration_increase = clamp_f(inflammation_intensity * 0.8f, 0.0f, 1.0f);

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

float executive_immune_compute_capacity_reduction(const executive_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;

    /* Combine cytokine-induced and inflammation-induced reduction */
    float cytokine_reduction = -bridge->cytokine_effects.total_capacity_impact;
    cytokine_reduction = clamp_f(cytokine_reduction, 0.0f, 1.0f);

    float inflammation_reduction = bridge->inflammation_state.capacity_reduction;

    /* Take maximum (not additive - they're correlated) */
    float total_reduction = fmaxf(cytokine_reduction, inflammation_reduction);

    /* Apply floor */
    total_reduction = clamp_f(total_reduction, 0.0f, 1.0f - INFLAMMATION_CAPACITY_FLOOR);

    return total_reduction;
}

float executive_immune_compute_switch_cost_increase(const executive_immune_bridge_t* bridge) {
    if (!bridge) return 1.0f; /* No increase */

    /* Use inflammation state switch cost multiplier */
    return bridge->inflammation_state.switch_cost_increase;
}

float executive_immune_compute_inhibition_impairment(const executive_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;

    /* Use inflammation state inhibition impairment */
    return bridge->inflammation_state.inhibition_impairment;
}

float executive_immune_compute_planning_reduction(const executive_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;

    /* Use inflammation state planning depth reduction */
    return bridge->inflammation_state.planning_depth_reduction;
}

/* ============================================================================
 * Executive → Immune Implementation
 * ============================================================================ */

int executive_immune_trigger_from_overload(executive_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_executive_immune_trigger) return 0;
    if (!bridge->immune_system || !bridge->executive_controller) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    executive_immune_trigger_t* trigger = &bridge->executive_trigger;

    /* Get current cognitive load */
    trigger->cognitive_load = executive_get_cognitive_load(bridge->executive_controller);

    /* Check if overload threshold exceeded */
    if (trigger->cognitive_load >= EXECUTIVE_OVERLOAD_THRESHOLD) {
        /* Trigger immune response (IL-6 release) */
        uint32_t cytokine_id = 0;
        brain_immune_release_cytokine(
            bridge->immune_system,
            BRAIN_CYTOKINE_IL6,
            0, /* source cell (executive system) */
            trigger->cognitive_load, /* concentration based on load */
            0, /* broadcast */
            &cytokine_id
        );

        trigger->cortisol_triggered = true;
        trigger->immune_suppression = clamp_f(trigger->cognitive_load * 0.5f, 0.0f, 1.0f);

        bridge->executive_triggered_responses++;
        bridge->overload_events++;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int executive_immune_amplify_from_frustration(executive_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_executive_immune_trigger) return 0;
    if (!bridge->immune_system || !bridge->executive_controller) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    executive_immune_trigger_t* trigger = &bridge->executive_trigger;

    /* Get executive stats to count failed tasks */
    executive_stats_t stats;
    if (!executive_get_stats(bridge->executive_controller, &stats)) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return -1;
    }

    /* Calculate recent failures (delta from previous) */
    uint32_t recent_failures = stats.failed_tasks - bridge->prev_failed_tasks;
    bridge->prev_failed_tasks = stats.failed_tasks;

    trigger->failed_tasks = recent_failures;

    /* Frustration amplifies inflammation */
    if (recent_failures > 0 && bridge->immune_system->inflammation_count > 0) {
        /* Escalate existing inflammation sites */
        for (size_t i = 0; i < bridge->immune_system->inflammation_count; i++) {
            brain_immune_escalate_inflammation(bridge->immune_system,
                bridge->immune_system->inflammation_sites[i].id);
        }
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int executive_immune_detect_burnout(executive_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_overload_monitoring) return 0;
    if (!bridge->immune_system || !bridge->executive_controller) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    executive_immune_trigger_t* trigger = &bridge->executive_trigger;

    /* Get current cognitive load */
    trigger->cognitive_load = executive_get_cognitive_load(bridge->executive_controller);

    /* Track overload duration */
    if (trigger->cognitive_load >= EXECUTIVE_BURNOUT_THRESHOLD) {
        /* Would increment overload duration based on delta_ms in update() */
        trigger->chronic_overload = (trigger->overload_duration_sec >= CHRONIC_OVERLOAD_THRESHOLD);

        if (trigger->chronic_overload) {
            /* Calculate burnout level */
            float duration_factor = clamp_f(
                trigger->overload_duration_sec / (CHRONIC_OVERLOAD_THRESHOLD * 2.0f),
                0.0f, 1.0f
            );
            trigger->burnout_level = clamp_f(duration_factor * 0.9f, 0.0f, 1.0f);
            trigger->immune_dysregulation = trigger->burnout_level;

            /* Trigger systemic inflammation */
            if (trigger->burnout_level > 0.7f) {
                uint32_t site_id = 0;
                uint32_t antigen_id = 0;
                uint8_t epitope[64] = "burnout_stress";

                brain_immune_present_antigen(bridge->immune_system, ANTIGEN_SOURCE_MANUAL,
                    epitope, strlen((const char*)epitope), 8, 0, &antigen_id);
                brain_immune_initiate_inflammation(bridge->immune_system, 0, antigen_id, &site_id);

                bridge->burnout_events++;
            }
        }
    } else {
        /* Reset overload duration if load is low */
        trigger->overload_duration_sec = 0.0f;
        trigger->chronic_overload = false;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int executive_immune_boost_from_success(executive_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_success_immune_boost) return 0;
    if (!bridge->immune_system || !bridge->executive_controller) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    executive_success_immune_boost_t* boost = &bridge->success_boost;

    /* Get executive stats */
    executive_stats_t stats;
    if (!executive_get_stats(bridge->executive_controller, &stats)) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return -1;
    }

    boost->completed_tasks = stats.completed_tasks;

    /* Calculate success rate */
    uint32_t total_tasks = stats.completed_tasks + stats.failed_tasks;
    if (total_tasks > 0) {
        boost->success_rate = (float)stats.completed_tasks / (float)total_tasks;
    } else {
        boost->success_rate = 0.0f;
    }

    /* High success rate → immune boost */
    if (boost->success_rate > 0.7f) {
        boost->immune_enhancement = clamp_f(boost->success_rate * 0.5f, 0.0f, 1.0f);
        boost->il10_release_boost = boost->success_rate * 0.3f;

        /* Release IL-10 (anti-inflammatory) */
        uint32_t cytokine_id = 0;
        brain_immune_release_cytokine(
            bridge->immune_system,
            BRAIN_CYTOKINE_IL10,
            0, /* source cell */
            boost->il10_release_boost,
            0, /* broadcast */
            &cytokine_id
        );

        /* Reduce inflammation */
        boost->inflammation_reduction = boost->success_rate * 0.4f;
        boost->stress_recovery = boost->success_rate * 0.6f;

        bridge->success_boosts++;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

/* ============================================================================
 * Bidirectional Update Implementation
 * ============================================================================ */

int executive_immune_bridge_update(
    executive_immune_bridge_t* bridge,
    uint64_t delta_ms
) {
    /* Guard clauses */
    if (!bridge) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Update overload duration tracking */
    float delta_sec = (float)delta_ms / 1000.0f;
    if (bridge->executive_trigger.cognitive_load >= EXECUTIVE_BURNOUT_THRESHOLD) {
        bridge->executive_trigger.overload_duration_sec += delta_sec;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    /* Immune → Executive: Apply cytokine and inflammation effects */
    executive_immune_apply_cytokine_effects(bridge);
    executive_immune_apply_inflammation_effects(bridge);

    /* Executive → Immune: Check for overload, frustration, success */
    executive_immune_trigger_from_overload(bridge);
    executive_immune_amplify_from_frustration(bridge);
    executive_immune_detect_burnout(bridge);
    executive_immune_boost_from_success(bridge);

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    bridge->total_updates++;
    bridge->last_update_time += delta_ms;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    return 0;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int executive_immune_get_cytokine_effects(
    const executive_immune_bridge_t* bridge,
    cytokine_executive_effects_t* effects
) {
    if (!bridge || !effects) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    *effects = bridge->cytokine_effects;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    return 0;
}

int executive_immune_get_inflammation_state(
    const executive_immune_bridge_t* bridge,
    inflammation_executive_state_t* state
) {
    if (!bridge || !state) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    *state = bridge->inflammation_state;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    return 0;
}

bool executive_immune_is_cognitive_fog(const executive_immune_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->cytokine_effects.cognitive_fog_level > 0.5f;
}

float executive_immune_get_cognitive_fog_severity(const executive_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->cytokine_effects.cognitive_fog_level;
}

bool executive_immune_is_burnout(const executive_immune_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->executive_trigger.burnout_level > 0.5f;
}

float executive_immune_get_burnout_severity(const executive_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->executive_trigger.burnout_level;
}
