/**
 * @file nimcp_autobiographical_immune_bridge.c
 * @brief Autobiographical Memory-Immune System Integration Implementation
 * @version 1.0.0
 * @date 2025-12-11
 *
 * WHAT: Bidirectional coupling between brain immune and autobiographical memory
 * WHY:  Biological realism - cytokines impair encoding, inflammation affects hippocampus
 * HOW:  Monitor cytokine levels to modulate encoding, track sickness landmarks
 */

#include "cognitive/immune/nimcp_autobiographical_immune_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>

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
 * WHAT: Query specific cytokine level
 * WHY:  Need individual cytokine concentrations for encoding modulation
 * HOW:  Iterate through immune system cytokines, find matching type
 */
static float get_cytokine_concentration(
    const brain_immune_system_t* immune,
    brain_cytokine_type_t type
) {
    if (!immune) return 0.0f;

    /* Iterate through active cytokines */
    float total = 0.0f;
    for (size_t i = 0; i < immune->cytokine_count; i++) {
        if (immune->cytokines[i].type == type) {
            total += immune->cytokines[i].concentration;
        }
    }

    return clamp_f(total, 0.0f, 1.0f);
}

/**
 * @brief Get inflammation duration
 *
 * WHAT: Calculate how long inflammation has persisted
 * WHY:  Chronic inflammation (>7 days) has different memory effects
 * HOW:  Find oldest active inflammation site, compute duration
 */
static float get_inflammation_duration_sec(const brain_immune_system_t* immune) {
    if (!immune || immune->inflammation_count == 0) return 0.0f;

    /* Find oldest inflammation site */
    uint64_t oldest_time = immune->inflammation_sites[0].start_time;
    for (size_t i = 1; i < immune->inflammation_count; i++) {
        if (immune->inflammation_sites[i].start_time < oldest_time) {
            oldest_time = immune->inflammation_sites[i].start_time;
        }
    }

    /* Compute duration (simplified - would use actual timestamp) */
    uint64_t current_time = 0; /* Would get from system */
    if (current_time > oldest_time) {
        return (float)(current_time - oldest_time) / 1000.0f;
    }
    return 0.0f;
}

/**
 * @brief Get current inflammation level
 *
 * WHAT: Get highest inflammation level in system
 * WHY:  Max inflammation determines memory impact
 * HOW:  Query immune system for max inflammation site level
 */
static brain_inflammation_level_t get_max_inflammation_level(
    const brain_immune_system_t* immune
) {
    if (!immune || immune->inflammation_count == 0) {
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

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int autobio_immune_default_config(autobio_immune_config_t* config) {
    if (!config) return -1;

    /* All features enabled by default */
    config->enable_cytokine_encoding_modulation = true;
    config->enable_inflammation_consolidation_impairment = true;
    config->enable_sickness_landmark_creation = true;
    config->enable_trauma_memory_immune_trigger = true;
    config->enable_positive_memory_immune_boost = true;
    config->enable_rumination_tracking = true;

    /* Biologically-based default sensitivities */
    config->cytokine_sensitivity = 1.0f;
    config->inflammation_sensitivity = 1.0f;
    config->memory_trigger_sensitivity = 1.0f;

    /* Evidence-based thresholds */
    config->trauma_trigger_threshold = TRAUMA_MEMORY_IMMUNE_THRESHOLD;
    config->negative_stress_threshold = NEGATIVE_MEMORY_STRESS_THRESHOLD;

    /* Sickness landmark settings */
    config->max_sickness_landmarks = 100;

    return 0;
}

autobio_immune_bridge_t* autobio_immune_bridge_create(
    const autobio_immune_config_t* config,
    brain_immune_system_t* immune_system,
    autobiographical_memory_t* autobio_memory
) {
    /* Guard: require both systems */
    if (!immune_system || !autobio_memory) {
        nimcp_log(NIMCP_LOG_ERROR, "autobio_immune_bridge",
                  "Cannot create bridge without immune and autobio memory systems");
        return NULL;
    }

    /* Allocate bridge */
    autobio_immune_bridge_t* bridge = (autobio_immune_bridge_t*)
        nimcp_malloc(sizeof(autobio_immune_bridge_t));
    if (!bridge) {
        nimcp_log(NIMCP_LOG_ERROR, "autobio_immune_bridge", "Allocation failed");
        return NULL;
    }

    /* Initialize to zero */
    memset(bridge, 0, sizeof(autobio_immune_bridge_t));

    /* Link systems */
    bridge->immune_system = immune_system;
    bridge->autobio_memory = autobio_memory;

    /* Apply configuration */
    autobio_immune_config_t default_cfg;
    if (!config) {
        autobio_immune_default_config(&default_cfg);
        config = &default_cfg;
    }

    bridge->enable_cytokine_encoding_modulation = config->enable_cytokine_encoding_modulation;
    bridge->enable_inflammation_consolidation_impairment = config->enable_inflammation_consolidation_impairment;
    bridge->enable_sickness_landmark_creation = config->enable_sickness_landmark_creation;
    bridge->enable_trauma_memory_immune_trigger = config->enable_trauma_memory_immune_trigger;
    bridge->enable_positive_memory_immune_boost = config->enable_positive_memory_immune_boost;
    bridge->enable_rumination_tracking = config->enable_rumination_tracking;

    /* Allocate sickness landmarks array */
    bridge->sickness_landmark_capacity = config->max_sickness_landmarks;
    bridge->sickness_landmarks = (sickness_landmark_t*)
        nimcp_malloc(sizeof(sickness_landmark_t) * bridge->sickness_landmark_capacity);
    if (!bridge->sickness_landmarks) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Create mutex */
    bridge->mutex = nimcp_malloc(sizeof(pthread_mutex_t));
    if (!bridge->mutex) {
        nimcp_free(bridge->sickness_landmarks);
        nimcp_free(bridge);
        return NULL;
    }
    pthread_mutex_init((pthread_mutex_t*)bridge->mutex, NULL);

    nimcp_log(NIMCP_LOG_INFO, "autobio_immune_bridge", "Bridge created successfully");
    return bridge;
}

void autobio_immune_bridge_destroy(autobio_immune_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->mutex) {
        pthread_mutex_destroy((pthread_mutex_t*)bridge->mutex);
        nimcp_free(bridge->mutex);
    }

    /* Free sickness landmarks */
    if (bridge->sickness_landmarks) {
        nimcp_free(bridge->sickness_landmarks);
    }

    /* Free bridge (don't destroy linked systems - we don't own them) */
    nimcp_free(bridge);
    nimcp_log(NIMCP_LOG_INFO, "autobio_immune_bridge", "Bridge destroyed");
}

/* ============================================================================
 * Immune → Memory Implementation
 * ============================================================================ */

int autobio_immune_apply_cytokine_encoding_effects(autobio_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_cytokine_encoding_modulation) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Query cytokine concentrations */
    float il1 = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_IL1);
    float il6 = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_IL6);
    float tnf = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_TNF);
    float ifn_gamma = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_IFN_GAMMA);
    float il10 = get_cytokine_concentration(bridge->immune_system, BRAIN_CYTOKINE_IL10);

    /* Compute encoding impairments */
    bridge->cytokine_effects.il1_encoding_impairment = il1 * fabsf(CYTOKINE_IL1_ENCODING_IMPACT);
    bridge->cytokine_effects.il6_encoding_impairment = il6 * fabsf(CYTOKINE_IL6_ENCODING_IMPACT);
    bridge->cytokine_effects.tnf_encoding_impairment = tnf * fabsf(CYTOKINE_TNF_ENCODING_IMPACT);
    bridge->cytokine_effects.ifn_gamma_impairment = ifn_gamma * fabsf(CYTOKINE_IFN_GAMMA_ENCODING_IMPACT);
    bridge->cytokine_effects.il10_encoding_boost = il10 * CYTOKINE_IL10_ENCODING_BOOST;

    /* Total encoding modulation (1.0 = normal, <1.0 = impaired, >1.0 = boosted) */
    float impairment = bridge->cytokine_effects.il1_encoding_impairment +
                       bridge->cytokine_effects.il6_encoding_impairment +
                       bridge->cytokine_effects.tnf_encoding_impairment +
                       bridge->cytokine_effects.ifn_gamma_impairment;

    bridge->cytokine_effects.total_encoding_modulation =
        1.0f - impairment + bridge->cytokine_effects.il10_encoding_boost;
    bridge->cytokine_effects.total_encoding_modulation =
        clamp_f(bridge->cytokine_effects.total_encoding_modulation, 0.0f, 1.5f);

    /* Emotional salience modulation */
    float inflammation_level = (float)get_max_inflammation_level(bridge->immune_system) /
                               (float)INFLAMMATION_STORM;
    bridge->cytokine_effects.negative_salience_boost =
        inflammation_level * INFLAMMATION_NEGATIVE_SALIENCE_BOOST;
    bridge->cytokine_effects.positive_salience_reduction =
        inflammation_level * INFLAMMATION_POSITIVE_SALIENCE_REDUCE;

    /* Consolidation impairment */
    bridge->cytokine_effects.consolidation_impairment = impairment;

    bridge->encoding_modulations++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    return 0;
}

int autobio_immune_apply_inflammation_consolidation_effects(
    autobio_immune_bridge_t* bridge
) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_inflammation_consolidation_impairment) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Get inflammation state */
    brain_inflammation_level_t level = get_max_inflammation_level(bridge->immune_system);
    float duration = get_inflammation_duration_sec(bridge->immune_system);

    bridge->inflammation_state.current_level = level;
    bridge->inflammation_state.inflammation_duration_sec = duration;
    bridge->inflammation_state.is_chronic =
        (duration >= CHRONIC_INFLAMMATION_MEMORY_THRESHOLD);

    /* Compute memory impacts based on inflammation level */
    float severity = (float)level / (float)INFLAMMATION_STORM;

    /* Encoding efficiency decreases with inflammation */
    bridge->inflammation_state.encoding_efficiency =
        1.0f - (severity * 0.5f);

    /* Consolidation quality impaired */
    bridge->inflammation_state.consolidation_quality =
        1.0f - (severity * 0.6f);

    /* Retrieval accuracy slightly impaired */
    bridge->inflammation_state.retrieval_accuracy =
        1.0f - (severity * 0.3f);

    /* False memory risk increases */
    bridge->inflammation_state.false_memory_risk = severity * 0.4f;

    /* Chronic inflammation accelerates decline */
    if (bridge->inflammation_state.is_chronic) {
        bridge->inflammation_state.memory_decline_rate = severity * 0.3f;
        bridge->inflammation_state.hippocampal_impairment = severity * 0.5f;
    } else {
        bridge->inflammation_state.memory_decline_rate = 0.0f;
        bridge->inflammation_state.hippocampal_impairment = severity * 0.2f;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

float autobio_immune_modulate_memory_salience(
    const autobio_immune_bridge_t* bridge,
    const autobiographical_memory_entry_t* memory
) {
    /* Guard clauses */
    if (!bridge || !memory) return 1.0f;
    if (!bridge->enable_cytokine_encoding_modulation) return 1.0f;

    float modulation = 1.0f;

    /* Enhance negative memories during inflammation */
    if (memory->valence < VALENCE_NEUTRAL) {
        modulation += bridge->cytokine_effects.negative_salience_boost;
    }
    /* Reduce positive memories during inflammation */
    else if (memory->valence > VALENCE_NEUTRAL) {
        modulation += bridge->cytokine_effects.positive_salience_reduction;
    }

    return clamp_f(modulation, 0.5f, 1.5f);
}

int autobio_immune_create_sickness_landmark(
    autobio_immune_bridge_t* bridge,
    brain_inflammation_level_t severity,
    uint64_t* landmark_id
) {
    /* Guard clauses */
    if (!bridge || !landmark_id) return -1;
    if (!bridge->enable_sickness_landmark_creation) return 0;
    if (!bridge->autobio_memory) return -1;
    if (severity < INFLAMMATION_SYSTEMIC) return 0; /* Only create for systemic+ */

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Check capacity */
    if (bridge->sickness_landmark_count >= bridge->sickness_landmark_capacity) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return -1;
    }

    /* Create autobiographical memory for sickness episode */
    autobiographical_memory_entry_t memory;
    memset(&memory, 0, sizeof(memory));

    memory.type = AUTOBIO_CRISIS;
    memory.valence = VALENCE_NEGATIVE;
    memory.emotional_intensity = (float)severity / (float)INFLAMMATION_STORM;
    memory.arousal = 0.7f; /* Sickness is moderate arousal */
    memory.importance = SICKNESS_LANDMARK_IMPORTANCE;
    memory.self_relevance = 1.0f; /* Highly self-relevant */
    memory.identity_defining = (severity == INFLAMMATION_STORM); /* Storm is identity-defining */

    snprintf(memory.what_happened, AUTOBIO_MAX_DESCRIPTION_LEN,
             "I experienced %s inflammation (sickness episode)",
             brain_immune_inflammation_to_string(severity));
    snprintf(memory.why_it_happened, AUTOBIO_MAX_REASONING_LEN,
             "Immune system activated against threat");
    snprintf(memory.outcome, AUTOBIO_MAX_OUTCOME_LEN,
             "Episode ongoing, experiencing sickness behavior");

    /* Store in autobiographical memory */
    uint64_t mem_id = autobio_store(bridge->autobio_memory, &memory);
    if (mem_id == 0) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return -1;
    }

    /* Create sickness landmark record */
    sickness_landmark_t* landmark =
        &bridge->sickness_landmarks[bridge->sickness_landmark_count];

    landmark->memory_id = mem_id;
    landmark->start_time_ms = 0; /* Would get from system */
    landmark->end_time_ms = 0; /* Ongoing */
    landmark->severity = severity;
    landmark->emotional_intensity = memory.emotional_intensity;
    landmark->identity_defining = memory.identity_defining;
    snprintf(landmark->description, sizeof(landmark->description),
             "%s", memory.what_happened);

    bridge->sickness_landmark_count++;
    bridge->active_sickness_landmark_id = (uint32_t)mem_id;
    bridge->sickness_landmarks_created++;

    *landmark_id = mem_id;

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    nimcp_log(NIMCP_LOG_INFO, "autobio_immune_bridge",
              "Created sickness landmark: %s", landmark->description);

    return 0;
}

int autobio_immune_close_sickness_landmark(
    autobio_immune_bridge_t* bridge,
    uint64_t landmark_id
) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->autobio_memory) return -1;
    if (landmark_id == 0) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Find landmark */
    sickness_landmark_t* landmark = NULL;
    for (uint32_t i = 0; i < bridge->sickness_landmark_count; i++) {
        if (bridge->sickness_landmarks[i].memory_id == landmark_id) {
            landmark = &bridge->sickness_landmarks[i];
            break;
        }
    }

    if (!landmark) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return -1;
    }

    /* Mark as ended */
    landmark->end_time_ms = 0; /* Would get from system */

    /* Update memory with outcome */
    autobiographical_memory_entry_t memory;
    if (autobio_retrieve(bridge->autobio_memory, landmark_id, &memory)) {
        snprintf(memory.outcome, AUTOBIO_MAX_OUTCOME_LEN,
                 "Recovered from %s inflammation, immune system resolved threat",
                 brain_immune_inflammation_to_string(landmark->severity));

        /* Update importance based on how it was resolved */
        memory.importance = clamp_f(memory.importance * 0.9f, 0.3f, 1.0f);
    }

    /* Clear active landmark if this was it */
    if (bridge->active_sickness_landmark_id == (uint32_t)landmark_id) {
        bridge->active_sickness_landmark_id = 0;
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    nimcp_log(NIMCP_LOG_INFO, "autobio_immune_bridge",
              "Closed sickness landmark: %llu", (unsigned long long)landmark_id);

    return 0;
}

float autobio_immune_get_encoding_efficiency(const autobio_immune_bridge_t* bridge) {
    if (!bridge) return 1.0f;
    return bridge->cytokine_effects.total_encoding_modulation;
}

/* ============================================================================
 * Memory → Immune Implementation
 * ============================================================================ */

int autobio_immune_trigger_from_trauma_recall(
    autobio_immune_bridge_t* bridge,
    const autobiographical_memory_entry_t* memory
) {
    /* Guard clauses */
    if (!bridge || !memory) return -1;
    if (!bridge->enable_trauma_memory_immune_trigger) return 0;
    if (!bridge->immune_system) return -1;

    /* Check if memory is traumatic enough to trigger immune */
    bool is_trauma = (memory->type == AUTOBIO_FAILURE ||
                      memory->type == AUTOBIO_CRISIS) &&
                     (memory->valence <= VALENCE_NEGATIVE) &&
                     (memory->importance >= TRAUMA_MEMORY_IMMUNE_THRESHOLD);

    if (!is_trauma) return 0;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Update trigger state */
    bridge->memory_trigger.memory_type = memory->type;
    bridge->memory_trigger.valence = memory->valence;
    bridge->memory_trigger.importance = memory->importance;
    bridge->memory_trigger.emotional_intensity = memory->emotional_intensity;
    bridge->memory_trigger.trauma_triggered = true;

    /* Compute cortisol and inflammatory response */
    bridge->memory_trigger.cortisol_release =
        memory->importance * memory->emotional_intensity;
    bridge->memory_trigger.inflammatory_response =
        bridge->memory_trigger.cortisol_release * 0.6f; /* Rebound after cortisol */

    bridge->memory_triggered_responses++;
    bridge->trauma_recalls++;

    /* Trigger immune system (simplified - would create antigen) */
    if (bridge->memory_trigger.inflammatory_response > 0.5f) {
        /* High trauma recall triggers immune activation */
        nimcp_log(NIMCP_LOG_DEBUG, "autobio_immune_bridge",
                  "Trauma recall triggered immune response: importance=%.2f",
                  memory->importance);
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int autobio_immune_ruminate_on_negative_memory(
    autobio_immune_bridge_t* bridge,
    uint64_t memory_id
) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_rumination_tracking) return 0;
    if (!bridge->autobio_memory) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Track rumination */
    bridge->memory_trigger.rumination_count++;
    bridge->memory_trigger.chronic_stress_active =
        (bridge->memory_trigger.rumination_count > 5);

    /* Chronic rumination escalates inflammation */
    if (bridge->memory_trigger.chronic_stress_active) {
        bridge->memory_trigger.inflammatory_response =
            clamp_f(bridge->memory_trigger.inflammatory_response * 1.2f, 0.0f, 1.0f);

        nimcp_log(NIMCP_LOG_DEBUG, "autobio_immune_bridge",
                  "Chronic rumination detected, escalating inflammation");
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int autobio_immune_boost_from_positive_memory(
    autobio_immune_bridge_t* bridge,
    const autobiographical_memory_entry_t* memory
) {
    /* Guard clauses */
    if (!bridge || !memory) return -1;
    if (!bridge->enable_positive_memory_immune_boost) return 0;
    if (!bridge->immune_system) return -1;

    /* Check if memory is positive enough to boost immune */
    bool is_positive = (memory->valence > VALENCE_NEUTRAL) &&
                       (memory->importance > 0.4f);

    if (!is_positive) return 0;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Track positive memory types */
    if (memory->type == AUTOBIO_ACHIEVEMENT) {
        bridge->positive_boost.achievement_count++;
    } else if (memory->type == AUTOBIO_LEARNING) {
        bridge->positive_boost.learning_count++;
    } else if (memory->type == AUTOBIO_INTERACTION) {
        bridge->positive_boost.social_bond_count++;
    }

    /* Compute immune benefits */
    bridge->positive_boost.positive_valence_avg =
        (float)(memory->valence - VALENCE_NEUTRAL) / 2.0f;
    bridge->positive_boost.immune_enhancement =
        memory->importance * bridge->positive_boost.positive_valence_avg * 0.3f;
    bridge->positive_boost.cortisol_reduction =
        bridge->positive_boost.immune_enhancement * 0.5f;
    bridge->positive_boost.il10_release_boost =
        bridge->positive_boost.immune_enhancement * 0.4f;
    bridge->positive_boost.resilience_factor =
        clamp_f(bridge->positive_boost.immune_enhancement, 0.0f, 1.0f);

    bridge->positive_boosts++;

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

bool autobio_immune_is_identity_threatening(
    const autobiographical_memory_entry_t* memory
) {
    if (!memory) return false;

    /* Identity-threatening: core memory + negative + high importance */
    return memory->is_core_memory &&
           (memory->valence <= VALENCE_NEGATIVE) &&
           (memory->importance >= 0.7f);
}

/* ============================================================================
 * Bidirectional Update Implementation
 * ============================================================================ */

int autobio_immune_bridge_update(
    autobio_immune_bridge_t* bridge,
    uint64_t delta_ms
) {
    /* Guard clauses */
    if (!bridge) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Update inflammation state */
    autobio_immune_apply_inflammation_consolidation_effects(bridge);

    /* Check if should create sickness landmark */
    if (bridge->enable_sickness_landmark_creation) {
        brain_inflammation_level_t level =
            get_max_inflammation_level(bridge->immune_system);

        /* Create landmark if systemic+ and no active landmark */
        if (level >= INFLAMMATION_SYSTEMIC &&
            bridge->active_sickness_landmark_id == 0) {
            uint64_t landmark_id;
            autobio_immune_create_sickness_landmark(bridge, level, &landmark_id);
        }
        /* Close landmark if inflammation resolved */
        else if (level < INFLAMMATION_REGIONAL &&
                 bridge->active_sickness_landmark_id != 0) {
            autobio_immune_close_sickness_landmark(
                bridge, bridge->active_sickness_landmark_id);
        }
    }

    /* Apply cytokine encoding effects */
    autobio_immune_apply_cytokine_encoding_effects(bridge);

    /* Update rumination duration */
    if (bridge->memory_trigger.chronic_stress_active) {
        bridge->memory_trigger.rumination_duration_sec += (float)delta_ms / 1000.0f;
    } else {
        bridge->memory_trigger.rumination_duration_sec = 0.0f;
    }

    bridge->total_updates++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    return 0;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int autobio_immune_get_cytokine_effects(
    const autobio_immune_bridge_t* bridge,
    cytokine_memory_effects_t* effects
) {
    if (!bridge || !effects) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    memcpy(effects, &bridge->cytokine_effects, sizeof(cytokine_memory_effects_t));
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    return 0;
}

int autobio_immune_get_inflammation_state(
    const autobio_immune_bridge_t* bridge,
    inflammation_memory_state_t* state
) {
    if (!bridge || !state) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    memcpy(state, &bridge->inflammation_state, sizeof(inflammation_memory_state_t));
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    return 0;
}

bool autobio_immune_is_sickness_affecting_memory(const autobio_immune_bridge_t* bridge) {
    if (!bridge) return false;

    return (bridge->inflammation_state.current_level >= INFLAMMATION_REGIONAL) &&
           (bridge->cytokine_effects.total_encoding_modulation < 0.8f);
}

int autobio_immune_get_sickness_landmarks(
    const autobio_immune_bridge_t* bridge,
    sickness_landmark_t* landmarks,
    uint32_t max_landmarks,
    uint32_t* num_found
) {
    if (!bridge || !landmarks || !num_found) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    uint32_t count = bridge->sickness_landmark_count;
    if (count > max_landmarks) count = max_landmarks;

    memcpy(landmarks, bridge->sickness_landmarks,
           count * sizeof(sickness_landmark_t));
    *num_found = count;

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

float autobio_immune_get_consolidation_impairment(
    const autobio_immune_bridge_t* bridge
) {
    if (!bridge) return 0.0f;
    return bridge->cytokine_effects.consolidation_impairment;
}

float autobio_immune_get_memory_decline_rate(const autobio_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->inflammation_state.memory_decline_rate;
}
