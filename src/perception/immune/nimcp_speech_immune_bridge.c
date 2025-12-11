/**
 * @file nimcp_speech_immune_bridge.c
 * @brief Speech Cortex-Immune System Integration Implementation
 * @version 1.0.0
 * @date 2025-12-11
 *
 * WHAT: Bidirectional coupling between brain immune and speech processing systems
 * WHY:  Biological realism - cytokines affect speech fluency, distress affects immunity
 * HOW:  Monitor cytokine levels to modulate speech, monitor speech to trigger immune responses
 */

#include "perception/immune/nimcp_speech_immune_bridge.h"
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
 * @brief Get inflammation duration from immune system
 *
 * WHAT: Calculate how long inflammation has persisted
 * WHY:  Chronic inflammation (>3 days) has different speech effects
 * HOW:  Find oldest active inflammation site, compute duration
 */
static float get_inflammation_duration_sec(const brain_immune_system_t* immune) {
    if (!immune) return 0.0f;

    /* Query immune system for oldest inflammation site */
    float max_duration = 0.0f;
    for (size_t i = 0; i < immune->inflammation_count; i++) {
        brain_inflammation_site_t* site = &immune->inflammation_sites[i];
        uint64_t current_time = 0; /* Would get from system time */
        float duration = (current_time - site->start_time) / 1000.0f;
        if (duration > max_duration) {
            max_duration = duration;
        }
    }
    return max_duration;
}

/**
 * @brief Get current inflammation level
 *
 * WHAT: Get highest inflammation level in system
 * WHY:  Max inflammation determines speech impact
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
 * @brief Check if word is illness-related
 *
 * WHAT: Determine if word indicates illness expression
 * WHY:  Illness words trigger immune modulation
 * HOW:  Simple keyword matching (could be more sophisticated)
 */
static bool is_illness_word(const char* word) {
    if (!word) return false;

    /* Common illness-related words */
    const char* illness_words[] = {
        "sick", "ill", "pain", "hurt", "ache", "tired", "fatigue",
        "weak", "dizzy", "nauseous", "fever", "cold", "flu",
        NULL
    };

    for (int i = 0; illness_words[i] != NULL; i++) {
        if (strcmp(word, illness_words[i]) == 0) {
            return true;
        }
    }
    return false;
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int speech_immune_default_config(speech_immune_config_t* config) {
    if (!config) return -1;

    /* All features enabled by default */
    config->enable_cytokine_speech_modulation = true;
    config->enable_inflammation_impairment = true;
    config->enable_speech_immune_trigger = true;
    config->enable_distress_vocalization_trigger = true;

    /* Biologically-based default sensitivities */
    config->cytokine_sensitivity = 1.0f;
    config->inflammation_sensitivity = 1.0f;
    config->speech_trigger_sensitivity = 1.0f;

    /* Evidence-based thresholds */
    config->effort_trigger_threshold = SPEECH_EFFORT_IMMUNE_TRIGGER;
    config->distress_threshold = DISTRESS_VOCALIZATION_THRESHOLD;

    return 0;
}

speech_immune_bridge_t* speech_immune_bridge_create(
    const speech_immune_config_t* config,
    brain_immune_system_t* immune_system,
    speech_cortex_t* speech_cortex
) {
    /* Guard: require immune and speech systems */
    if (!immune_system || !speech_cortex) {
        nimcp_log(NIMCP_LOG_ERROR, "speech_immune_bridge",
                  "Cannot create bridge without immune and speech systems");
        return NULL;
    }

    /* Allocate bridge */
    speech_immune_bridge_t* bridge = (speech_immune_bridge_t*)
        nimcp_malloc(sizeof(speech_immune_bridge_t));
    if (!bridge) {
        nimcp_log(NIMCP_LOG_ERROR, "speech_immune_bridge", "Allocation failed");
        return NULL;
    }

    /* Initialize to zero */
    memset(bridge, 0, sizeof(speech_immune_bridge_t));

    /* Link systems */
    bridge->immune_system = immune_system;
    bridge->speech_cortex = speech_cortex;

    /* Apply configuration */
    if (config) {
        bridge->enable_cytokine_speech_modulation = config->enable_cytokine_speech_modulation;
        bridge->enable_inflammation_impairment = config->enable_inflammation_impairment;
        bridge->enable_speech_immune_trigger = config->enable_speech_immune_trigger;
        bridge->enable_distress_vocalization_trigger = config->enable_distress_vocalization_trigger;
    } else {
        /* Use defaults */
        speech_immune_config_t default_cfg;
        speech_immune_default_config(&default_cfg);
        bridge->enable_cytokine_speech_modulation = default_cfg.enable_cytokine_speech_modulation;
        bridge->enable_inflammation_impairment = default_cfg.enable_inflammation_impairment;
        bridge->enable_speech_immune_trigger = default_cfg.enable_speech_immune_trigger;
        bridge->enable_distress_vocalization_trigger = default_cfg.enable_distress_vocalization_trigger;
    }

    /* Create mutex */
    bridge->mutex = nimcp_malloc(sizeof(pthread_mutex_t));
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    pthread_mutex_init((pthread_mutex_t*)bridge->mutex, NULL);

    nimcp_log(NIMCP_LOG_INFO, "speech_immune_bridge", "Bridge created successfully");
    return bridge;
}

void speech_immune_bridge_destroy(speech_immune_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->mutex) {
        pthread_mutex_destroy((pthread_mutex_t*)bridge->mutex);
        nimcp_free(bridge->mutex);
    }

    /* Free bridge (don't destroy linked systems - we don't own them) */
    nimcp_free(bridge);
    nimcp_log(NIMCP_LOG_INFO, "speech_immune_bridge", "Bridge destroyed");
}

/* ============================================================================
 * Immune → Speech Implementation
 * ============================================================================ */

int speech_immune_apply_cytokine_effects(speech_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_cytokine_speech_modulation) return 0;
    if (!bridge->immune_system || !bridge->speech_cortex) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Query cytokine levels from immune system */
    /* In a full implementation, we'd query actual cytokine concentrations */
    float il1_level = 0.0f;
    float il6_level = 0.0f;
    float tnf_level = 0.0f;
    float ifn_gamma_level = 0.0f;

    /* Iterate through active cytokines */
    for (size_t i = 0; i < bridge->immune_system->cytokine_count; i++) {
        brain_cytokine_t* cytokine = &bridge->immune_system->cytokines[i];
        if (!cytokine->delivered) continue;

        switch (cytokine->type) {
            case BRAIN_CYTOKINE_IL1:
                il1_level += cytokine->concentration;
                break;
            case BRAIN_CYTOKINE_IL6:
                il6_level += cytokine->concentration;
                break;
            case BRAIN_CYTOKINE_TNF:
                tnf_level += cytokine->concentration;
                break;
            case BRAIN_CYTOKINE_IFN_GAMMA:
                ifn_gamma_level += cytokine->concentration;
                break;
            default:
                break;
        }
    }

    /* Compute individual cytokine effects */
    bridge->cytokine_effects.il1_fluency_reduction =
        clamp_f(il1_level * CYTOKINE_IL1_FLUENCY_IMPACT, -1.0f, 0.0f);
    bridge->cytokine_effects.il6_word_retrieval_delay =
        clamp_f(il6_level * CYTOKINE_IL6_FLUENCY_IMPACT, -1.0f, 0.0f);
    bridge->cytokine_effects.tnf_phoneme_discrimination =
        clamp_f(tnf_level * CYTOKINE_TNF_FLUENCY_IMPACT, -1.0f, 0.0f);
    bridge->cytokine_effects.ifn_gamma_prosody_reduction =
        clamp_f(ifn_gamma_level * CYTOKINE_IFN_GAMMA_FLUENCY_IMPACT, -1.0f, 0.0f);

    /* Compute aggregate effects */
    bridge->cytokine_effects.total_fluency_impairment = clamp_f(
        -(bridge->cytokine_effects.il1_fluency_reduction +
          bridge->cytokine_effects.il6_word_retrieval_delay +
          bridge->cytokine_effects.tnf_phoneme_discrimination),
        0.0f, 1.0f
    );

    /* IL-6 specifically affects word retrieval latency (Marsland et al. 2006) */
    bridge->cytokine_effects.word_retrieval_latency_ms =
        il6_level * 200.0f; /* Up to 200ms added latency */

    /* TNF-α increases phoneme errors */
    bridge->cytokine_effects.phoneme_error_rate =
        clamp_f(tnf_level * 0.3f, 0.0f, 0.3f);

    /* Speech rate reduction from overall cytokine burden */
    bridge->cytokine_effects.speech_rate_reduction =
        bridge->cytokine_effects.total_fluency_impairment;

    /* Prosody flattening */
    bridge->cytokine_effects.prosody_flattening =
        clamp_f(-bridge->cytokine_effects.ifn_gamma_prosody_reduction, 0.0f, 0.8f);

    bridge->cytokine_modulations++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    return 0;
}

int speech_immune_apply_inflammation_effects(speech_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_inflammation_impairment) return 0;
    if (!bridge->immune_system) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Get inflammation state */
    brain_inflammation_level_t level = get_max_inflammation_level(bridge->immune_system);
    float duration = get_inflammation_duration_sec(bridge->immune_system);

    bridge->inflammation_state.current_level = level;
    bridge->inflammation_state.inflammation_duration_sec = duration;
    bridge->inflammation_state.is_chronic = (duration >= CHRONIC_SICKNESS_THRESHOLD);

    /* Map inflammation level to speech impairments */
    float inflammation_factor = 0.0f;
    switch (level) {
        case INFLAMMATION_NONE:
            inflammation_factor = 0.0f;
            break;
        case INFLAMMATION_LOCAL:
            inflammation_factor = 0.2f;
            break;
        case INFLAMMATION_REGIONAL:
            inflammation_factor = 0.4f;
            break;
        case INFLAMMATION_SYSTEMIC:
            inflammation_factor = 0.7f;
            break;
        case INFLAMMATION_STORM:
            inflammation_factor = 1.0f;
            break;
    }

    /* Chronic inflammation amplifies effects */
    if (bridge->inflammation_state.is_chronic) {
        inflammation_factor *= 1.3f;
        inflammation_factor = clamp_f(inflammation_factor, 0.0f, 1.0f);
    }

    /* Compute speech impacts */
    bridge->inflammation_state.verbal_fluency_reduction =
        inflammation_factor * INFLAMMATION_MAX_SPEECH_IMPAIRMENT;

    bridge->inflammation_state.word_finding_difficulty =
        (inflammation_factor > INFLAMMATION_WORD_RETRIEVAL_THRESHOLD)
            ? (inflammation_factor - INFLAMMATION_WORD_RETRIEVAL_THRESHOLD) * 2.0f
            : 0.0f;

    bridge->inflammation_state.phonological_error_rate =
        (inflammation_factor > INFLAMMATION_PHONEME_ERROR_THRESHOLD)
            ? (inflammation_factor - INFLAMMATION_PHONEME_ERROR_THRESHOLD) * 2.5f
            : 0.0f;

    bridge->inflammation_state.articulation_slowing =
        inflammation_factor * 0.6f;

    bridge->inflammation_state.comprehension_impairment =
        inflammation_factor * 0.5f;

    /* Phonological working memory capacity reduction (7±2 items reduced) */
    bridge->inflammation_state.working_memory_capacity =
        1.0f - (inflammation_factor * 0.4f);  /* Max 40% reduction */

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

float speech_immune_compute_impairment(const speech_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;

    /* Combined impairment from cytokines and inflammation */
    float cytokine_impairment = bridge->cytokine_effects.total_fluency_impairment;
    float inflammation_impairment = bridge->inflammation_state.verbal_fluency_reduction;

    /* Take maximum (not additive to avoid over-impairment) */
    float total_impairment = fmaxf(cytokine_impairment, inflammation_impairment);
    return clamp_f(total_impairment, 0.0f, 1.0f);
}

float speech_immune_get_retrieval_latency_increase(
    const speech_immune_bridge_t* bridge
) {
    if (!bridge) return 0.0f;
    return bridge->cytokine_effects.word_retrieval_latency_ms;
}

float speech_immune_get_phoneme_error_rate(
    const speech_immune_bridge_t* bridge
) {
    if (!bridge) return 0.0f;

    /* Combine cytokine and inflammation error rates */
    float cytokine_errors = bridge->cytokine_effects.phoneme_error_rate;
    float inflammation_errors = bridge->inflammation_state.phonological_error_rate;

    return clamp_f(cytokine_errors + inflammation_errors, 0.0f, 0.5f);
}

float speech_immune_get_speech_rate_factor(
    const speech_immune_bridge_t* bridge
) {
    if (!bridge) return 1.0f;

    /* Speech rate multiplier (1.0 = normal, 0.5 = half speed) */
    float reduction = bridge->cytokine_effects.speech_rate_reduction;
    return clamp_f(1.0f - reduction, 0.3f, 1.0f);  /* Min 30% of normal speed */
}

/* ============================================================================
 * Speech → Immune Implementation
 * ============================================================================ */

int speech_immune_trigger_from_effort(speech_immune_bridge_t* bridge) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_speech_immune_trigger) return 0;
    if (!bridge->immune_system || !bridge->speech_cortex) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Check if speech effort exceeds threshold */
    if (bridge->speech_trigger.speech_effort_level < SPEECH_EFFORT_IMMUNE_TRIGGER) {
        pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
        return 0;
    }

    /* High effort triggers stress response */
    bridge->speech_trigger.cortisol_triggered = true;
    bridge->speech_trigger.immune_suppression =
        bridge->speech_trigger.speech_effort_level * 0.3f;

    /* Present antigen to immune system (stress-induced inflammation) */
    uint8_t epitope[BRAIN_IMMUNE_EPITOPE_SIZE];
    memset(epitope, 0, BRAIN_IMMUNE_EPITOPE_SIZE);
    snprintf((char*)epitope, BRAIN_IMMUNE_EPITOPE_SIZE, "speech_effort_stress");

    uint32_t antigen_id;
    brain_immune_present_antigen(
        bridge->immune_system,
        ANTIGEN_SOURCE_MANUAL,
        epitope,
        strlen((char*)epitope),
        3,  /* Moderate severity */
        0,  /* No specific node */
        &antigen_id
    );

    /* Release IL-6 (stress cytokine) */
    uint32_t cytokine_id;
    brain_immune_release_cytokine(
        bridge->immune_system,
        BRAIN_CYTOKINE_IL6,
        0,  /* No specific source cell */
        bridge->speech_trigger.frustration_level,
        0,  /* Broadcast */
        &cytokine_id
    );

    bridge->speech_triggered_responses++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    nimcp_log(NIMCP_LOG_INFO, "speech_immune_bridge",
              "Speech effort triggered immune response");
    return 0;
}

int speech_immune_detect_distress_vocalization(
    speech_immune_bridge_t* bridge,
    const void* prosody_features
) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_distress_vocalization_trigger) return 0;
    if (!prosody_features) return -1;

    /* In full implementation, would analyze prosody features for:
     * - High pitch (pain/distress marker)
     * - Irregular pitch contour (emotional distress)
     * - High intensity variability (agitation)
     * - Tremor in voice (anxiety/pain)
     */

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Simplified distress detection */
    bool distress_detected = bridge->speech_trigger.distress_intensity > DISTRESS_VOCALIZATION_THRESHOLD;

    if (distress_detected) {
        bridge->speech_trigger.distress_detected = true;

        /* Distress triggers HPA axis → cortisol → immune activation */
        uint8_t epitope[BRAIN_IMMUNE_EPITOPE_SIZE];
        snprintf((char*)epitope, BRAIN_IMMUNE_EPITOPE_SIZE, "distress_vocalization");

        uint32_t antigen_id;
        brain_immune_present_antigen(
            bridge->immune_system,
            ANTIGEN_SOURCE_MANUAL,
            epitope,
            strlen((char*)epitope),
            5,  /* Higher severity */
            0,
            &antigen_id
        );

        /* Release TNF-α (acute stress response) */
        uint32_t cytokine_id;
        brain_immune_release_cytokine(
            bridge->immune_system,
            BRAIN_CYTOKINE_TNF,
            0,
            bridge->speech_trigger.distress_intensity,
            0,
            &cytokine_id
        );

        bridge->distress_events++;
        nimcp_log(NIMCP_LOG_INFO, "speech_immune_bridge",
                  "Distress vocalization triggered immune response");
    }

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);
    return 0;
}

int speech_immune_trigger_from_illness_expression(
    speech_immune_bridge_t* bridge,
    const char* word
) {
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->enable_speech_immune_trigger) return 0;
    if (!word || !is_illness_word(word)) return 0;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);

    /* Verbalizing illness modulates immune response (Pennebaker 1997) */
    /* Release IL-10 (anti-inflammatory, associated with expression/disclosure) */
    uint32_t cytokine_id;
    brain_immune_release_cytokine(
        bridge->immune_system,
        BRAIN_CYTOKINE_IL10,
        0,
        0.5f,  /* Moderate concentration */
        0,
        &cytokine_id
    );

    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    nimcp_log(NIMCP_LOG_DEBUG, "speech_immune_bridge",
              "Illness word '%s' modulated immune response", word);
    return 0;
}

/* ============================================================================
 * Bidirectional Update Implementation
 * ============================================================================ */

int speech_immune_bridge_update(
    speech_immune_bridge_t* bridge,
    uint64_t delta_ms
) {
    if (!bridge) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    bridge->total_updates++;
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    /* Apply immune effects to speech */
    speech_immune_apply_cytokine_effects(bridge);
    speech_immune_apply_inflammation_effects(bridge);

    /* Check speech state for immune triggers */
    /* In full implementation, would query speech cortex for:
     * - Current error rate
     * - Retrieval latencies
     * - Speech effort metrics
     */

    /* Update speech effort level (would be computed from speech cortex metrics) */
    /* For now, use placeholder logic */
    float error_rate = speech_immune_get_phoneme_error_rate(bridge);
    bridge->speech_trigger.error_rate = error_rate;
    bridge->speech_trigger.speech_effort_level = error_rate * 1.5f;
    bridge->speech_trigger.frustration_level =
        clamp_f(error_rate * 2.0f, 0.0f, 1.0f);

    /* Trigger immune from high effort if threshold exceeded */
    speech_immune_trigger_from_effort(bridge);

    return 0;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int speech_immune_get_cytokine_effects(
    const speech_immune_bridge_t* bridge,
    cytokine_speech_effects_t* effects
) {
    if (!bridge || !effects) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    memcpy(effects, &bridge->cytokine_effects, sizeof(cytokine_speech_effects_t));
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    return 0;
}

int speech_immune_get_inflammation_state(
    const speech_immune_bridge_t* bridge,
    inflammation_speech_state_t* state
) {
    if (!bridge || !state) return -1;

    pthread_mutex_lock((pthread_mutex_t*)bridge->mutex);
    memcpy(state, &bridge->inflammation_state, sizeof(inflammation_speech_state_t));
    pthread_mutex_unlock((pthread_mutex_t*)bridge->mutex);

    return 0;
}

bool speech_immune_is_speech_impaired(const speech_immune_bridge_t* bridge) {
    if (!bridge) return false;

    float impairment = speech_immune_compute_impairment(bridge);
    return (impairment > 0.3f);  /* Threshold for noticeable impairment */
}

float speech_immune_get_fluency_reduction(const speech_immune_bridge_t* bridge) {
    if (!bridge) return 0.0f;
    return bridge->cytokine_effects.total_fluency_impairment;
}

float speech_immune_get_working_memory_capacity(
    const speech_immune_bridge_t* bridge
) {
    if (!bridge) return 1.0f;
    return bridge->inflammation_state.working_memory_capacity;
}
