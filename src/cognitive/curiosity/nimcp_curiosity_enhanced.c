/**
 * @file nimcp_curiosity_enhanced.c
 * @brief Enhanced Curiosity System Implementation
 * @version 1.0.0
 * @date 2025-12-20
 *
 * WHAT: Implementation of 10 curiosity enhancements
 * WHY:  Complete biologically-inspired curiosity dynamics
 * HOW:  Modular sub-systems with unified interface
 *
 * @author NIMCP Development Team
 */

#include "cognitive/curiosity/nimcp_curiosity_enhanced.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/platform/nimcp_platform_time.h"
#include "utils/containers/nimcp_hash_table.h"
#include "utils/error/nimcp_error_codes.h"
#include "async/nimcp_bio_router.h"
#include "nimcp.h"

/* Include quantum bridge implementation */
#define NIMCP_CURIOSITY_QUANTUM_BRIDGE_IMPLEMENTATION
#include "cognitive/curiosity/nimcp_curiosity_quantum_bridge.h"

#include <string.h>
#include <math.h>
#include <stdio.h>

#define LOG_MODULE "curiosity_enhanced"

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Interest tracking entry for hash table
 */
typedef struct interest_entry_s {
    curiosity_topic_interest_t interest;
} interest_entry_t;

/**
 * @brief Recent stimulus for boredom detection
 */
typedef struct recent_stimulus_s {
    uint64_t hash;
    float novelty;
    uint64_t timestamp_ms;
} recent_stimulus_t;

#define MAX_RECENT_STIMULI 64

/**
 * @brief Enhanced curiosity system internal structure
 */
struct curiosity_enhanced_system_s {
    /* Configuration */
    curiosity_enhanced_config_t config;

    /* Base engine reference */
    curiosity_engine_t base_engine;

    /* State */
    curiosity_enhanced_state_t state;

    /* Statistics */
    curiosity_enhanced_stats_t stats;

    /* Interest tracking hash table */
    hash_table_t* interest_table;

    /* Recent stimuli for boredom detection */
    recent_stimulus_t recent_stimuli[MAX_RECENT_STIMULI];
    uint32_t num_recent_stimuli;
    uint32_t stimulus_write_idx;

    /* Connected systems */
    anxiety_system_t* anxiety_system;
    theory_of_mind_t* tom_system;

    /* Quantum bridge */
    curiosity_quantum_bridge_t* quantum_bridge;

    /* Bio-async */
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    /* Thread safety */
    nimcp_platform_mutex_t* mutex;

    /* Timing */
    uint64_t creation_time_ms;
    uint64_t last_update_ms;
};

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Get current time in milliseconds
 */
static uint64_t get_time_ms(void) {
    return nimcp_platform_time_monotonic_ms();
}

/**
 * @brief Clamp float to range
 */
static float clampf(float val, float min_val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

/**
 * @brief Simple string hash
 */
static uint64_t hash_string(const char* str) {
    if (!str) return 0;
    uint64_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

/**
 * @brief Interest entry destructor for hash table
 */
static void interest_entry_destructor(void* value, size_t value_size) {
    (void)value_size;
    (void)value;
    /* No dynamic memory in interest_entry_t */
}

/* ============================================================================
 * Boredom Sub-system
 * ============================================================================ */

static void boredom_init(curiosity_boredom_state_t* state,
                         const curiosity_boredom_config_t* config) {
    if (!state) return;

    memset(state, 0, sizeof(*state));
    state->boredom_threshold = config ? config->boredom_threshold : BOREDOM_THRESHOLD_DEFAULT;
    state->novelty_seeking_boost = 1.0f;
    state->last_novel_event_ms = get_time_ms();
}

static void boredom_update(curiosity_enhanced_system_t* sys, float dt_ms) {
    if (!sys) return;

    curiosity_boredom_state_t* state = &sys->state.boredom;
    const curiosity_boredom_config_t* config = &sys->config.boredom;

    uint64_t now = get_time_ms();
    float time_since_novelty = (float)(now - state->last_novel_event_ms);

    /* Update understimulation duration */
    state->understimulation_duration_ms = time_since_novelty;

    /* Compute monotony based on repetition */
    if (state->repetition_count > 3) {
        state->monotony_level += config->monotony_decay_rate * dt_ms / 1000.0f;
    } else {
        state->monotony_level -= config->monotony_decay_rate * 2.0f * dt_ms / 1000.0f;
    }
    state->monotony_level = clampf(state->monotony_level, 0.0f, 1.0f);

    /* Check understimulation timeout */
    float understim_factor = 0.0f;
    if (time_since_novelty > config->understimulation_timeout_ms) {
        understim_factor = (time_since_novelty - config->understimulation_timeout_ms) /
                          config->understimulation_timeout_ms;
        understim_factor = clampf(understim_factor, 0.0f, 1.0f);
    }

    /* Combine factors for boredom */
    float boredom_level = (state->monotony_level + understim_factor) / 2.0f;

    /* Check if bored */
    bool was_bored = state->is_bored;
    state->is_bored = boredom_level > state->boredom_threshold;

    if (state->is_bored && !was_bored) {
        sys->stats.boredom_episodes++;
        NIMCP_LOGGING_DEBUG("Boredom triggered: monotony=%.2f, understim=%.2f",
                           state->monotony_level, understim_factor);
    }

    /* Compute novelty seeking boost */
    if (state->is_bored) {
        state->novelty_seeking_boost = 1.0f + (boredom_level - state->boredom_threshold) *
                                       config->novelty_boost_factor;
        state->novelty_seeking_boost = clampf(state->novelty_seeking_boost,
                                              1.0f, BOREDOM_NOVELTY_SEEK_BOOST);
    } else {
        state->novelty_seeking_boost = 1.0f;
    }
}

static void boredom_report_stimulus(curiosity_enhanced_system_t* sys,
                                    uint64_t stimulus_hash, float novelty) {
    if (!sys) return;

    curiosity_boredom_state_t* state = &sys->state.boredom;
    uint64_t now = get_time_ms();

    /* Check if stimulus is repeated */
    bool is_repeat = false;
    for (uint32_t i = 0; i < sys->num_recent_stimuli; i++) {
        if (sys->recent_stimuli[i].hash == stimulus_hash) {
            is_repeat = true;
            break;
        }
    }

    /* Update repetition tracking */
    if (is_repeat) {
        state->repetition_count++;
    } else {
        state->repetition_count = 0;
    }

    /* Store stimulus */
    sys->recent_stimuli[sys->stimulus_write_idx].hash = stimulus_hash;
    sys->recent_stimuli[sys->stimulus_write_idx].novelty = novelty;
    sys->recent_stimuli[sys->stimulus_write_idx].timestamp_ms = now;
    sys->stimulus_write_idx = (sys->stimulus_write_idx + 1) % MAX_RECENT_STIMULI;
    if (sys->num_recent_stimuli < MAX_RECENT_STIMULI) {
        sys->num_recent_stimuli++;
    }

    /* Update novelty tracking */
    state->last_stimulus_novelty = novelty;
    if (novelty > 0.5f) {
        state->last_novel_event_ms = now;
        state->is_bored = false;
        state->monotony_level *= 0.5f;  /* Reset monotony on novelty */
        sys->stats.novelty_events++;
    }
}

/* ============================================================================
 * Interest Decay Sub-system
 * ============================================================================ */

static curiosity_topic_interest_t* interest_get_or_create(
    curiosity_enhanced_system_t* sys, const char* topic) {

    if (!sys || !topic || !sys->interest_table) return NULL;

    /* Look up existing */
    interest_entry_t* entry = (interest_entry_t*)hash_table_lookup_string(
        sys->interest_table, topic);

    if (entry) {
        return &entry->interest;
    }

    /* Create new entry */
    interest_entry_t new_entry;
    memset(&new_entry, 0, sizeof(new_entry));
    strncpy(new_entry.interest.topic, topic, sizeof(new_entry.interest.topic) - 1);
    new_entry.interest.initial_interest = 1.0f;
    new_entry.interest.current_interest = 1.0f;
    new_entry.interest.peak_interest = 1.0f;
    new_entry.interest.decay_rate = sys->config.interest.base_decay_rate;
    new_entry.interest.first_exposure_ms = get_time_ms();
    new_entry.interest.last_exposure_ms = new_entry.interest.first_exposure_ms;
    new_entry.interest.exposure_count = 1;

    bool success = hash_table_insert_string(sys->interest_table, topic,
                                            &new_entry, sizeof(new_entry));
    if (!success) {
        NIMCP_LOGGING_ERROR("Failed to create interest entry for topic: %s", topic);
        return NULL;
    }

    entry = (interest_entry_t*)hash_table_lookup_string(sys->interest_table, topic);
    return entry ? &entry->interest : NULL;
}

static float interest_compute_decay(const curiosity_interest_config_t* config,
                                    const curiosity_topic_interest_t* interest,
                                    uint64_t now_ms) {
    if (!config || !interest) return 1.0f;

    float time_since_last = (float)(now_ms - interest->last_exposure_ms);

    if (config->enable_hyperbolic_decay) {
        /* Hyperbolic decay: I(t) = I0 / (1 + k*t) */
        float k = interest->decay_rate / 1000.0f;
        return interest->initial_interest / (1.0f + k * time_since_last);
    } else {
        /* Exponential decay: I(t) = I0 * exp(-k*t) */
        float half_life = (float)config->half_life_ms;
        float decay = powf(0.5f, time_since_last / half_life);
        return interest->initial_interest * decay;
    }
}

static float interest_compute_satiation(const curiosity_topic_interest_t* interest) {
    if (!interest) return 0.0f;

    /* Satiation based on exposure count */
    float satiation = 1.0f - expf(-(float)interest->exposure_count / 10.0f);
    return clampf(satiation, 0.0f, 1.0f);
}

/* ============================================================================
 * Curiosity Types Sub-system
 * ============================================================================ */

static void types_init(curiosity_type_profile_t* profile,
                       const curiosity_type_config_t* config) {
    if (!profile) return;

    memset(profile, 0, sizeof(*profile));
    profile->dominant_type = CURIOSITY_TYPE_EPISTEMIC;  /* Default to knowledge-seeking */

    /* Initialize intensities from config weights */
    if (config) {
        for (int i = 0; i < CURIOSITY_TYPE_COUNT; i++) {
            profile->type_intensities[i] = config->type_weights[i];
        }
    } else {
        /* Default equal distribution */
        for (int i = 0; i < CURIOSITY_TYPE_COUNT; i++) {
            profile->type_intensities[i] = 1.0f / CURIOSITY_TYPE_COUNT;
        }
    }

    profile->last_transition_ms = get_time_ms();
}

static void types_update(curiosity_enhanced_system_t* sys, float dt_ms) {
    if (!sys) return;

    curiosity_type_profile_t* profile = &sys->state.types;

    /* Find dominant type */
    float max_intensity = 0.0f;
    curiosity_type_t new_dominant = profile->dominant_type;

    for (int i = 0; i < CURIOSITY_TYPE_COUNT; i++) {
        if (profile->type_intensities[i] > max_intensity) {
            max_intensity = profile->type_intensities[i];
            new_dominant = (curiosity_type_t)i;
        }
    }

    /* Check for transition */
    if (new_dominant != profile->dominant_type) {
        uint64_t now = get_time_ms();
        profile->type_durations_ms[profile->dominant_type] +=
            now - profile->last_transition_ms;
        profile->dominant_type = new_dominant;
        profile->last_transition_ms = now;
        sys->stats.type_transitions++;
    }

    /* Normalize intensities */
    float total = 0.0f;
    for (int i = 0; i < CURIOSITY_TYPE_COUNT; i++) {
        total += profile->type_intensities[i];
    }
    if (total > 0.0f) {
        for (int i = 0; i < CURIOSITY_TYPE_COUNT; i++) {
            profile->type_intensities[i] /= total;
        }
    }
}

/* ============================================================================
 * Approach-Avoidance Sub-system
 * ============================================================================ */

static void anxiety_balance_init(curiosity_approach_avoidance_t* state,
                                  const curiosity_anxiety_config_t* config) {
    if (!state) return;

    memset(state, 0, sizeof(*state));
    state->approach_tendency = 0.5f;
    state->avoidance_tendency = 0.0f;
    state->resolution_bias = config ? config->approach_bias : 0.6f;
}

static void anxiety_balance_update(curiosity_enhanced_system_t* sys, float dt_ms) {
    if (!sys) return;

    curiosity_approach_avoidance_t* state = &sys->state.anxiety_balance;

    /* Compute net motivation */
    state->net_motivation = state->approach_tendency - state->avoidance_tendency;

    /* Compute conflict */
    float min_tendency = fminf(state->approach_tendency, state->avoidance_tendency);
    state->conflict_level = min_tendency * 2.0f;  /* High when both are high */

    bool was_in_conflict = state->in_conflict;
    state->in_conflict = state->conflict_level > 0.5f;

    if (state->in_conflict && !was_in_conflict) {
        state->conflict_start_ms = get_time_ms();
        sys->stats.approach_avoidance_conflicts++;
    }

    /* Resolve conflict over time */
    if (state->in_conflict) {
        float resolution_rate = sys->config.anxiety.conflict_resolution_rate * dt_ms / 1000.0f;
        if (state->resolution_bias > 0.5f) {
            state->avoidance_tendency -= resolution_rate;
        } else {
            state->approach_tendency -= resolution_rate;
        }
        state->approach_tendency = clampf(state->approach_tendency, 0.0f, 1.0f);
        state->avoidance_tendency = clampf(state->avoidance_tendency, 0.0f, 1.0f);
    }
}

/* ============================================================================
 * Social Curiosity Sub-system
 * ============================================================================ */

static void social_init(curiosity_social_state_t* state,
                        const curiosity_social_config_t* config) {
    if (!state) return;

    memset(state, 0, sizeof(*state));
    state->gossip_interest = 0.5f;
    state->reputation_tracking = 0.5f;
    state->coalition_mapping = 0.3f;
    state->deception_detection = 0.3f;
}

static curiosity_social_target_t* social_get_or_create_target(
    curiosity_social_state_t* state, const char* agent_id) {

    if (!state || !agent_id) return NULL;

    /* Find existing */
    for (uint32_t i = 0; i < state->num_targets; i++) {
        if (strcmp(state->targets[i].agent_id, agent_id) == 0) {
            return &state->targets[i];
        }
    }

    /* Create new if space available */
    if (state->num_targets >= MAX_SOCIAL_TARGETS) {
        return NULL;
    }

    curiosity_social_target_t* target = &state->targets[state->num_targets++];
    memset(target, 0, sizeof(*target));
    strncpy(target->agent_id, agent_id, sizeof(target->agent_id) - 1);
    target->social_interest = 0.5f;
    target->trust_level = 0.5f;
    target->is_tracked = true;

    return target;
}

static void social_update(curiosity_enhanced_system_t* sys, float dt_ms) {
    if (!sys) return;

    curiosity_social_state_t* state = &sys->state.social;
    const curiosity_social_config_t* config = &sys->config.social;

    /* Decay gossip interest */
    state->gossip_interest -= config->gossip_decay_rate * dt_ms / 1000.0f;
    state->gossip_interest = clampf(state->gossip_interest, 0.1f, 1.0f);

    /* Decay social interest in targets */
    uint64_t now = get_time_ms();
    for (uint32_t i = 0; i < state->num_targets; i++) {
        float time_since = (float)(now - state->targets[i].last_interaction_ms);
        float decay = expf(-time_since / 3600000.0f);  /* 1 hour half-life */
        state->targets[i].social_interest *= decay;
    }
}

/* ============================================================================
 * Meta-Curiosity Sub-system
 * ============================================================================ */

static void meta_init(curiosity_meta_state_t* state,
                      const curiosity_meta_config_t* config) {
    if (!state) return;

    memset(state, 0, sizeof(*state));
    state->self_awareness_of_interests = 0.3f;
    state->curiosity_pattern_recognition = 0.2f;
    state->learning_strategy_awareness = 0.2f;
    state->meta_curiosity_level = 0.3f;
}

static void meta_update(curiosity_enhanced_system_t* sys, float dt_ms) {
    if (!sys) return;

    curiosity_meta_state_t* state = &sys->state.meta;
    const curiosity_meta_config_t* config = &sys->config.meta;

    /* Grow meta-awareness over time */
    state->self_awareness_of_interests += config->meta_awareness_growth_rate * dt_ms / 1000.0f;
    state->self_awareness_of_interests = clampf(state->self_awareness_of_interests, 0.0f, 1.0f);

    /* Check for automatic introspection */
    uint64_t now = get_time_ms();
    if (config->enable_automatic_introspection &&
        (now - state->last_introspection_ms) > (uint64_t)config->introspection_frequency_ms) {
        state->last_introspection_ms = now;
        sys->stats.introspection_events++;

        /* Simple meta-curiosity computation */
        state->meta_curiosity_level = state->self_awareness_of_interests * 0.5f +
                                      state->curiosity_pattern_recognition * 0.3f +
                                      state->learning_strategy_awareness * 0.2f;
    }
}

/* ============================================================================
 * Contagion Sub-system
 * ============================================================================ */

static void contagion_init(curiosity_contagion_state_t* state,
                           const curiosity_contagion_config_t* config) {
    if (!state) return;

    memset(state, 0, sizeof(*state));
    state->contagion_susceptibility = config ? config->base_susceptibility :
                                      CONTAGION_SUSCEPTIBILITY_DEFAULT;
}

static void contagion_update(curiosity_enhanced_system_t* sys, float dt_ms) {
    if (!sys) return;

    curiosity_contagion_state_t* state = &sys->state.contagion;
    const curiosity_contagion_config_t* config = &sys->config.contagion;

    /* Decay accumulated contagion */
    state->accumulated_contagion -= config->contagion_decay_rate * dt_ms / 1000.0f;
    state->accumulated_contagion = clampf(state->accumulated_contagion, 0.0f, 1.0f);
}

/* ============================================================================
 * Surprise Learning Sub-system
 * ============================================================================ */

static void surprise_init(curiosity_surprise_learning_t* state,
                          const curiosity_surprise_config_t* config) {
    if (!state) return;

    memset(state, 0, sizeof(*state));
    state->learning_rate_boost = 1.0f;
}

static void surprise_update(curiosity_enhanced_system_t* sys, float dt_ms) {
    if (!sys) return;

    curiosity_surprise_learning_t* state = &sys->state.surprise;
    const curiosity_surprise_config_t* config = &sys->config.surprise;

    /* Decay surprise effects */
    state->surprise_magnitude -= config->surprise_decay_rate * dt_ms / 1000.0f;
    state->surprise_magnitude = clampf(state->surprise_magnitude, 0.0f, 1.0f);

    /* Update learning rate boost */
    if (state->surprise_magnitude > config->surprise_threshold) {
        float excess = state->surprise_magnitude - config->surprise_threshold;
        state->learning_rate_boost = 1.0f + excess * (config->max_lr_boost - 1.0f);
    } else {
        state->learning_rate_boost = 1.0f;
    }
}

static float surprise_report(curiosity_enhanced_system_t* sys,
                             float prediction_error, const char* context) {
    if (!sys) return 1.0f;

    curiosity_surprise_learning_t* state = &sys->state.surprise;
    const curiosity_surprise_config_t* config = &sys->config.surprise;

    state->prediction_error = prediction_error;
    state->surprise_magnitude = clampf(prediction_error, 0.0f, 1.0f);
    state->last_surprise_ms = get_time_ms();
    state->surprise_count++;

    /* Update running average */
    float alpha = 0.1f;
    state->avg_surprise = alpha * state->surprise_magnitude +
                          (1.0f - alpha) * state->avg_surprise;

    /* Compute boost */
    if (state->surprise_magnitude > config->surprise_threshold) {
        float excess = state->surprise_magnitude - config->surprise_threshold;
        state->learning_rate_boost = 1.0f + excess * (config->max_lr_boost - 1.0f);
        state->learning_rate_boost = clampf(state->learning_rate_boost, 1.0f, config->max_lr_boost);

        state->memory_consolidation_priority = state->surprise_magnitude;
        state->attention_capture_strength = state->surprise_magnitude * 1.5f;

        sys->stats.surprise_events++;
        NIMCP_LOGGING_DEBUG("Surprise event: error=%.2f, boost=%.2f",
                           prediction_error, state->learning_rate_boost);
    }

    return state->learning_rate_boost;
}

/* ============================================================================
 * Fatigue Sub-system
 * ============================================================================ */

static void fatigue_init(curiosity_fatigue_state_t* state,
                         const curiosity_fatigue_config_t* config) {
    if (!state) return;

    memset(state, 0, sizeof(*state));
    state->rest_threshold = config ? config->rest_threshold : FATIGUE_REST_THRESHOLD;
    state->recovery_rate = config ? config->base_recovery_rate : FATIGUE_RECOVERY_RATE;
}

static void fatigue_update(curiosity_enhanced_system_t* sys, float dt_ms) {
    if (!sys) return;

    curiosity_fatigue_state_t* state = &sys->state.fatigue;
    const curiosity_fatigue_config_t* config = &sys->config.fatigue;

    if (state->is_resting) {
        /* Recovery mode */
        state->exploration_fatigue -= config->base_recovery_rate * dt_ms / 1000.0f;
        state->cognitive_depletion -= config->base_recovery_rate * dt_ms / 1000.0f;
        state->total_rest_ms += (uint64_t)dt_ms;

        state->exploration_fatigue = clampf(state->exploration_fatigue, 0.0f, 1.0f);
        state->cognitive_depletion = clampf(state->cognitive_depletion, 0.0f, 1.0f);

        /* Check if recovered enough */
        if (state->exploration_fatigue < 0.2f) {
            state->is_resting = false;
        }
    } else {
        /* Active mode - accumulate fatigue */
        state->exploration_fatigue += config->fatigue_accumulation_rate * dt_ms / 1000.0f;
        state->cognitive_depletion += config->fatigue_accumulation_rate * 0.5f * dt_ms / 1000.0f;
        state->total_exploration_ms += (uint64_t)dt_ms;

        state->exploration_fatigue = clampf(state->exploration_fatigue, 0.0f, 1.0f);
        state->cognitive_depletion = clampf(state->cognitive_depletion, 0.0f, 1.0f);

        /* Check if rest is needed */
        state->needs_rest = state->exploration_fatigue > state->rest_threshold;

        if (state->needs_rest && config->enable_auto_rest) {
            state->is_resting = true;
            state->last_rest_period_ms = get_time_ms();
            sys->stats.fatigue_rest_periods++;
        }
    }
}

/* ============================================================================
 * Counterfactual Sub-system
 * ============================================================================ */

static void counterfactual_init(curiosity_counterfactual_state_t* state,
                                const curiosity_counterfactual_config_t* config) {
    if (!state) return;

    memset(state, 0, sizeof(*state));
    state->counterfactual_curiosity = 0.3f;
    state->regret_sensitivity = 0.5f;
}

static int counterfactual_generate(curiosity_enhanced_system_t* sys,
                                   const char* decision_point,
                                   const char* actual_outcome,
                                   curiosity_counterfactual_t* cf) {
    if (!sys || !decision_point || !actual_outcome || !cf) return -1;

    curiosity_counterfactual_state_t* state = &sys->state.counterfactual;

    if (state->num_items >= MAX_COUNTERFACTUALS) {
        /* Replace oldest */
        memmove(&state->items[0], &state->items[1],
                (MAX_COUNTERFACTUALS - 1) * sizeof(curiosity_counterfactual_t));
        state->num_items = MAX_COUNTERFACTUALS - 1;
    }

    memset(cf, 0, sizeof(*cf));
    strncpy(cf->actual_outcome, actual_outcome, sizeof(cf->actual_outcome) - 1);
    snprintf(cf->counterfactual_question, sizeof(cf->counterfactual_question),
             "What if I had done differently at: %s?", decision_point);
    cf->decision_time_ms = get_time_ms();
    cf->learning_value = 0.5f;
    cf->exploration_cost = COUNTERFACTUAL_EXPLORATION_COST;

    /* Copy to state */
    state->items[state->num_items++] = *cf;
    sys->stats.counterfactuals_generated++;

    return 0;
}

/* ============================================================================
 * Aggregate Functions
 * ============================================================================ */

static void compute_aggregates(curiosity_enhanced_system_t* sys) {
    if (!sys) return;

    curiosity_enhanced_state_t* state = &sys->state;

    /* Combine all factors for overall drive */
    float boredom_factor = state->boredom.is_bored ?
                          state->boredom.novelty_seeking_boost : 1.0f;
    float anxiety_factor = clampf(1.0f - state->anxiety_balance.avoidance_tendency, 0.2f, 1.0f);
    float fatigue_factor = clampf(1.0f - state->fatigue.exploration_fatigue, 0.1f, 1.0f);
    float surprise_factor = state->surprise.learning_rate_boost;

    /* Get base curiosity from dominant type */
    float type_intensity = state->types.type_intensities[state->types.dominant_type];

    /* Combine */
    state->overall_curiosity_drive = type_intensity * boredom_factor * anxiety_factor *
                                     fatigue_factor * (0.5f + surprise_factor * 0.5f);
    state->overall_curiosity_drive = clampf(state->overall_curiosity_drive, 0.0f, 1.0f);

    state->effective_exploration_rate = state->overall_curiosity_drive;
    if (state->fatigue.is_resting) {
        state->effective_exploration_rate *= 0.1f;
    }

    /* Update average stats */
    float alpha = 0.01f;
    sys->stats.avg_curiosity_level = alpha * state->overall_curiosity_drive +
                                     (1.0f - alpha) * sys->stats.avg_curiosity_level;
    sys->stats.avg_boredom_level = alpha * state->boredom.monotony_level +
                                   (1.0f - alpha) * sys->stats.avg_boredom_level;
    sys->stats.avg_fatigue_level = alpha * state->fatigue.exploration_fatigue +
                                   (1.0f - alpha) * sys->stats.avg_fatigue_level;
}

/* ============================================================================
 * Public API Implementation
 * ============================================================================ */

void curiosity_enhanced_config_default(curiosity_enhanced_config_t* config) {
    if (!config) return;

    memset(config, 0, sizeof(*config));

    /* Boredom config */
    config->boredom.boredom_threshold = BOREDOM_THRESHOLD_DEFAULT;
    config->boredom.monotony_decay_rate = BOREDOM_MONOTONY_DECAY;
    config->boredom.novelty_boost_factor = BOREDOM_NOVELTY_SEEK_BOOST - 1.0f;
    config->boredom.understimulation_timeout_ms = 60000.0f;  /* 1 minute */
    config->boredom.enable_auto_novelty_seek = true;

    /* Interest config */
    config->interest.base_decay_rate = INTEREST_DECAY_RATE_DEFAULT;
    config->interest.satiation_threshold = SATIATION_THRESHOLD;
    config->interest.half_life_ms = INTEREST_HALF_LIFE_MS;
    config->interest.residual_interest_min = 0.1f;
    config->interest.enable_hyperbolic_decay = true;

    /* Types config */
    for (int i = 0; i < CURIOSITY_TYPE_COUNT; i++) {
        config->types.type_weights[i] = 1.0f / CURIOSITY_TYPE_COUNT;
    }
    config->types.transition_threshold = 0.3f;
    config->types.enable_dynamic_switching = true;

    /* Anxiety config */
    config->anxiety.anxiety_suppress_threshold = ANXIETY_SUPPRESS_THRESHOLD;
    config->anxiety.approach_bias = 0.6f;
    config->anxiety.conflict_resolution_rate = 0.1f;
    config->anxiety.enable_gradual_exposure = true;

    /* Social config */
    config->social.gossip_decay_rate = GOSSIP_INTEREST_DECAY;
    config->social.trust_influence = 0.5f;
    config->social.coalition_update_rate = 0.01f;
    config->social.enable_deception_vigilance = true;

    /* Meta config */
    config->meta.introspection_frequency_ms = 60000.0f;
    config->meta.blind_spot_detection_threshold = 0.2f;
    config->meta.meta_awareness_growth_rate = 0.001f;
    config->meta.enable_automatic_introspection = true;

    /* Contagion config */
    config->contagion.base_susceptibility = CONTAGION_SUSCEPTIBILITY_DEFAULT;
    config->contagion.trust_susceptibility_factor = 0.5f;
    config->contagion.contagion_decay_rate = CONTAGION_DECAY_RATE;
    config->contagion.enable_selective_contagion = true;

    /* Surprise config */
    config->surprise.surprise_threshold = SURPRISE_THRESHOLD_DEFAULT;
    config->surprise.max_lr_boost = SURPRISE_LR_BOOST_MAX;
    config->surprise.surprise_decay_rate = 0.1f;
    config->surprise.memory_priority_factor = 1.5f;

    /* Fatigue config */
    config->fatigue.fatigue_accumulation_rate = FATIGUE_ACCUMULATION_RATE;
    config->fatigue.base_recovery_rate = FATIGUE_RECOVERY_RATE;
    config->fatigue.rest_threshold = FATIGUE_REST_THRESHOLD;
    config->fatigue.min_rest_duration_ms = 10000.0f;
    config->fatigue.enable_auto_rest = false;

    /* Counterfactual config */
    config->counterfactual.regret_threshold = 0.5f;
    config->counterfactual.exploration_cost_limit = 0.3f;
    config->counterfactual.learning_value_threshold = 0.3f;
    config->counterfactual.enable_automatic_generation = true;

    /* Global config */
    config->update_interval_ms = 100.0f;
    config->enable_bio_async = true;
    config->enable_all_enhancements = true;
    config->enable_quantum_curiosity = true;
}

curiosity_enhanced_system_t* curiosity_enhanced_create(
    const curiosity_enhanced_config_t* config,
    curiosity_engine_t base_engine) {

    curiosity_enhanced_system_t* sys = (curiosity_enhanced_system_t*)
        nimcp_calloc(1, sizeof(curiosity_enhanced_system_t));

    if (!sys) {
        NIMCP_LOGGING_ERROR("Failed to allocate enhanced curiosity system");
        return NULL;
    }

    /* Store config */
    if (config) {
        sys->config = *config;
    } else {
        curiosity_enhanced_config_default(&sys->config);
    }

    sys->base_engine = base_engine;

    /* Create mutex */
    sys->mutex = nimcp_platform_mutex_create();
    if (!sys->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(sys);
        return NULL;
    }

    /* Create interest hash table */
    hash_table_config_t ht_config = {
        .initial_buckets = 256,
        .key_type = HASH_KEY_STRING,
        .hash_algorithm = HASH_ALG_FNV1A,
        .value_destructor = interest_entry_destructor,
        .case_insensitive = false,
        .thread_safe = false
    };

    sys->interest_table = hash_table_create(&ht_config);
    if (!sys->interest_table) {
        NIMCP_LOGGING_ERROR("Failed to create interest table");
        nimcp_platform_mutex_destroy(sys->mutex);
        nimcp_free(sys);
        return NULL;
    }

    /* Initialize sub-systems */
    boredom_init(&sys->state.boredom, &sys->config.boredom);
    types_init(&sys->state.types, &sys->config.types);
    anxiety_balance_init(&sys->state.anxiety_balance, &sys->config.anxiety);
    social_init(&sys->state.social, &sys->config.social);
    meta_init(&sys->state.meta, &sys->config.meta);
    contagion_init(&sys->state.contagion, &sys->config.contagion);
    surprise_init(&sys->state.surprise, &sys->config.surprise);
    fatigue_init(&sys->state.fatigue, &sys->config.fatigue);
    counterfactual_init(&sys->state.counterfactual, &sys->config.counterfactual);

    sys->creation_time_ms = get_time_ms();
    sys->last_update_ms = sys->creation_time_ms;

    /* Create quantum bridge if enabled */
    if (sys->config.enable_quantum_curiosity) {
        curiosity_quantum_config_t quantum_config;
        curiosity_quantum_default_config(&quantum_config);

        sys->quantum_bridge = curiosity_quantum_create(&quantum_config);
        if (!sys->quantum_bridge) {
            NIMCP_LOGGING_WARN("Failed to create quantum bridge, continuing without quantum exploration");
        } else {
            NIMCP_LOGGING_INFO("Quantum curiosity bridge enabled (max_topics=%u, steps=%u)",
                              quantum_config.max_topics, quantum_config.exploration_steps);
        }
    }

    NIMCP_LOGGING_INFO("Created enhanced curiosity system with %d enhancements%s",
                       sys->config.enable_all_enhancements ? 10 : 0,
                       sys->quantum_bridge ? " + quantum exploration" : "");

    return sys;
}

void curiosity_enhanced_destroy(curiosity_enhanced_system_t* system) {
    if (!system) return;

    /* Disconnect bio-async */
    if (system->bio_async_enabled) {
        curiosity_enhanced_disconnect_bio_async(system);
    }

    /* Destroy quantum bridge */
    if (system->quantum_bridge) {
        curiosity_quantum_destroy(system->quantum_bridge);
    }

    /* Free blind spots */
    for (uint32_t i = 0; i < system->state.meta.num_blind_spots; i++) {
        if (system->state.meta.identified_blind_spots[i]) {
            nimcp_free(system->state.meta.identified_blind_spots[i]);
        }
    }

    /* Destroy hash table */
    if (system->interest_table) {
        hash_table_destroy(system->interest_table);
    }

    /* Destroy mutex */
    if (system->mutex) {
        nimcp_platform_mutex_destroy(system->mutex);
    }

    nimcp_free(system);
    NIMCP_LOGGING_INFO("Destroyed enhanced curiosity system");
}

int curiosity_enhanced_update(curiosity_enhanced_system_t* system, float dt_ms) {
    if (!system) return NIMCP_ERROR_NULL_ARG;

    nimcp_platform_mutex_lock(system->mutex);

    /* Update all sub-systems */
    boredom_update(system, dt_ms);
    types_update(system, dt_ms);
    anxiety_balance_update(system, dt_ms);
    social_update(system, dt_ms);
    meta_update(system, dt_ms);
    contagion_update(system, dt_ms);
    surprise_update(system, dt_ms);
    fatigue_update(system, dt_ms);

    /* Compute aggregates */
    compute_aggregates(system);

    system->state.last_update_ms = get_time_ms();
    system->last_update_ms = system->state.last_update_ms;

    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

/* Boredom functions */
bool curiosity_enhanced_is_bored(
    const curiosity_enhanced_system_t* system,
    curiosity_boredom_state_t* state) {

    if (!system) return false;

    if (state) {
        *state = system->state.boredom;
    }

    return system->state.boredom.is_bored;
}

int curiosity_enhanced_report_stimulus(
    curiosity_enhanced_system_t* system,
    uint64_t stimulus_hash,
    float novelty) {

    if (!system) return NIMCP_ERROR_NULL_ARG;

    nimcp_platform_mutex_lock(system->mutex);
    boredom_report_stimulus(system, stimulus_hash, novelty);
    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

float curiosity_enhanced_get_boredom_boost(
    const curiosity_enhanced_system_t* system) {

    if (!system) return 1.0f;
    return system->state.boredom.novelty_seeking_boost;
}

/* Interest decay functions */
float curiosity_enhanced_get_topic_interest(
    const curiosity_enhanced_system_t* system,
    const char* topic) {

    if (!system || !topic) return 0.0f;

    interest_entry_t* entry = (interest_entry_t*)hash_table_lookup_string(
        system->interest_table, topic);

    if (!entry) return 1.0f;  /* Unknown topic = full interest */

    return interest_compute_decay(&system->config.interest,
                                 &entry->interest, get_time_ms());
}

int curiosity_enhanced_record_exposure(
    curiosity_enhanced_system_t* system,
    const char* topic,
    float learning_value) {

    if (!system || !topic) return NIMCP_ERROR_NULL_ARG;

    nimcp_platform_mutex_lock(system->mutex);

    curiosity_topic_interest_t* interest = interest_get_or_create(system, topic);
    if (!interest) {
        nimcp_platform_mutex_unlock(system->mutex);
        return NIMCP_ERROR;
    }

    interest->exposure_count++;
    interest->last_exposure_ms = get_time_ms();
    interest->satiation_level += learning_value * 0.1f;
    interest->satiation_level = clampf(interest->satiation_level, 0.0f, 1.0f);

    if (interest->satiation_level > system->config.interest.satiation_threshold) {
        system->stats.interest_satiation_events++;
    }

    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

float curiosity_enhanced_compute_satiation(
    const curiosity_enhanced_system_t* system,
    const char* topic) {

    if (!system || !topic) return 0.0f;

    interest_entry_t* entry = (interest_entry_t*)hash_table_lookup_string(
        system->interest_table, topic);

    if (!entry) return 0.0f;

    return interest_compute_satiation(&entry->interest);
}

float curiosity_enhanced_get_residual_interest(
    const curiosity_enhanced_system_t* system,
    const char* topic) {

    if (!system || !topic) return 0.0f;

    float current = curiosity_enhanced_get_topic_interest(system, topic);
    float satiation = curiosity_enhanced_compute_satiation(system, topic);
    float residual = current * (1.0f - satiation * 0.8f);

    return fmaxf(residual, system->config.interest.residual_interest_min);
}

/* Curiosity type functions */
curiosity_type_t curiosity_enhanced_get_dominant_type(
    const curiosity_enhanced_system_t* system) {

    if (!system) return CURIOSITY_TYPE_EPISTEMIC;
    return system->state.types.dominant_type;
}

int curiosity_enhanced_get_type_profile(
    const curiosity_enhanced_system_t* system,
    curiosity_type_profile_t* profile) {

    if (!system || !profile) return NIMCP_ERROR_NULL_ARG;

    *profile = system->state.types;
    return 0;
}

int curiosity_enhanced_set_type_intensity(
    curiosity_enhanced_system_t* system,
    curiosity_type_t type,
    float intensity) {

    if (!system) return NIMCP_ERROR_NULL_ARG;
    if (type < 0 || type >= CURIOSITY_TYPE_COUNT) return NIMCP_ERROR_INVALID;

    nimcp_platform_mutex_lock(system->mutex);
    system->state.types.type_intensities[type] = clampf(intensity, 0.0f, 1.0f);
    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

int curiosity_enhanced_transition_type(
    curiosity_enhanced_system_t* system,
    curiosity_type_t new_type) {

    if (!system) return NIMCP_ERROR_NULL_ARG;
    if (new_type < 0 || new_type >= CURIOSITY_TYPE_COUNT) return NIMCP_ERROR_INVALID;

    nimcp_platform_mutex_lock(system->mutex);

    /* Boost new type, reduce others */
    for (int i = 0; i < CURIOSITY_TYPE_COUNT; i++) {
        if (i == new_type) {
            system->state.types.type_intensities[i] = 0.5f;
        } else {
            system->state.types.type_intensities[i] *= 0.7f;
        }
    }

    system->state.types.dominant_type = new_type;
    system->state.types.last_transition_ms = get_time_ms();
    system->stats.type_transitions++;

    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

/* Anxiety balance functions */
int curiosity_enhanced_connect_anxiety(
    curiosity_enhanced_system_t* system,
    anxiety_system_t* anxiety) {

    if (!system) return NIMCP_ERROR_NULL_ARG;

    nimcp_platform_mutex_lock(system->mutex);
    system->anxiety_system = anxiety;
    nimcp_platform_mutex_unlock(system->mutex);

    NIMCP_LOGGING_INFO("Connected anxiety system to enhanced curiosity");
    return 0;
}

float curiosity_enhanced_get_net_motivation(
    const curiosity_enhanced_system_t* system) {

    if (!system) return 0.0f;
    return system->state.anxiety_balance.net_motivation;
}

bool curiosity_enhanced_should_explore(
    const curiosity_enhanced_system_t* system,
    float threat_level) {

    if (!system) return false;

    float adjusted_avoidance = system->state.anxiety_balance.avoidance_tendency + threat_level;
    float net = system->state.anxiety_balance.approach_tendency - adjusted_avoidance;

    return net > 0.0f;
}

int curiosity_enhanced_report_conflict_resolution(
    curiosity_enhanced_system_t* system,
    bool approach_won) {

    if (!system) return NIMCP_ERROR_NULL_ARG;

    nimcp_platform_mutex_lock(system->mutex);

    if (approach_won) {
        system->state.anxiety_balance.approach_wins++;
        system->state.anxiety_balance.resolution_bias += 0.01f;
    } else {
        system->state.anxiety_balance.avoidance_wins++;
        system->state.anxiety_balance.resolution_bias -= 0.01f;
    }

    system->state.anxiety_balance.resolution_bias =
        clampf(system->state.anxiety_balance.resolution_bias, 0.0f, 1.0f);
    system->state.anxiety_balance.in_conflict = false;

    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

/* Social curiosity functions */
int curiosity_enhanced_connect_tom(
    curiosity_enhanced_system_t* system,
    theory_of_mind_t* tom) {

    if (!system) return NIMCP_ERROR_NULL_ARG;

    nimcp_platform_mutex_lock(system->mutex);
    system->tom_system = tom;
    nimcp_platform_mutex_unlock(system->mutex);

    NIMCP_LOGGING_INFO("Connected ToM system to enhanced curiosity");
    return 0;
}

float curiosity_enhanced_assess_social_target(
    curiosity_enhanced_system_t* system,
    const char* agent_id,
    curiosity_social_target_t* target) {

    if (!system || !agent_id) return 0.0f;

    nimcp_platform_mutex_lock(system->mutex);

    curiosity_social_target_t* t = social_get_or_create_target(
        &system->state.social, agent_id);

    if (!t) {
        nimcp_platform_mutex_unlock(system->mutex);
        return 0.0f;
    }

    if (target) {
        *target = *t;
    }

    float result = t->social_interest;
    nimcp_platform_mutex_unlock(system->mutex);

    return result;
}

int curiosity_enhanced_record_social_interaction(
    curiosity_enhanced_system_t* system,
    const char* agent_id,
    float info_gained) {

    if (!system || !agent_id) return NIMCP_ERROR_NULL_ARG;

    nimcp_platform_mutex_lock(system->mutex);

    curiosity_social_target_t* t = social_get_or_create_target(
        &system->state.social, agent_id);

    if (t) {
        t->interaction_count++;
        t->last_interaction_ms = get_time_ms();
        t->information_value = info_gained;
        t->social_interest += info_gained * 0.1f;
        t->social_interest = clampf(t->social_interest, 0.0f, 1.0f);
    }

    system->stats.social_curiosity_events++;
    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

float curiosity_enhanced_get_gossip_interest(
    const curiosity_enhanced_system_t* system) {

    if (!system) return 0.0f;
    return system->state.social.gossip_interest;
}

/* Meta-curiosity functions */
int curiosity_enhanced_introspect(
    curiosity_enhanced_system_t* system,
    curiosity_meta_state_t* state) {

    if (!system) return NIMCP_ERROR_NULL_ARG;

    nimcp_platform_mutex_lock(system->mutex);

    system->state.meta.last_introspection_ms = get_time_ms();
    system->state.meta.introspection_depth += 0.1f;
    system->state.meta.introspection_depth =
        clampf(system->state.meta.introspection_depth, 0.0f, 1.0f);

    /* Update self-awareness based on introspection */
    system->state.meta.self_awareness_of_interests += 0.05f;
    system->state.meta.self_awareness_of_interests =
        clampf(system->state.meta.self_awareness_of_interests, 0.0f, 1.0f);

    system->stats.introspection_events++;

    if (state) {
        *state = system->state.meta;
    }

    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

uint32_t curiosity_enhanced_identify_blind_spots(
    curiosity_enhanced_system_t* system) {

    if (!system) return 0;

    nimcp_platform_mutex_lock(system->mutex);

    /* Simple blind spot detection: types with low intensity */
    uint32_t count = 0;
    for (int i = 0; i < CURIOSITY_TYPE_COUNT && count < MAX_BLIND_SPOTS; i++) {
        if (system->state.types.type_intensities[i] <
            system->config.meta.blind_spot_detection_threshold) {

            const char* type_str = curiosity_type_to_string((curiosity_type_t)i);
            if (system->state.meta.identified_blind_spots[count]) {
                nimcp_free(system->state.meta.identified_blind_spots[count]);
            }
            system->state.meta.identified_blind_spots[count] =
                (char*)nimcp_malloc(strlen(type_str) + 32);
            if (system->state.meta.identified_blind_spots[count]) {
                snprintf(system->state.meta.identified_blind_spots[count], 256,
                        "Low curiosity in: %s", type_str);
                count++;
            }
        }
    }

    system->state.meta.num_blind_spots = count;
    nimcp_platform_mutex_unlock(system->mutex);

    return count;
}

float curiosity_enhanced_get_meta_curiosity(
    const curiosity_enhanced_system_t* system) {

    if (!system) return 0.0f;
    return system->state.meta.meta_curiosity_level;
}

/* Contagion functions */
bool curiosity_enhanced_observe_curiosity(
    curiosity_enhanced_system_t* system,
    const curiosity_contagion_event_t* event) {

    if (!system || !event) return false;

    nimcp_platform_mutex_lock(system->mutex);

    curiosity_contagion_state_t* state = &system->state.contagion;

    /* Compute contagion strength */
    float strength = event->observed_curiosity_intensity * state->contagion_susceptibility;

    /* Add to recent events */
    if (state->num_recent_events < 16) {
        state->recent_events[state->num_recent_events++] = *event;
    }

    state->accumulated_contagion += strength;
    state->accumulated_contagion = clampf(state->accumulated_contagion, 0.0f, 1.0f);
    state->total_contagion_events++;

    /* Adopt curiosity if strong enough */
    bool adopted = strength > 0.3f;
    if (adopted) {
        state->curiosities_adopted++;

        /* Boost epistemic curiosity for observed topic */
        system->state.types.type_intensities[CURIOSITY_TYPE_EPISTEMIC] += strength * 0.1f;
    }

    system->stats.contagion_events++;
    nimcp_platform_mutex_unlock(system->mutex);

    return adopted;
}

float curiosity_enhanced_get_contagion_susceptibility(
    const curiosity_enhanced_system_t* system) {

    if (!system) return 0.0f;
    return system->state.contagion.contagion_susceptibility;
}

int curiosity_enhanced_set_contagion_susceptibility(
    curiosity_enhanced_system_t* system,
    float susceptibility) {

    if (!system) return NIMCP_ERROR_NULL_ARG;

    nimcp_platform_mutex_lock(system->mutex);
    system->state.contagion.contagion_susceptibility = clampf(susceptibility, 0.0f, 1.0f);
    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

/* Surprise learning functions */
float curiosity_enhanced_report_surprise(
    curiosity_enhanced_system_t* system,
    float prediction_error,
    const char* context) {

    if (!system) return 1.0f;

    nimcp_platform_mutex_lock(system->mutex);
    float boost = surprise_report(system, prediction_error, context);
    nimcp_platform_mutex_unlock(system->mutex);

    return boost;
}

float curiosity_enhanced_get_surprise_boost(
    const curiosity_enhanced_system_t* system) {

    if (!system) return 1.0f;
    return system->state.surprise.learning_rate_boost;
}

float curiosity_enhanced_prioritize_surprise(
    curiosity_enhanced_system_t* system,
    const char* experience) {

    if (!system) return 0.0f;

    (void)experience;  /* Could be used for more sophisticated prioritization */

    return system->state.surprise.memory_consolidation_priority;
}

/* Fatigue functions */
float curiosity_enhanced_check_fatigue(
    const curiosity_enhanced_system_t* system,
    curiosity_fatigue_state_t* state) {

    if (!system) return 0.0f;

    if (state) {
        *state = system->state.fatigue;
    }

    return system->state.fatigue.exploration_fatigue;
}

int curiosity_enhanced_initiate_recovery(
    curiosity_enhanced_system_t* system,
    float rest_duration_ms) {

    if (!system) return NIMCP_ERROR_NULL_ARG;

    (void)rest_duration_ms;  /* Could be used to set expected rest duration */

    nimcp_platform_mutex_lock(system->mutex);

    system->state.fatigue.is_resting = true;
    system->state.fatigue.last_rest_period_ms = get_time_ms();
    system->stats.fatigue_rest_periods++;

    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

bool curiosity_enhanced_needs_rest(
    const curiosity_enhanced_system_t* system) {

    if (!system) return false;
    return system->state.fatigue.needs_rest;
}

int curiosity_enhanced_end_recovery(
    curiosity_enhanced_system_t* system) {

    if (!system) return NIMCP_ERROR_NULL_ARG;

    nimcp_platform_mutex_lock(system->mutex);
    system->state.fatigue.is_resting = false;
    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

/* Counterfactual functions */
int curiosity_enhanced_generate_counterfactual(
    curiosity_enhanced_system_t* system,
    const char* decision_point,
    const char* actual_outcome,
    curiosity_counterfactual_t* counterfactual) {

    if (!system || !decision_point || !actual_outcome || !counterfactual) {
        return NIMCP_ERROR_NULL_ARG;
    }

    nimcp_platform_mutex_lock(system->mutex);
    int ret = counterfactual_generate(system, decision_point, actual_outcome, counterfactual);
    nimcp_platform_mutex_unlock(system->mutex);

    return ret;
}

int curiosity_enhanced_explore_counterfactual(
    curiosity_enhanced_system_t* system,
    curiosity_counterfactual_t* counterfactual,
    float* learning_outcome) {

    if (!system || !counterfactual) return NIMCP_ERROR_NULL_ARG;

    nimcp_platform_mutex_lock(system->mutex);

    counterfactual->is_explored = true;
    system->state.counterfactual.counterfactuals_explored++;

    /* Simulate learning outcome */
    float outcome = counterfactual->learning_value * (1.0f - counterfactual->exploration_cost);
    outcome = clampf(outcome, 0.0f, 1.0f);

    if (learning_outcome) {
        *learning_outcome = outcome;
    }

    /* Update average */
    float alpha = 0.1f;
    system->state.counterfactual.avg_learning_value =
        alpha * outcome + (1.0f - alpha) * system->state.counterfactual.avg_learning_value;

    nimcp_platform_mutex_unlock(system->mutex);

    return 0;
}

float curiosity_enhanced_get_counterfactual_curiosity(
    const curiosity_enhanced_system_t* system) {

    if (!system) return 0.0f;
    return system->state.counterfactual.counterfactual_curiosity;
}

/* State and statistics functions */
int curiosity_enhanced_get_state(
    const curiosity_enhanced_system_t* system,
    curiosity_enhanced_state_t* state) {

    if (!system || !state) return NIMCP_ERROR_NULL_ARG;

    *state = system->state;
    return 0;
}

int curiosity_enhanced_get_stats(
    const curiosity_enhanced_system_t* system,
    curiosity_enhanced_stats_t* stats) {

    if (!system || !stats) return NIMCP_ERROR_NULL_ARG;

    *stats = system->stats;
    return 0;
}

void curiosity_enhanced_reset_stats(
    curiosity_enhanced_system_t* system) {

    if (!system) return;

    nimcp_platform_mutex_lock(system->mutex);
    memset(&system->stats, 0, sizeof(system->stats));
    nimcp_platform_mutex_unlock(system->mutex);
}

float curiosity_enhanced_get_overall_drive(
    const curiosity_enhanced_system_t* system) {

    if (!system) return 0.0f;
    return system->state.overall_curiosity_drive;
}

/* Bio-async functions */
int curiosity_enhanced_connect_bio_async(
    curiosity_enhanced_system_t* system) {

    if (!system) return NIMCP_ERROR_NULL_ARG;
    if (system->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_CURIOSITY_BOREDOM,
        .module_name = "curiosity_enhanced",
        .inbox_capacity = 64,
        .user_data = system
    };

    system->bio_ctx = bio_router_register_module(&info);
    if (system->bio_ctx) {
        system->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected enhanced curiosity to bio-async router");
    }

    return 0;
}

int curiosity_enhanced_disconnect_bio_async(
    curiosity_enhanced_system_t* system) {

    if (!system) return NIMCP_ERROR_NULL_ARG;
    if (!system->bio_async_enabled) return 0;

    bio_router_unregister_module(system->bio_ctx);
    system->bio_async_enabled = false;
    system->bio_ctx = NULL;

    NIMCP_LOGGING_INFO("Disconnected enhanced curiosity from bio-async router");
    return 0;
}

bool curiosity_enhanced_is_bio_async_connected(
    const curiosity_enhanced_system_t* system) {

    if (!system) return false;
    return system->bio_async_enabled;
}

/* String conversion functions */
const char* curiosity_type_to_string(curiosity_type_t type) {
    switch (type) {
        case CURIOSITY_TYPE_DIVERSIVE:  return "diversive";
        case CURIOSITY_TYPE_SPECIFIC:   return "specific";
        case CURIOSITY_TYPE_PERCEPTUAL: return "perceptual";
        case CURIOSITY_TYPE_EPISTEMIC:  return "epistemic";
        case CURIOSITY_TYPE_SOCIAL:     return "social";
        case CURIOSITY_TYPE_MORBID:     return "morbid";
        default:                        return "unknown";
    }
}

curiosity_type_t curiosity_type_from_string(const char* str) {
    if (!str) return CURIOSITY_TYPE_EPISTEMIC;

    if (strcmp(str, "diversive") == 0)  return CURIOSITY_TYPE_DIVERSIVE;
    if (strcmp(str, "specific") == 0)   return CURIOSITY_TYPE_SPECIFIC;
    if (strcmp(str, "perceptual") == 0) return CURIOSITY_TYPE_PERCEPTUAL;
    if (strcmp(str, "epistemic") == 0)  return CURIOSITY_TYPE_EPISTEMIC;
    if (strcmp(str, "social") == 0)     return CURIOSITY_TYPE_SOCIAL;
    if (strcmp(str, "morbid") == 0)     return CURIOSITY_TYPE_MORBID;

    return CURIOSITY_TYPE_EPISTEMIC;
}

/* ============================================================================
 * Quantum Bridge Integration
 * ============================================================================ */

curiosity_quantum_bridge_t* curiosity_enhanced_get_quantum_bridge(
    curiosity_enhanced_system_t* system) {

    if (!system) return NULL;
    return system->quantum_bridge;
}

float curiosity_enhanced_quantum_explore(
    curiosity_enhanced_system_t* system,
    const char* start_topic,
    char* novel_topic) {

    if (!system || !system->quantum_bridge) return -1.0f;

    nimcp_platform_mutex_lock(system->mutex);

    /* Perform quantum exploration */
    float novelty = curiosity_quantum_explore(
        system->quantum_bridge,
        start_topic,
        0,  /* Use default steps */
        novel_topic
    );

    /* Update stats */
    if (novelty >= 0.0f) {
        system->stats.quantum_explorations++;

        /* Update average speedup */
        curiosity_quantum_stats_t qstats;
        if (curiosity_quantum_get_stats(system->quantum_bridge, &qstats) == 0) {
            system->stats.avg_quantum_speedup = qstats.avg_exploration_speedup;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);

    return novelty;
}

int curiosity_enhanced_add_quantum_topic(
    curiosity_enhanced_system_t* system,
    const char* topic,
    float curiosity_level,
    float novelty_score) {

    if (!system || !topic) return NIMCP_ERROR_NULL_ARG;
    if (!system->quantum_bridge) return NIMCP_ERROR_INVALID;

    nimcp_platform_mutex_lock(system->mutex);

    int result = curiosity_quantum_add_topic(
        system->quantum_bridge,
        topic,
        curiosity_level,
        novelty_score
    );

    /* Also add to interest tracking */
    if (result >= 0) {
        curiosity_topic_interest_t* interest = interest_get_or_create(system, topic);
        if (interest) {
            interest->initial_interest = curiosity_level;
            interest->current_interest = curiosity_level;
            interest->novelty_score = novelty_score;
        }
    }

    nimcp_platform_mutex_unlock(system->mutex);

    return result >= 0 ? 0 : NIMCP_ERROR;
}

float curiosity_enhanced_quantum_evaluate_novelty(
    curiosity_enhanced_system_t* system,
    const char* topic) {

    if (!system || !topic || !system->quantum_bridge) return -1.0f;

    nimcp_platform_mutex_lock(system->mutex);

    float novelty = curiosity_quantum_evaluate_novelty(
        system->quantum_bridge,
        topic
    );

    nimcp_platform_mutex_unlock(system->mutex);

    return novelty;
}
