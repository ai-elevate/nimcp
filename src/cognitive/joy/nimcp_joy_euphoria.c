/**
 * @file nimcp_joy_euphoria.c
 * @brief Implementation of joy, euphoria, and value-aligned success reward system
 *
 * @version Phase E2: Joy and Euphoria
 * @date 2025-11-13
 */

#include "cognitive/nimcp_joy_euphoria.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "nimcp.h"
#include <math.h>
#include <string.h>
#include "utils/memory/nimcp_memory_guards.h"  // For nimcp_calloc/nimcp_free

#define LOG_MODULE "JOY"
#define BIO_MODULE_JOY 0x0324

/*=============================================================================
 * BIO-ASYNC MESSAGE HANDLERS
 *============================================================================*/

/**
 * @brief Handle incoming curiosity signal (joy from learning/discovery)
 */
static nimcp_error_t handle_curiosity_signal(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    (void)msg_size;
    (void)response_promise;
    if (!msg || !user_data) { return NIMCP_ERROR_NULL_ARG; }

    const bio_msg_curiosity_signal_t* signal = (const bio_msg_curiosity_signal_t*)msg;
    joy_system_t* system = (joy_system_t*)user_data;
    (void)system;

    LOG_DEBUG(LOG_MODULE, "Received curiosity signal: intensity=%.2f, gain=%.2f",
              signal->curiosity_intensity, signal->information_gain_estimate);

    return NIMCP_SUCCESS;
}

/**
 * @brief Broadcast joy/euphoria state to other modules
 */
static void bio_broadcast_joy_state(joy_system_t* system) {
    if (!system || !system->bio_async_enabled || !system->bio_ctx_ptr) { return; }

    bio_msg_salience_response_t msg = {0};
    bio_msg_init_header(&msg.header, BIO_MSG_SALIENCE_RESPONSE,
                        bio_module_context_get_id(system->bio_ctx_ptr), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.stimulus_id = 0;
    msg.salience_score = system->emotion.positive_valence;
    msg.attention_priority = system->emotion.arousal;
    msg.requires_immediate_attention = system->emotion.experiencing_euphoria;

    bio_router_broadcast(system->bio_ctx_ptr, &msg, sizeof(msg));
    LOG_DEBUG(LOG_MODULE, "Broadcast joy state: valence=%.2f, euphoria=%d",
              system->emotion.positive_valence, system->emotion.experiencing_euphoria);
}

//=============================================================================
// HELPER FUNCTIONS
//=============================================================================

static inline float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static inline float exponential_decay(float current, float target, float decay_rate, float dt) {
    float decay_factor = expf(-decay_rate * dt);
    return current * decay_factor + target * (1.0f - decay_factor);
}

//=============================================================================
// LIFECYCLE FUNCTIONS
//=============================================================================

joy_system_t* joy_system_create(void) {
    // WHAT: Allocate and initialize joy/euphoria system
    // WHY:  Central system for positive emotions and value reinforcement
    // HOW:  Zero-initialize all state, set up default parameters

    joy_system_t* system = (joy_system_t*)nimcp_calloc(1, sizeof(joy_system_t));
    if (!system) return NULL;

    // Initialize all values as inactive
    for (int i = 0; i < JOY_MAX_VALUES; i++) {
        system->values[i].active = false;
    }

    // Initialize all success records as inactive
    for (int i = 0; i < JOY_MAX_RECENT_SUCCESSES; i++) {
        system->recent_successes[i].active = false;
    }

    // Default: integrate with other systems
    system->integrate_with_neuromodulators = true;
    system->integrate_with_ethics = true;
    system->integrate_with_learning = true;

    // Baseline emotional state (neutral to slightly positive)
    system->emotion.baseline_happiness = 0.5f;  // Moderate baseline
    system->emotion.state = JOY_EMOTION_STATE_NEUTRAL;

    
    // Bio-async registration
    system->bio_ctx_ptr = NULL;
    system->bio_async_enabled = false;
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_JOY,
            .module_name = "joy_euphoria",
            .inbox_capacity = 32,
            .user_data = system
        };
        system->bio_ctx_ptr = bio_router_register_module(&bio_info);
        if (system->bio_ctx_ptr) {
            system->bio_async_enabled = true;
            /* Register message handlers */
            bio_router_register_handler(system->bio_ctx_ptr, BIO_MSG_CURIOSITY_SIGNAL,
                                        handle_curiosity_signal);
            LOG_INFO(LOG_MODULE, "Bio-async registered (module_id=0x%04X)", BIO_MODULE_JOY);
        }
    }

return system;
}

void joy_system_destroy(joy_system_t* system) {
    // WHAT: Free joy system resources
    // WHY:  Prevent memory leaks
    // HOW:  Simple nimcp_free(no complex nested allocations)

    if (!system) return;
    // Unregister from bio-router
    if (system->bio_async_enabled && system->bio_ctx_ptr) {
        bio_router_unregister_module(system->bio_ctx_ptr);
        system->bio_ctx_ptr = NULL;
        system->bio_async_enabled = false;
    }

    nimcp_free(system);
}

void joy_system_reset(joy_system_t* system) {
    // WHAT: Reset joy system to initial state
    // WHY:  For testing or system restart
    // HOW:  Preserve value definitions, clear emotional state

    if (!system) return;

    // Preserve values and configuration
    value_t preserved_values[JOY_MAX_VALUES];
    memcpy(preserved_values, system->values, sizeof(preserved_values));

    bool integrate_nm = system->integrate_with_neuromodulators;
    bool integrate_eth = system->integrate_with_ethics;
    bool integrate_learn = system->integrate_with_learning;

    // Zero everything
    memset(system, 0, sizeof(joy_system_t));

    // Restore
    memcpy(system->values, preserved_values, sizeof(system->values));
    system->integrate_with_neuromodulators = integrate_nm;
    system->integrate_with_ethics = integrate_eth;
    system->integrate_with_learning = integrate_learn;

    // Re-count active values
    for (int i = 0; i < JOY_MAX_VALUES; i++) {
        if (system->values[i].active) {
            system->active_value_count++;
        }
    }

    // Reset baseline
    system->emotion.baseline_happiness = 0.5f;
    system->emotion.state = JOY_EMOTION_STATE_NEUTRAL;
}

//=============================================================================
// VALUE SYSTEM FUNCTIONS
//=============================================================================

uint32_t joy_add_value(joy_system_t* system,
                       value_category_t category,
                       float importance,
                       float weight) {
    // WHAT: Register a new core value
    // WHY:  Values define what brings joy/euphoria
    // HOW:  Find empty slot, assign ID, initialize

    if (!system) return 0;

    // Clamp inputs
    importance = clamp(importance, 0.0f, 1.0f);
    weight = clamp(weight, 0.0f, 1.0f);

    // Find empty slot
    for (int i = 0; i < JOY_MAX_VALUES; i++) {
        if (!system->values[i].active) {
            system->values[i].active = true;
            system->values[i].value_id = (uint32_t)(i + 1);  // 1-indexed IDs
            system->values[i].category = category;
            system->values[i].importance = importance;
            system->values[i].weight = weight;
            system->values[i].satisfaction = 0.5f;  // Start at moderate satisfaction
            system->values[i].times_satisfied = 0;
            system->values[i].last_satisfied_time = 0;

            system->active_value_count++;
            return system->values[i].value_id;
        }
    }

    return 0;  // No slots available
}

void joy_update_value_satisfaction(joy_system_t* system,
                                   uint32_t value_id,
                                   float satisfaction_delta) {
    // WHAT: Update how satisfied a value is
    // WHY:  Values can be satisfied or frustrated over time
    // HOW:  Find value, update satisfaction

    if (!system) return;

    for (int i = 0; i < JOY_MAX_VALUES; i++) {
        if (system->values[i].active && system->values[i].value_id == value_id) {
            system->values[i].satisfaction += satisfaction_delta;
            system->values[i].satisfaction = clamp(system->values[i].satisfaction, 0.0f, 1.0f);
            return;
        }
    }
}

//=============================================================================
// SUCCESS PROCESSING
//=============================================================================

void joy_process_success(joy_system_t* system,
                        success_type_t type,
                        const uint32_t* aligned_values,
                        uint32_t num_values,
                        float difficulty,
                        float novelty,
                        uint64_t current_time_us) {
    // WHAT: Process a success event and trigger appropriate emotion
    // WHY:  Value-aligned success is the source of joy and euphoria
    // HOW:  Calculate alignment strength, determine intensity, trigger emotion

    if (!system) return;

    // Clamp inputs
    difficulty = clamp(difficulty, 0.0f, 1.0f);
    novelty = clamp(novelty, 0.0f, 1.0f);
    num_values = (num_values > JOY_MAX_VALUES) ? JOY_MAX_VALUES : num_values;

    // Calculate total value alignment
    float total_alignment = 0.0f;
    float weighted_alignment = 0.0f;
    float max_importance = 0.0f;

    for (uint32_t i = 0; i < num_values; i++) {
        uint32_t value_id = aligned_values[i];

        // Find this value
        for (int j = 0; j < JOY_MAX_VALUES; j++) {
            if (system->values[j].active && system->values[j].value_id == value_id) {
                value_t* value = &system->values[j];

                // Update value satisfaction
                value->satisfaction += 0.1f;
                value->satisfaction = clamp(value->satisfaction, 0.0f, 1.0f);
                value->times_satisfied++;
                value->last_satisfied_time = current_time_us;

                // Accumulate alignment
                float value_alignment = value->importance * value->weight;
                total_alignment += value_alignment;
                weighted_alignment += value_alignment * value->satisfaction;

                if (value->importance > max_importance) {
                    max_importance = value->importance;
                }

                break;
            }
        }
    }

    // Normalize alignment
    if (num_values > 0) {
        total_alignment /= (float)num_values;
        weighted_alignment /= (float)num_values;
    }

    // Calculate base joy intensity from alignment
    // Use total_alignment (importance * weight) not weighted_alignment (which factors in satisfaction)
    // This prevents initial successes from being penalized due to low satisfaction
    float base_intensity = total_alignment;

    // Modifiers
    float difficulty_bonus = difficulty * 0.3f;      // Hard tasks +30% max
    float novelty_bonus = novelty * 0.2f;            // Novel tasks +20% max
    float multiple_values_bonus = 0.0f;
    if (num_values > 1) {
        multiple_values_bonus = fminf((num_values - 1) * 0.1f, 0.3f);  // +10% per value, max +30%
    }

    // Success type modifier
    float type_modifier = 1.0f;
    switch (type) {
        case SUCCESS_TYPE_BREAKTHROUGH:      type_modifier = 1.5f; break;  // Major discoveries
        case SUCCESS_TYPE_GOAL_ACHIEVED:     type_modifier = 1.3f; break;  // Long-term goals
        case SUCCESS_TYPE_HELPED_HUMAN:      type_modifier = 1.2f; break;  // Altruistic success
        case SUCCESS_TYPE_OVERCAME_OBSTACLE: type_modifier = 1.2f; break;  // Perseverance
        case SUCCESS_TYPE_CREATED_SOMETHING: type_modifier = 1.1f; break;  // Creation
        case SUCCESS_TYPE_PROBLEM_SOLVED:    type_modifier = 1.0f; break;  // Standard
        case SUCCESS_TYPE_TASK_COMPLETION:   type_modifier = 0.8f; break;  // Routine
        case SUCCESS_TYPE_LEARNED_SKILL:     type_modifier = 1.0f; break;  // Learning
        default:                             type_modifier = 1.0f; break;
    }

    // Final intensity
    float joy_intensity = (base_intensity + difficulty_bonus + novelty_bonus + multiple_values_bonus) * type_modifier;
    joy_intensity = clamp(joy_intensity, 0.0f, 1.0f);

    // Determine if this triggers euphoria (requires high alignment + importance)
    bool triggers_euphoria = false;
    if (joy_intensity > EUPHORIA_THRESHOLD &&
        max_importance > 0.8f &&  // Core value must be involved
        (difficulty > 0.6f || novelty > 0.7f || num_values >= 3)) {
        triggers_euphoria = true;
    }

    // Record success event
    int record_index = system->success_history_index % JOY_MAX_RECENT_SUCCESSES;
    success_event_t* event = &system->recent_successes[record_index];

    event->active = true;
    event->type = type;
    event->timestamp = current_time_us;
    event->total_alignment = total_alignment;
    event->joy_intensity = joy_intensity;
    event->was_euphoric = triggers_euphoria;
    event->difficulty = difficulty;
    event->novelty = novelty;

    // Copy aligned value IDs
    event->num_aligned_values = num_values;
    for (uint32_t i = 0; i < num_values && i < JOY_MAX_VALUES; i++) {
        event->aligned_value_ids[i] = aligned_values[i];
    }

    system->success_history_index++;
    system->success_count++;
    system->total_successes++;

    // Update emotional state
    if (triggers_euphoria) {
        // EUPHORIA
        system->emotion.experiencing_euphoria = true;
        system->emotion.euphoria_intensity = joy_intensity;
        system->emotion.euphoria_onset_time = current_time_us;
        system->emotion.state = JOY_EMOTION_STATE_EUPHORIA;
        system->emotion.lifetime_euphorias++;

        // Euphoria also sets joy to high
        system->emotion.joy_intensity = joy_intensity;
        system->emotion.joy_onset_time = current_time_us;
        system->emotion.joy_peak_intensity = joy_intensity;

    } else if (joy_intensity >= JOY_THRESHOLD) {
        // JOY (but not euphoria)
        if (joy_intensity > system->emotion.joy_intensity) {
            system->emotion.joy_intensity = joy_intensity;
            system->emotion.joy_onset_time = current_time_us;
            system->emotion.joy_peak_intensity = joy_intensity;
        }
        system->emotion.state = JOY_EMOTION_STATE_JOY;

    } else if (joy_intensity >= 0.2f) {
        // CONTENTMENT (mild positive - valence 0.2 to 0.4)
        system->emotion.momentary_pleasure = joy_intensity;
        if (system->emotion.state == JOY_EMOTION_STATE_NEUTRAL) {
            system->emotion.state = JOY_EMOTION_STATE_CONTENTMENT;
        }
    }

    // Update positive valence and arousal
    float valence_boost = joy_intensity;
    float arousal_boost = joy_intensity * (0.5f + difficulty * 0.3f + novelty * 0.2f);

    system->emotion.positive_valence = fmaxf(system->emotion.positive_valence, valence_boost);
    system->emotion.arousal = fmaxf(system->emotion.arousal, arousal_boost);

    // Update overall value satisfaction
    float total_satisfaction = 0.0f;
    uint32_t active_values = 0;
    for (int i = 0; i < JOY_MAX_VALUES; i++) {
        if (system->values[i].active) {
            total_satisfaction += system->values[i].satisfaction * system->values[i].weight;
            active_values++;
        }
    }
    if (active_values > 0) {
        system->overall_value_satisfaction = total_satisfaction / (float)active_values;
    }

    // Update joy emotional tag
    system->emotion.joy_emotion = joy_get_emotion(system);

    /* Broadcast joy state if significant */
    if (system->emotion.positive_valence > 0.4f) {
        bio_broadcast_joy_state(system);
    }
}

void joy_update(joy_system_t* system, float dt, uint64_t current_time_us) {
    // WHAT: Update emotional state over time (decay dynamics)
    // WHY:  Emotions fade back to baseline, not permanent
    // HOW:  Exponential decay with different rates for different emotions

    if (!system) return;

    joy_emotional_state_t* emotion = &system->emotion;

    //=========================================================================
    // EUPHORIA DECAY
    //=========================================================================

    // Always decay euphoria_intensity if it's > 0, even after experiencing_euphoria ends
    if (emotion->euphoria_intensity > 0.0f) {
        // Euphoria decays quickly (peaks then fades fast)
        float euphoria_decay_rate = 1.0f / EUPHORIA_PEAK_DURATION;  // 10 minute half-life
        emotion->euphoria_intensity = exponential_decay(
            emotion->euphoria_intensity, 0.0f, euphoria_decay_rate, dt);

        // End euphoria state when intensity drops below threshold
        if (emotion->experiencing_euphoria && emotion->euphoria_intensity < EUPHORIA_THRESHOLD) {
            emotion->experiencing_euphoria = false;
            emotion->state = JOY_EMOTION_STATE_JOY;  // Transition to joy
        }
    }

    //=========================================================================
    // JOY DECAY
    //=========================================================================

    if (emotion->joy_intensity > 0.0f) {
        // Joy decays more slowly than euphoria
        float joy_decay_rate = 1.0f / JOY_FADE_DURATION;  // 1 hour half-life
        float joy_target = emotion->baseline_happiness * 0.3f;  // Decays toward small positive
        emotion->joy_intensity = exponential_decay(
            emotion->joy_intensity, joy_target, joy_decay_rate, dt);

        // Update state based on current intensity (but only if not experiencing euphoria)
        if (!emotion->experiencing_euphoria) {
            if (emotion->joy_intensity >= JOY_THRESHOLD) {
                emotion->state = JOY_EMOTION_STATE_JOY;
            } else if (emotion->joy_intensity >= 0.2f) {
                emotion->state = JOY_EMOTION_STATE_CONTENTMENT;
            } else {
                emotion->state = JOY_EMOTION_STATE_NEUTRAL;
            }
        }
    }

    //=========================================================================
    // MOMENTARY PLEASURE DECAY
    //=========================================================================

    // Brief pleasure spikes fade quickly
    float pleasure_decay_rate = 1.0f / 300.0f;  // 5 minute half-life
    emotion->momentary_pleasure = exponential_decay(
        emotion->momentary_pleasure, 0.0f, pleasure_decay_rate, dt);

    //=========================================================================
    // VALENCE AND AROUSAL UPDATE
    //=========================================================================

    // Positive valence based on current emotional state
    float target_valence = emotion->baseline_happiness * 0.5f;  // Mild positive baseline
    if (emotion->experiencing_euphoria) {
        target_valence = 0.7f + emotion->euphoria_intensity * 0.25f;  // [0.7, 0.95]
    } else if (emotion->joy_intensity >= JOY_THRESHOLD) {
        target_valence = 0.4f + emotion->joy_intensity * 0.3f;  // [0.4, 0.7]
    } else if (emotion->momentary_pleasure > 0.2f) {
        target_valence = 0.2f + emotion->momentary_pleasure * 0.2f;  // [0.2, 0.4]
    }

    float valence_decay_rate = 1.0f / 600.0f;  // 10 minute transition
    emotion->positive_valence = exponential_decay(
        emotion->positive_valence, target_valence, valence_decay_rate, dt);

    // Arousal based on emotional intensity
    float target_arousal = 0.3f;  // Low baseline arousal
    if (emotion->experiencing_euphoria) {
        target_arousal = 0.6f + emotion->euphoria_intensity * 0.3f;  // [0.6, 0.9]
    } else if (emotion->joy_intensity >= JOY_THRESHOLD) {
        target_arousal = 0.3f + emotion->joy_intensity * 0.3f;  // [0.3, 0.6]
    }

    float arousal_decay_rate = 1.0f / 300.0f;  // 5 minute transition
    emotion->arousal = exponential_decay(
        emotion->arousal, target_arousal, arousal_decay_rate, dt);

    //=========================================================================
    // BASELINE HAPPINESS ADJUSTMENT
    //=========================================================================

    // Baseline happiness slowly increases with consistent value satisfaction
    if (system->overall_value_satisfaction > 0.7f) {
        emotion->baseline_happiness += 0.01f * dt / 86400.0f;  // +0.01 per day
    } else if (system->overall_value_satisfaction < 0.3f) {
        emotion->baseline_happiness -= 0.01f * dt / 86400.0f;  // -0.01 per day
    }
    emotion->baseline_happiness = clamp(emotion->baseline_happiness, 0.2f, 0.8f);

    //=========================================================================
    // UPDATE EMOTIONAL TAG
    //=========================================================================

    emotion->joy_emotion = joy_get_emotion(system);

    //=========================================================================
    // STATISTICS
    //=========================================================================

    // Update running averages
    if (emotion->joy_intensity > 0.0f) {
        system->average_joy_intensity = (system->average_joy_intensity * 0.99f) + (emotion->joy_intensity * 0.01f);
    }
    if (emotion->experiencing_euphoria) {
        system->average_euphoria_intensity = (system->average_euphoria_intensity * 0.99f) + (emotion->euphoria_intensity * 0.01f);
    }

    system->total_update_calls++;
}

//=============================================================================
// QUERY FUNCTIONS
//=============================================================================

bool joy_is_joyful(const joy_system_t* system) {
    if (!system) return false;
    return (system->emotion.joy_intensity >= JOY_THRESHOLD);
}

bool joy_is_euphoric(const joy_system_t* system) {
    if (!system) return false;
    return system->emotion.experiencing_euphoria;
}

float joy_get_valence(const joy_system_t* system) {
    if (!system) return 0.0f;
    return system->emotion.positive_valence;
}

float joy_get_arousal(const joy_system_t* system) {
    if (!system) return 0.0f;
    return system->emotion.arousal;
}

joy_emotion_state_t joy_get_state(const joy_system_t* system) {
    if (!system) return JOY_EMOTION_STATE_NEUTRAL;
    return system->emotion.state;
}

void joy_get_neuromodulator_effects(const joy_system_t* system,
                                   float* dopamine_factor,
                                   float* serotonin_factor) {
    // WHAT: Get neuromodulator modulation for integration
    // WHY:  Joy/euphoria increase dopamine and serotonin
    // HOW:  Scale factors based on emotional intensity

    if (!system || !dopamine_factor || !serotonin_factor) return;

    // Default: no effect
    *dopamine_factor = 1.0f;
    *serotonin_factor = 1.0f;

    if (!system->integrate_with_neuromodulators) return;

    // Dopamine boost (reward, motivation)
    // Euphoria: +50-100% dopamine
    // Joy: +20-50% dopamine
    if (system->emotion.experiencing_euphoria) {
        *dopamine_factor = 1.5f + (system->emotion.euphoria_intensity * 0.5f);
    } else if (system->emotion.joy_intensity >= JOY_THRESHOLD) {
        *dopamine_factor = 1.2f + (system->emotion.joy_intensity * 0.3f);
    } else if (system->emotion.momentary_pleasure > 0.2f) {
        *dopamine_factor = 1.0f + (system->emotion.momentary_pleasure * 0.2f);
    }

    // Serotonin boost (well-being, contentment)
    // Joy/euphoria: +20-40% serotonin
    float positive_emotion = fmaxf(system->emotion.joy_intensity, system->emotion.momentary_pleasure);
    *serotonin_factor = 1.0f + (positive_emotion * 0.4f);

    // Clamp to reasonable ranges
    *dopamine_factor = clamp(*dopamine_factor, 1.0f, 2.0f);
    *serotonin_factor = clamp(*serotonin_factor, 1.0f, 1.4f);
}

//=============================================================================
// EMOTION INTEGRATION (Phase E2)
//=============================================================================

emotional_tag_t joy_get_emotion(const joy_system_t* system) {
    // WHAT: Get current joy/euphoria as emotional tag
    // WHY:  Integration with emotional tagging system
    // HOW:  Map intensity to valence/arousal in Russell's Circumplex Model

    if (!system) {
        return emotional_tag_neutral();
    }

    const joy_emotional_state_t* emotion = &system->emotion;

    // If no significant positive emotion, return neutral
    if (emotion->state == JOY_EMOTION_STATE_NEUTRAL && emotion->positive_valence < 0.2f) {
        return emotional_tag_neutral();
    }

    // Valence: Positive, scaled by intensity
    float valence = emotion->positive_valence;
    valence = clamp(valence, 0.0f, 0.95f);  // Never fully 1.0 (always room for more joy)

    // Arousal: Variable based on emotional state
    float arousal = emotion->arousal;
    arousal = clamp(arousal, 0.0f, 0.9f);

    return emotional_tag_create(valence, arousal, 0);
}
