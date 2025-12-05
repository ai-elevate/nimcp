/**
 * @file nimcp_emotional_system.c
 * @brief Phase 10.2: Integrated Emotional System Implementation
 *
 * WHAT: Centralized emotional processing coordinating tagging, recognition, and regulation
 * WHY:  Emotions enhance memory, guide attention, enable empathy, and ensure ethical behavior
 * HOW:  Integrate emotional_tagging, emotion_recognition, shadow_emotions with unified API
 *
 * IMPLEMENTATION NOTES:
 * - Uses Russell's Circumplex Model for dimensional emotions (valence × arousal)
 * - Integrates Ekman's categorical emotions for recognition
 * - Implements CBT/DBT regulation strategies
 * - Tracks shadow emotions (Dark Triad patterns)
 * - Exponential decay for natural emotion fading
 * - Auto-regulation when intensity exceeds threshold
 *
 * @author NIMCP Development Team
 * @date 2025-11-15
 * @version 1.0.0
 */

#include "cognitive/nimcp_emotional_system.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "cognitive/nimcp_emotional_tagging.h"
#include "nimcp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils/memory/nimcp_memory_guards.h"  // For nimcp_calloc/nimcp_free

#define LOG_MODULE "EMOTIONS"
#define BIO_MODULE_EMOTIONS 0x0320

//=============================================================================
// Constants
//=============================================================================

#define EMOTION_DECAY_DEFAULT 0.05f         /**< Default decay rate per second */
#define AROUSAL_SENSITIVITY_DEFAULT 1.2f    /**< Default arousal multiplier */
#define VALENCE_SENSITIVITY_DEFAULT 1.0f    /**< Default valence multiplier */
#define REGULATION_THRESHOLD_DEFAULT 0.75f  /**< Default regulation trigger */
#define MAX_SHADOW_TRACKED_DEFAULT 10       /**< Default max shadow emotions */
#define SHADOW_THRESHOLD_DEFAULT 0.6f       /**< Default shadow intervention */
#define REGULATION_STRENGTH 0.3f            /**< How much regulation reduces intensity */

//=============================================================================
// Internal Structure
//=============================================================================

/**
 * @brief Emotional system internal structure
 */
struct emotional_system {
    // === Configuration ===
    emotion_config_t config;

    // === Current State ===
    emotion_state_t state;

    // === Statistics ===
    emotion_stats_t stats;

    // === Bio-Async Integration ===
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;
};

/*=============================================================================
 * BIO-ASYNC MESSAGE HANDLERS
 *============================================================================*/

/**
 * @brief Handle incoming salience query for emotional boost
 */
static nimcp_error_t handle_salience_query(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    (void)msg_size;
    (void)response_promise;
    if (!msg || !user_data) { return NIMCP_ERROR_NULL_ARG; }

    const bio_msg_salience_query_t* query = (const bio_msg_salience_query_t*)msg;
    emotional_system_t* system = (emotional_system_t*)user_data;

    LOG_DEBUG(LOG_MODULE, "Received salience query: stimulus=%u, intensity=%.2f",
              query->stimulus_id, query->raw_intensity);

    /* Could respond with emotional boost for salience calculation */
    float boost = emotion_system_get_salience_boost(system);
    LOG_DEBUG(LOG_MODULE, "Emotional salience boost: %.2f", boost);

    return NIMCP_SUCCESS;
}

/**
 * @brief Broadcast emotional state change to other modules
 */
static void bio_broadcast_emotion_state(emotional_system_t* system) {
    if (!system || !system->bio_async_enabled || !system->bio_ctx) { return; }

    bio_msg_salience_response_t msg = {0};
    bio_msg_init_header(&msg.header, BIO_MSG_SALIENCE_RESPONSE,
                        bio_module_context_get_id(system->bio_ctx), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.stimulus_id = 0;
    msg.salience_score = system->state.intensity;
    msg.attention_priority = system->state.arousal;
    msg.requires_immediate_attention = (system->state.intensity > 0.7f);

    bio_router_broadcast(system->bio_ctx, &msg, sizeof(msg));
    LOG_DEBUG(LOG_MODULE, "Broadcast emotion state: intensity=%.2f, arousal=%.2f",
              system->state.intensity, system->state.arousal);
}

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Clamp value to range [min, max]
 *
 * WHAT: Ensure value stays within bounds
 * WHY:  Prevent invalid emotional states
 * HOW:  Return min if too low, max if too high, value otherwise
 */
static float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief Calculate emotional intensity from valence and arousal
 *
 * WHAT: Compute overall emotional intensity
 * WHY:  Combine valence extremity with arousal
 * HOW:  intensity = arousal * (1 + |valence|) / 2
 */
static float calculate_intensity(float valence, float arousal) {
    float valence_extremity = fabsf(valence);
    return arousal * (1.0f + valence_extremity) / 2.0f;
}

/**
 * @brief Update running average
 *
 * WHAT: Exponential moving average
 * WHY:  Track average emotional state over time
 * HOW:  new_avg = old_avg * 0.9 + new_value * 0.1
 */
static float update_average(float old_avg, float new_value, uint64_t count) {
    if (count == 0) {
        return new_value;
    }

    // Use exponential moving average with alpha = 0.1
    return old_avg * 0.9f + new_value * 0.1f;
}

/**
 * @brief Calculate emotional stability
 *
 * WHAT: Measure how stable emotions are over time
 * WHY:  Rapid changes indicate instability
 * HOW:  Compare current to average, return 1.0 - difference
 */
static float calculate_stability(float current_valence, float current_arousal,
                                float avg_valence, float avg_arousal) {
    float valence_diff = fabsf(current_valence - avg_valence);
    float arousal_diff = fabsf(current_arousal - avg_arousal);
    float total_diff = (valence_diff + arousal_diff) / 2.0f;

    return clamp(1.0f - total_diff, 0.0f, 1.0f);
}

//=============================================================================
// Lifecycle API Implementation
//=============================================================================

emotion_config_t emotion_system_default_config(void) {
    emotion_config_t config = {0};

    // Core features enabled by default
    config.enable_emotion_recognition = true;
    config.enable_emotional_tagging = true;
    config.enable_shadow_detection = true;
    config.enable_emotion_regulation = true;

    // Integration features enabled
    config.integrate_with_memory = true;
    config.integrate_with_salience = true;
    config.integrate_with_mental_health = true;
    config.integrate_with_ethics = true;

    // Regulation parameters
    config.emotion_decay_rate = EMOTION_DECAY_DEFAULT;
    config.arousal_sensitivity = AROUSAL_SENSITIVITY_DEFAULT;
    config.valence_sensitivity = VALENCE_SENSITIVITY_DEFAULT;
    config.regulation_threshold = REGULATION_THRESHOLD_DEFAULT;

    // Shadow emotion limits
    config.max_shadow_tracked = MAX_SHADOW_TRACKED_DEFAULT;
    config.shadow_intervention_threshold = SHADOW_THRESHOLD_DEFAULT;

    return config;
}

emotional_system_t* emotion_system_create(const emotion_config_t* config) {
    LOG_DEBUG(LOG_MODULE, "emotion_system_create called");

    // Allocate system with unified memory
    emotional_system_t* system = (emotional_system_t*)nimcp_calloc(1, sizeof(emotional_system_t));
    if (!system) {
        LOG_ERROR(LOG_MODULE, "Failed to allocate emotional system");
        return NULL;
    }
    LOG_DEBUG(LOG_MODULE, "Allocated %zu bytes for emotional system", sizeof(emotional_system_t));

    // Use provided config or defaults
    if (config) {
        system->config = *config;
        LOG_DEBUG(LOG_MODULE, "Using provided configuration");
    } else {
        system->config = emotion_system_default_config();
        LOG_DEBUG(LOG_MODULE, "Using default configuration");
    }

    // Initialize state to neutral
    memset(&system->state, 0, sizeof(emotion_state_t));
    system->state.valence = 0.0f;
    system->state.arousal = 0.0f;
    system->state.intensity = 0.0f;
    system->state.emotional_stability = 1.0f;
    LOG_INFO(LOG_MODULE, "Initialized emotional state to neutral (valence=0.0, arousal=0.0)");

    // Initialize statistics
    memset(&system->stats, 0, sizeof(emotion_stats_t));
    LOG_DEBUG(LOG_MODULE, "Initialized statistics");

    // Initialize bio-async
    system->bio_ctx = NULL;
    system->bio_async_enabled = false;
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_EMOTIONS,
            .module_name = "emotions",
            .inbox_capacity = 64,
            .user_data = system
        };
        system->bio_ctx = bio_router_register_module(&bio_info);
        if (system->bio_ctx) {
            system->bio_async_enabled = true;
            /* Register message handlers */
            bio_router_register_handler(system->bio_ctx, BIO_MSG_SALIENCE_QUERY,
                                        handle_salience_query);
            LOG_INFO(LOG_MODULE, "Bio-async registered (module_id=0x%04X)", BIO_MODULE_EMOTIONS);
        }
    }

    LOG_INFO(LOG_MODULE, "Emotional system created successfully");
    return system;
}

void emotion_system_destroy(emotional_system_t* system) {
    if (!system) {
        LOG_WARN(LOG_MODULE, "emotion_system_destroy called with NULL system");
        return;
    }

    LOG_DEBUG(LOG_MODULE, "Destroying emotional system");

    // Unregister from bio-router
    if (system->bio_async_enabled && system->bio_ctx) {
        bio_router_unregister_module(system->bio_ctx);
        system->bio_ctx = NULL;
        LOG_DEBUG(LOG_MODULE, "Unregistered from bio-router");
    }

    LOG_INFO(LOG_MODULE, "Emotional system destroyed (total_updates=%lu, total_regulations=%lu)",
             system->stats.total_updates, system->stats.total_regulations);

    nimcp_free(system);
}

//=============================================================================
// State Query API Implementation
//=============================================================================

bool emotion_system_get_state(const emotional_system_t* system, emotion_state_t* state) {
    if (!system || !state) {
        return false;
    }

    *state = system->state;
    return true;
}

bool emotion_system_get_tag(const emotional_system_t* system, emotional_tag_t* tag) {
    if (!system || !tag) {
        return false;
    }

    // Convert emotional state to tag
    tag->valence = system->state.valence;
    tag->arousal = system->state.arousal;

    return true;
}

bool emotion_system_is_active(const emotional_system_t* system, uint32_t emotion_id,
                              float threshold) {
    if (!system) {
        return false;
    }

    // Check if the specified emotion is dominant with sufficient confidence
    if (system->state.dominant_emotion == emotion_id &&
        system->state.emotion_confidence >= threshold) {
        return true;
    }

    return false;
}

//=============================================================================
// Update API Implementation
//=============================================================================

bool emotion_system_set_state(emotional_system_t* system, float valence, float arousal,
                              uint64_t timestamp_ms) {
    if (!system) {
        LOG_ERROR(LOG_MODULE, "emotion_system_set_state called with NULL system");
        return false;
    }

    LOG_DEBUG(LOG_MODULE, "Setting emotional state: valence=%.3f, arousal=%.3f", valence, arousal);

    // Clamp inputs to valid ranges
    valence = clamp(valence, -1.0f, 1.0f);
    arousal = clamp(arousal, 0.0f, 1.0f);

    // Update state
    system->state.valence = valence * system->config.valence_sensitivity;
    system->state.arousal = arousal * system->config.arousal_sensitivity;
    system->state.valence = clamp(system->state.valence, -1.0f, 1.0f);
    system->state.arousal = clamp(system->state.arousal, 0.0f, 1.0f);

    // Recalculate intensity
    system->state.intensity = calculate_intensity(system->state.valence,
                                                  system->state.arousal);

    LOG_INFO(LOG_MODULE, "Emotional state updated: valence=%.3f, arousal=%.3f, intensity=%.3f",
             system->state.valence, system->state.arousal, system->state.intensity);

    // Update timestamp
    system->state.last_update_ms = timestamp_ms;

    // Update statistics
    system->stats.total_updates++;
    system->stats.avg_valence = update_average(system->stats.avg_valence,
                                              system->state.valence,
                                              system->stats.total_updates);
    system->stats.avg_arousal = update_average(system->stats.avg_arousal,
                                              system->state.arousal,
                                              system->stats.total_updates);

    // Update stability
    system->state.emotional_stability = calculate_stability(
        system->state.valence, system->state.arousal,
        system->stats.avg_valence, system->stats.avg_arousal
    );

    system->stats.avg_stability = update_average(system->stats.avg_stability,
                                                 system->state.emotional_stability,
                                                 system->stats.total_updates);

    LOG_DEBUG(LOG_MODULE, "Statistics: total_updates=%lu, stability=%.3f",
              system->stats.total_updates, system->state.emotional_stability);

    /* Broadcast emotional state change if significant */
    if (system->state.intensity > 0.4f) {
        bio_broadcast_emotion_state(system);
    }

    return true;
}

bool emotion_system_decay(emotional_system_t* system, float delta_time,
                         uint64_t current_time_ms) {
    if (!system) {
        return false;
    }

    // Apply exponential decay to arousal
    float decay_factor = expf(-system->config.emotion_decay_rate * delta_time);
    system->state.arousal *= decay_factor;

    // Clamp to prevent negative values
    system->state.arousal = clamp(system->state.arousal, 0.0f, 1.0f);

    // Recalculate intensity
    system->state.intensity = calculate_intensity(system->state.valence,
                                                  system->state.arousal);

    // Update timestamp
    system->state.last_update_ms = current_time_ms;

    return true;
}

bool emotion_system_update_multimodal(emotional_system_t* system,
                                      const float* visual_data, uint32_t visual_dim,
                                      const float* audio_data, uint32_t audio_dim,
                                      const char* text, uint64_t timestamp_ms) {
    if (!system) {
        return false;
    }

    // For now, simple placeholder implementation
    // In full implementation, would:
    // 1. Pass to emotion_recognition subsystem
    // 2. Extract emotional features from each modality
    // 3. Fuse multimodal features
    // 4. Update emotional state based on recognition

    // Placeholder: if any input provided, generate small arousal bump
    if (visual_data || audio_data || text) {
        float current_arousal = system->state.arousal;
        float new_arousal = current_arousal + 0.05f;
        emotion_system_set_state(system, system->state.valence, new_arousal, timestamp_ms);
    }

    return true;
}

//=============================================================================
// Regulation API Implementation
//=============================================================================

bool emotion_system_regulate(emotional_system_t* system, uint32_t strategy) {
    if (!system) {
        return false;
    }

    if (!system->config.enable_emotion_regulation) {
        return false;
    }

    // Apply regulation strategy
    // Strategy 0 = Reappraisal (reduce negative valence)
    // Strategy 1 = Suppression (reduce arousal)
    // Strategy 2 = Distraction (reduce both)

    system->stats.total_regulations++;

    switch (strategy) {
        case 0:  // Reappraisal
            if (system->state.valence < 0.0f) {
                system->state.valence *= (1.0f - REGULATION_STRENGTH);
            }
            break;

        case 1:  // Suppression
            system->state.arousal *= (1.0f - REGULATION_STRENGTH);
            break;

        case 2:  // Distraction
            if (system->state.valence < 0.0f) {
                system->state.valence *= (1.0f - REGULATION_STRENGTH);
            }
            system->state.arousal *= (1.0f - REGULATION_STRENGTH);
            break;

        default:
            return false;
    }

    // Recalculate intensity
    system->state.intensity = calculate_intensity(system->state.valence,
                                                  system->state.arousal);

    // Mark as regulating
    system->state.in_self_regulation = true;
    system->stats.successful_regulations++;

    return true;
}

bool emotion_system_auto_regulate(emotional_system_t* system) {
    if (!system) {
        return false;
    }

    if (!system->config.enable_emotion_regulation) {
        return false;
    }

    // Check if regulation is needed
    bool needs_regulation = false;

    // Trigger on high intensity
    if (system->state.intensity >= system->config.regulation_threshold) {
        needs_regulation = true;
    }

    // Trigger on high shadow intensity
    if (system->state.shadow_intensity >= system->config.shadow_intervention_threshold) {
        needs_regulation = true;
    }

    if (!needs_regulation) {
        system->state.in_self_regulation = false;
        return false;
    }

    // Apply regulation
    // Choose strategy based on state
    uint32_t strategy;
    if (system->state.valence < -0.5f && system->state.arousal > 0.7f) {
        strategy = 2;  // Distraction for severe negative high-arousal
    } else if (system->state.valence < 0.0f) {
        strategy = 0;  // Reappraisal for negative valence
    } else {
        strategy = 1;  // Suppression for high arousal
    }

    return emotion_system_regulate(system, strategy);
}

//=============================================================================
// Integration API Implementation
//=============================================================================

float emotion_system_get_salience_boost(const emotional_system_t* system) {
    if (!system) {
        return 1.0f;  // No boost if NULL
    }

    // High arousal increases salience
    // Boost range: [1.0, 2.0]
    float boost = 1.0f + system->state.arousal * system->config.arousal_sensitivity;

    return clamp(boost, 1.0f, 3.0f);
}

float emotion_system_get_memory_priority(const emotional_system_t* system) {
    if (!system) {
        return 0.0f;
    }

    // Memory priority based on emotional intensity
    // More intense emotions → better memory encoding
    // Priority = (arousal + |valence|) / 2

    float valence_extremity = fabsf(system->state.valence);
    float priority = (system->state.arousal + valence_extremity) / 2.0f;

    return clamp(priority, 0.0f, 1.0f);
}

float emotion_system_get_mental_health_impact(const emotional_system_t* system) {
    if (!system) {
        return 0.0f;
    }

    // Mental health impact based on:
    // 1. Extreme negative valence
    // 2. High sustained arousal
    // 3. Low stability
    // 4. Shadow emotion intensity

    float negative_valence_impact = 0.0f;
    if (system->state.valence < 0.0f) {
        negative_valence_impact = fabsf(system->state.valence);
    }

    float arousal_impact = system->state.arousal;
    float stability_impact = 1.0f - system->state.emotional_stability;
    float shadow_impact = system->state.shadow_intensity;

    // Weighted combination
    float impact = (negative_valence_impact * 0.3f +
                   arousal_impact * 0.2f +
                   stability_impact * 0.2f +
                   shadow_impact * 0.3f);

    return clamp(impact, 0.0f, 1.0f);
}

//=============================================================================
// Statistics API Implementation
//=============================================================================

bool emotion_system_get_stats(const emotional_system_t* system, emotion_stats_t* stats) {
    if (!system || !stats) {
        return false;
    }

    *stats = system->stats;
    return true;
}
