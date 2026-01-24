/**
 * @file nimcp_mirror_resonance.c
 * @brief Motor Resonance Implementation for Mirror Neurons
 * @version 1.0.0
 * @date 2025-11-25
 *
 * WHAT: Implementation of motor resonance with suppression circuits
 * WHY:  Enable automatic imitation tendency with appropriate behavioral control
 * HOW:  Track resonance per channel, apply BG/PFC gating, detect conflicts
 *
 * @see nimcp_mirror_resonance.h for API documentation
 */

#include "cognitive/mirror_neurons/nimcp_mirror_resonance.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
// Internal Constants
//=============================================================================

#define RESONANCE_ACTIVE_THRESHOLD  0.1f
#define MAX_CONFLICTS_PER_CHANNEL   8

//=============================================================================
// Internal Structure Definition
//=============================================================================

/**
 * @brief Conflict relationship
 */
typedef struct {
    uint32_t channel_a;
    uint32_t channel_b;
} conflict_pair_t;

/**
 * @brief Internal motor resonance system state
 */
struct motor_resonance_system {
    // Configuration
    motor_resonance_config_t config;

    // Channel storage
    motor_channel_t* channels;
    uint32_t max_channels;
    uint32_t num_channels;

    // Action-to-channel mapping
    uint32_t* action_map;
    uint32_t action_map_size;

    // Global BG inhibition
    float bg_inhibition;
    float bg_adaptation_state;

    // Conflict relationships
    conflict_pair_t* conflicts;
    uint32_t max_conflicts;
    uint32_t num_conflicts;

    // Global state
    uint64_t current_time_us;
    float social_context;
    float learning_context;

    // Statistics
    uint32_t total_activations;
    uint32_t bg_suppressions;
    uint32_t pfc_suppressions;
    uint32_t conflict_suppressions;
    uint32_t learning_releases;
    uint32_t social_releases;
    uint32_t voluntary_releases;
};

//=============================================================================
// Helper Functions
//=============================================================================

static inline float clamp_f(float val, float min_val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

static inline float exp_decay(float dt_ms, float tau_ms) {
    return expf(-dt_ms / tau_ms);
}

static inline uint32_t hash_action(uint32_t action_id, uint32_t map_size) {
    uint32_t h = action_id;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h % map_size;
}

/**
 * @brief Check if two channels are in conflict
 */
static bool channels_conflict(motor_resonance_t resonance, uint32_t ch_a, uint32_t ch_b) {
    for (uint32_t i = 0; i < resonance->num_conflicts; i++) {
        if ((resonance->conflicts[i].channel_a == ch_a &&
             resonance->conflicts[i].channel_b == ch_b) ||
            (resonance->conflicts[i].channel_a == ch_b &&
             resonance->conflicts[i].channel_b == ch_a)) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Compute conflict signal for a channel
 */
static float compute_conflict_signal(motor_resonance_t resonance, uint32_t channel_id) {
    motor_channel_t* ch = &resonance->channels[channel_id];
    float max_conflict = 0.0F;
    uint32_t conflicting_id = UINT32_MAX;

    // Check all other active channels for conflicts
    for (uint32_t i = 0; i < resonance->num_channels; i++) {
        if (i == channel_id) continue;

        motor_channel_t* other = &resonance->channels[i];

        // Only consider channels with significant resonance
        if (other->resonance_level < RESONANCE_ACTIVE_THRESHOLD) continue;

        // Check if they conflict
        if (channels_conflict(resonance, channel_id, i)) {
            // Conflict signal is product of resonance levels
            float conflict = ch->resonance_level * other->resonance_level;
            if (conflict > max_conflict) {
                max_conflict = conflict;
                conflicting_id = i;
            }
        }
    }

    ch->conflict_signal = max_conflict;
    ch->conflicting_channel = conflicting_id;

    return max_conflict;
}

//=============================================================================
// Lifecycle Management
//=============================================================================

motor_resonance_config_t motor_resonance_get_default_config(void) {
    motor_resonance_config_t config = {
        // Resonance parameters
        .resonance_gain = NIMCP_RESONANCE_DEFAULT_GAIN,
        .tau_resonance_decay = NIMCP_RESONANCE_TAU_DECAY,
        .tau_resonance_rise = 20.0F,
        .execution_threshold = NIMCP_RESONANCE_EXEC_THRESHOLD,

        // Basal ganglia parameters
        .bg_tonic_inhibition = NIMCP_RESONANCE_BG_TONIC_INHIB,
        .tau_bg_adaptation = NIMCP_RESONANCE_TAU_BG_ADAPT,
        .bg_release_rate = 0.1F,

        // Conflict detection
        .conflict_threshold = NIMCP_RESONANCE_CONFLICT_THRESH,
        .conflict_decay = 0.05F,

        // Voluntary control
        .pfc_gain = 1.0F,
        .enable_voluntary_override = true,

        // Fatigue modeling
        .enable_fatigue = true,
        .fatigue_threshold = 0.8F,
        .tau_fatigue_recovery = 5000.0F,

        // Social modulation
        .enable_social_context = true,
        .social_gain = 1.2F
    };
    return config;
}

motor_resonance_t motor_resonance_create(const motor_resonance_config_t* config,
                                          uint32_t max_channels) {
    LOG_DEBUG("Creating motor resonance system with max_channels=%u", max_channels);

    if (max_channels == 0 || max_channels > NIMCP_RESONANCE_MAX_CHANNELS) {
        LOG_ERROR("Invalid max_channels: %u (max allowed: %u)", max_channels, NIMCP_RESONANCE_MAX_CHANNELS);
        return NULL;
    }

    motor_resonance_t resonance = (motor_resonance_t)nimcp_calloc(1, sizeof(struct motor_resonance_system));
    if (!resonance) {
        LOG_ERROR("Failed to allocate motor resonance system");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "resonance is NULL");

        return NULL;
    }

    // Copy configuration
    if (config) {
        resonance->config = *config;
    } else {
        resonance->config = motor_resonance_get_default_config();
    }

    // Allocate channel storage
    resonance->channels = (motor_channel_t*)nimcp_calloc(max_channels, sizeof(motor_channel_t));
    if (!resonance->channels) {
        LOG_ERROR("Failed to allocate channel storage");
        nimcp_free(resonance);
        return NULL;
    }
    resonance->max_channels = max_channels;
    resonance->num_channels = 0;

    // Allocate action map
    resonance->action_map_size = max_channels * 2;
    resonance->action_map = (uint32_t*)nimcp_malloc(resonance->action_map_size * sizeof(uint32_t));
    if (!resonance->action_map) {
        LOG_ERROR("Failed to allocate action map");
        nimcp_free(resonance->channels);
        nimcp_free(resonance);
        return NULL;
    }
    for (uint32_t i = 0; i < resonance->action_map_size; i++) {
        resonance->action_map[i] = UINT32_MAX;
    }

    // Allocate conflict storage
    resonance->max_conflicts = max_channels * MAX_CONFLICTS_PER_CHANNEL / 2;
    resonance->conflicts = (conflict_pair_t*)nimcp_calloc(resonance->max_conflicts, sizeof(conflict_pair_t));
    if (!resonance->conflicts) {
        LOG_ERROR("Failed to allocate conflict storage");
        nimcp_free(resonance->action_map);
        nimcp_free(resonance->channels);
        nimcp_free(resonance);
        return NULL;
    }
    resonance->num_conflicts = 0;

    // Initialize global state
    resonance->bg_inhibition = resonance->config.bg_tonic_inhibition;
    resonance->bg_adaptation_state = 0.0F;
    resonance->social_context = 0.0F;
    resonance->learning_context = 0.0F;

    LOG_INFO("Motor resonance system created successfully (channels=%u, conflicts=%u)",
             max_channels, resonance->max_conflicts);

    return resonance;
}

void motor_resonance_destroy(motor_resonance_t resonance) {
    if (!resonance) return;

    LOG_DEBUG("Destroying motor resonance system (channels=%u)", resonance->num_channels);

    if (resonance->conflicts) nimcp_free(resonance->conflicts);
    if (resonance->action_map) nimcp_free(resonance->action_map);
    if (resonance->channels) nimcp_free(resonance->channels);
    nimcp_free(resonance);
}

//=============================================================================
// Channel Management
//=============================================================================

uint32_t motor_resonance_create_channel(motor_resonance_t resonance, uint32_t action_id) {
    if (!resonance || resonance->num_channels >= resonance->max_channels) {
        return UINT32_MAX;
    }

    // Find slot in action map
    uint32_t hash = hash_action(action_id, resonance->action_map_size);
    uint32_t slot = hash;
    uint32_t attempts = 0;

    while (resonance->action_map[slot] != UINT32_MAX && attempts < resonance->action_map_size) {
        // Check if action already exists
        if (resonance->channels[resonance->action_map[slot]].action_id == action_id) {
            return resonance->action_map[slot];
        }
        slot = (slot + 1) % resonance->action_map_size;
        attempts++;
    }

    if (attempts >= resonance->action_map_size) {
        return UINT32_MAX;
    }

    // Create new channel
    uint32_t channel_id = resonance->num_channels++;
    motor_channel_t* ch = &resonance->channels[channel_id];

    // Initialize channel
    ch->channel_id = channel_id;
    ch->action_id = action_id;
    ch->resonance_level = 0.0F;
    ch->peak_resonance = 0.0F;
    ch->target_resonance = 0.0F;
    ch->suppression_level = resonance->bg_inhibition;
    ch->suppress_reason = RESONANCE_SUPPRESS_BG_TONIC;
    ch->release_reason = RESONANCE_RELEASE_NONE;
    ch->motor_output = 0.0F;
    ch->above_threshold = false;
    ch->conflict_signal = 0.0F;
    ch->conflicting_channel = UINT32_MAX;
    ch->fatigue_level = 0.0F;
    ch->last_activation_us = 0;
    ch->activation_count = 0;
    ch->suppression_count = 0;
    ch->total_resonance_time = 0.0F;

    // Add to action map
    resonance->action_map[slot] = channel_id;

    return channel_id;
}

bool motor_resonance_get_channel(motor_resonance_t resonance, uint32_t channel_id,
                                  motor_channel_t* out_channel) {
    if (!resonance || !out_channel || channel_id >= resonance->num_channels) {
        return false;
    }

    *out_channel = resonance->channels[channel_id];
    return true;
}

uint32_t motor_resonance_find_channel(motor_resonance_t resonance, uint32_t action_id) {
    if (!resonance) return UINT32_MAX;

    uint32_t hash = hash_action(action_id, resonance->action_map_size);
    uint32_t slot = hash;
    uint32_t attempts = 0;

    while (resonance->action_map[slot] != UINT32_MAX && attempts < resonance->action_map_size) {
        uint32_t ch_id = resonance->action_map[slot];
        if (resonance->channels[ch_id].action_id == action_id) {
            return ch_id;
        }
        slot = (slot + 1) % resonance->action_map_size;
        attempts++;
    }

    return UINT32_MAX;
}

//=============================================================================
// Resonance Input
//=============================================================================

float motor_resonance_observe(motor_resonance_t resonance, uint32_t channel_id,
                               float observation_strength, uint64_t timestamp_us) {
    if (!resonance || channel_id >= resonance->num_channels) {
        return 0.0F;
    }

    motor_channel_t* ch = &resonance->channels[channel_id];
    const motor_resonance_config_t* cfg = &resonance->config;

    resonance->current_time_us = timestamp_us;

    // Compute target resonance from observation
    float scaled_obs = observation_strength * cfg->resonance_gain;

    // Apply social context boost
    if (cfg->enable_social_context && resonance->social_context > 0.0F) {
        scaled_obs *= (1.0F + resonance->social_context * (cfg->social_gain - 1.0F));
    }

    // Apply fatigue penalty
    if (cfg->enable_fatigue && ch->fatigue_level > cfg->fatigue_threshold) {
        float fatigue_penalty = (ch->fatigue_level - cfg->fatigue_threshold) /
                               (1.0F - cfg->fatigue_threshold);
        scaled_obs *= (1.0F - fatigue_penalty * 0.5F);  // Max 50% reduction
    }

    ch->target_resonance = clamp_f(scaled_obs, 0.0F, NIMCP_RESONANCE_MAX);

    // Instant update toward target (for responsiveness)
    float rise_factor = 0.3F;  // Portion to move instantly
    ch->resonance_level += (ch->target_resonance - ch->resonance_level) * rise_factor;
    ch->resonance_level = clamp_f(ch->resonance_level, 0.0F, NIMCP_RESONANCE_MAX);

    // Track peak
    if (ch->resonance_level > ch->peak_resonance) {
        ch->peak_resonance = ch->resonance_level;
    }

    // Update fatigue
    if (cfg->enable_fatigue && ch->resonance_level > 0.5F) {
        ch->fatigue_level += 0.01F;
        ch->fatigue_level = clamp_f(ch->fatigue_level, 0.0F, 1.0F);
    }

    return ch->resonance_level;
}

void motor_resonance_observe_batch(motor_resonance_t resonance,
                                    const uint32_t* channel_ids,
                                    const float* strengths,
                                    uint32_t count,
                                    uint64_t timestamp_us) {
    if (!resonance || !channel_ids || !strengths) return;

    for (uint32_t i = 0; i < count; i++) {
        motor_resonance_observe(resonance, channel_ids[i], strengths[i], timestamp_us);
    }
}

//=============================================================================
// Suppression Control
//=============================================================================

void motor_resonance_set_bg_inhibition(motor_resonance_t resonance, float level) {
    if (!resonance) return;

    resonance->bg_inhibition = clamp_f(level, 0.0F, 1.0F);

    // Update all channels with BG suppression
    for (uint32_t i = 0; i < resonance->num_channels; i++) {
        motor_channel_t* ch = &resonance->channels[i];
        if (ch->suppress_reason == RESONANCE_SUPPRESS_BG_TONIC ||
            ch->suppress_reason == RESONANCE_SUPPRESS_NONE) {
            ch->suppression_level = resonance->bg_inhibition;
            if (resonance->bg_inhibition > 0.1F) {
                ch->suppress_reason = RESONANCE_SUPPRESS_BG_TONIC;
            }
        }
    }
}

void motor_resonance_set_pfc_suppression(motor_resonance_t resonance,
                                          uint32_t channel_id, float level) {
    if (!resonance || channel_id >= resonance->num_channels) return;

    motor_channel_t* ch = &resonance->channels[channel_id];

    level = clamp_f(level, 0.0F, 1.0F);

    // PFC suppression takes priority over BG
    if (level > 0.1F) {
        ch->suppression_level = fmaxf(ch->suppression_level, level * resonance->config.pfc_gain);
        ch->suppress_reason = RESONANCE_SUPPRESS_PFC_VOLUNTARY;
        ch->suppression_count++;
        resonance->pfc_suppressions++;
    }
}

void motor_resonance_release_for_learning(motor_resonance_t resonance,
                                           int32_t channel_id, float learning_context) {
    if (!resonance) return;

    resonance->learning_context = clamp_f(learning_context, 0.0F, 1.0F);

    // Reduce suppression proportional to learning context
    float release_amount = learning_context * resonance->config.bg_release_rate;

    if (channel_id < 0) {
        // Release all channels
        for (uint32_t i = 0; i < resonance->num_channels; i++) {
            motor_channel_t* ch = &resonance->channels[i];
            ch->suppression_level -= release_amount;
            ch->suppression_level = clamp_f(ch->suppression_level, 0.0F, 1.0F);

            if (ch->suppression_level < 0.2F && learning_context > 0.5F) {
                ch->release_reason = RESONANCE_RELEASE_LEARNING;
                resonance->learning_releases++;
            }
        }
    } else if ((uint32_t)channel_id < resonance->num_channels) {
        motor_channel_t* ch = &resonance->channels[channel_id];
        ch->suppression_level -= release_amount;
        ch->suppression_level = clamp_f(ch->suppression_level, 0.0F, 1.0F);

        if (ch->suppression_level < 0.2F && learning_context > 0.5F) {
            ch->release_reason = RESONANCE_RELEASE_LEARNING;
            resonance->learning_releases++;
        }
    }
}

void motor_resonance_release_for_social(motor_resonance_t resonance,
                                         int32_t channel_id, float social_strength) {
    if (!resonance || !resonance->config.enable_social_context) return;

    resonance->social_context = clamp_f(social_strength, 0.0F, 1.0F);

    float release_amount = social_strength * resonance->config.bg_release_rate * 0.5F;

    if (channel_id < 0) {
        for (uint32_t i = 0; i < resonance->num_channels; i++) {
            motor_channel_t* ch = &resonance->channels[i];
            ch->suppression_level -= release_amount;
            ch->suppression_level = clamp_f(ch->suppression_level, 0.0F, 1.0F);

            if (ch->suppression_level < 0.2F && social_strength > 0.5F) {
                ch->release_reason = RESONANCE_RELEASE_SOCIAL;
                resonance->social_releases++;
            }
        }
    } else if ((uint32_t)channel_id < resonance->num_channels) {
        motor_channel_t* ch = &resonance->channels[channel_id];
        ch->suppression_level -= release_amount;
        ch->suppression_level = clamp_f(ch->suppression_level, 0.0F, 1.0F);

        if (ch->suppression_level < 0.2F && social_strength > 0.5F) {
            ch->release_reason = RESONANCE_RELEASE_SOCIAL;
            resonance->social_releases++;
        }
    }
}

//=============================================================================
// Conflict Detection
//=============================================================================

float motor_resonance_get_conflict(motor_resonance_t resonance, uint32_t channel_id) {
    if (!resonance || channel_id >= resonance->num_channels) {
        return 0.0F;
    }
    return resonance->channels[channel_id].conflict_signal;
}

void motor_resonance_set_conflict(motor_resonance_t resonance,
                                   uint32_t channel_a, uint32_t channel_b) {
    if (!resonance || channel_a >= resonance->num_channels ||
        channel_b >= resonance->num_channels ||
        channel_a == channel_b) {
        return;
    }

    // Check if conflict already exists
    if (channels_conflict(resonance, channel_a, channel_b)) {
        return;
    }

    // Add new conflict
    if (resonance->num_conflicts < resonance->max_conflicts) {
        resonance->conflicts[resonance->num_conflicts].channel_a = channel_a;
        resonance->conflicts[resonance->num_conflicts].channel_b = channel_b;
        resonance->num_conflicts++;
    }
}

//=============================================================================
// Output Query
//=============================================================================

float motor_resonance_get_output(motor_resonance_t resonance, uint32_t channel_id) {
    if (!resonance || channel_id >= resonance->num_channels) {
        return -1.0F;
    }
    return resonance->channels[channel_id].motor_output;
}

bool motor_resonance_above_threshold(motor_resonance_t resonance, uint32_t channel_id) {
    if (!resonance || channel_id >= resonance->num_channels) {
        return false;
    }
    return resonance->channels[channel_id].above_threshold;
}

uint32_t motor_resonance_get_active_channels(motor_resonance_t resonance,
                                              uint32_t* out_channels,
                                              uint32_t max_channels) {
    if (!resonance || !out_channels || max_channels == 0) {
        return 0;
    }

    uint32_t count = 0;
    for (uint32_t i = 0; i < resonance->num_channels && count < max_channels; i++) {
        if (resonance->channels[i].above_threshold) {
            out_channels[count++] = i;
        }
    }
    return count;
}

//=============================================================================
// Time Update
//=============================================================================

void motor_resonance_step(motor_resonance_t resonance, float dt_ms) {
    if (!resonance) return;

    const motor_resonance_config_t* cfg = &resonance->config;

    // Compute decay factors
    float resonance_decay = exp_decay(dt_ms, cfg->tau_resonance_decay);
    float fatigue_recovery = exp_decay(dt_ms, cfg->tau_fatigue_recovery);
    float conflict_decay = 1.0F - cfg->conflict_decay;

    // Update all channels
    for (uint32_t i = 0; i < resonance->num_channels; i++) {
        motor_channel_t* ch = &resonance->channels[i];

        // Decay resonance toward target (or zero if no input)
        ch->resonance_level *= resonance_decay;
        ch->resonance_level = fmaxf(ch->resonance_level, 0.0F);

        // Also decay peak
        ch->peak_resonance *= 0.999F;

        // Recover from fatigue
        if (cfg->enable_fatigue) {
            ch->fatigue_level *= fatigue_recovery;
        }

        // Decay conflict signal
        ch->conflict_signal *= conflict_decay;

        // Compute conflict with other channels
        compute_conflict_signal(resonance, i);

        // Apply conflict suppression if needed
        if (ch->conflict_signal > cfg->conflict_threshold) {
            ch->suppression_level = fmaxf(ch->suppression_level, ch->conflict_signal);
            ch->suppress_reason = RESONANCE_SUPPRESS_CONFLICT;
            resonance->conflict_suppressions++;
        }

        // Restore BG tonic inhibition if no other suppression
        if (ch->suppression_level < resonance->bg_inhibition &&
            ch->release_reason == RESONANCE_RELEASE_NONE) {
            ch->suppression_level += 0.01F;
            ch->suppression_level = fminf(ch->suppression_level, resonance->bg_inhibition);
        }

        // Compute final motor output
        ch->motor_output = ch->resonance_level - ch->suppression_level;
        ch->motor_output = clamp_f(ch->motor_output, 0.0F, 1.0F);

        // Check threshold
        bool was_above = ch->above_threshold;
        ch->above_threshold = ch->motor_output >= cfg->execution_threshold;

        // Track activation
        if (ch->above_threshold && !was_above) {
            ch->activation_count++;
            resonance->total_activations++;
            ch->last_activation_us = resonance->current_time_us;
        }

        // Track total resonance time
        if (ch->resonance_level > RESONANCE_ACTIVE_THRESHOLD) {
            ch->total_resonance_time += dt_ms;
        }

        // Track suppression
        if (ch->suppression_level > 0.5F && ch->resonance_level > 0.3F) {
            ch->suppression_count++;
            if (ch->suppress_reason == RESONANCE_SUPPRESS_BG_TONIC) {
                resonance->bg_suppressions++;
            }
        }

        // Clear release reason if suppression restored
        if (ch->suppression_level > 0.3F) {
            ch->release_reason = RESONANCE_RELEASE_NONE;
        }
    }

    // Decay context signals
    resonance->learning_context *= 0.99F;
    resonance->social_context *= 0.99F;
}

//=============================================================================
// Statistics
//=============================================================================

bool motor_resonance_get_stats(motor_resonance_t resonance, motor_resonance_stats_t* stats) {
    if (!resonance || !stats) return false;

    memset(stats, 0, sizeof(motor_resonance_stats_t));

    stats->num_channels = resonance->num_channels;

    float sum_resonance = 0.0F;
    float sum_suppression = 0.0F;
    float sum_conflict = 0.0F;
    uint32_t conflict_count = 0;

    for (uint32_t i = 0; i < resonance->num_channels; i++) {
        const motor_channel_t* ch = &resonance->channels[i];

        sum_resonance += ch->resonance_level;
        sum_suppression += ch->suppression_level;

        if (ch->resonance_level > RESONANCE_ACTIVE_THRESHOLD) {
            stats->active_channels++;
        }

        if (ch->suppression_level > 0.5F) {
            stats->suppressed_channels++;
        }

        if (ch->above_threshold) {
            stats->above_threshold++;
        }

        if (ch->resonance_level > stats->max_resonance) {
            stats->max_resonance = ch->resonance_level;
        }

        if (ch->conflict_signal > 0.1F) {
            sum_conflict += ch->conflict_signal;
            conflict_count++;
        }
    }

    if (resonance->num_channels > 0) {
        stats->mean_resonance = sum_resonance / resonance->num_channels;
        stats->mean_suppression = sum_suppression / resonance->num_channels;
    }

    stats->total_activations = resonance->total_activations;
    stats->bg_suppressions = resonance->bg_suppressions;
    stats->pfc_suppressions = resonance->pfc_suppressions;
    stats->conflict_suppressions = resonance->conflict_suppressions;
    stats->learning_releases = resonance->learning_releases;
    stats->social_releases = resonance->social_releases;
    stats->voluntary_releases = resonance->voluntary_releases;

    if (resonance->num_channels > 0) {
        stats->conflict_rate = (float)conflict_count / resonance->num_channels;
    }
    if (conflict_count > 0) {
        stats->avg_conflict_signal = sum_conflict / conflict_count;
    }

    return true;
}

void motor_resonance_reset_stats(motor_resonance_t resonance) {
    if (!resonance) return;

    resonance->total_activations = 0;
    resonance->bg_suppressions = 0;
    resonance->pfc_suppressions = 0;
    resonance->conflict_suppressions = 0;
    resonance->learning_releases = 0;
    resonance->social_releases = 0;
    resonance->voluntary_releases = 0;

    for (uint32_t i = 0; i < resonance->num_channels; i++) {
        motor_channel_t* ch = &resonance->channels[i];
        ch->activation_count = 0;
        ch->suppression_count = 0;
        ch->total_resonance_time = 0.0F;
        ch->peak_resonance = 0.0F;
    }
}

//=============================================================================
// KG Self-Awareness Integration
//=============================================================================

int motor_resonance_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Motor_Resonance");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            LOG_DEBUG("Motor resonance self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Motor_Resonance");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Motor_Resonance");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
