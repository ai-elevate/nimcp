/**
 * @file nimcp_self_awareness_plasticity_bridge.c
 * @brief Self-Awareness - Plasticity Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/self_awareness/nimcp_self_awareness_plasticity_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"

#include <string.h>
#include <math.h>
#include <stdio.h>

//=============================================================================
// Internal Structures
//=============================================================================

typedef struct synapse_entry {
    self_awareness_plasticity_synapse_t synapse;
    bool in_use;
} synapse_entry_t;

struct self_awareness_plasticity_bridge {
    self_awareness_plasticity_config_t config;
    nimcp_mutex_t* mutex;

    /* State */
    self_awareness_plasticity_state_t state;
    uint64_t current_time_us;
    bool bio_async_connected;

    /* Synapse storage */
    synapse_entry_t* synapses;
    uint32_t synapse_count;
    uint32_t max_synapses;

    /* Learning state */
    self_awareness_learning_state_t learning_state;

    /* Global learning state */
    float current_reward;
    float learning_rate_effective;
    float bcm_global_threshold;

    /* Callbacks */
    self_awareness_plasticity_learn_callback_t learn_callback;
    void* learn_callback_data;
    self_awareness_plasticity_awareness_callback_t awareness_callback;
    void* awareness_callback_data;

    /* Statistics */
    self_awareness_plasticity_stats_t stats;
};

//=============================================================================
// Helper Functions
//=============================================================================

static inline float clamp_f(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

static synapse_entry_t* find_synapse(self_awareness_plasticity_bridge_t* bridge, uint32_t synapse_id) {
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use &&
            bridge->synapses[i].synapse.synapse_id == synapse_id) {
            return &bridge->synapses[i];
        }
    }
    return NULL;
}

static synapse_entry_t* find_free_slot(self_awareness_plasticity_bridge_t* bridge) {
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (!bridge->synapses[i].in_use) {
            return &bridge->synapses[i];
        }
    }
    return NULL;
}

static bool is_protected_type(self_awareness_synapse_type_t type) {
    return type == SELF_SYNAPSE_BODY_OWNERSHIP ||
           type == SELF_SYNAPSE_CONTINUITY;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

self_awareness_plasticity_config_t self_awareness_plasticity_config_default(void) {
    self_awareness_plasticity_config_t config = {
        .base_learning_rate = SELF_AWARENESS_PLASTICITY_DEFAULT_LR,
        .stdp_tau_plus_ms = 20.0f,
        .stdp_tau_minus_ms = 20.0f,
        .stdp_a_plus = 0.01f,
        .stdp_a_minus = 0.012f,

        .bcm_tau_ms = 1000.0f,
        .bcm_target_rate = 5.0f,

        .homeostatic_tau_ms = 10000.0f,
        .target_self_awareness = 0.5f,

        .recognition_learning_boost = 1.5f,
        .agency_learning_boost = 1.3f,
        .metacog_modulation = 1.2f,

        .weight_min = 0.0f,
        .weight_max = 1.0f,

        .protect_body_ownership = true,
        .protect_temporal_continuity = true,
        .protection_strength = 1.0f,

        .max_synapses = SELF_AWARENESS_PLASTICITY_MAX_SYNAPSES,

        .enable_bio_async = false
    };
    return config;
}

self_awareness_plasticity_bridge_t* self_awareness_plasticity_create(
    const self_awareness_plasticity_config_t* config
) {
    self_awareness_plasticity_bridge_t* bridge = nimcp_calloc(1, sizeof(self_awareness_plasticity_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = self_awareness_plasticity_config_default();
    }

    /* Create mutex */
    mutex_attr_t mutex_attr = {.type = MUTEX_TYPE_NORMAL};
    bridge->mutex = nimcp_mutex_create(&mutex_attr);
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate synapse storage */
    bridge->max_synapses = bridge->config.max_synapses;
    bridge->synapses = nimcp_calloc(bridge->max_synapses, sizeof(synapse_entry_t));
    if (!bridge->synapses) {
        nimcp_mutex_destroy(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize learning state */
    bridge->learning_state.recognition_sensitivity = 1.0f;
    bridge->learning_state.body_ownership_calibration = 0.5f;
    bridge->learning_state.agency_sensitivity = 1.0f;
    bridge->learning_state.metacog_strength = 0.5f;
    bridge->learning_state.reflection_depth = 0.5f;
    bridge->learning_state.learning_rate_mod = 1.0f;
    bridge->learning_state.last_learning_us = 0;

    bridge->state = SELF_PLASTICITY_STATE_IDLE;
    bridge->current_time_us = 0;
    bridge->bio_async_connected = false;
    bridge->synapse_count = 0;

    bridge->current_reward = 0.0f;
    bridge->learning_rate_effective = bridge->config.base_learning_rate;
    bridge->bcm_global_threshold = bridge->config.bcm_target_rate;

    return bridge;
}

void self_awareness_plasticity_destroy(self_awareness_plasticity_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge->synapses);
    nimcp_free(bridge);
}

int self_awareness_plasticity_reset(self_awareness_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Reset all synapses to initial weights */
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use) {
            bridge->synapses[i].synapse.weight = bridge->synapses[i].synapse.initial_weight;
            bridge->synapses[i].synapse.eligibility_trace = 0.0f;
            bridge->synapses[i].synapse.bcm_threshold = bridge->config.bcm_target_rate;
            bridge->synapses[i].synapse.avg_activity = 0.0f;
            bridge->synapses[i].synapse.update_count = 0;
        }
    }

    /* Reset learning state */
    bridge->learning_state.recognition_sensitivity = 1.0f;
    bridge->learning_state.body_ownership_calibration = 0.5f;
    bridge->learning_state.agency_sensitivity = 1.0f;
    bridge->learning_state.learning_rate_mod = 1.0f;

    bridge->state = SELF_PLASTICITY_STATE_IDLE;
    bridge->current_reward = 0.0f;
    bridge->learning_rate_effective = bridge->config.base_learning_rate;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Synapse Management
//=============================================================================

int self_awareness_plasticity_register_synapse(
    self_awareness_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    self_awareness_synapse_type_t type,
    float initial_weight
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Check for duplicate */
    if (find_synapse(bridge, synapse_id)) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Find free slot */
    synapse_entry_t* slot = find_free_slot(bridge);
    if (!slot) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Initialize synapse */
    slot->in_use = true;
    slot->synapse.synapse_id = synapse_id;
    slot->synapse.type = type;
    slot->synapse.weight = clamp_f(initial_weight, bridge->config.weight_min, bridge->config.weight_max);
    slot->synapse.initial_weight = slot->synapse.weight;
    slot->synapse.eligibility_trace = 0.0f;
    slot->synapse.bcm_threshold = bridge->config.bcm_target_rate;
    slot->synapse.avg_activity = 0.0f;
    slot->synapse.last_update_us = bridge->current_time_us;
    slot->synapse.update_count = 0;

    /* Auto-protect body ownership and continuity synapses */
    slot->synapse.is_protected = is_protected_type(type) &&
        (bridge->config.protect_body_ownership || bridge->config.protect_temporal_continuity);

    bridge->synapse_count++;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_plasticity_unregister_synapse(
    self_awareness_plasticity_bridge_t* bridge,
    uint32_t synapse_id
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    synapse_entry_t* entry = find_synapse(bridge, synapse_id);
    if (!entry) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    entry->in_use = false;
    bridge->synapse_count--;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_plasticity_get_synapse(
    self_awareness_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    self_awareness_plasticity_synapse_t* synapse
) {
    if (!bridge || !synapse) return -1;

    nimcp_mutex_lock(bridge->mutex);

    synapse_entry_t* entry = find_synapse(bridge, synapse_id);
    if (!entry) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    *synapse = entry->synapse;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_plasticity_protect_synapse(
    self_awareness_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    bool protect
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    synapse_entry_t* entry = find_synapse(bridge, synapse_id);
    if (!entry) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    entry->synapse.is_protected = protect;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Learning Functions
//=============================================================================

int self_awareness_plasticity_learn(
    self_awareness_plasticity_bridge_t* bridge,
    self_awareness_learn_event_t event,
    float magnitude,
    uint32_t synapse_id,
    float context
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = SELF_PLASTICITY_STATE_LEARNING;

    synapse_entry_t* entry = find_synapse(bridge, synapse_id);
    if (!entry) {
        bridge->state = SELF_PLASTICITY_STATE_IDLE;
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Check protection */
    if (entry->synapse.is_protected) {
        bridge->stats.protected_updates_blocked++;
        bridge->state = SELF_PLASTICITY_STATE_IDLE;
        nimcp_mutex_unlock(bridge->mutex);
        return 0; /* Not an error, just blocked */
    }

    float lr = bridge->learning_rate_effective * magnitude;
    float weight_change = 0.0f;

    switch (event) {
        case SELF_LEARN_RECOGNITION_CONFIRMED:
            weight_change = lr * bridge->config.recognition_learning_boost;
            bridge->stats.recognition_confirmed_events++;
            break;

        case SELF_LEARN_FALSE_RECOGNITION:
            weight_change = -lr * 0.5f;
            bridge->stats.false_recognition_events++;
            break;

        case SELF_LEARN_AGENCY_CONFIRMED:
            weight_change = lr * bridge->config.agency_learning_boost;
            bridge->stats.agency_confirmed_events++;
            break;

        case SELF_LEARN_AGENCY_MISMATCH:
            weight_change = -lr * 0.3f;
            break;

        case SELF_LEARN_BODY_OWNERSHIP_GAIN:
            weight_change = lr * 1.2f;
            break;

        case SELF_LEARN_BODY_OWNERSHIP_LOSS:
            weight_change = -lr * 0.2f;
            break;

        case SELF_LEARN_METACOG_ACCURATE:
            weight_change = lr * bridge->config.metacog_modulation;
            bridge->stats.metacog_accurate_events++;
            break;

        case SELF_LEARN_METACOG_INACCURATE:
            weight_change = -lr * 0.3f;
            break;

        case SELF_LEARN_REFLECTION_INSIGHT:
            weight_change = lr * 1.0f;
            break;

        case SELF_LEARN_CONTINUITY_MAINTAINED:
            weight_change = lr * 0.5f;
            break;
    }

    /* Context modulation */
    weight_change *= (0.5f + 0.5f * context);

    /* Apply weight change */
    float old_weight = entry->synapse.weight;
    entry->synapse.weight = clamp_f(
        entry->synapse.weight + weight_change,
        bridge->config.weight_min,
        bridge->config.weight_max
    );

    /* Update statistics */
    float actual_change = entry->synapse.weight - old_weight;
    if (actual_change > 0) {
        bridge->stats.total_potentiation += actual_change;
    } else {
        bridge->stats.total_depression += -actual_change;
    }

    entry->synapse.update_count++;
    entry->synapse.last_update_us = bridge->current_time_us;

    bridge->stats.total_learning_events++;
    bridge->stats.weight_updates++;
    bridge->stats.mean_weight_change =
        (bridge->stats.mean_weight_change * (bridge->stats.weight_updates - 1) +
         fabsf(actual_change)) / bridge->stats.weight_updates;

    /* Invoke callback */
    if (bridge->learn_callback) {
        bridge->learn_callback(bridge, event, magnitude, bridge->learn_callback_data);
    }

    bridge->state = SELF_PLASTICITY_STATE_IDLE;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float self_awareness_plasticity_apply_stdp(
    self_awareness_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    float pre_time,
    float post_time
) {
    if (!bridge) return NAN;

    nimcp_mutex_lock(bridge->mutex);

    synapse_entry_t* entry = find_synapse(bridge, synapse_id);
    if (!entry) {
        nimcp_mutex_unlock(bridge->mutex);
        return NAN;
    }

    /* Protected synapses don't get STDP */
    if (entry->synapse.is_protected) {
        nimcp_mutex_unlock(bridge->mutex);
        return 0.0f;
    }

    float dt = post_time - pre_time;
    float delta_w = 0.0f;

    if (dt > 0) {
        /* Potentiation: post after pre */
        delta_w = bridge->config.stdp_a_plus * expf(-dt / bridge->config.stdp_tau_plus_ms);
    } else {
        /* Depression: pre after post */
        delta_w = -bridge->config.stdp_a_minus * expf(dt / bridge->config.stdp_tau_minus_ms);
    }

    /* Apply weight change */
    float old_weight = entry->synapse.weight;
    entry->synapse.weight = clamp_f(
        entry->synapse.weight + delta_w,
        bridge->config.weight_min,
        bridge->config.weight_max
    );

    float actual_change = entry->synapse.weight - old_weight;
    if (actual_change > 0) {
        bridge->stats.total_potentiation += actual_change;
    } else {
        bridge->stats.total_depression += -actual_change;
    }

    entry->synapse.update_count++;
    bridge->stats.weight_updates++;

    nimcp_mutex_unlock(bridge->mutex);
    return delta_w;
}

int self_awareness_plasticity_apply_reward(
    self_awareness_plasticity_bridge_t* bridge,
    float reward
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    reward = clamp_f(reward, -1.0f, 1.0f);
    bridge->current_reward = reward;

    /* Apply reward modulation to all eligible synapses */
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use && !bridge->synapses[i].synapse.is_protected) {
            float trace = bridge->synapses[i].synapse.eligibility_trace;
            if (fabsf(trace) > 0.001f) {
                float delta = bridge->config.base_learning_rate * reward * trace;
                bridge->synapses[i].synapse.weight = clamp_f(
                    bridge->synapses[i].synapse.weight + delta,
                    bridge->config.weight_min,
                    bridge->config.weight_max
                );
                bridge->stats.weight_updates++;
            }
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_plasticity_update_bcm(
    self_awareness_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;
    if (dt_ms <= 0.0f) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = SELF_PLASTICITY_STATE_UPDATING;

    float decay = expf(-dt_ms / bridge->config.bcm_tau_ms);

    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use) {
            /* Update sliding threshold towards average activity */
            float target = bridge->synapses[i].synapse.avg_activity;
            bridge->synapses[i].synapse.bcm_threshold =
                bridge->synapses[i].synapse.bcm_threshold * decay +
                target * (1.0f - decay);
        }
    }

    bridge->state = SELF_PLASTICITY_STATE_IDLE;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_plasticity_homeostatic_update(
    self_awareness_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = SELF_PLASTICITY_STATE_UPDATING;

    float decay = expf(-dt_ms / bridge->config.homeostatic_tau_ms);

    /* Calculate mean self-awareness (from recognition synapses) */
    float mean_awareness = 0.0f;
    uint32_t awareness_count = 0;
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use &&
            bridge->synapses[i].synapse.type == SELF_SYNAPSE_RECOGNITION) {
            mean_awareness += bridge->synapses[i].synapse.weight;
            awareness_count++;
        }
    }
    if (awareness_count > 0) {
        mean_awareness /= awareness_count;
    }

    /* Scale non-protected synapses toward target */
    float target = bridge->config.target_self_awareness;
    float scale_factor = 1.0f;
    if (mean_awareness > 0.0f) {
        scale_factor = target / mean_awareness;
        scale_factor = clamp_f(scale_factor, 0.9f, 1.1f);
    }

    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use && !bridge->synapses[i].synapse.is_protected) {
            float scaled = bridge->synapses[i].synapse.weight * (1.0f + (scale_factor - 1.0f) * (1.0f - decay));
            bridge->synapses[i].synapse.weight = clamp_f(
                scaled,
                bridge->config.weight_min,
                bridge->config.weight_max
            );
        }
    }

    bridge->state = SELF_PLASTICITY_STATE_IDLE;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_plasticity_update_traces(
    self_awareness_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float decay = expf(-dt_ms / bridge->config.stdp_tau_plus_ms);

    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use) {
            bridge->synapses[i].synapse.eligibility_trace *= decay;
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_plasticity_consolidate(self_awareness_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = SELF_PLASTICITY_STATE_CONSOLIDATING;

    /* Clear eligibility traces */
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use) {
            bridge->synapses[i].synapse.eligibility_trace = 0.0f;
        }
    }

    /* Update learning state based on synapse weights */
    float recognition_sum = 0.0f, recognition_count = 0;
    float agency_sum = 0.0f, agency_count = 0;
    float metacog_sum = 0.0f, metacog_count = 0;
    float reflection_sum = 0.0f, reflection_count = 0;

    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use) {
            switch (bridge->synapses[i].synapse.type) {
                case SELF_SYNAPSE_RECOGNITION:
                    recognition_sum += bridge->synapses[i].synapse.weight;
                    recognition_count++;
                    break;
                case SELF_SYNAPSE_AGENCY:
                    agency_sum += bridge->synapses[i].synapse.weight;
                    agency_count++;
                    break;
                case SELF_SYNAPSE_METACOGNITIVE:
                    metacog_sum += bridge->synapses[i].synapse.weight;
                    metacog_count++;
                    break;
                case SELF_SYNAPSE_REFLECTION:
                    reflection_sum += bridge->synapses[i].synapse.weight;
                    reflection_count++;
                    break;
                default:
                    break;
            }
        }
    }

    if (recognition_count > 0) {
        bridge->learning_state.recognition_sensitivity = recognition_sum / recognition_count * 2.0f;
    }
    if (agency_count > 0) {
        bridge->learning_state.agency_sensitivity = agency_sum / agency_count * 2.0f;
    }
    if (metacog_count > 0) {
        bridge->learning_state.metacog_strength = metacog_sum / metacog_count;
    }
    if (reflection_count > 0) {
        bridge->learning_state.reflection_depth = reflection_sum / reflection_count;
    }

    bridge->learning_state.last_learning_us = bridge->current_time_us;
    bridge->state = SELF_PLASTICITY_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// State Query Functions
//=============================================================================

int self_awareness_plasticity_get_learning_state(
    self_awareness_plasticity_bridge_t* bridge,
    self_awareness_learning_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->learning_state;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int self_awareness_plasticity_get_state(
    self_awareness_plasticity_bridge_t* bridge,
    self_awareness_plasticity_bridge_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->mutex);

    state->state = bridge->state;
    state->active_synapses = bridge->synapse_count;
    state->learning_rate_effective = bridge->learning_rate_effective;

    /* Calculate mean weight and variance */
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use) {
            float w = bridge->synapses[i].synapse.weight;
            sum += w;
            sum_sq += w * w;
        }
    }

    if (bridge->synapse_count > 0) {
        state->mean_weight = sum / bridge->synapse_count;
        state->weight_variance = (sum_sq / bridge->synapse_count) -
                                (state->mean_weight * state->mean_weight);
    } else {
        state->mean_weight = 0.0f;
        state->weight_variance = 0.0f;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_plasticity_get_stats(
    self_awareness_plasticity_bridge_t* bridge,
    self_awareness_plasticity_stats_t* stats
) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int self_awareness_plasticity_reset_stats(self_awareness_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(self_awareness_plasticity_stats_t));
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Callback Registration
//=============================================================================

int self_awareness_plasticity_register_learn_callback(
    self_awareness_plasticity_bridge_t* bridge,
    self_awareness_plasticity_learn_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->learn_callback = callback;
    bridge->learn_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int self_awareness_plasticity_register_awareness_callback(
    self_awareness_plasticity_bridge_t* bridge,
    self_awareness_plasticity_awareness_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->awareness_callback = callback;
    bridge->awareness_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int self_awareness_plasticity_bio_async_connect(self_awareness_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return -1;

    nimcp_mutex_lock(bridge->mutex);
    /* Bio-async connection would be implemented here */
    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int self_awareness_plasticity_bio_async_disconnect(self_awareness_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

bool self_awareness_plasticity_is_bio_async_connected(self_awareness_plasticity_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    bool connected = bridge->bio_async_connected;
    nimcp_mutex_unlock(bridge->mutex);

    return connected;
}
