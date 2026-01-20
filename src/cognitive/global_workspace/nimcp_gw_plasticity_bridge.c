/**
 * @file nimcp_gw_plasticity_bridge.c
 * @brief Global Workspace - Plasticity Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/global_workspace/nimcp_gw_plasticity_bridge.h"
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

struct gw_plasticity_bridge {
    gw_plasticity_config_t config;
    nimcp_mutex_t* mutex;

    /* State */
    gw_plasticity_state_t state;
    uint64_t current_time_us;
    bool bio_async_connected;

    /* Synapses */
    gw_plasticity_synapse_t* synapses;
    uint32_t num_synapses;
    uint32_t max_synapses;

    /* Access learning state */
    gw_access_learning_state_t access_learning;

    /* Reward eligibility */
    float global_eligibility;
    float last_reward;

    /* Callbacks */
    gw_plasticity_learn_callback_t learn_callback;
    void* learn_callback_data;
    gw_plasticity_access_callback_t access_callback;
    void* access_callback_data;

    /* Statistics */
    gw_plasticity_stats_t stats;
};

//=============================================================================
// Helper Functions
//=============================================================================

static inline float clamp_f(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

static gw_plasticity_synapse_t* find_synapse(
    gw_plasticity_bridge_t* bridge,
    uint32_t synapse_id
) {
    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        if (bridge->synapses[i].synapse_id == synapse_id) {
            return &bridge->synapses[i];
        }
    }
    return NULL;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

gw_plasticity_config_t gw_plasticity_config_default(void) {
    gw_plasticity_config_t config = {
        .base_learning_rate = GW_PLASTICITY_DEFAULT_LR,
        .stdp_tau_plus_ms = 20.0f,
        .stdp_tau_minus_ms = 25.0f,
        .stdp_a_plus = 0.01f,
        .stdp_a_minus = 0.012f,

        .bcm_tau_ms = 1000.0f,
        .bcm_target_rate = 10.0f,

        .homeostatic_tau_ms = 10000.0f,
        .target_ignition_rate = 0.6f,

        .broadcast_success_boost = 2.0f,
        .competition_win_boost = 1.5f,
        .ignition_modulation = 1.0f,

        .weight_min = 0.0f,
        .weight_max = 2.0f,

        .protect_broadcast_synapses = true,
        .protect_integration_synapses = true,
        .protection_strength = 0.9f,

        .max_synapses = GW_PLASTICITY_MAX_SYNAPSES,

        .enable_bio_async = false
    };
    return config;
}

gw_plasticity_bridge_t* gw_plasticity_create(
    const gw_plasticity_config_t* config
) {
    gw_plasticity_bridge_t* bridge = nimcp_calloc(1, sizeof(gw_plasticity_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = gw_plasticity_config_default();
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
    bridge->synapses = nimcp_calloc(bridge->max_synapses, sizeof(gw_plasticity_synapse_t));
    if (!bridge->synapses) {
        nimcp_mutex_free(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->num_synapses = 0;
    bridge->state = GW_PLASTICITY_STATE_IDLE;
    bridge->current_time_us = 0;
    bridge->bio_async_connected = false;
    bridge->global_eligibility = 0.0f;
    bridge->last_reward = 0.0f;

    /* Initialize access learning state */
    bridge->access_learning.broadcast_sensitivity = 0.5f;
    bridge->access_learning.ignition_calibration = 0.6f;
    bridge->access_learning.competition_strength = 0.5f;
    bridge->access_learning.binding_strength = 0.5f;
    bridge->access_learning.coalition_strength = 0.5f;
    bridge->access_learning.learning_rate_mod = 1.0f;
    bridge->access_learning.last_learning_us = 0;

    return bridge;
}

void gw_plasticity_destroy(gw_plasticity_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }

    nimcp_free(bridge->synapses);
    nimcp_free(bridge);
}

int gw_plasticity_reset(gw_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Reset all synapses to initial weights */
    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        bridge->synapses[i].weight = bridge->synapses[i].initial_weight;
        bridge->synapses[i].eligibility_trace = 0.0f;
        bridge->synapses[i].bcm_threshold = bridge->config.bcm_target_rate;
        bridge->synapses[i].avg_activity = 0.0f;
        bridge->synapses[i].update_count = 0;
    }

    /* Reset access learning state */
    bridge->access_learning.broadcast_sensitivity = 0.5f;
    bridge->access_learning.ignition_calibration = 0.6f;
    bridge->access_learning.competition_strength = 0.5f;
    bridge->access_learning.binding_strength = 0.5f;
    bridge->access_learning.coalition_strength = 0.5f;
    bridge->access_learning.learning_rate_mod = 1.0f;

    bridge->global_eligibility = 0.0f;
    bridge->last_reward = 0.0f;
    bridge->state = GW_PLASTICITY_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Synapse Management
//=============================================================================

int gw_plasticity_register_synapse(
    gw_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    gw_synapse_type_t type,
    float initial_weight
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Check if already exists */
    if (find_synapse(bridge, synapse_id)) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Check capacity */
    if (bridge->num_synapses >= bridge->max_synapses) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Add synapse */
    gw_plasticity_synapse_t* syn = &bridge->synapses[bridge->num_synapses];
    syn->synapse_id = synapse_id;
    syn->type = type;
    syn->weight = clamp_f(initial_weight, bridge->config.weight_min, bridge->config.weight_max);
    syn->initial_weight = syn->weight;
    syn->eligibility_trace = 0.0f;
    syn->bcm_threshold = bridge->config.bcm_target_rate;
    syn->avg_activity = 0.0f;
    syn->last_update_us = bridge->current_time_us;
    syn->update_count = 0;

    /* Auto-protect broadcast and integration synapses */
    syn->is_protected = false;
    if (bridge->config.protect_broadcast_synapses &&
        type == GW_SYNAPSE_BROADCAST) {
        syn->is_protected = true;
    }
    if (bridge->config.protect_integration_synapses &&
        type == GW_SYNAPSE_INTEGRATION) {
        syn->is_protected = true;
    }

    bridge->num_synapses++;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int gw_plasticity_unregister_synapse(
    gw_plasticity_bridge_t* bridge,
    uint32_t synapse_id
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        if (bridge->synapses[i].synapse_id == synapse_id) {
            /* Move last to current position */
            if (i < bridge->num_synapses - 1) {
                bridge->synapses[i] = bridge->synapses[bridge->num_synapses - 1];
            }
            bridge->num_synapses--;
            nimcp_mutex_unlock(bridge->mutex);
            return 0;
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return -1;
}

int gw_plasticity_get_synapse(
    gw_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    gw_plasticity_synapse_t* synapse
) {
    if (!bridge || !synapse) return -1;

    nimcp_mutex_lock(bridge->mutex);

    gw_plasticity_synapse_t* syn = find_synapse(bridge, synapse_id);
    if (!syn) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    *synapse = *syn;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int gw_plasticity_protect_synapse(
    gw_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    bool protect
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    gw_plasticity_synapse_t* syn = find_synapse(bridge, synapse_id);
    if (!syn) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    syn->is_protected = protect;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Learning Functions
//=============================================================================

int gw_plasticity_learn(
    gw_plasticity_bridge_t* bridge,
    gw_learn_event_t event,
    float magnitude,
    uint32_t synapse_id,
    float context
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = GW_PLASTICITY_STATE_LEARNING;

    gw_plasticity_synapse_t* syn = find_synapse(bridge, synapse_id);
    if (!syn) {
        bridge->state = GW_PLASTICITY_STATE_IDLE;
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Check protection */
    if (syn->is_protected) {
        bridge->stats.protected_updates_blocked++;
        bridge->state = GW_PLASTICITY_STATE_IDLE;
        nimcp_mutex_unlock(bridge->mutex);
        return 0;
    }

    /* Calculate learning rate with modulation */
    float lr = bridge->config.base_learning_rate * bridge->access_learning.learning_rate_mod;

    /* Apply event-specific learning */
    float delta_weight = 0.0f;
    switch (event) {
        case GW_LEARN_BROADCAST_SUCCESS:
            delta_weight = lr * magnitude * bridge->config.broadcast_success_boost;
            bridge->stats.broadcast_success_events++;
            break;

        case GW_LEARN_BROADCAST_FAILURE:
            delta_weight = -lr * magnitude * 0.5f;
            bridge->stats.broadcast_failure_events++;
            break;

        case GW_LEARN_IGNITION_TRIGGERED:
            delta_weight = lr * magnitude * bridge->config.ignition_modulation;
            bridge->stats.ignition_events++;
            break;

        case GW_LEARN_IGNITION_SUBTHRESHOLD:
            delta_weight = -lr * magnitude * 0.3f;
            break;

        case GW_LEARN_COMPETITION_WON:
            delta_weight = lr * magnitude * bridge->config.competition_win_boost;
            bridge->stats.competition_events++;
            break;

        case GW_LEARN_COMPETITION_LOST:
            delta_weight = -lr * magnitude * 0.5f;
            break;

        case GW_LEARN_BINDING_FORMED:
            delta_weight = lr * magnitude * context;
            break;

        case GW_LEARN_COALITION_STRENGTHENED:
            delta_weight = lr * magnitude * 0.8f;
            break;

        default:
            delta_weight = lr * magnitude;
            break;
    }

    /* Apply eligibility trace modulation */
    delta_weight *= (1.0f + syn->eligibility_trace);

    /* Update weight with bounds */
    float old_weight = syn->weight;
    syn->weight = clamp_f(syn->weight + delta_weight, bridge->config.weight_min, bridge->config.weight_max);
    float actual_delta = syn->weight - old_weight;

    /* Update statistics */
    if (actual_delta > 0) {
        bridge->stats.total_potentiation += actual_delta;
    } else {
        bridge->stats.total_depression += fabsf(actual_delta);
    }
    bridge->stats.weight_updates++;
    bridge->stats.total_learning_events++;

    /* Update running mean */
    float n = (float)bridge->stats.weight_updates;
    bridge->stats.mean_weight_change = bridge->stats.mean_weight_change * ((n - 1) / n) +
                                       fabsf(actual_delta) / n;

    syn->last_update_us = bridge->current_time_us;
    syn->update_count++;
    bridge->access_learning.last_learning_us = bridge->current_time_us;

    /* Invoke callback */
    if (bridge->learn_callback) {
        bridge->learn_callback(bridge, event, magnitude, bridge->learn_callback_data);
    }

    bridge->state = GW_PLASTICITY_STATE_IDLE;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float gw_plasticity_apply_stdp(
    gw_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    float pre_time,
    float post_time
) {
    if (!bridge) return NAN;

    nimcp_mutex_lock(bridge->mutex);

    gw_plasticity_synapse_t* syn = find_synapse(bridge, synapse_id);
    if (!syn) {
        nimcp_mutex_unlock(bridge->mutex);
        return NAN;
    }

    /* Check protection */
    if (syn->is_protected) {
        bridge->stats.protected_updates_blocked++;
        nimcp_mutex_unlock(bridge->mutex);
        return 0.0f;
    }

    float dt = post_time - pre_time;
    float delta_weight = 0.0f;

    if (dt > 0) {
        /* LTP: post after pre */
        delta_weight = bridge->config.stdp_a_plus *
                      expf(-dt / bridge->config.stdp_tau_plus_ms);
    } else if (dt < 0) {
        /* LTD: pre after post */
        delta_weight = -bridge->config.stdp_a_minus *
                      expf(dt / bridge->config.stdp_tau_minus_ms);
    }

    /* Apply with learning rate */
    delta_weight *= bridge->config.base_learning_rate * bridge->access_learning.learning_rate_mod;

    /* Update weight */
    float old_weight = syn->weight;
    syn->weight = clamp_f(syn->weight + delta_weight, bridge->config.weight_min, bridge->config.weight_max);
    float actual_delta = syn->weight - old_weight;

    /* Update statistics */
    if (actual_delta > 0) {
        bridge->stats.total_potentiation += actual_delta;
    } else {
        bridge->stats.total_depression += fabsf(actual_delta);
    }
    bridge->stats.weight_updates++;

    syn->last_update_us = bridge->current_time_us;
    syn->update_count++;

    nimcp_mutex_unlock(bridge->mutex);
    return actual_delta;
}

int gw_plasticity_apply_reward(
    gw_plasticity_bridge_t* bridge,
    float reward
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    reward = clamp_f(reward, -1.0f, 1.0f);
    bridge->last_reward = reward;

    /* Apply reward-modulated learning to all synapses with eligibility */
    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        gw_plasticity_synapse_t* syn = &bridge->synapses[i];

        if (syn->is_protected || syn->eligibility_trace < 0.001f) {
            continue;
        }

        float delta = bridge->config.base_learning_rate * reward *
                     syn->eligibility_trace * bridge->config.ignition_modulation;

        syn->weight = clamp_f(syn->weight + delta, bridge->config.weight_min, bridge->config.weight_max);

        if (delta > 0) {
            bridge->stats.total_potentiation += delta;
        } else {
            bridge->stats.total_depression += fabsf(delta);
        }
        bridge->stats.weight_updates++;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int gw_plasticity_update_bcm(
    gw_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;
    if (dt_ms <= 0.0f) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float decay = expf(-dt_ms / bridge->config.bcm_tau_ms);

    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        gw_plasticity_synapse_t* syn = &bridge->synapses[i];

        /* Update sliding threshold towards average activity */
        syn->bcm_threshold = syn->bcm_threshold * decay +
                            syn->avg_activity * (1.0f - decay);

        /* BCM learning rule: weight change depends on activity relative to threshold */
        if (!syn->is_protected && syn->avg_activity > 0.001f) {
            float bcm_factor = (syn->avg_activity - syn->bcm_threshold) * syn->avg_activity;
            float delta = bridge->config.base_learning_rate * bcm_factor * 0.001f;

            syn->weight = clamp_f(syn->weight + delta, bridge->config.weight_min, bridge->config.weight_max);
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int gw_plasticity_homeostatic_update(
    gw_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;
    if (dt_ms <= 0.0f) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Calculate mean weight */
    float mean_weight = 0.0f;
    uint32_t active_count = 0;

    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        if (!bridge->synapses[i].is_protected) {
            mean_weight += bridge->synapses[i].weight;
            active_count++;
        }
    }

    if (active_count > 0) {
        mean_weight /= active_count;
    }

    /* Target weight is mid-range */
    float target = (bridge->config.weight_max + bridge->config.weight_min) / 2.0f;
    float error = target - mean_weight;

    /* Homeostatic scaling */
    float scale_factor = 1.0f + error * dt_ms / bridge->config.homeostatic_tau_ms;
    scale_factor = clamp_f(scale_factor, 0.99f, 1.01f);

    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        if (!bridge->synapses[i].is_protected) {
            bridge->synapses[i].weight = clamp_f(
                bridge->synapses[i].weight * scale_factor,
                bridge->config.weight_min,
                bridge->config.weight_max
            );
        }
    }

    /* Update ignition calibration towards target */
    float calib_decay = expf(-dt_ms / bridge->config.homeostatic_tau_ms);
    float old_calib = bridge->access_learning.ignition_calibration;
    bridge->access_learning.ignition_calibration =
        old_calib * calib_decay + bridge->config.target_ignition_rate * (1.0f - calib_decay);

    /* Invoke access callback if significant change */
    if (bridge->access_callback &&
        fabsf(bridge->access_learning.ignition_calibration - old_calib) > 0.01f) {
        bridge->access_callback(bridge, old_calib,
                                bridge->access_learning.ignition_calibration,
                                bridge->access_callback_data);
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int gw_plasticity_update_traces(
    gw_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;
    if (dt_ms <= 0.0f) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float decay = expf(-dt_ms / bridge->config.stdp_tau_plus_ms);

    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        bridge->synapses[i].eligibility_trace *= decay;
    }

    bridge->global_eligibility *= decay;
    bridge->current_time_us += (uint64_t)(dt_ms * 1000.0f);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int gw_plasticity_consolidate(gw_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = GW_PLASTICITY_STATE_CONSOLIDATING;

    /* Consolidate learning by resetting eligibility traces */
    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        bridge->synapses[i].eligibility_trace = 0.0f;
    }

    bridge->global_eligibility = 0.0f;
    bridge->state = GW_PLASTICITY_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// State Query Functions
//=============================================================================

int gw_plasticity_get_access_learning_state(
    gw_plasticity_bridge_t* bridge,
    gw_access_learning_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->access_learning;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int gw_plasticity_get_state(
    gw_plasticity_bridge_t* bridge,
    gw_plasticity_bridge_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->mutex);

    state->state = bridge->state;
    state->active_synapses = bridge->num_synapses;

    /* Calculate mean weight and variance */
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        sum += bridge->synapses[i].weight;
        sum_sq += bridge->synapses[i].weight * bridge->synapses[i].weight;
    }

    if (bridge->num_synapses > 0) {
        state->mean_weight = sum / bridge->num_synapses;
        state->weight_variance = (sum_sq / bridge->num_synapses) -
                                (state->mean_weight * state->mean_weight);
    } else {
        state->mean_weight = 0.0f;
        state->weight_variance = 0.0f;
    }

    state->learning_rate_effective = bridge->config.base_learning_rate *
                                     bridge->access_learning.learning_rate_mod;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int gw_plasticity_get_stats(
    gw_plasticity_bridge_t* bridge,
    gw_plasticity_stats_t* stats
) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int gw_plasticity_reset_stats(gw_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(gw_plasticity_stats_t));
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Callback Registration
//=============================================================================

int gw_plasticity_register_learn_callback(
    gw_plasticity_bridge_t* bridge,
    gw_plasticity_learn_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->learn_callback = callback;
    bridge->learn_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int gw_plasticity_register_access_callback(
    gw_plasticity_bridge_t* bridge,
    gw_plasticity_access_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->access_callback = callback;
    bridge->access_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int gw_plasticity_bio_async_connect(gw_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int gw_plasticity_bio_async_disconnect(gw_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

bool gw_plasticity_is_bio_async_connected(gw_plasticity_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    bool connected = bridge->bio_async_connected;
    nimcp_mutex_unlock(bridge->mutex);

    return connected;
}
