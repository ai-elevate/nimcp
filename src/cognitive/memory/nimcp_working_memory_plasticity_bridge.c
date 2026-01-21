/**
 * @file nimcp_working_memory_plasticity_bridge.c
 * @brief Working Memory - Plasticity Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/memory/nimcp_working_memory_plasticity_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>

//=============================================================================
// Internal Structure
//=============================================================================

struct wm_plasticity_bridge {
    /* Configuration */
    wm_plasticity_config_t config;

    /* Synapses */
    wm_plasticity_synapse_t* synapses;
    uint32_t synapse_count;
    uint32_t synapse_capacity;

    /* Per-slot state */
    wm_slot_plasticity_t* slot_states;
    uint32_t num_slots;

    /* State */
    wm_plasticity_state_t state;

    /* Current modulation values */
    float current_capacity_pressure;
    float current_salience_mod;
    float global_learning_rate;

    /* Pending reward */
    float pending_reward;
    uint64_t last_reward_time_us;

    /* Statistics */
    wm_plasticity_stats_t stats;

    /* Callbacks */
    wm_weight_change_cb weight_callback;
    void* weight_callback_data;
    wm_consolidation_cb consolidation_callback;
    void* consolidation_callback_data;

    /* Bio-async */
    bool bio_async_connected;

    /* Thread safety */
    nimcp_mutex_t* mutex;
};

//=============================================================================
// Helper Functions
//=============================================================================

static inline float clamp_f(float x, float min_val, float max_val) {
    return (x < min_val) ? min_val : (x > max_val) ? max_val : x;
}

/**
 * @brief Find synapse by ID
 */
static wm_plasticity_synapse_t* find_synapse(
    wm_plasticity_bridge_t* bridge,
    uint32_t synapse_id)
{
    for (uint32_t i = 0; i < bridge->synapse_count; i++) {
        if (bridge->synapses[i].synapse_id == synapse_id) {
            return &bridge->synapses[i];
        }
    }
    return NULL;
}

/**
 * @brief Compute STDP weight change
 */
static float compute_stdp_weight_change(
    const wm_plasticity_config_t* config,
    float dt_ms,
    float modulation)
{
    float dw = 0.0f;

    if (dt_ms > 0) {
        /* Pre before post -> LTP */
        dw = config->stdp_a_plus * expf(-dt_ms / config->stdp_tau_plus);
        dw *= modulation;
    } else if (dt_ms < 0) {
        /* Post before pre -> LTD */
        dw = -config->stdp_a_minus * expf(dt_ms / config->stdp_tau_minus);
        dw *= modulation;
    }

    return dw;
}

/**
 * @brief Apply eligibility trace modulation
 */
static float apply_eligibility(
    wm_plasticity_synapse_t* synapse,
    float reward,
    float gain)
{
    float dw = synapse->eligibility_trace * reward * gain;
    return dw;
}

/**
 * @brief Update BCM threshold
 */
static void update_bcm_threshold(
    wm_plasticity_synapse_t* synapse,
    float activity,
    float tau)
{
    float target = activity * activity;
    synapse->bcm_threshold += (target - synapse->bcm_threshold) / tau;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

wm_plasticity_config_t wm_plasticity_config_default(void) {
    wm_plasticity_config_t config = {
        /* STDP parameters */
        .stdp_ltp_window_ms = WM_PLASTICITY_STDP_WINDOW,
        .stdp_ltd_window_ms = WM_PLASTICITY_STDP_WINDOW,
        .stdp_a_plus = 0.01f,
        .stdp_a_minus = 0.012f,
        .stdp_tau_plus = 20.0f,
        .stdp_tau_minus = 20.0f,

        /* Maintenance modulation */
        .enable_maintenance_ltp = true,
        .maintenance_ltp_rate = 0.005f,
        .maintenance_interval_ms = 50.0f,

        /* Rehearsal learning */
        .enable_rehearsal_boost = true,
        .rehearsal_ltp_gain = 1.5f,
        .rehearsal_window_ms = 200.0f,

        /* Consolidation */
        .enable_consolidation = true,
        .consolidation_threshold = 5000.0f,  /* 5 seconds */
        .consolidation_ltp_boost = 2.0f,

        /* Capacity-based plasticity */
        .enable_capacity_ltd = true,
        .capacity_ltd_rate = 0.02f,
        .lateral_inhibition_gain = 0.5f,

        /* BCM */
        .enable_bcm = true,
        .bcm_threshold_tau = 1000.0f,
        .bcm_activity_tau = 100.0f,

        /* Homeostatic */
        .enable_homeostatic = true,
        .target_capacity_utilization = 0.7f,
        .homeostatic_tau_ms = 10000.0f,

        /* Eligibility */
        .enable_eligibility = true,
        .eligibility_decay = 0.95f,
        .reward_modulation_gain = 1.0f,

        /* Weights */
        .weight_min = 0.0f,
        .weight_max = 1.0f,
        .initial_weight = 0.3f,

        /* Salience */
        .enable_salience_modulation = true,
        .salience_ltp_gain = 1.5f,

        /* Integration */
        .enable_bio_async = false,
        .enable_immune_modulation = false,
        .enable_sleep_consolidation = true
    };
    return config;
}

wm_plasticity_bridge_t* wm_plasticity_create(const wm_plasticity_config_t* config) {
    wm_plasticity_bridge_t* bridge = nimcp_calloc(1, sizeof(wm_plasticity_bridge_t));
    if (!bridge) return NULL;

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = wm_plasticity_config_default();
    }

    /* Create mutex */
    mutex_attr_t mutex_attr = {.type = MUTEX_TYPE_NORMAL};
    bridge->mutex = nimcp_mutex_create(&mutex_attr);
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate synapse array */
    bridge->synapse_capacity = WM_PLASTICITY_MAX_SYNAPSES;
    bridge->synapses = nimcp_calloc(bridge->synapse_capacity, sizeof(wm_plasticity_synapse_t));
    if (!bridge->synapses) {
        nimcp_mutex_free(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }
    bridge->synapse_count = 0;

    /* Allocate per-slot state */
    bridge->num_slots = WM_PLASTICITY_MAX_SLOTS;
    bridge->slot_states = nimcp_calloc(bridge->num_slots, sizeof(wm_slot_plasticity_t));
    if (!bridge->slot_states) {
        nimcp_free(bridge->synapses);
        nimcp_mutex_free(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize slot states */
    for (uint32_t s = 0; s < bridge->num_slots; s++) {
        bridge->slot_states[s].occupied = false;
        bridge->slot_states[s].encoding_strength = 0.0f;
        bridge->slot_states[s].current_strength = 0.0f;
        bridge->slot_states[s].consolidation_progress = 0.0f;
        bridge->slot_states[s].rehearsal_count = 0;
        bridge->slot_states[s].last_access_time_us = 0;
        bridge->slot_states[s].salience = 0.0f;
    }

    /* Initialize state */
    bridge->state = WM_PLASTICITY_STATE_IDLE;
    bridge->current_capacity_pressure = 0.0f;
    bridge->current_salience_mod = 1.0f;
    bridge->global_learning_rate = 1.0f;
    bridge->pending_reward = 0.0f;
    bridge->last_reward_time_us = 0;
    bridge->bio_async_connected = false;

    return bridge;
}

void wm_plasticity_destroy(wm_plasticity_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->bio_async_connected) {
        wm_plasticity_disconnect_bio_async(bridge);
    }

    if (bridge->synapses) nimcp_free(bridge->synapses);
    if (bridge->slot_states) nimcp_free(bridge->slot_states);

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }

    nimcp_free(bridge);
}

int wm_plasticity_reset(wm_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Reset all synapses to initial state */
    for (uint32_t i = 0; i < bridge->synapse_count; i++) {
        bridge->synapses[i].weight = bridge->synapses[i].initial_weight;
        bridge->synapses[i].eligibility_trace = 0.0f;
        bridge->synapses[i].bcm_threshold = 0.5f;
        bridge->synapses[i].avg_activity = 0.0f;
        bridge->synapses[i].consolidation_level = 0.0f;
        bridge->synapses[i].maintenance_count = 0;
    }

    /* Reset slot states */
    for (uint32_t s = 0; s < bridge->num_slots; s++) {
        bridge->slot_states[s].occupied = false;
        bridge->slot_states[s].encoding_strength = 0.0f;
        bridge->slot_states[s].current_strength = 0.0f;
        bridge->slot_states[s].consolidation_progress = 0.0f;
        bridge->slot_states[s].rehearsal_count = 0;
        bridge->slot_states[s].last_access_time_us = 0;
        bridge->slot_states[s].salience = 0.0f;
    }

    /* Reset state */
    bridge->state = WM_PLASTICITY_STATE_IDLE;
    bridge->current_capacity_pressure = 0.0f;
    bridge->current_salience_mod = 1.0f;
    bridge->global_learning_rate = 1.0f;
    bridge->pending_reward = 0.0f;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Synapse Management
//=============================================================================

int wm_plasticity_register_synapse(
    wm_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    wm_synapse_type_t type,
    int32_t slot_idx,
    float initial_weight)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Check if already registered */
    if (find_synapse(bridge, synapse_id)) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Check capacity */
    if (bridge->synapse_count >= bridge->synapse_capacity) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Add synapse */
    wm_plasticity_synapse_t* synapse = &bridge->synapses[bridge->synapse_count];
    synapse->synapse_id = synapse_id;
    synapse->type = type;
    synapse->slot_idx = (slot_idx >= 0 && slot_idx < (int32_t)bridge->num_slots) ? slot_idx : -1;
    synapse->weight = clamp_f(initial_weight, bridge->config.weight_min, bridge->config.weight_max);
    synapse->initial_weight = synapse->weight;
    synapse->last_pre_spike_us = 0;
    synapse->last_post_spike_us = 0;
    synapse->eligibility_trace = 0.0f;
    synapse->bcm_threshold = 0.5f;
    synapse->avg_activity = 0.0f;
    synapse->consolidation_level = 0.0f;
    synapse->encoding_time_us = nimcp_time_get_us();
    synapse->maintenance_count = 0;

    bridge->synapse_count++;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_plasticity_unregister_synapse(
    wm_plasticity_bridge_t* bridge,
    uint32_t synapse_id)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    for (uint32_t i = 0; i < bridge->synapse_count; i++) {
        if (bridge->synapses[i].synapse_id == synapse_id) {
            if (i < bridge->synapse_count - 1) {
                bridge->synapses[i] = bridge->synapses[bridge->synapse_count - 1];
            }
            bridge->synapse_count--;
            nimcp_mutex_unlock(bridge->mutex);
            return 0;
        }
    }

    nimcp_mutex_unlock(bridge->mutex);
    return -1;
}

int wm_plasticity_get_synapse(
    wm_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    wm_plasticity_synapse_t* synapse)
{
    if (!bridge || !synapse) return -1;

    nimcp_mutex_lock(bridge->mutex);

    wm_plasticity_synapse_t* found = find_synapse(bridge, synapse_id);
    if (!found) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    *synapse = *found;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Event Recording
//=============================================================================

int wm_plasticity_encode(
    wm_plasticity_bridge_t* bridge,
    uint32_t slot_idx,
    float salience,
    uint64_t timestamp_us)
{
    if (!bridge) return -1;
    if (slot_idx >= bridge->num_slots) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->state = WM_PLASTICITY_STATE_ENCODING;

    wm_slot_plasticity_t* slot = &bridge->slot_states[slot_idx];
    slot->occupied = true;
    slot->encoding_strength = 1.0f;
    slot->current_strength = 1.0f;
    slot->salience = clamp_f(salience, 0.0f, 1.0f);
    slot->consolidation_progress = 0.0f;
    slot->rehearsal_count = 0;
    slot->last_access_time_us = timestamp_us;

    /* Apply LTP to encoding synapses for this slot */
    float salience_mod = bridge->config.enable_salience_modulation ?
                         (1.0f + salience * bridge->config.salience_ltp_gain) : 1.0f;

    for (uint32_t i = 0; i < bridge->synapse_count; i++) {
        wm_plasticity_synapse_t* syn = &bridge->synapses[i];
        if (syn->slot_idx == (int32_t)slot_idx && syn->type == WM_SYNAPSE_ENCODING) {
            syn->last_pre_spike_us = timestamp_us;
            syn->encoding_time_us = timestamp_us;

            /* Encoding LTP */
            float dw = bridge->config.stdp_a_plus * salience_mod * bridge->global_learning_rate;
            float old_weight = syn->weight;
            syn->weight = clamp_f(syn->weight + dw, bridge->config.weight_min, bridge->config.weight_max);

            if (bridge->weight_callback) {
                bridge->weight_callback(syn->synapse_id, slot_idx, old_weight,
                                       syn->weight, WM_LEARN_ENCODING,
                                       bridge->weight_callback_data);
            }

            /* Update eligibility */
            if (bridge->config.enable_eligibility) {
                syn->eligibility_trace += salience;
            }

            bridge->stats.ltp_events++;
        }
    }

    bridge->stats.total_encodings++;
    bridge->state = WM_PLASTICITY_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_plasticity_maintain(
    wm_plasticity_bridge_t* bridge,
    uint32_t slot_idx,
    float activity_level,
    uint64_t timestamp_us)
{
    if (!bridge) return -1;
    if (slot_idx >= bridge->num_slots) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->state = WM_PLASTICITY_STATE_MAINTAINING;

    wm_slot_plasticity_t* slot = &bridge->slot_states[slot_idx];
    if (!slot->occupied) {
        bridge->state = WM_PLASTICITY_STATE_IDLE;
        nimcp_mutex_unlock(bridge->mutex);
        return 0;
    }

    /* Update slot state */
    slot->current_strength = activity_level;
    slot->last_access_time_us = timestamp_us;
    slot->rehearsal_count++;

    /* Apply maintenance LTP to recurrent synapses */
    if (bridge->config.enable_maintenance_ltp) {
        float rehearsal_boost = bridge->config.enable_rehearsal_boost ?
                               bridge->config.rehearsal_ltp_gain : 1.0f;

        for (uint32_t i = 0; i < bridge->synapse_count; i++) {
            wm_plasticity_synapse_t* syn = &bridge->synapses[i];
            if (syn->slot_idx == (int32_t)slot_idx && syn->type == WM_SYNAPSE_MAINTENANCE) {
                syn->last_pre_spike_us = timestamp_us;
                syn->maintenance_count++;

                float dw = bridge->config.maintenance_ltp_rate * activity_level * rehearsal_boost;
                float old_weight = syn->weight;
                syn->weight = clamp_f(syn->weight + dw, bridge->config.weight_min, bridge->config.weight_max);

                if (bridge->weight_callback) {
                    bridge->weight_callback(syn->synapse_id, slot_idx, old_weight,
                                           syn->weight, WM_LEARN_MAINTENANCE,
                                           bridge->weight_callback_data);
                }

                bridge->stats.ltp_events++;
            }
        }
    }

    /* Update consolidation progress */
    float time_held_ms = (float)(timestamp_us - slot->last_access_time_us + 1000) / 1000.0f;
    if (time_held_ms > 0 && bridge->config.enable_consolidation) {
        slot->consolidation_progress += time_held_ms / bridge->config.consolidation_threshold;
        slot->consolidation_progress = clamp_f(slot->consolidation_progress, 0.0f, 1.0f);
    }

    bridge->stats.total_maintenance_cycles++;
    bridge->state = WM_PLASTICITY_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_plasticity_retrieve(
    wm_plasticity_bridge_t* bridge,
    uint32_t slot_idx,
    float retrieval_strength,
    uint64_t timestamp_us)
{
    if (!bridge) return -1;
    if (slot_idx >= bridge->num_slots) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->state = WM_PLASTICITY_STATE_RETRIEVING;

    wm_slot_plasticity_t* slot = &bridge->slot_states[slot_idx];
    slot->last_access_time_us = timestamp_us;

    /* Retrieval strengthens both encoding and retrieval synapses */
    for (uint32_t i = 0; i < bridge->synapse_count; i++) {
        wm_plasticity_synapse_t* syn = &bridge->synapses[i];
        if (syn->slot_idx == (int32_t)slot_idx &&
            (syn->type == WM_SYNAPSE_RETRIEVAL || syn->type == WM_SYNAPSE_ENCODING)) {
            syn->last_post_spike_us = timestamp_us;

            /* STDP: post spike -> check for LTP with recent pre spikes */
            if (syn->last_pre_spike_us > 0) {
                float dt_ms = (float)(timestamp_us - syn->last_pre_spike_us) / 1000.0f;
                if (dt_ms > 0 && dt_ms < bridge->config.stdp_ltp_window_ms) {
                    float dw = compute_stdp_weight_change(&bridge->config, dt_ms, retrieval_strength);
                    float old_weight = syn->weight;
                    syn->weight = clamp_f(syn->weight + dw, bridge->config.weight_min, bridge->config.weight_max);

                    if (bridge->weight_callback) {
                        bridge->weight_callback(syn->synapse_id, slot_idx, old_weight,
                                               syn->weight, WM_LEARN_RETRIEVAL,
                                               bridge->weight_callback_data);
                    }

                    if (dw > 0) bridge->stats.ltp_events++;
                    else bridge->stats.ltd_events++;
                }
            }
        }
    }

    bridge->stats.total_retrievals++;
    bridge->state = WM_PLASTICITY_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_plasticity_evict(
    wm_plasticity_bridge_t* bridge,
    uint32_t slot_idx,
    uint64_t timestamp_us)
{
    if (!bridge) return -1;
    if (slot_idx >= bridge->num_slots) return -1;

    nimcp_mutex_lock(bridge->mutex);

    wm_slot_plasticity_t* slot = &bridge->slot_states[slot_idx];
    slot->occupied = false;

    /* Capacity-based LTD for evicted items */
    if (bridge->config.enable_capacity_ltd) {
        for (uint32_t i = 0; i < bridge->synapse_count; i++) {
            wm_plasticity_synapse_t* syn = &bridge->synapses[i];
            if (syn->slot_idx == (int32_t)slot_idx) {
                float dw = -bridge->config.capacity_ltd_rate * bridge->current_capacity_pressure;
                float old_weight = syn->weight;
                syn->weight = clamp_f(syn->weight + dw, bridge->config.weight_min, bridge->config.weight_max);

                if (bridge->weight_callback) {
                    bridge->weight_callback(syn->synapse_id, slot_idx, old_weight,
                                           syn->weight, WM_LEARN_EVICTION,
                                           bridge->weight_callback_data);
                }

                bridge->stats.ltd_events++;
            }
        }
    }

    bridge->stats.total_evictions++;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_plasticity_decay(
    wm_plasticity_bridge_t* bridge,
    uint32_t slot_idx,
    float new_strength,
    uint64_t timestamp_us)
{
    if (!bridge) return -1;
    if (slot_idx >= bridge->num_slots) return -1;

    nimcp_mutex_lock(bridge->mutex);

    wm_slot_plasticity_t* slot = &bridge->slot_states[slot_idx];
    float decay_amount = slot->current_strength - new_strength;
    slot->current_strength = new_strength;

    /* Decay causes minor LTD - track any decay as an LTD event */
    if (decay_amount > 0.0f) {
        bridge->stats.ltd_events++;

        /* Apply weight changes for significant decay */
        if (decay_amount > 0.1f) {
            for (uint32_t i = 0; i < bridge->synapse_count; i++) {
                wm_plasticity_synapse_t* syn = &bridge->synapses[i];
                if (syn->slot_idx == (int32_t)slot_idx && syn->type == WM_SYNAPSE_MAINTENANCE) {
                    float dw = -decay_amount * 0.001f;
                    float old_weight = syn->weight;
                    syn->weight = clamp_f(syn->weight + dw, bridge->config.weight_min, bridge->config.weight_max);

                    if (bridge->weight_callback) {
                        bridge->weight_callback(syn->synapse_id, slot_idx, old_weight,
                                               syn->weight, WM_LEARN_DECAY,
                                               bridge->weight_callback_data);
                    }
                }
            }
        }
    }

    /* Check if item has fully decayed */
    if (new_strength < 0.01f) {
        slot->occupied = false;
    }

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_plasticity_reward(
    wm_plasticity_bridge_t* bridge,
    float reward,
    uint64_t timestamp_us)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->pending_reward = reward;
    bridge->last_reward_time_us = timestamp_us;

    /* Apply reward to all eligible synapses */
    if (bridge->config.enable_eligibility) {
        for (uint32_t i = 0; i < bridge->synapse_count; i++) {
            wm_plasticity_synapse_t* syn = &bridge->synapses[i];
            if (syn->eligibility_trace > 0.01f) {
                float dw = apply_eligibility(syn, reward, bridge->config.reward_modulation_gain);
                float old_weight = syn->weight;
                syn->weight = clamp_f(syn->weight + dw, bridge->config.weight_min, bridge->config.weight_max);

                if (reward > 0) bridge->stats.ltp_events++;
                else bridge->stats.ltd_events++;

                bridge->stats.avg_weight_change += fabsf(syn->weight - old_weight);
            }
        }
    }

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Update Functions
//=============================================================================

int wm_plasticity_update(
    wm_plasticity_bridge_t* bridge,
    float dt_ms)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    uint64_t now_us = nimcp_time_get_us();

    /* Update capacity pressure */
    uint32_t occupied_count = 0;
    for (uint32_t s = 0; s < bridge->num_slots; s++) {
        if (bridge->slot_states[s].occupied) occupied_count++;
    }
    bridge->current_capacity_pressure = (float)occupied_count / (float)bridge->num_slots;
    bridge->stats.avg_capacity_utilization = bridge->current_capacity_pressure;

    /* Update synapses */
    for (uint32_t i = 0; i < bridge->synapse_count; i++) {
        wm_plasticity_synapse_t* syn = &bridge->synapses[i];

        /* Decay eligibility traces */
        if (bridge->config.enable_eligibility) {
            syn->eligibility_trace *= bridge->config.eligibility_decay;
        }

        /* Update BCM threshold */
        if (bridge->config.enable_bcm) {
            update_bcm_threshold(syn, syn->avg_activity, bridge->config.bcm_threshold_tau);
        }

        /* Update consolidation level */
        if (bridge->config.enable_consolidation && syn->slot_idx >= 0) {
            wm_slot_plasticity_t* slot = &bridge->slot_states[syn->slot_idx];
            if (slot->occupied && slot->consolidation_progress >= 1.0f) {
                /* Full consolidation reached */
                float consol_boost = bridge->config.consolidation_ltp_boost;
                syn->consolidation_level = 1.0f;
                syn->weight = clamp_f(syn->weight * consol_boost,
                                     bridge->config.weight_min, bridge->config.weight_max);
            }
        }

        /* Homeostatic scaling */
        if (bridge->config.enable_homeostatic) {
            float diff = bridge->config.target_capacity_utilization - bridge->current_capacity_pressure;
            float scale = 1.0f + diff * (dt_ms / bridge->config.homeostatic_tau_ms);
            syn->weight *= scale;
            syn->weight = clamp_f(syn->weight, bridge->config.weight_min, bridge->config.weight_max);
        }

        /* Update activity average */
        float alpha = dt_ms / bridge->config.bcm_activity_tau;
        float recent_activity = (syn->last_pre_spike_us > now_us - 100000) ? 1.0f : 0.0f;
        syn->avg_activity = (1.0f - alpha) * syn->avg_activity + alpha * recent_activity;
    }

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_plasticity_consolidate_slot(
    wm_plasticity_bridge_t* bridge,
    uint32_t slot_idx)
{
    if (!bridge) return -1;
    if (slot_idx >= bridge->num_slots) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->state = WM_PLASTICITY_STATE_CONSOLIDATING;

    wm_slot_plasticity_t* slot = &bridge->slot_states[slot_idx];

    if (slot->occupied && bridge->config.enable_consolidation) {
        slot->consolidation_progress = 1.0f;

        /* Apply consolidation LTP boost */
        for (uint32_t i = 0; i < bridge->synapse_count; i++) {
            wm_plasticity_synapse_t* syn = &bridge->synapses[i];
            if (syn->slot_idx == (int32_t)slot_idx) {
                syn->consolidation_level = 1.0f;
                float old_weight = syn->weight;
                syn->weight *= bridge->config.consolidation_ltp_boost;
                syn->weight = clamp_f(syn->weight, bridge->config.weight_min, bridge->config.weight_max);

                if (bridge->weight_callback) {
                    bridge->weight_callback(syn->synapse_id, slot_idx, old_weight,
                                           syn->weight, WM_LEARN_CONSOLIDATION,
                                           bridge->weight_callback_data);
                }

                bridge->stats.ltp_events++;
            }
        }

        /* Fire consolidation callback */
        if (bridge->consolidation_callback) {
            bridge->consolidation_callback(slot_idx, 1.0f, bridge->consolidation_callback_data);
        }

        bridge->stats.total_consolidations++;
    }

    bridge->state = WM_PLASTICITY_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_plasticity_consolidate_all(wm_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_sleep_consolidation) return 0;

    nimcp_mutex_lock(bridge->mutex);

    bridge->state = WM_PLASTICITY_STATE_CONSOLIDATING;

    /* Consolidate all occupied slots */
    for (uint32_t s = 0; s < bridge->num_slots; s++) {
        if (bridge->slot_states[s].occupied) {
            nimcp_mutex_unlock(bridge->mutex);
            wm_plasticity_consolidate_slot(bridge, s);
            nimcp_mutex_lock(bridge->mutex);
        }
    }

    /* Reset eligibility traces */
    for (uint32_t i = 0; i < bridge->synapse_count; i++) {
        bridge->synapses[i].eligibility_trace = 0.0f;
    }

    bridge->state = WM_PLASTICITY_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Query Functions
//=============================================================================

float wm_plasticity_get_encoding_strength(
    wm_plasticity_bridge_t* bridge,
    uint32_t slot_idx)
{
    if (!bridge) return -1.0f;
    if (slot_idx >= bridge->num_slots) return -1.0f;

    nimcp_mutex_lock(bridge->mutex);

    float strength = bridge->slot_states[slot_idx].encoding_strength;

    nimcp_mutex_unlock(bridge->mutex);

    return strength;
}

float wm_plasticity_get_consolidation_level(
    wm_plasticity_bridge_t* bridge,
    uint32_t slot_idx)
{
    if (!bridge) return -1.0f;
    if (slot_idx >= bridge->num_slots) return -1.0f;

    nimcp_mutex_lock(bridge->mutex);

    float level = bridge->slot_states[slot_idx].consolidation_progress;

    nimcp_mutex_unlock(bridge->mutex);

    return level;
}

float wm_plasticity_get_retrieval_priority(
    wm_plasticity_bridge_t* bridge,
    uint32_t slot_idx)
{
    if (!bridge) return -1.0f;
    if (slot_idx >= bridge->num_slots) return -1.0f;

    nimcp_mutex_lock(bridge->mutex);

    wm_slot_plasticity_t* slot = &bridge->slot_states[slot_idx];

    /* Priority based on salience, consolidation, and recency */
    float priority = slot->salience * 0.4f +
                    slot->consolidation_progress * 0.3f +
                    slot->current_strength * 0.3f;

    nimcp_mutex_unlock(bridge->mutex);

    return priority;
}

int wm_plasticity_get_maintenance_modulation(
    wm_plasticity_bridge_t* bridge,
    float* modulation,
    uint32_t num_slots)
{
    if (!bridge || !modulation) return -1;

    nimcp_mutex_lock(bridge->mutex);

    uint32_t n = (num_slots < bridge->num_slots) ? num_slots : bridge->num_slots;

    for (uint32_t s = 0; s < n; s++) {
        /* Modulation based on encoding strength and rehearsal count */
        wm_slot_plasticity_t* slot = &bridge->slot_states[s];
        modulation[s] = slot->encoding_strength *
                       (1.0f + 0.1f * (float)slot->rehearsal_count);
        modulation[s] = clamp_f(modulation[s], 0.0f, 2.0f);
    }

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// State and Statistics
//=============================================================================

int wm_plasticity_get_state(
    const wm_plasticity_bridge_t* bridge,
    wm_plasticity_bridge_state_t* state)
{
    if (!bridge || !state) return -1;

    wm_plasticity_bridge_t* mutable_bridge = (wm_plasticity_bridge_t*)bridge;

    nimcp_mutex_lock(mutable_bridge->mutex);

    state->state = bridge->state;
    state->registered_synapses = bridge->synapse_count;

    /* Count active slots */
    state->active_slots = 0;
    for (uint32_t s = 0; s < bridge->num_slots; s++) {
        if (bridge->slot_states[s].occupied) state->active_slots++;
    }

    state->global_learning_rate = bridge->global_learning_rate;
    state->current_capacity_pressure = bridge->current_capacity_pressure;
    state->current_salience_mod = bridge->current_salience_mod;
    state->bio_async_connected = bridge->bio_async_connected;

    nimcp_mutex_unlock(mutable_bridge->mutex);

    return 0;
}

int wm_plasticity_get_stats(
    const wm_plasticity_bridge_t* bridge,
    wm_plasticity_stats_t* stats)
{
    if (!bridge || !stats) return -1;

    wm_plasticity_bridge_t* mutable_bridge = (wm_plasticity_bridge_t*)bridge;

    nimcp_mutex_lock(mutable_bridge->mutex);

    *stats = bridge->stats;

    uint64_t total_events = stats->ltp_events + stats->ltd_events;
    if (total_events > 0) {
        stats->avg_weight_change /= (float)total_events;
    }

    nimcp_mutex_unlock(mutable_bridge->mutex);

    return 0;
}

void wm_plasticity_reset_stats(wm_plasticity_bridge_t* bridge) {
    if (!bridge) return;

    nimcp_mutex_lock(bridge->mutex);

    memset(&bridge->stats, 0, sizeof(wm_plasticity_stats_t));

    nimcp_mutex_unlock(bridge->mutex);
}

//=============================================================================
// Callbacks
//=============================================================================

int wm_plasticity_set_weight_callback(
    wm_plasticity_bridge_t* bridge,
    wm_weight_change_cb callback,
    void* user_data)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->weight_callback = callback;
    bridge->weight_callback_data = user_data;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_plasticity_set_consolidation_callback(
    wm_plasticity_bridge_t* bridge,
    wm_consolidation_cb callback,
    void* user_data)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->consolidation_callback = callback;
    bridge->consolidation_callback_data = user_data;

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Modulation Functions
//=============================================================================

int wm_plasticity_set_capacity_pressure(
    wm_plasticity_bridge_t* bridge,
    float pressure)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->current_capacity_pressure = clamp_f(pressure, 0.0f, 1.0f);

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_plasticity_set_salience_modulation(
    wm_plasticity_bridge_t* bridge,
    float salience_level)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->current_salience_mod = clamp_f(salience_level, 0.0f, 2.0f);

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int wm_plasticity_connect_bio_async(wm_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return 0;

    nimcp_mutex_lock(bridge->mutex);

    bridge->bio_async_connected = true;

    nimcp_mutex_unlock(bridge->mutex);

    return bridge->bio_async_connected ? 0 : -1;
}

int wm_plasticity_disconnect_bio_async(wm_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    if (bridge->bio_async_connected) {
        bridge->bio_async_connected = false;
    }

    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

bool wm_plasticity_is_bio_async_connected(const wm_plasticity_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->bio_async_connected;
}
