/**
 * @file nimcp_working_memory_snn_bridge.c
 * @brief Working Memory - SNN Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/memory/nimcp_working_memory_snn_bridge.h"
#include "snn/nimcp_snn_network.h"
#include "snn/nimcp_snn_config.h"
#include "core/neuron_types/nimcp_neuron_types.h"
#include "core/synapse_types/nimcp_synapse_types.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "utils/thread/nimcp_thread.h"

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

//=============================================================================
// Module Constants
//=============================================================================

#define LOG_MODULE "WM_SNN"

//=============================================================================
// Internal Structures
//=============================================================================

struct wm_snn_bridge {
    wm_snn_config_t config;

    /* SNN network */
    snn_network_t* snn;
    bool owns_snn;

    /* Population IDs */
    uint32_t* slot_pops;             /* One population per slot */
    uint32_t hidden_pop;
    uint32_t output_pop;
    uint32_t inhibition_pop;

    /* Per-slot state */
    wm_slot_state_t* slot_states;
    float* slot_buffer;              /* Temporary buffer for encoding */
    float* output_buffer;            /* Temporary buffer for decoding */

    /* Bridge state */
    wm_snn_state_t state;
    wm_snn_stats_t stats;
    uint64_t current_time_ms;

    /* Callbacks */
    wm_snn_spike_callback_t spike_callback;
    void* spike_user_data;
    wm_snn_encoding_callback_t encoding_callback;
    void* encoding_user_data;
    wm_snn_retrieval_callback_t retrieval_callback;
    void* retrieval_user_data;

    /* Bio-async */
    bool bio_async_connected;

    /* Thread safety */
    nimcp_mutex_t* mutex;
};

//=============================================================================
// Helper Functions
//=============================================================================

static inline float clamp_f(float v, float lo, float hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

//=============================================================================
// Default Configuration
//=============================================================================

wm_snn_config_t wm_snn_config_default(void) {
    return (wm_snn_config_t) {
        .max_slots = 8,
        .neurons_per_slot = WM_SNN_NEURONS_PER_SLOT,
        .slot_dim = WM_SNN_SLOT_DIM,
        .hidden_dim = WM_SNN_HIDDEN_DIM,

        .dt_ms = 1.0f,
        .maintenance_rate = 20.0f,
        .decay_tau_ms = 500.0f,

        .encoding_type = WM_SNN_ENCODE_RATE,
        .encoding_gain = 1.0f,
        .noise_stddev = 0.05f,

        .enable_lateral_inhibition = true,
        .inhibition_strength = 0.3f,
        .enable_recurrence = true,
        .recurrence_strength = 0.4f,

        .enable_bio_async = false
    };
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

wm_snn_bridge_t* wm_snn_create(const wm_snn_config_t* config) {
    wm_snn_bridge_t* bridge = nimcp_calloc(1, sizeof(wm_snn_bridge_t));
    if (!bridge) {
        NIMCP_LOG_ERROR(LOG_MODULE, "Failed to allocate bridge");
        return NULL;
    }

    /* Apply configuration */
    bridge->config = config ? *config : wm_snn_config_default();

    /* Validate */
    if (bridge->config.max_slots == 0 ||
        bridge->config.max_slots > WM_SNN_MAX_SLOTS) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Create mutex */
    mutex_attr_t mutex_attr = {.type = MUTEX_TYPE_NORMAL};
    bridge->mutex = nimcp_mutex_create(&mutex_attr);
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Create SNN network */
    snn_config_t snn_config;
    uint32_t input_dim = bridge->config.max_slots * bridge->config.neurons_per_slot;

    snn_config_feedforward(&snn_config,
        input_dim,
        bridge->config.hidden_dim,
        bridge->config.max_slots);

    snn_config.dt = bridge->config.dt_ms;
    snn_config.enable_bio_async = bridge->config.enable_bio_async;
    snn_config.n_populations = 0;

    bridge->snn = snn_network_create(&snn_config);
    if (!bridge->snn) {
        NIMCP_LOG_ERROR(LOG_MODULE, "Failed to create SNN network");
        nimcp_mutex_free(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }
    bridge->owns_snn = true;

    /* Allocate population IDs */
    bridge->slot_pops = nimcp_calloc(bridge->config.max_slots, sizeof(uint32_t));
    if (!bridge->slot_pops) {
        wm_snn_destroy(bridge);
        return NULL;
    }

    /* Create populations for each memory slot */
    char pop_name[64];
    for (uint32_t s = 0; s < bridge->config.max_slots; s++) {
        snprintf(pop_name, sizeof(pop_name), "wm_slot_%u", s);
        bridge->slot_pops[s] = snn_network_add_population(
            bridge->snn, bridge->config.neurons_per_slot,
            NEURON_GENERIC_LIF, pop_name);
    }

    /* Hidden population */
    bridge->hidden_pop = snn_network_add_population(
        bridge->snn, bridge->config.hidden_dim,
        NEURON_GENERIC_LIF, "wm_hidden");

    /* Output population (one neuron per slot for retrieval) */
    bridge->output_pop = snn_network_add_population(
        bridge->snn, bridge->config.max_slots,
        NEURON_GENERIC_LIF, "wm_output");

    /* Connect slot populations to hidden */
    for (uint32_t s = 0; s < bridge->config.max_slots; s++) {
        snn_network_connect_populations(bridge->snn,
            bridge->slot_pops[s], bridge->hidden_pop,
            SNN_TOPO_RANDOM, 0.3f, SYNAPSE_AMPA, 0.5f, 0.1f);
    }

    /* Hidden to output */
    snn_network_connect_populations(bridge->snn,
        bridge->hidden_pop, bridge->output_pop,
        SNN_TOPO_FEEDFORWARD, 1.0f, SYNAPSE_AMPA, 0.5f, 0.1f);

    /* Recurrent connections for maintenance */
    if (bridge->config.enable_recurrence) {
        for (uint32_t s = 0; s < bridge->config.max_slots; s++) {
            snn_network_connect_populations(bridge->snn,
                bridge->slot_pops[s], bridge->slot_pops[s],
                SNN_TOPO_RANDOM, 0.5f, SYNAPSE_AMPA,
                bridge->config.recurrence_strength, 0.05f);
        }
    }

    /* Lateral inhibition between slots */
    if (bridge->config.enable_lateral_inhibition) {
        bridge->inhibition_pop = snn_network_add_population(
            bridge->snn, bridge->config.max_slots,
            NEURON_GENERIC_LIF, "wm_inhibition");

        for (uint32_t s = 0; s < bridge->config.max_slots; s++) {
            /* Slot -> inhibition */
            snn_network_connect_populations(bridge->snn,
                bridge->slot_pops[s], bridge->inhibition_pop,
                SNN_TOPO_FEEDFORWARD, 1.0f, SYNAPSE_AMPA, 0.3f, 0.05f);

            /* Inhibition -> all other slots */
            for (uint32_t t = 0; t < bridge->config.max_slots; t++) {
                if (t != s) {
                    snn_network_connect_populations(bridge->snn,
                        bridge->inhibition_pop, bridge->slot_pops[t],
                        SNN_TOPO_RANDOM, 0.3f, SYNAPSE_GABA_A,
                        -bridge->config.inhibition_strength, 0.05f);
                }
            }
        }
    }

    /* Allocate per-slot state */
    bridge->slot_states = nimcp_calloc(bridge->config.max_slots,
                                        sizeof(wm_slot_state_t));
    if (!bridge->slot_states) {
        wm_snn_destroy(bridge);
        return NULL;
    }

    /* Allocate buffers */
    bridge->slot_buffer = nimcp_calloc(input_dim, sizeof(float));
    bridge->output_buffer = nimcp_calloc(bridge->config.max_slots, sizeof(float));
    if (!bridge->slot_buffer || !bridge->output_buffer) {
        wm_snn_destroy(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->state = WM_SNN_STATE_IDLE;
    bridge->current_time_ms = 0;

    NIMCP_LOG_INFO(LOG_MODULE, "Created WM-SNN bridge with %u slots",
                   bridge->config.max_slots);

    return bridge;
}

void wm_snn_destroy(wm_snn_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->snn && bridge->owns_snn) {
        snn_network_destroy(bridge->snn);
    }

    if (bridge->slot_pops) nimcp_free(bridge->slot_pops);
    if (bridge->slot_states) nimcp_free(bridge->slot_states);
    if (bridge->slot_buffer) nimcp_free(bridge->slot_buffer);
    if (bridge->output_buffer) nimcp_free(bridge->output_buffer);
    if (bridge->mutex) nimcp_mutex_free(bridge->mutex);

    nimcp_free(bridge);
}

int wm_snn_reset(wm_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Reset slot states */
    memset(bridge->slot_states, 0,
           bridge->config.max_slots * sizeof(wm_slot_state_t));

    /* Reset SNN */
    if (bridge->snn) {
        snn_network_reset(bridge->snn);
    }

    /* Clear buffers */
    uint32_t input_dim = bridge->config.max_slots * bridge->config.neurons_per_slot;
    memset(bridge->slot_buffer, 0, input_dim * sizeof(float));
    memset(bridge->output_buffer, 0, bridge->config.max_slots * sizeof(float));

    bridge->state = WM_SNN_STATE_IDLE;
    bridge->current_time_ms = 0;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Encoding Functions
//=============================================================================

int wm_snn_encode_item(
    wm_snn_bridge_t* bridge,
    uint32_t slot,
    const float* features,
    uint32_t feature_count,
    float salience)
{
    if (!bridge || !features || slot >= bridge->config.max_slots) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->state = WM_SNN_STATE_ENCODING;

    /* Calculate spike pattern for this slot */
    uint32_t neurons = bridge->config.neurons_per_slot;
    uint32_t offset = slot * neurons;
    int spike_count = 0;

    /* Map features to neuron firing rates */
    for (uint32_t n = 0; n < neurons; n++) {
        float rate = 0.0f;

        /* Average features that map to this neuron */
        uint32_t feat_start = (n * feature_count) / neurons;
        uint32_t feat_end = ((n + 1) * feature_count) / neurons;
        if (feat_end > feature_count) feat_end = feature_count;

        for (uint32_t f = feat_start; f < feat_end; f++) {
            rate += features[f];
        }
        if (feat_end > feat_start) {
            rate /= (float)(feat_end - feat_start);
        }

        /* Apply encoding gain and salience */
        rate = clamp_f(rate * bridge->config.encoding_gain * salience, 0.0f, 1.0f);
        bridge->slot_buffer[offset + n] = rate;

        /* Count spikes (probabilistic based on rate) */
        if (rate > 0.1f) spike_count++;
    }

    /* Update slot state */
    bridge->slot_states[slot].occupied = true;
    bridge->slot_states[slot].activity_level = (float)spike_count / (float)neurons;
    bridge->slot_states[slot].salience = salience;
    bridge->slot_states[slot].encode_time = bridge->current_time_ms;
    bridge->slot_states[slot].persistence = 0.0f;

    /* Set inputs to SNN */
    uint32_t input_dim = bridge->config.max_slots * bridge->config.neurons_per_slot;
    snn_network_set_inputs(bridge->snn, bridge->slot_buffer, input_dim);

    /* Update statistics */
    bridge->stats.total_encodings++;
    bridge->stats.total_spikes += (uint64_t)spike_count;

    /* Trigger callback */
    if (bridge->encoding_callback) {
        bridge->encoding_callback(bridge, slot, spike_count,
                                  bridge->encoding_user_data);
    }

    bridge->state = WM_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);
    return spike_count;
}

int wm_snn_update_item(
    wm_snn_bridge_t* bridge,
    uint32_t slot,
    const float* features,
    uint32_t feature_count)
{
    if (!bridge || slot >= bridge->config.max_slots) return -1;
    if (!bridge->slot_states[slot].occupied) return -1;

    /* Re-encode with existing salience */
    return wm_snn_encode_item(bridge, slot, features, feature_count,
                              bridge->slot_states[slot].salience);
}

int wm_snn_clear_slot(wm_snn_bridge_t* bridge, uint32_t slot) {
    if (!bridge || slot >= bridge->config.max_slots) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Clear slot buffer */
    uint32_t offset = slot * bridge->config.neurons_per_slot;
    memset(&bridge->slot_buffer[offset], 0,
           bridge->config.neurons_per_slot * sizeof(float));

    /* Clear slot state */
    memset(&bridge->slot_states[slot], 0, sizeof(wm_slot_state_t));

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Simulation Functions
//=============================================================================

int wm_snn_simulate(wm_snn_bridge_t* bridge, float duration_ms) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->state = WM_SNN_STATE_SIMULATING;

    int ret = snn_network_run(bridge->snn, duration_ms);

    if (ret >= 0) {  /* snn_network_run returns spike count, >= 0 means success */
        bridge->current_time_ms += (uint64_t)duration_ms;
        bridge->stats.total_simulations++;
        bridge->stats.total_spikes += (uint64_t)ret;

        /* Update slot persistence */
        for (uint32_t s = 0; s < bridge->config.max_slots; s++) {
            if (bridge->slot_states[s].occupied) {
                bridge->slot_states[s].persistence += duration_ms;

                /* Apply decay */
                float decay = expf(-duration_ms / bridge->config.decay_tau_ms);
                bridge->slot_states[s].activity_level *= decay;
            }
        }
    }

    bridge->state = WM_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);
    return ret;
}

int wm_snn_step(wm_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    return wm_snn_simulate(bridge, bridge->config.dt_ms);
}

int wm_snn_forward(
    wm_snn_bridge_t* bridge,
    const float* inputs,
    uint32_t input_count)
{
    if (!bridge || !inputs) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->state = WM_SNN_STATE_SIMULATING;

    uint32_t input_dim = bridge->config.max_slots * bridge->config.neurons_per_slot;
    uint32_t n = (input_count < input_dim) ? input_count : input_dim;

    memcpy(bridge->slot_buffer, inputs, n * sizeof(float));
    snn_network_set_inputs(bridge->snn, bridge->slot_buffer, input_dim);

    int ret = snn_network_step(bridge->snn, bridge->config.dt_ms);

    int spike_count = 0;
    if (ret >= 0) {  /* snn_network_step returns spike count, >= 0 means success */
        bridge->stats.total_simulations++;
        spike_count = ret;
        bridge->stats.total_spikes += (uint64_t)spike_count;
    }

    bridge->state = WM_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);
    return spike_count;
}

//=============================================================================
// Retrieval Functions
//=============================================================================

int wm_snn_retrieve_item(
    wm_snn_bridge_t* bridge,
    uint32_t slot,
    float* output,
    uint32_t output_size)
{
    if (!bridge || !output || slot >= bridge->config.max_slots) return -1;
    if (!bridge->slot_states[slot].occupied) return -1;

    nimcp_mutex_lock(bridge->mutex);

    bridge->state = WM_SNN_STATE_RETRIEVING;

    /* Get slot activity */
    uint32_t neurons = bridge->config.neurons_per_slot;
    uint32_t offset = slot * neurons;
    uint32_t n = (output_size < neurons) ? output_size : neurons;

    memcpy(output, &bridge->slot_buffer[offset], n * sizeof(float));

    /* Update statistics */
    bridge->stats.total_retrievals++;
    bridge->slot_states[slot].retrieval_count++;

    /* Calculate confidence from activity */
    float confidence = bridge->slot_states[slot].activity_level;
    if (confidence > 0.3f) {
        bridge->stats.successful_retrievals++;
    }

    /* Trigger callback */
    if (bridge->retrieval_callback) {
        bridge->retrieval_callback(bridge, slot, confidence,
                                   bridge->retrieval_user_data);
    }

    bridge->state = WM_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int wm_snn_get_slot_activities(
    wm_snn_bridge_t* bridge,
    float* activities,
    uint32_t slot_count)
{
    if (!bridge || !activities) return -1;

    nimcp_mutex_lock(bridge->mutex);

    uint32_t n = (slot_count < bridge->config.max_slots) ?
                  slot_count : bridge->config.max_slots;

    for (uint32_t s = 0; s < n; s++) {
        activities[s] = bridge->slot_states[s].activity_level;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int wm_snn_get_most_active_slot(
    wm_snn_bridge_t* bridge,
    float* confidence)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    int best_slot = -1;
    float best_activity = 0.0f;

    for (uint32_t s = 0; s < bridge->config.max_slots; s++) {
        if (bridge->slot_states[s].occupied &&
            bridge->slot_states[s].activity_level > best_activity) {
            best_activity = bridge->slot_states[s].activity_level;
            best_slot = (int)s;
        }
    }

    if (confidence) {
        *confidence = best_activity;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return best_slot;
}

//=============================================================================
// State Query Functions
//=============================================================================

int wm_snn_get_slot_state(
    wm_snn_bridge_t* bridge,
    uint32_t slot,
    wm_slot_state_t* state)
{
    if (!bridge || !state || slot >= bridge->config.max_slots) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->slot_states[slot];
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_snn_get_state(
    wm_snn_bridge_t* bridge,
    wm_snn_bridge_state_t* state)
{
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->mutex);

    state->state = bridge->state;
    state->active_slots = 0;
    state->total_activity = 0.0f;
    state->mean_persistence = 0.0f;

    for (uint32_t s = 0; s < bridge->config.max_slots; s++) {
        if (bridge->slot_states[s].occupied) {
            state->active_slots++;
            state->total_activity += bridge->slot_states[s].activity_level;
            state->mean_persistence += bridge->slot_states[s].persistence;
        }
    }

    if (state->active_slots > 0) {
        state->mean_persistence /= (float)state->active_slots;
    }

    state->capacity_used = (float)state->active_slots /
                           (float)bridge->config.max_slots;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int wm_snn_get_stats(wm_snn_bridge_t* bridge, wm_snn_stats_t* stats) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;

    /* Calculate derived statistics */
    if (stats->total_encodings > 0) {
        stats->mean_encoding_spikes = (float)stats->total_spikes /
                                       (float)stats->total_encodings;
    }
    if (stats->total_retrievals > 0) {
        stats->mean_retrieval_accuracy = (float)stats->successful_retrievals /
                                          (float)stats->total_retrievals;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int wm_snn_reset_stats(wm_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(wm_snn_stats_t));
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

float wm_snn_get_capacity(wm_snn_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(bridge->mutex);

    uint32_t active = 0;
    for (uint32_t s = 0; s < bridge->config.max_slots; s++) {
        if (bridge->slot_states[s].occupied) active++;
    }

    float capacity = (float)active / (float)bridge->config.max_slots;

    nimcp_mutex_unlock(bridge->mutex);
    return capacity;
}

float wm_snn_get_total_activity(wm_snn_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(bridge->mutex);

    float total = 0.0f;
    for (uint32_t s = 0; s < bridge->config.max_slots; s++) {
        total += bridge->slot_states[s].activity_level;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return total;
}

//=============================================================================
// Callback Registration
//=============================================================================

int wm_snn_register_spike_callback(
    wm_snn_bridge_t* bridge,
    wm_snn_spike_callback_t callback,
    void* user_data)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->spike_callback = callback;
    bridge->spike_user_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_snn_register_encoding_callback(
    wm_snn_bridge_t* bridge,
    wm_snn_encoding_callback_t callback,
    void* user_data)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->encoding_callback = callback;
    bridge->encoding_user_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_snn_register_retrieval_callback(
    wm_snn_bridge_t* bridge,
    wm_snn_retrieval_callback_t callback,
    void* user_data)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->retrieval_callback = callback;
    bridge->retrieval_user_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int wm_snn_bio_async_connect(wm_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int wm_snn_bio_async_disconnect(wm_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

bool wm_snn_is_bio_async_connected(wm_snn_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    bool connected = bridge->bio_async_connected;
    nimcp_mutex_unlock(bridge->mutex);

    return connected;
}
