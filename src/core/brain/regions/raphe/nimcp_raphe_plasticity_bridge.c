/**
 * @file nimcp_raphe_plasticity_bridge.c
 * @brief Implementation of Raphe-Plasticity Bidirectional Integration Bridge
 * @version 1.0.0
 * @date 2026-01-11
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/brain/regions/raphe/nimcp_raphe_plasticity_bridge.h"
#include "core/brain/regions/raphe/nimcp_raphe_adapter.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(raphe_plasticity_bridge)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_raphe_plasticity_bridge_mesh_id = 0;
static mesh_participant_registry_t* g_raphe_plasticity_bridge_mesh_registry = NULL;

nimcp_error_t raphe_plasticity_bridge_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_raphe_plasticity_bridge_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "raphe_plasticity_bridge", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "raphe_plasticity_bridge";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_raphe_plasticity_bridge_mesh_id);
    if (err == NIMCP_SUCCESS) g_raphe_plasticity_bridge_mesh_registry = registry;
    return err;
}

void raphe_plasticity_bridge_mesh_unregister(void) {
    if (g_raphe_plasticity_bridge_mesh_registry && g_raphe_plasticity_bridge_mesh_id != 0) {
        mesh_participant_unregister(g_raphe_plasticity_bridge_mesh_registry, g_raphe_plasticity_bridge_mesh_id);
        g_raphe_plasticity_bridge_mesh_id = 0;
        g_raphe_plasticity_bridge_mesh_registry = NULL;
    }
}


#define LOG_MODULE "RAPHE_PLASTICITY_BRIDGE"


struct nimcp_raphe_plasticity_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    nimcp_raphe_plasticity_config_t config;
    nimcp_raphe_adapter_t raphe_adapter;
    void* plasticity_coordinator;
    nimcp_raphe_plasticity_synapse_t* synapses;
    uint32_t synapse_count;
    uint32_t synapse_capacity;
    nimcp_raphe_plasticity_bridge_state_t state;
    nimcp_raphe_plasticity_modulation_t current_modulation;
    nimcp_raphe_weight_change_cb weight_callback;
    void* callback_user_data;
    nimcp_raphe_plasticity_stats_t stats;
};

static float clamp(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }
static nimcp_raphe_plasticity_synapse_t* find_synapse(nimcp_raphe_plasticity_bridge_t* b, uint32_t id) {
    for (uint32_t i = 0; i < b->synapse_count; i++) if (b->synapses[i].synapse_id == id) return &b->synapses[i];
    return NULL;
}

nimcp_raphe_plasticity_config_t nimcp_raphe_plasticity_config_default(void) {
    return (nimcp_raphe_plasticity_config_t){
        .enable_ht_gating = true, .ht_lr_multiplier_min = 0.5f, .ht_lr_multiplier_max = 2.0f, .ht_gating_threshold = 0.3f,
        .enable_mood_modulation = true, .positive_mood_ltp_boost = 1.5f, .negative_mood_ltd_boost = 1.5f, .mood_bias_strength = 0.5f,
        .stdp_ltp_window_ms = RAPHE_PLASTICITY_STDP_WINDOW, .stdp_ltd_window_ms = RAPHE_PLASTICITY_STDP_WINDOW, .ht_stdp_modulation = 0.5f,
        .enable_inhibition_learning = true, .inhibition_learning_rate = 0.1f, .successful_inhibition_boost = 1.5f,
        .enable_extinction = true, .extinction_rate = 0.05f, .safety_learning_rate = 0.1f, .extinction_ht_requirement = 0.5f,
        .enable_patience_learning = true, .patience_learning_rate = 0.1f, .delayed_reward_boost = 2.0f,
        .weight_min = 0.0f, .weight_max = 1.0f, .initial_weight = 0.5f, .enable_bio_async = false
    };
}

nimcp_raphe_plasticity_bridge_t* nimcp_raphe_plasticity_create(const nimcp_raphe_plasticity_config_t* config) {
    nimcp_raphe_plasticity_bridge_t* b = nimcp_calloc(1, sizeof(*b));
    if (!b) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "b is NULL");

        return NULL;

    }
    b->config = config ? *config : nimcp_raphe_plasticity_config_default();
    b->synapse_capacity = RAPHE_PLASTICITY_MAX_SYNAPSES;
    b->synapses = nimcp_calloc(b->synapse_capacity, sizeof(nimcp_raphe_plasticity_synapse_t));
    if (!b->synapses) { nimcp_free(b); return NULL; }
    b->state.state = RAPHE_PLASTICITY_STATE_IDLE;
    b->current_modulation.lr_multiplier = 1.0f;
    return b;
}

void nimcp_raphe_plasticity_destroy(nimcp_raphe_plasticity_bridge_t* b) {
    if (!b) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "raphe_plasticity");
    nimcp_free(b->synapses);
    nimcp_free(b);
}

int nimcp_raphe_plasticity_reset(nimcp_raphe_plasticity_bridge_t* b) {
    if (!b) return -1;
    for (uint32_t i = 0; i < b->synapse_count; i++) b->synapses[i].weight = b->synapses[i].initial_weight;
    memset(&b->state, 0, sizeof(b->state));
    b->state.state = RAPHE_PLASTICITY_STATE_IDLE;
    memset(&b->stats, 0, sizeof(b->stats));
    return 0;
}

int nimcp_raphe_plasticity_connect_raphe(nimcp_raphe_plasticity_bridge_t* b, nimcp_raphe_adapter_t a) { if (!b||!a) return -1; b->raphe_adapter = a; return 0; }
int nimcp_raphe_plasticity_connect_coordinator(nimcp_raphe_plasticity_bridge_t* b, struct nimcp_plasticity_coordinator* c) { if (!b||!c) return -1; b->plasticity_coordinator = c; return 0; }

int nimcp_raphe_plasticity_register_synapse(nimcp_raphe_plasticity_bridge_t* b, uint32_t id, nimcp_raphe_synapse_type_t t, float w) {
    if (!b || b->synapse_count >= b->synapse_capacity) return -1;
    nimcp_raphe_plasticity_synapse_t* s = &b->synapses[b->synapse_count++];
    s->synapse_id = id; s->type = t; s->weight = s->initial_weight = w;
    b->state.registered_synapses = b->synapse_count;
    return 0;
}

int nimcp_raphe_plasticity_unregister_synapse(nimcp_raphe_plasticity_bridge_t* b, uint32_t id) {
    if (!b) return -1;
    for (uint32_t i = 0; i < b->synapse_count; i++) {
        if (b->synapses[i].synapse_id == id) {
            if (i < b->synapse_count - 1) b->synapses[i] = b->synapses[b->synapse_count - 1];
            b->synapse_count--;
            return 0;
        }
    }
    return -1;
}

int nimcp_raphe_plasticity_get_synapse(nimcp_raphe_plasticity_bridge_t* b, uint32_t id, nimcp_raphe_plasticity_synapse_t* o) {
    if (!b || !o) return -1;
    nimcp_raphe_plasticity_synapse_t* s = find_synapse(b, id);
    if (!s) return -1;
    *o = *s;
    return 0;
}

int nimcp_raphe_plasticity_pre_spike(nimcp_raphe_plasticity_bridge_t* b, uint32_t id, uint64_t ts) {
    if (!b) return -1;
    nimcp_raphe_plasticity_synapse_t* s = find_synapse(b, id);
    if (!s) return -1;
    s->last_pre_spike_us = ts;
    b->stats.total_pre_spikes++;
    return 0;
}

int nimcp_raphe_plasticity_post_spike(nimcp_raphe_plasticity_bridge_t* b, uint32_t id, uint64_t ts) {
    if (!b) return -1;
    nimcp_raphe_plasticity_synapse_t* s = find_synapse(b, id);
    if (!s) return -1;
    s->last_post_spike_us = ts;
    b->stats.total_post_spikes++;
    return 0;
}

int nimcp_raphe_plasticity_inhibition_success(nimcp_raphe_plasticity_bridge_t* b, uint64_t ts) {
    if (!b) return -1;
    b->stats.inhibition_successes++;
    return 0;
}

int nimcp_raphe_plasticity_inhibition_failure(nimcp_raphe_plasticity_bridge_t* b, uint64_t ts) {
    if (!b) return -1;
    b->stats.inhibition_failures++;
    return 0;
}

int nimcp_raphe_plasticity_extinction_trial(nimcp_raphe_plasticity_bridge_t* b, uint64_t ts) {
    if (!b) return -1;
    b->stats.extinction_trials++;
    return 0;
}

int nimcp_raphe_plasticity_set_ht_state(nimcp_raphe_plasticity_bridge_t* b, float ht, float mood) {
    if (!b) return -1;
    b->state.ht.serotonin_level = clamp(ht, 0.0f, 100.0f);
    b->state.ht.mood_valence = clamp(mood, -1.0f, 1.0f);
    float n = ht / 100.0f;
    b->current_modulation.lr_multiplier = b->config.ht_lr_multiplier_min +
        (b->config.ht_lr_multiplier_max - b->config.ht_lr_multiplier_min) * n;
    b->current_modulation.mood_bias = mood * b->config.mood_bias_strength;
    return 0;
}

int nimcp_raphe_plasticity_update(nimcp_raphe_plasticity_bridge_t* b, float dt_ms) {
    if (!b) return -1;
    b->stats.total_updates++;
    return 0;
}

int nimcp_raphe_plasticity_start_patience(nimcp_raphe_plasticity_bridge_t* b, uint64_t ts) {
    if (!b) return -1;
    b->state.ht.patience_active = true;
    b->state.ht.patience_start_us = ts;
    return 0;
}

int nimcp_raphe_plasticity_complete_patience(nimcp_raphe_plasticity_bridge_t* b, float r, uint64_t ts) {
    if (!b) return -1;
    b->state.ht.patience_active = false;
    return 0;
}

int nimcp_raphe_plasticity_get_modulation(nimcp_raphe_plasticity_bridge_t* b, nimcp_raphe_plasticity_modulation_t* m) {
    if (!b || !m) return -1;
    *m = b->current_modulation;
    return 0;
}

float nimcp_raphe_plasticity_get_inhibition_strength(nimcp_raphe_plasticity_bridge_t* b) {
    return b ? b->state.avg_inhibition_strength : 0.0f;
}

float nimcp_raphe_plasticity_get_extinction_progress(nimcp_raphe_plasticity_bridge_t* b) {
    if (!b || b->synapse_count == 0) return 0.0f;
    float total = 0.0f;
    for (uint32_t i = 0; i < b->synapse_count; i++) total += b->synapses[i].extinction_level;
    return total / b->synapse_count;
}

float nimcp_raphe_plasticity_get_mood_feedback(nimcp_raphe_plasticity_bridge_t* b) {
    return b ? b->state.ht.mood_valence : 0.0f;
}

int nimcp_raphe_plasticity_get_state(const nimcp_raphe_plasticity_bridge_t* b, nimcp_raphe_plasticity_bridge_state_t* s) { if (!b||!s) return -1; *s = b->state; return 0; }
int nimcp_raphe_plasticity_get_stats(const nimcp_raphe_plasticity_bridge_t* b, nimcp_raphe_plasticity_stats_t* s) { if (!b||!s) return -1; *s = b->stats; return 0; }
void nimcp_raphe_plasticity_reset_stats(nimcp_raphe_plasticity_bridge_t* b) { if (b) memset(&b->stats, 0, sizeof(b->stats)); }
int nimcp_raphe_plasticity_set_weight_callback(nimcp_raphe_plasticity_bridge_t* b, nimcp_raphe_weight_change_cb cb, void* ud) { if (!b) return -1; b->weight_callback = cb; b->callback_user_data = ud; return 0; }
int nimcp_raphe_plasticity_connect_bio_async(nimcp_raphe_plasticity_bridge_t* b) { if (!b) return -1; b->state.bio_async_connected = true; return 0; }
int nimcp_raphe_plasticity_disconnect_bio_async(nimcp_raphe_plasticity_bridge_t* b) { if (!b) return -1; b->state.bio_async_connected = false; return 0; }
bool nimcp_raphe_plasticity_is_bio_async_connected(const nimcp_raphe_plasticity_bridge_t* b) { return b ? b->state.bio_async_connected : false; }
