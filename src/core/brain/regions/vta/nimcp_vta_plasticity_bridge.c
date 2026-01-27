/**
 * @file nimcp_vta_plasticity_bridge.c
 * @brief Implementation of VTA-Plasticity Bidirectional Integration Bridge
 * @version 1.0.0
 * @date 2026-01-11
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/brain/regions/vta/nimcp_vta_plasticity_bridge.h"
#include "core/brain/regions/vta/nimcp_vta_adapter.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for vta_plasticity_bridge module */
static nimcp_health_agent_t* g_vta_plasticity_bridge_health_agent = NULL;

/**
 * @brief Set health agent for vta_plasticity_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void vta_plasticity_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_vta_plasticity_bridge_health_agent = agent;
}

/** @brief Send heartbeat from vta_plasticity_bridge module */
static inline void vta_plasticity_bridge_heartbeat(const char* operation, float progress) {
    if (g_vta_plasticity_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_vta_plasticity_bridge_health_agent, operation, progress);
    }
}

#define LOG_MODULE "VTA_PLASTICITY_BRIDGE"


struct nimcp_vta_plasticity_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    nimcp_vta_plasticity_config_t config;
    nimcp_vta_adapter_t vta_adapter;
    void* plasticity_coordinator;

    nimcp_vta_plasticity_synapse_t* synapses;
    uint32_t synapse_count;
    uint32_t synapse_capacity;

    nimcp_vta_plasticity_bridge_state_t state;
    nimcp_vta_plasticity_modulation_t current_modulation;

    nimcp_vta_weight_change_cb weight_callback;
    void* callback_user_data;

    nimcp_vta_plasticity_stats_t stats;
    uint64_t current_time_us;
};

static float clamp(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

static nimcp_vta_plasticity_synapse_t* find_synapse(nimcp_vta_plasticity_bridge_t* b, uint32_t id) {
    for (uint32_t i = 0; i < b->synapse_count; i++)
        if (b->synapses[i].synapse_id == id) return &b->synapses[i];
    return NULL;
}

nimcp_vta_plasticity_config_t nimcp_vta_plasticity_config_default(void) {
    return (nimcp_vta_plasticity_config_t){
        .enable_da_gating = true,
        .da_lr_multiplier_min = 0.5f,
        .da_lr_multiplier_max = 2.0f,
        .da_gating_threshold = 0.3f,
        .enable_rpe_learning = true,
        .rpe_learning_rate = 0.1f,
        .positive_rpe_boost = 1.5f,
        .negative_rpe_scale = 0.5f,
        .stdp_ltp_window_ms = 50.0f,
        .stdp_ltd_window_ms = 50.0f,
        .da_stdp_modulation = 0.5f,
        .enable_eligibility_traces = true,
        .eligibility_decay_tau = 500.0f,
        .da_trace_conversion = 0.5f,
        .trace_accumulation_rate = 0.1f,
        .enable_td_learning = true,
        .td_discount_factor = 0.99f,
        .td_learning_rate = 0.1f,
        .enable_motivation_mod = true,
        .high_motivation_boost = 1.5f,
        .effort_cost_penalty = 0.2f,
        .weight_min = 0.0f,
        .weight_max = 1.0f,
        .initial_weight = 0.5f,
        .enable_bio_async = false
    };
}

nimcp_vta_plasticity_bridge_t* nimcp_vta_plasticity_create(const nimcp_vta_plasticity_config_t* config) {
    nimcp_vta_plasticity_bridge_t* b = calloc(1, sizeof(*b));
    if (!b) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "b is NULL");

        return NULL;

    }
    b->config = config ? *config : nimcp_vta_plasticity_config_default();
    b->synapse_capacity = VTA_PLASTICITY_MAX_SYNAPSES;
    b->synapses = calloc(b->synapse_capacity, sizeof(nimcp_vta_plasticity_synapse_t));
    if (!b->synapses) { free(b); return NULL; }
    b->state.state = VTA_PLASTICITY_STATE_IDLE;
    b->current_modulation.lr_multiplier = 1.0f;
    return b;
}

void nimcp_vta_plasticity_destroy(nimcp_vta_plasticity_bridge_t* b) {
    if (!b) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "vta_plasticity");
    free(b->synapses);
    free(b);
}

int nimcp_vta_plasticity_reset(nimcp_vta_plasticity_bridge_t* b) {
    if (!b) return -1;
    for (uint32_t i = 0; i < b->synapse_count; i++) {
        b->synapses[i].weight = b->synapses[i].initial_weight;
        b->synapses[i].eligibility_trace = 0.0f;
    }
    memset(&b->state, 0, sizeof(b->state));
    b->state.state = VTA_PLASTICITY_STATE_IDLE;
    memset(&b->stats, 0, sizeof(b->stats));
    return 0;
}

int nimcp_vta_plasticity_connect_vta(nimcp_vta_plasticity_bridge_t* b, nimcp_vta_adapter_t a) {
    if (!b || !a) return -1;
    b->vta_adapter = a;
    return 0;
}

int nimcp_vta_plasticity_connect_coordinator(nimcp_vta_plasticity_bridge_t* b, struct nimcp_plasticity_coordinator* c) {
    if (!b || !c) return -1;
    b->plasticity_coordinator = c;
    return 0;
}

int nimcp_vta_plasticity_register_synapse(nimcp_vta_plasticity_bridge_t* b, uint32_t id, nimcp_vta_synapse_type_t t, float w) {
    if (!b || b->synapse_count >= b->synapse_capacity) return -1;
    nimcp_vta_plasticity_synapse_t* s = &b->synapses[b->synapse_count++];
    s->synapse_id = id;
    s->type = t;
    s->weight = s->initial_weight = w;
    b->state.registered_synapses = b->synapse_count;
    return 0;
}

int nimcp_vta_plasticity_unregister_synapse(nimcp_vta_plasticity_bridge_t* b, uint32_t id) {
    if (!b) return -1;
    for (uint32_t i = 0; i < b->synapse_count; i++) {
        if (b->synapses[i].synapse_id == id) {
            if (i < b->synapse_count - 1) b->synapses[i] = b->synapses[b->synapse_count - 1];
            b->synapse_count--;
            b->state.registered_synapses = b->synapse_count;
            return 0;
        }
    }
    return -1;
}

int nimcp_vta_plasticity_get_synapse(nimcp_vta_plasticity_bridge_t* b, uint32_t id, nimcp_vta_plasticity_synapse_t* out) {
    if (!b || !out) return -1;
    nimcp_vta_plasticity_synapse_t* s = find_synapse(b, id);
    if (!s) return -1;
    *out = *s;
    return 0;
}

int nimcp_vta_plasticity_pre_spike(nimcp_vta_plasticity_bridge_t* b, uint32_t id, uint64_t ts) {
    if (!b) return -1;
    nimcp_vta_plasticity_synapse_t* s = find_synapse(b, id);
    if (!s) return -1;
    s->last_pre_spike_us = ts;
    s->eligibility_trace = 1.0f;
    b->stats.total_pre_spikes++;
    return 0;
}

int nimcp_vta_plasticity_post_spike(nimcp_vta_plasticity_bridge_t* b, uint32_t id, uint64_t ts) {
    if (!b) return -1;
    nimcp_vta_plasticity_synapse_t* s = find_synapse(b, id);
    if (!s) return -1;
    s->last_post_spike_us = ts;
    b->stats.total_post_spikes++;
    return 0;
}

int nimcp_vta_plasticity_reward(nimcp_vta_plasticity_bridge_t* b, float r, uint64_t ts) {
    if (!b) return -1;
    b->state.da.reward_received = true;
    b->state.da.last_reward_us = ts;
    b->stats.reward_events++;
    b->stats.total_reward += r;
    return 0;
}

int nimcp_vta_plasticity_rpe(nimcp_vta_plasticity_bridge_t* b, float rpe, uint64_t ts) {
    if (!b) return -1;
    b->state.da.current_rpe = rpe;
    b->current_modulation.rpe_signal = rpe;
    return 0;
}

int nimcp_vta_plasticity_set_da_level(nimcp_vta_plasticity_bridge_t* b, float da, float m) {
    if (!b) return -1;
    b->state.da.da_level = clamp(da, 0.0f, 100.0f);
    b->state.da.motivation = clamp(m, 0.0f, 1.0f);
    float n = da / 100.0f;
    b->current_modulation.lr_multiplier = b->config.da_lr_multiplier_min +
        (b->config.da_lr_multiplier_max - b->config.da_lr_multiplier_min) * n;
    b->state.da.lr_multiplier = b->current_modulation.lr_multiplier;
    return 0;
}

int nimcp_vta_plasticity_update(nimcp_vta_plasticity_bridge_t* b, float dt_ms) {
    if (!b) return -1;
    b->stats.total_updates++;
    float decay = expf(-dt_ms / b->config.eligibility_decay_tau);
    for (uint32_t i = 0; i < b->synapse_count; i++) {
        b->synapses[i].eligibility_trace *= decay;
    }
    return 0;
}

int nimcp_vta_plasticity_td_update(nimcp_vta_plasticity_bridge_t* b, float cur, float next, float r) {
    if (!b) return -1;
    float td_error = r + b->config.td_discount_factor * next - cur;
    b->state.da.current_rpe = td_error;
    b->stats.td_updates++;
    return 0;
}

int nimcp_vta_plasticity_convert_traces(nimcp_vta_plasticity_bridge_t* b, float da) {
    if (!b) return -1;
    for (uint32_t i = 0; i < b->synapse_count; i++) {
        if (b->synapses[i].eligibility_trace > 0.1f) {
            float dw = b->synapses[i].eligibility_trace * da * b->config.da_trace_conversion;
            b->synapses[i].weight = clamp(b->synapses[i].weight + dw * 0.01f,
                b->config.weight_min, b->config.weight_max);
        }
    }
    return 0;
}

int nimcp_vta_plasticity_get_modulation(nimcp_vta_plasticity_bridge_t* b, nimcp_vta_plasticity_modulation_t* m) {
    if (!b || !m) return -1;
    *m = b->current_modulation;
    return 0;
}

float nimcp_vta_plasticity_get_learning_progress(nimcp_vta_plasticity_bridge_t* b) {
    if (!b || b->synapse_count == 0) return 0.0f;
    float total = 0.0f;
    for (uint32_t i = 0; i < b->synapse_count; i++)
        total += fabsf(b->synapses[i].weight - b->synapses[i].initial_weight);
    return total / b->synapse_count;
}

float nimcp_vta_plasticity_get_prediction_accuracy(nimcp_vta_plasticity_bridge_t* b) {
    if (!b) return 0.0f;
    return 1.0f - clamp(fabsf(b->stats.avg_rpe), 0.0f, 1.0f);
}

float nimcp_vta_plasticity_get_avg_value(nimcp_vta_plasticity_bridge_t* b) {
    if (!b) return 0.0f;
    return b->state.avg_value_estimate;
}

int nimcp_vta_plasticity_get_state(const nimcp_vta_plasticity_bridge_t* b, nimcp_vta_plasticity_bridge_state_t* s) {
    if (!b || !s) return -1;
    *s = b->state;
    return 0;
}

int nimcp_vta_plasticity_get_stats(const nimcp_vta_plasticity_bridge_t* b, nimcp_vta_plasticity_stats_t* s) {
    if (!b || !s) return -1;
    *s = b->stats;
    return 0;
}

void nimcp_vta_plasticity_reset_stats(nimcp_vta_plasticity_bridge_t* b) {
    if (b) memset(&b->stats, 0, sizeof(b->stats));
}

int nimcp_vta_plasticity_set_weight_callback(nimcp_vta_plasticity_bridge_t* b, nimcp_vta_weight_change_cb cb, void* ud) {
    if (!b) return -1;
    b->weight_callback = cb;
    b->callback_user_data = ud;
    return 0;
}

int nimcp_vta_plasticity_connect_bio_async(nimcp_vta_plasticity_bridge_t* b) {
    if (!b) return -1;
    b->state.bio_async_connected = true;
    return 0;
}

int nimcp_vta_plasticity_disconnect_bio_async(nimcp_vta_plasticity_bridge_t* b) {
    if (!b) return -1;
    b->state.bio_async_connected = false;
    return 0;
}

bool nimcp_vta_plasticity_is_bio_async_connected(const nimcp_vta_plasticity_bridge_t* b) {
    return b ? b->state.bio_async_connected : false;
}
