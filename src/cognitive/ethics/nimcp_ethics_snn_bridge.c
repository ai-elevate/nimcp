/**
 * @file nimcp_ethics_snn_bridge.c
 * @brief Ethics Engine - SNN Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/ethics/nimcp_ethics_snn_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "snn/nimcp_snn_network.h"
#include "snn/nimcp_snn_config.h"
#include "core/neuron_types/nimcp_neuron_types.h"
#include "core/synapse_types/nimcp_synapse_types.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>
#include <stdio.h>

//=============================================================================
// Internal Structures
//=============================================================================

struct ethics_snn_bridge {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */
    ethics_snn_config_t config;
    snn_network_t* snn;

    /* State */
    ethics_snn_state_t state;
    uint64_t current_time_us;
    bool bio_async_connected;

    /* Dimension state */
    ethics_dim_state_t dim_states[ETHICS_SNN_MAX_DIMENSIONS];

    /* Buffers */
    float* encoding_buffer;
    float* output_buffer;
    float* judgment_buffer;

    /* Judgment state */
    ethics_judgment_t last_judgment;
    float harm_signal;
    float conflict_signal;

    /* Callbacks */
    ethics_snn_harm_callback_t harm_callback;
    void* harm_callback_data;
    ethics_snn_judgment_callback_t judgment_callback;
    void* judgment_callback_data;
    ethics_snn_conflict_callback_t conflict_callback;
    void* conflict_callback_data;

    /* Statistics */
    ethics_snn_stats_t stats;
};

//=============================================================================
// Helper Functions
//=============================================================================

static inline float clamp_f(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

static void softmax(float* values, uint32_t n) {
    if (n == 0) return;

    float max_val = values[0];
    for (uint32_t i = 1; i < n; i++) {
        if (values[i] > max_val) max_val = values[i];
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        values[i] = expf(values[i] - max_val);
        sum += values[i];
    }

    if (sum > 0.0f) {
        for (uint32_t i = 0; i < n; i++) {
            values[i] /= sum;
        }
    }
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

ethics_snn_config_t ethics_snn_config_default(void) {
    ethics_snn_config_t config = {
        .num_dimensions = ETHICS_DIM_COUNT,
        .neurons_per_dim = ETHICS_SNN_NEURONS_PER_DIM,
        .hidden_dim = 128,

        .dt_ms = 1.0f,
        .encoding_window_ms = ETHICS_SNN_ENCODING_WINDOW,
        .integration_tau_ms = 100.0f,

        .encoding = ETHICS_SNN_ENCODE_POPULATION,
        .encoding_gain = 1.0f,
        .baseline_rate_hz = 5.0f,
        .max_rate_hz = 100.0f,

        .decoding = ETHICS_SNN_DECODE_INTEGRATION,
        .decision_threshold = 0.5f,
        .harm_threshold = ETHICS_SNN_HARM_THRESHOLD,
        .confidence_threshold = 0.6f,

        .enable_competition = true,
        .inhibition_strength = 0.3f,
        .enable_conflict_detection = true,
        .conflict_threshold = 0.4f,

        .enable_asimov_populations = true,
        .first_law_priority = 2.0f,
        .zeroth_law_priority = 3.0f,

        .enable_bio_async = false,
        .enable_plasticity_integration = true
    };
    return config;
}

ethics_snn_bridge_t* ethics_snn_create(const ethics_snn_config_t* config) {
    ethics_snn_bridge_t* bridge = nimcp_calloc(1, sizeof(ethics_snn_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = ethics_snn_config_default();
    }

    /* Validate config */
    if (bridge->config.num_dimensions == 0 ||
        bridge->config.num_dimensions > ETHICS_SNN_MAX_DIMENSIONS) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize base bridge */
    if (bridge_base_init(&bridge->base, 0, "ethics_snn") != 0) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Create SNN network */
    snn_config_t snn_config;
    uint32_t input_dim = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
    uint32_t hidden_dim = bridge->config.hidden_dim;
    uint32_t output_dim = 4; /* allow, block, modify, confidence */

    snn_config_feedforward(&snn_config, input_dim, hidden_dim, output_dim);
    snn_config.dt = bridge->config.dt_ms;
    snn_config.enable_bio_async = bridge->config.enable_bio_async;

    bridge->snn = snn_network_create(&snn_config);
    if (!bridge->snn) {
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate buffers */
    bridge->encoding_buffer = nimcp_calloc(input_dim, sizeof(float));
    bridge->output_buffer = nimcp_calloc(output_dim, sizeof(float));
    bridge->judgment_buffer = nimcp_calloc(bridge->config.num_dimensions, sizeof(float));

    if (!bridge->encoding_buffer || !bridge->output_buffer || !bridge->judgment_buffer) {
        ethics_snn_destroy(bridge);
        return NULL;
    }

    /* Initialize dimension states */
    for (uint32_t i = 0; i < bridge->config.num_dimensions; i++) {
        bridge->dim_states[i].activation = 0.0f;
        bridge->dim_states[i].accumulated_evidence = 0.0f;
        bridge->dim_states[i].spike_count = 0;
        bridge->dim_states[i].mean_rate_hz = bridge->config.baseline_rate_hz;
        bridge->dim_states[i].last_spike_time_us = 0;
    }

    /* Initialize judgment to neutral */
    bridge->last_judgment.allow_score = 0.5f;
    bridge->last_judgment.block_score = 0.5f;
    bridge->last_judgment.modify_score = 0.0f;
    bridge->last_judgment.confidence = 0.0f;
    bridge->last_judgment.harm_detected = false;
    bridge->last_judgment.conflict_detected = false;
    bridge->last_judgment.golden_rule_activation = 0.5f;
    bridge->last_judgment.first_law_activation = 1.0f;

    bridge->state = ETHICS_SNN_STATE_IDLE;
    bridge->current_time_us = nimcp_time_get_us();
    bridge->bio_async_connected = false;

    return bridge;
}

void ethics_snn_destroy(ethics_snn_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->snn) snn_network_destroy(bridge->snn);
    if (bridge->encoding_buffer) nimcp_free(bridge->encoding_buffer);
    if (bridge->output_buffer) nimcp_free(bridge->output_buffer);
    if (bridge->judgment_buffer) nimcp_free(bridge->judgment_buffer);
    if (bridge->base.mutex) bridge_base_cleanup(&bridge->base);

    nimcp_free(bridge);
}

int ethics_snn_reset(ethics_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Reset SNN */
    if (bridge->snn) {
        snn_network_reset(bridge->snn);
    }

    /* Reset buffers */
    uint32_t input_dim = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
    memset(bridge->encoding_buffer, 0, input_dim * sizeof(float));
    memset(bridge->output_buffer, 0, 4 * sizeof(float));
    memset(bridge->judgment_buffer, 0, bridge->config.num_dimensions * sizeof(float));

    /* Reset dimension states */
    for (uint32_t i = 0; i < bridge->config.num_dimensions; i++) {
        bridge->dim_states[i].activation = 0.0f;
        bridge->dim_states[i].accumulated_evidence = 0.0f;
        bridge->dim_states[i].spike_count = 0;
        bridge->dim_states[i].mean_rate_hz = bridge->config.baseline_rate_hz;
    }

    /* Reset judgment to neutral */
    bridge->last_judgment.allow_score = 0.5f;
    bridge->last_judgment.block_score = 0.5f;
    bridge->last_judgment.modify_score = 0.0f;
    bridge->last_judgment.confidence = 0.0f;
    bridge->last_judgment.harm_detected = false;
    bridge->last_judgment.conflict_detected = false;
    bridge->last_judgment.golden_rule_activation = 0.5f;
    bridge->last_judgment.first_law_activation = 1.0f;

    bridge->harm_signal = 0.0f;
    bridge->conflict_signal = 0.0f;
    bridge->state = ETHICS_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

//=============================================================================
// Encoding Functions
//=============================================================================

int ethics_snn_encode_context(
    ethics_snn_bridge_t* bridge,
    const float* dimensions,
    uint32_t num_dims)
{
    if (!bridge || !dimensions) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->state = ETHICS_SNN_STATE_ENCODING;

    uint32_t n = (num_dims < bridge->config.num_dimensions) ?
                  num_dims : bridge->config.num_dimensions;

    /* Encode each dimension as population activity */
    int total_spikes = 0;
    for (uint32_t d = 0; d < n; d++) {
        float value = clamp_f(dimensions[d], 0.0f, 1.0f);
        bridge->dim_states[d].activation = value;

        /* Calculate firing rate */
        float rate = bridge->config.baseline_rate_hz +
                    value * (bridge->config.max_rate_hz - bridge->config.baseline_rate_hz);

        /* Apply priority weighting for Asimov dimensions */
        if (bridge->config.enable_asimov_populations) {
            if (d == ETHICS_DIM_ASIMOV_FIRST) {
                rate *= bridge->config.first_law_priority;
            } else if (d == ETHICS_DIM_ASIMOV_ZEROTH) {
                rate *= bridge->config.zeroth_law_priority;
            }
        }

        /* Encode into buffer */
        uint32_t base_idx = d * bridge->config.neurons_per_dim;
        for (uint32_t n_idx = 0; n_idx < bridge->config.neurons_per_dim; n_idx++) {
            float neuron_rate = rate * bridge->config.encoding_gain;
            bridge->encoding_buffer[base_idx + n_idx] = neuron_rate;

            /* Count expected spikes */
            if (neuron_rate > bridge->config.baseline_rate_hz) {
                total_spikes++;
            }
        }

        bridge->dim_states[d].spike_count += total_spikes;
    }

    /* Set network inputs */
    uint32_t input_size = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
    snn_network_set_inputs(bridge->snn, bridge->encoding_buffer, input_size);

    bridge->stats.total_evaluations++;
    bridge->state = ETHICS_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);

    return total_spikes;
}

int ethics_snn_encode_harm(
    ethics_snn_bridge_t* bridge,
    float harm_level,
    float urgency)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->state = ETHICS_SNN_STATE_ENCODING;

    harm_level = clamp_f(harm_level, 0.0f, 1.0f);
    urgency = clamp_f(urgency, 0.0f, 1.0f);

    /* Encode harm with urgency boost */
    float effective_harm = harm_level * (1.0f + urgency);
    effective_harm = clamp_f(effective_harm, 0.0f, 1.0f);

    /* Update harm dimension */
    bridge->dim_states[ETHICS_DIM_HARM].activation = effective_harm;

    /* Check for rapid harm detection */
    if (effective_harm > bridge->config.harm_threshold) {
        bridge->harm_signal = effective_harm;
        bridge->last_judgment.harm_detected = true;
        bridge->stats.harm_detections++;

        /* Fire harm callback */
        if (bridge->harm_callback) {
            uint64_t latency = nimcp_time_get_us() - bridge->current_time_us;
            nimcp_mutex_unlock(bridge->base.mutex);
            bridge->harm_callback(bridge, effective_harm, latency,
                                 bridge->harm_callback_data);
            nimcp_mutex_lock(bridge->base.mutex);
        }
    }

    /* Encode into harm neurons with high priority */
    uint32_t base_idx = ETHICS_DIM_HARM * bridge->config.neurons_per_dim;
    float rate = bridge->config.max_rate_hz * effective_harm * 2.0f; /* High priority */
    int spike_count = 0;

    for (uint32_t n = 0; n < bridge->config.neurons_per_dim; n++) {
        bridge->encoding_buffer[base_idx + n] = rate;
        if (rate > bridge->config.baseline_rate_hz) spike_count++;
    }

    bridge->state = ETHICS_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);

    return spike_count;
}

int ethics_snn_encode_golden_rule(
    ethics_snn_bridge_t* bridge,
    float self_impact,
    float other_impact,
    float empathy_level)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->state = ETHICS_SNN_STATE_ENCODING;

    self_impact = clamp_f(self_impact, 0.0f, 1.0f);
    other_impact = clamp_f(other_impact, 0.0f, 1.0f);
    empathy_level = clamp_f(empathy_level, 0.0f, 1.0f);

    /* Golden Rule: would I want this done to me? */
    float golden_rule_score = fabsf(self_impact - other_impact);
    golden_rule_score = 1.0f - golden_rule_score; /* Higher = more aligned */
    golden_rule_score *= empathy_level; /* Modulated by empathy */

    bridge->dim_states[ETHICS_DIM_GOLDEN_RULE].activation = golden_rule_score;
    bridge->dim_states[ETHICS_DIM_EMPATHY].activation = empathy_level;
    bridge->last_judgment.golden_rule_activation = golden_rule_score;

    /* Encode Golden Rule dimension */
    uint32_t gr_base = ETHICS_DIM_GOLDEN_RULE * bridge->config.neurons_per_dim;
    uint32_t emp_base = ETHICS_DIM_EMPATHY * bridge->config.neurons_per_dim;
    int spike_count = 0;

    for (uint32_t n = 0; n < bridge->config.neurons_per_dim; n++) {
        float gr_rate = bridge->config.baseline_rate_hz +
                       golden_rule_score * (bridge->config.max_rate_hz -
                                           bridge->config.baseline_rate_hz);
        float emp_rate = bridge->config.baseline_rate_hz +
                        empathy_level * (bridge->config.max_rate_hz -
                                        bridge->config.baseline_rate_hz);

        bridge->encoding_buffer[gr_base + n] = gr_rate;
        bridge->encoding_buffer[emp_base + n] = emp_rate;

        if (gr_rate > bridge->config.baseline_rate_hz) spike_count++;
        if (emp_rate > bridge->config.baseline_rate_hz) spike_count++;
    }

    bridge->state = ETHICS_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);

    return spike_count;
}

//=============================================================================
// Simulation Functions
//=============================================================================

int ethics_snn_simulate(ethics_snn_bridge_t* bridge, float duration_ms) {
    if (!bridge || duration_ms <= 0) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->state = ETHICS_SNN_STATE_SIMULATING;

    int ret = snn_network_run(bridge->snn, duration_ms);
    if (ret >= 0) {
        bridge->current_time_us += (uint64_t)(duration_ms * 1000);
        bridge->stats.total_simulations++;
        bridge->stats.total_spikes += (uint64_t)ret;
    }

    bridge->state = ETHICS_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);

    return ret >= 0 ? 0 : -1;
}

int ethics_snn_step(ethics_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->state = ETHICS_SNN_STATE_SIMULATING;

    int ret = snn_network_step(bridge->snn, bridge->config.dt_ms);
    if (ret >= 0) {
        bridge->current_time_us += (uint64_t)(bridge->config.dt_ms * 1000);
        bridge->stats.total_simulations++;
        bridge->stats.total_spikes += (uint64_t)ret;
    }

    bridge->state = ETHICS_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);

    return ret >= 0 ? 0 : -1;
}

int ethics_snn_forward(
    ethics_snn_bridge_t* bridge,
    const float* inputs,
    uint32_t input_count)
{
    if (!bridge || !inputs) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->state = ETHICS_SNN_STATE_PROCESSING;

    snn_network_set_inputs(bridge->snn, inputs, input_count);
    int spikes = snn_network_run(bridge->snn, bridge->config.encoding_window_ms);

    if (spikes >= 0) {
        bridge->stats.total_spikes += (uint64_t)spikes;
        bridge->stats.total_evaluations++;
    }

    bridge->state = ETHICS_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);

    return spikes;
}

//=============================================================================
// Decoding Functions
//=============================================================================

int ethics_snn_get_judgment(
    ethics_snn_bridge_t* bridge,
    ethics_judgment_t* judgment)
{
    if (!bridge || !judgment) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->state = ETHICS_SNN_STATE_DECODING;

    /* Get network outputs */
    snn_network_get_outputs(bridge->snn, bridge->output_buffer, 4);

    /* Decode judgment scores */
    float outputs[4];
    memcpy(outputs, bridge->output_buffer, 4 * sizeof(float));

    /* Normalize to [0,1] range */
    float max_out = bridge->config.max_rate_hz;
    float min_out = bridge->config.baseline_rate_hz;
    for (int i = 0; i < 4; i++) {
        outputs[i] = (outputs[i] - min_out) / (max_out - min_out);
        outputs[i] = clamp_f(outputs[i], 0.0f, 1.0f);
    }

    /* Apply decoding method */
    switch (bridge->config.decoding) {
        case ETHICS_SNN_DECODE_SOFTMAX:
            softmax(outputs, 3); /* Softmax over allow/block/modify */
            break;

        case ETHICS_SNN_DECODE_COMPETITION:
            /* Winner take all */
            {
                int max_idx = 0;
                for (int i = 1; i < 3; i++) {
                    if (outputs[i] > outputs[max_idx]) max_idx = i;
                }
                for (int i = 0; i < 3; i++) {
                    outputs[i] = (i == max_idx) ? 1.0f : 0.0f;
                }
            }
            break;

        case ETHICS_SNN_DECODE_INTEGRATION:
        case ETHICS_SNN_DECODE_THRESHOLD:
        default:
            /* Keep normalized values */
            break;
    }

    /* Update judgment */
    bridge->last_judgment.allow_score = outputs[0];
    bridge->last_judgment.block_score = outputs[1];
    bridge->last_judgment.modify_score = outputs[2];
    bridge->last_judgment.confidence = outputs[3];

    /* Check First Law activation */
    bridge->last_judgment.first_law_activation =
        bridge->dim_states[ETHICS_DIM_ASIMOV_FIRST].activation;

    /* High harm should always block */
    if (bridge->harm_signal > bridge->config.harm_threshold) {
        bridge->last_judgment.block_score = fmaxf(
            bridge->last_judgment.block_score,
            bridge->harm_signal
        );
        bridge->last_judgment.harm_detected = true;
    }

    /* Check for moral conflict */
    if (bridge->config.enable_conflict_detection) {
        float allow_block_diff = fabsf(outputs[0] - outputs[1]);
        if (allow_block_diff < bridge->config.conflict_threshold &&
            outputs[0] > 0.3f && outputs[1] > 0.3f) {
            bridge->conflict_signal = 1.0f - allow_block_diff;
            bridge->last_judgment.conflict_detected = true;
            bridge->stats.conflict_detections++;

            if (bridge->conflict_callback) {
                nimcp_mutex_unlock(bridge->base.mutex);
                bridge->conflict_callback(bridge, bridge->conflict_signal,
                                         0, 1, bridge->conflict_callback_data);
                nimcp_mutex_lock(bridge->base.mutex);
            }
        }
    }

    *judgment = bridge->last_judgment;

    /* Fire judgment callback if registered */
    if (bridge->judgment_callback) {
        nimcp_mutex_unlock(bridge->base.mutex);
        bridge->judgment_callback(bridge, &bridge->last_judgment,
                                 bridge->judgment_callback_data);
        nimcp_mutex_lock(bridge->base.mutex);
    }

    bridge->state = ETHICS_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int ethics_snn_get_activations(
    ethics_snn_bridge_t* bridge,
    float* activations,
    uint32_t num_dims)
{
    if (!bridge || !activations) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    uint32_t n = (num_dims < bridge->config.num_dimensions) ?
                  num_dims : bridge->config.num_dimensions;

    for (uint32_t i = 0; i < n; i++) {
        activations[i] = bridge->dim_states[i].activation;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

bool ethics_snn_check_harm(
    ethics_snn_bridge_t* bridge,
    float* harm_level)
{
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->base.mutex);

    bool detected = bridge->harm_signal > bridge->config.harm_threshold;
    if (harm_level) {
        *harm_level = bridge->harm_signal;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return detected;
}

bool ethics_snn_check_conflict(
    ethics_snn_bridge_t* bridge,
    float* conflict_level)
{
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->base.mutex);

    bool detected = bridge->conflict_signal > bridge->config.conflict_threshold;
    if (conflict_level) {
        *conflict_level = bridge->conflict_signal;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return detected;
}

//=============================================================================
// State Query Functions
//=============================================================================

int ethics_snn_get_dim_state(
    ethics_snn_bridge_t* bridge,
    uint32_t dim,
    ethics_dim_state_t* state)
{
    if (!bridge || !state || dim >= bridge->config.num_dimensions) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->dim_states[dim];
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int ethics_snn_get_state(
    ethics_snn_bridge_t* bridge,
    ethics_snn_bridge_state_t* state)
{
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    state->state = bridge->state;
    state->harm_signal = bridge->harm_signal;
    state->conflict_signal = bridge->conflict_signal;
    state->mean_confidence = bridge->last_judgment.confidence;

    /* Count active dimensions */
    state->active_dimensions = 0;
    state->total_activity = 0.0f;
    for (uint32_t i = 0; i < bridge->config.num_dimensions; i++) {
        if (bridge->dim_states[i].activation > 0.1f) {
            state->active_dimensions++;
        }
        state->total_activity += bridge->dim_states[i].activation;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int ethics_snn_get_stats(ethics_snn_bridge_t* bridge, ethics_snn_stats_t* stats) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int ethics_snn_reset_stats(ethics_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(ethics_snn_stats_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float ethics_snn_get_confidence(ethics_snn_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(bridge->base.mutex);
    float conf = bridge->last_judgment.confidence;
    nimcp_mutex_unlock(bridge->base.mutex);

    return conf;
}

float ethics_snn_get_total_activity(ethics_snn_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(bridge->base.mutex);

    float total = 0.0f;
    for (uint32_t i = 0; i < bridge->config.num_dimensions; i++) {
        total += bridge->dim_states[i].activation;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return total;
}

//=============================================================================
// Callback Registration
//=============================================================================

int ethics_snn_register_harm_callback(
    ethics_snn_bridge_t* bridge,
    ethics_snn_harm_callback_t callback,
    void* user_data)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->harm_callback = callback;
    bridge->harm_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int ethics_snn_register_judgment_callback(
    ethics_snn_bridge_t* bridge,
    ethics_snn_judgment_callback_t callback,
    void* user_data)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->judgment_callback = callback;
    bridge->judgment_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int ethics_snn_register_conflict_callback(
    ethics_snn_bridge_t* bridge,
    ethics_snn_conflict_callback_t callback,
    void* user_data)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->conflict_callback = callback;
    bridge->conflict_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int ethics_snn_bio_async_connect(ethics_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    /* Bio-async connection would be implemented here */
    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int ethics_snn_bio_async_disconnect(ethics_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

bool ethics_snn_is_bio_async_connected(ethics_snn_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->base.mutex);
    bool connected = bridge->bio_async_connected;
    nimcp_mutex_unlock(bridge->base.mutex);

    return connected;
}
