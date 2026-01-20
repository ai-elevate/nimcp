/**
 * @file nimcp_executive_snn_bridge.c
 * @brief Executive Function - SNN Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/executive/nimcp_executive_snn_bridge.h"
#include "snn/nimcp_snn_network.h"
#include "snn/nimcp_snn_config.h"
#include "core/neuron_types/nimcp_neuron_types.h"
#include "core/synapse_types/nimcp_synapse_types.h"
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

struct executive_snn_bridge {
    executive_snn_config_t config;
    snn_network_t* snn;
    nimcp_mutex_t* mutex;

    /* State */
    executive_snn_state_t state;
    uint64_t current_time_us;
    bool bio_async_connected;

    /* Dimension state */
    executive_dim_state_t dim_states[EXECUTIVE_SNN_MAX_DIMENSIONS];

    /* Buffers */
    float* encoding_buffer;
    float* output_buffer;
    float* control_buffer;

    /* Control output state */
    executive_control_output_t last_output;
    float conflict_signal;
    float error_signal;

    /* Previous state for change detection */
    float* prev_state;

    /* Callbacks */
    executive_snn_conflict_callback_t conflict_callback;
    void* conflict_callback_data;
    executive_snn_control_callback_t control_callback;
    void* control_callback_data;
    executive_snn_error_callback_t error_callback;
    void* error_callback_data;

    /* Statistics */
    executive_snn_stats_t stats;
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

executive_snn_config_t executive_snn_config_default(void) {
    executive_snn_config_t config = {
        .num_dimensions = EXEC_DIM_COUNT,
        .neurons_per_dim = EXECUTIVE_SNN_NEURONS_PER_DIM,
        .hidden_dim = 128,

        .dt_ms = 1.0f,
        .encoding_window_ms = EXECUTIVE_SNN_ENCODING_WINDOW,
        .integration_tau_ms = 100.0f,

        .encoding = EXECUTIVE_SNN_ENCODE_POPULATION,
        .encoding_gain = 1.0f,
        .baseline_rate_hz = 5.0f,
        .max_rate_hz = 100.0f,

        .decoding = EXECUTIVE_SNN_DECODE_INTEGRATION,
        .conflict_threshold = EXECUTIVE_SNN_CONFLICT_THRESH,
        .inhibition_threshold = 0.6f,
        .goal_change_threshold = 0.2f,

        .enable_competition = true,
        .inhibition_strength = 0.3f,
        .enable_conflict_detection = true,
        .error_threshold = 0.4f,

        .enable_goal_maintenance = true,
        .goal_persistence_gain = 1.5f,
        .enable_task_switching = true,

        .enable_bio_async = false,
        .enable_plasticity_integration = true
    };
    return config;
}

executive_snn_bridge_t* executive_snn_create(const executive_snn_config_t* config) {
    executive_snn_bridge_t* bridge = nimcp_calloc(1, sizeof(executive_snn_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = executive_snn_config_default();
    }

    /* Validate config */
    if (bridge->config.num_dimensions == 0 ||
        bridge->config.num_dimensions > EXECUTIVE_SNN_MAX_DIMENSIONS) {
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
    uint32_t input_dim = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
    uint32_t hidden_dim = bridge->config.hidden_dim;
    uint32_t output_dim = 6; /* inhibition, flexibility, planning, goal, conflict, control */

    snn_config_feedforward(&snn_config, input_dim, hidden_dim, output_dim);
    snn_config.dt = bridge->config.dt_ms;
    snn_config.enable_bio_async = bridge->config.enable_bio_async;

    bridge->snn = snn_network_create(&snn_config);
    if (!bridge->snn) {
        nimcp_mutex_free(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate buffers */
    bridge->encoding_buffer = nimcp_calloc(input_dim, sizeof(float));
    bridge->output_buffer = nimcp_calloc(output_dim, sizeof(float));
    bridge->control_buffer = nimcp_calloc(bridge->config.num_dimensions, sizeof(float));
    bridge->prev_state = nimcp_calloc(bridge->config.num_dimensions, sizeof(float));

    if (!bridge->encoding_buffer || !bridge->output_buffer ||
        !bridge->control_buffer || !bridge->prev_state) {
        executive_snn_destroy(bridge);
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

    /* Initialize control output to neutral */
    bridge->last_output.inhibition_level = 0.5f;
    bridge->last_output.working_memory_load = 0.5f;
    bridge->last_output.flexibility_level = 0.5f;
    bridge->last_output.planning_activity = 0.5f;
    bridge->last_output.task_switching_cost = 0.5f;
    bridge->last_output.attention_control = 0.5f;
    bridge->last_output.goal_change_detected = false;
    bridge->last_output.conflict_detected = false;
    bridge->last_output.conflict_magnitude = 0.0f;
    bridge->last_output.goal_strength = 0.5f;
    bridge->last_output.resource_allocation = 0.5f;

    bridge->state = EXECUTIVE_SNN_STATE_IDLE;
    bridge->current_time_us = 0;
    bridge->bio_async_connected = false;
    bridge->conflict_signal = 0.0f;
    bridge->error_signal = 0.0f;

    return bridge;
}

void executive_snn_destroy(executive_snn_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->snn) {
        snn_network_destroy(bridge->snn);
    }

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }

    nimcp_free(bridge->encoding_buffer);
    nimcp_free(bridge->output_buffer);
    nimcp_free(bridge->control_buffer);
    nimcp_free(bridge->prev_state);
    nimcp_free(bridge);
}

int executive_snn_reset(executive_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Reset SNN network */
    if (bridge->snn) {
        snn_network_reset(bridge->snn);
    }

    /* Reset dimension states */
    for (uint32_t i = 0; i < bridge->config.num_dimensions; i++) {
        bridge->dim_states[i].activation = 0.0f;
        bridge->dim_states[i].accumulated_evidence = 0.0f;
        bridge->dim_states[i].spike_count = 0;
        bridge->dim_states[i].mean_rate_hz = bridge->config.baseline_rate_hz;
        bridge->dim_states[i].last_spike_time_us = 0;
    }

    /* Reset control output */
    memset(&bridge->last_output, 0, sizeof(executive_control_output_t));
    bridge->last_output.inhibition_level = 0.5f;
    bridge->last_output.working_memory_load = 0.5f;
    bridge->last_output.flexibility_level = 0.5f;

    /* Reset buffers */
    uint32_t input_dim = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
    memset(bridge->encoding_buffer, 0, input_dim * sizeof(float));
    memset(bridge->output_buffer, 0, 6 * sizeof(float));
    memset(bridge->control_buffer, 0, bridge->config.num_dimensions * sizeof(float));
    memset(bridge->prev_state, 0, bridge->config.num_dimensions * sizeof(float));

    bridge->state = EXECUTIVE_SNN_STATE_IDLE;
    bridge->conflict_signal = 0.0f;
    bridge->error_signal = 0.0f;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Encoding Functions
//=============================================================================

int executive_snn_encode_state(
    executive_snn_bridge_t* bridge,
    const float* dimensions,
    uint32_t num_dims
) {
    if (!bridge || !dimensions) return -1;
    if (num_dims == 0 || num_dims > bridge->config.num_dimensions) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = EXECUTIVE_SNN_STATE_ENCODING;

    uint32_t neurons_per_dim = bridge->config.neurons_per_dim;
    int total_spikes = 0;

    /* Population encoding for each dimension */
    for (uint32_t d = 0; d < num_dims; d++) {
        float value = clamp_f(dimensions[d], 0.0f, 1.0f);
        float rate = bridge->config.baseline_rate_hz +
                    value * (bridge->config.max_rate_hz - bridge->config.baseline_rate_hz);

        /* Update dimension state */
        bridge->dim_states[d].activation = value;
        bridge->dim_states[d].mean_rate_hz = rate;

        /* Population encode */
        for (uint32_t n = 0; n < neurons_per_dim; n++) {
            float preferred = (float)n / (neurons_per_dim - 1);
            float diff = value - preferred;
            float tuning = expf(-diff * diff / 0.1f);
            uint32_t idx = d * neurons_per_dim + n;
            bridge->encoding_buffer[idx] = tuning * bridge->config.encoding_gain;
            if (bridge->encoding_buffer[idx] > 0.5f) {
                total_spikes++;
                bridge->dim_states[d].spike_count++;
            }
        }
    }

    /* Detect goal state change */
    float change_magnitude = 0.0f;
    for (uint32_t d = 0; d < num_dims; d++) {
        float diff = dimensions[d] - bridge->prev_state[d];
        change_magnitude += diff * diff;
        bridge->prev_state[d] = dimensions[d];
    }
    change_magnitude = sqrtf(change_magnitude / num_dims);

    if (change_magnitude > bridge->config.goal_change_threshold) {
        bridge->last_output.goal_change_detected = true;
        bridge->stats.goal_changes++;
    } else {
        bridge->last_output.goal_change_detected = false;
    }

    bridge->stats.total_spikes += total_spikes;

    nimcp_mutex_unlock(bridge->mutex);
    return total_spikes;
}

int executive_snn_encode_inhibition(
    executive_snn_bridge_t* bridge,
    float inhibition_strength,
    float urgency
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float dims[EXEC_DIM_COUNT] = {0};
    dims[EXEC_DIM_INHIBITION] = clamp_f(inhibition_strength, 0.0f, 1.0f);
    dims[EXEC_DIM_ATTENTION_CONTROL] = clamp_f(urgency, 0.0f, 1.0f);
    dims[EXEC_DIM_CONFLICT_MONITOR] = clamp_f(urgency * 0.5f, 0.0f, 1.0f);

    nimcp_mutex_unlock(bridge->mutex);

    return executive_snn_encode_state(bridge, dims, 3);
}

int executive_snn_encode_task(
    executive_snn_bridge_t* bridge,
    float task_load,
    uint32_t task_count
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float dims[EXEC_DIM_COUNT] = {0};
    dims[EXEC_DIM_WORKING_MEMORY] = clamp_f(task_load, 0.0f, 1.0f);
    dims[EXEC_DIM_TASK_SWITCHING] = clamp_f((float)task_count / 10.0f, 0.0f, 1.0f);
    dims[EXEC_DIM_RESOURCE_ALLOCATION] = clamp_f(task_load * 0.8f, 0.0f, 1.0f);

    nimcp_mutex_unlock(bridge->mutex);

    return executive_snn_encode_state(bridge, dims, 3);
}

int executive_snn_encode_conflict(
    executive_snn_bridge_t* bridge,
    float conflict_magnitude,
    uint32_t conflict_type
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float dims[EXEC_DIM_COUNT] = {0};
    dims[EXEC_DIM_CONFLICT_MONITOR] = clamp_f(conflict_magnitude, 0.0f, 1.0f);
    dims[EXEC_DIM_ERROR_CORRECTION] = clamp_f(conflict_magnitude * 0.5f, 0.0f, 1.0f);

    bridge->conflict_signal = conflict_magnitude;

    if (conflict_magnitude > bridge->config.conflict_threshold) {
        bridge->last_output.conflict_detected = true;
        bridge->last_output.conflict_magnitude = conflict_magnitude;
        bridge->stats.conflict_detections++;

        if (bridge->conflict_callback) {
            bridge->conflict_callback(bridge, conflict_magnitude, conflict_type,
                                      bridge->conflict_callback_data);
        }
    }

    nimcp_mutex_unlock(bridge->mutex);

    return executive_snn_encode_state(bridge, dims, 2);
}

//=============================================================================
// Simulation Functions
//=============================================================================

int executive_snn_simulate(executive_snn_bridge_t* bridge, float duration_ms) {
    if (!bridge) return -1;
    if (duration_ms <= 0.0f) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = EXECUTIVE_SNN_STATE_SIMULATING;

    float dt = bridge->config.dt_ms;
    uint32_t steps = (uint32_t)(duration_ms / dt);

    /* Set inputs before simulation */
    if (bridge->snn) {
        uint32_t input_size = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
        snn_network_set_inputs(bridge->snn, bridge->encoding_buffer, input_size);
    }

    for (uint32_t s = 0; s < steps; s++) {
        if (bridge->snn) {
            snn_network_step(bridge->snn, dt);
        }

        /* Update evidence integration */
        float decay = expf(-dt / bridge->config.integration_tau_ms);
        for (uint32_t d = 0; d < bridge->config.num_dimensions; d++) {
            bridge->dim_states[d].accumulated_evidence *= decay;
            bridge->dim_states[d].accumulated_evidence +=
                bridge->dim_states[d].activation * dt / bridge->config.integration_tau_ms;
        }

        bridge->current_time_us += (uint64_t)(dt * 1000.0f);
        bridge->stats.total_simulations++;
    }

    /* Get output from SNN */
    if (bridge->snn) {
        snn_network_get_outputs(bridge->snn, bridge->output_buffer, 6);
    }

    /* Decode outputs */
    bridge->last_output.inhibition_level = clamp_f(bridge->output_buffer[0], 0.0f, 1.0f);
    bridge->last_output.flexibility_level = clamp_f(bridge->output_buffer[1], 0.0f, 1.0f);
    bridge->last_output.planning_activity = clamp_f(bridge->output_buffer[2], 0.0f, 1.0f);
    bridge->last_output.goal_strength = clamp_f(bridge->output_buffer[3], 0.0f, 1.0f);
    bridge->last_output.conflict_magnitude = clamp_f(bridge->output_buffer[4], 0.0f, 1.0f);
    bridge->last_output.attention_control = clamp_f(bridge->output_buffer[5], 0.0f, 1.0f);

    /* Check conflict threshold */
    if (bridge->last_output.conflict_magnitude > bridge->config.conflict_threshold) {
        bridge->stats.conflict_detections++;
        bridge->last_output.conflict_detected = true;

        if (bridge->conflict_callback) {
            bridge->conflict_callback(bridge, bridge->last_output.conflict_magnitude,
                                     bridge->current_time_us, bridge->conflict_callback_data);
        }
    }

    bridge->stats.total_evaluations++;
    bridge->state = EXECUTIVE_SNN_STATE_IDLE;

    /* Invoke control callback */
    if (bridge->control_callback) {
        bridge->control_callback(bridge, &bridge->last_output, bridge->control_callback_data);
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int executive_snn_step(executive_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    return executive_snn_simulate(bridge, bridge->config.dt_ms);
}

int executive_snn_forward(
    executive_snn_bridge_t* bridge,
    const float* inputs,
    uint32_t input_count
) {
    if (!bridge || !inputs) return -1;

    int spike_count = executive_snn_encode_state(bridge, inputs, input_count);
    if (spike_count < 0) return -1;

    if (executive_snn_simulate(bridge, bridge->config.encoding_window_ms) < 0) {
        return -1;
    }

    return spike_count;
}

//=============================================================================
// Decoding Functions
//=============================================================================

int executive_snn_get_control_output(
    executive_snn_bridge_t* bridge,
    executive_control_output_t* output
) {
    if (!bridge || !output) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *output = bridge->last_output;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int executive_snn_get_activations(
    executive_snn_bridge_t* bridge,
    float* activations,
    uint32_t num_dims
) {
    if (!bridge || !activations) return -1;
    if (num_dims == 0 || num_dims > bridge->config.num_dimensions) return -1;

    nimcp_mutex_lock(bridge->mutex);
    for (uint32_t d = 0; d < num_dims; d++) {
        activations[d] = bridge->dim_states[d].activation;
    }
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

bool executive_snn_check_conflict(
    executive_snn_bridge_t* bridge,
    float* conflict_level
) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    float level = bridge->last_output.conflict_magnitude;
    if (conflict_level) {
        *conflict_level = level;
    }
    bool detected = level > bridge->config.conflict_threshold;
    nimcp_mutex_unlock(bridge->mutex);

    return detected;
}

bool executive_snn_check_error(
    executive_snn_bridge_t* bridge,
    float* error_level
) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    float level = bridge->error_signal;
    if (error_level) {
        *error_level = level;
    }
    bool detected = level > bridge->config.error_threshold;
    nimcp_mutex_unlock(bridge->mutex);

    return detected;
}

bool executive_snn_check_goal_change(
    executive_snn_bridge_t* bridge,
    float* change_magnitude
) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    bool changed = bridge->last_output.goal_change_detected;
    if (change_magnitude && changed) {
        /* Calculate magnitude from prev_state differences */
        float mag = 0.0f;
        for (uint32_t d = 0; d < bridge->config.num_dimensions; d++) {
            float diff = bridge->dim_states[d].activation - bridge->prev_state[d];
            mag += diff * diff;
        }
        *change_magnitude = sqrtf(mag / bridge->config.num_dimensions);
    }
    nimcp_mutex_unlock(bridge->mutex);

    return changed;
}

//=============================================================================
// State Query Functions
//=============================================================================

int executive_snn_get_dim_state(
    executive_snn_bridge_t* bridge,
    uint32_t dim,
    executive_dim_state_t* state
) {
    if (!bridge || !state) return -1;
    if (dim >= bridge->config.num_dimensions) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->dim_states[dim];
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int executive_snn_get_state(
    executive_snn_bridge_t* bridge,
    executive_snn_bridge_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->mutex);

    state->state = bridge->state;
    state->mean_inhibition = bridge->last_output.inhibition_level;
    state->conflict_signal = bridge->conflict_signal;
    state->error_signal = bridge->error_signal;

    /* Count active dimensions */
    state->active_dimensions = 0;
    state->total_activity = 0.0f;
    for (uint32_t d = 0; d < bridge->config.num_dimensions; d++) {
        if (bridge->dim_states[d].activation > 0.1f) {
            state->active_dimensions++;
        }
        state->total_activity += bridge->dim_states[d].activation;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int executive_snn_get_stats(executive_snn_bridge_t* bridge, executive_snn_stats_t* stats) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int executive_snn_reset_stats(executive_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(executive_snn_stats_t));
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

float executive_snn_get_inhibition(executive_snn_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(bridge->mutex);
    float inhibition = bridge->last_output.inhibition_level;
    nimcp_mutex_unlock(bridge->mutex);

    return inhibition;
}

float executive_snn_get_total_activity(executive_snn_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(bridge->mutex);
    float total = 0.0f;
    for (uint32_t d = 0; d < bridge->config.num_dimensions; d++) {
        total += bridge->dim_states[d].activation;
    }
    nimcp_mutex_unlock(bridge->mutex);

    return total;
}

//=============================================================================
// Callback Registration
//=============================================================================

int executive_snn_register_conflict_callback(
    executive_snn_bridge_t* bridge,
    executive_snn_conflict_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->conflict_callback = callback;
    bridge->conflict_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int executive_snn_register_control_callback(
    executive_snn_bridge_t* bridge,
    executive_snn_control_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->control_callback = callback;
    bridge->control_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int executive_snn_register_error_callback(
    executive_snn_bridge_t* bridge,
    executive_snn_error_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->error_callback = callback;
    bridge->error_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int executive_snn_bio_async_connect(executive_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return -1;

    nimcp_mutex_lock(bridge->mutex);
    /* Bio-async connection would be implemented here */
    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int executive_snn_bio_async_disconnect(executive_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

bool executive_snn_is_bio_async_connected(executive_snn_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    bool connected = bridge->bio_async_connected;
    nimcp_mutex_unlock(bridge->mutex);

    return connected;
}
