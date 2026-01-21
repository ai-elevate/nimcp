/**
 * @file nimcp_empathy_snn_bridge.c
 * @brief Empathy - SNN Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/empathetic_response/nimcp_empathy_snn_bridge.h"
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

struct empathy_snn_bridge {
    empathy_snn_config_t config;
    snn_network_t* snn;
    nimcp_mutex_t* mutex;

    /* State */
    empathy_snn_state_t state;
    uint64_t current_time_us;
    bool bio_async_connected;

    /* Dimension state */
    empathy_dim_state_t dim_states[EMPATHY_SNN_MAX_DIMENSIONS];

    /* Buffers */
    float* encoding_buffer;
    float* output_buffer;
    float* response_buffer;

    /* Response state */
    empathy_response_t last_response;
    float mirroring_signal;
    float compassion_signal;

    /* Previous state for change detection */
    float* prev_state;

    /* Callbacks */
    empathy_snn_mirroring_callback_t mirroring_callback;
    void* mirroring_callback_data;
    empathy_snn_response_callback_t response_callback;
    void* response_callback_data;
    empathy_snn_compassion_callback_t compassion_callback;
    void* compassion_callback_data;

    /* Statistics */
    empathy_snn_stats_t stats;
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

empathy_snn_config_t empathy_snn_config_default(void) {
    empathy_snn_config_t config = {
        .num_dimensions = EMPATHY_DIM_COUNT,
        .neurons_per_dim = EMPATHY_SNN_NEURONS_PER_DIM,
        .hidden_dim = 128,

        .dt_ms = 1.0f,
        .encoding_window_ms = EMPATHY_SNN_ENCODING_WINDOW,
        .integration_tau_ms = 100.0f,

        .encoding = EMPATHY_SNN_ENCODE_POPULATION,
        .encoding_gain = 1.0f,
        .baseline_rate_hz = 5.0f,
        .max_rate_hz = 100.0f,

        .decoding = EMPATHY_SNN_DECODE_INTEGRATION,
        .compassion_threshold = EMPATHY_SNN_COMPASSION_THRESH,
        .empathic_concern_threshold = 0.6f,
        .state_change_threshold = 0.2f,

        .enable_competition = true,
        .inhibition_strength = 0.3f,
        .enable_mirroring_detection = true,
        .mirroring_sensitivity = 1.0f,

        .enable_compassion = true,
        .compassion_gain = 1.5f,
        .enable_distress_modulation = true,

        .enable_bio_async = false,
        .enable_plasticity_integration = true
    };
    return config;
}

empathy_snn_bridge_t* empathy_snn_create(const empathy_snn_config_t* config) {
    empathy_snn_bridge_t* bridge = nimcp_calloc(1, sizeof(empathy_snn_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = empathy_snn_config_default();
    }

    /* Validate config */
    if (bridge->config.num_dimensions == 0 ||
        bridge->config.num_dimensions > EMPATHY_SNN_MAX_DIMENSIONS) {
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
    uint32_t output_dim = 6; /* mirroring, perspective, affective_sharing, empathic_concern, compassion, validation */

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
    bridge->response_buffer = nimcp_calloc(bridge->config.num_dimensions, sizeof(float));
    bridge->prev_state = nimcp_calloc(bridge->config.num_dimensions, sizeof(float));

    if (!bridge->encoding_buffer || !bridge->output_buffer ||
        !bridge->response_buffer || !bridge->prev_state) {
        empathy_snn_destroy(bridge);
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

    /* Initialize response to neutral empathic baseline */
    bridge->last_response.mirroring_level = 0.5f;
    bridge->last_response.perspective_taking = 0.5f;
    bridge->last_response.affective_sharing = 0.5f;
    bridge->last_response.empathic_concern = 0.5f;
    bridge->last_response.compassion_response = 0.5f;
    bridge->last_response.high_empathy_detected = false;
    bridge->last_response.compassion_activated = false;
    bridge->last_response.distress_tolerance = 0.7f;
    bridge->last_response.prosocial_motivation = 0.5f;
    bridge->last_response.validation_readiness = 0.5f;

    bridge->state = EMPATHY_SNN_STATE_IDLE;
    bridge->current_time_us = 0;
    bridge->bio_async_connected = false;
    bridge->mirroring_signal = 0.0f;
    bridge->compassion_signal = 0.0f;

    return bridge;
}

void empathy_snn_destroy(empathy_snn_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->snn) {
        snn_network_destroy(bridge->snn);
    }

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }

    nimcp_free(bridge->encoding_buffer);
    nimcp_free(bridge->output_buffer);
    nimcp_free(bridge->response_buffer);
    nimcp_free(bridge->prev_state);
    nimcp_free(bridge);
}

int empathy_snn_reset(empathy_snn_bridge_t* bridge) {
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

    /* Reset response */
    memset(&bridge->last_response, 0, sizeof(empathy_response_t));
    bridge->last_response.mirroring_level = 0.5f;
    bridge->last_response.perspective_taking = 0.5f;
    bridge->last_response.empathic_concern = 0.5f;

    /* Reset buffers */
    uint32_t input_dim = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
    memset(bridge->encoding_buffer, 0, input_dim * sizeof(float));
    memset(bridge->output_buffer, 0, 6 * sizeof(float));
    memset(bridge->response_buffer, 0, bridge->config.num_dimensions * sizeof(float));
    memset(bridge->prev_state, 0, bridge->config.num_dimensions * sizeof(float));

    bridge->state = EMPATHY_SNN_STATE_IDLE;
    bridge->mirroring_signal = 0.0f;
    bridge->compassion_signal = 0.0f;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Encoding Functions
//=============================================================================

int empathy_snn_encode_state(
    empathy_snn_bridge_t* bridge,
    const float* dimensions,
    uint32_t num_dims
) {
    if (!bridge || !dimensions) return -1;
    if (num_dims == 0 || num_dims > bridge->config.num_dimensions) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = EMPATHY_SNN_STATE_ENCODING;

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

    /* Detect state change */
    float change_magnitude = 0.0f;
    for (uint32_t d = 0; d < num_dims; d++) {
        float diff = dimensions[d] - bridge->prev_state[d];
        change_magnitude += diff * diff;
        bridge->prev_state[d] = dimensions[d];
    }
    change_magnitude = sqrtf(change_magnitude / num_dims);

    if (change_magnitude > bridge->config.state_change_threshold) {
        bridge->stats.state_changes++;
    }

    bridge->stats.total_spikes += total_spikes;

    nimcp_mutex_unlock(bridge->mutex);
    return total_spikes;
}

int empathy_snn_encode_mirroring(
    empathy_snn_bridge_t* bridge,
    float mirroring,
    float target_emotion
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float dims[EMPATHY_DIM_COUNT] = {0};
    dims[EMPATHY_DIM_EMOTIONAL_MIRRORING] = clamp_f(mirroring, 0.0f, 1.0f);
    dims[EMPATHY_DIM_AFFECTIVE_SHARING] = clamp_f(target_emotion * mirroring, 0.0f, 1.0f);
    dims[EMPATHY_DIM_SELF_OTHER_DISTINCTION] = clamp_f(1.0f - mirroring * 0.3f, 0.0f, 1.0f);

    bridge->mirroring_signal = mirroring;

    nimcp_mutex_unlock(bridge->mutex);

    return empathy_snn_encode_state(bridge, dims, 3);
}

int empathy_snn_encode_perspective(
    empathy_snn_bridge_t* bridge,
    float perspective,
    float self_other_clarity
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float dims[EMPATHY_DIM_COUNT] = {0};
    dims[EMPATHY_DIM_PERSPECTIVE_TAKING] = clamp_f(perspective, 0.0f, 1.0f);
    dims[EMPATHY_DIM_SELF_OTHER_DISTINCTION] = clamp_f(self_other_clarity, 0.0f, 1.0f);

    nimcp_mutex_unlock(bridge->mutex);

    return empathy_snn_encode_state(bridge, dims, 2);
}

int empathy_snn_encode_compassion(
    empathy_snn_bridge_t* bridge,
    float compassion,
    float empathic_concern
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float dims[EMPATHY_DIM_COUNT] = {0};
    dims[EMPATHY_DIM_COMPASSION] = clamp_f(compassion, 0.0f, 1.0f);
    dims[EMPATHY_DIM_EMPATHIC_CONCERN] = clamp_f(empathic_concern, 0.0f, 1.0f);
    dims[EMPATHY_DIM_PROSOCIAL_MOTIVATION] = clamp_f((compassion + empathic_concern) / 2.0f, 0.0f, 1.0f);

    bridge->compassion_signal = compassion;

    if (compassion > bridge->config.compassion_threshold) {
        bridge->last_response.compassion_activated = true;
        bridge->stats.compassion_activations++;

        if (bridge->compassion_callback) {
            bridge->compassion_callback(bridge, compassion, EMPATHY_DIM_COMPASSION,
                                       bridge->compassion_callback_data);
        }
    }

    nimcp_mutex_unlock(bridge->mutex);

    return empathy_snn_encode_state(bridge, dims, 3);
}

//=============================================================================
// Simulation Functions
//=============================================================================

int empathy_snn_simulate(empathy_snn_bridge_t* bridge, float duration_ms) {
    if (!bridge) return -1;
    if (duration_ms <= 0.0f) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = EMPATHY_SNN_STATE_SIMULATING;

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
    bridge->last_response.mirroring_level = clamp_f(bridge->output_buffer[0], 0.0f, 1.0f);
    bridge->last_response.perspective_taking = clamp_f(bridge->output_buffer[1], 0.0f, 1.0f);
    bridge->last_response.affective_sharing = clamp_f(bridge->output_buffer[2], 0.0f, 1.0f);
    bridge->last_response.empathic_concern = clamp_f(bridge->output_buffer[3], 0.0f, 1.0f);
    bridge->last_response.compassion_response = clamp_f(bridge->output_buffer[4], 0.0f, 1.0f);
    bridge->last_response.validation_readiness = clamp_f(bridge->output_buffer[5], 0.0f, 1.0f);

    /* Calculate combined empathy level */
    float empathy_level = (bridge->last_response.mirroring_level +
                          bridge->last_response.perspective_taking +
                          bridge->last_response.empathic_concern +
                          bridge->last_response.compassion_response) / 4.0f;

    /* Check for high empathy threshold */
    if (empathy_level > bridge->config.compassion_threshold) {
        bridge->last_response.high_empathy_detected = true;
        bridge->stats.high_empathy_events++;

        if (bridge->mirroring_callback) {
            bridge->mirroring_callback(bridge, bridge->last_response.mirroring_level,
                                      bridge->current_time_us, bridge->mirroring_callback_data);
        }
    } else {
        bridge->last_response.high_empathy_detected = false;
    }

    /* Check compassion activation */
    if (bridge->last_response.compassion_response > bridge->config.compassion_threshold) {
        bridge->last_response.compassion_activated = true;
    }

    /* Update prosocial motivation */
    bridge->last_response.prosocial_motivation =
        (bridge->last_response.empathic_concern + bridge->last_response.compassion_response) / 2.0f;

    /* Update distress tolerance based on self-other distinction */
    bridge->last_response.distress_tolerance =
        0.5f + 0.5f * bridge->dim_states[EMPATHY_DIM_SELF_OTHER_DISTINCTION].activation;

    bridge->stats.total_evaluations++;
    bridge->stats.mean_empathy_level =
        (bridge->stats.mean_empathy_level * (bridge->stats.total_evaluations - 1) +
         empathy_level) / bridge->stats.total_evaluations;

    bridge->state = EMPATHY_SNN_STATE_IDLE;

    /* Invoke response callback */
    if (bridge->response_callback) {
        bridge->response_callback(bridge, &bridge->last_response, bridge->response_callback_data);
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int empathy_snn_step(empathy_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    return empathy_snn_simulate(bridge, bridge->config.dt_ms);
}

int empathy_snn_forward(
    empathy_snn_bridge_t* bridge,
    const float* inputs,
    uint32_t input_count
) {
    if (!bridge || !inputs) return -1;

    int spike_count = empathy_snn_encode_state(bridge, inputs, input_count);
    if (spike_count < 0) return -1;

    if (empathy_snn_simulate(bridge, bridge->config.encoding_window_ms) < 0) {
        return -1;
    }

    return spike_count;
}

//=============================================================================
// Decoding Functions
//=============================================================================

int empathy_snn_get_response(
    empathy_snn_bridge_t* bridge,
    empathy_response_t* response
) {
    if (!bridge || !response) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *response = bridge->last_response;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int empathy_snn_get_activations(
    empathy_snn_bridge_t* bridge,
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

bool empathy_snn_check_empathy(
    empathy_snn_bridge_t* bridge,
    float* empathy_level
) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    float level = (bridge->last_response.mirroring_level +
                   bridge->last_response.perspective_taking +
                   bridge->last_response.empathic_concern +
                   bridge->last_response.compassion_response) / 4.0f;
    if (empathy_level) {
        *empathy_level = level;
    }
    bool detected = level > bridge->config.compassion_threshold;
    nimcp_mutex_unlock(bridge->mutex);

    return detected;
}

bool empathy_snn_check_compassion(
    empathy_snn_bridge_t* bridge,
    float* compassion_level
) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    float level = bridge->last_response.compassion_response;
    if (compassion_level) {
        *compassion_level = level;
    }
    bool detected = level > bridge->config.compassion_threshold;
    nimcp_mutex_unlock(bridge->mutex);

    return detected;
}

bool empathy_snn_check_state_change(
    empathy_snn_bridge_t* bridge,
    float* change_magnitude
) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    /* Calculate magnitude from prev_state differences */
    float mag = 0.0f;
    for (uint32_t d = 0; d < bridge->config.num_dimensions; d++) {
        float diff = bridge->dim_states[d].activation - bridge->prev_state[d];
        mag += diff * diff;
    }
    mag = sqrtf(mag / bridge->config.num_dimensions);

    if (change_magnitude) {
        *change_magnitude = mag;
    }
    bool changed = mag > bridge->config.state_change_threshold;
    nimcp_mutex_unlock(bridge->mutex);

    return changed;
}

//=============================================================================
// State Query Functions
//=============================================================================

int empathy_snn_get_dim_state(
    empathy_snn_bridge_t* bridge,
    uint32_t dim,
    empathy_dim_state_t* state
) {
    if (!bridge || !state) return -1;
    if (dim >= bridge->config.num_dimensions) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->dim_states[dim];
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int empathy_snn_get_state(
    empathy_snn_bridge_t* bridge,
    empathy_snn_bridge_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->mutex);

    state->state = bridge->state;
    state->mean_empathy = (bridge->last_response.mirroring_level +
                          bridge->last_response.perspective_taking +
                          bridge->last_response.empathic_concern +
                          bridge->last_response.compassion_response) / 4.0f;
    state->mirroring_signal = bridge->mirroring_signal;
    state->compassion_signal = bridge->compassion_signal;

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

int empathy_snn_get_stats(empathy_snn_bridge_t* bridge, empathy_snn_stats_t* stats) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int empathy_snn_reset_stats(empathy_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(empathy_snn_stats_t));
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

float empathy_snn_get_empathic_concern(empathy_snn_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(bridge->mutex);
    float concern = bridge->last_response.empathic_concern;
    nimcp_mutex_unlock(bridge->mutex);

    return concern;
}

float empathy_snn_get_total_activity(empathy_snn_bridge_t* bridge) {
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

int empathy_snn_register_mirroring_callback(
    empathy_snn_bridge_t* bridge,
    empathy_snn_mirroring_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->mirroring_callback = callback;
    bridge->mirroring_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int empathy_snn_register_response_callback(
    empathy_snn_bridge_t* bridge,
    empathy_snn_response_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->response_callback = callback;
    bridge->response_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int empathy_snn_register_compassion_callback(
    empathy_snn_bridge_t* bridge,
    empathy_snn_compassion_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->compassion_callback = callback;
    bridge->compassion_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int empathy_snn_bio_async_connect(empathy_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return -1;

    nimcp_mutex_lock(bridge->mutex);
    /* Bio-async connection would be implemented here */
    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int empathy_snn_bio_async_disconnect(empathy_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

bool empathy_snn_is_bio_async_connected(empathy_snn_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    bool connected = bridge->bio_async_connected;
    nimcp_mutex_unlock(bridge->mutex);

    return connected;
}
