/**
 * @file nimcp_predictive_snn_bridge.c
 * @brief Predictive Processing - SNN Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/predictive/nimcp_predictive_snn_bridge.h"
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
#include "utils/exception/nimcp_exception_immune.h"

#include <string.h>
#include <math.h>
#include <stdio.h>

//=============================================================================
// Internal Structures
//=============================================================================

struct predictive_snn_bridge {
    bridge_base_t base;
    predictive_snn_config_t config;
    snn_network_t* snn;

    /* State */
    predictive_snn_state_t state;
    uint64_t current_time_us;
    bool bio_async_connected;

    /* Dimension state */
    predictive_dim_state_t dim_states[PREDICTIVE_SNN_MAX_DIMENSIONS];

    /* Buffers */
    float* encoding_buffer;
    float* output_buffer;
    float* anticipation_buffer;

    /* Anticipation state */
    predictive_anticipation_t last_anticipation;
    float error_signal;
    float precision_signal;

    /* Previous state for change detection */
    float* prev_state;

    /* Callbacks */
    predictive_snn_error_callback_t error_callback;
    void* error_callback_data;
    predictive_snn_anticipation_callback_t anticipation_callback;
    void* anticipation_callback_data;
    predictive_snn_high_anticipation_callback_t high_anticipation_callback;
    void* high_anticipation_callback_data;

    /* Statistics */
    predictive_snn_stats_t stats;
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

predictive_snn_config_t predictive_snn_config_default(void) {
    predictive_snn_config_t config = {
        .num_dimensions = PREDICTIVE_DIM_COUNT,
        .neurons_per_dim = PREDICTIVE_SNN_NEURONS_PER_DIM,
        .hidden_dim = 128,

        .dt_ms = 1.0f,
        .encoding_window_ms = PREDICTIVE_SNN_ENCODING_WINDOW,
        .integration_tau_ms = 100.0f,

        .encoding = PREDICTIVE_SNN_ENCODE_POPULATION,
        .encoding_gain = 1.0f,
        .baseline_rate_hz = 5.0f,
        .max_rate_hz = 100.0f,

        .decoding = PREDICTIVE_SNN_DECODE_INTEGRATION,
        .error_threshold = PREDICTIVE_SNN_ERROR_THRESH,
        .anticipation_threshold = 0.6f,
        .state_change_threshold = 0.2f,

        .enable_competition = true,
        .inhibition_strength = 0.3f,
        .enable_error_detection = true,
        .error_sensitivity = 1.0f,

        .enable_anticipation = true,
        .anticipation_gain = 1.5f,
        .enable_model_updating = true,

        .enable_bio_async = false,
        .enable_plasticity_integration = true
    };
    return config;
}

predictive_snn_bridge_t* predictive_snn_create(const predictive_snn_config_t* config) {
    predictive_snn_bridge_t* bridge = nimcp_calloc(1, sizeof(predictive_snn_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = predictive_snn_config_default();
    }

    /* Validate config */
    if (bridge->config.num_dimensions == 0 ||
        bridge->config.num_dimensions > PREDICTIVE_SNN_MAX_DIMENSIONS) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize bridge base */
    if (bridge_base_init(&bridge->base, 0, "predictive_snn") != 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to initialize bridge base in predictive_snn_create");
        nimcp_free(bridge);
        return NULL;
    }

    /* Create SNN network */
    snn_config_t snn_config;
    uint32_t input_dim = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
    uint32_t hidden_dim = bridge->config.hidden_dim;
    uint32_t output_dim = 6; /* prediction, error, precision, anticipation, model, free_energy */

    snn_config_feedforward(&snn_config, input_dim, hidden_dim, output_dim);
    snn_config.dt = bridge->config.dt_ms;
    snn_config.enable_bio_async = bridge->config.enable_bio_async;

    bridge->snn = snn_network_create(&snn_config);
    if (!bridge->snn) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to create SNN network in predictive_snn_create");
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate buffers */
    bridge->encoding_buffer = nimcp_calloc(input_dim, sizeof(float));
    bridge->output_buffer = nimcp_calloc(output_dim, sizeof(float));
    bridge->anticipation_buffer = nimcp_calloc(bridge->config.num_dimensions, sizeof(float));
    bridge->prev_state = nimcp_calloc(bridge->config.num_dimensions, sizeof(float));

    if (!bridge->encoding_buffer || !bridge->output_buffer ||
        !bridge->anticipation_buffer || !bridge->prev_state) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate buffers in predictive_snn_create");
        predictive_snn_destroy(bridge);
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

    /* Initialize anticipation to neutral */
    bridge->last_anticipation.prediction_level = 0.5f;
    bridge->last_anticipation.error_level = 0.5f;
    bridge->last_anticipation.precision_level = 0.5f;
    bridge->last_anticipation.anticipation_drive = 0.5f;
    bridge->last_anticipation.model_confidence = 0.5f;
    bridge->last_anticipation.error_detected = false;
    bridge->last_anticipation.high_anticipation = false;
    bridge->last_anticipation.anticipation_magnitude = 0.0f;
    bridge->last_anticipation.free_energy_level = 0.5f;
    bridge->last_anticipation.expectation_strength = 0.5f;

    bridge->state = PREDICTIVE_SNN_STATE_IDLE;
    bridge->current_time_us = 0;
    bridge->bio_async_connected = false;
    bridge->error_signal = 0.0f;
    bridge->precision_signal = 0.0f;

    return bridge;
}

void predictive_snn_destroy(predictive_snn_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->snn) {
        snn_network_destroy(bridge->snn);
    }

    bridge_base_cleanup(&bridge->base);

    nimcp_free(bridge->encoding_buffer);
    nimcp_free(bridge->output_buffer);
    nimcp_free(bridge->anticipation_buffer);
    nimcp_free(bridge->prev_state);
    nimcp_free(bridge);
}

int predictive_snn_reset(predictive_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

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

    /* Reset anticipation */
    memset(&bridge->last_anticipation, 0, sizeof(predictive_anticipation_t));
    bridge->last_anticipation.prediction_level = 0.5f;
    bridge->last_anticipation.error_level = 0.5f;
    bridge->last_anticipation.anticipation_drive = 0.5f;

    /* Reset buffers */
    uint32_t input_dim = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
    memset(bridge->encoding_buffer, 0, input_dim * sizeof(float));
    memset(bridge->output_buffer, 0, 6 * sizeof(float));
    memset(bridge->anticipation_buffer, 0, bridge->config.num_dimensions * sizeof(float));
    memset(bridge->prev_state, 0, bridge->config.num_dimensions * sizeof(float));

    bridge->state = PREDICTIVE_SNN_STATE_IDLE;
    bridge->error_signal = 0.0f;
    bridge->precision_signal = 0.0f;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

//=============================================================================
// Encoding Functions
//=============================================================================

int predictive_snn_encode_state(
    predictive_snn_bridge_t* bridge,
    const float* dimensions,
    uint32_t num_dims
) {
    if (!bridge || !dimensions) return -1;
    if (num_dims == 0 || num_dims > bridge->config.num_dimensions) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = PREDICTIVE_SNN_STATE_ENCODING;

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

    nimcp_mutex_unlock(bridge->base.mutex);
    return total_spikes;
}

int predictive_snn_encode_error(
    predictive_snn_bridge_t* bridge,
    float error,
    float precision
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    float dims[PREDICTIVE_DIM_COUNT] = {0};
    dims[PREDICTIVE_DIM_ERROR] = clamp_f(error, 0.0f, 1.0f);
    dims[PREDICTIVE_DIM_PRECISION] = clamp_f(precision, 0.0f, 1.0f);
    dims[PREDICTIVE_DIM_ANTICIPATION] = (1.0f - error) * precision;

    bridge->error_signal = error;
    bridge->precision_signal = precision;

    nimcp_mutex_unlock(bridge->base.mutex);

    return predictive_snn_encode_state(bridge, dims, 3);
}

int predictive_snn_encode_model_state(
    predictive_snn_bridge_t* bridge,
    float model_state,
    uint32_t hierarchy_level
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    float dims[PREDICTIVE_DIM_COUNT] = {0};
    dims[PREDICTIVE_DIM_MODEL_STATE] = clamp_f(model_state, 0.0f, 1.0f);
    dims[PREDICTIVE_DIM_HIERARCHY_LEVEL] = clamp_f((float)hierarchy_level / 10.0f, 0.0f, 1.0f);

    nimcp_mutex_unlock(bridge->base.mutex);

    return predictive_snn_encode_state(bridge, dims, 2);
}

int predictive_snn_encode_free_energy(
    predictive_snn_bridge_t* bridge,
    float free_energy,
    uint32_t energy_type
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    float dims[PREDICTIVE_DIM_COUNT] = {0};
    dims[PREDICTIVE_DIM_FREE_ENERGY] = clamp_f(free_energy, 0.0f, 1.0f);
    dims[PREDICTIVE_DIM_SURPRISE] = clamp_f(free_energy * 0.8f, 0.0f, 1.0f);

    if (free_energy > bridge->config.error_threshold) {
        bridge->last_anticipation.high_anticipation = true;
        bridge->last_anticipation.anticipation_magnitude = free_energy;
        bridge->stats.high_anticipation_events++;

        if (bridge->high_anticipation_callback) {
            bridge->high_anticipation_callback(bridge, free_energy, energy_type,
                                     bridge->high_anticipation_callback_data);
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return predictive_snn_encode_state(bridge, dims, 2);
}

//=============================================================================
// Simulation Functions
//=============================================================================

int predictive_snn_simulate(predictive_snn_bridge_t* bridge, float duration_ms) {
    if (!bridge) return -1;
    if (duration_ms <= 0.0f) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = PREDICTIVE_SNN_STATE_SIMULATING;

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
    bridge->last_anticipation.prediction_level = clamp_f(bridge->output_buffer[0], 0.0f, 1.0f);
    bridge->last_anticipation.error_level = clamp_f(bridge->output_buffer[1], 0.0f, 1.0f);
    bridge->last_anticipation.precision_level = clamp_f(bridge->output_buffer[2], 0.0f, 1.0f);
    bridge->last_anticipation.anticipation_drive = clamp_f(bridge->output_buffer[3], 0.0f, 1.0f);
    bridge->last_anticipation.anticipation_magnitude = clamp_f(bridge->output_buffer[4], 0.0f, 1.0f);
    bridge->last_anticipation.free_energy_level = clamp_f(bridge->output_buffer[5], 0.0f, 1.0f);

    /* Check error threshold */
    if (bridge->last_anticipation.error_level > bridge->config.error_threshold) {
        bridge->last_anticipation.error_detected = true;
        bridge->stats.error_detections++;

        if (bridge->error_callback) {
            bridge->error_callback(bridge, bridge->last_anticipation.error_level,
                                    bridge->current_time_us, bridge->error_callback_data);
        }
    } else {
        bridge->last_anticipation.error_detected = false;
    }

    bridge->stats.total_evaluations++;
    bridge->state = PREDICTIVE_SNN_STATE_IDLE;

    /* Invoke anticipation callback */
    if (bridge->anticipation_callback) {
        bridge->anticipation_callback(bridge, &bridge->last_anticipation, bridge->anticipation_callback_data);
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_snn_step(predictive_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    return predictive_snn_simulate(bridge, bridge->config.dt_ms);
}

int predictive_snn_forward(
    predictive_snn_bridge_t* bridge,
    const float* inputs,
    uint32_t input_count
) {
    if (!bridge || !inputs) return -1;

    int spike_count = predictive_snn_encode_state(bridge, inputs, input_count);
    if (spike_count < 0) return -1;

    if (predictive_snn_simulate(bridge, bridge->config.encoding_window_ms) < 0) {
        return -1;
    }

    return spike_count;
}

//=============================================================================
// Decoding Functions
//=============================================================================

int predictive_snn_get_anticipation(
    predictive_snn_bridge_t* bridge,
    predictive_anticipation_t* anticipation
) {
    if (!bridge || !anticipation) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    *anticipation = bridge->last_anticipation;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int predictive_snn_get_activations(
    predictive_snn_bridge_t* bridge,
    float* activations,
    uint32_t num_dims
) {
    if (!bridge || !activations) return -1;
    if (num_dims == 0 || num_dims > bridge->config.num_dimensions) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    for (uint32_t d = 0; d < num_dims; d++) {
        activations[d] = bridge->dim_states[d].activation;
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

bool predictive_snn_check_error(
    predictive_snn_bridge_t* bridge,
    float* error_level
) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->base.mutex);
    float level = bridge->last_anticipation.error_level;
    if (error_level) {
        *error_level = level;
    }
    bool detected = level > bridge->config.error_threshold;
    nimcp_mutex_unlock(bridge->base.mutex);

    return detected;
}

bool predictive_snn_check_anticipation(
    predictive_snn_bridge_t* bridge,
    float* anticipation_level
) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->base.mutex);
    float level = bridge->last_anticipation.anticipation_magnitude;
    if (anticipation_level) {
        *anticipation_level = level;
    }
    bool detected = level > bridge->config.error_threshold;
    nimcp_mutex_unlock(bridge->base.mutex);

    return detected;
}

bool predictive_snn_check_state_change(
    predictive_snn_bridge_t* bridge,
    float* change_magnitude
) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->base.mutex);
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
    nimcp_mutex_unlock(bridge->base.mutex);

    return changed;
}

//=============================================================================
// State Query Functions
//=============================================================================

int predictive_snn_get_dim_state(
    predictive_snn_bridge_t* bridge,
    uint32_t dim,
    predictive_dim_state_t* state
) {
    if (!bridge || !state) return -1;
    if (dim >= bridge->config.num_dimensions) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->dim_states[dim];
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int predictive_snn_get_state(
    predictive_snn_bridge_t* bridge,
    predictive_snn_bridge_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    state->state = bridge->state;
    state->mean_anticipation = bridge->last_anticipation.anticipation_drive;
    state->error_signal = bridge->error_signal;
    state->precision_signal = bridge->precision_signal;

    /* Count active dimensions */
    state->active_dimensions = 0;
    state->total_activity = 0.0f;
    for (uint32_t d = 0; d < bridge->config.num_dimensions; d++) {
        if (bridge->dim_states[d].activation > 0.1f) {
            state->active_dimensions++;
        }
        state->total_activity += bridge->dim_states[d].activation;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_snn_get_stats(predictive_snn_bridge_t* bridge, predictive_snn_stats_t* stats) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int predictive_snn_reset_stats(predictive_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(predictive_snn_stats_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float predictive_snn_get_anticipation_level(predictive_snn_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(bridge->base.mutex);
    float anticipation = bridge->last_anticipation.anticipation_drive;
    nimcp_mutex_unlock(bridge->base.mutex);

    return anticipation;
}

float predictive_snn_get_total_activity(predictive_snn_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(bridge->base.mutex);
    float total = 0.0f;
    for (uint32_t d = 0; d < bridge->config.num_dimensions; d++) {
        total += bridge->dim_states[d].activation;
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return total;
}

//=============================================================================
// Callback Registration
//=============================================================================

int predictive_snn_register_error_callback(
    predictive_snn_bridge_t* bridge,
    predictive_snn_error_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->error_callback = callback;
    bridge->error_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int predictive_snn_register_anticipation_callback(
    predictive_snn_bridge_t* bridge,
    predictive_snn_anticipation_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->anticipation_callback = callback;
    bridge->anticipation_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int predictive_snn_register_high_anticipation_callback(
    predictive_snn_bridge_t* bridge,
    predictive_snn_high_anticipation_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->high_anticipation_callback = callback;
    bridge->high_anticipation_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int predictive_snn_bio_async_connect(predictive_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    /* Bio-async connection would be implemented here */
    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int predictive_snn_bio_async_disconnect(predictive_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

bool predictive_snn_is_bio_async_connected(predictive_snn_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->base.mutex);
    bool connected = bridge->bio_async_connected;
    nimcp_mutex_unlock(bridge->base.mutex);

    return connected;
}
