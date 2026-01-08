/**
 * @file nimcp_autobio_snn_bridge.c
 * @brief Autobiographical Memory - SNN Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/autobiographical_memory/nimcp_autobio_snn_bridge.h"
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

struct autobio_snn_bridge {
    autobio_snn_config_t config;
    snn_network_t* snn;
    nimcp_mutex_t* mutex;

    /* State */
    autobio_snn_state_t state;
    uint64_t current_time_us;
    bool bio_async_connected;

    /* Dimension state */
    autobio_dim_state_t dim_states[AUTOBIO_SNN_MAX_DIMENSIONS];

    /* Buffers */
    float* encoding_buffer;
    float* output_buffer;
    float* recall_buffer;

    /* Recall state */
    autobio_recall_t last_recall;
    float temporal_signal;
    float emotional_signal;

    /* Previous state for change detection */
    float* prev_state;

    /* Callbacks */
    autobio_snn_recall_callback_t recall_callback;
    void* recall_callback_data;
    autobio_snn_encoded_callback_t encoded_callback;
    void* encoded_callback_data;
    autobio_snn_emotional_callback_t emotional_callback;
    void* emotional_callback_data;

    /* Statistics */
    autobio_snn_stats_t stats;
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

autobio_snn_config_t autobio_snn_config_default(void) {
    autobio_snn_config_t config = {
        .num_dimensions = AUTOBIO_DIM_COUNT,
        .neurons_per_dim = AUTOBIO_SNN_NEURONS_PER_DIM,
        .hidden_dim = 128,

        .dt_ms = 1.0f,
        .encoding_window_ms = AUTOBIO_SNN_ENCODING_WINDOW,
        .integration_tau_ms = 100.0f,

        .encoding = AUTOBIO_SNN_ENCODE_POPULATION,
        .encoding_gain = 1.0f,
        .baseline_rate_hz = 5.0f,
        .max_rate_hz = 100.0f,

        .decoding = AUTOBIO_SNN_DECODE_INTEGRATION,
        .recall_threshold = AUTOBIO_SNN_RECALL_THRESH,
        .vividness_threshold = 0.4f,
        .state_change_threshold = 0.2f,

        .enable_competition = true,
        .inhibition_strength = 0.3f,
        .enable_temporal_context = true,
        .temporal_sensitivity = 1.0f,

        .enable_emotional_modulation = true,
        .emotional_gain = 1.5f,
        .enable_consolidation = true,

        .enable_bio_async = false,
        .enable_plasticity_integration = true
    };
    return config;
}

autobio_snn_bridge_t* autobio_snn_create(const autobio_snn_config_t* config) {
    autobio_snn_bridge_t* bridge = nimcp_calloc(1, sizeof(autobio_snn_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = autobio_snn_config_default();
    }

    /* Validate config */
    if (bridge->config.num_dimensions == 0 ||
        bridge->config.num_dimensions > AUTOBIO_SNN_MAX_DIMENSIONS) {
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
    uint32_t output_dim = 6; /* temporal, valence, intensity, self_relevance, vividness, confidence */

    snn_config_feedforward(&snn_config, input_dim, hidden_dim, output_dim);
    snn_config.dt = bridge->config.dt_ms;
    snn_config.enable_bio_async = bridge->config.enable_bio_async;

    bridge->snn = snn_network_create(&snn_config);
    if (!bridge->snn) {
        nimcp_mutex_destroy(bridge->mutex);
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate buffers */
    bridge->encoding_buffer = nimcp_calloc(input_dim, sizeof(float));
    bridge->output_buffer = nimcp_calloc(output_dim, sizeof(float));
    bridge->recall_buffer = nimcp_calloc(bridge->config.num_dimensions, sizeof(float));
    bridge->prev_state = nimcp_calloc(bridge->config.num_dimensions, sizeof(float));

    if (!bridge->encoding_buffer || !bridge->output_buffer ||
        !bridge->recall_buffer || !bridge->prev_state) {
        autobio_snn_destroy(bridge);
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

    /* Initialize recall to neutral */
    bridge->last_recall.temporal_context = 0.5f;
    bridge->last_recall.emotional_valence = 0.0f;
    bridge->last_recall.emotional_intensity = 0.5f;
    bridge->last_recall.self_relevance = 0.5f;
    bridge->last_recall.vividness = 0.5f;
    bridge->last_recall.recall_confidence = 0.5f;
    bridge->last_recall.recall_successful = false;
    bridge->last_recall.emotional_memory = false;
    bridge->last_recall.importance_level = 0.5f;
    bridge->last_recall.encoding_depth = 0.5f;

    bridge->state = AUTOBIO_SNN_STATE_IDLE;
    bridge->current_time_us = 0;
    bridge->bio_async_connected = false;
    bridge->temporal_signal = 0.0f;
    bridge->emotional_signal = 0.0f;

    return bridge;
}

void autobio_snn_destroy(autobio_snn_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->snn) {
        snn_network_destroy(bridge->snn);
    }

    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge->encoding_buffer);
    nimcp_free(bridge->output_buffer);
    nimcp_free(bridge->recall_buffer);
    nimcp_free(bridge->prev_state);
    nimcp_free(bridge);
}

int autobio_snn_reset(autobio_snn_bridge_t* bridge) {
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

    /* Reset recall */
    memset(&bridge->last_recall, 0, sizeof(autobio_recall_t));
    bridge->last_recall.temporal_context = 0.5f;
    bridge->last_recall.emotional_valence = 0.0f;
    bridge->last_recall.recall_confidence = 0.5f;

    /* Reset buffers */
    uint32_t input_dim = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
    memset(bridge->encoding_buffer, 0, input_dim * sizeof(float));
    memset(bridge->output_buffer, 0, 6 * sizeof(float));
    memset(bridge->recall_buffer, 0, bridge->config.num_dimensions * sizeof(float));
    memset(bridge->prev_state, 0, bridge->config.num_dimensions * sizeof(float));

    bridge->state = AUTOBIO_SNN_STATE_IDLE;
    bridge->temporal_signal = 0.0f;
    bridge->emotional_signal = 0.0f;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Encoding Functions
//=============================================================================

int autobio_snn_encode_state(
    autobio_snn_bridge_t* bridge,
    const float* dimensions,
    uint32_t num_dims
) {
    if (!bridge || !dimensions) return -1;
    if (num_dims == 0 || num_dims > bridge->config.num_dimensions) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = AUTOBIO_SNN_STATE_ENCODING;

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
    bridge->stats.total_encodings++;

    nimcp_mutex_unlock(bridge->mutex);
    return total_spikes;
}

int autobio_snn_encode_episodic(
    autobio_snn_bridge_t* bridge,
    float importance,
    float self_relevance,
    float vividness
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float dims[AUTOBIO_DIM_COUNT] = {0};
    dims[AUTOBIO_DIM_IMPORTANCE] = clamp_f(importance, 0.0f, 1.0f);
    dims[AUTOBIO_DIM_SELF_RELEVANCE] = clamp_f(self_relevance, 0.0f, 1.0f);
    dims[AUTOBIO_DIM_VIVIDNESS] = clamp_f(vividness, 0.0f, 1.0f);
    dims[AUTOBIO_DIM_ENCODING_DEPTH] = (importance + vividness) / 2.0f;

    nimcp_mutex_unlock(bridge->mutex);

    return autobio_snn_encode_state(bridge, dims, 4);
}

int autobio_snn_encode_temporal(
    autobio_snn_bridge_t* bridge,
    float recency,
    uint64_t temporal_tag
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float dims[AUTOBIO_DIM_COUNT] = {0};
    dims[AUTOBIO_DIM_TEMPORAL_CONTEXT] = clamp_f(recency, 0.0f, 1.0f);
    dims[AUTOBIO_DIM_RETRIEVAL_STRENGTH] = recency * 0.8f; /* Recent = easier to retrieve */

    bridge->temporal_signal = recency;
    (void)temporal_tag; /* Could be used for temporal ordering */

    nimcp_mutex_unlock(bridge->mutex);

    return autobio_snn_encode_state(bridge, dims, 2);
}

int autobio_snn_encode_emotional(
    autobio_snn_bridge_t* bridge,
    float valence,
    float intensity,
    float arousal
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    float dims[AUTOBIO_DIM_COUNT] = {0};
    /* Convert valence from [-1,1] to [0,1] for encoding */
    dims[AUTOBIO_DIM_EMOTIONAL_VALENCE] = clamp_f((valence + 1.0f) / 2.0f, 0.0f, 1.0f);
    dims[AUTOBIO_DIM_EMOTIONAL_INTENSITY] = clamp_f(intensity, 0.0f, 1.0f);
    dims[AUTOBIO_DIM_AROUSAL] = clamp_f(arousal, 0.0f, 1.0f);

    bridge->emotional_signal = intensity;

    /* Check for emotional memory */
    if (intensity > bridge->config.vividness_threshold) {
        bridge->last_recall.emotional_memory = true;
        bridge->stats.emotional_memories++;

        if (bridge->emotional_callback) {
            bridge->emotional_callback(bridge, intensity, valence,
                                       bridge->emotional_callback_data);
        }
    }

    nimcp_mutex_unlock(bridge->mutex);

    return autobio_snn_encode_state(bridge, dims, 3);
}

//=============================================================================
// Simulation Functions
//=============================================================================

int autobio_snn_simulate(autobio_snn_bridge_t* bridge, float duration_ms) {
    if (!bridge) return -1;
    if (duration_ms <= 0.0f) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = AUTOBIO_SNN_STATE_SIMULATING;

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
    bridge->last_recall.temporal_context = clamp_f(bridge->output_buffer[0], 0.0f, 1.0f);
    /* Convert back from [0,1] to [-1,1] for valence */
    bridge->last_recall.emotional_valence = clamp_f(bridge->output_buffer[1] * 2.0f - 1.0f, -1.0f, 1.0f);
    bridge->last_recall.emotional_intensity = clamp_f(bridge->output_buffer[2], 0.0f, 1.0f);
    bridge->last_recall.self_relevance = clamp_f(bridge->output_buffer[3], 0.0f, 1.0f);
    bridge->last_recall.vividness = clamp_f(bridge->output_buffer[4], 0.0f, 1.0f);
    bridge->last_recall.recall_confidence = clamp_f(bridge->output_buffer[5], 0.0f, 1.0f);

    /* Check recall threshold */
    if (bridge->last_recall.vividness > bridge->config.recall_threshold) {
        bridge->last_recall.recall_successful = true;
        bridge->stats.successful_recalls++;

        if (bridge->recall_callback) {
            bridge->recall_callback(bridge, bridge->last_recall.vividness,
                                   bridge->current_time_us, bridge->recall_callback_data);
        }
    } else {
        bridge->last_recall.recall_successful = false;
    }

    bridge->state = AUTOBIO_SNN_STATE_IDLE;

    /* Invoke encoded callback */
    if (bridge->encoded_callback) {
        bridge->encoded_callback(bridge, &bridge->last_recall, bridge->encoded_callback_data);
    }

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int autobio_snn_step(autobio_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    return autobio_snn_simulate(bridge, bridge->config.dt_ms);
}

int autobio_snn_forward(
    autobio_snn_bridge_t* bridge,
    const float* inputs,
    uint32_t input_count
) {
    if (!bridge || !inputs) return -1;

    int spike_count = autobio_snn_encode_state(bridge, inputs, input_count);
    if (spike_count < 0) return -1;

    if (autobio_snn_simulate(bridge, bridge->config.encoding_window_ms) < 0) {
        return -1;
    }

    return spike_count;
}

//=============================================================================
// Decoding Functions
//=============================================================================

int autobio_snn_get_recall(
    autobio_snn_bridge_t* bridge,
    autobio_recall_t* recall
) {
    if (!bridge || !recall) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *recall = bridge->last_recall;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int autobio_snn_get_activations(
    autobio_snn_bridge_t* bridge,
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

bool autobio_snn_check_recall(
    autobio_snn_bridge_t* bridge,
    float* recall_strength
) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    float strength = bridge->last_recall.vividness;
    if (recall_strength) {
        *recall_strength = strength;
    }
    bool successful = strength > bridge->config.recall_threshold;
    nimcp_mutex_unlock(bridge->mutex);

    return successful;
}

bool autobio_snn_check_emotional(
    autobio_snn_bridge_t* bridge,
    float* emotional_intensity
) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    float intensity = bridge->last_recall.emotional_intensity;
    if (emotional_intensity) {
        *emotional_intensity = intensity;
    }
    bool detected = intensity > bridge->config.vividness_threshold;
    nimcp_mutex_unlock(bridge->mutex);

    return detected;
}

bool autobio_snn_check_state_change(
    autobio_snn_bridge_t* bridge,
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

int autobio_snn_get_dim_state(
    autobio_snn_bridge_t* bridge,
    uint32_t dim,
    autobio_dim_state_t* state
) {
    if (!bridge || !state) return -1;
    if (dim >= bridge->config.num_dimensions) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->dim_states[dim];
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int autobio_snn_get_state(
    autobio_snn_bridge_t* bridge,
    autobio_snn_bridge_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->mutex);

    state->state = bridge->state;
    state->mean_recall_strength = bridge->last_recall.vividness;
    state->temporal_signal = bridge->temporal_signal;
    state->emotional_signal = bridge->emotional_signal;

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

int autobio_snn_get_stats(autobio_snn_bridge_t* bridge, autobio_snn_stats_t* stats) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int autobio_snn_reset_stats(autobio_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(autobio_snn_stats_t));
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

float autobio_snn_get_recall_strength(autobio_snn_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    nimcp_mutex_lock(bridge->mutex);
    float strength = bridge->last_recall.vividness;
    nimcp_mutex_unlock(bridge->mutex);

    return strength;
}

float autobio_snn_get_total_activity(autobio_snn_bridge_t* bridge) {
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

int autobio_snn_register_recall_callback(
    autobio_snn_bridge_t* bridge,
    autobio_snn_recall_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->recall_callback = callback;
    bridge->recall_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int autobio_snn_register_encoded_callback(
    autobio_snn_bridge_t* bridge,
    autobio_snn_encoded_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->encoded_callback = callback;
    bridge->encoded_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int autobio_snn_register_emotional_callback(
    autobio_snn_bridge_t* bridge,
    autobio_snn_emotional_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->emotional_callback = callback;
    bridge->emotional_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int autobio_snn_bio_async_connect(autobio_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return -1;

    nimcp_mutex_lock(bridge->mutex);
    /* Bio-async connection would be implemented here */
    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int autobio_snn_bio_async_disconnect(autobio_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

bool autobio_snn_is_bio_async_connected(autobio_snn_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    bool connected = bridge->bio_async_connected;
    nimcp_mutex_unlock(bridge->mutex);

    return connected;
}
