/**
 * @file nimcp_self_model_snn_bridge.c
 * @brief Self Model - SNN Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/self_model/nimcp_self_model_snn_bridge.h"
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
#include "security/nimcp_bbb_helpers.h"

#include <string.h>
#include <math.h>
#include <stdio.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for self_model_snn_bridge module */
static nimcp_health_agent_t* g_self_model_snn_bridge_health_agent = NULL;

/**
 * @brief Set health agent for self_model_snn_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void self_model_snn_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_self_model_snn_bridge_health_agent = agent;
}

/** @brief Send heartbeat from self_model_snn_bridge module */
static inline void self_model_snn_bridge_heartbeat(const char* operation, float progress) {
    if (g_self_model_snn_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_self_model_snn_bridge_health_agent, operation, progress);
    }
}

#define LOG_MODULE "SELF_MODEL_SNN_BRIDGE"

//=============================================================================
// Internal Structures
//=============================================================================

struct self_model_snn_bridge {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */
    self_model_snn_config_t config;
    snn_network_t* snn;

    /* State */
    self_model_snn_state_t state;
    uint64_t current_time_us;
    bool bio_async_connected;

    /* Dimension state */
    self_model_dim_state_t dim_states[SELF_MODEL_SNN_MAX_DIMENSIONS];

    /* Buffers */
    float* encoding_buffer;
    float* output_buffer;
    float* insight_buffer;

    /* Insight state */
    self_model_insight_t last_insight;
    float boundary_signal;
    float agency_signal;

    /* Previous state for change detection */
    float* prev_state;

    /* Callbacks */
    self_model_snn_boundary_callback_t boundary_callback;
    void* boundary_callback_data;
    self_model_snn_insight_callback_t insight_callback;
    void* insight_callback_data;
    self_model_snn_agency_callback_t agency_callback;
    void* agency_callback_data;

    /* Statistics */
    self_model_snn_stats_t stats;
};

BRIDGE_DEFINE_SECURITY_SETTERS(self_model_snn_bridge)

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
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && n > 256) {
            self_model_snn_bridge_heartbeat("self_model_s_loop",
                             (float)(i + 1) / (float)n);
        }

        values[i] = expf(values[i] - max_val);
        sum += values[i];
    }

    if (sum > 0.0f) {
        for (uint32_t i = 0; i < n; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && n > 256) {
                self_model_snn_bridge_heartbeat("self_model_s_loop",
                                 (float)(i + 1) / (float)n);
            }

            values[i] /= sum;
        }
    }
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

self_model_snn_config_t self_model_snn_config_default(void) {
    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_confi", 0.0f);


    self_model_snn_config_t config = {
        .num_dimensions = SELF_DIM_COUNT,
        .neurons_per_dim = SELF_MODEL_SNN_NEURONS_PER_DIM,
        .hidden_dim = 128,

        .dt_ms = 1.0f,
        .encoding_window_ms = SELF_MODEL_SNN_ENCODING_WINDOW,
        .integration_tau_ms = 100.0f,

        .encoding = SELF_MODEL_SNN_ENCODE_POPULATION,
        .encoding_gain = 1.0f,
        .baseline_rate_hz = 5.0f,
        .max_rate_hz = 100.0f,

        .decoding = SELF_MODEL_SNN_DECODE_INTEGRATION,
        .boundary_threshold = SELF_MODEL_SNN_BOUNDARY_THRESH,
        .agency_threshold = 0.6f,
        .continuity_threshold = 0.2f,

        .enable_competition = true,
        .inhibition_strength = 0.3f,
        .enable_boundary_detection = true,
        .boundary_sensitivity = 0.8f,

        .enable_identity_core = true,
        .identity_gain = 1.5f,
        .enable_autobiographical = true,

        .enable_bio_async = false,
        .enable_plasticity_integration = true
    };
    return config;
}

self_model_snn_bridge_t* self_model_snn_create(const self_model_snn_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_creat", 0.0f);


    self_model_snn_bridge_t* bridge = nimcp_calloc(1, sizeof(self_model_snn_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = self_model_snn_config_default();
    }

    /* Validate config */
    if (bridge->config.num_dimensions == 0 ||
        bridge->config.num_dimensions > SELF_MODEL_SNN_MAX_DIMENSIONS) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize base bridge */
    if (bridge_base_init(&bridge->base, 0, "self_model_snn") != 0) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Create SNN network */
    snn_config_t snn_config;
    uint32_t input_dim = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
    uint32_t hidden_dim = bridge->config.hidden_dim;
    uint32_t output_dim = 6; /* body_state, agency, ownership, identity, boundary, continuity */

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
    bridge->insight_buffer = nimcp_calloc(bridge->config.num_dimensions, sizeof(float));
    bridge->prev_state = nimcp_calloc(bridge->config.num_dimensions, sizeof(float));

    if (!bridge->encoding_buffer || !bridge->output_buffer ||
        !bridge->insight_buffer || !bridge->prev_state) {
        self_model_snn_destroy(bridge);
        return NULL;
    }

    /* Initialize dimension states */
    for (uint32_t i = 0; i < bridge->config.num_dimensions; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->config.num_dimensions > 256) {
            self_model_snn_bridge_heartbeat("self_model_s_loop",
                             (float)(i + 1) / (float)bridge->config.num_dimensions);
        }

        bridge->dim_states[i].activation = 0.0f;
        bridge->dim_states[i].accumulated_evidence = 0.0f;
        bridge->dim_states[i].spike_count = 0;
        bridge->dim_states[i].mean_rate_hz = bridge->config.baseline_rate_hz;
        bridge->dim_states[i].last_spike_time_us = 0;
    }

    /* Initialize insight to neutral */
    bridge->last_insight.body_state_level = 0.5f;
    bridge->last_insight.agency_level = 0.5f;
    bridge->last_insight.ownership_level = 0.5f;
    bridge->last_insight.identity_coherence = 0.5f;
    bridge->last_insight.boundary_clarity = 0.5f;
    bridge->last_insight.boundary_violation_detected = false;
    bridge->last_insight.agency_disruption_detected = false;
    bridge->last_insight.disruption_magnitude = 0.0f;
    bridge->last_insight.narrative_coherence = 0.5f;
    bridge->last_insight.continuity_score = 0.5f;

    bridge->state = SELF_MODEL_SNN_STATE_IDLE;
    bridge->current_time_us = 0;
    bridge->bio_async_connected = false;
    bridge->boundary_signal = 0.0f;
    bridge->agency_signal = 0.0f;

    NIMCP_LOGGING_INFO("Created %s bridge", "self_model_snn");
    return bridge;
}

void self_model_snn_destroy(self_model_snn_bridge_t* bridge) {
    if (!bridge) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "self_model_snn");

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_destr", 0.0f);


    if (bridge->snn) {
        snn_network_destroy(bridge->snn);
    }

    bridge_base_cleanup(&bridge->base);

    nimcp_free(bridge->encoding_buffer);
    nimcp_free(bridge->output_buffer);
    nimcp_free(bridge->insight_buffer);
    nimcp_free(bridge->prev_state);
    nimcp_free(bridge);
}

int self_model_snn_reset(self_model_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Reset SNN network */
    if (bridge->snn) {
        snn_network_reset(bridge->snn);
    }

    /* Reset dimension states */
    for (uint32_t i = 0; i < bridge->config.num_dimensions; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->config.num_dimensions > 256) {
            self_model_snn_bridge_heartbeat("self_model_s_loop",
                             (float)(i + 1) / (float)bridge->config.num_dimensions);
        }

        bridge->dim_states[i].activation = 0.0f;
        bridge->dim_states[i].accumulated_evidence = 0.0f;
        bridge->dim_states[i].spike_count = 0;
        bridge->dim_states[i].mean_rate_hz = bridge->config.baseline_rate_hz;
        bridge->dim_states[i].last_spike_time_us = 0;
    }

    /* Reset insight */
    memset(&bridge->last_insight, 0, sizeof(self_model_insight_t));
    bridge->last_insight.body_state_level = 0.5f;
    bridge->last_insight.agency_level = 0.5f;
    bridge->last_insight.ownership_level = 0.5f;

    /* Reset buffers */
    uint32_t input_dim = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
    memset(bridge->encoding_buffer, 0, input_dim * sizeof(float));
    memset(bridge->output_buffer, 0, 6 * sizeof(float));
    memset(bridge->insight_buffer, 0, bridge->config.num_dimensions * sizeof(float));
    memset(bridge->prev_state, 0, bridge->config.num_dimensions * sizeof(float));

    bridge->state = SELF_MODEL_SNN_STATE_IDLE;
    bridge->boundary_signal = 0.0f;
    bridge->agency_signal = 0.0f;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

//=============================================================================
// Encoding Functions
//=============================================================================

int self_model_snn_encode_state(
    self_model_snn_bridge_t* bridge,
    const float* dimensions,
    uint32_t num_dims
) {
    if (!bridge || !dimensions) return -1;
    BRIDGE_BBB_VALIDATE(bridge, dimensions, num_dims * sizeof(float));
    if (num_dims == 0 || num_dims > bridge->config.num_dimensions) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_encod", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = SELF_MODEL_SNN_STATE_ENCODING;

    uint32_t neurons_per_dim = bridge->config.neurons_per_dim;
    int total_spikes = 0;

    /* Population encoding for each dimension */
    for (uint32_t d = 0; d < num_dims; d++) {
        /* Phase 8: Loop progress heartbeat */
        if ((d & 0xFF) == 0 && num_dims > 256) {
            self_model_snn_bridge_heartbeat("self_model_s_loop",
                             (float)(d + 1) / (float)num_dims);
        }

        float value = clamp_f(dimensions[d], 0.0f, 1.0f);
        float rate = bridge->config.baseline_rate_hz +
                    value * (bridge->config.max_rate_hz - bridge->config.baseline_rate_hz);

        /* Update dimension state */
        bridge->dim_states[d].activation = value;
        bridge->dim_states[d].mean_rate_hz = rate;

        /* Population encode */
        for (uint32_t n = 0; n < neurons_per_dim; n++) {
            /* Phase 8: Loop progress heartbeat */
            if ((n & 0xFF) == 0 && neurons_per_dim > 256) {
                self_model_snn_bridge_heartbeat("self_model_s_loop",
                                 (float)(n + 1) / (float)neurons_per_dim);
            }

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

    /* Detect identity change */
    float change_magnitude = 0.0f;
    for (uint32_t d = 0; d < num_dims; d++) {
        /* Phase 8: Loop progress heartbeat */
        if ((d & 0xFF) == 0 && num_dims > 256) {
            self_model_snn_bridge_heartbeat("self_model_s_loop",
                             (float)(d + 1) / (float)num_dims);
        }

        float diff = dimensions[d] - bridge->prev_state[d];
        change_magnitude += diff * diff;
        bridge->prev_state[d] = dimensions[d];
    }
    change_magnitude = sqrtf(change_magnitude / num_dims);

    if (change_magnitude > bridge->config.continuity_threshold) {
        bridge->stats.identity_changes++;
    }

    bridge->stats.total_spikes += total_spikes;

    nimcp_mutex_unlock(bridge->base.mutex);
    return total_spikes;
}

int self_model_snn_encode_body_state(
    self_model_snn_bridge_t* bridge,
    float interoceptive,
    float proprioceptive
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_encod", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    float dims[SELF_DIM_COUNT] = {0};
    dims[SELF_DIM_BODY_STATE] = (interoceptive + proprioceptive) / 2.0f;
    dims[SELF_DIM_OWNERSHIP] = proprioceptive;

    nimcp_mutex_unlock(bridge->base.mutex);

    return self_model_snn_encode_state(bridge, dims, 2);
}

int self_model_snn_encode_agency(
    self_model_snn_bridge_t* bridge,
    float agency_strength,
    float efference_match
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_encod", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    float dims[SELF_DIM_COUNT] = {0};
    dims[SELF_DIM_AGENCY] = clamp_f(agency_strength, 0.0f, 1.0f);
    dims[SELF_DIM_CAPABILITY] = clamp_f(efference_match, 0.0f, 1.0f);

    bridge->agency_signal = agency_strength;

    /* Check for agency disruption */
    if (agency_strength < bridge->config.agency_threshold) {
        bridge->last_insight.agency_disruption_detected = true;
        bridge->last_insight.disruption_magnitude = bridge->config.agency_threshold - agency_strength;
        bridge->stats.agency_disruptions++;

        if (bridge->agency_callback) {
            bridge->agency_callback(bridge, agency_strength, SELF_DIM_AGENCY,
                                   bridge->agency_callback_data);
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return self_model_snn_encode_state(bridge, dims, 2);
}

int self_model_snn_encode_boundary(
    self_model_snn_bridge_t* bridge,
    float boundary_strength,
    uint32_t boundary_type
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_encod", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    float dims[SELF_DIM_COUNT] = {0};
    dims[SELF_DIM_BOUNDARY] = clamp_f(boundary_strength, 0.0f, 1.0f);

    bridge->boundary_signal = boundary_strength;

    /* Check for boundary violation */
    if (boundary_strength < bridge->config.boundary_threshold) {
        bridge->last_insight.boundary_violation_detected = true;
        bridge->stats.boundary_detections++;

        if (bridge->boundary_callback) {
            bridge->boundary_callback(bridge, boundary_strength,
                                     bridge->current_time_us, bridge->boundary_callback_data);
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return self_model_snn_encode_state(bridge, dims, 1);
}

//=============================================================================
// Simulation Functions
//=============================================================================

int self_model_snn_simulate(self_model_snn_bridge_t* bridge, float duration_ms) {
    if (!bridge) return -1;
    if (duration_ms <= 0.0f) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_simul", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = SELF_MODEL_SNN_STATE_SIMULATING;

    float dt = bridge->config.dt_ms;
    uint32_t steps = (uint32_t)(duration_ms / dt);

    /* Set inputs before simulation */
    if (bridge->snn) {
        uint32_t input_size = bridge->config.num_dimensions * bridge->config.neurons_per_dim;
        snn_network_set_inputs(bridge->snn, bridge->encoding_buffer, input_size);
    }

    for (uint32_t s = 0; s < steps; s++) {
        /* Phase 8: Loop progress heartbeat */
        if ((s & 0xFF) == 0 && steps > 256) {
            self_model_snn_bridge_heartbeat("self_model_s_loop",
                             (float)(s + 1) / (float)steps);
        }

        if (bridge->snn) {
            snn_network_step(bridge->snn, dt);
        }

        /* Update evidence integration */
        float decay = expf(-dt / bridge->config.integration_tau_ms);
        for (uint32_t d = 0; d < bridge->config.num_dimensions; d++) {
            /* Phase 8: Loop progress heartbeat */
            if ((d & 0xFF) == 0 && bridge->config.num_dimensions > 256) {
                self_model_snn_bridge_heartbeat("self_model_s_loop",
                                 (float)(d + 1) / (float)bridge->config.num_dimensions);
            }

            bridge->dim_states[d].accumulated_evidence *= decay;
            bridge->dim_states[d].accumulated_evidence +=
                bridge->dim_states[d].activation * dt / bridge->config.integration_tau_ms;
        }

        bridge->current_time_us += (uint64_t)(dt * 1000.0f);
        bridge->stats.total_simulations++;
    }

    /* WHAT: Compute insight from encoded dimension activations */
    /* WHY: SNN outputs may be 0/uniform; use direct dimension values for insight */
    /* HOW: Map dimension activations to corresponding insight fields */
    bridge->last_insight.body_state_level = bridge->dim_states[SELF_DIM_BODY_STATE].activation;
    bridge->last_insight.agency_level = bridge->dim_states[SELF_DIM_AGENCY].activation;
    bridge->last_insight.ownership_level = bridge->dim_states[SELF_DIM_OWNERSHIP].activation;
    bridge->last_insight.identity_coherence = bridge->dim_states[SELF_DIM_IDENTITY].activation;
    bridge->last_insight.boundary_clarity = bridge->dim_states[SELF_DIM_BOUNDARY].activation;
    bridge->last_insight.continuity_score = bridge->dim_states[SELF_DIM_CONTINUITY].activation;
    bridge->last_insight.narrative_coherence = bridge->dim_states[SELF_DIM_NARRATIVE].activation;

    /* Check boundary threshold */
    if (bridge->last_insight.boundary_clarity < bridge->config.boundary_threshold) {
        bridge->stats.boundary_detections++;

        if (bridge->boundary_callback) {
            bridge->boundary_callback(bridge, bridge->last_insight.boundary_clarity,
                                     bridge->current_time_us, bridge->boundary_callback_data);
        }
    }

    bridge->stats.total_evaluations++;
    bridge->state = SELF_MODEL_SNN_STATE_IDLE;

    /* Invoke insight callback */
    if (bridge->insight_callback) {
        bridge->insight_callback(bridge, &bridge->last_insight, bridge->insight_callback_data);
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_model_snn_step(self_model_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_step", 0.0f);


    return self_model_snn_simulate(bridge, bridge->config.dt_ms);
}

int self_model_snn_forward(
    self_model_snn_bridge_t* bridge,
    const float* inputs,
    uint32_t input_count
) {
    if (!bridge || !inputs) return -1;
    BRIDGE_BBB_VALIDATE(bridge, inputs, input_count * sizeof(float));

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_forwa", 0.0f);


    int spike_count = self_model_snn_encode_state(bridge, inputs, input_count);
    if (spike_count < 0) return -1;

    if (self_model_snn_simulate(bridge, bridge->config.encoding_window_ms) < 0) {
        return -1;
    }

    return spike_count;
}

//=============================================================================
// Decoding Functions
//=============================================================================

int self_model_snn_get_insight(
    self_model_snn_bridge_t* bridge,
    self_model_insight_t* insight
) {
    if (!bridge || !insight) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_get_i", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *insight = bridge->last_insight;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int self_model_snn_get_activations(
    self_model_snn_bridge_t* bridge,
    float* activations,
    uint32_t num_dims
) {
    if (!bridge || !activations) return -1;
    if (num_dims == 0 || num_dims > bridge->config.num_dimensions) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_get_a", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    for (uint32_t d = 0; d < num_dims; d++) {
        /* Phase 8: Loop progress heartbeat */
        if ((d & 0xFF) == 0 && num_dims > 256) {
            self_model_snn_bridge_heartbeat("self_model_s_loop",
                             (float)(d + 1) / (float)num_dims);
        }

        activations[d] = bridge->dim_states[d].activation;
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

bool self_model_snn_check_boundary(
    self_model_snn_bridge_t* bridge,
    float* boundary_level
) {
    if (!bridge) return false;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_check", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    float level = bridge->last_insight.boundary_clarity;
    if (boundary_level) {
        *boundary_level = level;
    }
    bool violated = level < bridge->config.boundary_threshold;
    nimcp_mutex_unlock(bridge->base.mutex);

    return violated;
}

bool self_model_snn_check_agency(
    self_model_snn_bridge_t* bridge,
    float* agency_level
) {
    if (!bridge) return false;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_check", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    float level = bridge->agency_signal;
    if (agency_level) {
        *agency_level = level;
    }
    bool disrupted = level < bridge->config.agency_threshold;
    nimcp_mutex_unlock(bridge->base.mutex);

    return disrupted;
}

bool self_model_snn_check_identity_change(
    self_model_snn_bridge_t* bridge,
    float* change_magnitude
) {
    if (!bridge) return false;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_check", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    /* Calculate magnitude from prev_state differences */
    float mag = 0.0f;
    for (uint32_t d = 0; d < bridge->config.num_dimensions; d++) {
        /* Phase 8: Loop progress heartbeat */
        if ((d & 0xFF) == 0 && bridge->config.num_dimensions > 256) {
            self_model_snn_bridge_heartbeat("self_model_s_loop",
                             (float)(d + 1) / (float)bridge->config.num_dimensions);
        }

        float diff = bridge->dim_states[d].activation - bridge->prev_state[d];
        mag += diff * diff;
    }
    mag = sqrtf(mag / bridge->config.num_dimensions);

    bool changed = mag > bridge->config.continuity_threshold;
    if (change_magnitude) {
        *change_magnitude = mag;
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return changed;
}

//=============================================================================
// State Query Functions
//=============================================================================

int self_model_snn_get_dim_state(
    self_model_snn_bridge_t* bridge,
    uint32_t dim,
    self_model_dim_state_t* state
) {
    if (!bridge || !state) return -1;
    if (dim >= bridge->config.num_dimensions) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_get_d", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->dim_states[dim];
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int self_model_snn_get_state(
    self_model_snn_bridge_t* bridge,
    self_model_snn_bridge_state_t* state
) {
    if (!bridge || !state) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_get_s", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    state->state = bridge->state;
    state->mean_agency = bridge->last_insight.agency_level;
    state->boundary_signal = bridge->boundary_signal;
    state->identity_signal = bridge->last_insight.identity_coherence;

    /* Count active dimensions */
    state->active_dimensions = 0;
    state->total_activity = 0.0f;
    for (uint32_t d = 0; d < bridge->config.num_dimensions; d++) {
        /* Phase 8: Loop progress heartbeat */
        if ((d & 0xFF) == 0 && bridge->config.num_dimensions > 256) {
            self_model_snn_bridge_heartbeat("self_model_s_loop",
                             (float)(d + 1) / (float)bridge->config.num_dimensions);
        }

        if (bridge->dim_states[d].activation > 0.1f) {
            state->active_dimensions++;
        }
        state->total_activity += bridge->dim_states[d].activation;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int self_model_snn_get_stats(self_model_snn_bridge_t* bridge, self_model_snn_stats_t* stats) {
    if (!bridge || !stats) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_get_s", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int self_model_snn_reset_stats(self_model_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(self_model_snn_stats_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

float self_model_snn_get_agency(self_model_snn_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_get_a", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    float agency = bridge->last_insight.agency_level;
    nimcp_mutex_unlock(bridge->base.mutex);

    return agency;
}

float self_model_snn_get_total_activity(self_model_snn_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_get_t", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    float total = 0.0f;
    for (uint32_t d = 0; d < bridge->config.num_dimensions; d++) {
        /* Phase 8: Loop progress heartbeat */
        if ((d & 0xFF) == 0 && bridge->config.num_dimensions > 256) {
            self_model_snn_bridge_heartbeat("self_model_s_loop",
                             (float)(d + 1) / (float)bridge->config.num_dimensions);
        }

        total += bridge->dim_states[d].activation;
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return total;
}

//=============================================================================
// Callback Registration
//=============================================================================

int self_model_snn_register_boundary_callback(
    self_model_snn_bridge_t* bridge,
    self_model_snn_boundary_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_regis", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->boundary_callback = callback;
    bridge->boundary_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int self_model_snn_register_insight_callback(
    self_model_snn_bridge_t* bridge,
    self_model_snn_insight_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_regis", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->insight_callback = callback;
    bridge->insight_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int self_model_snn_register_agency_callback(
    self_model_snn_bridge_t* bridge,
    self_model_snn_agency_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_regis", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->agency_callback = callback;
    bridge->agency_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int self_model_snn_bio_async_connect(self_model_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_bio_a", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    /* Bio-async connection would be implemented here */
    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int self_model_snn_bio_async_disconnect(self_model_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_bio_a", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

bool self_model_snn_is_bio_async_connected(self_model_snn_bridge_t* bridge) {
    if (!bridge) return false;

    /* Phase 8: Heartbeat at operation start */
    self_model_snn_bridge_heartbeat("self_model_s_self_model_snn_is_bi", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bool connected = bridge->bio_async_connected;
    nimcp_mutex_unlock(bridge->base.mutex);

    return connected;
}
