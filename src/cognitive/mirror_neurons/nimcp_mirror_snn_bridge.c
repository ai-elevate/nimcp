/**
 * @file nimcp_mirror_snn_bridge.c
 * @brief Mirror Neuron - SNN Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-05
 */

#include "cognitive/mirror_neurons/nimcp_mirror_snn_bridge.h"
#include "core/neuron_types/nimcp_neuron_types.h"
#include "core/synapse_types/nimcp_synapse_types.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#define LOG_MODULE "mirror_snn_bridge"

//=============================================================================
// Internal Structures
//=============================================================================

struct mirror_snn_bridge {
    /* Configuration */
    mirror_snn_config_t config;

    /* SNN network */
    snn_network_t* snn;
    bool owns_snn;                       /**< true if bridge created the SNN */

    /* State tracking */
    mirror_snn_state_t state;
    mirror_snn_action_state_t actions[MIRROR_SNN_MAX_ACTIONS];
    uint32_t num_actions;

    /* Population mapping */
    uint32_t observation_pop;            /**< Observation input population */
    uint32_t execution_pop;              /**< Execution input population */
    uint32_t hidden_pop;                 /**< Hidden layer population */
    uint32_t output_pop;                 /**< Output/recognition population */

    /* Callbacks */
    mirror_snn_spike_callback_t spike_callback;
    void* spike_callback_data;
    mirror_snn_recognition_callback_t recognition_callback;
    void* recognition_callback_data;
    mirror_snn_training_callback_t training_callback;
    void* training_callback_data;
    mirror_snn_health_callback_t health_callback;
    void* health_callback_data;

    /* Bio-async integration */
    bool bio_async_connected;
    bio_module_context_t bio_ctx;

    /* Immune integration */
    void* immune_system;
    bool immune_connected;

    /* Statistics */
    mirror_snn_stats_t stats;

    /* Thread safety */
    nimcp_mutex_t* mutex;

    /* Timing */
    uint64_t last_update_us;
    snn_state_health_t last_health;
};

//=============================================================================
// Helper Functions
//=============================================================================

static inline float clamp_f(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static uint64_t get_time_us(void) {
    return nimcp_time_get_us();
}

static void init_action_state(mirror_snn_action_state_t* action, uint32_t id) {
    memset(action, 0, sizeof(*action));
    action->action_id = id;
    action->status = MIRROR_SNN_ACTION_INACTIVE;
}

//=============================================================================
// Bio-Async Message Handlers
//=============================================================================

static int handle_spike_event(
    snn_network_t* network,
    snn_bio_msg_type_t type,
    const void* msg,
    size_t msg_size,
    void* user_data
) {
    mirror_snn_bridge_t* bridge = (mirror_snn_bridge_t*)user_data;
    if (!bridge || !msg || type != SNN_BIO_MSG_SPIKE_EVENT) return -1;

    const snn_bio_spike_msg_t* spike = (const snn_bio_spike_msg_t*)msg;

    /* Update statistics */
    bridge->stats.bio_messages_received++;

    /* Invoke user callback if registered */
    if (bridge->spike_callback) {
        bridge->spike_callback(
            spike->population_id,
            spike->neuron_id,
            spike->spike_time,
            bridge->spike_callback_data
        );
    }

    return 0;
}

static int handle_training_event(
    snn_network_t* network,
    snn_bio_msg_type_t type,
    const void* msg,
    size_t msg_size,
    void* user_data
) {
    mirror_snn_bridge_t* bridge = (mirror_snn_bridge_t*)user_data;
    if (!bridge || !msg || type != SNN_BIO_MSG_STDP_EVENT) return -1;

    const snn_bio_stdp_msg_t* stdp = (const snn_bio_stdp_msg_t*)msg;

    /* Update statistics */
    bridge->stats.stdp_events++;
    bridge->stats.avg_weight_change =
        0.99f * bridge->stats.avg_weight_change + 0.01f * fabsf(stdp->delta_w);

    /* Invoke user callback if registered */
    if (bridge->training_callback) {
        bridge->training_callback(
            stdp->pre_id,  /* Use as synapse identifier */
            stdp->delta_w,
            stdp->new_weight,
            bridge->training_callback_data
        );
    }

    return 0;
}

static int handle_population_activity(
    snn_network_t* network,
    snn_bio_msg_type_t type,
    const void* msg,
    size_t msg_size,
    void* user_data
) {
    mirror_snn_bridge_t* bridge = (mirror_snn_bridge_t*)user_data;
    if (!bridge || !msg || type != SNN_BIO_MSG_POPULATION_ACTIVITY) return -1;

    const snn_bio_population_msg_t* pop = (const snn_bio_population_msg_t*)msg;

    /* Update action state if this is the output population */
    if (pop->population_id == bridge->output_pop) {
        /* Find or create action entry */
        for (uint32_t i = 0; i < bridge->num_actions; i++) {
            if (bridge->actions[i].action_id == pop->population_id) {
                bridge->actions[i].spike_count = pop->n_active;
                bridge->actions[i].last_spike_us = get_time_us();
                break;
            }
        }
    }

    return 0;
}

//=============================================================================
// Lifecycle API Implementation
//=============================================================================

mirror_snn_config_t mirror_snn_config_default(void) {
    mirror_snn_config_t config = {
        .input_dim = MIRROR_SNN_INPUT_DIM,
        .hidden_dim = 256,
        .output_dim = MIRROR_SNN_OUTPUT_DIM,
        .neurons_per_action = MIRROR_SNN_NEURONS_PER_ACTION,

        .encoding_method = MIRROR_SNN_ENCODE_POPULATION,
        .encoding_gain = 1.0f,
        .encoding_threshold = 0.1f,
        .encoding_window_ms = MIRROR_SNN_ENCODING_WINDOW,

        .dt_ms = MIRROR_SNN_DEFAULT_DT,
        .simulation_duration_ms = 100.0f,
        .enable_recurrence = true,

        .enable_training = true,
        .learning_rate = 0.001f,
        .enable_reward_modulation = true,

        .decoding_threshold = 0.2f,
        .confidence_gain = 1.0f,

        .enable_bio_async = true,
        .bio_async_priority = 1,

        .enable_immune_integration = true,
        .update_interval_ms = 10.0f
    };
    return config;
}

mirror_snn_bridge_t* mirror_snn_create(const mirror_snn_config_t* config) {
    mirror_snn_bridge_t* bridge = nimcp_calloc(1, sizeof(mirror_snn_bridge_t));
    if (!bridge) {
        NIMCP_LOG_ERROR(LOG_MODULE, "Failed to allocate bridge");
        return NULL;
    }

    /* Copy configuration */
    bridge->config = config ? *config : mirror_snn_config_default();

    /* Create SNN network */
    snn_config_t snn_config;
    snn_config_feedforward(&snn_config,
        bridge->config.input_dim,
        bridge->config.hidden_dim,
        bridge->config.output_dim);

    snn_config.dt = bridge->config.dt_ms;
    snn_config.enable_bio_async = bridge->config.enable_bio_async;
    /* Allow additional populations (default from feedforward is only 3) */
    snn_config.n_populations = 0;  /* Use SNN_MAX_POPULATIONS (64) */

    bridge->snn = snn_network_create(&snn_config);
    if (!bridge->snn) {
        NIMCP_LOG_ERROR(LOG_MODULE, "Failed to create SNN network");
        nimcp_free(bridge);
        return NULL;
    }
    bridge->owns_snn = true;

    /* Add populations for observation and execution pathways */
    bridge->observation_pop = snn_network_add_population(
        bridge->snn, bridge->config.input_dim, NEURON_GENERIC_LIF, "observation");
    bridge->execution_pop = snn_network_add_population(
        bridge->snn, bridge->config.input_dim, NEURON_GENERIC_LIF, "execution");
    bridge->hidden_pop = snn_network_add_population(
        bridge->snn, bridge->config.hidden_dim, NEURON_GENERIC_LIF, "hidden");
    bridge->output_pop = snn_network_add_population(
        bridge->snn, bridge->config.output_dim, NEURON_GENERIC_LIF, "output");

    /* Connect populations */
    snn_network_connect_populations(bridge->snn, bridge->observation_pop,
        bridge->hidden_pop, SNN_TOPO_RANDOM, 0.3f, SYNAPSE_AMPA, 0.5f, 0.1f);
    snn_network_connect_populations(bridge->snn, bridge->execution_pop,
        bridge->hidden_pop, SNN_TOPO_RANDOM, 0.3f, SYNAPSE_AMPA, 0.5f, 0.1f);
    snn_network_connect_populations(bridge->snn, bridge->hidden_pop,
        bridge->output_pop, SNN_TOPO_FEEDFORWARD, 1.0f, SYNAPSE_AMPA, 0.5f, 0.1f);

    /* Add recurrent connections if enabled */
    if (bridge->config.enable_recurrence) {
        snn_network_connect_populations(bridge->snn, bridge->hidden_pop,
            bridge->hidden_pop, SNN_TOPO_RANDOM, 0.1f, SYNAPSE_AMPA, 0.3f, 0.1f);
    }

    /* Create mutex */
    mutex_attr_t attr = { .type = MUTEX_TYPE_NORMAL };
    bridge->mutex = nimcp_mutex_create(&attr);
    if (!bridge->mutex) {
        NIMCP_LOG_ERROR(LOG_MODULE, "Failed to create mutex");
        snn_network_destroy(bridge->snn);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->state = MIRROR_SNN_STATE_IDLE;
    bridge->last_health = SNN_STATE_HEALTHY;
    bridge->last_update_us = get_time_us();

    /* Initialize action states */
    for (uint32_t i = 0; i < MIRROR_SNN_MAX_ACTIONS; i++) {
        init_action_state(&bridge->actions[i], i);
    }

    NIMCP_LOG_INFO(LOG_MODULE, "Created mirror-SNN bridge (input=%u, hidden=%u, output=%u)",
        bridge->config.input_dim, bridge->config.hidden_dim, bridge->config.output_dim);

    return bridge;
}

mirror_snn_bridge_t* mirror_snn_create_with_network(
    const mirror_snn_config_t* config,
    snn_network_t* snn
) {
    if (!snn) {
        NIMCP_LOG_ERROR(LOG_MODULE, "NULL SNN network provided");
        return NULL;
    }

    mirror_snn_bridge_t* bridge = nimcp_calloc(1, sizeof(mirror_snn_bridge_t));
    if (!bridge) {
        NIMCP_LOG_ERROR(LOG_MODULE, "Failed to allocate bridge");
        return NULL;
    }

    bridge->config = config ? *config : mirror_snn_config_default();
    bridge->snn = snn;
    bridge->owns_snn = false;  /* Do NOT destroy on cleanup */

    /* Create mutex */
    mutex_attr_t attr = { .type = MUTEX_TYPE_NORMAL };
    bridge->mutex = nimcp_mutex_create(&attr);
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->state = MIRROR_SNN_STATE_IDLE;
    bridge->last_health = SNN_STATE_HEALTHY;
    bridge->last_update_us = get_time_us();

    for (uint32_t i = 0; i < MIRROR_SNN_MAX_ACTIONS; i++) {
        init_action_state(&bridge->actions[i], i);
    }

    return bridge;
}

void mirror_snn_destroy(mirror_snn_bridge_t* bridge) {
    if (!bridge) return;

    /* Disconnect integrations */
    mirror_snn_disconnect_bio_async(bridge);
    mirror_snn_disconnect_immune(bridge);

    /* Destroy SNN only if we own it */
    if (bridge->owns_snn && bridge->snn) {
        snn_network_destroy(bridge->snn);
    }

    if (bridge->mutex) {
        nimcp_mutex_free(bridge->mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOG_INFO(LOG_MODULE, "Destroyed mirror-SNN bridge");
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int mirror_snn_connect_bio_async(mirror_snn_bridge_t* bridge) {
    if (!bridge) return -1;
    if (bridge->bio_async_connected) return 0;  /* Already connected */

    nimcp_mutex_lock(bridge->mutex);

    /* Connect SNN to bio-async */
    int ret = snn_network_connect_bio_async(bridge->snn);
    if (ret != SNN_SUCCESS) {
        nimcp_mutex_unlock(bridge->mutex);
        NIMCP_LOG_ERROR(LOG_MODULE, "Failed to connect SNN to bio-async");
        return ret;
    }

    /* Register message handlers */
    snn_bio_async_register_handler(bridge->snn, SNN_BIO_MSG_SPIKE_EVENT,
        handle_spike_event, bridge);
    snn_bio_async_register_handler(bridge->snn, SNN_BIO_MSG_STDP_EVENT,
        handle_training_event, bridge);
    snn_bio_async_register_handler(bridge->snn, SNN_BIO_MSG_POPULATION_ACTIVITY,
        handle_population_activity, bridge);

    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOG_INFO(LOG_MODULE, "Connected to bio-async");
    return 0;
}

int mirror_snn_disconnect_bio_async(mirror_snn_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_connected) return 0;

    nimcp_mutex_lock(bridge->mutex);
    snn_network_disconnect_bio_async(bridge->snn);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOG_INFO(LOG_MODULE, "Disconnected from bio-async");
    return 0;
}

bool mirror_snn_is_bio_async_connected(const mirror_snn_bridge_t* bridge) {
    return bridge && bridge->bio_async_connected;
}

int mirror_snn_process_messages(mirror_snn_bridge_t* bridge, int timeout_ms) {
    if (!bridge || !bridge->bio_async_connected) return 0;
    return snn_bio_async_process(bridge->snn, timeout_ms);
}

//=============================================================================
// Immune System Integration
//=============================================================================

int mirror_snn_connect_immune(mirror_snn_bridge_t* bridge, void* immune_system) {
    if (!bridge || !immune_system) return -1;

    nimcp_mutex_lock(bridge->mutex);

    int ret = snn_network_connect_immune(bridge->snn, immune_system);
    if (ret == SNN_SUCCESS) {
        bridge->immune_system = immune_system;
        bridge->immune_connected = true;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return ret;
}

int mirror_snn_disconnect_immune(mirror_snn_bridge_t* bridge) {
    if (!bridge || !bridge->immune_connected) return 0;

    nimcp_mutex_lock(bridge->mutex);
    bridge->immune_system = NULL;
    bridge->immune_connected = false;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int mirror_snn_apply_immune_modulation(mirror_snn_bridge_t* bridge) {
    if (!bridge || !bridge->immune_connected) return -1;
    return snn_network_apply_immune_modulation(bridge->snn);
}

//=============================================================================
// Mirror --> SNN Pathway (Observation Encoding)
//=============================================================================

int mirror_snn_encode_observation(
    mirror_snn_bridge_t* bridge,
    uint32_t action_id,
    const float* features,
    uint32_t feature_dim,
    float observation_strength
) {
    if (!bridge || !features || feature_dim == 0) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = MIRROR_SNN_STATE_ENCODING;

    /* Scale features by observation strength */
    uint32_t dim = feature_dim < bridge->config.input_dim ?
                   feature_dim : bridge->config.input_dim;
    float scaled[MIRROR_SNN_INPUT_DIM];
    for (uint32_t i = 0; i < dim; i++) {
        scaled[i] = features[i] * observation_strength * bridge->config.encoding_gain;
    }
    for (uint32_t i = dim; i < bridge->config.input_dim; i++) {
        scaled[i] = 0.0f;
    }

    /* Set SNN inputs */
    int ret = snn_network_set_inputs(bridge->snn, scaled, bridge->config.input_dim);
    if (ret != SNN_SUCCESS) {
        bridge->state = MIRROR_SNN_STATE_IDLE;
        nimcp_mutex_unlock(bridge->mutex);
        return -1;
    }

    /* Update action state */
    if (action_id < MIRROR_SNN_MAX_ACTIONS) {
        bridge->actions[action_id].status = MIRROR_SNN_ACTION_OBSERVED;
        bridge->actions[action_id].observation_start_us = get_time_us();
    }

    /* Update statistics */
    bridge->stats.total_observations++;

    nimcp_mutex_unlock(bridge->mutex);
    return (int)dim;
}

int mirror_snn_encode_execution(
    mirror_snn_bridge_t* bridge,
    uint32_t action_id,
    const float* motor_command,
    uint32_t command_dim,
    float execution_strength
) {
    if (!bridge || !motor_command || command_dim == 0) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = MIRROR_SNN_STATE_ENCODING;

    /* Scale motor command */
    uint32_t dim = command_dim < bridge->config.input_dim ?
                   command_dim : bridge->config.input_dim;
    float scaled[MIRROR_SNN_INPUT_DIM];
    for (uint32_t i = 0; i < dim; i++) {
        scaled[i] = motor_command[i] * execution_strength * bridge->config.encoding_gain;
    }
    for (uint32_t i = dim; i < bridge->config.input_dim; i++) {
        scaled[i] = 0.0f;
    }

    /* Set SNN inputs (execution pathway) */
    int ret = snn_network_set_inputs(bridge->snn, scaled, bridge->config.input_dim);

    /* Update action state */
    if (action_id < MIRROR_SNN_MAX_ACTIONS) {
        bridge->actions[action_id].status = MIRROR_SNN_ACTION_EXECUTING;
    }

    nimcp_mutex_unlock(bridge->mutex);
    return ret == SNN_SUCCESS ? (int)dim : -1;
}

int mirror_snn_set_input_tensor(
    mirror_snn_bridge_t* bridge,
    const nimcp_tensor_t* input
) {
    if (!bridge || !input) return -1;
    return snn_network_set_input_tensor(bridge->snn, input);
}

//=============================================================================
// SNN --> Mirror Pathway (Output Decoding)
//=============================================================================

int mirror_snn_simulate(mirror_snn_bridge_t* bridge, float duration_ms) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = MIRROR_SNN_STATE_SIMULATING;

    uint64_t start = get_time_us();
    int spikes = snn_network_run(bridge->snn, duration_ms);
    uint64_t elapsed = get_time_us() - start;

    /* Update statistics */
    bridge->stats.total_simulation_steps++;
    bridge->stats.total_spikes_generated += spikes > 0 ? spikes : 0;
    bridge->stats.avg_simulation_time_us =
        0.99f * bridge->stats.avg_simulation_time_us + 0.01f * (float)elapsed;

    bridge->state = MIRROR_SNN_STATE_DECODING;
    nimcp_mutex_unlock(bridge->mutex);

    return spikes;
}

int mirror_snn_get_recognized_action(
    mirror_snn_bridge_t* bridge,
    uint32_t* action_id,
    float* confidence
) {
    if (!bridge || !action_id || !confidence) return -1;

    nimcp_mutex_lock(bridge->mutex);

    /* Get SNN outputs */
    float outputs[MIRROR_SNN_OUTPUT_DIM];
    int ret = snn_network_get_outputs(bridge->snn, outputs, bridge->config.output_dim);
    if (ret != SNN_SUCCESS) {
        nimcp_mutex_unlock(bridge->mutex);
        return ret;
    }

    /* Find maximum output */
    uint32_t best_action = 0;
    float best_conf = outputs[0];
    for (uint32_t i = 1; i < bridge->config.output_dim; i++) {
        if (outputs[i] > best_conf) {
            best_conf = outputs[i];
            best_action = i;
        }
    }

    /* Apply confidence gain and threshold */
    best_conf *= bridge->config.confidence_gain;
    if (best_conf < bridge->config.decoding_threshold) {
        nimcp_mutex_unlock(bridge->mutex);
        return -1;  /* No confident recognition */
    }

    *action_id = best_action;
    *confidence = clamp_f(best_conf, 0.0f, 1.0f);

    /* Update action state */
    if (best_action < MIRROR_SNN_MAX_ACTIONS) {
        bridge->actions[best_action].status = MIRROR_SNN_ACTION_RECOGNIZED;
        bridge->actions[best_action].recognition_confidence = best_conf;
    }

    /* Invoke callback */
    if (bridge->recognition_callback) {
        float latency = (float)(get_time_us() -
            bridge->actions[best_action].observation_start_us) / 1000.0f;
        bridge->recognition_callback(
            best_action, best_conf, latency, bridge->recognition_callback_data);
    }

    bridge->stats.total_recognitions++;
    bridge->state = MIRROR_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int mirror_snn_get_action_confidences(
    mirror_snn_bridge_t* bridge,
    float* confidences,
    uint32_t num_actions
) {
    if (!bridge || !confidences) return -1;

    uint32_t n = num_actions < bridge->config.output_dim ?
                 num_actions : bridge->config.output_dim;

    int ret = snn_network_get_outputs(bridge->snn, confidences, n);
    if (ret != SNN_SUCCESS) return ret;

    /* Normalize firing rates to [0, 1] confidence
     * SNN outputs are in Hz - typical max firing rates ~100-200 Hz
     * We use sigmoid normalization for smooth scaling */
    const float max_rate = 100.0f;  /* Reference max rate */

    /* Count confident actions and normalize */
    int count = 0;
    for (uint32_t i = 0; i < n; i++) {
        /* Normalize: rate/max_rate then apply gain and clamp */
        float normalized = confidences[i] / max_rate;
        normalized = clamp_f(normalized * bridge->config.confidence_gain, 0.0f, 1.0f);
        confidences[i] = normalized;
        if (confidences[i] > bridge->config.decoding_threshold) count++;
    }

    return count;
}

float mirror_snn_get_population_rate(
    mirror_snn_bridge_t* bridge,
    uint32_t action_id,
    float window_ms
) {
    if (!bridge) return -1.0f;
    return snn_network_get_population_rate(bridge->snn, action_id, window_ms);
}

int mirror_snn_get_output_tensor(
    mirror_snn_bridge_t* bridge,
    nimcp_tensor_t* output
) {
    if (!bridge || !output) return -1;
    return snn_network_get_output_tensor(bridge->snn, output);
}

//=============================================================================
// Complete Forward Pass
//=============================================================================

int mirror_snn_forward(
    mirror_snn_bridge_t* bridge,
    uint32_t action_id,
    const float* features,
    uint32_t feature_dim,
    float observation_strength,
    uint32_t* recognized_action,
    float* recognition_confidence
) {
    if (!bridge || !features) return -1;

    /* Encode observation */
    int spikes = mirror_snn_encode_observation(
        bridge, action_id, features, feature_dim, observation_strength);
    if (spikes < 0) return spikes;

    /* Simulate */
    spikes = mirror_snn_simulate(bridge, bridge->config.simulation_duration_ms);
    if (spikes < 0) return spikes;

    /* Decode recognition */
    return mirror_snn_get_recognized_action(
        bridge, recognized_action, recognition_confidence);
}

//=============================================================================
// Training API
//=============================================================================

int mirror_snn_set_training(mirror_snn_bridge_t* bridge, bool enable) {
    if (!bridge) return -1;
    return snn_network_set_training(bridge->snn, enable);
}

int mirror_snn_apply_stdp(mirror_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = MIRROR_SNN_STATE_TRAINING;

    int updates = snn_network_apply_stdp(bridge->snn);
    bridge->stats.training_iterations++;

    bridge->state = MIRROR_SNN_STATE_IDLE;
    nimcp_mutex_unlock(bridge->mutex);

    return updates;
}

int mirror_snn_apply_reward(mirror_snn_bridge_t* bridge, float reward) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);
    bridge->state = MIRROR_SNN_STATE_TRAINING;

    int updates = snn_network_apply_rstdp(bridge->snn, reward);
    bridge->stats.training_iterations++;

    bridge->state = MIRROR_SNN_STATE_IDLE;
    nimcp_mutex_unlock(bridge->mutex);

    return updates;
}

float mirror_snn_train_step(
    mirror_snn_bridge_t* bridge,
    const float* features,
    uint32_t feature_dim,
    uint32_t target_action
) {
    if (!bridge || !features) return -1.0f;

    /* Create target tensor */
    float targets[MIRROR_SNN_OUTPUT_DIM] = {0};
    if (target_action < bridge->config.output_dim) {
        targets[target_action] = 1.0f;
    }

    return snn_network_train_step(
        bridge->snn,
        features, feature_dim,
        targets, bridge->config.output_dim,
        bridge->config.simulation_duration_ms
    );
}

//=============================================================================
// Callback Registration
//=============================================================================

int mirror_snn_register_spike_callback(
    mirror_snn_bridge_t* bridge,
    mirror_snn_spike_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->mutex);
    bridge->spike_callback = callback;
    bridge->spike_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int mirror_snn_register_recognition_callback(
    mirror_snn_bridge_t* bridge,
    mirror_snn_recognition_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->mutex);
    bridge->recognition_callback = callback;
    bridge->recognition_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int mirror_snn_register_training_callback(
    mirror_snn_bridge_t* bridge,
    mirror_snn_training_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->mutex);
    bridge->training_callback = callback;
    bridge->training_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int mirror_snn_register_health_callback(
    mirror_snn_bridge_t* bridge,
    mirror_snn_health_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->mutex);
    bridge->health_callback = callback;
    bridge->health_callback_data = user_data;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// State Query API
//=============================================================================

int mirror_snn_get_state(
    const mirror_snn_bridge_t* bridge,
    mirror_snn_bridge_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(((mirror_snn_bridge_t*)bridge)->mutex);

    state->state = bridge->state;
    state->snn_health = snn_network_check_health(bridge->snn);

    snn_stats_t snn_stats;
    if (snn_network_get_stats(bridge->snn, &snn_stats) == SNN_SUCCESS) {
        state->mean_firing_rate = snn_stats.mean_firing_rate;
        state->sparsity = snn_stats.sparsity;
    }

    /* Count active observations and recognitions */
    state->active_observations = 0;
    state->recognized_actions = 0;
    for (uint32_t i = 0; i < MIRROR_SNN_MAX_ACTIONS; i++) {
        if (bridge->actions[i].status == MIRROR_SNN_ACTION_OBSERVED) {
            state->active_observations++;
        } else if (bridge->actions[i].status == MIRROR_SNN_ACTION_RECOGNIZED) {
            state->recognized_actions++;
        }
    }

    state->training_active = bridge->config.enable_training;
    state->weight_updates = (uint32_t)bridge->stats.stdp_events;

    nimcp_mutex_unlock(((mirror_snn_bridge_t*)bridge)->mutex);
    return 0;
}

int mirror_snn_get_action_state(
    const mirror_snn_bridge_t* bridge,
    uint32_t action_id,
    mirror_snn_action_state_t* state
) {
    if (!bridge || !state || action_id >= MIRROR_SNN_MAX_ACTIONS) return -1;

    nimcp_mutex_lock(((mirror_snn_bridge_t*)bridge)->mutex);
    *state = bridge->actions[action_id];
    nimcp_mutex_unlock(((mirror_snn_bridge_t*)bridge)->mutex);

    return 0;
}

int mirror_snn_get_stats(
    const mirror_snn_bridge_t* bridge,
    mirror_snn_stats_t* stats
) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(((mirror_snn_bridge_t*)bridge)->mutex);
    *stats = bridge->stats;

    /* Get bio-async stats */
    if (bridge->bio_async_connected) {
        snn_bio_async_get_stats(
            bridge->snn,
            &stats->bio_messages_sent,
            &stats->bio_messages_received,
            &stats->bio_messages_dropped
        );
    }

    nimcp_mutex_unlock(((mirror_snn_bridge_t*)bridge)->mutex);
    return 0;
}

void mirror_snn_reset_stats(mirror_snn_bridge_t* bridge) {
    if (!bridge) return;
    nimcp_mutex_lock(bridge->mutex);
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->mutex);
}

snn_state_health_t mirror_snn_check_health(const mirror_snn_bridge_t* bridge) {
    if (!bridge) return SNN_STATE_NAN_DETECTED;
    return snn_network_check_health(bridge->snn);
}

//=============================================================================
// Main Update Loop
//=============================================================================

int mirror_snn_update(mirror_snn_bridge_t* bridge, float dt_ms) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    uint64_t now = get_time_us();
    float elapsed_ms = (float)(now - bridge->last_update_us) / 1000.0f;

    /* Rate limiting */
    if (elapsed_ms < bridge->config.update_interval_ms) {
        nimcp_mutex_unlock(bridge->mutex);
        return 0;
    }

    /* Check health status changes */
    snn_state_health_t health = snn_network_check_health(bridge->snn);
    if (health != bridge->last_health && bridge->health_callback) {
        bridge->health_callback(
            bridge->last_health, health, bridge->health_callback_data);
    }
    bridge->last_health = health;

    /* Process bio-async messages */
    if (bridge->bio_async_connected) {
        snn_bio_async_process(bridge->snn, 0);
    }

    /* Apply immune modulation if connected */
    if (bridge->immune_connected) {
        snn_network_apply_immune_modulation(bridge->snn);
    }

    /* Update firing rates for tracked actions */
    for (uint32_t i = 0; i < bridge->num_actions; i++) {
        if (bridge->actions[i].status != MIRROR_SNN_ACTION_INACTIVE) {
            bridge->actions[i].firing_rate =
                snn_network_get_population_rate(bridge->snn, i, 100.0f);
        }
    }

    bridge->last_update_us = now;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

int mirror_snn_reset(mirror_snn_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    snn_network_reset(bridge->snn);

    for (uint32_t i = 0; i < MIRROR_SNN_MAX_ACTIONS; i++) {
        init_action_state(&bridge->actions[i], i);
    }

    bridge->state = MIRROR_SNN_STATE_IDLE;

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

//=============================================================================
// Direct SNN Access
//=============================================================================

snn_network_t* mirror_snn_get_network(mirror_snn_bridge_t* bridge) {
    return bridge ? bridge->snn : NULL;
}

int mirror_snn_get_snn_stats(
    const mirror_snn_bridge_t* bridge,
    snn_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    return snn_network_get_stats(bridge->snn, stats);
}
