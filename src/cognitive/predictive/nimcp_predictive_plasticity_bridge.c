/**
 * @file nimcp_predictive_plasticity_bridge.c
 * @brief Predictive Processing - Plasticity Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/predictive/nimcp_predictive_plasticity_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
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

typedef struct synapse_entry {
    predictive_plasticity_synapse_t synapse;
    bool in_use;
} synapse_entry_t;

struct predictive_plasticity_bridge {
    bridge_base_t base;
    predictive_plasticity_config_t config;

    /* State */
    predictive_plasticity_state_t state;
    uint64_t current_time_us;
    bool bio_async_connected;

    /* Synapse storage */
    synapse_entry_t* synapses;
    uint32_t synapse_count;
    uint32_t max_synapses;

    /* Model state */
    predictive_model_state_t model_state;

    /* Global learning state */
    float current_error;
    float learning_rate_effective;
    float bcm_global_threshold;

    /* Callbacks */
    predictive_plasticity_learn_callback_t learn_callback;
    void* learn_callback_data;
    predictive_plasticity_model_callback_t model_callback;
    void* model_callback_data;

    /* Statistics */
    predictive_plasticity_stats_t stats;
};

//=============================================================================
// Helper Functions
//=============================================================================

static inline float clamp_f(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

static synapse_entry_t* find_synapse(predictive_plasticity_bridge_t* bridge, uint32_t synapse_id) {
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use &&
            bridge->synapses[i].synapse.synapse_id == synapse_id) {
            return &bridge->synapses[i];
        }
    }
    return NULL;
}

static synapse_entry_t* find_free_slot(predictive_plasticity_bridge_t* bridge) {
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (!bridge->synapses[i].in_use) {
            return &bridge->synapses[i];
        }
    }
    return NULL;
}

static bool is_protected_type(predictive_synapse_type_t type) {
    return type == PREDICTIVE_SYNAPSE_ERROR ||
           type == PREDICTIVE_SYNAPSE_HIERARCHY;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

predictive_plasticity_config_t predictive_plasticity_config_default(void) {
    predictive_plasticity_config_t config = {
        .base_learning_rate = PREDICTIVE_PLASTICITY_DEFAULT_LR,
        .stdp_tau_plus_ms = 20.0f,
        .stdp_tau_minus_ms = 20.0f,
        .stdp_a_plus = 0.01f,
        .stdp_a_minus = 0.012f,

        .bcm_tau_ms = 1000.0f,
        .bcm_target_rate = 5.0f,

        .homeostatic_tau_ms = 10000.0f,
        .target_prediction = 0.5f,

        .error_reduction_boost = 1.5f,
        .model_update_boost = 1.3f,
        .precision_modulation = 1.2f,

        .weight_min = 0.0f,
        .weight_max = 1.0f,

        .protect_error_pathways = true,
        .protect_hierarchy = true,
        .protection_strength = 1.0f,

        .max_synapses = PREDICTIVE_PLASTICITY_MAX_SYNAPSES,

        .enable_bio_async = false
    };
    return config;
}

predictive_plasticity_bridge_t* predictive_plasticity_create(
    const predictive_plasticity_config_t* config
) {
    predictive_plasticity_bridge_t* bridge = nimcp_calloc(1, sizeof(predictive_plasticity_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = predictive_plasticity_config_default();
    }

    /* Initialize bridge base */
    if (bridge_base_init(&bridge->base, 0, "predictive_plasticity") != 0) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate synapse storage */
    bridge->max_synapses = bridge->config.max_synapses;
    bridge->synapses = nimcp_calloc(bridge->max_synapses, sizeof(synapse_entry_t));
    if (!bridge->synapses) {
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize model state */
    bridge->model_state.prediction_accuracy = 1.0f;
    bridge->model_state.model_calibration = 0.5f;
    bridge->model_state.error_sensitivity = 1.0f;
    bridge->model_state.precision_strength = 0.5f;
    bridge->model_state.anticipation_strength = 0.5f;
    bridge->model_state.learning_rate_mod = 1.0f;
    bridge->model_state.last_learning_us = 0;

    bridge->state = PREDICTIVE_PLASTICITY_STATE_IDLE;
    bridge->current_time_us = 0;
    bridge->bio_async_connected = false;
    bridge->synapse_count = 0;

    bridge->current_error = 0.0f;
    bridge->learning_rate_effective = bridge->config.base_learning_rate;
    bridge->bcm_global_threshold = bridge->config.bcm_target_rate;

    return bridge;
}

void predictive_plasticity_destroy(predictive_plasticity_bridge_t* bridge) {
    if (!bridge) return;

    bridge_base_cleanup(&bridge->base);

    nimcp_free(bridge->synapses);
    nimcp_free(bridge);
}

int predictive_plasticity_reset(predictive_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Reset all synapses to initial weights */
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use) {
            bridge->synapses[i].synapse.weight = bridge->synapses[i].synapse.initial_weight;
            bridge->synapses[i].synapse.eligibility_trace = 0.0f;
            bridge->synapses[i].synapse.bcm_threshold = bridge->config.bcm_target_rate;
            bridge->synapses[i].synapse.avg_activity = 0.0f;
            bridge->synapses[i].synapse.update_count = 0;
        }
    }

    /* Reset model state */
    bridge->model_state.prediction_accuracy = 1.0f;
    bridge->model_state.model_calibration = 0.5f;
    bridge->model_state.error_sensitivity = 1.0f;
    bridge->model_state.learning_rate_mod = 1.0f;

    bridge->state = PREDICTIVE_PLASTICITY_STATE_IDLE;
    bridge->current_error = 0.0f;
    bridge->learning_rate_effective = bridge->config.base_learning_rate;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

//=============================================================================
// Synapse Management
//=============================================================================

int predictive_plasticity_register_synapse(
    predictive_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    predictive_synapse_type_t type,
    float initial_weight
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Check for duplicate */
    if (find_synapse(bridge, synapse_id)) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Find free slot */
    synapse_entry_t* slot = find_free_slot(bridge);
    if (!slot) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Initialize synapse */
    slot->in_use = true;
    slot->synapse.synapse_id = synapse_id;
    slot->synapse.type = type;
    slot->synapse.weight = clamp_f(initial_weight, bridge->config.weight_min, bridge->config.weight_max);
    slot->synapse.initial_weight = slot->synapse.weight;
    slot->synapse.eligibility_trace = 0.0f;
    slot->synapse.bcm_threshold = bridge->config.bcm_target_rate;
    slot->synapse.avg_activity = 0.0f;
    slot->synapse.last_update_us = bridge->current_time_us;
    slot->synapse.update_count = 0;

    /* Auto-protect error and hierarchy synapses */
    slot->synapse.is_protected = is_protected_type(type) &&
        (bridge->config.protect_error_pathways || bridge->config.protect_hierarchy);

    bridge->synapse_count++;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_plasticity_unregister_synapse(
    predictive_plasticity_bridge_t* bridge,
    uint32_t synapse_id
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    synapse_entry_t* entry = find_synapse(bridge, synapse_id);
    if (!entry) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    entry->in_use = false;
    bridge->synapse_count--;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_plasticity_get_synapse(
    predictive_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    predictive_plasticity_synapse_t* synapse
) {
    if (!bridge || !synapse) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    synapse_entry_t* entry = find_synapse(bridge, synapse_id);
    if (!entry) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    *synapse = entry->synapse;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_plasticity_protect_synapse(
    predictive_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    bool protect
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    synapse_entry_t* entry = find_synapse(bridge, synapse_id);
    if (!entry) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    entry->synapse.is_protected = protect;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

//=============================================================================
// Learning Functions
//=============================================================================

int predictive_plasticity_learn(
    predictive_plasticity_bridge_t* bridge,
    predictive_learn_event_t event,
    float magnitude,
    uint32_t synapse_id,
    float context
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = PREDICTIVE_PLASTICITY_STATE_LEARNING;

    synapse_entry_t* entry = find_synapse(bridge, synapse_id);
    if (!entry) {
        bridge->state = PREDICTIVE_PLASTICITY_STATE_IDLE;
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Check protection */
    if (entry->synapse.is_protected) {
        bridge->stats.protected_updates_blocked++;
        bridge->state = PREDICTIVE_PLASTICITY_STATE_IDLE;
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0; /* Not an error, just blocked */
    }

    float lr = bridge->learning_rate_effective * magnitude;
    float weight_change = 0.0f;

    switch (event) {
        case PREDICTIVE_LEARN_PREDICTION_CONFIRMED:
            weight_change = lr * bridge->config.precision_modulation;
            bridge->stats.prediction_confirmed_events++;
            break;

        case PREDICTIVE_LEARN_PREDICTION_VIOLATED:
            weight_change = -lr * 0.5f;
            bridge->stats.prediction_violated_events++;
            break;

        case PREDICTIVE_LEARN_ERROR_MINIMIZED:
            weight_change = lr * bridge->config.error_reduction_boost;
            bridge->stats.error_minimized_events++;
            break;

        case PREDICTIVE_LEARN_ERROR_PERSISTED:
            weight_change = -lr * 0.3f;
            break;

        case PREDICTIVE_LEARN_MODEL_UPDATED:
            weight_change = lr * bridge->config.model_update_boost;
            bridge->stats.model_update_events++;
            break;

        case PREDICTIVE_LEARN_MODEL_STABLE:
            weight_change = lr * 0.1f;
            break;

        case PREDICTIVE_LEARN_PRECISION_INCREASED:
            weight_change = lr * 0.8f;
            break;

        case PREDICTIVE_LEARN_ANTICIPATION_CORRECT:
            weight_change = lr * 1.0f;
            break;

        case PREDICTIVE_LEARN_ANTICIPATION_WRONG:
            weight_change = -lr * 0.3f;
            break;

        case PREDICTIVE_LEARN_FREE_ENERGY_REDUCED:
            weight_change = lr * 0.5f;
            break;
    }

    /* Context modulation */
    weight_change *= (0.5f + 0.5f * context);

    /* Apply weight change */
    float old_weight = entry->synapse.weight;
    entry->synapse.weight = clamp_f(
        entry->synapse.weight + weight_change,
        bridge->config.weight_min,
        bridge->config.weight_max
    );

    /* Update statistics */
    float actual_change = entry->synapse.weight - old_weight;
    if (actual_change > 0) {
        bridge->stats.total_potentiation += actual_change;
    } else {
        bridge->stats.total_depression += -actual_change;
    }

    entry->synapse.update_count++;
    entry->synapse.last_update_us = bridge->current_time_us;

    bridge->stats.total_learning_events++;
    bridge->stats.weight_updates++;
    bridge->stats.mean_weight_change =
        (bridge->stats.mean_weight_change * (bridge->stats.weight_updates - 1) +
         fabsf(actual_change)) / bridge->stats.weight_updates;

    /* Invoke callback */
    if (bridge->learn_callback) {
        bridge->learn_callback(bridge, event, magnitude, bridge->learn_callback_data);
    }

    bridge->state = PREDICTIVE_PLASTICITY_STATE_IDLE;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float predictive_plasticity_apply_stdp(
    predictive_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    float pre_time,
    float post_time
) {
    if (!bridge) return NAN;

    nimcp_mutex_lock(bridge->base.mutex);

    synapse_entry_t* entry = find_synapse(bridge, synapse_id);
    if (!entry) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return NAN;
    }

    /* Protected synapses don't get STDP */
    if (entry->synapse.is_protected) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0.0f;
    }

    float dt = post_time - pre_time;
    float delta_w = 0.0f;

    if (dt > 0) {
        /* Potentiation: post after pre */
        delta_w = bridge->config.stdp_a_plus * expf(-dt / bridge->config.stdp_tau_plus_ms);
    } else {
        /* Depression: pre after post */
        delta_w = -bridge->config.stdp_a_minus * expf(dt / bridge->config.stdp_tau_minus_ms);
    }

    /* Apply weight change */
    float old_weight = entry->synapse.weight;
    entry->synapse.weight = clamp_f(
        entry->synapse.weight + delta_w,
        bridge->config.weight_min,
        bridge->config.weight_max
    );

    float actual_change = entry->synapse.weight - old_weight;
    if (actual_change > 0) {
        bridge->stats.total_potentiation += actual_change;
    } else {
        bridge->stats.total_depression += -actual_change;
    }

    entry->synapse.update_count++;
    bridge->stats.weight_updates++;

    nimcp_mutex_unlock(bridge->base.mutex);
    return delta_w;
}

int predictive_plasticity_apply_error(
    predictive_plasticity_bridge_t* bridge,
    float error
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    error = clamp_f(error, -1.0f, 1.0f);
    bridge->current_error = error;

    /* Apply error modulation to all eligible synapses */
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use && !bridge->synapses[i].synapse.is_protected) {
            float trace = bridge->synapses[i].synapse.eligibility_trace;
            if (fabsf(trace) > 0.001f) {
                float delta = bridge->config.base_learning_rate * error * trace;
                bridge->synapses[i].synapse.weight = clamp_f(
                    bridge->synapses[i].synapse.weight + delta,
                    bridge->config.weight_min,
                    bridge->config.weight_max
                );
                bridge->stats.weight_updates++;
            }
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_plasticity_update_bcm(
    predictive_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;
    if (dt_ms <= 0.0f) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = PREDICTIVE_PLASTICITY_STATE_UPDATING;

    float decay = expf(-dt_ms / bridge->config.bcm_tau_ms);

    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use) {
            /* Update sliding threshold towards average activity */
            float target = bridge->synapses[i].synapse.avg_activity;
            bridge->synapses[i].synapse.bcm_threshold =
                bridge->synapses[i].synapse.bcm_threshold * decay +
                target * (1.0f - decay);
        }
    }

    bridge->state = PREDICTIVE_PLASTICITY_STATE_IDLE;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_plasticity_homeostatic_update(
    predictive_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = PREDICTIVE_PLASTICITY_STATE_UPDATING;

    float decay = expf(-dt_ms / bridge->config.homeostatic_tau_ms);

    /* Calculate mean prediction */
    float mean_prediction = 0.0f;
    uint32_t prediction_count = 0;
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use &&
            bridge->synapses[i].synapse.type == PREDICTIVE_SYNAPSE_PREDICTION) {
            mean_prediction += bridge->synapses[i].synapse.weight;
            prediction_count++;
        }
    }
    if (prediction_count > 0) {
        mean_prediction /= prediction_count;
    }

    /* Scale non-protected synapses toward target */
    float target = bridge->config.target_prediction;
    float scale_factor = 1.0f;
    if (mean_prediction > 0.0f) {
        scale_factor = target / mean_prediction;
        scale_factor = clamp_f(scale_factor, 0.9f, 1.1f);
    }

    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use && !bridge->synapses[i].synapse.is_protected) {
            float scaled = bridge->synapses[i].synapse.weight * (1.0f + (scale_factor - 1.0f) * (1.0f - decay));
            bridge->synapses[i].synapse.weight = clamp_f(
                scaled,
                bridge->config.weight_min,
                bridge->config.weight_max
            );
        }
    }

    bridge->state = PREDICTIVE_PLASTICITY_STATE_IDLE;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_plasticity_update_traces(
    predictive_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    float decay = expf(-dt_ms / bridge->config.stdp_tau_plus_ms);

    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use) {
            bridge->synapses[i].synapse.eligibility_trace *= decay;
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_plasticity_consolidate(predictive_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = PREDICTIVE_PLASTICITY_STATE_CONSOLIDATING;

    /* Clear eligibility traces */
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use) {
            bridge->synapses[i].synapse.eligibility_trace = 0.0f;
        }
    }

    /* Update model state based on synapse weights */
    float prediction_sum = 0.0f, prediction_count = 0;
    float precision_sum = 0.0f, precision_count = 0;
    float anticipation_sum = 0.0f, anticipation_count = 0;
    float model_sum = 0.0f, model_count = 0;

    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use) {
            switch (bridge->synapses[i].synapse.type) {
                case PREDICTIVE_SYNAPSE_PREDICTION:
                    prediction_sum += bridge->synapses[i].synapse.weight;
                    prediction_count++;
                    break;
                case PREDICTIVE_SYNAPSE_PRECISION:
                    precision_sum += bridge->synapses[i].synapse.weight;
                    precision_count++;
                    break;
                case PREDICTIVE_SYNAPSE_ANTICIPATION:
                    anticipation_sum += bridge->synapses[i].synapse.weight;
                    anticipation_count++;
                    break;
                case PREDICTIVE_SYNAPSE_MODEL:
                    model_sum += bridge->synapses[i].synapse.weight;
                    model_count++;
                    break;
                default:
                    break;
            }
        }
    }

    if (prediction_count > 0) {
        bridge->model_state.prediction_accuracy = prediction_sum / prediction_count * 2.0f;
    }
    if (precision_count > 0) {
        bridge->model_state.precision_strength = precision_sum / precision_count * 2.0f;
    }
    if (anticipation_count > 0) {
        bridge->model_state.anticipation_strength = anticipation_sum / anticipation_count;
    }
    if (model_count > 0) {
        bridge->model_state.model_calibration = model_sum / model_count;
    }

    bridge->model_state.last_learning_us = bridge->current_time_us;
    bridge->state = PREDICTIVE_PLASTICITY_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

//=============================================================================
// State Query Functions
//=============================================================================

int predictive_plasticity_get_model_state(
    predictive_plasticity_bridge_t* bridge,
    predictive_model_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->model_state;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int predictive_plasticity_get_state(
    predictive_plasticity_bridge_t* bridge,
    predictive_plasticity_bridge_state_t* state
) {
    if (!bridge || !state) return -1;

    nimcp_mutex_lock(bridge->base.mutex);

    state->state = bridge->state;
    state->active_synapses = bridge->synapse_count;
    state->learning_rate_effective = bridge->learning_rate_effective;

    /* Calculate mean weight and variance */
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        if (bridge->synapses[i].in_use) {
            float w = bridge->synapses[i].synapse.weight;
            sum += w;
            sum_sq += w * w;
        }
    }

    if (bridge->synapse_count > 0) {
        state->mean_weight = sum / bridge->synapse_count;
        state->weight_variance = (sum_sq / bridge->synapse_count) -
                                (state->mean_weight * state->mean_weight);
    } else {
        state->mean_weight = 0.0f;
        state->weight_variance = 0.0f;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_plasticity_get_stats(
    predictive_plasticity_bridge_t* bridge,
    predictive_plasticity_stats_t* stats
) {
    if (!bridge || !stats) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int predictive_plasticity_reset_stats(predictive_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(predictive_plasticity_stats_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

//=============================================================================
// Callback Registration
//=============================================================================

int predictive_plasticity_register_learn_callback(
    predictive_plasticity_bridge_t* bridge,
    predictive_plasticity_learn_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->learn_callback = callback;
    bridge->learn_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int predictive_plasticity_register_model_callback(
    predictive_plasticity_bridge_t* bridge,
    predictive_plasticity_model_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->model_callback = callback;
    bridge->model_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int predictive_plasticity_bio_async_connect(predictive_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    /* Bio-async connection would be implemented here */
    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int predictive_plasticity_bio_async_disconnect(predictive_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

bool predictive_plasticity_is_bio_async_connected(predictive_plasticity_bridge_t* bridge) {
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->base.mutex);
    bool connected = bridge->bio_async_connected;
    nimcp_mutex_unlock(bridge->base.mutex);

    return connected;
}
