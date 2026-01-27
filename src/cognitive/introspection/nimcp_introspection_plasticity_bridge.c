/**
 * @file nimcp_introspection_plasticity_bridge.c
 * @brief Introspection - Plasticity Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/introspection/nimcp_introspection_plasticity_bridge.h"
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
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for introspection_plasticity_bridge module */
static nimcp_health_agent_t* g_introspection_plasticity_bridge_health_agent = NULL;

/**
 * @brief Set health agent for introspection_plasticity_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void introspection_plasticity_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_introspection_plasticity_bridge_health_agent = agent;
}

/** @brief Send heartbeat from introspection_plasticity_bridge module */
static inline void introspection_plasticity_bridge_heartbeat(const char* operation, float progress) {
    if (g_introspection_plasticity_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_introspection_plasticity_bridge_health_agent, operation, progress);
    }
}

#define LOG_MODULE "INTROSPECTION_PLASTICITY_BRIDGE"


//=============================================================================
// Internal Structures
//=============================================================================

struct introspection_plasticity_bridge {
    bridge_base_t base;               /**< MUST be first: base bridge infrastructure */
    introspection_plasticity_config_t config;

    /* State */
    introspection_plasticity_state_t state;
    uint64_t current_time_us;
    bool bio_async_connected;

    /* Synapses */
    introspection_plasticity_synapse_t* synapses;
    uint32_t num_synapses;
    uint32_t max_synapses;

    /* Calibration state */
    introspection_calibration_state_t calibration;

    /* Reward eligibility */
    float global_eligibility;
    float last_reward;

    /* Callbacks */
    introspection_plasticity_learn_callback_t learn_callback;
    void* learn_callback_data;
    introspection_plasticity_calibration_callback_t calibration_callback;
    void* calibration_callback_data;

    /* Statistics */
    introspection_plasticity_stats_t stats;
};

//=============================================================================
// Helper Functions
//=============================================================================

static inline float clamp_f(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

static introspection_plasticity_synapse_t* find_synapse(
    introspection_plasticity_bridge_t* bridge,
    uint32_t synapse_id
) {
    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_synapses > 256) {
            introspection_plasticity_bridge_heartbeat("introspectio_loop",
                             (float)(i + 1) / (float)bridge->num_synapses);
        }

        if (bridge->synapses[i].synapse_id == synapse_id) {
            return &bridge->synapses[i];
        }
    }
    return NULL;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

introspection_plasticity_config_t introspection_plasticity_config_default(void) {
    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    introspection_plasticity_config_t config = {
        .base_learning_rate = INTROSPECTION_PLASTICITY_DEFAULT_LR,
        .stdp_tau_plus_ms = 20.0f,
        .stdp_tau_minus_ms = 25.0f,
        .stdp_a_plus = 0.01f,
        .stdp_a_minus = 0.012f,

        .bcm_tau_ms = 1000.0f,
        .bcm_target_rate = 10.0f,

        .homeostatic_tau_ms = 10000.0f,
        .target_confidence = 0.7f,

        .accuracy_learning_boost = 2.0f,
        .error_learning_boost = 1.5f,
        .calibration_modulation = 1.0f,

        .weight_min = 0.0f,
        .weight_max = 2.0f,

        .protect_core_patterns = true,
        .protect_error_detection = true,
        .protection_strength = 0.9f,

        .max_synapses = INTROSPECTION_PLASTICITY_MAX_SYNAPSES,

        .enable_bio_async = false
    };
    return config;
}

introspection_plasticity_bridge_t* introspection_plasticity_create(
    const introspection_plasticity_config_t* config
) {
    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    introspection_plasticity_bridge_t* bridge = nimcp_calloc(1, sizeof(introspection_plasticity_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = introspection_plasticity_config_default();
    }

    /* Initialize base bridge */
    if (bridge_base_init(&bridge->base, 0, "introspection_plasticity") != 0) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Allocate synapse storage */
    bridge->max_synapses = bridge->config.max_synapses;
    bridge->synapses = nimcp_calloc(bridge->max_synapses, sizeof(introspection_plasticity_synapse_t));
    if (!bridge->synapses) {
        bridge_base_cleanup(&bridge->base);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->num_synapses = 0;
    bridge->state = INTROSPECTION_PLASTICITY_STATE_IDLE;
    bridge->current_time_us = 0;
    bridge->bio_async_connected = false;
    bridge->global_eligibility = 0.0f;
    bridge->last_reward = 0.0f;

    /* Initialize calibration state */
    bridge->calibration.uncertainty_sensitivity = 0.5f;
    bridge->calibration.confidence_calibration = 0.7f;
    bridge->calibration.error_sensitivity = 0.5f;
    bridge->calibration.pattern_strength = 0.5f;
    bridge->calibration.metacognition_strength = 0.5f;
    bridge->calibration.learning_rate_mod = 1.0f;
    bridge->calibration.last_learning_us = 0;

    return bridge;
}

void introspection_plasticity_destroy(introspection_plasticity_bridge_t* bridge) {
    if (!bridge) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "introspection_plasticity");

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }

    nimcp_free(bridge->synapses);
    nimcp_free(bridge);
}

int introspection_plasticity_reset(introspection_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Reset all synapses to initial weights */
    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_synapses > 256) {
            introspection_plasticity_bridge_heartbeat("introspectio_loop",
                             (float)(i + 1) / (float)bridge->num_synapses);
        }

        bridge->synapses[i].weight = bridge->synapses[i].initial_weight;
        bridge->synapses[i].eligibility_trace = 0.0f;
        bridge->synapses[i].bcm_threshold = bridge->config.bcm_target_rate;
        bridge->synapses[i].avg_activity = 0.0f;
        bridge->synapses[i].update_count = 0;
    }

    /* Reset calibration state */
    bridge->calibration.uncertainty_sensitivity = 0.5f;
    bridge->calibration.confidence_calibration = 0.7f;
    bridge->calibration.error_sensitivity = 0.5f;
    bridge->calibration.pattern_strength = 0.5f;
    bridge->calibration.metacognition_strength = 0.5f;
    bridge->calibration.learning_rate_mod = 1.0f;

    bridge->global_eligibility = 0.0f;
    bridge->last_reward = 0.0f;
    bridge->state = INTROSPECTION_PLASTICITY_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

//=============================================================================
// Synapse Management
//=============================================================================

int introspection_plasticity_register_synapse(
    introspection_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    introspection_synapse_type_t type,
    float initial_weight
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Check if already exists */
    if (find_synapse(bridge, synapse_id)) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Check capacity */
    if (bridge->num_synapses >= bridge->max_synapses) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Add synapse */
    introspection_plasticity_synapse_t* syn = &bridge->synapses[bridge->num_synapses];
    syn->synapse_id = synapse_id;
    syn->type = type;
    syn->weight = clamp_f(initial_weight, bridge->config.weight_min, bridge->config.weight_max);
    syn->initial_weight = syn->weight;
    syn->eligibility_trace = 0.0f;
    syn->bcm_threshold = bridge->config.bcm_target_rate;
    syn->avg_activity = 0.0f;
    syn->last_update_us = bridge->current_time_us;
    syn->update_count = 0;

    /* Auto-protect core patterns and error detection */
    syn->is_protected = false;
    if (bridge->config.protect_core_patterns &&
        (type == INTROSPECTION_SYNAPSE_METACOGNITION ||
         type == INTROSPECTION_SYNAPSE_CALIBRATION)) {
        syn->is_protected = true;
    }
    if (bridge->config.protect_error_detection &&
        type == INTROSPECTION_SYNAPSE_ERROR) {
        syn->is_protected = true;
    }

    bridge->num_synapses++;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int introspection_plasticity_unregister_synapse(
    introspection_plasticity_bridge_t* bridge,
    uint32_t synapse_id
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_synapses > 256) {
            introspection_plasticity_bridge_heartbeat("introspectio_loop",
                             (float)(i + 1) / (float)bridge->num_synapses);
        }

        if (bridge->synapses[i].synapse_id == synapse_id) {
            /* Move last to current position */
            if (i < bridge->num_synapses - 1) {
                bridge->synapses[i] = bridge->synapses[bridge->num_synapses - 1];
            }
            bridge->num_synapses--;
            nimcp_mutex_unlock(bridge->base.mutex);
            return 0;
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return -1;
}

int introspection_plasticity_get_synapse(
    introspection_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    introspection_plasticity_synapse_t* synapse
) {
    if (!bridge || !synapse) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    introspection_plasticity_synapse_t* syn = find_synapse(bridge, synapse_id);
    if (!syn) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    *synapse = *syn;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int introspection_plasticity_protect_synapse(
    introspection_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    bool protect
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    introspection_plasticity_synapse_t* syn = find_synapse(bridge, synapse_id);
    if (!syn) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    syn->is_protected = protect;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

//=============================================================================
// Learning Functions
//=============================================================================

int introspection_plasticity_learn(
    introspection_plasticity_bridge_t* bridge,
    introspection_learn_event_t event,
    float magnitude,
    uint32_t synapse_id,
    float context
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = INTROSPECTION_PLASTICITY_STATE_LEARNING;

    introspection_plasticity_synapse_t* syn = find_synapse(bridge, synapse_id);
    if (!syn) {
        bridge->state = INTROSPECTION_PLASTICITY_STATE_IDLE;
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Check protection */
    if (syn->is_protected) {
        bridge->stats.protected_updates_blocked++;
        bridge->state = INTROSPECTION_PLASTICITY_STATE_IDLE;
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }

    /* Calculate learning rate with modulation */
    float lr = bridge->config.base_learning_rate * bridge->calibration.learning_rate_mod;

    /* Apply event-specific learning */
    float delta_weight = 0.0f;
    switch (event) {
        case INTROSPECTION_LEARN_CORRECT_CONFIDENCE:
            delta_weight = lr * magnitude * bridge->config.accuracy_learning_boost;
            bridge->stats.correct_confidence_events++;
            break;

        case INTROSPECTION_LEARN_OVERCONFIDENCE:
            delta_weight = -lr * magnitude * bridge->config.error_learning_boost;
            bridge->stats.overconfidence_events++;
            break;

        case INTROSPECTION_LEARN_UNDERCONFIDENCE:
            delta_weight = lr * magnitude * 0.5f;
            bridge->stats.underconfidence_events++;
            break;

        case INTROSPECTION_LEARN_ERROR_DETECTED:
            delta_weight = lr * magnitude * bridge->config.error_learning_boost;
            bridge->stats.error_detection_events++;
            break;

        case INTROSPECTION_LEARN_ERROR_MISSED:
            delta_weight = -lr * magnitude * bridge->config.error_learning_boost;
            break;

        case INTROSPECTION_LEARN_PATTERN_MATCH:
            delta_weight = lr * magnitude * context;
            break;

        case INTROSPECTION_LEARN_STATE_TRACKED:
            delta_weight = lr * magnitude * 0.5f;
            break;

        case INTROSPECTION_LEARN_UNCERTAINTY_CALIBRATED:
            delta_weight = lr * magnitude * bridge->config.calibration_modulation;
            break;

        default:
            delta_weight = lr * magnitude;
            break;
    }

    /* Apply eligibility trace modulation */
    delta_weight *= (1.0f + syn->eligibility_trace);

    /* Update weight with bounds */
    float old_weight = syn->weight;
    syn->weight = clamp_f(syn->weight + delta_weight, bridge->config.weight_min, bridge->config.weight_max);
    float actual_delta = syn->weight - old_weight;

    /* Update statistics */
    if (actual_delta > 0) {
        bridge->stats.total_potentiation += actual_delta;
    } else {
        bridge->stats.total_depression += fabsf(actual_delta);
    }
    bridge->stats.weight_updates++;
    bridge->stats.total_learning_events++;

    /* Update running mean */
    float n = (float)bridge->stats.weight_updates;
    bridge->stats.mean_weight_change = bridge->stats.mean_weight_change * ((n - 1) / n) +
                                       fabsf(actual_delta) / n;

    syn->last_update_us = bridge->current_time_us;
    syn->update_count++;
    bridge->calibration.last_learning_us = bridge->current_time_us;

    /* Invoke callback */
    if (bridge->learn_callback) {
        bridge->learn_callback(bridge, event, magnitude, bridge->learn_callback_data);
    }

    bridge->state = INTROSPECTION_PLASTICITY_STATE_IDLE;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float introspection_plasticity_apply_stdp(
    introspection_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    float pre_time,
    float post_time
) {
    if (!bridge) return NAN;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    introspection_plasticity_synapse_t* syn = find_synapse(bridge, synapse_id);
    if (!syn) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return NAN;
    }

    /* Check protection */
    if (syn->is_protected) {
        bridge->stats.protected_updates_blocked++;
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0.0f;
    }

    float dt = post_time - pre_time;
    float delta_weight = 0.0f;

    if (dt > 0) {
        /* LTP: post after pre */
        delta_weight = bridge->config.stdp_a_plus *
                      expf(-dt / bridge->config.stdp_tau_plus_ms);
    } else if (dt < 0) {
        /* LTD: pre after post */
        delta_weight = -bridge->config.stdp_a_minus *
                      expf(dt / bridge->config.stdp_tau_minus_ms);
    }

    /* Apply with learning rate */
    delta_weight *= bridge->config.base_learning_rate * bridge->calibration.learning_rate_mod;

    /* Update weight */
    float old_weight = syn->weight;
    syn->weight = clamp_f(syn->weight + delta_weight, bridge->config.weight_min, bridge->config.weight_max);
    float actual_delta = syn->weight - old_weight;

    /* Update statistics */
    if (actual_delta > 0) {
        bridge->stats.total_potentiation += actual_delta;
    } else {
        bridge->stats.total_depression += fabsf(actual_delta);
    }
    bridge->stats.weight_updates++;

    syn->last_update_us = bridge->current_time_us;
    syn->update_count++;

    nimcp_mutex_unlock(bridge->base.mutex);
    return actual_delta;
}

int introspection_plasticity_apply_reward(
    introspection_plasticity_bridge_t* bridge,
    float reward
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    reward = clamp_f(reward, -1.0f, 1.0f);
    bridge->last_reward = reward;

    /* Apply reward-modulated learning to all synapses with eligibility */
    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_synapses > 256) {
            introspection_plasticity_bridge_heartbeat("introspectio_loop",
                             (float)(i + 1) / (float)bridge->num_synapses);
        }

        introspection_plasticity_synapse_t* syn = &bridge->synapses[i];

        if (syn->is_protected || syn->eligibility_trace < 0.001f) {
            continue;
        }

        float delta = bridge->config.base_learning_rate * reward *
                     syn->eligibility_trace * bridge->config.calibration_modulation;

        syn->weight = clamp_f(syn->weight + delta, bridge->config.weight_min, bridge->config.weight_max);

        if (delta > 0) {
            bridge->stats.total_potentiation += delta;
        } else {
            bridge->stats.total_depression += fabsf(delta);
        }
        bridge->stats.weight_updates++;
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int introspection_plasticity_update_bcm(
    introspection_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;
    if (dt_ms <= 0.0f) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    float decay = expf(-dt_ms / bridge->config.bcm_tau_ms);

    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_synapses > 256) {
            introspection_plasticity_bridge_heartbeat("introspectio_loop",
                             (float)(i + 1) / (float)bridge->num_synapses);
        }

        introspection_plasticity_synapse_t* syn = &bridge->synapses[i];

        /* Update sliding threshold towards average activity */
        syn->bcm_threshold = syn->bcm_threshold * decay +
                            syn->avg_activity * (1.0f - decay);

        /* BCM learning rule: weight change depends on activity relative to threshold */
        if (!syn->is_protected && syn->avg_activity > 0.001f) {
            float bcm_factor = (syn->avg_activity - syn->bcm_threshold) * syn->avg_activity;
            float delta = bridge->config.base_learning_rate * bcm_factor * 0.001f;

            syn->weight = clamp_f(syn->weight + delta, bridge->config.weight_min, bridge->config.weight_max);
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int introspection_plasticity_homeostatic_update(
    introspection_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;
    if (dt_ms <= 0.0f) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Calculate mean weight */
    float mean_weight = 0.0f;
    uint32_t active_count = 0;

    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_synapses > 256) {
            introspection_plasticity_bridge_heartbeat("introspectio_loop",
                             (float)(i + 1) / (float)bridge->num_synapses);
        }

        if (!bridge->synapses[i].is_protected) {
            mean_weight += bridge->synapses[i].weight;
            active_count++;
        }
    }

    if (active_count > 0) {
        mean_weight /= active_count;
    }

    /* Target weight is mid-range */
    float target = (bridge->config.weight_max + bridge->config.weight_min) / 2.0f;
    float error = target - mean_weight;

    /* Homeostatic scaling */
    float scale_factor = 1.0f + error * dt_ms / bridge->config.homeostatic_tau_ms;
    scale_factor = clamp_f(scale_factor, 0.99f, 1.01f);

    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_synapses > 256) {
            introspection_plasticity_bridge_heartbeat("introspectio_loop",
                             (float)(i + 1) / (float)bridge->num_synapses);
        }

        if (!bridge->synapses[i].is_protected) {
            bridge->synapses[i].weight = clamp_f(
                bridge->synapses[i].weight * scale_factor,
                bridge->config.weight_min,
                bridge->config.weight_max
            );
        }
    }

    /* Update calibration confidence towards target */
    float conf_decay = expf(-dt_ms / bridge->config.homeostatic_tau_ms);
    float old_calib = bridge->calibration.confidence_calibration;
    bridge->calibration.confidence_calibration =
        old_calib * conf_decay + bridge->config.target_confidence * (1.0f - conf_decay);

    /* Invoke calibration callback if significant change */
    if (bridge->calibration_callback &&
        fabsf(bridge->calibration.confidence_calibration - old_calib) > 0.01f) {
        bridge->calibration_callback(bridge, old_calib,
                                     bridge->calibration.confidence_calibration,
                                     bridge->calibration_callback_data);
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int introspection_plasticity_update_traces(
    introspection_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;
    if (dt_ms <= 0.0f) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    float decay = expf(-dt_ms / bridge->config.stdp_tau_plus_ms);

    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_synapses > 256) {
            introspection_plasticity_bridge_heartbeat("introspectio_loop",
                             (float)(i + 1) / (float)bridge->num_synapses);
        }

        bridge->synapses[i].eligibility_trace *= decay;
    }

    bridge->global_eligibility *= decay;
    bridge->current_time_us += (uint64_t)(dt_ms * 1000.0f);

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int introspection_plasticity_consolidate(introspection_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = INTROSPECTION_PLASTICITY_STATE_CONSOLIDATING;

    /* Consolidate learning by resetting eligibility traces */
    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_synapses > 256) {
            introspection_plasticity_bridge_heartbeat("introspectio_loop",
                             (float)(i + 1) / (float)bridge->num_synapses);
        }

        bridge->synapses[i].eligibility_trace = 0.0f;
    }

    bridge->global_eligibility = 0.0f;
    bridge->state = INTROSPECTION_PLASTICITY_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

//=============================================================================
// State Query Functions
//=============================================================================

int introspection_plasticity_get_calibration_state(
    introspection_plasticity_bridge_t* bridge,
    introspection_calibration_state_t* state
) {
    if (!bridge || !state) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->calibration;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int introspection_plasticity_get_state(
    introspection_plasticity_bridge_t* bridge,
    introspection_plasticity_bridge_state_t* state
) {
    if (!bridge || !state) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    state->state = bridge->state;
    state->active_synapses = bridge->num_synapses;

    /* Calculate mean weight and variance */
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_synapses > 256) {
            introspection_plasticity_bridge_heartbeat("introspectio_loop",
                             (float)(i + 1) / (float)bridge->num_synapses);
        }

        sum += bridge->synapses[i].weight;
        sum_sq += bridge->synapses[i].weight * bridge->synapses[i].weight;
    }

    if (bridge->num_synapses > 0) {
        state->mean_weight = sum / bridge->num_synapses;
        state->weight_variance = (sum_sq / bridge->num_synapses) -
                                (state->mean_weight * state->mean_weight);
    } else {
        state->mean_weight = 0.0f;
        state->weight_variance = 0.0f;
    }

    state->learning_rate_effective = bridge->config.base_learning_rate *
                                     bridge->calibration.learning_rate_mod;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int introspection_plasticity_get_stats(
    introspection_plasticity_bridge_t* bridge,
    introspection_plasticity_stats_t* stats
) {
    if (!bridge || !stats) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int introspection_plasticity_reset_stats(introspection_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(introspection_plasticity_stats_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

//=============================================================================
// Callback Registration
//=============================================================================

int introspection_plasticity_register_learn_callback(
    introspection_plasticity_bridge_t* bridge,
    introspection_plasticity_learn_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->learn_callback = callback;
    bridge->learn_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int introspection_plasticity_register_calibration_callback(
    introspection_plasticity_bridge_t* bridge,
    introspection_plasticity_calibration_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->calibration_callback = callback;
    bridge->calibration_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int introspection_plasticity_bio_async_connect(introspection_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int introspection_plasticity_bio_async_disconnect(introspection_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

bool introspection_plasticity_is_bio_async_connected(introspection_plasticity_bridge_t* bridge) {
    if (!bridge) return false;

    /* Phase 8: Heartbeat at operation start */
    introspection_plasticity_bridge_heartbeat("introspectio_introspection_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bool connected = bridge->bio_async_connected;
    nimcp_mutex_unlock(bridge->base.mutex);

    return connected;
}
