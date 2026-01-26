/**
 * @file nimcp_mental_health_plasticity_bridge.c
 * @brief Mental Health - Plasticity Bidirectional Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-06
 */

#include "cognitive/mental_health/nimcp_mental_health_plasticity_bridge.h"
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

/** Global health agent for mental_health_plasticity_bridge module */
static nimcp_health_agent_t* g_mental_health_plasticity_bridge_health_agent = NULL;

/**
 * @brief Set health agent for mental_health_plasticity_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void mental_health_plasticity_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_mental_health_plasticity_bridge_health_agent = agent;
}

/** @brief Send heartbeat from mental_health_plasticity_bridge module */
static inline void mental_health_plasticity_bridge_heartbeat(const char* operation, float progress) {
    if (g_mental_health_plasticity_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_mental_health_plasticity_bridge_health_agent, operation, progress);
    }
}


//=============================================================================
// Internal Structures
//=============================================================================

typedef struct synapse_entry {
    mental_health_plasticity_synapse_t synapse;
    bool in_use;
} synapse_entry_t;

struct mental_health_plasticity_bridge {
    bridge_base_t base;
    mental_health_plasticity_config_t config;

    /* State */
    mental_health_plasticity_state_t state;
    uint64_t current_time_us;
    bool bio_async_connected;

    /* Synapse storage */
    synapse_entry_t* synapses;
    uint32_t synapse_count;
    uint32_t max_synapses;

    /* Regulation state */
    mental_health_regulation_state_t regulation_state;

    /* Global learning state */
    float current_stress_level;
    float learning_rate_effective;
    float bcm_global_threshold;

    /* Callbacks */
    mental_health_plasticity_learn_callback_t learn_callback;
    void* learn_callback_data;
    mental_health_plasticity_regulation_callback_t regulation_callback;
    void* regulation_callback_data;

    /* Statistics */
    mental_health_plasticity_stats_t stats;
};

//=============================================================================
// Helper Functions
//=============================================================================

static inline float clamp_f(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

static synapse_entry_t* find_synapse(mental_health_plasticity_bridge_t* bridge, uint32_t synapse_id) {
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->max_synapses > 256) {
            mental_health_plasticity_bridge_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)bridge->max_synapses);
        }

        if (bridge->synapses[i].in_use &&
            bridge->synapses[i].synapse.synapse_id == synapse_id) {
            return &bridge->synapses[i];
        }
    }
    return NULL;
}

static synapse_entry_t* find_free_slot(mental_health_plasticity_bridge_t* bridge) {
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->max_synapses > 256) {
            mental_health_plasticity_bridge_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)bridge->max_synapses);
        }

        if (!bridge->synapses[i].in_use) {
            return &bridge->synapses[i];
        }
    }
    return NULL;
}

static bool is_protected_type(mental_health_synapse_type_t type) {
    return type == MENTAL_HEALTH_SYNAPSE_MOOD_REGULATION ||
           type == MENTAL_HEALTH_SYNAPSE_DEPRESSION_BUFFER ||
           type == MENTAL_HEALTH_SYNAPSE_RESILIENCE;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

mental_health_plasticity_config_t mental_health_plasticity_config_default(void) {
    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    mental_health_plasticity_config_t config = {
        .base_learning_rate = MENTAL_HEALTH_PLASTICITY_DEFAULT_LR,
        .stdp_tau_plus_ms = 20.0f,
        .stdp_tau_minus_ms = 20.0f,
        .stdp_a_plus = 0.01f,
        .stdp_a_minus = 0.012f,

        .bcm_tau_ms = 1000.0f,
        .bcm_target_rate = 5.0f,

        .homeostatic_tau_ms = 10000.0f,
        .target_mood_level = 0.6f,

        .stress_learning_reduction = 0.3f,
        .resilience_learning_boost = 1.5f,
        .mood_improvement_boost = 1.3f,

        .weight_min = 0.0f,
        .weight_max = 1.0f,

        .protect_mood_regulation = true,
        .protect_depression_buffer = true,
        .protect_resilience = true,
        .protection_strength = 1.0f,

        .max_synapses = MENTAL_HEALTH_PLASTICITY_MAX_SYNAPSES,

        .enable_bio_async = false
    };
    return config;
}

mental_health_plasticity_bridge_t* mental_health_plasticity_create(
    const mental_health_plasticity_config_t* config
) {
    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    mental_health_plasticity_bridge_t* bridge = nimcp_calloc(1, sizeof(mental_health_plasticity_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = mental_health_plasticity_config_default();
    }

    /* Initialize bridge base */
    if (bridge_base_init(&bridge->base, 0, "mental_health_plasticity") != 0) {
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

    /* Initialize regulation state */
    bridge->regulation_state.mood_regulation_strength = 0.5f;
    bridge->regulation_state.anxiety_coping_calibration = 0.5f;
    bridge->regulation_state.depression_resistance = 0.5f;
    bridge->regulation_state.stress_resilience = 0.5f;
    bridge->regulation_state.social_support_sensitivity = 0.5f;
    bridge->regulation_state.learning_rate_mod = 1.0f;
    bridge->regulation_state.last_learning_us = 0;

    bridge->state = MENTAL_HEALTH_PLASTICITY_STATE_IDLE;
    bridge->current_time_us = 0;
    bridge->bio_async_connected = false;
    bridge->synapse_count = 0;

    bridge->current_stress_level = 0.0f;
    bridge->learning_rate_effective = bridge->config.base_learning_rate;
    bridge->bcm_global_threshold = bridge->config.bcm_target_rate;

    return bridge;
}

void mental_health_plasticity_destroy(mental_health_plasticity_bridge_t* bridge) {
    if (!bridge) return;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    bridge_base_cleanup(&bridge->base);
    nimcp_free(bridge->synapses);
    nimcp_free(bridge);
}

int mental_health_plasticity_reset(mental_health_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    /* Reset all synapses to initial weights */
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->max_synapses > 256) {
            mental_health_plasticity_bridge_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)bridge->max_synapses);
        }

        if (bridge->synapses[i].in_use) {
            bridge->synapses[i].synapse.weight = bridge->synapses[i].synapse.initial_weight;
            bridge->synapses[i].synapse.eligibility_trace = 0.0f;
            bridge->synapses[i].synapse.bcm_threshold = bridge->config.bcm_target_rate;
            bridge->synapses[i].synapse.avg_activity = 0.0f;
            bridge->synapses[i].synapse.update_count = 0;
        }
    }

    /* Reset regulation state */
    bridge->regulation_state.mood_regulation_strength = 0.5f;
    bridge->regulation_state.anxiety_coping_calibration = 0.5f;
    bridge->regulation_state.depression_resistance = 0.5f;
    bridge->regulation_state.stress_resilience = 0.5f;
    bridge->regulation_state.learning_rate_mod = 1.0f;

    bridge->state = MENTAL_HEALTH_PLASTICITY_STATE_IDLE;
    bridge->current_stress_level = 0.0f;
    bridge->learning_rate_effective = bridge->config.base_learning_rate;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

//=============================================================================
// Synapse Management
//=============================================================================

int mental_health_plasticity_register_synapse(
    mental_health_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    mental_health_synapse_type_t type,
    float initial_weight
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


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

    /* Auto-protect critical mental health pathways */
    slot->synapse.is_protected = is_protected_type(type) &&
        (bridge->config.protect_mood_regulation ||
         bridge->config.protect_depression_buffer ||
         bridge->config.protect_resilience);

    bridge->synapse_count++;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_plasticity_unregister_synapse(
    mental_health_plasticity_bridge_t* bridge,
    uint32_t synapse_id
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


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

int mental_health_plasticity_get_synapse(
    mental_health_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    mental_health_plasticity_synapse_t* synapse
) {
    if (!bridge || !synapse) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


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

int mental_health_plasticity_protect_synapse(
    mental_health_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    bool protect
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


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

int mental_health_plasticity_learn(
    mental_health_plasticity_bridge_t* bridge,
    mental_health_learn_event_t event,
    float magnitude,
    uint32_t synapse_id,
    float context
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = MENTAL_HEALTH_PLASTICITY_STATE_LEARNING;

    synapse_entry_t* entry = find_synapse(bridge, synapse_id);
    if (!entry) {
        bridge->state = MENTAL_HEALTH_PLASTICITY_STATE_IDLE;
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Check protection */
    if (entry->synapse.is_protected) {
        bridge->stats.protected_updates_blocked++;
        bridge->state = MENTAL_HEALTH_PLASTICITY_STATE_IDLE;
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0; /* Not an error, just blocked */
    }

    /* Apply stress reduction to learning rate */
    float stress_factor = 1.0f - (bridge->current_stress_level * bridge->config.stress_learning_reduction);
    float lr = bridge->learning_rate_effective * magnitude * stress_factor;
    float weight_change = 0.0f;

    switch (event) {
        case MENTAL_HEALTH_LEARN_MOOD_IMPROVED:
            weight_change = lr * bridge->config.mood_improvement_boost;
            bridge->stats.mood_improvement_events++;
            break;

        case MENTAL_HEALTH_LEARN_MOOD_DECLINED:
            weight_change = -lr * 0.5f;
            bridge->stats.mood_decline_events++;
            break;

        case MENTAL_HEALTH_LEARN_ANXIETY_REDUCED:
            weight_change = lr * 1.0f;
            break;

        case MENTAL_HEALTH_LEARN_ANXIETY_INCREASED:
            weight_change = -lr * 0.3f;
            break;

        case MENTAL_HEALTH_LEARN_COPING_SUCCESS:
            weight_change = lr * 1.2f;
            bridge->stats.coping_success_events++;
            break;

        case MENTAL_HEALTH_LEARN_COPING_FAILURE:
            weight_change = -lr * 0.2f;
            break;

        case MENTAL_HEALTH_LEARN_STRESS_RECOVERED:
            weight_change = lr * 0.8f;
            break;

        case MENTAL_HEALTH_LEARN_STRESS_PROLONGED:
            weight_change = -lr * 0.4f;
            break;

        case MENTAL_HEALTH_LEARN_RESILIENCE_DEMONSTRATED:
            weight_change = lr * bridge->config.resilience_learning_boost;
            bridge->stats.resilience_events++;
            break;

        case MENTAL_HEALTH_LEARN_SOCIAL_SUPPORT_RECEIVED:
            weight_change = lr * 0.7f;
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

    bridge->state = MENTAL_HEALTH_PLASTICITY_STATE_IDLE;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

float mental_health_plasticity_apply_stdp(
    mental_health_plasticity_bridge_t* bridge,
    uint32_t synapse_id,
    float pre_time,
    float post_time
) {
    if (!bridge) return NAN;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


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

int mental_health_plasticity_apply_stress(
    mental_health_plasticity_bridge_t* bridge,
    float stress_level
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    stress_level = clamp_f(stress_level, 0.0f, 1.0f);
    bridge->current_stress_level = stress_level;

    /* Stress reduces effective learning rate */
    float stress_factor = 1.0f - (stress_level * bridge->config.stress_learning_reduction);
    bridge->learning_rate_effective = bridge->config.base_learning_rate * stress_factor;

    /* Update stress resilience in regulation state based on how well we cope */
    /* High stress with intact learning = high resilience */
    if (stress_level > 0.5f && bridge->learning_rate_effective > bridge->config.base_learning_rate * 0.7f) {
        bridge->regulation_state.stress_resilience += 0.01f;
        if (bridge->regulation_state.stress_resilience > 1.0f) {
            bridge->regulation_state.stress_resilience = 1.0f;
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_plasticity_update_bcm(
    mental_health_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;
    if (dt_ms <= 0.0f) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = MENTAL_HEALTH_PLASTICITY_STATE_UPDATING;

    float decay = expf(-dt_ms / bridge->config.bcm_tau_ms);

    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->max_synapses > 256) {
            mental_health_plasticity_bridge_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)bridge->max_synapses);
        }

        if (bridge->synapses[i].in_use) {
            /* Update sliding threshold towards average activity */
            float target = bridge->synapses[i].synapse.avg_activity;
            bridge->synapses[i].synapse.bcm_threshold =
                bridge->synapses[i].synapse.bcm_threshold * decay +
                target * (1.0f - decay);
        }
    }

    bridge->state = MENTAL_HEALTH_PLASTICITY_STATE_IDLE;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_plasticity_homeostatic_update(
    mental_health_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = MENTAL_HEALTH_PLASTICITY_STATE_UPDATING;

    float decay = expf(-dt_ms / bridge->config.homeostatic_tau_ms);

    /* Calculate mean mood regulation */
    float mean_mood_regulation = 0.0f;
    uint32_t mood_count = 0;
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->max_synapses > 256) {
            mental_health_plasticity_bridge_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)bridge->max_synapses);
        }

        if (bridge->synapses[i].in_use &&
            bridge->synapses[i].synapse.type == MENTAL_HEALTH_SYNAPSE_MOOD_REGULATION) {
            mean_mood_regulation += bridge->synapses[i].synapse.weight;
            mood_count++;
        }
    }
    if (mood_count > 0) {
        mean_mood_regulation /= mood_count;
    }

    /* Scale non-protected synapses toward target */
    float target = bridge->config.target_mood_level;
    float scale_factor = 1.0f;
    if (mean_mood_regulation > 0.0f) {
        scale_factor = target / mean_mood_regulation;
        scale_factor = clamp_f(scale_factor, 0.9f, 1.1f);
    }

    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->max_synapses > 256) {
            mental_health_plasticity_bridge_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)bridge->max_synapses);
        }

        if (bridge->synapses[i].in_use && !bridge->synapses[i].synapse.is_protected) {
            float scaled = bridge->synapses[i].synapse.weight * (1.0f + (scale_factor - 1.0f) * (1.0f - decay));
            bridge->synapses[i].synapse.weight = clamp_f(
                scaled,
                bridge->config.weight_min,
                bridge->config.weight_max
            );
        }
    }

    bridge->state = MENTAL_HEALTH_PLASTICITY_STATE_IDLE;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_plasticity_update_traces(
    mental_health_plasticity_bridge_t* bridge,
    float dt_ms
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    float decay = expf(-dt_ms / bridge->config.stdp_tau_plus_ms);

    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->max_synapses > 256) {
            mental_health_plasticity_bridge_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)bridge->max_synapses);
        }

        if (bridge->synapses[i].in_use) {
            bridge->synapses[i].synapse.eligibility_trace *= decay;
        }
    }

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int mental_health_plasticity_consolidate(mental_health_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->state = MENTAL_HEALTH_PLASTICITY_STATE_CONSOLIDATING;

    /* Clear eligibility traces */
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->max_synapses > 256) {
            mental_health_plasticity_bridge_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)bridge->max_synapses);
        }

        if (bridge->synapses[i].in_use) {
            bridge->synapses[i].synapse.eligibility_trace = 0.0f;
        }
    }

    /* Update regulation state based on synapse weights */
    float mood_sum = 0.0f, mood_count = 0;
    float anxiety_sum = 0.0f, anxiety_count = 0;
    float depression_sum = 0.0f, depression_count = 0;
    float resilience_sum = 0.0f, resilience_count = 0;
    float social_sum = 0.0f, social_count = 0;

    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->max_synapses > 256) {
            mental_health_plasticity_bridge_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)bridge->max_synapses);
        }

        if (bridge->synapses[i].in_use) {
            switch (bridge->synapses[i].synapse.type) {
                case MENTAL_HEALTH_SYNAPSE_MOOD_REGULATION:
                    mood_sum += bridge->synapses[i].synapse.weight;
                    mood_count++;
                    break;
                case MENTAL_HEALTH_SYNAPSE_ANXIETY_RESPONSE:
                    anxiety_sum += bridge->synapses[i].synapse.weight;
                    anxiety_count++;
                    break;
                case MENTAL_HEALTH_SYNAPSE_DEPRESSION_BUFFER:
                    depression_sum += bridge->synapses[i].synapse.weight;
                    depression_count++;
                    break;
                case MENTAL_HEALTH_SYNAPSE_RESILIENCE:
                    resilience_sum += bridge->synapses[i].synapse.weight;
                    resilience_count++;
                    break;
                case MENTAL_HEALTH_SYNAPSE_SOCIAL_SUPPORT:
                    social_sum += bridge->synapses[i].synapse.weight;
                    social_count++;
                    break;
                default:
                    break;
            }
        }
    }

    if (mood_count > 0) {
        bridge->regulation_state.mood_regulation_strength = mood_sum / mood_count;
    }
    if (anxiety_count > 0) {
        bridge->regulation_state.anxiety_coping_calibration = anxiety_sum / anxiety_count;
    }
    if (depression_count > 0) {
        bridge->regulation_state.depression_resistance = depression_sum / depression_count;
    }
    if (resilience_count > 0) {
        bridge->regulation_state.stress_resilience = resilience_sum / resilience_count;
    }
    if (social_count > 0) {
        bridge->regulation_state.social_support_sensitivity = social_sum / social_count;
    }

    bridge->regulation_state.last_learning_us = bridge->current_time_us;
    bridge->state = MENTAL_HEALTH_PLASTICITY_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

//=============================================================================
// State Query Functions
//=============================================================================

int mental_health_plasticity_get_regulation_state(
    mental_health_plasticity_bridge_t* bridge,
    mental_health_regulation_state_t* state
) {
    if (!bridge || !state) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->regulation_state;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int mental_health_plasticity_get_state(
    mental_health_plasticity_bridge_t* bridge,
    mental_health_plasticity_bridge_state_t* state
) {
    if (!bridge || !state) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    state->state = bridge->state;
    state->active_synapses = bridge->synapse_count;
    state->learning_rate_effective = bridge->learning_rate_effective;

    /* Calculate mean weight and variance */
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < bridge->max_synapses; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->max_synapses > 256) {
            mental_health_plasticity_bridge_heartbeat("mental_healt_loop",
                             (float)(i + 1) / (float)bridge->max_synapses);
        }

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

int mental_health_plasticity_get_stats(
    mental_health_plasticity_bridge_t* bridge,
    mental_health_plasticity_stats_t* stats
) {
    if (!bridge || !stats) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int mental_health_plasticity_reset_stats(mental_health_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(mental_health_plasticity_stats_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

//=============================================================================
// Callback Registration
//=============================================================================

int mental_health_plasticity_register_learn_callback(
    mental_health_plasticity_bridge_t* bridge,
    mental_health_plasticity_learn_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->learn_callback = callback;
    bridge->learn_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int mental_health_plasticity_register_regulation_callback(
    mental_health_plasticity_bridge_t* bridge,
    mental_health_plasticity_regulation_callback_t callback,
    void* user_data
) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->regulation_callback = callback;
    bridge->regulation_callback_data = user_data;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int mental_health_plasticity_bio_async_connect(mental_health_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->config.enable_bio_async) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    /* Bio-async connection would be implemented here */
    bridge->bio_async_connected = true;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int mental_health_plasticity_bio_async_disconnect(mental_health_plasticity_bridge_t* bridge) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->bio_async_connected = false;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

bool mental_health_plasticity_is_bio_async_connected(mental_health_plasticity_bridge_t* bridge) {
    if (!bridge) return false;

    /* Phase 8: Heartbeat at operation start */
    mental_health_plasticity_bridge_heartbeat("mental_healt_mental_health_plasti", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bool connected = bridge->bio_async_connected;
    nimcp_mutex_unlock(bridge->base.mutex);

    return connected;
}
