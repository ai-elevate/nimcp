/**
 * @file nimcp_snn_prefrontal_bridge.c
 * @brief Implementation of SNN-prefrontal cortex bridge
 */

#include "snn/bridges/nimcp_snn_prefrontal_bridge.h"
#include "utils/validation/nimcp_common.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "async/nimcp_bio_async.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* Bio-async module ID for prefrontal bridge */
#define BIO_MODULE_SNN_PREFRONTAL 0x0609

//=============================================================================
// Default Configuration
//=============================================================================

void snn_prefrontal_config_default(snn_prefrontal_config_t* config) {
    if (!config) return;

    /* Persistent activity */
    config->persistent_baseline_rate = 2.0f;    /* 2 Hz baseline */
    config->persistent_active_rate = 30.0f;     /* 30 Hz during delay */
    config->recurrent_excitation = 0.7f;
    config->decay_time_constant = 200.0f;       /* 200ms decay */

    /* Working memory */
    config->max_wm_items = 7;                   /* Miller's 7±2 */
    config->neurons_per_item = 50;
    config->wm_threshold = 0.4f;

    /* Goals */
    config->num_goal_populations = 5;
    config->goal_encoding_rate = 25.0f;
    config->goal_switch_cost = 50.0f;           /* 50ms penalty */

    /* Inhibitory control */
    config->inhibitory_neuron_ratio = 0.2f;     /* 20% inhibitory */
    config->inhibition_strength = 0.8f;
    config->go_threshold = 0.6f;
    config->stop_threshold = 0.7f;

    /* Decision accumulation */
    config->accumulator_leak = 0.1f;
    config->decision_threshold = 1.0f;
    config->evidence_weight = 0.3f;
    config->enable_urgency_signal = true;

    /* Subregions */
    config->dlpfc_ratio = 0.5f;
    config->vmpfc_ratio = 0.3f;
    config->ofc_ratio = 0.2f;

    /* Bio-async */
    config->enable_bio_async = true;
}

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Create WM item
 */
static wm_item_t* create_wm_item(
    uint32_t item_id,
    const float* features,
    uint32_t feature_dim,
    uint32_t n_neurons
) {
    wm_item_t* item = nimcp_malloc(sizeof(wm_item_t));
    if (!item) return NULL;

    item->item_id = item_id;
    item->feature_dim = feature_dim;
    item->n_neurons = n_neurons;
    item->activation_strength = 0.0f;
    item->is_active = false;
    item->onset_time = 0;

    item->feature_vector = nimcp_malloc(feature_dim * sizeof(float));
    item->population_rates = nimcp_malloc(n_neurons * sizeof(float));

    if (!item->feature_vector || !item->population_rates) {
        if (item->feature_vector) nimcp_free(item->feature_vector);
        if (item->population_rates) nimcp_free(item->population_rates);
        nimcp_free(item);
        return NULL;
    }

    memcpy(item->feature_vector, features, feature_dim * sizeof(float));
    memset(item->population_rates, 0, n_neurons * sizeof(float));

    return item;
}

/**
 * @brief Destroy WM item
 */
static void destroy_wm_item(wm_item_t* item) {
    if (!item) return;
    if (item->feature_vector) nimcp_free(item->feature_vector);
    if (item->population_rates) nimcp_free(item->population_rates);
    nimcp_free(item);
}

/**
 * @brief Create goal state
 */
static goal_state_t* create_goal_state(uint32_t goal_id, const char* name) {
    goal_state_t* goal = nimcp_malloc(sizeof(goal_state_t));
    if (!goal) return NULL;

    goal->goal_id = goal_id;
    strncpy(goal->goal_name, name ? name : "unnamed", 63);
    goal->goal_name[63] = '\0';
    goal->priority = 0.5f;
    goal->activation = 0.0f;
    goal->is_active = false;
    goal->population_id = goal_id;
    goal->mean_spike_rate = 0.0f;

    return goal;
}

//=============================================================================
// Bridge Lifecycle
//=============================================================================

snn_prefrontal_bridge_t* snn_prefrontal_bridge_create(
    const snn_prefrontal_config_t* config,
    snn_network_t* network,
    brain_region_t* prefrontal_region
) {
    /* Guard clauses */
    if (!config || !network || !prefrontal_region) {
        NIMCP_LOGGING_ERROR("Null parameter in snn_prefrontal_bridge_create");
        return NULL;
    }

    /* Allocate bridge */
    snn_prefrontal_bridge_t* bridge = nimcp_malloc(sizeof(snn_prefrontal_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate prefrontal bridge");
        return NULL;
    }

    /* Initialize fields */
    memset(bridge, 0, sizeof(snn_prefrontal_bridge_t));
    bridge->network = network;
    bridge->prefrontal_region = prefrontal_region;
    bridge->config = *config;
    bridge->connected = true;
    bridge->max_wm_items = config->max_wm_items;

    /* Allocate WM storage */
    bridge->wm_items = nimcp_malloc(bridge->max_wm_items * sizeof(wm_item_t*));
    if (!bridge->wm_items) {
        NIMCP_LOGGING_ERROR("Failed to allocate WM storage");
        nimcp_free(bridge);
        return NULL;
    }
    memset(bridge->wm_items, 0, bridge->max_wm_items * sizeof(wm_item_t*));
    bridge->n_wm_items = 0;

    /* Allocate goal states */
    bridge->n_goals = config->num_goal_populations;
    bridge->goals = nimcp_malloc(bridge->n_goals * sizeof(goal_state_t*));
    if (!bridge->goals) {
        nimcp_free(bridge->wm_items);
        nimcp_free(bridge);
        return NULL;
    }

    for (uint32_t i = 0; i < bridge->n_goals; i++) {
        char name[64];
        snprintf(name, sizeof(name), "goal_%u", i);
        bridge->goals[i] = create_goal_state(i, name);
        if (!bridge->goals[i]) {
            for (uint32_t j = 0; j < i; j++) {
                nimcp_free(bridge->goals[j]);
            }
            nimcp_free(bridge->goals);
            nimcp_free(bridge->wm_items);
            nimcp_free(bridge);
            return NULL;
        }
    }
    bridge->active_goal = NULL;

    /* Allocate decision accumulator */
    bridge->accumulator = nimcp_malloc(sizeof(decision_accumulator_t));
    if (!bridge->accumulator) {
        for (uint32_t i = 0; i < bridge->n_goals; i++) {
            nimcp_free(bridge->goals[i]);
        }
        nimcp_free(bridge->goals);
        nimcp_free(bridge->wm_items);
        nimcp_free(bridge);
        return NULL;
    }
    memset(bridge->accumulator, 0, sizeof(decision_accumulator_t));
    bridge->accumulator->n_options = 2;  /* Binary decision default */
    bridge->accumulator->evidence = nimcp_malloc(2 * sizeof(float));
    bridge->accumulator->evidence_rates = nimcp_malloc(2 * sizeof(float));
    if (!bridge->accumulator->evidence || !bridge->accumulator->evidence_rates) {
        if (bridge->accumulator->evidence) nimcp_free(bridge->accumulator->evidence);
        if (bridge->accumulator->evidence_rates) nimcp_free(bridge->accumulator->evidence_rates);
        nimcp_free(bridge->accumulator);
        for (uint32_t i = 0; i < bridge->n_goals; i++) {
            nimcp_free(bridge->goals[i]);
        }
        nimcp_free(bridge->goals);
        nimcp_free(bridge->wm_items);
        nimcp_free(bridge);
        return NULL;
    }
    memset(bridge->accumulator->evidence, 0, 2 * sizeof(float));
    memset(bridge->accumulator->evidence_rates, 0, 2 * sizeof(float));

    /* Allocate inhibitory control */
    bridge->inhibition = nimcp_malloc(sizeof(inhibitory_control_t));
    if (!bridge->inhibition) {
        nimcp_free(bridge->accumulator->evidence);
        nimcp_free(bridge->accumulator->evidence_rates);
        nimcp_free(bridge->accumulator);
        for (uint32_t i = 0; i < bridge->n_goals; i++) {
            nimcp_free(bridge->goals[i]);
        }
        nimcp_free(bridge->goals);
        nimcp_free(bridge->wm_items);
        nimcp_free(bridge);
        return NULL;
    }
    memset(bridge->inhibition, 0, sizeof(inhibitory_control_t));
    bridge->inhibition->n_inhibitory = 100;  /* Placeholder */
    bridge->inhibition->n_excitatory = 400;

    /* NOTE: Populations would be created here in full implementation */
    bridge->dlpfc_pop = NULL;
    bridge->vmpfc_pop = NULL;
    bridge->ofc_pop = NULL;
    bridge->inhibitory_pop = NULL;

    /* Initialize statistics */
    bridge->total_decisions = 0;
    bridge->inhibited_responses = 0;
    bridge->mean_decision_time = 0.0f;

    NIMCP_LOGGING_INFO("Created SNN-prefrontal bridge with %u goal populations",
                       bridge->n_goals);
    return bridge;
}

void snn_prefrontal_bridge_destroy(snn_prefrontal_bridge_t* bridge) {
    if (!bridge) return;

    /* Free WM items */
    if (bridge->wm_items) {
        for (uint32_t i = 0; i < bridge->n_wm_items; i++) {
            destroy_wm_item(bridge->wm_items[i]);
        }
        nimcp_free(bridge->wm_items);
    }

    /* Free goals */
    if (bridge->goals) {
        for (uint32_t i = 0; i < bridge->n_goals; i++) {
            nimcp_free(bridge->goals[i]);
        }
        nimcp_free(bridge->goals);
    }

    /* Free accumulator */
    if (bridge->accumulator) {
        if (bridge->accumulator->evidence) nimcp_free(bridge->accumulator->evidence);
        if (bridge->accumulator->evidence_rates) nimcp_free(bridge->accumulator->evidence_rates);
        nimcp_free(bridge->accumulator);
    }

    /* Free inhibitory control */
    if (bridge->inhibition) {
        nimcp_free(bridge->inhibition);
    }

    /* Disconnect bio-async */
    if (bridge->bio_async_enabled) {
        snn_prefrontal_bridge_disconnect_bio_async(bridge);
    }

    nimcp_free(bridge);
}

//=============================================================================
// Bio-async Integration
//=============================================================================

int snn_prefrontal_bridge_connect_bio_async(snn_prefrontal_bridge_t* bridge) {
    if (!bridge) return -1;
    if (bridge->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_SNN_PREFRONTAL,
        .module_name = "snn_prefrontal_bridge",
        .inbox_capacity = 64,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    }
    return 0;
}

int snn_prefrontal_bridge_disconnect_bio_async(snn_prefrontal_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return 0;

    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }
    bridge->bio_async_enabled = false;
    return 0;
}

bool snn_prefrontal_bridge_is_bio_async_connected(const snn_prefrontal_bridge_t* bridge) {
    return bridge && bridge->bio_async_enabled;
}

//=============================================================================
// Processing Functions
//=============================================================================

int snn_prefrontal_bridge_process(
    snn_prefrontal_bridge_t* bridge,
    const float* input,
    uint32_t input_size,
    float* output,
    uint32_t output_size
) {
    /* Guard clauses */
    if (!bridge || !input || !output) return -1;
    if (!bridge->connected) return -1;

    /* Process input - would update populations */
    /* Placeholder: copy input to output */
    uint32_t copy_size = (output_size < input_size) ? output_size : input_size;
    memcpy(output, input, copy_size * sizeof(float));

    /* Update active goal if set */
    if (bridge->active_goal) {
        bridge->active_goal->mean_spike_rate = bridge->config.goal_encoding_rate;
    }

    return 0;
}

int snn_prefrontal_bridge_update(snn_prefrontal_bridge_t* bridge, float dt) {
    if (!bridge) return -1;
    if (!bridge->connected) return -1;

    /* Update time */
    bridge->last_update_time += dt;
    bridge->update_count++;

    /* Decay WM items */
    float decay_factor = expf(-dt / bridge->config.decay_time_constant);
    for (uint32_t i = 0; i < bridge->n_wm_items; i++) {
        wm_item_t* item = bridge->wm_items[i];
        if (!item) continue;

        item->activation_strength *= decay_factor;

        /* Remove if below threshold */
        if (item->activation_strength < bridge->config.wm_threshold) {
            item->is_active = false;
        }
    }

    /* Update decision accumulator with leak */
    if (bridge->accumulator && !bridge->accumulator->decision_made) {
        for (uint32_t i = 0; i < bridge->accumulator->n_options; i++) {
            bridge->accumulator->evidence[i] *= (1.0f - bridge->config.accumulator_leak * dt / 1000.0f);

            /* Check for threshold crossing */
            if (bridge->accumulator->evidence[i] >= bridge->config.decision_threshold) {
                bridge->accumulator->decision_made = true;
                bridge->accumulator->chosen_option = i;
                bridge->accumulator->decision_time = bridge->last_update_time;
                bridge->total_decisions++;
                break;
            }
        }
    }

    return 0;
}

uint32_t snn_prefrontal_add_wm_item(
    snn_prefrontal_bridge_t* bridge,
    const float* features,
    uint32_t feature_dim
) {
    if (!bridge || !features) return UINT32_MAX;
    if (bridge->n_wm_items >= bridge->max_wm_items) return UINT32_MAX;

    wm_item_t* item = create_wm_item(
        bridge->n_wm_items,
        features,
        feature_dim,
        bridge->config.neurons_per_item
    );

    if (!item) return UINT32_MAX;

    item->is_active = true;
    item->activation_strength = 1.0f;
    item->onset_time = (uint64_t)(bridge->last_update_time * 1000.0);

    bridge->wm_items[bridge->n_wm_items] = item;
    uint32_t item_id = bridge->n_wm_items;
    bridge->n_wm_items++;

    NIMCP_LOGGING_DEBUG("Added WM item %u (total: %u)", item_id, bridge->n_wm_items);

    return item_id;
}

int snn_prefrontal_remove_wm_item(snn_prefrontal_bridge_t* bridge, uint32_t item_id) {
    if (!bridge) return -1;
    if (item_id >= bridge->n_wm_items) return -1;

    wm_item_t* item = bridge->wm_items[item_id];
    if (!item) return -1;

    destroy_wm_item(item);
    bridge->wm_items[item_id] = NULL;

    /* Compact array if removing last item */
    if (item_id == bridge->n_wm_items - 1) {
        bridge->n_wm_items--;
    }

    return 0;
}

int snn_prefrontal_set_goal(snn_prefrontal_bridge_t* bridge, uint32_t goal_id) {
    if (!bridge) return -1;
    if (goal_id >= bridge->n_goals) return -1;

    /* Deactivate current goal */
    if (bridge->active_goal) {
        bridge->active_goal->is_active = false;
        bridge->active_goal->activation = 0.0f;
    }

    /* Activate new goal */
    goal_state_t* new_goal = bridge->goals[goal_id];
    new_goal->is_active = true;
    new_goal->activation = 1.0f;
    bridge->active_goal = new_goal;

    NIMCP_LOGGING_DEBUG("Set active goal to %u (%s)", goal_id, new_goal->goal_name);

    return 0;
}

int snn_prefrontal_accumulate_evidence(
    snn_prefrontal_bridge_t* bridge,
    const float* evidence,
    uint32_t n_options
) {
    if (!bridge || !evidence) return -1;
    if (!bridge->accumulator) return -1;
    if (bridge->accumulator->decision_made) return 1;

    /* Resize if needed */
    if (n_options != bridge->accumulator->n_options) {
        float* new_evidence = nimcp_malloc(n_options * sizeof(float));
        float* new_rates = nimcp_malloc(n_options * sizeof(float));
        if (!new_evidence || !new_rates) {
            if (new_evidence) nimcp_free(new_evidence);
            if (new_rates) nimcp_free(new_rates);
            return -1;
        }
        nimcp_free(bridge->accumulator->evidence);
        nimcp_free(bridge->accumulator->evidence_rates);
        bridge->accumulator->evidence = new_evidence;
        bridge->accumulator->evidence_rates = new_rates;
        bridge->accumulator->n_options = n_options;
        memset(bridge->accumulator->evidence, 0, n_options * sizeof(float));
    }

    /* Accumulate evidence */
    for (uint32_t i = 0; i < n_options; i++) {
        bridge->accumulator->evidence[i] += evidence[i] * bridge->config.evidence_weight;
        bridge->accumulator->evidence_rates[i] = evidence[i];
    }

    return 0;
}

int snn_prefrontal_apply_inhibition(snn_prefrontal_bridge_t* bridge, float stop_signal) {
    if (!bridge || !bridge->inhibition) return -1;

    bridge->inhibition->stop_signal = stop_signal;

    if (stop_signal >= bridge->config.stop_threshold) {
        bridge->inhibition->response_suppressed = true;
        bridge->inhibition->inhibition_onset_time = bridge->last_update_time;
        bridge->inhibited_responses++;
        NIMCP_LOGGING_DEBUG("Response inhibited (signal: %.2f)", stop_signal);
    } else {
        bridge->inhibition->response_suppressed = false;
    }

    return 0;
}

//=============================================================================
// Query Functions
//=============================================================================

uint32_t snn_prefrontal_get_wm_count(const snn_prefrontal_bridge_t* bridge) {
    if (!bridge) return 0;

    uint32_t count = 0;
    for (uint32_t i = 0; i < bridge->n_wm_items; i++) {
        if (bridge->wm_items[i] && bridge->wm_items[i]->is_active) {
            count++;
        }
    }
    return count;
}

const wm_item_t* snn_prefrontal_get_wm_item(
    const snn_prefrontal_bridge_t* bridge,
    uint32_t item_id
) {
    if (!bridge) return NULL;
    if (item_id >= bridge->n_wm_items) return NULL;
    return bridge->wm_items[item_id];
}

const goal_state_t* snn_prefrontal_get_active_goal(const snn_prefrontal_bridge_t* bridge) {
    return bridge ? bridge->active_goal : NULL;
}

bool snn_prefrontal_is_decision_made(const snn_prefrontal_bridge_t* bridge) {
    if (!bridge || !bridge->accumulator) return false;
    return bridge->accumulator->decision_made;
}

uint32_t snn_prefrontal_get_decision(const snn_prefrontal_bridge_t* bridge) {
    if (!bridge || !bridge->accumulator) return UINT32_MAX;
    if (!bridge->accumulator->decision_made) return UINT32_MAX;
    return bridge->accumulator->chosen_option;
}

float snn_prefrontal_bridge_get_activity(const snn_prefrontal_bridge_t* bridge) {
    if (!bridge) return -1.0f;

    /* Combine WM activity and goal activation */
    float wm_activity = 0.0f;
    uint32_t active_wm = snn_prefrontal_get_wm_count(bridge);
    wm_activity = (float)active_wm / (float)bridge->max_wm_items;

    float goal_activity = bridge->active_goal ? bridge->active_goal->activation : 0.0f;

    return (wm_activity + goal_activity) / 2.0f;
}

//=============================================================================
// Statistics
//=============================================================================

int snn_prefrontal_get_stats(
    const snn_prefrontal_bridge_t* bridge,
    uint32_t* total_decisions,
    uint32_t* inhibited_responses,
    uint32_t* updates
) {
    if (!bridge) return -1;

    if (total_decisions) {
        *total_decisions = bridge->total_decisions;
    }

    if (inhibited_responses) {
        *inhibited_responses = bridge->inhibited_responses;
    }

    if (updates) {
        *updates = bridge->update_count;
    }

    return 0;
}

void snn_prefrontal_reset_stats(snn_prefrontal_bridge_t* bridge) {
    if (!bridge) return;

    bridge->update_count = 0;
    bridge->last_update_time = 0.0f;
    bridge->total_decisions = 0;
    bridge->inhibited_responses = 0;
    bridge->mean_decision_time = 0.0f;

    /* Reset accumulator */
    if (bridge->accumulator) {
        bridge->accumulator->decision_made = false;
        bridge->accumulator->chosen_option = 0;
        bridge->accumulator->decision_time = 0.0f;
        memset(bridge->accumulator->evidence, 0,
               bridge->accumulator->n_options * sizeof(float));
    }
}
