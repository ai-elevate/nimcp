/**
 * @file nimcp_snn_scaling_bridge.c
 * @brief SNN-Synaptic Scaling Integration Bridge Implementation
 */

#include "snn/bridges/nimcp_snn_scaling_bridge.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>
#include <math.h>

#define BIO_MODULE_SNN_SCALING_BRIDGE 0x0613

void snn_scaling_bridge_config_default(snn_scaling_bridge_config_t* config) {
    if (!config) return;

    config->target_rate_hz = 5.0f;
    config->scaling_exponent = 1.0f;
    config->min_scaling_factor = 0.1f;
    config->max_scaling_factor = 10.0f;
    config->update_interval_ms = 1000.0f; /* 1 second */
    config->bidirectional_updates = true;
    config->enable_bio_async = true;
}

snn_scaling_bridge_t* snn_scaling_bridge_create(
    const snn_scaling_bridge_config_t* config,
    snn_network_t* network,
    synaptic_scaling_state_t* scaling_states,
    uint32_t n_neurons,
    uint32_t n_synapses
) {
    if (!config || !network || !scaling_states || n_neurons == 0 || n_synapses == 0) {
        NIMCP_LOGGING_ERROR("Invalid parameters for SNN-Scaling bridge");
        return NULL;
    }

    snn_scaling_bridge_t* bridge = (snn_scaling_bridge_t*)
        nimcp_malloc(sizeof(snn_scaling_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate SNN-Scaling bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(snn_scaling_bridge_t));

    bridge->network = network;
    bridge->scaling_states = scaling_states;
    bridge->n_neurons = n_neurons;
    bridge->n_synapses = n_synapses;
    bridge->config = *config;

    bridge->effects.avg_scaling_factor = 1.0f;
    bridge->effects.scaling_variance = 0.0f;
    bridge->effects.scaled_synapses = 0;
    bridge->effects.convergence_achieved = false;

    bridge->mutex = nimcp_platform_mutex_create();
    bridge->connected = true;

    if (config->enable_bio_async) {
        snn_scaling_bridge_connect_bio_async(bridge);
    }

    NIMCP_LOGGING_INFO("Created SNN-Scaling bridge: %u neurons, %u synapses",
                       n_neurons, n_synapses);
    return bridge;
}

void snn_scaling_bridge_destroy(snn_scaling_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->bio_async_enabled) {
        snn_scaling_bridge_disconnect_bio_async(bridge);
    }

    if (bridge->mutex) {
        nimcp_platform_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
}

int snn_scaling_bridge_connect_bio_async(snn_scaling_bridge_t* bridge) {
    if (!bridge) return -1;
    if (bridge->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_SNN_SCALING_BRIDGE,
        .module_name = "snn_scaling_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected SNN-Scaling bridge to bio-async");
    }

    return 0;
}

int snn_scaling_bridge_disconnect_bio_async(snn_scaling_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return 0;

    bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_async_enabled = false;
    bridge->bio_ctx = NULL;

    return 0;
}

bool snn_scaling_bridge_is_bio_async_connected(const snn_scaling_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}

int snn_scaling_bridge_compute_factors(snn_scaling_bridge_t* bridge, float dt) {
    if (!bridge) return -1;

    if (bridge->mutex) nimcp_platform_mutex_lock(bridge->mutex);

    float factor_sum = 0.0f;
    float factor_sq_sum = 0.0f;

    for (uint32_t i = 0; i < bridge->n_neurons; i++) {
        synaptic_scaling_state_t* state = &bridge->scaling_states[i];

        /* Compute scaling factor: (target / actual)^exponent */
        float rate = state->average_rate;
        if (rate < 0.001f) rate = 0.001f; /* Prevent division by zero */

        float factor = powf(bridge->config.target_rate_hz / rate,
                           bridge->config.scaling_exponent);

        /* Clamp to valid range */
        factor = fmaxf(factor, bridge->config.min_scaling_factor);
        factor = fminf(factor, bridge->config.max_scaling_factor);

        state->scaling_factor = factor;

        factor_sum += factor;
        factor_sq_sum += factor * factor;
    }

    float n_inv = 1.0f / (float)bridge->n_neurons;
    bridge->effects.avg_scaling_factor = factor_sum * n_inv;

    float mean_sq = factor_sq_sum * n_inv;
    float sq_mean = bridge->effects.avg_scaling_factor * bridge->effects.avg_scaling_factor;
    bridge->effects.scaling_variance = mean_sq - sq_mean;

    if (bridge->mutex) nimcp_platform_mutex_unlock(bridge->mutex);

    return 0;
}

int snn_scaling_bridge_apply_plasticity(snn_scaling_bridge_t* bridge, float dt) {
    if (!bridge) return -1;

    if (bridge->mutex) nimcp_platform_mutex_lock(bridge->mutex);

    /* Apply scaling factors to synapses */
    bridge->scaling_events++;
    bridge->effects.scaled_synapses = bridge->n_synapses;

    if (bridge->mutex) nimcp_platform_mutex_unlock(bridge->mutex);

    return 0;
}

int snn_scaling_bridge_update(snn_scaling_bridge_t* bridge, float dt) {
    if (!bridge) return -1;

    int ret = snn_scaling_bridge_compute_factors(bridge, dt);
    if (ret != 0) return ret;

    ret = snn_scaling_bridge_apply_plasticity(bridge, dt);
    if (ret != 0) return ret;

    bridge->last_update_time_ms += dt;

    return 0;
}

int snn_scaling_bridge_get_weight_changes(
    const snn_scaling_bridge_t* bridge,
    uint32_t* synapse_ids,
    float* scaling_factors,
    uint32_t max_changes,
    uint32_t* n_changes
) {
    if (!bridge || !synapse_ids || !scaling_factors || !n_changes) {
        return -1;
    }

    *n_changes = 0;
    return 0;
}

int snn_scaling_bridge_get_effects(
    const snn_scaling_bridge_t* bridge,
    snn_scaling_effects_t* effects
) {
    if (!bridge || !effects) return -1;

    if (bridge->mutex) {
        nimcp_platform_mutex_lock((void*)bridge->mutex);
    }

    *effects = bridge->effects;

    if (bridge->mutex) {
        nimcp_platform_mutex_unlock((void*)bridge->mutex);
    }

    return 0;
}

float snn_scaling_bridge_get_avg_factor(const snn_scaling_bridge_t* bridge) {
    return bridge ? bridge->effects.avg_scaling_factor : 1.0f;
}

int snn_scaling_bridge_get_stats(
    const snn_scaling_bridge_t* bridge,
    uint32_t* scaling_events,
    uint32_t* updates
) {
    if (!bridge) return -1;

    if (scaling_events) *scaling_events = bridge->scaling_events;
    if (updates) *updates = (uint32_t)(bridge->last_update_time_ms /
                                       bridge->config.update_interval_ms);

    return 0;
}

void snn_scaling_bridge_reset_stats(snn_scaling_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->mutex) nimcp_platform_mutex_lock(bridge->mutex);

    bridge->scaling_events = 0;
    bridge->last_update_time_ms = 0.0f;

    if (bridge->mutex) nimcp_platform_mutex_unlock(bridge->mutex);
}
