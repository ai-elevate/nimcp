/**
 * @file nimcp_bcm_fep_bridge.c
 * @brief Free Energy Principle - BCM Integration Bridge Implementation
 */

#include "plasticity/bcm/nimcp_bcm_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>

#define LOG_MODULE_BCM_FEP "BCM_FEP_BRIDGE"

int bcm_fep_bridge_default_config(bcm_fep_config_t* config) {
    if (!config) return -1;
    config->complexity_threshold_gain = BCM_FEP_THRESHOLD_SCALING;
    config->precision_selectivity_gain = BCM_FEP_SELECTIVITY_GAIN;
    config->pe_lr_sensitivity = 1.0f;
    config->enable_complexity_regularization = true;
    config->enable_precision_modulation = true;
    config->enable_pe_gating = true;
    config->enable_sparsity_tracking = true;
    config->lr_min_factor = BCM_FEP_LR_MIN_FACTOR;
    config->lr_max_factor = BCM_FEP_LR_MAX_FACTOR;
    return 0;
}

bcm_fep_bridge_t* bcm_fep_bridge_create(const bcm_fep_config_t* config) {
    bcm_fep_bridge_t* bridge = (bcm_fep_bridge_t*)nimcp_malloc(sizeof(bcm_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(bcm_fep_bridge_t));

    if (config) {
        bridge->config = *config;
    } else {
        bcm_fep_bridge_default_config(&bridge->config);
    }

    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.total_threshold_modulation = 1.0f;
    bridge->effects.total_lr_modulation = 1.0f;
    NIMCP_LOGGING_INFO("BCM-FEP bridge created");
    return bridge;
}

void bcm_fep_bridge_destroy(bcm_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.bio_async_enabled) bcm_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->base.mutex) nimcp_platform_mutex_destroy(bridge->base.mutex);
    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("BCM-FEP bridge destroyed");
}

int bcm_fep_bridge_connect_fep(bcm_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge) return -1;
    /* Allow NULL fep to disconnect/reset FEP connection */
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int bcm_fep_bridge_connect_bcm(bcm_fep_bridge_t* bridge, bcm_synapse_t* bcm, uint32_t num) {
    if (!bridge || !bcm || num == 0) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->bcm_system = bcm;
    bridge->num_synapses = num;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int bcm_fep_bridge_disconnect(bcm_fep_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    bridge->bcm_system = NULL;
    bridge->num_synapses = 0;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

float bcm_fep_apply_complexity_regularization(bcm_fep_bridge_t* bridge, float complexity) {
    if (!bridge || !bridge->config.enable_complexity_regularization) return 1.0f;
    float clamped = fminf(fmaxf(complexity, BCM_FEP_COMPLEXITY_MIN), BCM_FEP_COMPLEXITY_MAX);
    float scaling = 1.0f + clamped * bridge->config.complexity_threshold_gain;
    /* Store values for get_effective_threshold to use */
    bridge->effects.complexity_value = complexity;
    bridge->effects.complexity_threshold_scaling = scaling;
    bridge->effects.total_threshold_modulation = scaling * bridge->effects.precision_selectivity_scaling;
    if (bridge->effects.total_threshold_modulation == 0.0f) {
        bridge->effects.total_threshold_modulation = scaling;
    }
    return scaling;
}

float bcm_fep_apply_precision_modulation(bcm_fep_bridge_t* bridge, float precision) {
    if (!bridge || !bridge->config.enable_precision_modulation) return 1.0f;
    return precision * bridge->config.precision_selectivity_gain;
}

float bcm_fep_apply_pe_gating(bcm_fep_bridge_t* bridge, float pe) {
    if (!bridge || !bridge->config.enable_pe_gating) return 1.0f;
    float scaling = 1.0f + fabsf(pe) * bridge->config.pe_lr_sensitivity;
    return fminf(fmaxf(scaling, bridge->config.lr_min_factor), bridge->config.lr_max_factor);
}

float bcm_fep_get_effective_threshold(const bcm_fep_bridge_t* bridge, float base_threshold) {
    if (!bridge) return base_threshold;
    return base_threshold * bridge->effects.total_threshold_modulation;
}

float bcm_fep_get_effective_lr(const bcm_fep_bridge_t* bridge, float base_lr) {
    if (!bridge) return base_lr;
    return base_lr * bridge->effects.total_lr_modulation;
}

int bcm_fep_report_threshold_changes(bcm_fep_bridge_t* bridge, float threshold_delta) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->stats.complexity_adjustments++;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

float bcm_fep_compute_sparsity(const bcm_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bcm_system) return 0.0f;
    uint32_t active = 0;
    for (uint32_t i = 0; i < bridge->num_synapses; i++) {
        if (bridge->bcm_system[i].weight > 0.1f) active++;
    }
    return (float)active / (float)bridge->num_synapses;
}

int bcm_fep_report_sparsity(bcm_fep_bridge_t* bridge, uint32_t active_count) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->state.sparsity_level = active_count;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int bcm_fep_bridge_update(bcm_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->base.mutex);

    if (bridge->fep_system) {
        float complexity_scaling = bcm_fep_apply_complexity_regularization(bridge, bridge->effects.complexity_value);
        float precision_scaling = bcm_fep_apply_precision_modulation(bridge, bridge->effects.precision_value);
        float pe_scaling = bcm_fep_apply_pe_gating(bridge, bridge->effects.pe_magnitude);

        bridge->effects.complexity_threshold_scaling = complexity_scaling;
        bridge->effects.precision_selectivity_scaling = precision_scaling;
        bridge->effects.pe_lr_scaling = pe_scaling;
        bridge->effects.total_threshold_modulation = complexity_scaling * precision_scaling;
        bridge->effects.total_lr_modulation = pe_scaling;

        bridge->stats.avg_complexity =
            (bridge->stats.avg_complexity * bridge->stats.total_updates + bridge->effects.complexity_value) /
            (bridge->stats.total_updates + 1);
    }

    bridge->stats.total_updates++;
    bridge->state.last_update_time = delta_ms;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int bcm_fep_bridge_get_state(const bcm_fep_bridge_t* bridge, bcm_fep_state_t* state) {
    if (!bridge || !state) return -1;
    *state = bridge->state;
    return 0;
}

int bcm_fep_bridge_get_stats(const bcm_fep_bridge_t* bridge, bcm_fep_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

int bcm_fep_bridge_connect_bio_async(bcm_fep_bridge_t* bridge) {
    if (!bridge) return -1;
    if (bridge->base.bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_BCM_BRIDGE,
        .module_name = "bcm_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };
    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    }
    return 0;
}

int bcm_fep_bridge_disconnect_bio_async(bcm_fep_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->base.bio_async_enabled) return 0;  /* Already disconnected - success */
    bio_router_unregister_module(bridge->base.bio_ctx);
    bridge->base.bio_ctx = NULL;
    bridge->base.bio_async_enabled = false;
    return 0;
}

bool bcm_fep_bridge_is_bio_async_connected(const bcm_fep_bridge_t* bridge) {
    return bridge ? bridge->base.bio_async_enabled : false;
}
