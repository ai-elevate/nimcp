/**
 * @file nimcp_executive_fep_bridge.c
 * @brief Free Energy Principle - Executive Function Integration Bridge Implementation
 */

#include "cognitive/executive/nimcp_executive_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE "executive_fep_bridge"

int executive_fep_bridge_default_config(executive_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->efe_temperature = EXECUTIVE_FEP_DEFAULT_TEMPERATURE;
    config->precision_exploration_threshold = EXECUTIVE_FEP_PRECISION_THRESHOLD;
    config->pe_control_threshold = EXECUTIVE_FEP_PE_CONTROL_THRESHOLD;
    config->enable_efe_policy_selection = true;
    config->enable_precision_exploration = true;
    config->enable_pe_cognitive_control = true;
    config->goal_prior_strength = 1.0f;
    config->wm_belief_persistence = 0.9f;
    config->inhibition_precision_reduction = 0.5f;
    config->enable_goal_priors = true;
    config->enable_wm_belief_maintenance = true;
    config->enable_inhibition_precision = true;
    config->efe_sensitivity = 1.0f;
    config->executive_sensitivity = 1.0f;
    return NIMCP_OK;
}

executive_fep_bridge_t* executive_fep_bridge_create(const executive_fep_config_t* config) {
    executive_fep_bridge_t* bridge = nimcp_malloc(sizeof(executive_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(executive_fep_bridge_t));
    if (config) bridge->config = *config;
    else executive_fep_bridge_default_config(&bridge->config);
    bridge->mutex = nimcp_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }
    return bridge;
}

void executive_fep_bridge_destroy(executive_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) executive_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int executive_fep_bridge_connect_fep(executive_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int executive_fep_bridge_connect_executive(executive_fep_bridge_t* bridge, executive_controller_t* executive) {
    if (!bridge || !executive) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->executive_system = executive;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int executive_fep_bridge_disconnect(executive_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    bridge->executive_system = NULL;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int executive_fep_select_policy_by_efe(executive_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_efe_policy_selection) return NIMCP_OK;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_effects.selected_policy = 0;
    bridge->stats.policy_selections++;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int executive_fep_modulate_exploration(executive_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_precision_exploration) return NIMCP_OK;
    nimcp_mutex_lock(bridge->mutex);
    float precision = bridge->state.current_precision;
    if (precision < bridge->config.precision_exploration_threshold) {
        bridge->fep_effects.exploration_mode = true;
        bridge->fep_effects.exploration_probability = EXECUTIVE_FEP_LOW_PRECISION_EXPLORE;
        bridge->stats.exploration_events++;
    } else {
        bridge->fep_effects.exploration_mode = false;
        bridge->fep_effects.exploration_probability = 1.0f - EXECUTIVE_FEP_HIGH_PRECISION_EXPLOIT;
    }
    bridge->state.exploration_mode = bridge->fep_effects.exploration_mode;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int executive_fep_trigger_cognitive_control(executive_fep_bridge_t* bridge, float pe_magnitude) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_pe_cognitive_control) return NIMCP_OK;
    nimcp_mutex_lock(bridge->mutex);
    if (pe_magnitude > bridge->config.pe_control_threshold) {
        bridge->fep_effects.control_active = true;
        bridge->fep_effects.control_signal = EXECUTIVE_FEP_CONTROL_BOOST;
        bridge->stats.cognitive_control_triggers++;
    } else {
        bridge->fep_effects.control_active = false;
        bridge->fep_effects.control_signal = 1.0f;
    }
    bridge->state.control_active = bridge->fep_effects.control_active;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int executive_fep_apply_goal_priors(executive_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_goal_priors) return NIMCP_OK;
    nimcp_mutex_lock(bridge->mutex);
    bridge->executive_effects.goal_prior_bias = bridge->config.goal_prior_strength;
    bridge->executive_effects.goal_prior_active = true;
    bridge->stats.goal_prior_applications++;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int executive_fep_maintain_wm_beliefs(executive_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_wm_belief_maintenance) return NIMCP_OK;
    nimcp_mutex_lock(bridge->mutex);
    bridge->executive_effects.wm_belief_strength = bridge->config.wm_belief_persistence;
    bridge->executive_effects.wm_maintenance_active = true;
    bridge->stats.wm_maintenance_events++;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int executive_fep_apply_inhibition_precision(executive_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_inhibition_precision) return NIMCP_OK;
    nimcp_mutex_lock(bridge->mutex);
    bridge->executive_effects.precision_suppression = bridge->config.inhibition_precision_reduction;
    bridge->stats.inhibition_events++;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int executive_fep_bridge_update(executive_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    executive_fep_select_policy_by_efe(bridge);
    executive_fep_modulate_exploration(bridge);
    executive_fep_apply_goal_priors(bridge);
    executive_fep_maintain_wm_beliefs(bridge);
    executive_fep_apply_inhibition_precision(bridge);
    return NIMCP_OK;
}

int executive_fep_bridge_get_state(const executive_fep_bridge_t* bridge, executive_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int executive_fep_bridge_get_stats(const executive_fep_bridge_t* bridge, executive_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int executive_fep_bridge_connect_bio_async(executive_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return NIMCP_OK;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_EXECUTIVE_BRIDGE,
        .module_name = "executive_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };
    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) bridge->bio_async_enabled = true;
    return NIMCP_OK;
}

int executive_fep_bridge_disconnect_bio_async(executive_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return NIMCP_OK;
    if (bridge->bio_ctx) bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_async_enabled = false;
    return NIMCP_OK;
}

bool executive_fep_bridge_is_bio_async_connected(const executive_fep_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}
