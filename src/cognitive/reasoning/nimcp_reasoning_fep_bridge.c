/**
 * @file nimcp_reasoning_fep_bridge.c
 * @brief Free Energy Principle - Reasoning Integration Bridge Implementation
 */

#include "cognitive/reasoning/nimcp_reasoning_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE "reasoning_fep_bridge"

int reasoning_fep_bridge_default_config(reasoning_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->pe_abduction_threshold = REASONING_FEP_PE_ABDUCTION_THRESHOLD;
    config->hypothesis_selection_temperature = 1.0f;
    config->inference_precision_threshold = REASONING_FEP_PRECISION_THRESHOLD;
    config->enable_pe_abduction = true;
    config->enable_fe_hypothesis_selection = true;
    config->enable_precision_inference = true;
    config->rule_prior_strength = 1.0f;
    config->conclusion_belief_strength = 0.9f;
    config->explanation_fe_reduction = 0.5f;
    config->enable_rule_priors = true;
    config->enable_conclusion_constraints = true;
    config->enable_explanation_reduction = true;
    config->fe_sensitivity = 1.0f;
    config->reasoning_sensitivity = 1.0f;
    return NIMCP_OK;
}

reasoning_fep_bridge_t* reasoning_fep_bridge_create(const reasoning_fep_config_t* config) {
    reasoning_fep_bridge_t* bridge = nimcp_malloc(sizeof(reasoning_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(reasoning_fep_bridge_t));
    if (config) bridge->config = *config;
    else reasoning_fep_bridge_default_config(&bridge->config);
    bridge->mutex = nimcp_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }
    return bridge;
}

void reasoning_fep_bridge_destroy(reasoning_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) reasoning_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int reasoning_fep_bridge_connect_fep(reasoning_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int reasoning_fep_bridge_connect_reasoning(reasoning_fep_bridge_t* bridge, reasoning_integration_t* reasoning) {
    if (!bridge || !reasoning) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->reasoning_system = reasoning;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int reasoning_fep_bridge_disconnect(reasoning_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    bridge->reasoning_system = NULL;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int reasoning_fep_trigger_abduction(reasoning_fep_bridge_t* bridge, float pe_magnitude) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_pe_abduction) return NIMCP_OK;
    nimcp_mutex_lock(bridge->mutex);
    if (pe_magnitude > bridge->config.pe_abduction_threshold) {
        bridge->fep_effects.abduction_triggered = true;
        bridge->state.abduction_active = true;
        bridge->stats.abduction_events++;
    }
    bridge->fep_effects.current_prediction_error = pe_magnitude;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int reasoning_fep_select_hypothesis_by_fe(reasoning_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_fe_hypothesis_selection) return NIMCP_OK;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_effects.selected_hypothesis = 0;
    bridge->stats.hypothesis_selections++;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int reasoning_fep_modulate_inference_confidence(reasoning_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_precision_inference) return NIMCP_OK;
    nimcp_mutex_lock(bridge->mutex);
    float precision = bridge->state.current_precision;
    bridge->fep_effects.inference_confidence = precision;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int reasoning_fep_apply_rule_priors(reasoning_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_rule_priors) return NIMCP_OK;
    nimcp_mutex_lock(bridge->mutex);
    bridge->reasoning_effects.rule_prior_bias = bridge->config.rule_prior_strength;
    bridge->reasoning_effects.rule_priors_active = true;
    bridge->stats.rule_prior_applications++;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int reasoning_fep_apply_conclusion_constraints(reasoning_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_conclusion_constraints) return NIMCP_OK;
    nimcp_mutex_lock(bridge->mutex);
    bridge->reasoning_effects.conclusion_constraint_strength = bridge->config.conclusion_belief_strength;
    bridge->reasoning_effects.conclusions_constraining_beliefs = true;
    bridge->stats.conclusion_constraints++;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int reasoning_fep_apply_explanation_reduction(reasoning_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_explanation_reduction) return NIMCP_OK;
    nimcp_mutex_lock(bridge->mutex);
    bridge->reasoning_effects.fe_reduction = bridge->config.explanation_fe_reduction;
    bridge->stats.explanation_reductions++;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int reasoning_fep_bridge_update(reasoning_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    reasoning_fep_select_hypothesis_by_fe(bridge);
    reasoning_fep_modulate_inference_confidence(bridge);
    reasoning_fep_apply_rule_priors(bridge);
    reasoning_fep_apply_conclusion_constraints(bridge);
    reasoning_fep_apply_explanation_reduction(bridge);
    return NIMCP_OK;
}

int reasoning_fep_bridge_get_state(const reasoning_fep_bridge_t* bridge, reasoning_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int reasoning_fep_bridge_get_stats(const reasoning_fep_bridge_t* bridge, reasoning_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_OK;
}

int reasoning_fep_bridge_connect_bio_async(reasoning_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return NIMCP_OK;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_REASONING_BRIDGE,
        .module_name = "reasoning_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };
    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) bridge->bio_async_enabled = true;
    return NIMCP_OK;
}

int reasoning_fep_bridge_disconnect_bio_async(reasoning_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return NIMCP_OK;
    if (bridge->bio_ctx) bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_async_enabled = false;
    return NIMCP_OK;
}

bool reasoning_fep_bridge_is_bio_async_connected(const reasoning_fep_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}
