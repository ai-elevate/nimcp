/**
 * @file nimcp_symbolic_logic_fep_bridge.c
 * @brief Symbolic Logic FEP Bridge Implementation
 */

#include "cognitive/symbolic_logic/nimcp_symbolic_logic_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>

#define LOG_MODULE "symbolic_logic_fep_bridge"

int symbolic_logic_fep_bridge_default_config(symbolic_logic_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->pe_exploration_threshold = LOGIC_FEP_HIGH_PE_THRESHOLD;
    config->proof_precision_factor = LOGIC_FEP_PROOF_PRECISION_FACTOR;
    config->enable_pe_exploration = true;
    config->enable_proof_validation = true;
    config->enable_novelty_weighting = true;
    config->enable_salience_precision = true;
    config->fe_sensitivity = 1.0f;
    config->logic_sensitivity = 1.0f;
    return 0;
}

symbolic_logic_fep_bridge_t* symbolic_logic_fep_bridge_create(const symbolic_logic_fep_config_t* config) {
    symbolic_logic_fep_bridge_t* bridge = nimcp_malloc(sizeof(symbolic_logic_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(symbolic_logic_fep_bridge_t));
    if (config) bridge->config = *config;
    else symbolic_logic_fep_bridge_default_config(&bridge->config);
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }
    return bridge;
}

void symbolic_logic_fep_bridge_destroy(symbolic_logic_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) symbolic_logic_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int symbolic_logic_fep_bridge_connect_fep(symbolic_logic_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int symbolic_logic_fep_bridge_connect_logic(symbolic_logic_fep_bridge_t* bridge, symbolic_logic_t* logic) {
    if (!bridge || !logic) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->logic_system = logic;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int symbolic_logic_fep_bridge_disconnect(symbolic_logic_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    bridge->logic_system = NULL;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int symbolic_logic_fep_trigger_exploration(symbolic_logic_fep_bridge_t* bridge, float pe_magnitude) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_pe_exploration) return 0;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_effects.current_prediction_error = pe_magnitude;
    if (pe_magnitude > bridge->config.pe_exploration_threshold) {
        bridge->fep_effects.logical_exploration_triggered = true;
        bridge->state.exploration_active = true;
        bridge->stats.exploration_events++;
        NIMCP_LOGGING_INFO("Logical exploration triggered (PE=%.2f)", pe_magnitude);
    }
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int symbolic_logic_fep_weight_facts_by_confidence(symbolic_logic_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_salience_precision) return 0;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_effects.num_salient_facts = 0;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int symbolic_logic_fep_validate_beliefs_by_proof(symbolic_logic_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_proof_validation) return 0;
    nimcp_mutex_lock(bridge->mutex);
    bridge->logic_effects.logic_validating_beliefs = true;
    bridge->stats.proof_validations++;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int symbolic_logic_fep_trigger_revision_from_contradiction(symbolic_logic_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->logic_effects.belief_revision_triggered = true;
    bridge->stats.belief_revisions++;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int symbolic_logic_fep_bridge_update(symbolic_logic_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    symbolic_logic_fep_weight_facts_by_confidence(bridge);
    symbolic_logic_fep_validate_beliefs_by_proof(bridge);
    return 0;
}

int symbolic_logic_fep_bridge_get_state(const symbolic_logic_fep_bridge_t* bridge, symbolic_logic_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int symbolic_logic_fep_bridge_get_stats(const symbolic_logic_fep_bridge_t* bridge, symbolic_logic_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int symbolic_logic_fep_bridge_connect_bio_async(symbolic_logic_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_LOGIC_BRIDGE,
        .module_name = "symbolic_logic_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };
    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    }
    return 0;
}

int symbolic_logic_fep_bridge_disconnect_bio_async(symbolic_logic_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return 0;
    if (bridge->bio_ctx) bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_async_enabled = false;
    return 0;
}

bool symbolic_logic_fep_bridge_is_bio_async_connected(const symbolic_logic_fep_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}
