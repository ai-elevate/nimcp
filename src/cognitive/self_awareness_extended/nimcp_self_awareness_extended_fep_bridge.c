/**
 * @file nimcp_self_awareness_extended_fep_bridge.c
 * @brief Extended Self-Awareness FEP Bridge Implementation
 */

#include "cognitive/self_awareness_extended/nimcp_self_awareness_extended_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>

#define LOG_MODULE "self_awareness_extended_fep_bridge"

int self_awareness_extended_fep_bridge_default_config(self_awareness_extended_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->uncertainty_threshold = SELF_AWARENESS_FEP_HIGH_UNCERTAINTY_THRESHOLD;
    config->coherence_factor = SELF_AWARENESS_FEP_COHERENCE_FACTOR;
    config->enable_metacognitive_monitoring = true;
    config->enable_precision_modulation = true;
    config->enable_self_harm_detection = true;
    config->enable_narrative_coherence = true;
    config->metacognition_sensitivity = 0.7f;
    config->temporal_continuity_weight = 0.8f;
    config->enable_agency_attribution = true;
    config->fe_sensitivity = 1.0f;
    config->awareness_sensitivity = 1.0f;
    return 0;
}

self_awareness_extended_fep_bridge_t* self_awareness_extended_fep_bridge_create(const self_awareness_extended_fep_config_t* config) {
    self_awareness_extended_fep_bridge_t* bridge = nimcp_malloc(sizeof(self_awareness_extended_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(self_awareness_extended_fep_bridge_t));
    if (config) bridge->config = *config;
    else self_awareness_extended_fep_bridge_default_config(&bridge->config);
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }
    return bridge;
}

void self_awareness_extended_fep_bridge_destroy(self_awareness_extended_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) self_awareness_extended_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int self_awareness_extended_fep_bridge_connect_fep(self_awareness_extended_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_extended_fep_bridge_connect_awareness(self_awareness_extended_fep_bridge_t* bridge, self_awareness_system_t awareness) {
    if (!bridge || !awareness) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->awareness_system = awareness;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_extended_fep_bridge_disconnect(self_awareness_extended_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    bridge->awareness_system = NULL;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_extended_fep_trigger_monitoring(self_awareness_extended_fep_bridge_t* bridge, float uncertainty) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_metacognitive_monitoring) return 0;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_effects.current_uncertainty = uncertainty;
    if (uncertainty > bridge->config.uncertainty_threshold) {
        bridge->fep_effects.metacognitive_monitoring_triggered = true;
        bridge->state.monitoring_active = true;
        bridge->stats.monitoring_events++;
        NIMCP_LOGGING_INFO("Metacognitive monitoring triggered (uncertainty=%.2f)", uncertainty);
    }
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_extended_fep_check_self_harm(self_awareness_extended_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_self_harm_detection) return 0;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_effects.self_harm_check_active = true;
    bridge->stats.self_harm_detections++;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_extended_fep_modulate_depth(self_awareness_extended_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_precision_modulation) return 0;
    nimcp_mutex_lock(bridge->mutex);
    bridge->stats.regulation_actions++;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_extended_fep_apply_narrative_coherence(self_awareness_extended_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_narrative_coherence) return 0;
    nimcp_mutex_lock(bridge->mutex);
    bridge->awareness_effects.self_model_updating_beliefs = true;
    bridge->stats.belief_updates++;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_extended_fep_update_from_regulation(self_awareness_extended_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->awareness_effects.agency_attribution_active = true;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_extended_fep_bridge_update(self_awareness_extended_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    self_awareness_extended_fep_check_self_harm(bridge);
    self_awareness_extended_fep_modulate_depth(bridge);
    self_awareness_extended_fep_apply_narrative_coherence(bridge);
    self_awareness_extended_fep_update_from_regulation(bridge);
    return 0;
}

int self_awareness_extended_fep_bridge_get_state(const self_awareness_extended_fep_bridge_t* bridge, self_awareness_extended_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_extended_fep_bridge_get_stats(const self_awareness_extended_fep_bridge_t* bridge, self_awareness_extended_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int self_awareness_extended_fep_bridge_connect_bio_async(self_awareness_extended_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_SELF_AWARENESS_BRIDGE,
        .module_name = "self_awareness_extended_fep_bridge",
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

int self_awareness_extended_fep_bridge_disconnect_bio_async(self_awareness_extended_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return 0;
    if (bridge->bio_ctx) bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_async_enabled = false;
    return 0;
}

bool self_awareness_extended_fep_bridge_is_bio_async_connected(const self_awareness_extended_fep_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}
