/**
 * @file nimcp_introspection_fep_bridge.c
 * @brief Introspection FEP Bridge Implementation
 */

#include "cognitive/introspection/nimcp_introspection_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE_INTROSPECTION_FEP "[INTROSPECTION_FEP]"

int introspection_fep_bridge_default_config(introspection_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->pe_threshold = INTROSPECTION_FEP_HIGH_PE_THRESHOLD;
    config->uncertainty_threshold = INTROSPECTION_FEP_HIGH_UNCERTAINTY;
    config->meta_learning_rate = INTROSPECTION_FEP_META_UPDATE_RATE;
    config->enable_precision_monitoring = true;
    config->enable_meta_learning = true;
    config->pe_sensitivity = 1.0f;
    return NIMCP_SUCCESS;
}

introspection_fep_bridge_t* introspection_fep_bridge_create(const introspection_fep_config_t* config) {
    introspection_fep_bridge_t* bridge = (introspection_fep_bridge_t*)nimcp_malloc(sizeof(introspection_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(introspection_fep_bridge_t));
    if (config) bridge->config = *config;
    else introspection_fep_bridge_default_config(&bridge->config);
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }
    bridge->state.current_precision = 0.5f;
    NIMCP_LOGGING_INFO(LOG_MODULE_INTROSPECTION_FEP " Bridge created");
    return bridge;
}

void introspection_fep_bridge_destroy(introspection_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) introspection_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int introspection_fep_bridge_connect_fep(introspection_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int introspection_fep_bridge_connect_introspection(introspection_fep_bridge_t* bridge, introspection_context_t intro) {
    if (!bridge || !intro) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->introspection_system = intro;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int introspection_fep_estimate_precision(introspection_fep_bridge_t* bridge, float* precision) {
    if (!bridge || !precision) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *precision = bridge->state.current_precision;
    bridge->effects.precision_estimate = *precision;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int introspection_fep_monitor_uncertainty(introspection_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->effects.uncertainty_estimate = bridge->state.current_uncertainty;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int introspection_fep_meta_learn(introspection_fep_bridge_t* bridge, float prediction_error) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_meta_learning) return NIMCP_SUCCESS;
    nimcp_mutex_lock(bridge->mutex);
    float pe_abs = fabsf(prediction_error);
    if (pe_abs > bridge->config.pe_threshold) {
        float lr = bridge->config.meta_learning_rate;
        bridge->state.current_precision *= (1.0f - lr);
        bridge->state.current_uncertainty += lr * 0.5f;
        if (bridge->state.current_uncertainty > 1.0f) bridge->state.current_uncertainty = 1.0f;
        bridge->state.pe_events_monitored++;
    }
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int introspection_fep_bridge_update(introspection_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->effects.meta_confidence = 1.0f - bridge->state.current_uncertainty;
    bridge->stats.avg_precision = (bridge->stats.avg_precision * 0.99f) + (bridge->state.current_precision * 0.01f);
    bridge->stats.avg_uncertainty = (bridge->stats.avg_uncertainty * 0.99f) + (bridge->state.current_uncertainty * 0.01f);
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int introspection_fep_bridge_get_state(const introspection_fep_bridge_t* bridge, introspection_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int introspection_fep_bridge_get_stats(const introspection_fep_bridge_t* bridge, introspection_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int introspection_fep_bridge_connect_bio_async(introspection_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return NIMCP_SUCCESS;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_INTROSPECTION_BRIDGE,
        .module_name = "introspection_fep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };
    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        return NIMCP_SUCCESS;
    }
    return -1;
}

int introspection_fep_bridge_disconnect_bio_async(introspection_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return NIMCP_SUCCESS;
    if (bridge->bio_ctx) bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_async_enabled = false;
    return NIMCP_SUCCESS;
}

bool introspection_fep_bridge_is_bio_async_connected(const introspection_fep_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}
