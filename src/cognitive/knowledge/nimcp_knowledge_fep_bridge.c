/**
 * @file nimcp_knowledge_fep_bridge.c
 * @brief Knowledge FEP Bridge Implementation
 */

#include "cognitive/knowledge/nimcp_knowledge_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE_KNOWLEDGE_FEP "[KNOWLEDGE_FEP]"

int knowledge_fep_bridge_default_config(knowledge_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->knowledge_update_threshold = KNOWLEDGE_FEP_UPDATE_THRESHOLD;
    config->semantic_prior_weight = KNOWLEDGE_FEP_SEMANTIC_PRIOR_WEIGHT;
    config->enable_knowledge_updates = true;
    config->enable_semantic_priors = true;
    config->pe_sensitivity = 1.0f;
    return NIMCP_SUCCESS;
}

knowledge_fep_bridge_t* knowledge_fep_bridge_create(const knowledge_fep_config_t* config) {
    knowledge_fep_bridge_t* bridge = (knowledge_fep_bridge_t*)nimcp_malloc(sizeof(knowledge_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(knowledge_fep_bridge_t));
    if (config) bridge->config = *config;
    else knowledge_fep_bridge_default_config(&bridge->config);
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }
    bridge->state.current_semantic_prior = 0.5f;
    NIMCP_LOGGING_INFO(LOG_MODULE_KNOWLEDGE_FEP " Bridge created");
    return bridge;
}

void knowledge_fep_bridge_destroy(knowledge_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) knowledge_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int knowledge_fep_bridge_connect_fep(knowledge_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_bridge_connect_knowledge(knowledge_fep_bridge_t* bridge, knowledge_system_t knowledge) {
    if (!bridge || !knowledge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->knowledge_system = knowledge;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_update_knowledge(knowledge_fep_bridge_t* bridge, float prediction_error) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->config.enable_knowledge_updates) return NIMCP_SUCCESS;
    nimcp_mutex_lock(bridge->mutex);
    if (fabsf(prediction_error) > bridge->config.knowledge_update_threshold) {
        bridge->state.knowledge_updates++;
        bridge->stats.updates_total++;
        bridge->effects.semantic_pe = fabsf(prediction_error);
    }
    bridge->stats.avg_semantic_pe = (bridge->stats.avg_semantic_pe * 0.9f) + (fabsf(prediction_error) * 0.1f);
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_apply_semantic_priors(knowledge_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    /* Apply knowledge as semantic priors to FEP */
    bridge->state.current_semantic_prior = bridge->config.semantic_prior_weight;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_bridge_update(knowledge_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->effects.knowledge_confidence = 1.0f - (bridge->effects.semantic_pe / 10.0f);
    if (bridge->effects.knowledge_confidence < 0.0f) bridge->effects.knowledge_confidence = 0.0f;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_bridge_get_state(const knowledge_fep_bridge_t* bridge, knowledge_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_bridge_get_stats(const knowledge_fep_bridge_t* bridge, knowledge_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);
    return NIMCP_SUCCESS;
}

int knowledge_fep_bridge_connect_bio_async(knowledge_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return NIMCP_SUCCESS;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_KNOWLEDGE_BRIDGE,
        .module_name = "knowledge_fep_bridge",
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

int knowledge_fep_bridge_disconnect_bio_async(knowledge_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return NIMCP_SUCCESS;
    if (bridge->bio_ctx) bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_async_enabled = false;
    return NIMCP_SUCCESS;
}

bool knowledge_fep_bridge_is_bio_async_connected(const knowledge_fep_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}
