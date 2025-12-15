/**
 * @file nimcp_meta_learning_fep_bridge.c
 * @brief Free Energy Principle - Meta-Learning Integration Bridge Implementation
 */

#include "cognitive/nimcp_meta_learning_fep_bridge.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE "meta_learning_fep_bridge"

int meta_learning_fep_bridge_default_config(meta_learning_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->enable_task_similarity_fe = true;
    config->enable_adaptation_speed_fe = true;
    config->enable_meta_prior_optimization = true;
    config->meta_sensitivity = 1.0f;
    config->fep_sensitivity = 1.0f;
    return 0;
}

meta_learning_fep_bridge_t* meta_learning_fep_bridge_create(const meta_learning_fep_config_t* config) {
    meta_learning_fep_bridge_t* bridge = nimcp_malloc(sizeof(meta_learning_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate meta-learning FEP bridge");
        return NULL;
    }
    memset(bridge, 0, sizeof(meta_learning_fep_bridge_t));
    if (config) {
        bridge->config = *config;
    } else {
        meta_learning_fep_bridge_default_config(&bridge->config);
    }
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }
    NIMCP_LOGGING_INFO("Created meta-learning FEP bridge");
    return bridge;
}

void meta_learning_fep_bridge_destroy(meta_learning_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) {
        meta_learning_fep_bridge_disconnect_bio_async(bridge);
    }
    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
    }
    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed meta-learning FEP bridge");
}

int meta_learning_fep_bridge_connect_fep(meta_learning_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);
    NIMCP_LOGGING_INFO("Connected FEP system to meta-learning bridge");
    return 0;
}

int meta_learning_fep_bridge_connect_meta_learning(meta_learning_fep_bridge_t* bridge,
                                                    meta_learner_t meta) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->meta_learner = meta;
    nimcp_mutex_unlock(bridge->mutex);
    NIMCP_LOGGING_INFO("Connected meta-learner to FEP bridge");
    return 0;
}

int meta_learning_fep_bridge_disconnect(meta_learning_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    bridge->meta_learner = NULL;
    nimcp_mutex_unlock(bridge->mutex);
    NIMCP_LOGGING_INFO("Disconnected all systems from meta-learning FEP bridge");
    return 0;
}

int meta_learning_fep_bridge_update(meta_learning_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_INVALID_STATE;
    nimcp_mutex_lock(bridge->mutex);
    bridge->state.current_free_energy = fep_get_free_energy(bridge->fep_system);
    bridge->stats.avg_free_energy =
        (bridge->stats.avg_free_energy * 0.99f) + (bridge->state.current_free_energy * 0.01f);
    bridge->stats.adaptation_events++;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int meta_learning_fep_bridge_get_state(const meta_learning_fep_bridge_t* bridge,
                                        meta_learning_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int meta_learning_fep_bridge_get_stats(const meta_learning_fep_bridge_t* bridge,
                                        meta_learning_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int meta_learning_fep_bridge_connect_bio_async(meta_learning_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_META_LEARNING_BRIDGE,
        .module_name = "meta_learning_fep_bridge",
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

int meta_learning_fep_bridge_disconnect_bio_async(meta_learning_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return 0;
    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }
    bridge->bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected from bio-async router");
    return 0;
}

bool meta_learning_fep_bridge_is_bio_async_connected(const meta_learning_fep_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->bio_async_enabled;
}
