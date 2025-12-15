/**
 * @file nimcp_consolidation_fep_bridge.c
 * @brief Free Energy Principle - Consolidation Integration Bridge Implementation
 */

#include "cognitive/consolidation/nimcp_consolidation_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE "consolidation_fep_bridge"

int consolidation_fep_bridge_default_config(consolidation_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->complexity_replay_scaling = 1.0f;
    config->fe_urgency_threshold = CONSOLIDATION_FEP_HIGH_FE_THRESHOLD;
    config->enable_complexity_guided_consolidation = true;
    config->enable_pe_replay_selection = true;
    config->enable_fe_urgency = true;
    config->replay_fe_reduction = CONSOLIDATION_FEP_REPLAY_FE_REDUCTION;
    config->pruning_complexity_reduction = CONSOLIDATION_FEP_PRUNING_COMPLEXITY_REDUCTION;
    config->scaling_precision_boost = CONSOLIDATION_FEP_SCALING_PRECISION_BOOST;
    config->enable_replay_fe_reduction = true;
    config->enable_pruning_complexity = true;
    config->enable_scaling_precision = true;
    config->fe_sensitivity = 1.0f;
    config->consolidation_sensitivity = 1.0f;
    return 0;
}

consolidation_fep_bridge_t* consolidation_fep_bridge_create(const consolidation_fep_config_t* config) {
    consolidation_fep_bridge_t* bridge = nimcp_malloc(sizeof(consolidation_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate consolidation FEP bridge");
        return NULL;
    }
    memset(bridge, 0, sizeof(consolidation_fep_bridge_t));
    if (config) {
        bridge->config = *config;
    } else {
        consolidation_fep_bridge_default_config(&bridge->config);
    }
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }
    NIMCP_LOGGING_INFO("Created consolidation FEP bridge");
    return bridge;
}

void consolidation_fep_bridge_destroy(consolidation_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) {
        consolidation_fep_bridge_disconnect_bio_async(bridge);
    }
    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
    }
    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed consolidation FEP bridge");
}

int consolidation_fep_bridge_connect_fep(consolidation_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);
    NIMCP_LOGGING_INFO("Connected FEP system to consolidation bridge");
    return 0;
}

int consolidation_fep_bridge_disconnect(consolidation_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    nimcp_mutex_unlock(bridge->mutex);
    NIMCP_LOGGING_INFO("Disconnected all systems from consolidation FEP bridge");
    return 0;
}

int consolidation_fep_bridge_update(consolidation_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_INVALID_STATE;
    nimcp_mutex_lock(bridge->mutex);
    bridge->state.current_free_energy = fep_get_free_energy(bridge->fep_system);
    bridge->fep_effects.current_free_energy = bridge->state.current_free_energy;
    if (bridge->config.enable_fe_urgency) {
        if (bridge->state.current_free_energy > bridge->config.fe_urgency_threshold) {
            bridge->fep_effects.consolidation_urgency = 1.0f;
        }
    }
    bridge->stats.avg_free_energy =
        (bridge->stats.avg_free_energy * 0.99f) + (bridge->state.current_free_energy * 0.01f);
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int consolidation_fep_bridge_get_state(const consolidation_fep_bridge_t* bridge,
                                        consolidation_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int consolidation_fep_bridge_get_stats(const consolidation_fep_bridge_t* bridge,
                                        consolidation_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int consolidation_fep_bridge_connect_bio_async(consolidation_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_CONSOLIDATION_BRIDGE,
        .module_name = "consolidation_fep_bridge",
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

int consolidation_fep_bridge_disconnect_bio_async(consolidation_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return 0;
    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }
    bridge->bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected from bio-async router");
    return 0;
}

bool consolidation_fep_bridge_is_bio_async_connected(const consolidation_fep_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->bio_async_enabled;
}
