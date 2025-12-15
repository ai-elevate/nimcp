/**
 * @file nimcp_hierarchical_fep_bridge.c
 * @brief Free Energy Principle - Hierarchical Brain Integration Bridge Implementation
 */

#include
#include "utils/error/nimcp_error_codes.h" "cognitive/nimcp_hierarchical_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>
#include <math.h>

#define LOG_MODULE "hierarchical_fep_bridge"

int hierarchical_fep_bridge_default_config(hierarchical_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->enable_hierarchical_prediction = true;
    config->enable_pe_propagation = true;
    config->enable_layer_specific_lr = true;
    config->hierarchy_sensitivity = 1.0f;
    config->fep_sensitivity = 1.0f;
    return 0;
}

hierarchical_fep_bridge_t* hierarchical_fep_bridge_create(const hierarchical_fep_config_t* config) {
    hierarchical_fep_bridge_t* bridge = nimcp_malloc(sizeof(hierarchical_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate hierarchical FEP bridge");
        return NULL;
    }
    memset(bridge, 0, sizeof(hierarchical_fep_bridge_t));
    if (config) {
        bridge->config = *config;
    } else {
        hierarchical_fep_bridge_default_config(&bridge->config);
    }
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }
    NIMCP_LOGGING_INFO("Created hierarchical FEP bridge");
    return bridge;
}

void hierarchical_fep_bridge_destroy(hierarchical_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) {
        hierarchical_fep_bridge_disconnect_bio_async(bridge);
    }
    if (bridge->state.level_states) {
        nimcp_free(bridge->state.level_states);
    }
    if (bridge->fep_effects.level_free_energies) {
        nimcp_free(bridge->fep_effects.level_free_energies);
    }
    if (bridge->fep_effects.level_prediction_errors) {
        nimcp_free(bridge->fep_effects.level_prediction_errors);
    }
    if (bridge->hierarchical_effects.level_lr_modifiers) {
        nimcp_free(bridge->hierarchical_effects.level_lr_modifiers);
    }
    if (bridge->hierarchical_effects.level_precisions) {
        nimcp_free(bridge->hierarchical_effects.level_precisions);
    }
    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
    }
    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Destroyed hierarchical FEP bridge");
}

int hierarchical_fep_bridge_connect_fep(hierarchical_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);
    NIMCP_LOGGING_INFO("Connected FEP system to hierarchical bridge");
    return 0;
}

int hierarchical_fep_bridge_connect_hierarchical(hierarchical_fep_bridge_t* bridge,
                                                  hierarchical_brain_t hbrain) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->hierarchical_brain = hbrain;
    nimcp_mutex_unlock(bridge->mutex);
    NIMCP_LOGGING_INFO("Connected hierarchical brain to FEP bridge");
    return 0;
}

int hierarchical_fep_bridge_disconnect(hierarchical_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    bridge->hierarchical_brain = NULL;
    nimcp_mutex_unlock(bridge->mutex);
    NIMCP_LOGGING_INFO("Disconnected all systems from hierarchical FEP bridge");
    return 0;
}

int hierarchical_fep_bridge_update(hierarchical_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_INVALID_STATE;
    nimcp_mutex_lock(bridge->mutex);
    bridge->state.current_free_energy = fep_get_free_energy(bridge->fep_system);
    bridge->stats.avg_free_energy =
        (bridge->stats.avg_free_energy * 0.99f) + (bridge->state.current_free_energy * 0.01f);
    bridge->stats.prediction_events++;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int hierarchical_fep_bridge_get_state(const hierarchical_fep_bridge_t* bridge,
                                       hierarchical_fep_state_t* state) {
    if (!bridge || !state) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int hierarchical_fep_bridge_get_stats(const hierarchical_fep_bridge_t* bridge,
                                       hierarchical_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_mutex_lock(bridge->mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int hierarchical_fep_bridge_connect_bio_async(hierarchical_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_HIERARCHICAL_BRIDGE,
        .module_name = "hierarchical_fep_bridge",
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

int hierarchical_fep_bridge_disconnect_bio_async(hierarchical_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return 0;
    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }
    bridge->bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Disconnected from bio-async router");
    return 0;
}

bool hierarchical_fep_bridge_is_bio_async_connected(const hierarchical_fep_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->bio_async_enabled;
}
