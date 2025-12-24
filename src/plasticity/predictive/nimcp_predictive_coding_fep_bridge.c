/**
 * @file nimcp_predictive_coding_fep_bridge.c
 * @brief Predictive Coding FEP Bridge Implementation
 */

#include "plasticity/predictive/nimcp_predictive_coding_fep_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>
#include <math.h>

int predictive_coding_fep_bridge_default_config(predictive_coding_fep_config_t* config) {
    if (!config) return -1;
    config->precision_sensitivity = 1.0f;
    config->error_scaling = 1.0f;
    config->learning_rate_modulation = 0.5f;
    config->enable_precision_weighting = true;
    config->enable_hierarchical_fe = true;
    config->enable_convergence_detection = true;
    return 0;
}

predictive_coding_fep_bridge_t* predictive_coding_fep_bridge_create(const predictive_coding_fep_config_t* config) {
    predictive_coding_fep_bridge_t* bridge = (predictive_coding_fep_bridge_t*)nimcp_malloc(sizeof(predictive_coding_fep_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        memcpy(&bridge->config, config, sizeof(predictive_coding_fep_config_t));
    } else {
        predictive_coding_fep_bridge_default_config(&bridge->config);
    }

    memset(&bridge->fep_effects, 0, sizeof(predictive_coding_fep_effects_t));
    memset(&bridge->pc_effects, 0, sizeof(predictive_coding_fep_feedback_t));
    memset(&bridge->stats, 0, sizeof(predictive_coding_fep_stats_t));

    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->fep_system = NULL;
    bridge->pc_hierarchy = NULL;
    bridge->base.bio_async_enabled = false;

    return bridge;
}

void predictive_coding_fep_bridge_destroy(predictive_coding_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->base.bio_async_enabled) predictive_coding_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->fep_effects.level_free_energies) nimcp_free(bridge->fep_effects.level_free_energies);
    if (bridge->pc_effects.prediction_errors) nimcp_free(bridge->pc_effects.prediction_errors);
    if (bridge->base.mutex) nimcp_platform_mutex_destroy(bridge->base.mutex);
    nimcp_free(bridge);
}

int predictive_coding_fep_bridge_connect_fep(predictive_coding_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_coding_fep_bridge_connect_pc(predictive_coding_fep_bridge_t* bridge, pc_hierarchy_t hierarchy) {
    if (!bridge || !hierarchy) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->pc_hierarchy = hierarchy;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_coding_fep_bridge_disconnect(predictive_coding_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    bridge->fep_system = NULL;
    bridge->pc_hierarchy = NULL;
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

float predictive_coding_fep_apply_precision_weighting(predictive_coding_fep_bridge_t* bridge, uint32_t level, float error) {
    if (!bridge || !bridge->config.enable_precision_weighting) return error;
    if (level >= bridge->fep_effects.num_levels) return error;
    return error * bridge->fep_effects.precision_scaling * bridge->config.precision_sensitivity;
}

float predictive_coding_fep_compute_hierarchical_free_energy(const predictive_coding_fep_bridge_t* bridge) {
    if (!bridge || !bridge->config.enable_hierarchical_fe) return 0.0f;
    return bridge->fep_effects.total_free_energy;
}

int predictive_coding_fep_report_errors(predictive_coding_fep_bridge_t* bridge, const float* errors, uint32_t num_levels) {
    if (!bridge || !errors) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    if (bridge->pc_effects.prediction_errors) nimcp_free(bridge->pc_effects.prediction_errors);
    bridge->pc_effects.prediction_errors = (float*)nimcp_malloc(sizeof(float) * num_levels);
    if (bridge->pc_effects.prediction_errors) {
        memcpy(bridge->pc_effects.prediction_errors, errors, sizeof(float) * num_levels);
        bridge->pc_effects.num_levels = num_levels;
        float sum = 0.0f;
        for (uint32_t i = 0; i < num_levels; i++) sum += fabsf(errors[i]);
        bridge->pc_effects.mean_error = sum / num_levels;
    }
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_coding_fep_bridge_update(predictive_coding_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    if (bridge->fep_system && bridge->pc_hierarchy) {
        bridge->fep_effects.total_free_energy = pc_hierarchy_get_free_energy(bridge->pc_hierarchy);
        bridge->stats.total_updates++;
        bridge->stats.avg_free_energy = (bridge->stats.avg_free_energy * (bridge->stats.total_updates - 1) + bridge->fep_effects.total_free_energy) / bridge->stats.total_updates;
        bridge->stats.avg_prediction_error = (bridge->stats.avg_prediction_error * (bridge->stats.total_updates - 1) + bridge->pc_effects.mean_error) / bridge->stats.total_updates;
    }
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_coding_fep_bridge_get_stats(const predictive_coding_fep_bridge_t* bridge, predictive_coding_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->base.mutex);
    memcpy(stats, &bridge->stats, sizeof(predictive_coding_fep_stats_t));
    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int predictive_coding_fep_bridge_connect_bio_async(predictive_coding_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    bridge->base.bio_async_enabled = false;
    return 0;
}

int predictive_coding_fep_bridge_disconnect_bio_async(predictive_coding_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    bridge->base.bio_async_enabled = false;
    return 0;
}

bool predictive_coding_fep_bridge_is_bio_async_connected(const predictive_coding_fep_bridge_t* bridge) {
    return bridge ? bridge->base.bio_async_enabled : false;
}
