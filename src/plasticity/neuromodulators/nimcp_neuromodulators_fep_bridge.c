/**
 * @file nimcp_neuromodulators_fep_bridge.c
 * @brief Neuromodulators FEP Bridge Implementation
 */

#include "plasticity/neuromodulators/nimcp_neuromodulators_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include <string.h>
#include <math.h>

int neuromod_fep_bridge_default_config(neuromod_fep_config_t* config) {
    if (!config) return -1;
    config->da_pe_sensitivity = NEUROMOD_FEP_DA_PE_GAIN;
    config->ach_precision_sensitivity = NEUROMOD_FEP_ACH_PRECISION_GAIN;
    config->ne_uncertainty_sensitivity = NEUROMOD_FEP_NE_UNCERTAINTY_GAIN;
    config->sht_complexity_sensitivity = NEUROMOD_FEP_5HT_COMPLEXITY_GAIN;
    config->enable_da_pe_coupling = true;
    config->enable_ach_precision_coupling = true;
    config->enable_ne_uncertainty_coupling = true;
    config->enable_sht_complexity_coupling = true;
    return 0;
}

neuromod_fep_bridge_t* neuromod_fep_bridge_create(const neuromod_fep_config_t* config) {
    neuromod_fep_bridge_t* bridge = (neuromod_fep_bridge_t*)nimcp_malloc(sizeof(neuromod_fep_bridge_t));
    if (!bridge) return NULL;

    if (config) memcpy(&bridge->config, config, sizeof(neuromod_fep_config_t));
    else neuromod_fep_bridge_default_config(&bridge->config);

    memset(&bridge->fep_effects, 0, sizeof(neuromod_fep_effects_t));
    memset(&bridge->neuromod_effects, 0, sizeof(neuromod_fep_feedback_t));
    memset(&bridge->stats, 0, sizeof(neuromod_fep_stats_t));

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    bridge->fep_system = NULL;
    bridge->neuromod_system = NULL;
    bridge->bio_async_enabled = false;
    return bridge;
}

void neuromod_fep_bridge_destroy(neuromod_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) neuromod_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->mutex) nimcp_platform_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int neuromod_fep_bridge_connect_fep(neuromod_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int neuromod_fep_bridge_connect_neuromod(neuromod_fep_bridge_t* bridge, neuromodulator_system_t neuromod) {
    if (!bridge || !neuromod) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->neuromod_system = neuromod;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int neuromod_fep_bridge_disconnect(neuromod_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    bridge->neuromod_system = NULL;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

float neuromod_fep_compute_da_from_pe(neuromod_fep_bridge_t* bridge, float pe) {
    if (!bridge || !bridge->config.enable_da_pe_coupling) return 0.0f;
    return pe * bridge->config.da_pe_sensitivity;
}

float neuromod_fep_compute_ach_from_precision(neuromod_fep_bridge_t* bridge, float precision) {
    if (!bridge || !bridge->config.enable_ach_precision_coupling) return 0.0f;
    return precision * bridge->config.ach_precision_sensitivity;
}

float neuromod_fep_compute_ne_from_uncertainty(neuromod_fep_bridge_t* bridge, float uncertainty) {
    if (!bridge || !bridge->config.enable_ne_uncertainty_coupling) return 0.0f;
    return uncertainty * bridge->config.ne_uncertainty_sensitivity;
}

float neuromod_fep_get_learning_rate_modulation(const neuromod_fep_bridge_t* bridge) {
    if (!bridge) return 1.0f;
    return bridge->neuromod_effects.learning_rate_modulation;
}

int neuromod_fep_bridge_update(neuromod_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->mutex);
    if (bridge->fep_system && bridge->neuromod_system) {
        float pe = fep_get_prediction_error(bridge->fep_system, 0);
        bridge->fep_effects.pe_magnitude = fabsf(pe);
        bridge->fep_effects.da_release = neuromod_fep_compute_da_from_pe(bridge, pe);

        if (bridge->config.enable_da_pe_coupling && bridge->fep_effects.da_release != 0.0f) {
            neuromodulator_release_dopamine(bridge->neuromod_system, bridge->fep_effects.da_release, 0.0f);
            bridge->stats.da_releases++;
        }

        neuromodulator_pool_t pool;
        if (neuromodulator_get_levels(bridge->neuromod_system, &pool)) {
            bridge->neuromod_effects.da_level = pool.dopamine;
            bridge->neuromod_effects.ach_level = pool.acetylcholine;
            bridge->neuromod_effects.learning_rate_modulation = 1.0f + 0.5f * pool.dopamine;
        }

        bridge->stats.total_updates++;
        bridge->stats.avg_da_level = (bridge->stats.avg_da_level * (bridge->stats.total_updates - 1) + bridge->neuromod_effects.da_level) / bridge->stats.total_updates;
    }
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int neuromod_fep_bridge_get_stats(const neuromod_fep_bridge_t* bridge, neuromod_fep_stats_t* stats) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    nimcp_platform_mutex_lock(bridge->mutex);
    memcpy(stats, &bridge->stats, sizeof(neuromod_fep_stats_t));
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int neuromod_fep_bridge_connect_bio_async(neuromod_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    bridge->bio_async_enabled = false;
    return 0;
}

int neuromod_fep_bridge_disconnect_bio_async(neuromod_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    bridge->bio_async_enabled = false;
    return 0;
}

bool neuromod_fep_bridge_is_bio_async_connected(const neuromod_fep_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}
