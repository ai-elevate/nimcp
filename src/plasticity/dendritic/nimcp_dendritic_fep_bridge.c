/**
 * @file nimcp_dendritic_fep_bridge.c
 * @brief FEP-Dendritic Integration Bridge Implementation
 */

#include "plasticity/dendritic/nimcp_dendritic_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>

#define LOG_MODULE_DENDRITIC_FEP "DENDRITIC_FEP_BRIDGE"

int dendritic_fep_bridge_default_config(dendritic_fep_config_t* config) {
    if (!config) return -1;
    config->pe_nmda_gain = DENDRITIC_FEP_PE_NMDA_SCALING;
    config->precision_excitability_gain = 1.0f;
    config->calcium_belief_sensitivity = DENDRITIC_FEP_CALCIUM_PE_FACTOR;
    config->enable_pe_nmda_modulation = true;
    config->enable_precision_gain_control = true;
    config->enable_calcium_belief_updates = true;
    config->enable_hierarchical_predictions = true;
    return 0;
}

dendritic_fep_bridge_t* dendritic_fep_bridge_create(const dendritic_fep_config_t* config) {
    dendritic_fep_bridge_t* bridge = (dendritic_fep_bridge_t*)nimcp_malloc(sizeof(dendritic_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(dendritic_fep_bridge_t));
    if (config) bridge->config = *config;
    else dendritic_fep_bridge_default_config(&bridge->config);
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }
    bridge->effects.total_nmda_modulation = 1.0f;
    bridge->effects.total_gain_modulation = 1.0f;
    NIMCP_LOGGING_INFO("Dendritic-FEP bridge created");
    return bridge;
}

void dendritic_fep_bridge_destroy(dendritic_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) dendritic_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->mutex) nimcp_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int dendritic_fep_bridge_connect_fep(dendritic_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return -1;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int dendritic_fep_bridge_connect_dendritic(dendritic_fep_bridge_t* bridge, dendritic_tree_t dendritic) {
    if (!bridge || !dendritic) return -1;
    nimcp_mutex_lock(bridge->mutex);
    bridge->dendritic_system = dendritic;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int dendritic_fep_bridge_disconnect(dendritic_fep_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    bridge->dendritic_system = NULL;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

float dendritic_fep_apply_pe_nmda_modulation(dendritic_fep_bridge_t* bridge, float pe) {
    if (!bridge || !bridge->config.enable_pe_nmda_modulation) return 1.0f;
    return 1.0f + fabsf(pe) * bridge->config.pe_nmda_gain;
}

float dendritic_fep_apply_precision_gain_control(dendritic_fep_bridge_t* bridge, float precision) {
    if (!bridge || !bridge->config.enable_precision_gain_control) return 1.0f;
    float gain = precision * bridge->config.precision_excitability_gain;
    return fminf(fmaxf(gain, DENDRITIC_FEP_PRECISION_GAIN_MIN), DENDRITIC_FEP_PRECISION_GAIN_MAX);
}

float dendritic_fep_compute_calcium_belief_update(const dendritic_fep_bridge_t* bridge) {
    if (!bridge || !bridge->config.enable_calcium_belief_updates) return 0.0f;
    return bridge->effects.calcium_concentration * bridge->config.calcium_belief_sensitivity;
}

float dendritic_fep_get_effective_nmda_conductance(const dendritic_fep_bridge_t* bridge, float base_conductance) {
    if (!bridge) return base_conductance;
    return base_conductance * bridge->effects.total_nmda_modulation;
}

int dendritic_fep_report_dendritic_spike(dendritic_fep_bridge_t* bridge, float spike_amplitude) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->mutex);
    bridge->state.dendritic_spikes++;
    bridge->stats.dendritic_spike_events++;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int dendritic_fep_bridge_update(dendritic_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return -1;
    nimcp_mutex_lock(bridge->mutex);
    if (bridge->fep_system) {
        float pe_nmda = dendritic_fep_apply_pe_nmda_modulation(bridge, bridge->effects.pe_magnitude);
        float precision_gain = dendritic_fep_apply_precision_gain_control(bridge, bridge->effects.precision_value);
        float calcium_update = dendritic_fep_compute_calcium_belief_update(bridge);

        bridge->effects.pe_nmda_scaling = pe_nmda;
        bridge->effects.precision_gain_modulation = precision_gain;
        bridge->effects.calcium_belief_update = calcium_update;
        bridge->effects.total_nmda_modulation = pe_nmda;
        bridge->effects.total_gain_modulation = precision_gain;

        bridge->stats.avg_nmda_modulation =
            (bridge->stats.avg_nmda_modulation * bridge->stats.total_updates + pe_nmda) /
            (bridge->stats.total_updates + 1);
    }
    bridge->stats.total_updates++;
    bridge->state.last_update_time = delta_ms;
    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

int dendritic_fep_bridge_get_state(const dendritic_fep_bridge_t* bridge, dendritic_fep_state_t* state) {
    if (!bridge || !state) return -1;
    *state = bridge->state;
    return 0;
}

int dendritic_fep_bridge_get_stats(const dendritic_fep_bridge_t* bridge, dendritic_fep_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

int dendritic_fep_bridge_connect_bio_async(dendritic_fep_bridge_t* bridge) {
    if (!bridge || bridge->bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_DENDRITIC_BRIDGE,
        .module_name = "dendritic_fep_bridge",
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

int dendritic_fep_bridge_disconnect_bio_async(dendritic_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return -1;
    bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_async_enabled = false;
    return 0;
}

bool dendritic_fep_bridge_is_bio_async_connected(const dendritic_fep_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}
