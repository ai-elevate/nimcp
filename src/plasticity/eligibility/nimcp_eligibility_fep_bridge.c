/**
 * @file nimcp_eligibility_fep_bridge.c
 * @brief FEP-Eligibility Integration Bridge Implementation
 */

#include "plasticity/eligibility/nimcp_eligibility_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>

#define LOG_MODULE_ELIGIBILITY_FEP "ELIGIBILITY_FEP_BRIDGE"

int eligibility_fep_bridge_default_config(eligibility_fep_config_t* config) {
    if (!config) return -1;
    config->pe_trace_gain = ELIGIBILITY_FEP_PE_TRACE_SCALING;
    config->precision_decay_sensitivity = 1.0f;
    config->fe_consolidation_threshold = 1.0f;
    config->enable_pe_eligibility = true;
    config->enable_precision_decay_modulation = true;
    config->enable_fe_gated_consolidation = true;
    return 0;
}

eligibility_fep_bridge_t* eligibility_fep_bridge_create(const eligibility_fep_config_t* config) {
    eligibility_fep_bridge_t* bridge = (eligibility_fep_bridge_t*)nimcp_malloc(sizeof(eligibility_fep_bridge_t));
    if (!bridge) return NULL;
    memset(bridge, 0, sizeof(eligibility_fep_bridge_t));
    if (config) bridge->config = *config;
    else eligibility_fep_bridge_default_config(&bridge->config);
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) { nimcp_free(bridge); return NULL; }
    bridge->effects.total_decay_modulation = 1.0f;
    NIMCP_LOGGING_INFO("Eligibility-FEP bridge created");
    return bridge;
}

void eligibility_fep_bridge_destroy(eligibility_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) eligibility_fep_bridge_disconnect_bio_async(bridge);
    if (bridge->mutex) nimcp_platform_mutex_destroy(bridge->mutex);
    nimcp_free(bridge);
}

int eligibility_fep_bridge_connect_fep(eligibility_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) return -1;
    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int eligibility_fep_bridge_connect_eligibility(eligibility_fep_bridge_t* bridge, eligibility_trace_t* traces, uint32_t num) {
    if (!bridge || !traces || num == 0) return -1;
    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->eligibility_system = traces;
    bridge->num_traces = num;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int eligibility_fep_bridge_disconnect(eligibility_fep_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    bridge->eligibility_system = NULL;
    bridge->num_traces = 0;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

float eligibility_fep_apply_pe_eligibility(eligibility_fep_bridge_t* bridge, float pe) {
    if (!bridge || !bridge->config.enable_pe_eligibility) return 1.0f;
    return 1.0f + fabsf(pe) * bridge->config.pe_trace_gain;
}

float eligibility_fep_apply_precision_decay_modulation(eligibility_fep_bridge_t* bridge, float precision) {
    if (!bridge || !bridge->config.enable_precision_decay_modulation) return 1.0f;
    float decay = precision * bridge->config.precision_decay_sensitivity;
    return fminf(fmaxf(decay, ELIGIBILITY_FEP_PRECISION_DECAY_MIN), ELIGIBILITY_FEP_PRECISION_DECAY_MAX);
}

bool eligibility_fep_should_consolidate(const eligibility_fep_bridge_t* bridge) {
    if (!bridge || !bridge->config.enable_fe_gated_consolidation) return true;
    return bridge->effects.free_energy_value > bridge->config.fe_consolidation_threshold;
}

float eligibility_fep_get_effective_decay(const eligibility_fep_bridge_t* bridge, float base_decay) {
    if (!bridge) return base_decay;
    return base_decay * bridge->effects.total_decay_modulation;
}

int eligibility_fep_report_consolidation(eligibility_fep_bridge_t* bridge) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->stats.trace_consolidations++;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int eligibility_fep_bridge_update(eligibility_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return -1;
    nimcp_platform_mutex_lock(bridge->mutex);
    if (bridge->fep_system) {
        float pe_scaling = eligibility_fep_apply_pe_eligibility(bridge, bridge->effects.pe_magnitude);
        float precision_decay = eligibility_fep_apply_precision_decay_modulation(bridge, bridge->effects.precision_value);
        bridge->effects.pe_trace_scaling = pe_scaling;
        bridge->effects.precision_decay_modulation = precision_decay;
        bridge->effects.total_decay_modulation = precision_decay;
        bridge->state.consolidation_active = eligibility_fep_should_consolidate(bridge);
    }
    bridge->stats.total_updates++;
    bridge->state.last_update_time = delta_ms;
    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int eligibility_fep_bridge_get_state(const eligibility_fep_bridge_t* bridge, eligibility_fep_state_t* state) {
    if (!bridge || !state) return -1;
    *state = bridge->state;
    return 0;
}

int eligibility_fep_bridge_get_stats(const eligibility_fep_bridge_t* bridge, eligibility_fep_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

int eligibility_fep_bridge_connect_bio_async(eligibility_fep_bridge_t* bridge) {
    if (!bridge || bridge->bio_async_enabled) return 0;
    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_ELIGIBILITY_BRIDGE,
        .module_name = "eligibility_fep_bridge",
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

int eligibility_fep_bridge_disconnect_bio_async(eligibility_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return -1;
    bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_async_enabled = false;
    return 0;
}

bool eligibility_fep_bridge_is_bio_async_connected(const eligibility_fep_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}
