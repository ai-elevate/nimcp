/**
 * @file nimcp_stdp_fep_bridge.c
 * @brief Free Energy Principle - STDP Integration Bridge Implementation
 */

#include "plasticity/stdp/nimcp_stdp_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>

/* Logging module name */
#define LOG_MODULE_STDP_FEP "STDP_FEP_BRIDGE"

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

int stdp_fep_bridge_default_config(stdp_fep_config_t* config) {
    if (!config) {
        NIMCP_LOGGING_ERROR("NULL config pointer");
        return -1;
    }

    config->pe_min_threshold = STDP_FEP_PE_MIN_THRESHOLD;
    config->pe_max_threshold = STDP_FEP_PE_MAX_THRESHOLD;
    config->precision_sensitivity = STDP_FEP_PRECISION_SENSITIVITY;
    config->lr_min_factor = STDP_FEP_LR_MIN_FACTOR;
    config->lr_max_factor = STDP_FEP_LR_MAX_FACTOR;

    config->enable_pe_scaling = true;
    config->enable_precision_weighting = true;
    config->enable_belief_modulation = true;
    config->enable_complexity_regularization = true;

    config->pe_sensitivity = 1.0f;
    config->precision_gain = 1.0f;
    config->belief_sensitivity = 0.5f;

    return 0;
}

stdp_fep_bridge_t* stdp_fep_bridge_create(const stdp_fep_config_t* config) {
    stdp_fep_bridge_t* bridge = (stdp_fep_bridge_t*)nimcp_malloc(sizeof(stdp_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("Failed to allocate bridge");
        return NULL;
    }

    memset(bridge, 0, sizeof(stdp_fep_bridge_t));

    if (config) {
        bridge->config = *config;
    } else {
        stdp_fep_bridge_default_config(&bridge->config);
    }

    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create mutex");
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.total_lr_scaling = 1.0f;
    bridge->effects.effective_learning_rate = 0.01f;

    NIMCP_LOGGING_INFO("STDP-FEP bridge created");
    return bridge;
}

void stdp_fep_bridge_destroy(stdp_fep_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->bio_async_enabled) {
        stdp_fep_bridge_disconnect_bio_async(bridge);
    }

    if (bridge->mutex) {
        nimcp_platform_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("STDP-FEP bridge destroyed");
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int stdp_fep_bridge_connect_fep(stdp_fep_bridge_t* bridge, fep_system_t* fep) {
    if (!bridge || !fep) {
        NIMCP_LOGGING_ERROR("NULL bridge or FEP system");
        return -1;
    }

    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->fep_system = fep;
    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_INFO("Connected to FEP system");
    return 0;
}

int stdp_fep_bridge_connect_stdp(stdp_fep_bridge_t* bridge,
                                   stdp_synapse_t* stdp_synapses,
                                   uint32_t num_synapses) {
    if (!bridge || !stdp_synapses || num_synapses == 0) {
        NIMCP_LOGGING_ERROR("Invalid parameters");
        return -1;
    }

    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->stdp_system = stdp_synapses;
    bridge->num_synapses = num_synapses;
    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_INFO("Connected to STDP system");
    return 0;
}

int stdp_fep_bridge_disconnect(stdp_fep_bridge_t* bridge) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->fep_system = NULL;
    bridge->stdp_system = NULL;
    bridge->num_synapses = 0;
    nimcp_platform_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_INFO("Disconnected systems");
    return 0;
}

/* ============================================================================
 * FEP → STDP Direction
 * ============================================================================ */

float stdp_fep_apply_pe_scaling(stdp_fep_bridge_t* bridge, float pe) {
    if (!bridge) return 1.0f;

    if (!bridge->config.enable_pe_scaling) {
        return 1.0f;
    }

    float pe_abs = fabsf(pe);
    if (pe_abs < bridge->config.pe_min_threshold) {
        return bridge->config.lr_min_factor;
    }

    if (pe_abs > bridge->config.pe_max_threshold) {
        pe_abs = bridge->config.pe_max_threshold;
    }

    float normalized_pe = (pe_abs - bridge->config.pe_min_threshold) /
                          (bridge->config.pe_max_threshold - bridge->config.pe_min_threshold);

    float scaling = 1.0f + normalized_pe * bridge->config.pe_sensitivity;

    if (scaling < bridge->config.lr_min_factor) {
        scaling = bridge->config.lr_min_factor;
    }
    if (scaling > bridge->config.lr_max_factor) {
        scaling = bridge->config.lr_max_factor;
    }

    return scaling;
}

float stdp_fep_apply_precision_weighting(stdp_fep_bridge_t* bridge, float precision) {
    if (!bridge) return 1.0f;

    if (!bridge->config.enable_precision_weighting) {
        return 1.0f;
    }

    float scaling = powf(precision, bridge->config.precision_sensitivity);
    scaling *= bridge->config.precision_gain;

    if (scaling < bridge->config.lr_min_factor) {
        scaling = bridge->config.lr_min_factor;
    }
    if (scaling > bridge->config.lr_max_factor) {
        scaling = bridge->config.lr_max_factor;
    }

    return scaling;
}

float stdp_fep_apply_belief_modulation(stdp_fep_bridge_t* bridge, float belief_delta) {
    if (!bridge) return 1.0f;

    if (!bridge->config.enable_belief_modulation) {
        return 1.0f;
    }

    float scaling = 1.0f + fabsf(belief_delta) * bridge->config.belief_sensitivity;

    if (scaling < bridge->config.lr_min_factor) {
        scaling = bridge->config.lr_min_factor;
    }
    if (scaling > bridge->config.lr_max_factor) {
        scaling = bridge->config.lr_max_factor;
    }

    return scaling;
}

float stdp_fep_get_effective_lr(const stdp_fep_bridge_t* bridge, float base_lr) {
    if (!bridge) return base_lr;

    return base_lr * bridge->effects.total_lr_scaling;
}

/* ============================================================================
 * STDP → FEP Direction
 * ============================================================================ */

int stdp_fep_report_weight_changes(stdp_fep_bridge_t* bridge, float weight_delta) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->mutex);
    bridge->stats.total_weight_delta += weight_delta;
    bridge->stats.weight_updates++;
    nimcp_platform_mutex_unlock(bridge->mutex);

    return 0;
}

float stdp_fep_compute_complexity_regularization(const stdp_fep_bridge_t* bridge) {
    if (!bridge) return 1.0f;

    if (!bridge->config.enable_complexity_regularization) {
        return 1.0f;
    }

    return 0.99f;
}

/* ============================================================================
 * Update Cycle
 * ============================================================================ */

int stdp_fep_bridge_update(stdp_fep_bridge_t* bridge, uint64_t delta_ms) {
    if (!bridge) return -1;

    nimcp_platform_mutex_lock(bridge->mutex);

    if (bridge->fep_system) {
        float pe_lr_scaling = 1.0f;
        float precision_lr_scaling = 1.0f;
        float belief_lr_scaling = 1.0f;

        if (bridge->config.enable_pe_scaling) {
            pe_lr_scaling = stdp_fep_apply_pe_scaling(bridge, bridge->effects.pe_magnitude);
            bridge->effects.pe_lr_scaling = pe_lr_scaling;
        }

        if (bridge->config.enable_precision_weighting) {
            precision_lr_scaling = stdp_fep_apply_precision_weighting(bridge, bridge->effects.precision_value);
            bridge->effects.precision_lr_scaling = precision_lr_scaling;
        }

        if (bridge->config.enable_belief_modulation) {
            belief_lr_scaling = stdp_fep_apply_belief_modulation(bridge, bridge->effects.belief_delta);
            bridge->effects.belief_lr_scaling = belief_lr_scaling;
        }

        bridge->effects.total_lr_scaling = pe_lr_scaling * precision_lr_scaling * belief_lr_scaling;

        bridge->stats.avg_pe_scaling =
            (bridge->stats.avg_pe_scaling * bridge->stats.total_updates + pe_lr_scaling) /
            (bridge->stats.total_updates + 1);
        bridge->stats.avg_precision_scaling =
            (bridge->stats.avg_precision_scaling * bridge->stats.total_updates + precision_lr_scaling) /
            (bridge->stats.total_updates + 1);
        bridge->stats.avg_lr_modulation =
            (bridge->stats.avg_lr_modulation * bridge->stats.total_updates + bridge->effects.total_lr_scaling) /
            (bridge->stats.total_updates + 1);
    }

    bridge->stats.total_updates++;
    bridge->state.last_update_time = delta_ms;

    nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

/* ============================================================================
 * State/Stats API
 * ============================================================================ */

int stdp_fep_bridge_get_state(const stdp_fep_bridge_t* bridge, stdp_fep_state_t* state) {
    if (!bridge || !state) return -1;

    *state = bridge->state;
    return 0;
}

int stdp_fep_bridge_get_stats(const stdp_fep_bridge_t* bridge, stdp_fep_stats_t* stats) {
    if (!bridge || !stats) return -1;

    *stats = bridge->stats;
    return 0;
}

/* ============================================================================
 * Bio-Async Integration
 * ============================================================================ */

int stdp_fep_bridge_connect_bio_async(stdp_fep_bridge_t* bridge) {
    if (!bridge) return -1;
    if (bridge->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_STDP_BRIDGE,
        .module_name = "stdp_fep_bridge",
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

int stdp_fep_bridge_disconnect_bio_async(stdp_fep_bridge_t* bridge) {
    if (!bridge || !bridge->bio_async_enabled) return -1;

    bio_router_unregister_module(bridge->bio_ctx);
    bridge->bio_ctx = NULL;
    bridge->bio_async_enabled = false;

    NIMCP_LOGGING_INFO("Disconnected from bio-async router");
    return 0;
}

bool stdp_fep_bridge_is_bio_async_connected(const stdp_fep_bridge_t* bridge) {
    return bridge ? bridge->bio_async_enabled : false;
}
