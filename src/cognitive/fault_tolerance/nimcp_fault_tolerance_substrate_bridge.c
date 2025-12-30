/**
 * @file nimcp_fault_tolerance_substrate_bridge.c
 * @brief Fault Tolerance-Neural Substrate Bridge Implementation
 *
 * WHAT: Links fault tolerance to metabolic state
 * WHY: Error handling requires sustained prefrontal resources
 * HOW: Monitors ATP/fatigue; modulates detection, recovery, redundancy
 */

#include "cognitive/fault_tolerance/nimcp_fault_tolerance_substrate_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>
#include <math.h>

struct fault_tolerance_substrate_bridge {
    bridge_base_t base;                 /* MUST be first: base bridge infrastructure */
    void* fault_tolerance;
    neural_substrate_t* substrate;
    fault_tolerance_substrate_config_t config;
    fault_tolerance_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

fault_tolerance_substrate_config_t fault_tolerance_substrate_default_config(void) {
    fault_tolerance_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.3f  /* Higher min for critical fault tolerance */
    };
    return cfg;
}

fault_tolerance_substrate_bridge_t* fault_tolerance_substrate_bridge_create(void* fault_tolerance, neural_substrate_t* substrate, const fault_tolerance_substrate_config_t* config) {
    if (!substrate) return NULL;

    fault_tolerance_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(fault_tolerance_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->fault_tolerance = fault_tolerance;
    bridge->substrate = substrate;
    bridge->config = config ? *config : fault_tolerance_substrate_default_config();

    /* Initialize mutex */
    bridge->base.mutex = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex for fault tolerance substrate bridge");
        nimcp_free(bridge);
        return NULL;
    }

    if (nimcp_platform_mutex_init(bridge->base.mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex for fault tolerance substrate bridge");
        nimcp_free(bridge->base.mutex);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.detection_sensitivity = 1.0f;
    bridge->effects.recovery_speed = 1.0f;
    bridge->effects.redundancy_capacity = 1.0f;
    bridge->effects.monitoring_depth = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void fault_tolerance_substrate_bridge_destroy(fault_tolerance_substrate_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge->base.mutex);
    }

    nimcp_free(bridge);
}

int fault_tolerance_substrate_bridge_update(fault_tolerance_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) {
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Detection sensitivity requires stable ATP for continuous monitoring */
        bridge->effects.detection_sensitivity = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Redundancy maintenance is ATP-intensive */
        bridge->effects.redundancy_capacity = clamp_f(atp * 0.9f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Recovery speed decreases with fatigue */
        bridge->effects.recovery_speed = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Monitoring depth narrows under metabolic stress */
        bridge->effects.monitoring_depth = clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.detection_sensitivity +
                                        bridge->effects.recovery_speed +
                                        bridge->effects.redundancy_capacity +
                                        bridge->effects.monitoring_depth) / 4.0f;

    bridge->update_count++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fault_tolerance_substrate_bridge_get_effects(const fault_tolerance_substrate_bridge_t* bridge, fault_tolerance_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int fault_tolerance_substrate_bridge_apply_effects(fault_tolerance_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int fault_tolerance_substrate_bridge_register_bio_async(fault_tolerance_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
