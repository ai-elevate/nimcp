/**
 * @file nimcp_empathetic_response_substrate_bridge.c
 * @brief Empathetic Response-Neural Substrate Bridge Implementation
 *
 * WHAT: Links empathetic response to metabolic state
 * WHY: Empathy requires sustained prefrontal and limbic resources
 * HOW: Monitors ATP/fatigue; modulates accuracy, depth, perspective-taking
 */

#include "cognitive/empathetic_response/nimcp_empathetic_response_substrate_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>
#include <math.h>

struct empathetic_response_substrate_bridge {
    bridge_base_t base;                 /* MUST be first: base bridge infrastructure */
    void* empathetic_response;
    neural_substrate_t* substrate;
    empathetic_response_substrate_config_t config;
    empathetic_response_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

empathetic_response_substrate_config_t empathetic_response_substrate_default_config(void) {
    empathetic_response_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

empathetic_response_substrate_bridge_t* empathetic_response_substrate_bridge_create(void* empathetic_response, neural_substrate_t* substrate, const empathetic_response_substrate_config_t* config) {
    if (!substrate) return NULL;

    empathetic_response_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(empathetic_response_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->empathetic_response = empathetic_response;
    bridge->substrate = substrate;
    bridge->config = config ? *config : empathetic_response_substrate_default_config();

    /* Initialize mutex */
    bridge->base.mutex = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex for empathetic response substrate bridge");
        nimcp_free(bridge);
        return NULL;
    }

    if (nimcp_platform_mutex_init(bridge->base.mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex for empathetic response substrate bridge");
        nimcp_free(bridge->base.mutex);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.empathic_accuracy = 1.0f;
    bridge->effects.response_depth = 1.0f;
    bridge->effects.perspective_taking = 1.0f;
    bridge->effects.compassion_endurance = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void empathetic_response_substrate_bridge_destroy(empathetic_response_substrate_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge->base.mutex);
    }

    nimcp_free(bridge);
}

int empathetic_response_substrate_bridge_update(empathetic_response_substrate_bridge_t* bridge) {
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
        /* Empathic accuracy requires stable prefrontal resources */
        bridge->effects.empathic_accuracy = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Perspective-taking is cognitively demanding */
        bridge->effects.perspective_taking = clamp_f(atp * 0.95f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Response depth decreases with fatigue */
        bridge->effects.response_depth = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Compassion endurance is vulnerable to metabolic stress */
        bridge->effects.compassion_endurance = clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.empathic_accuracy +
                                        bridge->effects.response_depth +
                                        bridge->effects.perspective_taking +
                                        bridge->effects.compassion_endurance) / 4.0f;

    bridge->update_count++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int empathetic_response_substrate_bridge_get_effects(const empathetic_response_substrate_bridge_t* bridge, empathetic_response_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int empathetic_response_substrate_bridge_apply_effects(empathetic_response_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int empathetic_response_substrate_bridge_register_bio_async(empathetic_response_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
