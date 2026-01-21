/**
 * @file nimcp_motor_substrate_bridge.c
 * @brief Motor-Neural Substrate Bridge Implementation
 */

#include "core/motor/nimcp_motor_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct motor_substrate_bridge {
    void* motor;
    neural_substrate_t* substrate;
    motor_substrate_config_t config;
    motor_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

motor_substrate_config_t motor_substrate_default_config(void) {
    motor_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

motor_substrate_bridge_t* motor_substrate_bridge_create(void* motor, neural_substrate_t* substrate, const motor_substrate_config_t* config) {
    if (!substrate) return NULL;
    motor_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(motor_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->motor = motor;
    bridge->substrate = substrate;
    bridge->config = config ? *config : motor_substrate_default_config();
    bridge->effects.motor_precision = 1.0f;
    bridge->effects.motor_speed = 1.0f;
    bridge->effects.motor_endurance = 1.0f;
    bridge->effects.coordination = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void motor_substrate_bridge_destroy(motor_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int motor_substrate_bridge_update(motor_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP directly affects motor execution capacity */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.motor_precision = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.motor_endurance = clamp_f(atp * 1.2f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue affects speed and coordination */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.motor_speed = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.coordination = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.motor_precision + bridge->effects.motor_speed +
                                        bridge->effects.motor_endurance + bridge->effects.coordination) / 4.0f;
    bridge->update_count++;
    return 0;
}

int motor_substrate_bridge_get_effects(const motor_substrate_bridge_t* bridge, motor_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int motor_substrate_bridge_apply_effects(motor_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int motor_substrate_bridge_register_bio_async(motor_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
