/**
 * @file nimcp_joy_substrate_bridge.c
 * @brief Joy-Neural Substrate Bridge Implementation
 */

#include "cognitive/joy/nimcp_joy_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct joy_substrate_bridge {
    void* joy;
    neural_substrate_t* substrate;
    joy_substrate_config_t config;
    joy_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

joy_substrate_config_t joy_substrate_default_config(void) {
    joy_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

joy_substrate_bridge_t* joy_substrate_bridge_create(void* joy, neural_substrate_t* substrate, const joy_substrate_config_t* config) {
    if (!substrate) return NULL;
    joy_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(joy_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->joy = joy;
    bridge->substrate = substrate;
    bridge->config = config ? *config : joy_substrate_default_config();
    bridge->effects.hedonic_capacity = 1.0f;
    bridge->effects.joy_intensity = 1.0f;
    bridge->effects.savoring_ability = 1.0f;
    bridge->effects.positive_anticipation = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void joy_substrate_bridge_destroy(joy_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int joy_substrate_bridge_update(joy_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables hedonic capacity and joy intensity */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.hedonic_capacity = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.joy_intensity = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables savoring and anticipation */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.savoring_ability = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.positive_anticipation = clamp_f(metabolic_cap * 0.95f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.hedonic_capacity + bridge->effects.joy_intensity +
                                        bridge->effects.savoring_ability + bridge->effects.positive_anticipation) / 4.0f;
    bridge->update_count++;
    return 0;
}

int joy_substrate_bridge_get_effects(const joy_substrate_bridge_t* bridge, joy_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int joy_substrate_bridge_apply_effects(joy_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int joy_substrate_bridge_register_bio_async(joy_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
