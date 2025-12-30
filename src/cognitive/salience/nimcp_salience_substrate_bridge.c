/**
 * @file nimcp_salience_substrate_bridge.c
 * @brief Salience-Neural Substrate Bridge Implementation
 */

#include "cognitive/salience/nimcp_salience_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct salience_substrate_bridge {
    void* salience;
    neural_substrate_t* substrate;
    salience_substrate_config_t config;
    salience_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

salience_substrate_config_t salience_substrate_default_config(void) {
    salience_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

salience_substrate_bridge_t* salience_substrate_bridge_create(void* salience, neural_substrate_t* substrate, const salience_substrate_config_t* config) {
    if (!substrate) return NULL;
    salience_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(salience_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->salience = salience;
    bridge->substrate = substrate;
    bridge->config = config ? *config : salience_substrate_default_config();
    bridge->effects.detection_sensitivity = 1.0f;
    bridge->effects.priority_accuracy = 1.0f;
    bridge->effects.filtering_quality = 1.0f;
    bridge->effects.switching_speed = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void salience_substrate_bridge_destroy(salience_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int salience_substrate_bridge_update(salience_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.detection_sensitivity = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.priority_accuracy = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.filtering_quality = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.switching_speed = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.detection_sensitivity + bridge->effects.priority_accuracy +
                                        bridge->effects.filtering_quality + bridge->effects.switching_speed) / 4.0f;
    bridge->update_count++;
    return 0;
}

int salience_substrate_bridge_get_effects(const salience_substrate_bridge_t* bridge, salience_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int salience_substrate_bridge_apply_effects(salience_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int salience_substrate_bridge_register_bio_async(salience_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
