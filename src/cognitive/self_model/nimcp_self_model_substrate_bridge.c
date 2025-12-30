/**
 * @file nimcp_self_model_substrate_bridge.c
 * @brief Self-Model-Neural Substrate Bridge Implementation
 */

#include "cognitive/self_model/nimcp_self_model_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct self_model_substrate_bridge {
    void* self_model;
    neural_substrate_t* substrate;
    self_model_substrate_config_t config;
    self_model_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

self_model_substrate_config_t self_model_substrate_default_config(void) {
    self_model_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

self_model_substrate_bridge_t* self_model_substrate_bridge_create(void* self_model, neural_substrate_t* substrate, const self_model_substrate_config_t* config) {
    if (!substrate) return NULL;
    self_model_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(self_model_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->self_model = self_model;
    bridge->substrate = substrate;
    bridge->config = config ? *config : self_model_substrate_default_config();
    bridge->effects.self_representation = 1.0f;
    bridge->effects.body_schema = 1.0f;
    bridge->effects.agency_sense = 1.0f;
    bridge->effects.boundary_clarity = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void self_model_substrate_bridge_destroy(self_model_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int self_model_substrate_bridge_update(self_model_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.self_representation = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.agency_sense = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.body_schema = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.boundary_clarity = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.self_representation + bridge->effects.body_schema +
                                        bridge->effects.agency_sense + bridge->effects.boundary_clarity) / 4.0f;
    bridge->update_count++;
    return 0;
}

int self_model_substrate_bridge_get_effects(const self_model_substrate_bridge_t* bridge, self_model_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int self_model_substrate_bridge_apply_effects(self_model_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int self_model_substrate_bridge_register_bio_async(self_model_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
