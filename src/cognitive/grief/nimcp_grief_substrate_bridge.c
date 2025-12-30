/**
 * @file nimcp_grief_substrate_bridge.c
 * @brief Grief-Neural Substrate Bridge Implementation
 */

#include "cognitive/grief/nimcp_grief_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct grief_substrate_bridge {
    void* grief;
    neural_substrate_t* substrate;
    grief_substrate_config_t config;
    grief_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

grief_substrate_config_t grief_substrate_default_config(void) {
    grief_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

grief_substrate_bridge_t* grief_substrate_bridge_create(void* grief, neural_substrate_t* substrate, const grief_substrate_config_t* config) {
    if (!substrate) return NULL;
    grief_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(grief_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->grief = grief;
    bridge->substrate = substrate;
    bridge->config = config ? *config : grief_substrate_default_config();
    bridge->effects.processing_capacity = 1.0f;
    bridge->effects.emotion_regulation = 1.0f;
    bridge->effects.adaptation_rate = 1.0f;
    bridge->effects.resilience_level = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void grief_substrate_bridge_destroy(grief_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int grief_substrate_bridge_update(grief_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.processing_capacity = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.resilience_level = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.emotion_regulation = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.adaptation_rate = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.processing_capacity + bridge->effects.emotion_regulation +
                                        bridge->effects.adaptation_rate + bridge->effects.resilience_level) / 4.0f;
    bridge->update_count++;
    return 0;
}

int grief_substrate_bridge_get_effects(const grief_substrate_bridge_t* bridge, grief_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int grief_substrate_bridge_apply_effects(grief_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int grief_substrate_bridge_register_bio_async(grief_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
