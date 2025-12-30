/**
 * @file nimcp_curiosity_substrate_bridge.c
 * @brief Curiosity-Neural Substrate Bridge Implementation
 */

#include "cognitive/curiosity/nimcp_curiosity_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct curiosity_substrate_bridge {
    void* curiosity;
    neural_substrate_t* substrate;
    curiosity_substrate_config_t config;
    curiosity_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

curiosity_substrate_config_t curiosity_substrate_default_config(void) {
    curiosity_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

curiosity_substrate_bridge_t* curiosity_substrate_bridge_create(void* curiosity, neural_substrate_t* substrate, const curiosity_substrate_config_t* config) {
    if (!substrate) return NULL;
    curiosity_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(curiosity_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->curiosity = curiosity;
    bridge->substrate = substrate;
    bridge->config = config ? *config : curiosity_substrate_default_config();
    bridge->effects.exploration_drive = 1.0f;
    bridge->effects.novelty_seeking = 1.0f;
    bridge->effects.information_gain = 1.0f;
    bridge->effects.uncertainty_tolerance = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void curiosity_substrate_bridge_destroy(curiosity_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int curiosity_substrate_bridge_update(curiosity_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.exploration_drive = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.information_gain = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.novelty_seeking = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.uncertainty_tolerance = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.exploration_drive + bridge->effects.novelty_seeking +
                                        bridge->effects.information_gain + bridge->effects.uncertainty_tolerance) / 4.0f;
    bridge->update_count++;
    return 0;
}

int curiosity_substrate_bridge_get_effects(const curiosity_substrate_bridge_t* bridge, curiosity_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int curiosity_substrate_bridge_apply_effects(curiosity_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int curiosity_substrate_bridge_register_bio_async(curiosity_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
