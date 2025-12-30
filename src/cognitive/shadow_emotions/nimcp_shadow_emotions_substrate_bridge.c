/**
 * @file nimcp_shadow_emotions_substrate_bridge.c
 * @brief Shadow Emotions-Neural Substrate Bridge Implementation
 */

#include "cognitive/shadow_emotions/nimcp_shadow_emotions_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct shadow_emotions_substrate_bridge {
    void* shadow_emotions;
    neural_substrate_t* substrate;
    shadow_emotions_substrate_config_t config;
    shadow_emotions_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

shadow_emotions_substrate_config_t shadow_emotions_substrate_default_config(void) {
    shadow_emotions_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

shadow_emotions_substrate_bridge_t* shadow_emotions_substrate_bridge_create(void* shadow_emotions, neural_substrate_t* substrate, const shadow_emotions_substrate_config_t* config) {
    if (!substrate) return NULL;
    shadow_emotions_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(shadow_emotions_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->shadow_emotions = shadow_emotions;
    bridge->substrate = substrate;
    bridge->config = config ? *config : shadow_emotions_substrate_default_config();
    bridge->effects.suppression_strength = 1.0f;
    bridge->effects.emergence_threshold = 0.5f;
    bridge->effects.regulation_capacity = 1.0f;
    bridge->effects.integration_ability = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void shadow_emotions_substrate_bridge_destroy(shadow_emotions_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int shadow_emotions_substrate_bridge_update(shadow_emotions_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, fatigue = 1.0f - metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables suppression and regulation - low ATP allows emergence */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.suppression_strength = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.regulation_capacity = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue lowers emergence threshold and reduces integration */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.emergence_threshold = clamp_f(0.5f - fatigue * 0.3f * bridge->config.fatigue_sensitivity, 0.2f, 0.8f);
        bridge->effects.integration_ability = clamp_f((1.0f - fatigue) * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.suppression_strength + bridge->effects.regulation_capacity +
                                        bridge->effects.integration_ability) / 3.0f;
    bridge->update_count++;
    return 0;
}

int shadow_emotions_substrate_bridge_get_effects(const shadow_emotions_substrate_bridge_t* bridge, shadow_emotions_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int shadow_emotions_substrate_bridge_apply_effects(shadow_emotions_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int shadow_emotions_substrate_bridge_register_bio_async(shadow_emotions_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
