/**
 * @file nimcp_occipital_substrate_bridge.c
 * @brief Occipital Cortex-Neural Substrate Bridge Implementation
 */

#include "core/occipital/nimcp_occipital_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct occipital_substrate_bridge {
    void* occipital;
    neural_substrate_t* substrate;
    occipital_substrate_config_t config;
    occipital_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

occipital_substrate_config_t occipital_substrate_default_config(void) {
    occipital_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

occipital_substrate_bridge_t* occipital_substrate_bridge_create(void* occipital, neural_substrate_t* substrate, const occipital_substrate_config_t* config) {
    if (!substrate) return NULL;
    occipital_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(occipital_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->occipital = occipital;
    bridge->substrate = substrate;
    bridge->config = config ? *config : occipital_substrate_default_config();
    bridge->effects.primary_visual = 1.0f;
    bridge->effects.color_processing = 1.0f;
    bridge->effects.form_processing = 1.0f;
    bridge->effects.motion_perception = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void occipital_substrate_bridge_destroy(occipital_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int occipital_substrate_bridge_update(occipital_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP affects primary visual and color processing */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.primary_visual = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.color_processing = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue affects form and motion processing */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.form_processing = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.motion_perception = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.primary_visual + bridge->effects.color_processing +
                                        bridge->effects.form_processing + bridge->effects.motion_perception) / 4.0f;
    bridge->update_count++;
    return 0;
}

int occipital_substrate_bridge_get_effects(const occipital_substrate_bridge_t* bridge, occipital_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int occipital_substrate_bridge_apply_effects(occipital_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int occipital_substrate_bridge_register_bio_async(occipital_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
