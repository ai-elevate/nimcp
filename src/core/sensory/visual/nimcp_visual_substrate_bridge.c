/**
 * @file nimcp_visual_substrate_bridge.c
 * @brief Visual-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/sensory/visual/nimcp_visual_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct visual_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* visual;
    neural_substrate_t* substrate;
    visual_substrate_config_t config;
    visual_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

visual_substrate_config_t visual_substrate_default_config(void) {
    visual_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

visual_substrate_bridge_t* visual_substrate_bridge_create(void* visual, neural_substrate_t* substrate, const visual_substrate_config_t* config) {
    if (!substrate) return NULL;
    visual_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(visual_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->visual = visual;
    bridge->substrate = substrate;
    bridge->config = config ? *config : visual_substrate_default_config();
    bridge->effects.visual_acuity = 1.0f;
    bridge->effects.contrast_sensitivity = 1.0f;
    bridge->effects.motion_processing = 1.0f;
    bridge->effects.feature_binding = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void visual_substrate_bridge_destroy(visual_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int visual_substrate_bridge_update(visual_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP affects visual acuity and contrast sensitivity */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.visual_acuity = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.contrast_sensitivity = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue affects motion processing and feature binding */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.motion_processing = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.feature_binding = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.visual_acuity + bridge->effects.contrast_sensitivity +
                                        bridge->effects.motion_processing + bridge->effects.feature_binding) / 4.0f;
    bridge->update_count++;
    return 0;
}

int visual_substrate_bridge_get_effects(const visual_substrate_bridge_t* bridge, visual_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int visual_substrate_bridge_apply_effects(visual_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int visual_substrate_bridge_register_bio_async(visual_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
