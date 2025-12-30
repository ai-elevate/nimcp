/**
 * @file nimcp_analysis_substrate_bridge.c
 * @brief Analysis-Neural Substrate Bridge Implementation
 */

#include "cognitive/analysis/nimcp_analysis_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct analysis_substrate_bridge {
    void* analysis;
    neural_substrate_t* substrate;
    analysis_substrate_config_t config;
    analysis_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

analysis_substrate_config_t analysis_substrate_default_config(void) {
    analysis_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

analysis_substrate_bridge_t* analysis_substrate_bridge_create(void* analysis, neural_substrate_t* substrate, const analysis_substrate_config_t* config) {
    if (!substrate) return NULL;
    analysis_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(analysis_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->analysis = analysis;
    bridge->substrate = substrate;
    bridge->config = config ? *config : analysis_substrate_default_config();
    bridge->effects.analysis_depth = 1.0f;
    bridge->effects.precision_level = 1.0f;
    bridge->effects.processing_speed = 1.0f;
    bridge->effects.decomposition_ability = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void analysis_substrate_bridge_destroy(analysis_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int analysis_substrate_bridge_update(analysis_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables deep analysis and precision */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.analysis_depth = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.precision_level = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables processing speed and decomposition */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.processing_speed = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.decomposition_ability = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.analysis_depth + bridge->effects.precision_level +
                                        bridge->effects.processing_speed + bridge->effects.decomposition_ability) / 4.0f;
    bridge->update_count++;
    return 0;
}

int analysis_substrate_bridge_get_effects(const analysis_substrate_bridge_t* bridge, analysis_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int analysis_substrate_bridge_apply_effects(analysis_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int analysis_substrate_bridge_register_bio_async(analysis_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
