/**
 * @file nimcp_explanations_substrate_bridge.c
 * @brief Explanations-Neural Substrate Bridge Implementation
 */

#include "cognitive/explanations/nimcp_explanations_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct explanations_substrate_bridge {
    void* explanations;
    neural_substrate_t* substrate;
    explanations_substrate_config_t config;
    explanations_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

explanations_substrate_config_t explanations_substrate_default_config(void) {
    explanations_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

explanations_substrate_bridge_t* explanations_substrate_bridge_create(void* explanations, neural_substrate_t* substrate, const explanations_substrate_config_t* config) {
    if (!substrate) return NULL;
    explanations_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(explanations_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->explanations = explanations;
    bridge->substrate = substrate;
    bridge->config = config ? *config : explanations_substrate_default_config();
    bridge->effects.explanation_depth = 1.0f;
    bridge->effects.coherence_quality = 1.0f;
    bridge->effects.abstraction_level = 1.0f;
    bridge->effects.integration_breadth = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void explanations_substrate_bridge_destroy(explanations_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int explanations_substrate_bridge_update(explanations_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.explanation_depth = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.abstraction_level = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.coherence_quality = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.integration_breadth = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.explanation_depth + bridge->effects.coherence_quality +
                                        bridge->effects.abstraction_level + bridge->effects.integration_breadth) / 4.0f;
    bridge->update_count++;
    return 0;
}

int explanations_substrate_bridge_get_effects(const explanations_substrate_bridge_t* bridge, explanations_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int explanations_substrate_bridge_apply_effects(explanations_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int explanations_substrate_bridge_register_bio_async(explanations_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
