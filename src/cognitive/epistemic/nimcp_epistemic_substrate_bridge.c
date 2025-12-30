/**
 * @file nimcp_epistemic_substrate_bridge.c
 * @brief Epistemic-Neural Substrate Bridge Implementation
 */

#include "cognitive/epistemic/nimcp_epistemic_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct epistemic_substrate_bridge {
    void* epistemic;
    neural_substrate_t* substrate;
    epistemic_substrate_config_t config;
    epistemic_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

epistemic_substrate_config_t epistemic_substrate_default_config(void) {
    epistemic_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

epistemic_substrate_bridge_t* epistemic_substrate_bridge_create(void* epistemic, neural_substrate_t* substrate, const epistemic_substrate_config_t* config) {
    if (!substrate) return NULL;
    epistemic_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(epistemic_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->epistemic = epistemic;
    bridge->substrate = substrate;
    bridge->config = config ? *config : epistemic_substrate_default_config();
    bridge->effects.evidence_integration = 1.0f;
    bridge->effects.belief_updating = 1.0f;
    bridge->effects.certainty_calibration = 1.0f;
    bridge->effects.source_evaluation = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void epistemic_substrate_bridge_destroy(epistemic_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int epistemic_substrate_bridge_update(epistemic_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.evidence_integration = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.source_evaluation = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.belief_updating = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.certainty_calibration = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.evidence_integration + bridge->effects.belief_updating +
                                        bridge->effects.certainty_calibration + bridge->effects.source_evaluation) / 4.0f;
    bridge->update_count++;
    return 0;
}

int epistemic_substrate_bridge_get_effects(const epistemic_substrate_bridge_t* bridge, epistemic_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int epistemic_substrate_bridge_apply_effects(epistemic_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int epistemic_substrate_bridge_register_bio_async(epistemic_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
