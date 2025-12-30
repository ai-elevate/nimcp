/**
 * @file nimcp_mirror_substrate_bridge.c
 * @brief Mirror Neuron-Neural Substrate Bridge Implementation
 *
 * WHAT: Links mirror neuron simulation to metabolic state
 * WHY: Action understanding and imitation require motor-premotor resources
 * HOW: Monitors ATP/fatigue; modulates mirroring fidelity, empathy, imitation
 */

#include "cognitive/mirror_neurons/nimcp_mirror_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct mirror_substrate_bridge {
    void* mirror;
    neural_substrate_t* substrate;
    mirror_substrate_config_t config;
    mirror_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

mirror_substrate_config_t mirror_substrate_default_config(void) {
    mirror_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

mirror_substrate_bridge_t* mirror_substrate_bridge_create(void* mirror, neural_substrate_t* substrate, const mirror_substrate_config_t* config) {
    if (!substrate) return NULL;

    mirror_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(mirror_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->mirror = mirror;
    bridge->substrate = substrate;
    bridge->config = config ? *config : mirror_substrate_default_config();

    bridge->effects.mirroring_fidelity = 1.0f;
    bridge->effects.empathic_resonance = 1.0f;
    bridge->effects.imitation_capacity = 1.0f;
    bridge->effects.action_prediction = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void mirror_substrate_bridge_destroy(mirror_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int mirror_substrate_bridge_update(mirror_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        bridge->effects.mirroring_fidelity = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.action_prediction = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.empathic_resonance = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.imitation_capacity = clamp_f(metabolic_cap * 0.95f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.mirroring_fidelity +
                                        bridge->effects.empathic_resonance +
                                        bridge->effects.imitation_capacity +
                                        bridge->effects.action_prediction) / 4.0f;

    bridge->update_count++;
    return 0;
}

int mirror_substrate_bridge_get_effects(const mirror_substrate_bridge_t* bridge, mirror_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int mirror_substrate_bridge_apply_effects(mirror_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int mirror_substrate_bridge_register_bio_async(mirror_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
