/**
 * @file nimcp_theory_of_mind_substrate_bridge.c
 * @brief Theory of Mind-Neural Substrate Bridge Implementation
 *
 * WHAT: Links ToM to metabolic state
 * WHY: Mentalizing requires sustained prefrontal and TPJ resources
 * HOW: Monitors ATP/fatigue; modulates mentalizing, perspective-taking
 */

#include "cognitive/theory_of_mind/nimcp_theory_of_mind_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct tom_substrate_bridge {
    void* tom;
    neural_substrate_t* substrate;
    tom_substrate_config_t config;
    tom_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

tom_substrate_config_t tom_substrate_default_config(void) {
    tom_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

tom_substrate_bridge_t* tom_substrate_bridge_create(void* tom, neural_substrate_t* substrate, const tom_substrate_config_t* config) {
    if (!substrate) return NULL;

    tom_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(tom_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->tom = tom;
    bridge->substrate = substrate;
    bridge->config = config ? *config : tom_substrate_default_config();

    bridge->effects.mentalizing_capacity = 1.0f;
    bridge->effects.perspective_taking = 1.0f;
    bridge->effects.recursive_depth = 1.0f;
    bridge->effects.false_belief_reasoning = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void tom_substrate_bridge_destroy(tom_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int tom_substrate_bridge_update(tom_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Mentalizing capacity requires stable prefrontal resources */
        bridge->effects.mentalizing_capacity = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Recursive depth is exponentially sensitive to ATP (each level compounds) */
        bridge->effects.recursive_depth = clamp_f(powf(atp, 0.8f) * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Perspective-taking degrades with fatigue */
        bridge->effects.perspective_taking = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* False belief reasoning is particularly vulnerable to fatigue */
        bridge->effects.false_belief_reasoning = clamp_f(metabolic_cap * 0.8f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.mentalizing_capacity +
                                        bridge->effects.perspective_taking +
                                        bridge->effects.recursive_depth +
                                        bridge->effects.false_belief_reasoning) / 4.0f;

    bridge->update_count++;
    return 0;
}

int tom_substrate_bridge_get_effects(const tom_substrate_bridge_t* bridge, tom_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int tom_substrate_bridge_apply_effects(tom_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int tom_substrate_bridge_register_bio_async(tom_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
