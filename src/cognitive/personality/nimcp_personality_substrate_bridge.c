/**
 * @file nimcp_personality_substrate_bridge.c
 * @brief Personality-Neural Substrate Bridge Implementation
 *
 * WHAT: Links personality traits to metabolic state
 * WHY: Personality expression varies with energy and fatigue levels
 * HOW: Monitors ATP/fatigue; modulates trait expression, self-regulation, consistency
 */

#include "cognitive/personality/nimcp_personality_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct personality_substrate_bridge {
    void* personality;
    neural_substrate_t* substrate;
    personality_substrate_config_t config;
    personality_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

personality_substrate_config_t personality_substrate_default_config(void) {
    personality_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

personality_substrate_bridge_t* personality_substrate_bridge_create(void* personality, neural_substrate_t* substrate, const personality_substrate_config_t* config) {
    if (!substrate) return NULL;

    personality_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(personality_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->personality = personality;
    bridge->substrate = substrate;
    bridge->config = config ? *config : personality_substrate_default_config();

    bridge->effects.self_regulation = 1.0f;
    bridge->effects.trait_consistency = 1.0f;
    bridge->effects.stress_resilience = 1.0f;
    bridge->effects.social_energy = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void personality_substrate_bridge_destroy(personality_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int personality_substrate_bridge_update(personality_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    /* ATP depletion reduces self-control (ego depletion) */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.self_regulation = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.trait_consistency = clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    /* Fatigue increases impulsivity, reduces agreeableness */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.stress_resilience = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.social_energy = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.self_regulation +
                                        bridge->effects.trait_consistency +
                                        bridge->effects.stress_resilience +
                                        bridge->effects.social_energy) / 4.0f;

    bridge->update_count++;
    return 0;
}

int personality_substrate_bridge_get_effects(const personality_substrate_bridge_t* bridge, personality_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int personality_substrate_bridge_apply_effects(personality_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int personality_substrate_bridge_register_bio_async(personality_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
