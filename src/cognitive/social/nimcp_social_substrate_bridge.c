/**
 * @file nimcp_social_substrate_bridge.c
 * @brief Social Cognition-Neural Substrate Bridge Implementation
 *
 * WHAT: Links social cognition to metabolic state
 * WHY: Social processing requires sustained prefrontal and limbic resources
 * HOW: Monitors ATP/fatigue; modulates bonding, loyalty, trust
 */

#include "cognitive/social/nimcp_social_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct social_substrate_bridge {
    void* social;
    neural_substrate_t* substrate;
    social_substrate_config_t config;
    social_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

social_substrate_config_t social_substrate_default_config(void) {
    social_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

social_substrate_bridge_t* social_substrate_bridge_create(void* social, neural_substrate_t* substrate, const social_substrate_config_t* config) {
    if (!substrate) return NULL;

    social_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(social_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->social = social;
    bridge->substrate = substrate;
    bridge->config = config ? *config : social_substrate_default_config();

    bridge->effects.bonding_capacity = 1.0f;
    bridge->effects.loyalty_strength = 1.0f;
    bridge->effects.trust_evaluation = 1.0f;
    bridge->effects.prosocial_motivation = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void social_substrate_bridge_destroy(social_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int social_substrate_bridge_update(social_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Bonding capacity requires sustained neural resources */
        bridge->effects.bonding_capacity = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Trust evaluation is cognitively demanding */
        bridge->effects.trust_evaluation = clamp_f(atp * 0.95f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Loyalty strength maintained but may waver under fatigue */
        bridge->effects.loyalty_strength = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Prosocial motivation decreases with fatigue */
        bridge->effects.prosocial_motivation = clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.bonding_capacity +
                                        bridge->effects.loyalty_strength +
                                        bridge->effects.trust_evaluation +
                                        bridge->effects.prosocial_motivation) / 4.0f;

    bridge->update_count++;
    return 0;
}

int social_substrate_bridge_get_effects(const social_substrate_bridge_t* bridge, social_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int social_substrate_bridge_apply_effects(social_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int social_substrate_bridge_register_bio_async(social_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
