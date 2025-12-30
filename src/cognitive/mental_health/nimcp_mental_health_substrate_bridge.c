/**
 * @file nimcp_mental_health_substrate_bridge.c
 * @brief Mental Health-Neural Substrate Bridge Implementation
 */

#include "cognitive/mental_health/nimcp_mental_health_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct mental_health_substrate_bridge {
    void* mental_health;
    neural_substrate_t* substrate;
    mental_health_substrate_config_t config;
    mental_health_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

mental_health_substrate_config_t mental_health_substrate_default_config(void) {
    mental_health_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

mental_health_substrate_bridge_t* mental_health_substrate_bridge_create(void* mental_health, neural_substrate_t* substrate, const mental_health_substrate_config_t* config) {
    if (!substrate) return NULL;
    mental_health_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(mental_health_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->mental_health = mental_health;
    bridge->substrate = substrate;
    bridge->config = config ? *config : mental_health_substrate_default_config();
    bridge->effects.resilience_level = 1.0f;
    bridge->effects.coping_capacity = 1.0f;
    bridge->effects.emotional_stability = 1.0f;
    bridge->effects.wellbeing_level = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void mental_health_substrate_bridge_destroy(mental_health_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int mental_health_substrate_bridge_update(mental_health_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP fundamentally underpins mental health and resilience */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.resilience_level = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.wellbeing_level = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables coping and stability */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.coping_capacity = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.emotional_stability = clamp_f(metabolic_cap * 0.95f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.resilience_level + bridge->effects.coping_capacity +
                                        bridge->effects.emotional_stability + bridge->effects.wellbeing_level) / 4.0f;
    bridge->update_count++;
    return 0;
}

int mental_health_substrate_bridge_get_effects(const mental_health_substrate_bridge_t* bridge, mental_health_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int mental_health_substrate_bridge_apply_effects(mental_health_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int mental_health_substrate_bridge_register_bio_async(mental_health_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
