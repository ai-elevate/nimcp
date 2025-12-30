/**
 * @file nimcp_emotional_tagging_substrate_bridge.c
 * @brief Emotional Tagging-Neural Substrate Bridge Implementation
 */

#include "cognitive/emotional_tagging/nimcp_emotional_tagging_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct emotional_tagging_substrate_bridge {
    void* emotional_tagging;
    neural_substrate_t* substrate;
    emotional_tagging_substrate_config_t config;
    emotional_tagging_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

emotional_tagging_substrate_config_t emotional_tagging_substrate_default_config(void) {
    emotional_tagging_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

emotional_tagging_substrate_bridge_t* emotional_tagging_substrate_bridge_create(void* emotional_tagging, neural_substrate_t* substrate, const emotional_tagging_substrate_config_t* config) {
    if (!substrate) return NULL;
    emotional_tagging_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(emotional_tagging_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->emotional_tagging = emotional_tagging;
    bridge->substrate = substrate;
    bridge->config = config ? *config : emotional_tagging_substrate_default_config();
    bridge->effects.tagging_strength = 1.0f;
    bridge->effects.tag_specificity = 1.0f;
    bridge->effects.consolidation_quality = 1.0f;
    bridge->effects.retrieval_accuracy = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void emotional_tagging_substrate_bridge_destroy(emotional_tagging_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int emotional_tagging_substrate_bridge_update(emotional_tagging_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables tagging strength and consolidation */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.tagging_strength = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.consolidation_quality = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables specificity and retrieval */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.tag_specificity = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.retrieval_accuracy = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.tagging_strength + bridge->effects.tag_specificity +
                                        bridge->effects.consolidation_quality + bridge->effects.retrieval_accuracy) / 4.0f;
    bridge->update_count++;
    return 0;
}

int emotional_tagging_substrate_bridge_get_effects(const emotional_tagging_substrate_bridge_t* bridge, emotional_tagging_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int emotional_tagging_substrate_bridge_apply_effects(emotional_tagging_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int emotional_tagging_substrate_bridge_register_bio_async(emotional_tagging_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
