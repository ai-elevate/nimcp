/**
 * @file nimcp_autobio_substrate_bridge.c
 * @brief Autobiographical Memory-Neural Substrate Bridge Implementation
 */

#include "cognitive/autobiographical_memory/nimcp_autobio_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct autobio_substrate_bridge {
    void* autobio;
    neural_substrate_t* substrate;
    autobio_substrate_config_t config;
    autobio_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

autobio_substrate_config_t autobio_substrate_default_config(void) {
    autobio_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

autobio_substrate_bridge_t* autobio_substrate_bridge_create(void* autobio, neural_substrate_t* substrate, const autobio_substrate_config_t* config) {
    if (!substrate) return NULL;
    autobio_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(autobio_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->autobio = autobio;
    bridge->substrate = substrate;
    bridge->config = config ? *config : autobio_substrate_default_config();
    bridge->effects.recall_vividness = 1.0f;
    bridge->effects.detail_resolution = 1.0f;
    bridge->effects.temporal_accuracy = 1.0f;
    bridge->effects.narrative_coherence = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void autobio_substrate_bridge_destroy(autobio_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int autobio_substrate_bridge_update(autobio_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables vivid recall and detail resolution */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.recall_vividness = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.detail_resolution = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables temporal accuracy and coherence */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.temporal_accuracy = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.narrative_coherence = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.recall_vividness + bridge->effects.detail_resolution +
                                        bridge->effects.temporal_accuracy + bridge->effects.narrative_coherence) / 4.0f;
    bridge->update_count++;
    return 0;
}

int autobio_substrate_bridge_get_effects(const autobio_substrate_bridge_t* bridge, autobio_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int autobio_substrate_bridge_apply_effects(autobio_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int autobio_substrate_bridge_register_bio_async(autobio_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
