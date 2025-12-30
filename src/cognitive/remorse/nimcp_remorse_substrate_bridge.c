/**
 * @file nimcp_remorse_substrate_bridge.c
 * @brief Remorse-Neural Substrate Bridge Implementation
 */

#include "cognitive/remorse/nimcp_remorse_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct remorse_substrate_bridge {
    void* remorse;
    neural_substrate_t* substrate;
    remorse_substrate_config_t config;
    remorse_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

remorse_substrate_config_t remorse_substrate_default_config(void) {
    remorse_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

remorse_substrate_bridge_t* remorse_substrate_bridge_create(void* remorse, neural_substrate_t* substrate, const remorse_substrate_config_t* config) {
    if (!substrate) return NULL;
    remorse_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(remorse_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->remorse = remorse;
    bridge->substrate = substrate;
    bridge->config = config ? *config : remorse_substrate_default_config();
    bridge->effects.guilt_processing = 1.0f;
    bridge->effects.repair_motivation = 1.0f;
    bridge->effects.moral_learning = 1.0f;
    bridge->effects.self_forgiveness = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void remorse_substrate_bridge_destroy(remorse_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int remorse_substrate_bridge_update(remorse_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables guilt processing and repair motivation */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.guilt_processing = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.repair_motivation = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables moral learning and self-forgiveness */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.moral_learning = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.self_forgiveness = clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.guilt_processing + bridge->effects.repair_motivation +
                                        bridge->effects.moral_learning + bridge->effects.self_forgiveness) / 4.0f;
    bridge->update_count++;
    return 0;
}

int remorse_substrate_bridge_get_effects(const remorse_substrate_bridge_t* bridge, remorse_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int remorse_substrate_bridge_apply_effects(remorse_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int remorse_substrate_bridge_register_bio_async(remorse_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
