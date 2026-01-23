/**
 * @file nimcp_temporal_substrate_bridge.c
 * @brief Temporal Cortex-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/temporal/nimcp_temporal_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct temporal_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* temporal;
    neural_substrate_t* substrate;
    temporal_substrate_config_t config;
    temporal_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

temporal_substrate_config_t temporal_substrate_default_config(void) {
    temporal_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

temporal_substrate_bridge_t* temporal_substrate_bridge_create(void* temporal, neural_substrate_t* substrate, const temporal_substrate_config_t* config) {
    if (!substrate) return NULL;
    temporal_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(temporal_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->temporal = temporal;
    bridge->substrate = substrate;
    bridge->config = config ? *config : temporal_substrate_default_config();
    bridge->effects.language_processing = 1.0f;
    bridge->effects.object_recognition = 1.0f;
    bridge->effects.semantic_access = 1.0f;
    bridge->effects.auditory_processing = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void temporal_substrate_bridge_destroy(temporal_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int temporal_substrate_bridge_update(temporal_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP affects language and semantic access */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.language_processing = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.semantic_access = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue affects object recognition and auditory processing */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.object_recognition = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.auditory_processing = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.language_processing + bridge->effects.object_recognition +
                                        bridge->effects.semantic_access + bridge->effects.auditory_processing) / 4.0f;
    bridge->update_count++;
    return 0;
}

int temporal_substrate_bridge_get_effects(const temporal_substrate_bridge_t* bridge, temporal_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int temporal_substrate_bridge_apply_effects(temporal_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int temporal_substrate_bridge_register_bio_async(temporal_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
