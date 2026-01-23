/**
 * @file nimcp_brainstem_substrate_bridge.c
 * @brief Brainstem-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "core/brainstem/nimcp_brainstem_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

struct brainstem_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* brainstem;
    neural_substrate_t* substrate;
    brainstem_substrate_config_t config;
    brainstem_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

brainstem_substrate_config_t brainstem_substrate_default_config(void) {
    brainstem_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.3f };
    return cfg;
}

brainstem_substrate_bridge_t* brainstem_substrate_bridge_create(void* brainstem, neural_substrate_t* substrate, const brainstem_substrate_config_t* config) {
    if (!substrate) return NULL;
    brainstem_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(brainstem_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->brainstem = brainstem;
    bridge->substrate = substrate;
    bridge->config = config ? *config : brainstem_substrate_default_config();
    bridge->effects.arousal_level = 1.0f;
    bridge->effects.autonomic_balance = 0.5f; /* Balanced sympathetic/parasympathetic */
    bridge->effects.vital_stability = 1.0f;
    bridge->effects.reflex_speed = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void brainstem_substrate_bridge_destroy(brainstem_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int brainstem_substrate_bridge_update(brainstem_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, fatigue = 1.0f - metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP directly affects arousal and vital stability */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.arousal_level = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.vital_stability = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue modulates autonomic balance and reflex speed */
    if (bridge->config.enable_fatigue_modulation) {
        /* High fatigue shifts toward parasympathetic (rest) */
        bridge->effects.autonomic_balance = clamp_f(0.5f - fatigue * 0.3f * bridge->config.fatigue_sensitivity, 0.2f, 0.8f);
        bridge->effects.reflex_speed = clamp_f((1.0f - fatigue) * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.arousal_level + bridge->effects.vital_stability +
                                        bridge->effects.reflex_speed) / 3.0f;
    bridge->update_count++;
    return 0;
}

int brainstem_substrate_bridge_get_effects(const brainstem_substrate_bridge_t* bridge, brainstem_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int brainstem_substrate_bridge_apply_effects(brainstem_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int brainstem_substrate_bridge_register_bio_async(brainstem_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
