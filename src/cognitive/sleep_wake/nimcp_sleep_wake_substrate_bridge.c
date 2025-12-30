/**
 * @file nimcp_sleep_wake_substrate_bridge.c
 * @brief Sleep-Wake-Neural Substrate Bridge Implementation
 */

#include "cognitive/sleep_wake/nimcp_sleep_wake_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct sleep_wake_substrate_bridge {
    void* sleep_wake;
    neural_substrate_t* substrate;
    sleep_wake_substrate_config_t config;
    sleep_wake_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

sleep_wake_substrate_config_t sleep_wake_substrate_default_config(void) {
    sleep_wake_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

sleep_wake_substrate_bridge_t* sleep_wake_substrate_bridge_create(void* sleep_wake, neural_substrate_t* substrate, const sleep_wake_substrate_config_t* config) {
    if (!substrate) return NULL;
    sleep_wake_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(sleep_wake_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->sleep_wake = sleep_wake;
    bridge->substrate = substrate;
    bridge->config = config ? *config : sleep_wake_substrate_default_config();
    bridge->effects.arousal_level = 1.0f;
    bridge->effects.sleep_pressure = 0.0f;
    bridge->effects.circadian_phase = 1.0f;
    bridge->effects.recovery_rate = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void sleep_wake_substrate_bridge_destroy(sleep_wake_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int sleep_wake_substrate_bridge_update(sleep_wake_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP depletion increases sleep pressure */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.arousal_level = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.recovery_rate = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Fatigue directly drives sleep pressure */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.sleep_pressure = clamp_f((1.0f - metabolic_cap) * bridge->config.fatigue_sensitivity, 0.0f, 1.0f);
        bridge->effects.circadian_phase = clamp_f((1.0f - (1.0f - metabolic_cap) * 0.3f), 0.5f, 1.0f);
    }
    bridge->effects.overall_capacity = bridge->effects.arousal_level;
    bridge->update_count++;
    return 0;
}

int sleep_wake_substrate_bridge_get_effects(const sleep_wake_substrate_bridge_t* bridge, sleep_wake_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int sleep_wake_substrate_bridge_apply_effects(sleep_wake_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int sleep_wake_substrate_bridge_register_bio_async(sleep_wake_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
