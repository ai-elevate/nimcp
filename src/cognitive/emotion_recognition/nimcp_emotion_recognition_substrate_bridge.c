/**
 * @file nimcp_emotion_recognition_substrate_bridge.c
 * @brief Emotion Recognition-Neural Substrate Bridge Implementation
 */

#include "cognitive/emotion_recognition/nimcp_emotion_recognition_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct emotion_recognition_substrate_bridge {
    void* emotion_recognition;
    neural_substrate_t* substrate;
    emotion_recognition_substrate_config_t config;
    emotion_recognition_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

emotion_recognition_substrate_config_t emotion_recognition_substrate_default_config(void) {
    emotion_recognition_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

emotion_recognition_substrate_bridge_t* emotion_recognition_substrate_bridge_create(void* emotion_recognition, neural_substrate_t* substrate, const emotion_recognition_substrate_config_t* config) {
    if (!substrate) return NULL;
    emotion_recognition_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(emotion_recognition_substrate_bridge_t));
    if (!bridge) return NULL;
    bridge->emotion_recognition = emotion_recognition;
    bridge->substrate = substrate;
    bridge->config = config ? *config : emotion_recognition_substrate_default_config();
    bridge->effects.recognition_accuracy = 1.0f;
    bridge->effects.detection_speed = 1.0f;
    bridge->effects.subtle_sensitivity = 1.0f;
    bridge->effects.context_integration = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void emotion_recognition_substrate_bridge_destroy(emotion_recognition_substrate_bridge_t* bridge) { if (bridge) nimcp_free(bridge); }

int emotion_recognition_substrate_bridge_update(emotion_recognition_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables recognition accuracy and subtle sensitivity */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.recognition_accuracy = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.subtle_sensitivity = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables detection speed and context integration */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.detection_speed = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.context_integration = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.recognition_accuracy + bridge->effects.detection_speed +
                                        bridge->effects.subtle_sensitivity + bridge->effects.context_integration) / 4.0f;
    bridge->update_count++;
    return 0;
}

int emotion_recognition_substrate_bridge_get_effects(const emotion_recognition_substrate_bridge_t* bridge, emotion_recognition_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int emotion_recognition_substrate_bridge_apply_effects(emotion_recognition_substrate_bridge_t* bridge) { return bridge ? 0 : -1; }

int emotion_recognition_substrate_bridge_register_bio_async(emotion_recognition_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
