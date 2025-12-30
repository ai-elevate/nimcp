/**
 * @file nimcp_emotion_tensor_substrate_bridge.c
 * @brief Emotion Tensor-Neural Substrate Bridge Implementation
 *
 * WHAT: Links tensor emotional representation to metabolic state
 * WHY: Complex emotional tensors require high energy for computation
 * HOW: Monitors ATP/fatigue; modulates intensity, valence, complexity
 */

#include "cognitive/emotion_tensor/nimcp_emotion_tensor_substrate_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>
#include <math.h>

struct emotion_tensor_substrate_bridge {
    bridge_base_t base;                 /* MUST be first: base bridge infrastructure */
    void* emotion_tensor;
    neural_substrate_t* substrate;
    emotion_tensor_substrate_config_t config;
    emotion_tensor_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

emotion_tensor_substrate_config_t emotion_tensor_substrate_default_config(void) {
    emotion_tensor_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

emotion_tensor_substrate_bridge_t* emotion_tensor_substrate_bridge_create(void* emotion_tensor, neural_substrate_t* substrate, const emotion_tensor_substrate_config_t* config) {
    if (!substrate) return NULL;

    emotion_tensor_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(emotion_tensor_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->emotion_tensor = emotion_tensor;
    bridge->substrate = substrate;
    bridge->config = config ? *config : emotion_tensor_substrate_default_config();

    /* Initialize mutex */
    bridge->base.mutex = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex for emotion tensor substrate bridge");
        nimcp_free(bridge);
        return NULL;
    }

    if (nimcp_platform_mutex_init(bridge->base.mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex for emotion tensor substrate bridge");
        nimcp_free(bridge->base.mutex);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.intensity_capacity = 1.0f;
    bridge->effects.valence_resolution = 1.0f;
    bridge->effects.tensor_complexity = 1.0f;
    bridge->effects.regulation_capacity = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void emotion_tensor_substrate_bridge_destroy(emotion_tensor_substrate_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge->base.mutex);
    }

    nimcp_free(bridge);
}

int emotion_tensor_substrate_bridge_update(emotion_tensor_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    nimcp_platform_mutex_lock(bridge->base.mutex);

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) {
        nimcp_platform_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Tensor complexity scales with available ATP */
        bridge->effects.tensor_complexity = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.valence_resolution = clamp_f(atp * 0.9f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Regulation decreases with fatigue, intensity may paradoxically increase */
        bridge->effects.regulation_capacity = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Low metabolic capacity leads to heightened emotional reactivity */
        bridge->effects.intensity_capacity = clamp_f(1.0f + (1.0f - metabolic_cap) * 0.3f, min_cap, 1.3f);
    }

    bridge->effects.overall_capacity = (bridge->effects.intensity_capacity +
                                        bridge->effects.valence_resolution +
                                        bridge->effects.tensor_complexity +
                                        bridge->effects.regulation_capacity) / 4.0f;

    bridge->update_count++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int emotion_tensor_substrate_bridge_get_effects(const emotion_tensor_substrate_bridge_t* bridge, emotion_tensor_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int emotion_tensor_substrate_bridge_apply_effects(emotion_tensor_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int emotion_tensor_substrate_bridge_register_bio_async(emotion_tensor_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
