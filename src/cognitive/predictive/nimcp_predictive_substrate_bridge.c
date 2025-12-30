/**
 * @file nimcp_predictive_substrate_bridge.c
 * @brief Predictive Coding-Neural Substrate Bridge Implementation
 *
 * WHAT: Links predictive processing to metabolic state
 * WHY: Prediction requires hierarchical cortical resources
 * HOW: Monitors ATP/fatigue; modulates prediction accuracy, precision, update rate
 */

#include "cognitive/predictive/nimcp_predictive_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct predictive_substrate_bridge {
    void* predictive;
    neural_substrate_t* substrate;
    predictive_substrate_config_t config;
    predictive_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

predictive_substrate_config_t predictive_substrate_default_config(void) {
    predictive_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

predictive_substrate_bridge_t* predictive_substrate_bridge_create(void* predictive, neural_substrate_t* substrate, const predictive_substrate_config_t* config) {
    if (!substrate) return NULL;

    predictive_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(predictive_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->predictive = predictive;
    bridge->substrate = substrate;
    bridge->config = config ? *config : predictive_substrate_default_config();

    bridge->effects.prediction_precision = 1.0f;
    bridge->effects.error_sensitivity = 1.0f;
    bridge->effects.model_update_rate = 1.0f;
    bridge->effects.hierarchical_depth = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void predictive_substrate_bridge_destroy(predictive_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int predictive_substrate_bridge_update(predictive_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        bridge->effects.prediction_precision = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.hierarchical_depth = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.error_sensitivity = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.model_update_rate = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.prediction_precision +
                                        bridge->effects.error_sensitivity +
                                        bridge->effects.model_update_rate +
                                        bridge->effects.hierarchical_depth) / 4.0f;

    bridge->update_count++;
    return 0;
}

int predictive_substrate_bridge_get_effects(const predictive_substrate_bridge_t* bridge, predictive_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int predictive_substrate_bridge_apply_effects(predictive_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int predictive_substrate_bridge_register_bio_async(predictive_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
