/**
 * @file nimcp_fractal_cognitive_substrate_bridge.c
 * @brief Fractal Cognitive-Neural Substrate Bridge Implementation
 *
 * WHAT: Links fractal cognitive processing to metabolic state
 * WHY: Scale-free computations require sustained energy
 * HOW: Monitors ATP/fatigue; modulates depth, self-similarity, complexity
 */

#include "cognitive/fractal_cognitive/nimcp_fractal_cognitive_substrate_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>
#include <math.h>

struct fractal_cognitive_substrate_bridge {
    bridge_base_t base;                 /* MUST be first: base bridge infrastructure */
    void* fractal_cognitive;
    neural_substrate_t* substrate;
    fractal_cognitive_substrate_config_t config;
    fractal_cognitive_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

fractal_cognitive_substrate_config_t fractal_cognitive_substrate_default_config(void) {
    fractal_cognitive_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

fractal_cognitive_substrate_bridge_t* fractal_cognitive_substrate_bridge_create(void* fractal_cognitive, neural_substrate_t* substrate, const fractal_cognitive_substrate_config_t* config) {
    if (!substrate) return NULL;

    fractal_cognitive_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(fractal_cognitive_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->fractal_cognitive = fractal_cognitive;
    bridge->substrate = substrate;
    bridge->config = config ? *config : fractal_cognitive_substrate_default_config();

    /* Initialize mutex */
    bridge->base.mutex = (nimcp_mutex_t*)nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bridge->base.mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate mutex for fractal cognitive substrate bridge");
        nimcp_free(bridge);
        return NULL;
    }

    if (nimcp_platform_mutex_init(bridge->base.mutex, false) != 0) {
        NIMCP_LOGGING_ERROR("Failed to initialize mutex for fractal cognitive substrate bridge");
        nimcp_free(bridge->base.mutex);
        nimcp_free(bridge);
        return NULL;
    }

    bridge->effects.fractal_depth = 1.0f;
    bridge->effects.self_similarity = 1.0f;
    bridge->effects.scale_invariance = 1.0f;
    bridge->effects.complexity_capacity = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void fractal_cognitive_substrate_bridge_destroy(fractal_cognitive_substrate_bridge_t* bridge) {
    if (!bridge) return;

    /* Destroy mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
        nimcp_free(bridge->base.mutex);
    }

    nimcp_free(bridge);
}

int fractal_cognitive_substrate_bridge_update(fractal_cognitive_substrate_bridge_t* bridge) {
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
        /* Fractal depth is exponentially sensitive to ATP (each level compounds) */
        bridge->effects.fractal_depth = clamp_f(powf(atp, 0.7f) * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Scale invariance requires consistent ATP across scales */
        bridge->effects.scale_invariance = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Self-similarity degrades with fatigue */
        bridge->effects.self_similarity = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Complexity generation requires sustained metabolic resources */
        bridge->effects.complexity_capacity = clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.fractal_depth +
                                        bridge->effects.self_similarity +
                                        bridge->effects.scale_invariance +
                                        bridge->effects.complexity_capacity) / 4.0f;

    bridge->update_count++;

    nimcp_platform_mutex_unlock(bridge->base.mutex);
    return 0;
}

int fractal_cognitive_substrate_bridge_get_effects(const fractal_cognitive_substrate_bridge_t* bridge, fractal_cognitive_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int fractal_cognitive_substrate_bridge_apply_effects(fractal_cognitive_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int fractal_cognitive_substrate_bridge_register_bio_async(fractal_cognitive_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
