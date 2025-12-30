/**
 * @file nimcp_knowledge_substrate_bridge.c
 * @brief Knowledge-Neural Substrate Bridge Implementation
 *
 * WHAT: Links knowledge/semantic memory to metabolic state
 * WHY: Semantic retrieval requires temporal-parietal resources
 * HOW: Monitors ATP/fatigue; modulates retrieval fluency, accuracy, integration
 */

#include "cognitive/knowledge/nimcp_knowledge_substrate_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

struct knowledge_substrate_bridge {
    void* knowledge;
    neural_substrate_t* substrate;
    knowledge_substrate_config_t config;
    knowledge_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

static float clamp_f(float v, float min, float max) {
    return v < min ? min : (v > max ? max : v);
}

knowledge_substrate_config_t knowledge_substrate_default_config(void) {
    knowledge_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

knowledge_substrate_bridge_t* knowledge_substrate_bridge_create(void* knowledge, neural_substrate_t* substrate, const knowledge_substrate_config_t* config) {
    if (!substrate) return NULL;

    knowledge_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(knowledge_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->knowledge = knowledge;
    bridge->substrate = substrate;
    bridge->config = config ? *config : knowledge_substrate_default_config();

    bridge->effects.retrieval_speed = 1.0f;
    bridge->effects.retrieval_accuracy = 1.0f;
    bridge->effects.consolidation_rate = 1.0f;
    bridge->effects.association_strength = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void knowledge_substrate_bridge_destroy(knowledge_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int knowledge_substrate_bridge_update(knowledge_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        bridge->effects.retrieval_accuracy = clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.association_strength = clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.retrieval_speed = clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.consolidation_rate = clamp_f(metabolic_cap * 0.95f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.retrieval_speed +
                                        bridge->effects.retrieval_accuracy +
                                        bridge->effects.consolidation_rate +
                                        bridge->effects.association_strength) / 4.0f;

    bridge->update_count++;
    return 0;
}

int knowledge_substrate_bridge_get_effects(const knowledge_substrate_bridge_t* bridge, knowledge_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int knowledge_substrate_bridge_apply_effects(knowledge_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int knowledge_substrate_bridge_register_bio_async(knowledge_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}
