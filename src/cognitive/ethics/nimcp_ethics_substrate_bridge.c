/**
 * @file nimcp_ethics_substrate_bridge.c
 * @brief Ethics-Neural Substrate Bridge Implementation
 *
 * WHAT: Links ethical reasoning to metabolic state
 * WHY: Moral cognition requires sustained prefrontal resources
 * HOW: Monitors ATP/fatigue; modulates moral reasoning, judgment speed, consistency
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/ethics/nimcp_ethics_substrate_bridge.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

struct ethics_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* ethics;
    neural_substrate_t* substrate;
    ethics_substrate_config_t config;
    ethics_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

ethics_substrate_config_t ethics_substrate_default_config(void) {
    ethics_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

ethics_substrate_bridge_t* ethics_substrate_bridge_create(void* ethics, neural_substrate_t* substrate, const ethics_substrate_config_t* config) {
    if (!substrate) return NULL;

    ethics_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(ethics_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->ethics = ethics;
    bridge->substrate = substrate;
    bridge->config = config ? *config : ethics_substrate_default_config();

    bridge->effects.moral_clarity = 1.0f;
    bridge->effects.deliberation_depth = 1.0f;
    bridge->effects.bias_resistance = 1.0f;
    bridge->effects.empathy_capacity = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void ethics_substrate_bridge_destroy(ethics_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int ethics_substrate_bridge_update(ethics_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        bridge->effects.moral_clarity = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.bias_resistance = nimcp_clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.deliberation_depth = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.empathy_capacity = nimcp_clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.moral_clarity +
                                        bridge->effects.deliberation_depth +
                                        bridge->effects.bias_resistance +
                                        bridge->effects.empathy_capacity) / 4.0f;

    bridge->update_count++;
    return 0;
}

int ethics_substrate_bridge_get_effects(const ethics_substrate_bridge_t* bridge, ethics_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int ethics_substrate_bridge_apply_effects(ethics_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int ethics_substrate_bridge_register_bio_async(ethics_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Ethics Substrate Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int ethics_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Ethics_Substrate_Bridge_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("Ethics substrate bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Ethics_Substrate_Bridge_Module");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Ethics_Substrate_Bridge_Module");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
