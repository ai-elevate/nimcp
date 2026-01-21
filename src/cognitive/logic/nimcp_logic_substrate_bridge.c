/**
 * @file nimcp_logic_substrate_bridge.c
 * @brief Logic-Neural Substrate Bridge Implementation
 *
 * WHAT: Links logical reasoning to metabolic state
 * WHY: Logical inference requires sustained prefrontal-parietal activation
 * HOW: Monitors ATP/fatigue; modulates inference depth, accuracy, speed
 */

#include "cognitive/logic/nimcp_logic_substrate_bridge.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

struct logic_substrate_bridge {
    void* logic;
    neural_substrate_t* substrate;
    logic_substrate_config_t config;
    logic_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

logic_substrate_config_t logic_substrate_default_config(void) {
    logic_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

logic_substrate_bridge_t* logic_substrate_bridge_create(void* logic, neural_substrate_t* substrate, const logic_substrate_config_t* config) {
    if (!substrate) return NULL;

    logic_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(logic_substrate_bridge_t));
    if (!bridge) return NULL;

    bridge->logic = logic;
    bridge->substrate = substrate;
    bridge->config = config ? *config : logic_substrate_default_config();

    bridge->effects.inference_depth = 1.0f;
    bridge->effects.logical_accuracy = 1.0f;
    bridge->effects.processing_speed = 1.0f;
    bridge->effects.abstraction_capacity = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void logic_substrate_bridge_destroy(logic_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int logic_substrate_bridge_update(logic_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        bridge->effects.inference_depth = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.logical_accuracy = nimcp_clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.abstraction_capacity = nimcp_clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.processing_speed = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.inference_depth +
                                        bridge->effects.logical_accuracy +
                                        bridge->effects.processing_speed +
                                        bridge->effects.abstraction_capacity) / 4.0f;

    bridge->update_count++;
    return 0;
}

int logic_substrate_bridge_get_effects(const logic_substrate_bridge_t* bridge, logic_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int logic_substrate_bridge_apply_effects(logic_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    return 0;
}

int logic_substrate_bridge_register_bio_async(logic_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int logic_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Logic_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Logic_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Logic_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
