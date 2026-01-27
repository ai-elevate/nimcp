/**
 * @file nimcp_logic_substrate_bridge.c
 * @brief Logic-Neural Substrate Bridge Implementation
 *
 * WHAT: Links logical reasoning to metabolic state
 * WHY: Logical inference requires sustained prefrontal-parietal activation
 * HOW: Monitors ATP/fatigue; modulates inference depth, accuracy, speed
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/logic/nimcp_logic_substrate_bridge.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for logic_substrate_bridge module */
static nimcp_health_agent_t* g_logic_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for logic_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void logic_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_logic_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from logic_substrate_bridge module */
static inline void logic_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_logic_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_logic_substrate_bridge_health_agent, operation, progress);
    }
}

#define LOG_MODULE "LOGIC_SUBSTRATE_BRIDGE"


struct logic_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* logic;
    neural_substrate_t* substrate;
    logic_substrate_config_t config;
    logic_substrate_effects_t effects;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
};

logic_substrate_config_t logic_substrate_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    logic_substrate_bridge_heartbeat("logic_substr_logic_substrate_defa", 0.0f);


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
    if (!substrate) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "logic_substrate_bridge_create: substrate is NULL");
        return NULL;
    }

    /* Phase 8: Heartbeat at operation start */
    logic_substrate_bridge_heartbeat("logic_substr_create", 0.0f);


    logic_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(logic_substrate_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "logic_substrate_bridge_create: failed to allocate bridge");
        return NULL;
    }

    bridge->logic = logic;
    bridge->substrate = substrate;
    bridge->config = config ? *config : logic_substrate_default_config();

    bridge->effects.inference_depth = 1.0f;
    bridge->effects.logical_accuracy = 1.0f;
    bridge->effects.processing_speed = 1.0f;
    bridge->effects.abstraction_capacity = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    NIMCP_LOGGING_INFO("Created %s bridge", "logic_substrate");
    return bridge;
}

void logic_substrate_bridge_destroy(logic_substrate_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    logic_substrate_bridge_heartbeat("logic_substr_destroy", 0.0f);


    if (bridge) nimcp_free(bridge);
}

int logic_substrate_bridge_update(logic_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    /* Phase 8: Heartbeat at operation start */
    logic_substrate_bridge_heartbeat("logic_substr_update", 0.0f);


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
    /* Phase 8: Heartbeat at operation start */
    logic_substrate_bridge_heartbeat("logic_substr_get_effects", 0.0f);


    return 0;
}

int logic_substrate_bridge_apply_effects(logic_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    logic_substrate_bridge_heartbeat("logic_substr_apply_effects", 0.0f);


    return 0;
}

int logic_substrate_bridge_register_bio_async(logic_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    logic_substrate_bridge_heartbeat("logic_substr_register_bio_async", 0.0f);


    bridge->router = router;
    bridge->bio_async_connected = (router != NULL);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int logic_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    logic_substrate_bridge_heartbeat("logic_substr_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Logic_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                logic_substrate_bridge_heartbeat("logic_substr_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

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
