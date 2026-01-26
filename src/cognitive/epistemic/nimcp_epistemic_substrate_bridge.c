/**
 * @file nimcp_epistemic_substrate_bridge.c
 * @brief Epistemic-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/epistemic/nimcp_epistemic_substrate_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for epistemic_substrate_bridge module */
static nimcp_health_agent_t* g_epistemic_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for epistemic_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void epistemic_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_epistemic_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from epistemic_substrate_bridge module */
static inline void epistemic_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_epistemic_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_epistemic_substrate_bridge_health_agent, operation, progress);
    }
}


struct epistemic_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* epistemic;
    neural_substrate_t* substrate;
    epistemic_substrate_config_t config;
    epistemic_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

epistemic_substrate_config_t epistemic_substrate_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    epistemic_substrate_bridge_heartbeat("epistemic_su_epistemic_substrate_", 0.0f);


    epistemic_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

epistemic_substrate_bridge_t* epistemic_substrate_bridge_create(void* epistemic, neural_substrate_t* substrate, const epistemic_substrate_config_t* config) {
    if (!substrate) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;

    }
    /* Phase 8: Heartbeat at operation start */
    epistemic_substrate_bridge_heartbeat("epistemic_su_create", 0.0f);


    epistemic_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(epistemic_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    bridge->epistemic = epistemic;
    bridge->substrate = substrate;
    bridge->config = config ? *config : epistemic_substrate_default_config();
    bridge->effects.evidence_integration = 1.0f;
    bridge->effects.belief_updating = 1.0f;
    bridge->effects.certainty_calibration = 1.0f;
    bridge->effects.source_evaluation = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void epistemic_substrate_bridge_destroy(epistemic_substrate_bridge_t* bridge) {
    if (!bridge) return;
    /* Phase 8: Heartbeat at operation start */
    epistemic_substrate_bridge_heartbeat("epistemic_su_destroy", 0.0f);


    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int epistemic_substrate_bridge_update(epistemic_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    /* Phase 8: Heartbeat at operation start */
    epistemic_substrate_bridge_heartbeat("epistemic_su_update", 0.0f);


    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.evidence_integration = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.source_evaluation = nimcp_clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.belief_updating = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.certainty_calibration = nimcp_clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.evidence_integration + bridge->effects.belief_updating +
                                        bridge->effects.certainty_calibration + bridge->effects.source_evaluation) / 4.0f;
    bridge->update_count++;
    return 0;
}

int epistemic_substrate_bridge_get_effects(const epistemic_substrate_bridge_t* bridge, epistemic_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    /* Phase 8: Heartbeat at operation start */
    epistemic_substrate_bridge_heartbeat("epistemic_su_get_effects", 0.0f);


    return 0;
}

int epistemic_substrate_bridge_apply_effects(epistemic_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->bio_async_connected || !bridge->ctx) return 0;

    /* Phase 8: Heartbeat at operation start */
    epistemic_substrate_bridge_heartbeat("epistemic_su_apply_effects", 0.0f);


    substrate_metabolic_state_t metabolic;
    float atp_level = 1.0f, fatigue_level = 0.0f;
    if (bridge->substrate && substrate_get_metabolic_state(bridge->substrate, &metabolic) == 0) {
        atp_level = metabolic.atp_level;
        fatigue_level = 1.0f - metabolic.metabolic_capacity;
    }

    bio_msg_substrate_modulation_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION, BIO_MODULE_SUBSTRATE_EPISTEMIC, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_ACETYLCHOLINE;  /* Epistemic uses acetylcholine for evidence assessment */
    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_EPISTEMIC;
    msg.processing_capacity = bridge->effects.evidence_integration;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.evidence_integration;
    msg.effect_values[1] = bridge->effects.belief_updating;
    msg.effect_values[2] = bridge->effects.certainty_calibration;
    msg.effect_values[3] = bridge->effects.source_evaluation;
    msg.atp_level = atp_level;
    msg.fatigue_level = fatigue_level;
    msg.update_count = bridge->update_count;
    msg.critical_low = (atp_level < bridge->config.min_capacity);
    bio_router_broadcast(bridge->ctx, &msg, sizeof(msg));

    float delta = bridge->effects.overall_capacity - bridge->prev_overall_capacity;
    if (delta < -0.1f || delta > 0.1f) {
        bio_msg_substrate_capacity_update_t update_msg;
        memset(&update_msg, 0, sizeof(update_msg));
        bio_msg_init_header(&update_msg.header, BIO_MSG_SUBSTRATE_CAPACITY_UPDATE, BIO_MODULE_SUBSTRATE_EPISTEMIC, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_EPISTEMIC;
        update_msg.old_capacity = bridge->prev_overall_capacity;
        update_msg.new_capacity = bridge->effects.overall_capacity;
        update_msg.delta = delta;
        update_msg.significant_change = true;
        bio_router_broadcast(bridge->ctx, &update_msg, sizeof(update_msg));
    }
    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int epistemic_substrate_bridge_register_bio_async(epistemic_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    epistemic_substrate_bridge_heartbeat("epistemic_su_register_bio_async", 0.0f);


    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }
    if (!router) return 0;
    bio_module_info_t info = { .module_id = BIO_MODULE_SUBSTRATE_EPISTEMIC, .module_name = "epistemic_substrate_bridge", .inbox_capacity = 16, .user_data = bridge };
    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) bridge->bio_async_connected = true;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int epistemic_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    epistemic_substrate_bridge_heartbeat("epistemic_su_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Epistemic_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                epistemic_substrate_bridge_heartbeat("epistemic_su_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Epistemic_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Epistemic_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
