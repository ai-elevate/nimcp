/**
 * @file nimcp_brain_immune_substrate_bridge.c
 * @brief Brain Immune-Neural Substrate Bridge Implementation
 *
 * WHAT: Links brain immune coordination to metabolic state
 * WHY: Immune responses require substantial energy
 * HOW: Monitors ATP/fatigue; modulates response, antibody production, memory
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/immune/nimcp_brain_immune_substrate_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "api/nimcp_api_exception.h"
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

/** Global health agent for brain_immune_substrate_bridge module */
static nimcp_health_agent_t* g_brain_immune_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for brain_immune_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void brain_immune_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_brain_immune_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from brain_immune_substrate_bridge module */
static inline void brain_immune_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_brain_immune_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_brain_immune_substrate_bridge_health_agent, operation, progress);
    }
}


struct brain_immune_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* brain_immune;
    neural_substrate_t* substrate;
    brain_immune_substrate_config_t config;
    brain_immune_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

brain_immune_substrate_config_t brain_immune_substrate_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    brain_immune_substrate_bridge_heartbeat("brain_immune_brain_immune_substra", 0.0f);


    brain_immune_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.25f  /* Baseline immune function maintained */
    };
    return cfg;
}

brain_immune_substrate_bridge_t* brain_immune_substrate_bridge_create(void* brain_immune, neural_substrate_t* substrate, const brain_immune_substrate_config_t* config) {
    if (!substrate) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;

    }

    /* Phase 8: Heartbeat at operation start */
    brain_immune_substrate_bridge_heartbeat("brain_immune_create", 0.0f);


    brain_immune_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(brain_immune_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge->brain_immune = brain_immune;
    bridge->substrate = substrate;
    bridge->config = config ? *config : brain_immune_substrate_default_config();

    bridge->effects.response_strength = 1.0f;
    bridge->effects.antibody_production = 1.0f;
    bridge->effects.memory_formation = 1.0f;
    bridge->effects.cytokine_regulation = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void brain_immune_substrate_bridge_destroy(brain_immune_substrate_bridge_t* bridge) {
    if (!bridge) return;
    /* Phase 8: Heartbeat at operation start */
    brain_immune_substrate_bridge_heartbeat("brain_immune_destroy", 0.0f);


    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int brain_immune_substrate_bridge_update(brain_immune_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_substrate_bridge_heartbeat("brain_immune_update", 0.0f);


    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Response strength scales with ATP (immune activation is costly) */
        bridge->effects.response_strength = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Antibody production is ATP-intensive */
        bridge->effects.antibody_production = nimcp_clamp_f(atp * 0.9f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Memory formation requires sustained resources */
        bridge->effects.memory_formation = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* Cytokine regulation is vulnerable to metabolic stress */
        bridge->effects.cytokine_regulation = nimcp_clamp_f(metabolic_cap * 0.85f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.response_strength +
                                        bridge->effects.antibody_production +
                                        bridge->effects.memory_formation +
                                        bridge->effects.cytokine_regulation) / 4.0f;

    bridge->update_count++;
    return 0;
}

int brain_immune_substrate_bridge_get_effects(const brain_immune_substrate_bridge_t* bridge, brain_immune_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    /* Phase 8: Heartbeat at operation start */
    brain_immune_substrate_bridge_heartbeat("brain_immune_get_effects", 0.0f);


    return 0;
}

int brain_immune_substrate_bridge_apply_effects(brain_immune_substrate_bridge_t* bridge) {
    if (!bridge) return -1;

    if (!bridge->bio_async_connected || !bridge->ctx) {
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    brain_immune_substrate_bridge_heartbeat("brain_immune_apply_effects", 0.0f);


    substrate_metabolic_state_t metabolic;
    float atp_level = 1.0f, fatigue_level = 0.0f;
    if (bridge->substrate && substrate_get_metabolic_state(bridge->substrate, &metabolic) == 0) {
        atp_level = metabolic.atp_level;
        fatigue_level = 1.0f - metabolic.metabolic_capacity;
    }

    bio_msg_substrate_modulation_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION,
                        BIO_MODULE_SUBSTRATE_BRAIN_IMMUNE, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_NOREPINEPHRINE;  /* Immune uses norepinephrine for alerts */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_BRAIN_IMMUNE;
    msg.processing_capacity = bridge->effects.response_strength;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.response_strength;
    msg.effect_values[1] = bridge->effects.antibody_production;
    msg.effect_values[2] = bridge->effects.memory_formation;
    msg.effect_values[3] = bridge->effects.cytokine_regulation;
    msg.atp_level = atp_level;
    msg.fatigue_level = fatigue_level;
    msg.update_count = bridge->update_count;
    msg.critical_low = (atp_level < bridge->config.min_capacity);

    bio_router_broadcast(bridge->ctx, &msg, sizeof(msg));

    float delta = bridge->effects.overall_capacity - bridge->prev_overall_capacity;
    if (delta < -0.1f || delta > 0.1f) {
        bio_msg_substrate_capacity_update_t update_msg;
        memset(&update_msg, 0, sizeof(update_msg));
        bio_msg_init_header(&update_msg.header, BIO_MSG_SUBSTRATE_CAPACITY_UPDATE,
                            BIO_MODULE_SUBSTRATE_BRAIN_IMMUNE, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_BRAIN_IMMUNE;
        update_msg.old_capacity = bridge->prev_overall_capacity;
        update_msg.new_capacity = bridge->effects.overall_capacity;
        update_msg.delta = delta;
        update_msg.significant_change = true;
        bio_router_broadcast(bridge->ctx, &update_msg, sizeof(update_msg));
    }

    if (msg.critical_low) {
        bio_msg_substrate_atp_critical_t alert;
        memset(&alert, 0, sizeof(alert));
        bio_msg_init_header(&alert.header, BIO_MSG_SUBSTRATE_ATP_CRITICAL,
                            BIO_MODULE_SUBSTRATE_BRAIN_IMMUNE, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_BRAIN_IMMUNE;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int brain_immune_substrate_bridge_register_bio_async(brain_immune_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    brain_immune_substrate_bridge_heartbeat("brain_immune_register_bio_async", 0.0f);


    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }

    if (!router) {
        return 0;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_SUBSTRATE_BRAIN_IMMUNE,
        .module_name = "brain_immune_substrate_bridge",
        .inbox_capacity = 16,
        .user_data = bridge
    };

    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) {
        bridge->bio_async_connected = true;
    }
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * @brief Query self-knowledge from knowledge graph
 *
 * WHAT: Query KG for module self-awareness information
 * WHY:  Enable introspective self-knowledge about brain immune substrate bridge
 * HOW:  Look up entity and relations in KG
 *
 * @param kg Knowledge graph reader handle
 * @return 1 if self-knowledge found, 0 otherwise
 */
int brain_immune_substrate_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    brain_immune_substrate_bridge_heartbeat("brain_immune_brain_immune_substra", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Brain_Immune_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                brain_immune_substrate_bridge_heartbeat("brain_immune_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Brain immune substrate bridge self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Brain_Immune_Substrate_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Brain_Immune_Substrate_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
