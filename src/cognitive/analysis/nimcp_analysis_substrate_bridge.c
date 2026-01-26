/**
 * @file nimcp_analysis_substrate_bridge.c
 * @brief Analysis-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/analysis/nimcp_analysis_substrate_bridge.h"
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

/** Global health agent for analysis_substrate_bridge module */
static nimcp_health_agent_t* g_analysis_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for analysis_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void analysis_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_analysis_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from analysis_substrate_bridge module */
static inline void analysis_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_analysis_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_analysis_substrate_bridge_health_agent, operation, progress);
    }
}


struct analysis_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* analysis;
    neural_substrate_t* substrate;
    analysis_substrate_config_t config;
    analysis_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

analysis_substrate_config_t analysis_substrate_default_config(void) {
    analysis_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

analysis_substrate_bridge_t* analysis_substrate_bridge_create(void* analysis, neural_substrate_t* substrate, const analysis_substrate_config_t* config) {
    if (!substrate) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "analysis_substrate_bridge_create: substrate is NULL");
        return NULL;
    }
    analysis_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(analysis_substrate_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "analysis_substrate_bridge_create: failed to allocate bridge");
        return NULL;
    }
    bridge->analysis = analysis;
    bridge->substrate = substrate;
    bridge->config = config ? *config : analysis_substrate_default_config();
    bridge->effects.analysis_depth = 1.0f;
    bridge->effects.precision_level = 1.0f;
    bridge->effects.processing_speed = 1.0f;
    bridge->effects.decomposition_ability = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void analysis_substrate_bridge_destroy(analysis_substrate_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int analysis_substrate_bridge_update(analysis_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "analysis_substrate_bridge_update: bridge or substrate is NULL");
        return -1;
    }
    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "analysis_substrate_bridge_update: failed to get metabolic state");
        return -1;
    }
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables deep analysis and precision */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.analysis_depth = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.precision_level = nimcp_clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables processing speed and decomposition */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.processing_speed = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.decomposition_ability = nimcp_clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.analysis_depth + bridge->effects.precision_level +
                                        bridge->effects.processing_speed + bridge->effects.decomposition_ability) / 4.0f;
    bridge->update_count++;
    return 0;
}

int analysis_substrate_bridge_get_effects(const analysis_substrate_bridge_t* bridge, analysis_substrate_effects_t* effects) {
    if (!bridge || !effects) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "analysis_substrate_bridge_get_effects: bridge or effects is NULL");
        return -1;
    }
    *effects = bridge->effects;
    return 0;
}

int analysis_substrate_bridge_apply_effects(analysis_substrate_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "analysis_substrate_bridge_apply_effects: bridge is NULL");
        return -1;
    }
    if (!bridge->bio_async_connected || !bridge->ctx) return 0;

    substrate_metabolic_state_t metabolic;
    float atp_level = 1.0f, fatigue_level = 0.0f;
    if (bridge->substrate && substrate_get_metabolic_state(bridge->substrate, &metabolic) == 0) {
        atp_level = metabolic.atp_level;
        fatigue_level = 1.0f - metabolic.metabolic_capacity;
    }

    bio_msg_substrate_modulation_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION, BIO_MODULE_SUBSTRATE_ANALYSIS, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_ACETYLCHOLINE;  /* Analysis uses acetylcholine for focused processing */
    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_ANALYSIS;
    msg.processing_capacity = bridge->effects.analysis_depth;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.analysis_depth;
    msg.effect_values[1] = bridge->effects.precision_level;
    msg.effect_values[2] = bridge->effects.processing_speed;
    msg.effect_values[3] = bridge->effects.decomposition_ability;
    msg.atp_level = atp_level;
    msg.fatigue_level = fatigue_level;
    msg.update_count = bridge->update_count;
    msg.critical_low = (atp_level < bridge->config.min_capacity);
    bio_router_broadcast(bridge->ctx, &msg, sizeof(msg));

    float delta = bridge->effects.overall_capacity - bridge->prev_overall_capacity;
    if (delta < -0.1f || delta > 0.1f) {
        bio_msg_substrate_capacity_update_t update_msg;
        memset(&update_msg, 0, sizeof(update_msg));
        bio_msg_init_header(&update_msg.header, BIO_MSG_SUBSTRATE_CAPACITY_UPDATE, BIO_MODULE_SUBSTRATE_ANALYSIS, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_ANALYSIS;
        update_msg.old_capacity = bridge->prev_overall_capacity;
        update_msg.new_capacity = bridge->effects.overall_capacity;
        update_msg.delta = delta;
        update_msg.significant_change = true;
        bio_router_broadcast(bridge->ctx, &update_msg, sizeof(update_msg));
    }
    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int analysis_substrate_bridge_register_bio_async(analysis_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "analysis_substrate_bridge_register_bio_async: bridge is NULL");
        return -1;
    }
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }
    if (!router) return 0;
    bio_module_info_t info = { .module_id = BIO_MODULE_SUBSTRATE_ANALYSIS, .module_name = "analysis_substrate_bridge", .inbox_capacity = 16, .user_data = bridge };
    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) bridge->bio_async_connected = true;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Analysis Substrate Bridge self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int analysis_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Analysis_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Log observation */
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Analysis_Substrate_Bridge");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Analysis_Substrate_Bridge");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
