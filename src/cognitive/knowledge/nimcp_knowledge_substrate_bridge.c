/**
 * @file nimcp_knowledge_substrate_bridge.c
 * @brief Knowledge-Neural Substrate Bridge Implementation
 *
 * WHAT: Links knowledge/semantic memory to metabolic state
 * WHY: Semantic retrieval requires temporal-parietal resources
 * HOW: Monitors ATP/fatigue; modulates retrieval fluency, accuracy, integration
 *
 * Uses shared metabolic modulation utilities from nimcp_metabolic_modulation.h
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/knowledge/nimcp_knowledge_substrate_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
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

/** Global health agent for knowledge_substrate_bridge module */
static nimcp_health_agent_t* g_knowledge_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for knowledge_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void knowledge_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_knowledge_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from knowledge_substrate_bridge module */
static inline void knowledge_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_knowledge_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_knowledge_substrate_bridge_health_agent, operation, progress);
    }
}

#define LOG_MODULE "KNOWLEDGE_SUBSTRATE_BRIDGE"


struct knowledge_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* knowledge;
    neural_substrate_t* substrate;
    knowledge_substrate_config_t config;
    knowledge_substrate_effects_t effects;
    metabolic_modulation_config_t metabolic_config;  /* Shared metabolic config */
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

knowledge_substrate_config_t knowledge_substrate_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    knowledge_substrate_bridge_heartbeat("knowledge_su_knowledge_substrate_", 0.0f);


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
    if (!substrate) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "knowledge_substrate_bridge_create: substrate is NULL");
        return NULL;
    }

    /* Phase 8: Heartbeat at operation start */
    knowledge_substrate_bridge_heartbeat("knowledge_su_create", 0.0f);


    knowledge_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(knowledge_substrate_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "knowledge_substrate_bridge_create: failed to allocate bridge");
        return NULL;
    }

    bridge->knowledge = knowledge;
    bridge->substrate = substrate;
    bridge->config = config ? *config : knowledge_substrate_default_config();

    /* Initialize shared metabolic config with knowledge-specific multipliers */
    metabolic_effect_multipliers_t knowledge_mult = {
        .atp_primary_mult = 1.0f,
        .atp_secondary_mult = 1.1f,
        .fatigue_primary_mult = 1.0f,
        .fatigue_secondary_mult = 0.95f  /* Knowledge uses 0.95 for consolidation */
    };
    bridge->metabolic_config = metabolic_config_from_fields(
        bridge->config.enable_atp_modulation,
        bridge->config.enable_fatigue_modulation,
        bridge->config.enable_bio_async,
        bridge->config.atp_sensitivity,
        bridge->config.fatigue_sensitivity,
        bridge->config.min_capacity,
        &knowledge_mult
    );

    /* Initialize effects to full capacity */
    bridge->effects.retrieval_speed = 1.0f;
    bridge->effects.retrieval_accuracy = 1.0f;
    bridge->effects.consolidation_rate = 1.0f;
    bridge->effects.association_strength = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    NIMCP_LOGGING_INFO("Created %s bridge", "knowledge_substrate");
    return bridge;
}

void knowledge_substrate_bridge_destroy(knowledge_substrate_bridge_t* bridge) {
    if (!bridge) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "knowledge_substrate");
    /* Phase 8: Heartbeat at operation start */
    knowledge_substrate_bridge_heartbeat("knowledge_su_destroy", 0.0f);


    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int knowledge_substrate_bridge_update(knowledge_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    /* Phase 8: Heartbeat at operation start */
    knowledge_substrate_bridge_heartbeat("knowledge_su_update", 0.0f);


    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    /* Use shared metabolic computation */
    metabolic_input_t input = {
        .atp_level = metabolic.atp_level,
        .metabolic_capacity = metabolic.metabolic_capacity
    };
    metabolic_effects_t generic_effects;
    metabolic_effects_init_full(&generic_effects);

    if (metabolic_compute_effects(&input, &bridge->metabolic_config, &generic_effects) == 0) {
        /* Map generic effects to knowledge-specific effect names */
        /* Note: Knowledge swaps ATP/fatigue primary roles from standard pattern */
        bridge->effects.retrieval_accuracy = generic_effects.primary_atp;
        bridge->effects.association_strength = generic_effects.secondary_atp;
        bridge->effects.retrieval_speed = generic_effects.primary_fatigue;
        bridge->effects.consolidation_rate = generic_effects.secondary_fatigue;
        bridge->effects.overall_capacity = generic_effects.overall_capacity;
    }

    bridge->update_count++;
    return 0;
}

int knowledge_substrate_bridge_get_effects(const knowledge_substrate_bridge_t* bridge, knowledge_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    /* Phase 8: Heartbeat at operation start */
    knowledge_substrate_bridge_heartbeat("knowledge_su_get_effects", 0.0f);


    return 0;
}

int knowledge_substrate_bridge_apply_effects(knowledge_substrate_bridge_t* bridge) {
    if (!bridge) return -1;

    if (!bridge->bio_async_connected || !bridge->ctx) {
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    knowledge_substrate_bridge_heartbeat("knowledge_su_apply_effects", 0.0f);


    substrate_metabolic_state_t metabolic;
    float atp_level = 1.0f, fatigue_level = 0.0f;
    if (bridge->substrate && substrate_get_metabolic_state(bridge->substrate, &metabolic) == 0) {
        atp_level = metabolic.atp_level;
        fatigue_level = 1.0f - metabolic.metabolic_capacity;
    }

    bio_msg_substrate_modulation_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION,
                        BIO_MODULE_SUBSTRATE_KNOWLEDGE, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_ACETYLCHOLINE;  /* Knowledge uses acetylcholine */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_KNOWLEDGE;
    msg.processing_capacity = bridge->effects.retrieval_speed;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.retrieval_speed;
    msg.effect_values[1] = bridge->effects.retrieval_accuracy;
    msg.effect_values[2] = bridge->effects.consolidation_rate;
    msg.effect_values[3] = bridge->effects.association_strength;
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
                            BIO_MODULE_SUBSTRATE_KNOWLEDGE, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_KNOWLEDGE;
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
                            BIO_MODULE_SUBSTRATE_KNOWLEDGE, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_KNOWLEDGE;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int knowledge_substrate_bridge_register_bio_async(knowledge_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    knowledge_substrate_bridge_heartbeat("knowledge_su_register_bio_async", 0.0f);


    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }

    if (!router) {
        return 0;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_SUBSTRATE_KNOWLEDGE,
        .module_name = "knowledge_substrate_bridge",
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
 * @brief Query knowledge graph for self-knowledge about substrate knowledge bridge
 *
 * WHAT: Retrieves entity observations and relations for substrate-knowledge bridge
 * WHY: Enables self-aware introspection of module capabilities
 * HOW: Uses kg_reader to query JSONL knowledge graph
 *
 * @param kg Knowledge graph reader instance
 * @return 1 if self-knowledge found, 0 otherwise
 */
int knowledge_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    knowledge_substrate_bridge_heartbeat("knowledge_su_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Knowledge_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                knowledge_substrate_bridge_heartbeat("knowledge_su_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Knowledge_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Knowledge_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
