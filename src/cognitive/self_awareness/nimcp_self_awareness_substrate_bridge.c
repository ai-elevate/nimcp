/**
 * @file nimcp_self_awareness_substrate_bridge.c
 * @brief Self-Awareness-Neural Substrate Bridge Implementation
 *
 * WHAT: Links self-awareness to metabolic state
 * WHY: Metacognition requires medial prefrontal and cingulate resources
 * HOW: Monitors ATP/fatigue; modulates self-reflection, introspection, metacognition
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/self_awareness/nimcp_self_awareness_substrate_bridge.h"
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

/** Global health agent for self_awareness_substrate_bridge module */
static nimcp_health_agent_t* g_self_awareness_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for self_awareness_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void self_awareness_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_self_awareness_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from self_awareness_substrate_bridge module */
static inline void self_awareness_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_self_awareness_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_self_awareness_substrate_bridge_health_agent, operation, progress);
    }
}

#define LOG_MODULE "SELF_AWARENESS_SUBSTRATE_BRIDGE"


struct self_awareness_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* self_awareness;
    neural_substrate_t* substrate;
    self_awareness_substrate_config_t config;
    self_awareness_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

self_awareness_substrate_config_t self_awareness_substrate_default_config(void) {
    self_awareness_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

self_awareness_substrate_bridge_t* self_awareness_substrate_bridge_create(void* self_awareness, neural_substrate_t* substrate, const self_awareness_substrate_config_t* config) {
    if (!substrate) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;

    }

    self_awareness_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(self_awareness_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge->self_awareness = self_awareness;
    bridge->substrate = substrate;
    bridge->config = config ? *config : self_awareness_substrate_default_config();

    bridge->effects.introspection_depth = 1.0f;
    bridge->effects.metacognitive_accuracy = 1.0f;
    bridge->effects.self_monitoring = 1.0f;
    bridge->effects.identity_coherence = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    NIMCP_LOGGING_INFO("Created %s bridge", "self_awareness_substrate");
    return bridge;
}

void self_awareness_substrate_bridge_destroy(self_awareness_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int self_awareness_substrate_bridge_update(self_awareness_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        bridge->effects.introspection_depth = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.metacognitive_accuracy = nimcp_clamp_f(atp * 1.05f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.self_monitoring = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.identity_coherence = nimcp_clamp_f(metabolic_cap * 0.95f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.introspection_depth +
                                        bridge->effects.metacognitive_accuracy +
                                        bridge->effects.self_monitoring +
                                        bridge->effects.identity_coherence) / 4.0f;

    bridge->update_count++;
    return 0;
}

int self_awareness_substrate_bridge_get_effects(const self_awareness_substrate_bridge_t* bridge, self_awareness_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int self_awareness_substrate_bridge_apply_effects(self_awareness_substrate_bridge_t* bridge) {
    if (!bridge) return -1;

    if (!bridge->bio_async_connected || !bridge->ctx) {
        return 0;
    }

    substrate_metabolic_state_t metabolic;
    float atp_level = 1.0f, fatigue_level = 0.0f;
    if (bridge->substrate && substrate_get_metabolic_state(bridge->substrate, &metabolic) == 0) {
        atp_level = metabolic.atp_level;
        fatigue_level = 1.0f - metabolic.metabolic_capacity;
    }

    bio_msg_substrate_modulation_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION,
                        BIO_MODULE_SUBSTRATE_SELF_AWARENESS, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_ACETYLCHOLINE;  /* Self-awareness uses acetylcholine */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_SELF_AWARENESS;
    msg.processing_capacity = bridge->effects.introspection_depth;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.introspection_depth;
    msg.effect_values[1] = bridge->effects.metacognitive_accuracy;
    msg.effect_values[2] = bridge->effects.self_monitoring;
    msg.effect_values[3] = bridge->effects.identity_coherence;
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
                            BIO_MODULE_SUBSTRATE_SELF_AWARENESS, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_SELF_AWARENESS;
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
                            BIO_MODULE_SUBSTRATE_SELF_AWARENESS, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_SELF_AWARENESS;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int self_awareness_substrate_bridge_register_bio_async(self_awareness_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;

    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }

    if (!router) {
        bridge->router = NULL;
        return 0;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_SUBSTRATE_SELF_AWARENESS,
        .module_name = "self_awareness_substrate_bridge",
        .inbox_capacity = 16,
        .user_data = bridge
    };

    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) {
        bridge->bio_async_connected = true;
    }
    bridge->router = router;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int self_awareness_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Self_Awareness_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Self_Awareness_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Self_Awareness_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
