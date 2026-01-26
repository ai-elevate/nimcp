/**
 * @file nimcp_theory_of_mind_substrate_bridge.c
 * @brief Theory of Mind-Neural Substrate Bridge Implementation
 *
 * WHAT: Links ToM to metabolic state
 * WHY: Mentalizing requires sustained prefrontal and TPJ resources
 * HOW: Monitors ATP/fatigue; modulates mentalizing, perspective-taking
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/theory_of_mind/nimcp_theory_of_mind_substrate_bridge.h"
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

/** Global health agent for theory_of_mind_substrate_bridge module */
static nimcp_health_agent_t* g_theory_of_mind_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for theory_of_mind_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void theory_of_mind_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_theory_of_mind_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from theory_of_mind_substrate_bridge module */
static inline void theory_of_mind_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_theory_of_mind_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_theory_of_mind_substrate_bridge_health_agent, operation, progress);
    }
}


struct tom_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* tom;
    neural_substrate_t* substrate;
    tom_substrate_config_t config;
    tom_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

tom_substrate_config_t tom_substrate_default_config(void) {
    tom_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

tom_substrate_bridge_t* tom_substrate_bridge_create(void* tom, neural_substrate_t* substrate, const tom_substrate_config_t* config) {
    if (!substrate) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;

    }

    tom_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(tom_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge->tom = tom;
    bridge->substrate = substrate;
    bridge->config = config ? *config : tom_substrate_default_config();

    bridge->effects.mentalizing_capacity = 1.0f;
    bridge->effects.perspective_taking = 1.0f;
    bridge->effects.recursive_depth = 1.0f;
    bridge->effects.false_belief_reasoning = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    return bridge;
}

void tom_substrate_bridge_destroy(tom_substrate_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int tom_substrate_bridge_update(tom_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        /* Mentalizing capacity requires stable prefrontal resources */
        bridge->effects.mentalizing_capacity = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        /* Recursive depth is exponentially sensitive to ATP (each level compounds) */
        bridge->effects.recursive_depth = nimcp_clamp_f(powf(atp, 0.8f) * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        /* Perspective-taking degrades with fatigue */
        bridge->effects.perspective_taking = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        /* False belief reasoning is particularly vulnerable to fatigue */
        bridge->effects.false_belief_reasoning = nimcp_clamp_f(metabolic_cap * 0.8f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.mentalizing_capacity +
                                        bridge->effects.perspective_taking +
                                        bridge->effects.recursive_depth +
                                        bridge->effects.false_belief_reasoning) / 4.0f;

    bridge->update_count++;
    return 0;
}

int tom_substrate_bridge_get_effects(const tom_substrate_bridge_t* bridge, tom_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int tom_substrate_bridge_apply_effects(tom_substrate_bridge_t* bridge) {
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
                        BIO_MODULE_SUBSTRATE_TOM, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_ACETYLCHOLINE;  /* ToM uses acetylcholine for attention */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_TOM;
    msg.processing_capacity = bridge->effects.mentalizing_capacity;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.mentalizing_capacity;
    msg.effect_values[1] = bridge->effects.perspective_taking;
    msg.effect_values[2] = bridge->effects.recursive_depth;
    msg.effect_values[3] = bridge->effects.false_belief_reasoning;
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
                            BIO_MODULE_SUBSTRATE_TOM, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_TOM;
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
                            BIO_MODULE_SUBSTRATE_TOM, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_TOM;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int tom_substrate_bridge_register_bio_async(tom_substrate_bridge_t* bridge, bio_router_t* router) {
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
        .module_id = BIO_MODULE_SUBSTRATE_TOM,
        .module_name = "tom_substrate_bridge",
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

int tom_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Theory_Of_Mind_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Theory_Of_Mind_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Theory_Of_Mind_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
