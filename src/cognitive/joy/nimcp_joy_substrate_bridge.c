/**
 * @file nimcp_joy_substrate_bridge.c
 * @brief Joy-Neural Substrate Bridge Implementation
 *
 * Uses shared metabolic modulation utilities from nimcp_metabolic_modulation.h
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/joy/nimcp_joy_substrate_bridge.h"
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

/** Global health agent for joy_substrate_bridge module */
static nimcp_health_agent_t* g_joy_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for joy_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void joy_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_joy_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from joy_substrate_bridge module */
static inline void joy_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_joy_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_joy_substrate_bridge_health_agent, operation, progress);
    }
}


struct joy_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* joy;
    neural_substrate_t* substrate;
    joy_substrate_config_t config;
    joy_substrate_effects_t effects;
    metabolic_modulation_config_t metabolic_config;  /* Shared metabolic config */
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
};

joy_substrate_config_t joy_substrate_default_config(void) {
    joy_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

joy_substrate_bridge_t* joy_substrate_bridge_create(void* joy, neural_substrate_t* substrate, const joy_substrate_config_t* config) {
    if (!substrate) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;

    }
    joy_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(joy_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }
    bridge->joy = joy;
    bridge->substrate = substrate;
    bridge->config = config ? *config : joy_substrate_default_config();

    /* Initialize shared metabolic config with joy-specific multipliers */
    metabolic_effect_multipliers_t joy_mult = {
        .atp_primary_mult = 1.0f,
        .atp_secondary_mult = 1.1f,
        .fatigue_primary_mult = 1.0f,
        .fatigue_secondary_mult = 0.95f  /* Joy uses 0.95 instead of standard 0.9 */
    };
    bridge->metabolic_config = metabolic_config_from_fields(
        bridge->config.enable_atp_modulation,
        bridge->config.enable_fatigue_modulation,
        bridge->config.enable_bio_async,
        bridge->config.atp_sensitivity,
        bridge->config.fatigue_sensitivity,
        bridge->config.min_capacity,
        &joy_mult
    );

    /* Initialize effects to full capacity */
    bridge->effects.hedonic_capacity = 1.0f;
    bridge->effects.joy_intensity = 1.0f;
    bridge->effects.savoring_ability = 1.0f;
    bridge->effects.positive_anticipation = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    return bridge;
}

void joy_substrate_bridge_destroy(joy_substrate_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int joy_substrate_bridge_update(joy_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

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
        /* Map generic effects to joy-specific effect names */
        /* ATP enables hedonic capacity and joy intensity */
        bridge->effects.hedonic_capacity = generic_effects.primary_atp;
        bridge->effects.joy_intensity = generic_effects.secondary_atp;
        /* Low fatigue enables savoring and anticipation */
        bridge->effects.savoring_ability = generic_effects.primary_fatigue;
        bridge->effects.positive_anticipation = generic_effects.secondary_fatigue;
        bridge->effects.overall_capacity = generic_effects.overall_capacity;
    }

    bridge->update_count++;
    return 0;
}

int joy_substrate_bridge_get_effects(const joy_substrate_bridge_t* bridge, joy_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    return 0;
}

int joy_substrate_bridge_apply_effects(joy_substrate_bridge_t* bridge) {
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
                        BIO_MODULE_SUBSTRATE_JOY, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_DOPAMINE;  /* Joy uses dopamine channel */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_JOY;
    msg.processing_capacity = bridge->effects.hedonic_capacity;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.hedonic_capacity;
    msg.effect_values[1] = bridge->effects.joy_intensity;
    msg.effect_values[2] = bridge->effects.savoring_ability;
    msg.effect_values[3] = bridge->effects.positive_anticipation;
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
                            BIO_MODULE_SUBSTRATE_JOY, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_JOY;
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
                            BIO_MODULE_SUBSTRATE_JOY, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_JOY;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int joy_substrate_bridge_register_bio_async(joy_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;

    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }

    if (!router) {
        return 0;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_SUBSTRATE_JOY,
        .module_name = "joy_substrate_bridge",
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

int joy_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Joy_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Joy_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Joy_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
