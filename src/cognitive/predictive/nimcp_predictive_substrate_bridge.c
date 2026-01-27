/**
 * @file nimcp_predictive_substrate_bridge.c
 * @brief Predictive Coding-Neural Substrate Bridge Implementation
 *
 * WHAT: Links predictive processing to metabolic state
 * WHY: Prediction requires hierarchical cortical resources
 * HOW: Monitors ATP/fatigue; modulates prediction accuracy, precision, update rate
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/predictive/nimcp_predictive_substrate_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "utils/exception/nimcp_exception_immune.h"
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

/** Global health agent for predictive_substrate_bridge module */
static nimcp_health_agent_t* g_predictive_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for predictive_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void predictive_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_predictive_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from predictive_substrate_bridge module */
static inline void predictive_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_predictive_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_predictive_substrate_bridge_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat from predictive_substrate_bridge module (instance-level) */
static inline void predictive_substrate_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_predictive_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_predictive_substrate_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_predictive_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}

#define LOG_MODULE "PREDICTIVE_SUBSTRATE_BRIDGE"


struct predictive_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* predictive;
    neural_substrate_t* substrate;
    predictive_substrate_config_t config;
    predictive_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;

    /* Phase 8: Instance health agent (B24 upgrade) */
    nimcp_health_agent_t* health_agent;
};

predictive_substrate_config_t predictive_substrate_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    predictive_substrate_bridge_heartbeat("predictive_s_predictive_substrate", 0.0f);


    predictive_substrate_config_t cfg = {
        .enable_atp_modulation = true,
        .enable_fatigue_modulation = true,
        .enable_bio_async = false,
        .atp_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .min_capacity = 0.2f
    };
    return cfg;
}

predictive_substrate_bridge_t* predictive_substrate_bridge_create(void* predictive, neural_substrate_t* substrate, const predictive_substrate_config_t* config) {
    if (!substrate) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;

    }

    /* Phase 8: Heartbeat at operation start */
    predictive_substrate_bridge_heartbeat("predictive_s_create", 0.0f);


    predictive_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(predictive_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;

    }

    bridge->predictive = predictive;
    bridge->substrate = substrate;
    bridge->config = config ? *config : predictive_substrate_default_config();

    bridge->effects.prediction_precision = 1.0f;
    bridge->effects.error_sensitivity = 1.0f;
    bridge->effects.model_update_rate = 1.0f;
    bridge->effects.hierarchical_depth = 1.0f;
    bridge->effects.overall_capacity = 1.0f;

    NIMCP_LOGGING_INFO("Created %s bridge", "predictive_substrate");
    return bridge;
}

void predictive_substrate_bridge_destroy(predictive_substrate_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    predictive_substrate_bridge_heartbeat("predictive_s_destroy", 0.0f);


    if (bridge) nimcp_free(bridge);
}

int predictive_substrate_bridge_update(predictive_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;

    /* Phase 8: Heartbeat at operation start */
    predictive_substrate_bridge_heartbeat("predictive_s_update", 0.0f);


    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;

    float atp = metabolic.atp_level;
    float metabolic_cap = metabolic.metabolic_capacity;
    float min_cap = bridge->config.min_capacity;

    if (bridge->config.enable_atp_modulation) {
        bridge->effects.prediction_precision = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.hierarchical_depth = nimcp_clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }

    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.error_sensitivity = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.model_update_rate = nimcp_clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }

    bridge->effects.overall_capacity = (bridge->effects.prediction_precision +
                                        bridge->effects.error_sensitivity +
                                        bridge->effects.model_update_rate +
                                        bridge->effects.hierarchical_depth) / 4.0f;

    bridge->update_count++;
    return 0;
}

int predictive_substrate_bridge_get_effects(const predictive_substrate_bridge_t* bridge, predictive_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    /* Phase 8: Heartbeat at operation start */
    predictive_substrate_bridge_heartbeat("predictive_s_get_effects", 0.0f);


    return 0;
}

int predictive_substrate_bridge_apply_effects(predictive_substrate_bridge_t* bridge) {
    if (!bridge) return -1;

    if (!bridge->bio_async_connected || !bridge->ctx) {
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    predictive_substrate_bridge_heartbeat("predictive_s_apply_effects", 0.0f);


    substrate_metabolic_state_t metabolic;
    float atp_level = 1.0f, fatigue_level = 0.0f;
    if (bridge->substrate && substrate_get_metabolic_state(bridge->substrate, &metabolic) == 0) {
        atp_level = metabolic.atp_level;
        fatigue_level = 1.0f - metabolic.metabolic_capacity;
    }

    bio_msg_substrate_modulation_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION,
                        BIO_MODULE_SUBSTRATE_PREDICTIVE, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_ACETYLCHOLINE;  /* Predictive uses acetylcholine for precision */

    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_PREDICTIVE;
    msg.processing_capacity = bridge->effects.prediction_precision;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.prediction_precision;
    msg.effect_values[1] = bridge->effects.error_sensitivity;
    msg.effect_values[2] = bridge->effects.model_update_rate;
    msg.effect_values[3] = bridge->effects.hierarchical_depth;
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
                            BIO_MODULE_SUBSTRATE_PREDICTIVE, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_PREDICTIVE;
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
                            BIO_MODULE_SUBSTRATE_PREDICTIVE, 0, sizeof(alert));
        alert.header.channel = BIO_CHANNEL_NOREPINEPHRINE;
        alert.bridge_module_id = BIO_MODULE_SUBSTRATE_PREDICTIVE;
        alert.atp_level = atp_level;
        alert.threshold = bridge->config.min_capacity;
        alert.min_capacity = bridge->config.min_capacity;
        bio_router_broadcast(bridge->ctx, &alert, sizeof(alert));
    }

    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int predictive_substrate_bridge_register_bio_async(predictive_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;

    /* Phase 8: Heartbeat at operation start */
    predictive_substrate_bridge_heartbeat("predictive_s_register_bio_async", 0.0f);


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
        .module_id = BIO_MODULE_SUBSTRATE_PREDICTIVE,
        .module_name = "predictive_substrate_bridge",
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

int predictive_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    predictive_substrate_bridge_heartbeat("predictive_s_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Predictive_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                predictive_substrate_bridge_heartbeat("predictive_s_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Predictive_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Predictive_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}

//=============================================================================
// Instance Health Agent Setter (B24 Upgrade)
//=============================================================================

void predictive_substrate_bridge_set_instance_health_agent(
    predictive_substrate_bridge_t* bridge, nimcp_health_agent_t* agent)
{
    if (bridge) {
        bridge->health_agent = agent;
    }
}

//=============================================================================
// Training Hook Stubs (B24 Upgrade)
//=============================================================================

int predictive_substrate_bridge_training_begin(predictive_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    predictive_substrate_bridge_heartbeat_instance(bridge->health_agent, "predictive_substrate_bridge_training_begin", 0.0f);
    return 0;
}

int predictive_substrate_bridge_training_end(predictive_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    predictive_substrate_bridge_heartbeat_instance(bridge->health_agent, "predictive_substrate_bridge_training_end", 1.0f);
    return 0;
}

int predictive_substrate_bridge_training_step(predictive_substrate_bridge_t* bridge, float progress) {
    if (!bridge) return -1;
    predictive_substrate_bridge_heartbeat_instance(bridge->health_agent, "predictive_substrate_bridge_training_step", progress);
    return 0;
}
