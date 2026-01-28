/**
 * @file nimcp_emotion_recognition_substrate_bridge.c
 * @brief Emotion Recognition-Neural Substrate Bridge Implementation
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/emotion_recognition/nimcp_emotion_recognition_substrate_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "cognitive/common/nimcp_metabolic_modulation.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for emotion_recognition_substrate_bridge module */
static nimcp_health_agent_t* g_emotion_recognition_substrate_bridge_health_agent = NULL;

/**
 * @brief Set health agent for emotion_recognition_substrate_bridge heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
void emotion_recognition_substrate_bridge_set_health_agent(nimcp_health_agent_t* agent) {
    g_emotion_recognition_substrate_bridge_health_agent = agent;
}

/** @brief Send heartbeat from emotion_recognition_substrate_bridge module */
static inline void emotion_recognition_substrate_bridge_heartbeat(const char* operation, float progress) {
    if (g_emotion_recognition_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_emotion_recognition_substrate_bridge_health_agent, operation, progress);
    }
}

/** @brief Send heartbeat from emotion_recognition_substrate_bridge module (instance-level) */
static inline void emotion_recognition_substrate_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_emotion_recognition_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_emotion_recognition_substrate_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_emotion_recognition_substrate_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}

#define LOG_MODULE "EMOTION_RECOGNITION_SUBSTRATE_BRIDGE"


struct emotion_recognition_substrate_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* emotion_recognition;
    neural_substrate_t* substrate;
    emotion_recognition_substrate_config_t config;
    emotion_recognition_substrate_effects_t effects;
    bio_module_context_t ctx;
    bio_router_t* router;
    bool bio_async_connected;
    uint64_t update_count;
    float prev_overall_capacity;
    nimcp_health_agent_t* health_agent;  /**< Instance-level health agent (Phase 8) */
};

emotion_recognition_substrate_config_t emotion_recognition_substrate_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    emotion_recognition_substrate_bridge_heartbeat("emotion_reco_emotion_recognition_", 0.0f);


    emotion_recognition_substrate_config_t cfg = { .enable_atp_modulation = true, .enable_fatigue_modulation = true,
        .enable_bio_async = false, .atp_sensitivity = 1.0f, .fatigue_sensitivity = 1.0f, .min_capacity = 0.2f };
    return cfg;
}

emotion_recognition_substrate_bridge_t* emotion_recognition_substrate_bridge_create(void* emotion_recognition, neural_substrate_t* substrate, const emotion_recognition_substrate_config_t* config) {
    if (!substrate) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "substrate is NULL");

        return NULL;

    }
    /* Phase 8: Heartbeat at operation start */
    emotion_recognition_substrate_bridge_heartbeat("emotion_reco_create", 0.0f);


    emotion_recognition_substrate_bridge_t* bridge = nimcp_calloc(1, sizeof(emotion_recognition_substrate_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate bridge");

        return NULL;

    }
    bridge->emotion_recognition = emotion_recognition;
    bridge->substrate = substrate;
    bridge->config = config ? *config : emotion_recognition_substrate_default_config();
    bridge->effects.recognition_accuracy = 1.0f;
    bridge->effects.detection_speed = 1.0f;
    bridge->effects.subtle_sensitivity = 1.0f;
    bridge->effects.context_integration = 1.0f;
    bridge->effects.overall_capacity = 1.0f;
    NIMCP_LOGGING_INFO("Created %s bridge", "emotion_recognition_substrate");
    return bridge;
}

void emotion_recognition_substrate_bridge_destroy(emotion_recognition_substrate_bridge_t* bridge) {
    if (!bridge) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "emotion_recognition_substrate");
    /* Phase 8: Heartbeat at operation start */
    emotion_recognition_substrate_bridge_heartbeat("emotion_reco_destroy", 0.0f);


    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
    }
    nimcp_free(bridge);
}

int emotion_recognition_substrate_bridge_update(emotion_recognition_substrate_bridge_t* bridge) {
    if (!bridge || !bridge->substrate) return -1;
    /* Phase 8: Heartbeat at operation start */
    emotion_recognition_substrate_bridge_heartbeat("emotion_reco_update", 0.0f);


    substrate_metabolic_state_t metabolic;
    if (substrate_get_metabolic_state(bridge->substrate, &metabolic) != 0) return -1;
    float atp = metabolic.atp_level, metabolic_cap = metabolic.metabolic_capacity, min_cap = bridge->config.min_capacity;
    /* ATP enables recognition accuracy and subtle sensitivity */
    if (bridge->config.enable_atp_modulation) {
        bridge->effects.recognition_accuracy = nimcp_clamp_f(atp * bridge->config.atp_sensitivity, min_cap, 1.0f);
        bridge->effects.subtle_sensitivity = nimcp_clamp_f(atp * 1.1f * bridge->config.atp_sensitivity, min_cap, 1.0f);
    }
    /* Low fatigue enables detection speed and context integration */
    if (bridge->config.enable_fatigue_modulation) {
        bridge->effects.detection_speed = nimcp_clamp_f(metabolic_cap * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
        bridge->effects.context_integration = nimcp_clamp_f(metabolic_cap * 0.9f * bridge->config.fatigue_sensitivity, min_cap, 1.0f);
    }
    bridge->effects.overall_capacity = (bridge->effects.recognition_accuracy + bridge->effects.detection_speed +
                                        bridge->effects.subtle_sensitivity + bridge->effects.context_integration) / 4.0f;
    bridge->update_count++;
    return 0;
}

int emotion_recognition_substrate_bridge_get_effects(const emotion_recognition_substrate_bridge_t* bridge, emotion_recognition_substrate_effects_t* effects) {
    if (!bridge || !effects) return -1;
    *effects = bridge->effects;
    /* Phase 8: Heartbeat at operation start */
    emotion_recognition_substrate_bridge_heartbeat("emotion_reco_get_effects", 0.0f);


    return 0;
}

int emotion_recognition_substrate_bridge_apply_effects(emotion_recognition_substrate_bridge_t* bridge) {
    if (!bridge) return -1;
    if (!bridge->bio_async_connected || !bridge->ctx) return 0;

    /* Phase 8: Heartbeat at operation start */
    emotion_recognition_substrate_bridge_heartbeat("emotion_reco_apply_effects", 0.0f);


    substrate_metabolic_state_t metabolic;
    float atp_level = 1.0f, fatigue_level = 0.0f;
    if (bridge->substrate && substrate_get_metabolic_state(bridge->substrate, &metabolic) == 0) {
        atp_level = metabolic.atp_level;
        fatigue_level = 1.0f - metabolic.metabolic_capacity;
    }

    bio_msg_substrate_modulation_t msg;
    memset(&msg, 0, sizeof(msg));
    bio_msg_init_header(&msg.header, BIO_MSG_SUBSTRATE_MODULATION, BIO_MODULE_SUBSTRATE_EMOTION_RECOGNITION, 0, sizeof(msg));
    msg.header.channel = BIO_CHANNEL_SEROTONIN;  /* Emotion recognition uses serotonin */
    msg.bridge_module_id = BIO_MODULE_SUBSTRATE_EMOTION_RECOGNITION;
    msg.processing_capacity = bridge->effects.recognition_accuracy;
    msg.overall_capacity = bridge->effects.overall_capacity;
    msg.effect_values[0] = bridge->effects.recognition_accuracy;
    msg.effect_values[1] = bridge->effects.detection_speed;
    msg.effect_values[2] = bridge->effects.subtle_sensitivity;
    msg.effect_values[3] = bridge->effects.context_integration;
    msg.atp_level = atp_level;
    msg.fatigue_level = fatigue_level;
    msg.update_count = bridge->update_count;
    msg.critical_low = (atp_level < bridge->config.min_capacity);
    bio_router_broadcast(bridge->ctx, &msg, sizeof(msg));

    float delta = bridge->effects.overall_capacity - bridge->prev_overall_capacity;
    if (delta < -0.1f || delta > 0.1f) {
        bio_msg_substrate_capacity_update_t update_msg;
        memset(&update_msg, 0, sizeof(update_msg));
        bio_msg_init_header(&update_msg.header, BIO_MSG_SUBSTRATE_CAPACITY_UPDATE, BIO_MODULE_SUBSTRATE_EMOTION_RECOGNITION, 0, sizeof(update_msg));
        update_msg.bridge_module_id = BIO_MODULE_SUBSTRATE_EMOTION_RECOGNITION;
        update_msg.old_capacity = bridge->prev_overall_capacity;
        update_msg.new_capacity = bridge->effects.overall_capacity;
        update_msg.delta = delta;
        update_msg.significant_change = true;
        bio_router_broadcast(bridge->ctx, &update_msg, sizeof(update_msg));
    }
    bridge->prev_overall_capacity = bridge->effects.overall_capacity;
    return 0;
}

int emotion_recognition_substrate_bridge_register_bio_async(emotion_recognition_substrate_bridge_t* bridge, bio_router_t* router) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    emotion_recognition_substrate_bridge_heartbeat("emotion_reco_register_bio_async", 0.0f);


    if (bridge->bio_async_connected && bridge->ctx) {
        bio_router_unregister_module(bridge->ctx);
        bridge->ctx = NULL;
        bridge->bio_async_connected = false;
    }
    if (!router) return 0;
    bio_module_info_t info = { .module_id = BIO_MODULE_SUBSTRATE_EMOTION_RECOGNITION, .module_name = "emotion_recognition_substrate_bridge", .inbox_capacity = 16, .user_data = bridge };
    bridge->ctx = bio_router_register_module(&info);
    if (bridge->ctx) bridge->bio_async_connected = true;
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int emotion_recognition_substrate_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    emotion_recognition_substrate_bridge_heartbeat("emotion_reco_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Emotion_Recognition_Substrate_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                emotion_recognition_substrate_bridge_heartbeat("emotion_reco_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Emotion_Recognition_Substrate_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Emotion_Recognition_Substrate_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent + Training Stubs
 * ============================================================================ */

void emotion_recognition_substrate_bridge_set_instance_health_agent(emotion_recognition_substrate_bridge_t* bridge, nimcp_health_agent_t* agent) {
    if (bridge) {
        bridge->health_agent = agent;
    }
}

int emotion_recognition_substrate_bridge_training_begin(emotion_recognition_substrate_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "emotion_recognition_substrate_bridge_training_begin: NULL argument");
        return -1;
    }
    emotion_recognition_substrate_bridge_heartbeat_instance(bridge->health_agent, "erec_sub_training_begin", 0.0f);
    return 0;
}

int emotion_recognition_substrate_bridge_training_end(emotion_recognition_substrate_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "emotion_recognition_substrate_bridge_training_end: NULL argument");
        return -1;
    }
    emotion_recognition_substrate_bridge_heartbeat_instance(bridge->health_agent, "erec_sub_training_end", 1.0f);
    return 0;
}

int emotion_recognition_substrate_bridge_training_step(emotion_recognition_substrate_bridge_t* bridge, float progress) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "emotion_recognition_substrate_bridge_training_step: NULL argument");
        return -1;
    }
    emotion_recognition_substrate_bridge_heartbeat_instance(bridge->health_agent, "erec_sub_training_step", progress);
    return 0;
}
