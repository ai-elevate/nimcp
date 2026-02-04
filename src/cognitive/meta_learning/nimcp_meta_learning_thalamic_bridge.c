/**
 * @file nimcp_meta_learning_thalamic_bridge.c
 * @brief Meta-Learning-Thalamic Bridge Implementation
 */

#include "cognitive/meta_learning/nimcp_meta_learning_thalamic_bridge.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(meta_learning_thalamic_bridge)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_meta_learning_thalamic_bridge_mesh_id = 0;
static mesh_participant_registry_t* g_meta_learning_thalamic_bridge_mesh_registry = NULL;

nimcp_error_t meta_learning_thalamic_bridge_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_meta_learning_thalamic_bridge_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "meta_learning_thalamic_bridge", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "meta_learning_thalamic_bridge";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_meta_learning_thalamic_bridge_mesh_id);
    if (err == NIMCP_SUCCESS) g_meta_learning_thalamic_bridge_mesh_registry = registry;
    return err;
}

void meta_learning_thalamic_bridge_mesh_unregister(void) {
    if (g_meta_learning_thalamic_bridge_mesh_registry && g_meta_learning_thalamic_bridge_mesh_id != 0) {
        mesh_participant_unregister(g_meta_learning_thalamic_bridge_mesh_registry, g_meta_learning_thalamic_bridge_mesh_id);
        g_meta_learning_thalamic_bridge_mesh_id = 0;
        g_meta_learning_thalamic_bridge_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat from meta_learning_thalamic_bridge module (instance-level) */
static inline void meta_learning_thalamic_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_meta_learning_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_meta_learning_thalamic_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_meta_learning_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


#define LOG_MODULE "META_LEARNING_THALAMIC_BRIDGE"


struct meta_learning_thalamic_bridge {
    bridge_base_t base;
    nimcp_health_agent_t* health_agent;  /**< Phase 8: instance-level health agent */
    void* meta_learning;
    thalamic_router_t* router;
    meta_learning_thalamic_config_t config;
    meta_learning_thalamic_stats_t stats;
    float attention_weight;
};

meta_learning_thalamic_config_t meta_learning_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_meta_learning_thalam", 0.0f);


    meta_learning_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_strategy_broadcast = true,
        .min_strategy_confidence = 0.3f,
        .transfer_threshold = 0.5f
    };
    return cfg;
}

meta_learning_thalamic_bridge_t* meta_learning_thalamic_bridge_create(void* meta_learning, thalamic_router_t* router, const meta_learning_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_create", 0.0f);


    meta_learning_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(meta_learning_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate bridge");

        return NULL;

    }
    if (bridge_base_init(&bridge->base, 0, "meta_learning_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }
    bridge->meta_learning = meta_learning;
    bridge->router = router;
    bridge->config = config ? *config : meta_learning_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    NIMCP_LOGGING_INFO("Created %s bridge", "meta_learning_thalamic");
    return bridge;
}

void meta_learning_thalamic_bridge_destroy(meta_learning_thalamic_bridge_t* bridge) {
    if (!bridge) return;
    NIMCP_LOGGING_DEBUG("Destroying %s bridge", "meta_learning_thalamic");
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_destroy", 0.0f);


    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
    }
    nimcp_free(bridge);
}

int meta_learning_thalamic_bridge_reset(meta_learning_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int meta_learning_thalamic_route_strategy(meta_learning_thalamic_bridge_t* bridge, const meta_learning_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_meta_learning_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->strategy_confidence < bridge->config.min_strategy_confidence) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.strategies_routed++;
    bridge->stats.avg_learning_rate = (bridge->stats.avg_learning_rate * (bridge->stats.strategies_routed - 1) +
                                       signal->learning_rate) / bridge->stats.strategies_routed;
    if (signal->signal_type == META_LEARNING_SIGNAL_RATE) {
        bridge->stats.rate_adjustments++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int meta_learning_thalamic_route_transfer(meta_learning_thalamic_bridge_t* bridge, const void* knowledge, float potential) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_meta_learning_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    if (potential >= bridge->config.transfer_threshold) {
        bridge->stats.transfers_initiated++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int meta_learning_thalamic_set_attention(meta_learning_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_meta_learning_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int meta_learning_thalamic_get_attention(const meta_learning_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_meta_learning_thalam", 0.0f);


    return 0;
}

int meta_learning_thalamic_bridge_get_stats(const meta_learning_thalamic_bridge_t* bridge, meta_learning_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_get_stats", 0.0f);


    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int meta_learning_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    meta_learning_thalamic_bridge_heartbeat("meta_learnin_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Meta_Learning_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                meta_learning_thalamic_bridge_heartbeat("meta_learnin_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Meta_Learning_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Meta_Learning_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void meta_learning_thalamic_bridge_set_instance_health_agent(meta_learning_thalamic_bridge_t* bridge, nimcp_health_agent_t* agent) {
    if (!bridge) {
        NIMCP_THROW(NIMCP_ERROR_NULL_POINTER,
                    "meta_learning_thalamic_bridge_set_instance_health_agent: NULL bridge");
        return;
    }
    bridge->health_agent = agent;
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int meta_learning_thalamic_bridge_training_begin(meta_learning_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "meta_learning_thalamic_bridge_training_begin: NULL argument");
        return -1;
    }
    meta_learning_thalamic_bridge_heartbeat_instance(bridge->health_agent, "meta_learning_thalamic_bridge_training_begin", 0.0f);
    return 0;
}

int meta_learning_thalamic_bridge_training_end(meta_learning_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "meta_learning_thalamic_bridge_training_end: NULL argument");
        return -1;
    }
    meta_learning_thalamic_bridge_heartbeat_instance(bridge->health_agent, "meta_learning_thalamic_bridge_training_end", 1.0f);
    return 0;
}

int meta_learning_thalamic_bridge_training_step(meta_learning_thalamic_bridge_t* bridge, float progress) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "meta_learning_thalamic_bridge_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    meta_learning_thalamic_bridge_heartbeat_instance(bridge->health_agent, "meta_learning_thalamic_bridge_training_step", progress);
    return 0;
}
