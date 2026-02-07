/**
 * @file nimcp_consolidation_thalamic_bridge.c
 * @brief Memory Consolidation-Thalamic Bridge Implementation
 */

#include "cognitive/consolidation/nimcp_consolidation_thalamic_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(consolidation_thalamic_bridge)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_consolidation_thalamic_bridge_mesh_id = 0;
static mesh_participant_registry_t* g_consolidation_thalamic_bridge_mesh_registry = NULL;

nimcp_error_t consolidation_thalamic_bridge_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_consolidation_thalamic_bridge_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "consolidation_thalamic_bridge", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "consolidation_thalamic_bridge";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_consolidation_thalamic_bridge_mesh_id);
    if (err == NIMCP_SUCCESS) g_consolidation_thalamic_bridge_mesh_registry = registry;
    return err;
}

void consolidation_thalamic_bridge_mesh_unregister(void) {
    if (g_consolidation_thalamic_bridge_mesh_registry && g_consolidation_thalamic_bridge_mesh_id != 0) {
        mesh_participant_unregister(g_consolidation_thalamic_bridge_mesh_registry, g_consolidation_thalamic_bridge_mesh_id);
        g_consolidation_thalamic_bridge_mesh_id = 0;
        g_consolidation_thalamic_bridge_mesh_registry = NULL;
    }
}


/* ============================================================================
 * Phase 8 Instance-Level Health Agent Support
 * ============================================================================ */

static nimcp_health_agent_t* g_consolidation_thalamic_bridge_instance_health_agent = NULL;

static inline void consolidation_thalamic_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_consolidation_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_consolidation_thalamic_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_consolidation_thalamic_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}

#define LOG_MODULE "CONSOLIDATION_THALAMIC_BRIDGE"


struct consolidation_thalamic_bridge {
    bridge_base_t base;  /* MUST be first - provides mutex protection */
    nimcp_health_agent_t* health_agent;  /**< Phase 8: instance-level health agent */
    void* consolidation;
    thalamic_router_t* router;
    consolidation_thalamic_config_t config;
    consolidation_thalamic_stats_t stats;
    float attention_weight;
};

consolidation_thalamic_config_t consolidation_thalamic_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_consolidation_thalam", 0.0f);


    consolidation_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_emotional_boost = true,
        .min_memory_salience = 0.3f,
        .replay_boost = 0.3f
    };
    return cfg;
}

consolidation_thalamic_bridge_t* consolidation_thalamic_bridge_create(void* consolidation, thalamic_router_t* router, const consolidation_thalamic_config_t* config) {
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_create", 0.0f);


    consolidation_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(consolidation_thalamic_bridge_t));
    if (!bridge) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate bridge");

        return NULL;

    }

    /* Initialize mutex for thread safety */
    if (bridge_base_init(&bridge->base, 0, "consolidation_thalamic") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "consolidation_thalamic_bridge_create: bridge->base is NULL");
        return NULL;
    }

    bridge->consolidation = consolidation;
    bridge->router = router;
    bridge->config = config ? *config : consolidation_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    NIMCP_LOGGING_INFO("Created %s bridge", "consolidation_thalamic");
    return bridge;
}

void consolidation_thalamic_bridge_destroy(consolidation_thalamic_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_destroy", 0.0f);


    if (bridge) {
        if (bridge->base.mutex) {
            bridge_base_cleanup(&bridge->base);
        }
        nimcp_free(bridge);
    }
}

int consolidation_thalamic_bridge_reset(consolidation_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "consolidation_thalamic_bridge_reset: bridge is NULL");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_reset", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_route_encode(consolidation_thalamic_bridge_t* bridge, const consolidation_thalamic_signal_t* signal) {
    if (!bridge || !signal) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "consolidation_thalamic_route_encode: required parameter is NULL (bridge, signal)");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_consolidation_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    if (bridge->config.enable_attention_gating && signal->memory_salience < bridge->config.min_memory_salience) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return 0;
    }
    bridge->stats.encodes_routed++;
    bridge->stats.avg_memory_salience = (bridge->stats.avg_memory_salience * (bridge->stats.encodes_routed - 1) +
                                         signal->memory_salience) / bridge->stats.encodes_routed;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_route_replay(consolidation_thalamic_bridge_t* bridge, const void* memory, float importance) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "consolidation_thalamic_route_replay: bridge is NULL");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_consolidation_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.replays_triggered++;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_set_attention(consolidation_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "consolidation_thalamic_set_attention: bridge is NULL");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_consolidation_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_get_attention(const consolidation_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "consolidation_thalamic_get_attention: required parameter is NULL (bridge, attention)");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_consolidation_thalam", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *attention = bridge->attention_weight;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

int consolidation_thalamic_bridge_get_stats(const consolidation_thalamic_bridge_t* bridge, consolidation_thalamic_stats_t* stats) {
    if (!bridge || !stats) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "consolidation_thalamic_bridge_get_stats: required parameter is NULL (bridge, stats)");
        return -1;
    }
    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_get_stats", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);
    return 0;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int consolidation_thalamic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    consolidation_thalamic_bridge_heartbeat("consolidatio_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Consolidation_Thalamic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                consolidation_thalamic_bridge_heartbeat("consolidatio_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Consolidation_Thalamic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Consolidation_Thalamic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}

void consolidation_thalamic_bridge_set_instance_health_agent(consolidation_thalamic_bridge_t* bridge, nimcp_health_agent_t* agent) {
    if (!bridge) {
        NIMCP_THROW(NIMCP_ERROR_NULL_POINTER,
                    "consolidation_thalamic_bridge_set_instance_health_agent: NULL bridge");
        return;
    }
    bridge->health_agent = agent;
    g_consolidation_thalamic_bridge_instance_health_agent = agent;
    NIMCP_LOGGING_DEBUG("consolidation_thalamic_bridge: instance health agent %s", agent ? "set" : "cleared");
}

int consolidation_thalamic_bridge_training_begin(consolidation_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "consolidation_thalamic_bridge_training_begin: NULL argument");
        return -1;
    }
    consolidation_thalamic_bridge_heartbeat_instance(bridge, "consol_thl_training_begin", 0.0f);
    (void)bridge;
    return 0;
}

int consolidation_thalamic_bridge_training_end(consolidation_thalamic_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "consolidation_thalamic_bridge_training_end: NULL argument");
        return -1;
    }
    consolidation_thalamic_bridge_heartbeat_instance(bridge, "consol_thl_training_end", 1.0f);
    (void)bridge;
    return 0;
}

int consolidation_thalamic_bridge_training_step(consolidation_thalamic_bridge_t* bridge, float progress) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "consolidation_thalamic_bridge_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    consolidation_thalamic_bridge_heartbeat_instance(bridge, "consol_thl_training_step", progress);
    (void)bridge;
    return 0;
}
