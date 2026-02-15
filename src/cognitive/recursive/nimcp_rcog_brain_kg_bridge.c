/**
 * @file nimcp_rcog_brain_kg_bridge.c
 * @brief Brain Knowledge Graph Integration Bridge Implementation for Recursive Cognition
 * @version 1.0.0
 * @date 2026-01-03
 */

#include "cognitive/recursive/nimcp_rcog_brain_kg_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/platform/nimcp_platform_time.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <stdio.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
#include "utils/logging/nimcp_logging.h"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(rcog_brain_kg_bridge)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_rcog_brain_kg_bridge_mesh_id = 0;
static mesh_participant_registry_t* g_rcog_brain_kg_bridge_mesh_registry = NULL;

nimcp_error_t rcog_brain_kg_bridge_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_rcog_brain_kg_bridge_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "rcog_brain_kg_bridge", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "rcog_brain_kg_bridge";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_rcog_brain_kg_bridge_mesh_id);
    if (err == NIMCP_SUCCESS) g_rcog_brain_kg_bridge_mesh_registry = registry;
    return err;
}

void rcog_brain_kg_bridge_mesh_unregister(void) {
    if (g_rcog_brain_kg_bridge_mesh_registry && g_rcog_brain_kg_bridge_mesh_id != 0) {
        mesh_participant_unregister(g_rcog_brain_kg_bridge_mesh_registry, g_rcog_brain_kg_bridge_mesh_id);
        g_rcog_brain_kg_bridge_mesh_id = 0;
        g_rcog_brain_kg_bridge_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat from rcog_brain_kg_bridge module (instance-level) */
static inline void rcog_brain_kg_bridge_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_rcog_brain_kg_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_rcog_brain_kg_bridge_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_rcog_brain_kg_bridge_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}

static nimcp_health_agent_t* g_rcog_brain_kg_bridge_instance_health_agent = NULL;

void rcog_brain_kg_bridge_set_instance_health_agent(
    rcog_brain_kg_bridge_t* bridge, nimcp_health_agent_t* agent)
{
    (void)bridge;
    g_rcog_brain_kg_bridge_instance_health_agent = agent;
}

/* ============================================================================
 * Phase 8: Instance-level Training Functions
 * ============================================================================ */

int rcog_brain_kg_bridge_training_begin(rcog_brain_kg_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "rcog_brain_kg_bridge_training_begin: NULL argument");
        return -1;
    }
    rcog_brain_kg_bridge_heartbeat_instance(
        g_rcog_brain_kg_bridge_instance_health_agent, "rcog_kg_training_begin", 0.0f);
    return 0;
}

int rcog_brain_kg_bridge_training_step(rcog_brain_kg_bridge_t* bridge, float progress) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "rcog_brain_kg_bridge_training_step: NULL argument");
        return -1;
    }
    float clamped = progress < 0.0f ? 0.0f : (progress > 1.0f ? 1.0f : progress);
    rcog_brain_kg_bridge_heartbeat_instance(
        g_rcog_brain_kg_bridge_instance_health_agent, "rcog_kg_training_step", clamped);
    return 0;
}

int rcog_brain_kg_bridge_training_end(rcog_brain_kg_bridge_t* bridge) {
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "rcog_brain_kg_bridge_training_end: NULL argument");
        return -1;
    }
    rcog_brain_kg_bridge_heartbeat_instance(
        g_rcog_brain_kg_bridge_instance_health_agent, "rcog_kg_training_end", 1.0f);
    return 0;
}

#define LOG_MODULE "RCOG_BRAIN_KG_BRIDGE"


/*=============================================================================
 * INTERNAL STRUCTURES
 *===========================================================================*/

/**
 * @brief Registered node tracking
 */
typedef struct {
    rcog_kg_node_id_t node_id;
    rcog_kg_node_type_t node_type;
    bool registered;
} rcog_registered_node_t;

/**
 * @brief Cached capability entry
 */
typedef struct {
    rcog_capability_info_t info;
    uint64_t cached_at_ms;
    bool valid;
} rcog_cached_capability_t;

/**
 * @brief Brain KG bridge internal structure
 */
struct rcog_brain_kg_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */

    /* Configuration */
    rcog_brain_kg_bridge_config_t config;

    /* Connections */
    struct brain_kg* kg;
    struct kg_reader* reader;
    struct rcog_engine* engine;
    bool connected;

    /* Effects state */
    rcog_to_kg_effects_t outgoing_effects;
    kg_to_rcog_effects_t incoming_effects;

    /* Registered nodes */
    rcog_registered_node_t registered_nodes[RCOG_KG_NODE_COUNT];
    rcog_kg_node_id_t engine_node_id;
    bool engine_registered;

    /* Capability cache */
    rcog_cached_capability_t capabilities[RCOG_KG_MAX_CAPABILITIES];
    size_t num_capabilities;

    /* Processing state */
    rcog_processing_state_t current_state;
    uint64_t last_state_update_ms;

    /* Current focus */
    char current_focus[128];
    float system_health;

    /* Statistics */
    rcog_brain_kg_bridge_stats_t stats;

    /* Phase 8: Instance-level health agent */
    nimcp_health_agent_t* health_agent;
};

/*=============================================================================
 * INTERNAL HELPERS
 *===========================================================================*/

/**
 * @brief Generate a simple node ID based on type and time
 */
static rcog_kg_node_id_t generate_node_id(rcog_kg_node_type_t type) {
    /* Simple ID generation - in full implementation would use KG's system */
    uint64_t now = nimcp_platform_time_monotonic_ms();
    return ((uint64_t)type << 56) | (now & 0x00FFFFFFFFFFFFFFULL);
}

/*=============================================================================
 * LIFECYCLE
 *===========================================================================*/

rcog_brain_kg_bridge_config_t rcog_brain_kg_bridge_default_config(void) {
    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_default_config", 0.0f);


    rcog_brain_kg_bridge_config_t config = {0};

    config.auto_register_on_connect = true;
    config.register_all_components = true;

    config.state_update_interval_ms = RCOG_KG_DEFAULT_UPDATE_INTERVAL;
    config.enable_continuous_updates = true;

    config.enable_introspection = true;
    config.enable_semantic_queries = true;
    config.max_semantic_query_depth = 3;

    config.lazy_capability_query = true;
    config.capability_cache_ttl_ms = 5000;

    return config;
}

rcog_brain_kg_bridge_t* rcog_brain_kg_bridge_create(
    const rcog_brain_kg_bridge_config_t* config
) {
    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_create", 0.0f);


    rcog_brain_kg_bridge_t* bridge = nimcp_calloc(1, sizeof(rcog_brain_kg_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate bridge");

        return NULL;
    }

    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = rcog_brain_kg_bridge_default_config();
    }

    /* Initialize bridge base infrastructure (includes mutex) */
    if (bridge_base_init(&bridge->base, 0, "rcog_brain_kg") != 0) {
        nimcp_free(bridge);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_INITIALIZED, "rcog_brain_kg_bridge_create: validation failed");
        return NULL;
    }

    /* Initialize healthy state */
    bridge->system_health = 1.0f;
    strncpy(bridge->current_focus, "idle", sizeof(bridge->current_focus) - 1);

    return bridge;
}

rcog_brain_kg_bridge_t* rcog_brain_kg_bridge_create_default(void) {
    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_create_default", 0.0f);


    return rcog_brain_kg_bridge_create(NULL);
}

void rcog_brain_kg_bridge_destroy(rcog_brain_kg_bridge_t* bridge) {
    if (!bridge) {
        return;
        NIMCP_LOGGING_DEBUG("Destroying %s bridge", "rcog_brain_kg");
    }

    /* Cleanup base bridge infrastructure */
    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_destroy", 0.0f);


    bridge_base_cleanup(&bridge->base);

    nimcp_free(bridge);
}

/*=============================================================================
 * CONNECTION
 *===========================================================================*/

int rcog_brain_kg_bridge_connect(
    rcog_brain_kg_bridge_t* bridge,
    struct brain_kg* kg
) {
    if (!bridge || !kg) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_connect", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->kg = kg;
    bridge->connected = (bridge->kg != NULL && bridge->engine != NULL);

    /* Auto-register if enabled */
    if (bridge->connected && bridge->config.auto_register_on_connect && !bridge->engine_registered) {
        rcog_kg_node_id_t node_id;
        /* Note: recursive call requires unlock/relock */
        nimcp_mutex_unlock(bridge->base.mutex);
        rcog_brain_kg_bridge_register_engine(bridge, &node_id);
        nimcp_mutex_lock(bridge->base.mutex);
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

int rcog_brain_kg_bridge_connect_reader(
    rcog_brain_kg_bridge_t* bridge,
    struct kg_reader* reader
) {
    if (!bridge || !reader) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_connect_reader", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->reader = reader;
    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

int rcog_brain_kg_bridge_connect_engine(
    rcog_brain_kg_bridge_t* bridge,
    struct rcog_engine* engine
) {
    if (!bridge || !engine) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_connect_engine", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    bridge->engine = engine;
    bridge->connected = (bridge->kg != NULL && bridge->engine != NULL);
    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

bool rcog_brain_kg_bridge_is_connected(const rcog_brain_kg_bridge_t* bridge) {
    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_is_connected", 0.0f);


    return bridge && bridge->connected;
}

/*=============================================================================
 * UPDATE
 *===========================================================================*/

int rcog_brain_kg_bridge_update(
    rcog_brain_kg_bridge_t* bridge,
    float delta_time_ms
) {
    if (!bridge) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_update", 0.0f);


    uint64_t now = nimcp_platform_time_monotonic_ms();

    nimcp_mutex_lock(bridge->base.mutex);

    /* Check if state update is due */
    if (bridge->config.enable_continuous_updates) {
        uint64_t elapsed = now - bridge->last_state_update_ms;
        if (elapsed >= bridge->config.state_update_interval_ms) {
            bridge->outgoing_effects.update_processing_state = true;
            bridge->outgoing_effects.state = bridge->current_state;
            bridge->last_state_update_ms = now;
            bridge->stats.state_updates++;
        }
    }

    /* Invalidate stale capability cache entries */
    for (size_t i = 0; i < bridge->num_capabilities; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_capabilities > 256) {
            rcog_brain_kg_bridge_heartbeat("rcog_brain_k_loop",
                             (float)(i + 1) / (float)bridge->num_capabilities);
        }

        if (bridge->capabilities[i].valid) {
            uint64_t age = now - bridge->capabilities[i].cached_at_ms;
            if (age > bridge->config.capability_cache_ttl_ms) {
                bridge->capabilities[i].valid = false;
            }
        }
    }

    /* Update incoming effects */
    bridge->incoming_effects.self_model_available = bridge->engine_registered;
    bridge->incoming_effects.registered_capabilities = (uint32_t)bridge->num_capabilities;
    bridge->incoming_effects.overall_health = bridge->system_health;
    bridge->incoming_effects.introspection_available = bridge->config.enable_introspection;
    bridge->incoming_effects.current_module_focus = bridge->current_focus;

    /* Count connected modules */
    uint32_t connected = 0;
    for (int i = 0; i < RCOG_KG_NODE_COUNT; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && RCOG_KG_NODE_COUNT > 256) {
            rcog_brain_kg_bridge_heartbeat("rcog_brain_k_loop",
                             (float)(i + 1) / (float)RCOG_KG_NODE_COUNT);
        }

        if (bridge->registered_nodes[i].registered) {
            connected++;
        }
    }
    bridge->incoming_effects.connected_modules = connected;
    bridge->incoming_effects.full_graph_available = (bridge->kg != NULL);

    (void)delta_time_ms;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

/*=============================================================================
 * REGISTRATION
 *===========================================================================*/

int rcog_brain_kg_bridge_register_engine(
    rcog_brain_kg_bridge_t* bridge,
    rcog_kg_node_id_t* node_id
) {
    if (!bridge || !node_id) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_register_engine", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->kg) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_KG_NOT_CONNECTED;
    }

    if (bridge->engine_registered) {
        *node_id = bridge->engine_node_id;
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_OK;
    }

    /* Generate node ID and register */
    bridge->engine_node_id = generate_node_id(RCOG_KG_NODE_ENGINE);
    bridge->engine_registered = true;

    bridge->registered_nodes[RCOG_KG_NODE_ENGINE].node_id = bridge->engine_node_id;
    bridge->registered_nodes[RCOG_KG_NODE_ENGINE].node_type = RCOG_KG_NODE_ENGINE;
    bridge->registered_nodes[RCOG_KG_NODE_ENGINE].registered = true;

    *node_id = bridge->engine_node_id;

    /* Register all components if configured */
    if (bridge->config.register_all_components) {
        for (int i = 1; i < RCOG_KG_NODE_COUNT; i++) {
            if (!bridge->registered_nodes[i].registered) {
                bridge->registered_nodes[i].node_id = generate_node_id((rcog_kg_node_type_t)i);
                bridge->registered_nodes[i].node_type = (rcog_kg_node_type_t)i;
                bridge->registered_nodes[i].registered = true;
            }
        }
    }

    bridge->stats.registrations_performed++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

int rcog_brain_kg_bridge_register_component(
    rcog_brain_kg_bridge_t* bridge,
    rcog_kg_node_type_t component_type,
    rcog_kg_node_id_t* node_id
) {
    if (!bridge || !node_id) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_register_component", 0.0f);


    if (component_type < 0 || component_type >= RCOG_KG_NODE_COUNT) {
        return RCOG_ERROR_INVALID_CONFIG;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->kg) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_KG_NOT_CONNECTED;
    }

    if (bridge->registered_nodes[component_type].registered) {
        *node_id = bridge->registered_nodes[component_type].node_id;
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_OK;
    }

    bridge->registered_nodes[component_type].node_id = generate_node_id(component_type);
    bridge->registered_nodes[component_type].node_type = component_type;
    bridge->registered_nodes[component_type].registered = true;

    *node_id = bridge->registered_nodes[component_type].node_id;

    bridge->stats.registrations_performed++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

int rcog_brain_kg_bridge_register_capability(
    rcog_brain_kg_bridge_t* bridge,
    const rcog_capability_info_t* capability
) {
    if (!bridge || !capability) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_register_capability", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->num_capabilities >= RCOG_KG_MAX_CAPABILITIES) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_CONTEXT_FULL;
    }

    /* Check for duplicate */
    for (size_t i = 0; i < bridge->num_capabilities; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_capabilities > 256) {
            rcog_brain_kg_bridge_heartbeat("rcog_brain_k_loop",
                             (float)(i + 1) / (float)bridge->num_capabilities);
        }

        if (bridge->capabilities[i].info.name &&
            capability->name &&
            strcmp(bridge->capabilities[i].info.name, capability->name) == 0) {
            /* Update existing */
            bridge->capabilities[i].info = *capability;
            bridge->capabilities[i].cached_at_ms = nimcp_platform_time_monotonic_ms();
            bridge->capabilities[i].valid = true;
            nimcp_mutex_unlock(bridge->base.mutex);
            return RCOG_OK;
        }
    }

    /* Add new */
    bridge->capabilities[bridge->num_capabilities].info = *capability;
    bridge->capabilities[bridge->num_capabilities].cached_at_ms = nimcp_platform_time_monotonic_ms();
    bridge->capabilities[bridge->num_capabilities].valid = true;
    bridge->num_capabilities++;

    /* Update outgoing effects */
    bridge->outgoing_effects.register_capabilities = true;
    bridge->outgoing_effects.num_capabilities = (uint32_t)bridge->num_capabilities;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

int rcog_brain_kg_bridge_create_edge(
    rcog_brain_kg_bridge_t* bridge,
    rcog_kg_node_id_t from_node,
    rcog_kg_node_id_t to_node,
    rcog_kg_edge_type_t edge_type
) {
    if (!bridge) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_create_edge", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->kg) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_KG_NOT_CONNECTED;
    }

    /* In full implementation, would create edge in KG */
    (void)from_node;
    (void)to_node;
    (void)edge_type;

    bridge->stats.edges_created++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

/*=============================================================================
 * STATE UPDATES
 *===========================================================================*/

int rcog_brain_kg_bridge_update_state(
    rcog_brain_kg_bridge_t* bridge,
    const rcog_processing_state_t* state
) {
    if (!bridge || !state) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_update_state", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    bridge->current_state = *state;
    bridge->outgoing_effects.update_processing_state = true;
    bridge->outgoing_effects.state = *state;
    bridge->last_state_update_ms = nimcp_platform_time_monotonic_ms();

    /* Update focus based on state */
    if (state->is_processing) {
        snprintf(bridge->current_focus, sizeof(bridge->current_focus),
                 "processing (depth=%u)", state->current_depth);
    } else {
        strncpy(bridge->current_focus, "idle", sizeof(bridge->current_focus) - 1);
    }

    bridge->stats.state_updates++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

int rcog_brain_kg_bridge_set_property(
    rcog_brain_kg_bridge_t* bridge,
    rcog_kg_node_id_t node_id,
    const char* property_name,
    const char* value
) {
    if (!bridge || !property_name || !value) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_set_property", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->kg) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_KG_NOT_CONNECTED;
    }

    /* In full implementation, would set property in KG */
    (void)node_id;

    bridge->stats.properties_updated++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

/*=============================================================================
 * CAPABILITY QUERIES
 *===========================================================================*/

int rcog_brain_kg_bridge_query_capabilities(
    rcog_brain_kg_bridge_t* bridge,
    rcog_capability_info_t* capabilities,
    size_t max_capabilities,
    size_t* num_capabilities
) {
    if (!bridge || !capabilities || !num_capabilities) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_query_capabilities", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    size_t to_copy = bridge->num_capabilities;
    if (to_copy > max_capabilities) {
        to_copy = max_capabilities;
    }

    for (size_t i = 0; i < to_copy; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && to_copy > 256) {
            rcog_brain_kg_bridge_heartbeat("rcog_brain_k_loop",
                             (float)(i + 1) / (float)to_copy);
        }

        capabilities[i] = bridge->capabilities[i].info;
    }
    *num_capabilities = to_copy;

    bridge->stats.capability_queries++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

bool rcog_brain_kg_bridge_has_capability(
    const rcog_brain_kg_bridge_t* bridge,
    const char* capability_name
) {
    if (!bridge || !capability_name) {
        return false;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_has_capability", 0.0f);


    nimcp_mutex_lock(((rcog_brain_kg_bridge_t*)bridge)->base.mutex);

    for (size_t i = 0; i < bridge->num_capabilities; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && bridge->num_capabilities > 256) {
            rcog_brain_kg_bridge_heartbeat("rcog_brain_k_loop",
                             (float)(i + 1) / (float)bridge->num_capabilities);
        }

        if (bridge->capabilities[i].valid &&
            bridge->capabilities[i].info.name &&
            strcmp(bridge->capabilities[i].info.name, capability_name) == 0 &&
            bridge->capabilities[i].info.currently_available) {
            nimcp_mutex_unlock(((rcog_brain_kg_bridge_t*)bridge)->base.mutex);
            return true;
        }
    }

    nimcp_mutex_unlock(((rcog_brain_kg_bridge_t*)bridge)->base.mutex);
    return false;
}

/*=============================================================================
 * SEMANTIC QUERIES
 *===========================================================================*/

int rcog_brain_kg_bridge_load_semantic_knowledge(
    rcog_brain_kg_bridge_t* bridge,
    struct rcog_context_store* context_store,
    const char* variable_name,
    const char* query,
    size_t max_entities
) {
    if (!bridge || !context_store || !variable_name || !query) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_load_semantic_knowle", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->reader) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_KG_NOT_CONNECTED;
    }

    if (!bridge->config.enable_semantic_queries) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_TOOL_ACCESS_DENIED;
    }

    /* In full implementation, would query KG and load into context store */
    (void)max_entities;

    bridge->stats.semantic_queries++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

int rcog_brain_kg_bridge_semantic_query(
    rcog_brain_kg_bridge_t* bridge,
    const char* query,
    rcog_semantic_query_result_t* result
) {
    if (!bridge || !query || !result) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_semantic_query", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->reader) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_KG_NOT_CONNECTED;
    }

    if (!bridge->config.enable_semantic_queries) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return RCOG_ERROR_TOOL_ACCESS_DENIED;
    }

    /* Placeholder result */
    result->query = query;
    result->success = true;
    result->num_results = 0;
    result->result_data = NULL;
    result->result_size = 0;

    bridge->stats.semantic_queries++;
    bridge->incoming_effects.semantic_results_ready = true;

    nimcp_mutex_unlock(bridge->base.mutex);

    return RCOG_OK;
}

void rcog_brain_kg_bridge_free_semantic_result(
    rcog_semantic_query_result_t* result
) {
    if (!result) {
        return;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_free_semantic_result", 0.0f);


    if (result->result_data) {
        nimcp_free(result->result_data);
        result->result_data = NULL;
    }
    result->result_size = 0;
    result->num_results = 0;
}

/*=============================================================================
 * INTROSPECTION
 *===========================================================================*/

int rcog_brain_kg_bridge_get_focus(
    const rcog_brain_kg_bridge_t* bridge,
    char* focus,
    size_t max_len
) {
    if (!bridge || !focus || max_len == 0) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_get_focus", 0.0f);


    nimcp_mutex_lock(((rcog_brain_kg_bridge_t*)bridge)->base.mutex);

    strncpy(focus, bridge->current_focus, max_len - 1);
    focus[max_len - 1] = '\0';

    nimcp_mutex_unlock(((rcog_brain_kg_bridge_t*)bridge)->base.mutex);

    return RCOG_OK;
}

float rcog_brain_kg_bridge_get_system_health(
    const rcog_brain_kg_bridge_t* bridge
) {
    if (!bridge) {
        return 0.0f;
    }
    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_get_system_health", 0.0f);


    return bridge->system_health;
}

int rcog_brain_kg_bridge_get_connected_modules(
    const rcog_brain_kg_bridge_t* bridge,
    char** modules,
    size_t max_modules,
    size_t* num_modules
) {
    if (!bridge || !modules || !num_modules) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Placeholder - would query KG for connected modules */
    *num_modules = 0;
    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_get_connected_module", 0.0f);


    (void)max_modules;

    return RCOG_OK;
}

/*=============================================================================
 * EFFECTS ACCESS
 *===========================================================================*/

int rcog_brain_kg_bridge_get_outgoing_effects(
    const rcog_brain_kg_bridge_t* bridge,
    rcog_to_kg_effects_t* effects
) {
    if (!bridge || !effects) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_get_outgoing_effects", 0.0f);


    nimcp_mutex_lock(((rcog_brain_kg_bridge_t*)bridge)->base.mutex);
    *effects = bridge->outgoing_effects;
    nimcp_mutex_unlock(((rcog_brain_kg_bridge_t*)bridge)->base.mutex);

    return RCOG_OK;
}

int rcog_brain_kg_bridge_get_incoming_effects(
    const rcog_brain_kg_bridge_t* bridge,
    kg_to_rcog_effects_t* effects
) {
    if (!bridge || !effects) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_get_incoming_effects", 0.0f);


    nimcp_mutex_lock(((rcog_brain_kg_bridge_t*)bridge)->base.mutex);
    *effects = bridge->incoming_effects;
    nimcp_mutex_unlock(((rcog_brain_kg_bridge_t*)bridge)->base.mutex);

    return RCOG_OK;
}

/*=============================================================================
 * STATISTICS
 *===========================================================================*/

int rcog_brain_kg_bridge_get_stats(
    const rcog_brain_kg_bridge_t* bridge,
    rcog_brain_kg_bridge_stats_t* stats
) {
    if (!bridge || !stats) {
        return RCOG_ERROR_NULL_POINTER;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_get_stats", 0.0f);


    nimcp_mutex_lock(((rcog_brain_kg_bridge_t*)bridge)->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(((rcog_brain_kg_bridge_t*)bridge)->base.mutex);

    return RCOG_OK;
}

void rcog_brain_kg_bridge_reset_stats(rcog_brain_kg_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_reset_stats", 0.0f);


    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(rcog_brain_kg_bridge_stats_t));
    nimcp_mutex_unlock(bridge->base.mutex);
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int rcog_brain_kg_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    rcog_brain_kg_bridge_heartbeat("rcog_brain_k_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Recursive_Cognition_Brain_KG_Bridge_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                rcog_brain_kg_bridge_heartbeat("rcog_brain_k_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            /* Log self-knowledge observations */
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Recursive_Cognition_Brain_KG_Bridge_Module");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Recursive_Cognition_Brain_KG_Bridge_Module");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
