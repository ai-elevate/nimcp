/**
 * @file nimcp_brain_init_surface_geometry.c
 * @brief Brain Initialization for Surface Geometry Subsystem
 *
 * @version 1.0.0
 * @date 2026-01-13
 */

#include "core/brain/factory/init/nimcp_brain_init_surface_geometry.h"
#include "core/geometry/nimcp_surface_geometry.h"
#include "core/brain/bridges/nimcp_surface_geometry_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_init_surface_geometry)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_brain_init_surface_geometry_mesh_id = 0;
static mesh_participant_registry_t* g_brain_init_surface_geometry_mesh_registry = NULL;

nimcp_error_t brain_init_surface_geometry_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_brain_init_surface_geometry_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "brain_init_surface_geometry", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "brain_init_surface_geometry";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_brain_init_surface_geometry_mesh_id);
    if (err == NIMCP_SUCCESS) g_brain_init_surface_geometry_mesh_registry = registry;
    return err;
}

void brain_init_surface_geometry_mesh_unregister(void) {
    if (g_brain_init_surface_geometry_mesh_registry && g_brain_init_surface_geometry_mesh_id != 0) {
        mesh_participant_unregister(g_brain_init_surface_geometry_mesh_registry, g_brain_init_surface_geometry_mesh_id);
        g_brain_init_surface_geometry_mesh_id = 0;
        g_brain_init_surface_geometry_mesh_registry = NULL;
    }
}


//=============================================================================
// INTERNAL STORAGE STRUCTURE
//=============================================================================

/**
 * @brief Surface geometry subsystem state stored in brain
 *
 * This structure is allocated and stored in the brain's extension data
 * during initialization.
 */
typedef struct surface_geometry_subsystem_struct {
    surface_geometry_ctx_t* geometry_ctx;
    surface_geometry_bridge_t* bridge;
    surface_geometry_init_flags_t flags;

    /* Callbacks */
    dendrite_geometry_callback_t dendrite_callback;
    void* dendrite_callback_data;
    axon_geometry_callback_t axon_callback;
    void* axon_callback_data;

    /* KG registration */
    uint32_t kg_node_id;

    bool initialized;
} surface_geometry_subsystem_t;

/* Extension data key for storing in brain */
#define SURFACE_GEOMETRY_EXTENSION_KEY "surface_geometry_subsystem"

//=============================================================================
// INTERNAL HELPERS
//=============================================================================

/**
 * @brief Get or create subsystem storage in brain
 */
static surface_geometry_subsystem_t* get_or_create_subsystem(brain_t brain) {
    if (!brain) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;

    }

    /* For now, we'll use a static allocation since we don't have
     * direct access to brain's extension storage mechanism.
     * In a full implementation, this would use brain_get_extension()
     * or similar. */
    static surface_geometry_subsystem_t s_subsystem;
    static bool s_initialized = false;

    if (!s_initialized) {
        memset(&s_subsystem, 0, sizeof(s_subsystem));
        s_initialized = true;
    }

    return &s_subsystem;
}

/**
 * @brief Get existing subsystem (no create)
 */
static surface_geometry_subsystem_t* get_subsystem(brain_t brain) {
    if (!brain) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;

    }

    surface_geometry_subsystem_t* subsystem = get_or_create_subsystem(brain);
    if (!subsystem || !subsystem->initialized) {
        return NULL;
    }
    return subsystem;
}

//=============================================================================
// INITIALIZATION
//=============================================================================

bool nimcp_brain_factory_init_surface_geometry_subsystem(brain_t brain) {
    return nimcp_brain_factory_init_surface_geometry_with_flags(
        brain,
        SURFACE_INIT_DEFAULT
    );
}

bool nimcp_brain_factory_init_surface_geometry_with_flags(
    brain_t brain,
    surface_geometry_init_flags_t flags
) {
    if (!brain) {
        NIMCP_LOGGING_ERROR("NULL brain handle");
        return false;
    }

    surface_geometry_subsystem_t* subsystem = get_or_create_subsystem(brain);
    if (!subsystem) {
        NIMCP_LOGGING_ERROR("Failed to get subsystem storage");
        return false;
    }

    /* Check if already initialized */
    if (subsystem->initialized) {
        NIMCP_LOGGING_WARN("Surface geometry already initialized");
        return true;
    }

    subsystem->flags = flags;

    /* Step 1: Create surface geometry context */
    surface_geometry_config_t geo_config;
    surface_geometry_default_config(&geo_config);

    /* Disable bio-async in geometry context if requested */
    if (flags & SURFACE_INIT_NO_BIO_ASYNC) {
        geo_config.enable_bio_async = false;
    }

    /* Disable quantum in geometry context if requested */
    if (flags & SURFACE_INIT_NO_QUANTUM) {
        geo_config.enable_quantum = false;
    }

    subsystem->geometry_ctx = surface_geometry_create(&geo_config);
    if (!subsystem->geometry_ctx) {
        NIMCP_LOGGING_ERROR("Failed to create surface geometry context");
        return false;
    }

    /* Step 2: Create surface geometry bridge */
    surface_geometry_bridge_config_t bridge_config;
    surface_geometry_bridge_default_config(&bridge_config);

    if (flags & SURFACE_INIT_NO_BIO_ASYNC) {
        bridge_config.enable_bio_async = false;
    }

    subsystem->bridge = surface_geometry_bridge_create(&bridge_config);
    if (!subsystem->bridge) {
        NIMCP_LOGGING_ERROR("Failed to create surface geometry bridge");
        surface_geometry_destroy(subsystem->geometry_ctx);
        subsystem->geometry_ctx = NULL;
        return false;
    }

    /* Step 3: Connect geometry context to bridge */
    if (surface_geometry_bridge_connect_geometry(
            subsystem->bridge, subsystem->geometry_ctx) != 0) {
        NIMCP_LOGGING_ERROR("Failed to connect geometry to bridge");
        surface_geometry_bridge_destroy(subsystem->bridge);
        surface_geometry_destroy(subsystem->geometry_ctx);
        subsystem->bridge = NULL;
        subsystem->geometry_ctx = NULL;
        return false;
    }

    /* Step 4: Connect brain to bridge */
    if (surface_geometry_bridge_connect_brain(subsystem->bridge, brain) != 0) {
        NIMCP_LOGGING_ERROR("Failed to connect brain to bridge");
        surface_geometry_bridge_destroy(subsystem->bridge);
        surface_geometry_destroy(subsystem->geometry_ctx);
        subsystem->bridge = NULL;
        subsystem->geometry_ctx = NULL;
        return false;
    }

    /* Step 5: Connect bio-async if enabled */
    if (!(flags & SURFACE_INIT_NO_BIO_ASYNC)) {
        if (surface_geometry_bridge_connect_bio_async(subsystem->bridge) != 0) {
            NIMCP_LOGGING_WARN("Failed to connect bio-async (continuing without)");
            /* Non-fatal - continue without bio-async */
        }
    }

    /* Step 6: Register with KG wiring diagram */
    /* In a full implementation, this would call brain_kg_add_node() */
    subsystem->kg_node_id = 0;  /* Placeholder */

    /* Mark as initialized */
    subsystem->initialized = true;

    NIMCP_LOGGING_INFO("Surface geometry subsystem initialized successfully");

    return true;
}

bool nimcp_brain_factory_shutdown_surface_geometry_subsystem(brain_t brain) {
    if (!brain) return false;

    surface_geometry_subsystem_t* subsystem = get_subsystem(brain);
    if (!subsystem) {
        return true;  /* Not initialized, nothing to do */
    }

    /* Disconnect and destroy bridge */
    if (subsystem->bridge) {
        surface_geometry_bridge_disconnect_bio_async(subsystem->bridge);
        surface_geometry_bridge_disconnect_brain(subsystem->bridge);
        surface_geometry_bridge_disconnect_geometry(subsystem->bridge);
        surface_geometry_bridge_destroy(subsystem->bridge);
        subsystem->bridge = NULL;
    }

    /* Destroy geometry context */
    if (subsystem->geometry_ctx) {
        surface_geometry_destroy(subsystem->geometry_ctx);
        subsystem->geometry_ctx = NULL;
    }

    /* Clear callbacks */
    subsystem->dendrite_callback = NULL;
    subsystem->dendrite_callback_data = NULL;
    subsystem->axon_callback = NULL;
    subsystem->axon_callback_data = NULL;

    subsystem->initialized = false;

    NIMCP_LOGGING_INFO("Surface geometry subsystem shutdown complete");

    return true;
}

//=============================================================================
// ACCESSORS
//=============================================================================

surface_geometry_ctx_t* nimcp_brain_get_surface_geometry_ctx(brain_t brain) {
    surface_geometry_subsystem_t* subsystem = get_subsystem(brain);
    return subsystem ? subsystem->geometry_ctx : NULL;
}

surface_geometry_bridge_t* nimcp_brain_get_surface_geometry_bridge(brain_t brain) {
    surface_geometry_subsystem_t* subsystem = get_subsystem(brain);
    return subsystem ? subsystem->bridge : NULL;
}

bool nimcp_brain_has_surface_geometry(brain_t brain) {
    surface_geometry_subsystem_t* subsystem = get_subsystem(brain);
    return subsystem && subsystem->initialized;
}

//=============================================================================
// INTEGRATION HOOKS
//=============================================================================

int nimcp_brain_register_dendrite_geometry_callback(
    brain_t brain,
    dendrite_geometry_callback_t callback,
    void* user_data
) {
    surface_geometry_subsystem_t* subsystem = get_subsystem(brain);
    if (!subsystem) return -1;

    subsystem->dendrite_callback = callback;
    subsystem->dendrite_callback_data = user_data;

    return 0;
}

int nimcp_brain_register_axon_geometry_callback(
    brain_t brain,
    axon_geometry_callback_t callback,
    void* user_data
) {
    surface_geometry_subsystem_t* subsystem = get_subsystem(brain);
    if (!subsystem) return -1;

    subsystem->axon_callback = callback;
    subsystem->axon_callback_data = user_data;

    return 0;
}
