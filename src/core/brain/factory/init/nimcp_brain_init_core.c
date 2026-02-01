//=============================================================================
// nimcp_brain_init_core.c - Core Brain Initialization Functions
//=============================================================================
/**
 * @file nimcp_brain_init_core.c
 * @brief Core brain allocation and network creation
 *
 * WHAT: Brain structure allocation and basic initialization
 * WHY:  Separates core allocation from subsystem initialization
 * HOW:  Provides functions for brain allocation, network creation, labels, and event bus
 *
 * EXTRACTED FROM: nimcp_brain_init.c
 * DATE: 2025-12-08
 *
 * @version 2.7.0
 * @author NIMCP Development Team
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_core.h"
#include "core/brain/factory/init/nimcp_brain_init_config.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/events/nimcp_event_bus.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/exception/nimcp_exception_macros.h"

/* Forward declaration for mesh integration - include avoided to prevent type conflicts */
struct mesh_brain_integration;
typedef struct mesh_brain_integration mesh_brain_integration_t;

/* Mesh brain integration function declarations */
extern nimcp_error_t mesh_brain_integration_register_bbb(
    mesh_brain_integration_t* integration, void* bbb, void* health_agent);
extern nimcp_error_t mesh_brain_integration_register_immune_system(
    mesh_brain_integration_t* integration, void* immune, void* health_agent);
extern nimcp_error_t mesh_brain_integration_register_fep_orchestrator(
    mesh_brain_integration_t* integration, void* fep, void* health_agent);
extern nimcp_error_t mesh_brain_integration_register_working_memory(
    mesh_brain_integration_t* integration, void* wm, void* health_agent);
extern nimcp_error_t mesh_brain_integration_register_executive(
    mesh_brain_integration_t* integration, void* exec, void* health_agent);
extern nimcp_error_t mesh_brain_integration_register_global_workspace(
    mesh_brain_integration_t* integration, void* gw, void* health_agent);
extern nimcp_error_t mesh_brain_integration_register_plasticity_coordinator(
    mesh_brain_integration_t* integration, void* plasticity, void* health_agent);
extern nimcp_error_t mesh_brain_integration_register_bio_async_orchestrator(
    mesh_brain_integration_t* integration, void* bio_orch, void* health_agent);

/* Mesh brain integration stats structure */
typedef struct mesh_brain_integration_stats {
    size_t memory_modules_registered;
    size_t cognitive_modules_registered;
    size_t sensory_modules_registered;
    size_t motor_modules_registered;
    size_t security_modules_registered;
    size_t plasticity_modules_registered;
    size_t glial_modules_registered;
    size_t orchestrator_modules_registered;
    size_t total_modules_registered;
    size_t registration_failures;
    uint64_t last_registration_time_ns;
} mesh_brain_integration_stats_t;

extern nimcp_error_t mesh_brain_integration_get_stats(
    const mesh_brain_integration_t* integration,
    mesh_brain_integration_stats_t* stats);

#define LOG_MODULE "BRAIN_INIT_CORE"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for brain_init_core module */
static nimcp_health_agent_t* g_brain_init_core_health_agent = NULL;

/**
 * @brief Set health agent for brain_init_core heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void brain_init_core_set_health_agent(nimcp_health_agent_t* agent) {
    g_brain_init_core_health_agent = agent;
}

/** @brief Send heartbeat from brain_init_core module */
static inline void brain_init_core_heartbeat(const char* operation, float progress) {
    if (g_brain_init_core_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_brain_init_core_health_agent, operation, progress);
    }
}


// Compatibility macro for set_error (converts to LOG_ERROR)
#ifndef set_error
#define set_error(msg) LOG_ERROR(LOG_MODULE, "%s", msg)
#endif

//=============================================================================
// Brain Allocation
//=============================================================================

brain_t nimcp_brain_factory_allocate_brain(void)
{
    brain_t brain = nimcp_calloc(1, sizeof(struct brain_struct));
    if (!brain) {
        set_error("Failed to allocate brain structure");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;
    }

    brain->last_input = NULL;
    brain->cached_decision = NULL;
    brain->input_size = 0;

    // Initialize cache mutex for thread-safe access
    if (nimcp_platform_mutex_init(&brain->cache_mutex, false) != 0) {
        set_error("Failed to initialize cache mutex");
        nimcp_free(brain);
        return NULL;
    }

    brain->distributed = NULL;  // Initialize as standalone brain

    // Phase 11: Initialize long-term memory consolidation buffer
    brain->longterm_capacity = 100;  // Store up to 100 consolidated memories
    brain->longterm_count = 0;
    brain->longterm_memory = nimcp_calloc(brain->longterm_capacity,
                                          sizeof(*brain->longterm_memory));
    // Guard: If allocation fails, set capacity to 0 (consolidation will be disabled)
    if (!brain->longterm_memory) {
        brain->longterm_capacity = 0;
    }

    // Initialize COW fields
    brain->is_cow_clone = false;
    brain->owns_network = true;  // By default, brain owns its network
    brain->original_network = NULL;
    brain->network_is_cached = false;

    // Phase 3: Initialize reference counting fields
    brain->network_refcount_atomic = NULL;
    brain->can_use_readonly = false;

    // Community Detection: Initialize fields
    brain->functional_modules = NULL;
    brain->network_hubs = NULL;
    brain->topology_metrics = NULL;
    brain->auto_detect_communities = false;
    brain->community_detection_interval = 0.0F;  // Manual only by default

    return brain;
}

//=============================================================================
// Network Creation
//=============================================================================

adaptive_network_t nimcp_brain_factory_create_brain_network(uint32_t num_inputs, uint32_t num_outputs,
                                               uint32_t num_neurons, float sparsity_target,
                                               ode_integration_method_t integration_method)
{
    adaptive_network_config_t net_config =
        nimcp_brain_factory_build_network_config(num_inputs, num_outputs, num_neurons, sparsity_target, integration_method);

    // Guard: Check if layer_sizes allocation failed in build_base_network_config
    if (!net_config.base_config.layer_sizes) {
        return NULL;
    }

    adaptive_network_t network = adaptive_network_create(&net_config);

    // Free our copy of layer_sizes - adaptive_network_create makes its own deep copy
    if (net_config.base_config.layer_sizes) {
        nimcp_free((void*)net_config.base_config.layer_sizes);
    }

    return network;
}

//=============================================================================
// Output Labels
//=============================================================================

bool nimcp_brain_factory_init_output_labels(brain_t brain, uint32_t num_outputs)
{
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "nimcp_brain_factory_init_output_labels: brain is NULL");
        return false;
    }
    if (num_outputs == 0) {
        // Zero outputs - no allocation needed, but set to NULL
        brain->output_labels = NULL;
        brain->num_output_labels = 0;
        return true;
    }
    brain->output_labels = nimcp_calloc(num_outputs, sizeof(char*));
    if (!brain->output_labels) {
        set_error("Failed to allocate output labels");
        return false;
    }
    brain->num_output_labels = 0;
    return true;
}

//=============================================================================
// Event Bus Initialization
//=============================================================================

bool nimcp_brain_factory_init_event_bus(brain_t brain)
{
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "nimcp_brain_factory_init_event_bus: brain is NULL");
        return false;
    }

    // Check if already initialized
    if (brain->event_bus) {
        return true;  // Already initialized
    }

    // Create event bus with immediate delivery (synchronous for predictability)
    brain->event_bus = event_bus_create("brain_event_bus", EVENT_DELIVERY_IMMEDIATE);
    if (!brain->event_bus) {
        set_error("Failed to create brain event bus");
        return false;
    }

    LOG_INFO("Universal event bus initialized for brain");
    return true;
}

//=============================================================================
// Mesh Network Integration (Phase 15: Brain-Mesh Integration)
//=============================================================================

/**
 * @brief Global mesh brain integration handle for brain module registration
 *
 * Set via nimcp_brain_factory_set_mesh_integration() before brain creation.
 * When set, brain modules are automatically registered with the mesh network.
 */
static mesh_brain_integration_t* g_mesh_brain_integration = NULL;

/**
 * @brief Set the mesh brain integration handle for automatic registration
 *
 * WHAT: Configures brain factory to auto-register modules with mesh
 * WHY:  Enables real brain modules to participate in mesh consensus
 * HOW:  Stores integration handle, used during brain init
 *
 * @param integration Mesh brain integration handle (NULL to disable)
 */
void nimcp_brain_factory_set_mesh_integration(mesh_brain_integration_t* integration) {
    g_mesh_brain_integration = integration;
    if (integration) {
        LOG_INFO("Mesh brain integration enabled for brain factory");
    } else {
        LOG_INFO("Mesh brain integration disabled for brain factory");
    }
}

/**
 * @brief Get the current mesh brain integration handle
 *
 * @return Current integration handle or NULL if not set
 */
mesh_brain_integration_t* nimcp_brain_factory_get_mesh_integration(void) {
    return g_mesh_brain_integration;
}

/**
 * @brief Register brain modules with mesh network
 *
 * WHAT: Registers all available brain modules with the mesh network
 * WHY:  Enables real module instances to participate in mesh consensus
 * HOW:  Iterates brain subsystems, registers non-NULL modules
 *
 * BIOLOGICAL MOTIVATION:
 * - Brain regions form a distributed consensus network
 * - Each region has specialized receptive fields (what it responds to)
 * - Thalamus acts as gateway, amygdala has veto power
 * - Hippocampus is required for memory transactions
 *
 * @param brain Brain instance with initialized modules
 * @return true if registration successful, false on error
 */
bool nimcp_brain_factory_register_with_mesh(brain_t brain)
{
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "nimcp_brain_factory_register_with_mesh: brain is NULL");
        return false;
    }

    mesh_brain_integration_t* integration = g_mesh_brain_integration;
    if (!integration) {
        /* Mesh integration not configured - not an error */
        return true;
    }

    brain_init_core_heartbeat("mesh_registration", 0.0f);

    LOG_INFO("Registering brain modules with mesh network...");

    size_t registered = 0;
    size_t failed = 0;

    /* Security modules (highest priority - must be registered first) */
    if (brain->bbb_system) {
        if (mesh_brain_integration_register_bbb(
                integration, brain->bbb_system, NULL) == NIMCP_SUCCESS) {
            registered++;
        } else {
            failed++;
        }
    }

    if (brain->immune_system) {
        if (mesh_brain_integration_register_immune_system(
                integration, brain->immune_system, NULL) == NIMCP_SUCCESS) {
            registered++;
        } else {
            failed++;
        }
    }

    brain_init_core_heartbeat("mesh_registration", 0.2f);

    /* Core cognitive modules */
    if (brain->fep_orchestrator) {
        if (mesh_brain_integration_register_fep_orchestrator(
                integration, brain->fep_orchestrator, NULL) == NIMCP_SUCCESS) {
            registered++;
        } else {
            failed++;
        }
    }

    if (brain->working_memory) {
        if (mesh_brain_integration_register_working_memory(
                integration, brain->working_memory, NULL) == NIMCP_SUCCESS) {
            registered++;
        } else {
            failed++;
        }
    }

    if (brain->executive) {
        if (mesh_brain_integration_register_executive(
                integration, brain->executive, NULL) == NIMCP_SUCCESS) {
            registered++;
        } else {
            failed++;
        }
    }

    brain_init_core_heartbeat("mesh_registration", 0.4f);

    /* Global workspace for conscious access */
    if (brain->global_workspace) {
        if (mesh_brain_integration_register_global_workspace(
                integration, brain->global_workspace, NULL) == NIMCP_SUCCESS) {
            registered++;
        } else {
            failed++;
        }
    }

    brain_init_core_heartbeat("mesh_registration", 0.6f);

    /* Plasticity coordinator */
    if (brain->plasticity_coordinator) {
        if (mesh_brain_integration_register_plasticity_coordinator(
                integration, brain->plasticity_coordinator, NULL) == NIMCP_SUCCESS) {
            registered++;
        } else {
            failed++;
        }
    }

    /* Bio-async orchestrator */
    if (brain->bio_async_orchestrator) {
        if (mesh_brain_integration_register_bio_async_orchestrator(
                integration, brain->bio_async_orchestrator, NULL) == NIMCP_SUCCESS) {
            registered++;
        } else {
            failed++;
        }
    }

    brain_init_core_heartbeat("mesh_registration", 0.8f);

    /* Get statistics */
    mesh_brain_integration_stats_t stats;
    if (mesh_brain_integration_get_stats(integration, &stats) == NIMCP_SUCCESS) {
        LOG_INFO("Mesh registration complete: %zu total (%zu memory, %zu cognitive, "
                 "%zu security, %zu plasticity), %zu failed",
                 stats.total_modules_registered,
                 stats.memory_modules_registered,
                 stats.cognitive_modules_registered,
                 stats.security_modules_registered,
                 stats.plasticity_modules_registered,
                 failed);
    } else {
        LOG_INFO("Mesh registration complete: %zu registered, %zu failed",
                 registered, failed);
    }

    brain_init_core_heartbeat("mesh_registration", 1.0f);

    return (failed == 0);
}
