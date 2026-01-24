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

#define LOG_MODULE "BRAIN_INIT_CORE"

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
