//=============================================================================
// nimcp_brain_init_broca.c - Broca's Region Initialization
//=============================================================================
/**
 * @file nimcp_brain_init_broca.c
 * @brief Broca's Region Initialization Implementation
 *
 * WHAT: Initialization functions for Broca's region (language production)
 * WHY:  Enable language production capabilities in the brain
 * HOW:  Creates Broca adapter and connects all integration bridges
 *
 * EXTRACTED FROM: Brain factory initialization
 * DATE: 2025-12-30
 *
 * @version Phase B3: Broca Full Brain Integration
 * @author NIMCP Development Team
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_broca.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"

#define LOG_MODULE "BRAIN_INIT_BROCA"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_init_broca)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_brain_init_broca_mesh_id = 0;
static mesh_participant_registry_t* g_brain_init_broca_mesh_registry = NULL;

nimcp_error_t brain_init_broca_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_brain_init_broca_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "brain_init_broca", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "brain_init_broca";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_brain_init_broca_mesh_id);
    if (err == NIMCP_SUCCESS) g_brain_init_broca_mesh_registry = registry;
    return err;
}

void brain_init_broca_mesh_unregister(void) {
    if (g_brain_init_broca_mesh_registry && g_brain_init_broca_mesh_id != 0) {
        mesh_participant_unregister(g_brain_init_broca_mesh_registry, g_brain_init_broca_mesh_id);
        g_brain_init_broca_mesh_id = 0;
        g_brain_init_broca_mesh_registry = NULL;
    }
}


// Compatibility macro for set_error (converts to LOG_ERROR)
#ifndef set_error
#define set_error(msg) LOG_ERROR(LOG_MODULE, "%s", msg)
#endif

// Broca's region includes
// NOTE: Include broca_adapter.h first as brain_internal.h doesn't include it
#include "core/brain/regions/broca/nimcp_broca_adapter.h"
#include "core/brain/regions/broca/nimcp_broca_substrate_bridge.h"
// NOTE: Avoid including thalamic_bridge.h directly due to type conflicts
// We use forward declarations instead
#include "core/brain/regions/broca/nimcp_broca_quantum_bridge.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>

// Forward declarations and types to avoid header conflicts with thalamic router
struct broca_thalamic_bridge;
typedef struct broca_thalamic_bridge broca_thalamic_bridge_t;

// Broca thalamic config (duplicated to avoid header conflicts)
typedef struct {
    bool enable_attention_gating;
    bool enable_motor_priority;
    bool enable_syntax_routing;
    float min_urgency_threshold;
    float motor_boost;
    float attention_decay_rate;
} broca_thalamic_config_t;

// Thalamic bridge API declarations (to avoid header conflicts)
extern broca_thalamic_config_t broca_thalamic_default_config(void);
extern broca_thalamic_bridge_t* broca_thalamic_bridge_create(
    void* broca, void* router, const broca_thalamic_config_t* config);
extern void broca_thalamic_bridge_destroy(broca_thalamic_bridge_t* bridge);
extern int broca_thalamic_bridge_reset(broca_thalamic_bridge_t* bridge);

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Connect Broca to bio-async messaging
 */
static bool connect_broca_to_bio_async(brain_t brain) {
    if (!brain || !brain->broca) return true; /* Non-fatal if not available */

    /* Register with bio-async if enabled */
    if (brain->bio_async_enabled && brain->bio_async_ctx) {
        /*
         * TODO: Register Broca message handlers
         * bio_router_register_module(router, BIO_MODULE_BROCA, brain->broca);
         */
    }

    return true;
}

/**
 * @brief Connect Broca substrate bridge to neural substrate
 */
static bool connect_substrate_bridge(brain_t brain) {
    if (!brain || !brain->broca_substrate_bridge) return true;

    /* Apply initial metabolic effects */
    if (broca_substrate_bridge_update(brain->broca_substrate_bridge) != 0) {
        LOG_WARN(LOG_MODULE, "Initial substrate bridge update failed");
    }

    return true;
}

/**
 * @brief Connect Broca thalamic bridge to thalamic router
 */
static bool connect_thalamic_bridge(brain_t brain) {
    if (!brain || !brain->broca_thalamic_bridge) return true;

    /* Reset bridge to clean state */
    if (broca_thalamic_bridge_reset(brain->broca_thalamic_bridge) != 0) {
        LOG_WARN(LOG_MODULE, "Thalamic bridge reset failed");
    }

    return true;
}

//=============================================================================
// Public API Implementation
//=============================================================================

bool nimcp_brain_factory_init_broca_subsystem(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_broca_subsystem: brain is NULL");

            return false;
    }

    /* Check if already initialized */
    if (brain->broca) {
        return true;  /* Already initialized */
    }

    /* Check if Broca is enabled in config */
    /* Note: Default to enabled for language-capable brains */
    if (!brain->config.enable_speech_cortex &&
        !brain->config.enable_multimodal_integration) {
        brain->broca_enabled = false;
        return true;  /* Not enabled, not an error */
    }

    /* Create Broca adapter with default configuration */
    broca_config_t broca_cfg = broca_default_config();

    /* Scale configuration based on brain config */
    if (brain->config.working_memory_capacity > 0) {
        broca_cfg.working_memory_slots = brain->config.working_memory_capacity;
    }

    brain->broca = broca_create(&broca_cfg);
    if (!brain->broca) {
        set_error("Failed to create Broca adapter");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_broca_subsystem: brain->broca is NULL");
        return false;
    }

    brain->broca_enabled = true;
    brain->last_broca_update_us = 0;

    /* Initialize integration bridges */
    if (!nimcp_brain_factory_init_broca_substrate_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Broca substrate bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_broca_thalamic_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Broca thalamic bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_broca_quantum_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Broca quantum bridge init failed (non-fatal)");
    }

    /* Connect to other subsystems */
    if (!nimcp_brain_factory_connect_broca_to_working_memory(brain)) {
        LOG_WARN(LOG_MODULE, "Broca-WM connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_broca_to_training(brain)) {
        LOG_WARN(LOG_MODULE, "Broca-Training connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_broca_to_immune(brain)) {
        LOG_WARN(LOG_MODULE, "Broca-Immune connection failed (non-fatal)");
    }

    /* Connect to bio-async */
    if (!connect_broca_to_bio_async(brain)) {
        LOG_WARN(LOG_MODULE, "Broca bio-async connection failed (non-fatal)");
    }

    LOG_INFO(LOG_MODULE, "Broca's region initialized successfully");
    return true;
}

bool nimcp_brain_factory_init_broca_substrate_bridge(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_broca_substrate_bridge: brain is NULL");

            return false;
    }

    /* Check if already initialized */
    if (brain->broca_substrate_bridge) {
        return true;
    }

    /* Need Broca adapter first */
    if (!brain->broca) {
        return true;  /* Not ready yet, will be called again */
    }

    /* Get neural substrate - may be NULL in simple configurations */
    neural_substrate_t* substrate = NULL;
    /*
     * TODO: Get substrate from brain if available
     * substrate = brain->substrate;
     */

    /* Create substrate bridge with default config */
    broca_substrate_config_t config = broca_substrate_default_config();

    brain->broca_substrate_bridge = broca_substrate_bridge_create(
        brain->broca, substrate, &config);

    if (!brain->broca_substrate_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create Broca substrate bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_broca_substrate_bridge: brain->broca_substrate_bridge is NULL");
        return false;
    }

    /* Connect to substrate */
    connect_substrate_bridge(brain);

    LOG_DEBUG(LOG_MODULE, "Broca substrate bridge initialized");
    return true;
}

bool nimcp_brain_factory_init_broca_thalamic_bridge(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_broca_thalamic_bridge: brain is NULL");

            return false;
    }

    /* Check if already initialized */
    if (brain->broca_thalamic_bridge) {
        return true;
    }

    /* Need Broca adapter first */
    if (!brain->broca) {
        return true;  /* Not ready yet */
    }

    /* Get thalamic router - may be NULL in simple configurations */
    void* router = NULL;
    /*
     * TODO: Get thalamic router from brain if available
     * router = brain->thalamic_router;
     */

    /* Create thalamic bridge with default config */
    broca_thalamic_config_t thal_config = broca_thalamic_default_config();

    brain->broca_thalamic_bridge = broca_thalamic_bridge_create(
        brain->broca, router, &thal_config);

    if (!brain->broca_thalamic_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create Broca thalamic bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_broca_thalamic_bridge: brain->broca_thalamic_bridge is NULL");
        return false;
    }

    /* Connect to thalamus */
    connect_thalamic_bridge(brain);

    LOG_DEBUG(LOG_MODULE, "Broca thalamic bridge initialized");
    return true;
}

bool nimcp_brain_factory_init_broca_quantum_bridge(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_broca_quantum_bridge: brain is NULL");

            return false;
    }

    /* Check if already initialized */
    if (brain->broca_quantum_bridge) {
        return true;
    }

    /* Need Broca adapter first */
    if (!brain->broca) {
        return true;  /* Not ready yet */
    }

    /* Check if quantum reasoning is enabled */
    if (!brain->quantum_reasoning_enabled) {
        return true;  /* Not enabled, not an error */
    }

    /* Create quantum bridge with default config */
    broca_quantum_config_t config = broca_quantum_default_config();

    /* Scale Grover iterations based on vocabulary size */
    if (brain->config.num_outputs > 100) {
        config.lexicon_search_depth = brain->config.num_outputs;
    }

    brain->broca_quantum_bridge = broca_quantum_bridge_create(
        brain->broca, &config);

    if (!brain->broca_quantum_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create Broca quantum bridge");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_broca_quantum_bridge: brain->broca_quantum_bridge is NULL");
        return false;
    }

    LOG_DEBUG(LOG_MODULE, "Broca quantum bridge initialized");
    return true;
}

bool nimcp_brain_factory_connect_broca_to_working_memory(brain_t brain) {
    if (!brain || !brain->broca) {
        return true;  /* Nothing to connect */
    }

    /* Check if working memory is available */
    if (!brain->working_memory) {
        return true;  /* WM not initialized yet */
    }

    /*
     * Register Broca as a working memory consumer for lexical access.
     * This allows Broca to maintain active word candidates during
     * speech production.
     */

    /* TODO: Register with working memory
     * working_memory_register_consumer(brain->working_memory,
     *     WM_CONSUMER_BROCA, brain->broca);
     */

    LOG_DEBUG(LOG_MODULE, "Broca connected to working memory");
    return true;
}

bool nimcp_brain_factory_connect_broca_to_training(brain_t brain) {
    if (!brain || !brain->broca) {
        return true;  /* Nothing to connect */
    }

    /* Check if training is enabled */
    if (!brain->enable_training_integration || !brain->training_ctx) {
        return true;  /* Training not enabled */
    }

    /*
     * Register Broca adapter with training context.
     * This allows language production learning through:
     * - Syntax error correction
     * - Phonological improvement
     * - Fluency optimization
     */

    /* TODO: Register with training
     * nimcp_brain_training_register_module(brain->training_ctx,
     *     TRAIN_MODULE_BROCA, brain->broca);
     */

    LOG_DEBUG(LOG_MODULE, "Broca connected to training system");
    return true;
}

bool nimcp_brain_factory_connect_broca_to_immune(brain_t brain) {
    if (!brain || !brain->broca) {
        return true;  /* Nothing to connect */
    }

    /* Check if immune system is available */
    if (!brain->immune_enabled || !brain->immune_system) {
        return true;  /* Immune not enabled */
    }

    /*
     * Register for cytokine signals that affect language production.
     * Neuroinflammation (high IL-1β, TNF-α) can cause:
     * - Reduced fluency (slower word retrieval)
     * - Simplified syntax (reduced working memory)
     * - Articulation difficulties
     */

    /* TODO: Register with immune system
     * brain_immune_register_cytokine_callback(brain->immune_system,
     *     CYTOKINE_IL1B | CYTOKINE_TNF_A, broca_inflammation_callback, brain->broca);
     */

    LOG_DEBUG(LOG_MODULE, "Broca connected to immune system");
    return true;
}

//=============================================================================
// Destruction
//=============================================================================

/**
 * @brief Destroy Broca's area subsystem
 *
 * WHAT: Clean up all Broca resources and bridges
 * WHY:  Prevent memory leaks during brain destruction
 * HOW:  Destroy in reverse initialization order
 *
 * @param brain Brain instance
 */
void nimcp_brain_factory_destroy_broca_subsystem(brain_t brain) {
    if (!brain) return;

    LOG_DEBUG(LOG_MODULE, "Destroying Broca's area subsystem");

    /* Destroy quantum bridge first (depends on broca) */
    if (brain->broca_quantum_bridge) {
        broca_quantum_bridge_destroy(brain->broca_quantum_bridge);
        brain->broca_quantum_bridge = NULL;
    }

    /* Destroy thalamic bridge */
    if (brain->broca_thalamic_bridge) {
        broca_thalamic_bridge_destroy(brain->broca_thalamic_bridge);
        brain->broca_thalamic_bridge = NULL;
    }

    /* Destroy substrate bridge */
    if (brain->broca_substrate_bridge) {
        broca_substrate_bridge_destroy(brain->broca_substrate_bridge);
        brain->broca_substrate_bridge = NULL;
    }

    /* Destroy Broca adapter */
    if (brain->broca) {
        broca_destroy(brain->broca);
        brain->broca = NULL;
    }

    brain->broca_enabled = false;
    brain->last_broca_update_us = 0;

    LOG_DEBUG(LOG_MODULE, "Broca's area subsystem destroyed");
}
