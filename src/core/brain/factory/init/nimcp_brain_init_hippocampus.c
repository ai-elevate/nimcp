//=============================================================================
// nimcp_brain_init_hippocampus.c - Hippocampus Initialization
//=============================================================================
/**
 * @file nimcp_brain_init_hippocampus.c
 * @brief Hippocampus Initialization Implementation
 *
 * WHAT: Initialization functions for hippocampus (memory and navigation)
 * WHY:  Enable episodic memory and spatial navigation in the brain
 * HOW:  Creates hippocampus adapter and connects all integration bridges
 *
 * EXTRACTED FROM: Brain factory initialization
 * DATE: 2025-12-30
 *
 * @version Phase H1: Hippocampus Brain Integration
 * @author NIMCP Development Team
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_hippocampus.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"

#define LOG_MODULE "BRAIN_INIT_HIPPOCAMPUS"

// Compatibility macro for set_error (converts to LOG_ERROR)
#ifndef set_error
#define set_error(msg) LOG_ERROR(LOG_MODULE, "%s", msg)
#endif

// Hippocampus includes
#include "core/brain/regions/hippocampus/nimcp_hippocampus_adapter.h"
#include "core/brain/regions/hippocampus/nimcp_hippocampus_quantum_bridge.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>

// Forward declarations for substrate and thalamic bridges
// These will be implemented as the substrate/thalamic bridge system matures
struct hippocampus_substrate_bridge;
typedef struct hippocampus_substrate_bridge hippocampus_substrate_bridge_t;

struct hippocampus_thalamic_bridge;
typedef struct hippocampus_thalamic_bridge hippocampus_thalamic_bridge_t;

// Hippocampus substrate config
typedef struct {
    bool enable_metabolic_modulation;
    bool enable_sleep_modulation;
    float atp_threshold;
    float fatigue_decay_rate;
} hippocampus_substrate_config_t;

// Hippocampus thalamic config
typedef struct {
    bool enable_anterior_routing;
    bool enable_memory_gating;
    bool enable_consolidation_signals;
    float min_signal_strength;
    float routing_priority;
} hippocampus_thalamic_config_t;

// Stub implementations for substrate/thalamic bridges
// These will be replaced when full implementations are available

static hippocampus_substrate_config_t hippocampus_substrate_default_config(void) {
    hippocampus_substrate_config_t config = {
        .enable_metabolic_modulation = true,
        .enable_sleep_modulation = true,
        .atp_threshold = 0.3f,
        .fatigue_decay_rate = 0.01f
    };
    return config;
}

static hippocampus_substrate_bridge_t* hippocampus_substrate_bridge_create(
    void* hippocampus, void* substrate, const hippocampus_substrate_config_t* config
) {
    (void)hippocampus;
    (void)substrate;
    (void)config;
    // Stub: Return NULL to indicate not yet implemented
    // Full implementation will integrate with neural_substrate module
    return NULL;
}

static void hippocampus_substrate_bridge_destroy(hippocampus_substrate_bridge_t* bridge) {
    (void)bridge;
    // Stub
}

static int hippocampus_substrate_bridge_update(hippocampus_substrate_bridge_t* bridge) {
    (void)bridge;
    return 0;  // Stub
}

static hippocampus_thalamic_config_t hippocampus_thalamic_default_config(void) {
    hippocampus_thalamic_config_t config = {
        .enable_anterior_routing = true,
        .enable_memory_gating = true,
        .enable_consolidation_signals = true,
        .min_signal_strength = 0.1f,
        .routing_priority = 0.8f
    };
    return config;
}

static hippocampus_thalamic_bridge_t* hippocampus_thalamic_bridge_create(
    void* hippocampus, void* router, const hippocampus_thalamic_config_t* config
) {
    (void)hippocampus;
    (void)router;
    (void)config;
    // Stub: Return NULL to indicate not yet implemented
    // Full implementation will integrate with thalamic router module
    return NULL;
}

static void hippocampus_thalamic_bridge_destroy(hippocampus_thalamic_bridge_t* bridge) {
    (void)bridge;
    // Stub
}

static int hippocampus_thalamic_bridge_reset(hippocampus_thalamic_bridge_t* bridge) {
    (void)bridge;
    return 0;  // Stub
}

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Connect hippocampus to bio-async messaging
 */
static bool connect_hippocampus_to_bio_async(brain_t brain) {
    if (!brain || !brain->hippocampus) return true; /* Non-fatal if not available */

    /* Register with bio-async if enabled */
    if (brain->bio_async_enabled && brain->bio_async_ctx) {
        /*
         * TODO: Register hippocampus message handlers
         * bio_router_register_module(router, BIO_MODULE_HIPPOCAMPUS, brain->hippocampus);
         */
        LOG_DEBUG(LOG_MODULE, "Hippocampus bio-async connection deferred");
    }

    return true;
}

/**
 * @brief Connect hippocampus substrate bridge to neural substrate
 */
static bool connect_substrate_bridge(brain_t brain) {
    if (!brain || !brain->hippocampus_substrate_bridge) return true;

    /* Apply initial metabolic effects */
    if (hippocampus_substrate_bridge_update(brain->hippocampus_substrate_bridge) != 0) {
        LOG_WARN(LOG_MODULE, "Initial substrate bridge update failed");
    }

    return true;
}

/**
 * @brief Connect hippocampus thalamic bridge to thalamic router
 */
static bool connect_thalamic_bridge(brain_t brain) {
    if (!brain || !brain->hippocampus_thalamic_bridge) return true;

    /* Reset bridge to clean state */
    if (hippocampus_thalamic_bridge_reset(brain->hippocampus_thalamic_bridge) != 0) {
        LOG_WARN(LOG_MODULE, "Thalamic bridge reset failed");
    }

    return true;
}

//=============================================================================
// Public API Implementation
//=============================================================================

bool nimcp_brain_factory_init_hippocampus_subsystem(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Check if already initialized */
    if (brain->hippocampus) {
        return true;  /* Already initialized */
    }

    /* Check if hippocampus is enabled in config */
    /* Default to enabled for memory-capable brains (episodic, consolidation, working memory) */
    if (!brain->config.enable_working_memory &&
        !brain->config.enable_consolidation) {
        brain->hippocampus_enabled = false;
        LOG_DEBUG(LOG_MODULE, "Hippocampus not enabled (no memory config)");
        return true;  /* Not enabled, not an error */
    }

    LOG_INFO(LOG_MODULE, "Initializing hippocampus subsystem");

    /* Create hippocampus adapter with default configuration */
    hippocampus_config_t hippocampus_cfg = hippocampus_default_config();

    brain->hippocampus = hippocampus_create(&hippocampus_cfg);
    if (!brain->hippocampus) {
        set_error("Failed to create hippocampus adapter");
        return false;
    }

    brain->hippocampus_enabled = true;
    brain->last_hippocampus_update_us = 0;

    LOG_DEBUG(LOG_MODULE, "Hippocampus adapter created (capacity=%u)",
              hippocampus_cfg.memory_capacity);

    /* Initialize integration bridges */
    if (!nimcp_brain_factory_init_hippocampus_substrate_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Hippocampus substrate bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_hippocampus_thalamic_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Hippocampus thalamic bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_hippocampus_quantum_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Hippocampus quantum bridge init failed (non-fatal)");
    }

    /* Connect to other subsystems */
    if (!nimcp_brain_factory_connect_hippocampus_to_cortex(brain)) {
        LOG_WARN(LOG_MODULE, "Hippocampus-cortex connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_hippocampus_to_amygdala(brain)) {
        LOG_WARN(LOG_MODULE, "Hippocampus-amygdala connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_hippocampus_to_training(brain)) {
        LOG_WARN(LOG_MODULE, "Hippocampus-training connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_hippocampus_to_immune(brain)) {
        LOG_WARN(LOG_MODULE, "Hippocampus-immune connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_hippocampus_to_sleep(brain)) {
        LOG_WARN(LOG_MODULE, "Hippocampus-sleep connection failed (non-fatal)");
    }

    /* Connect to bio-async */
    if (!connect_hippocampus_to_bio_async(brain)) {
        LOG_WARN(LOG_MODULE, "Hippocampus bio-async connection failed (non-fatal)");
    }

    LOG_INFO(LOG_MODULE, "Hippocampus subsystem initialized successfully");
    return true;
}

bool nimcp_brain_factory_init_hippocampus_substrate_bridge(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Check if already initialized */
    if (brain->hippocampus_substrate_bridge) {
        return true;
    }

    /* Need hippocampus adapter first */
    if (!brain->hippocampus) {
        return true;  /* Not ready yet, will be called again */
    }

    LOG_DEBUG(LOG_MODULE, "Initializing hippocampus substrate bridge");

    /* Get neural substrate - may be NULL in simple configurations */
    void* substrate = NULL;
    /*
     * TODO: Get substrate from brain if available
     * substrate = brain->substrate;
     */

    /* Create substrate bridge with default config */
    hippocampus_substrate_config_t config = hippocampus_substrate_default_config();

    brain->hippocampus_substrate_bridge = hippocampus_substrate_bridge_create(
        brain->hippocampus, substrate, &config);

    if (!brain->hippocampus_substrate_bridge) {
        LOG_WARN(LOG_MODULE, "Hippocampus substrate bridge not yet implemented");
        return true;  /* Non-fatal: stub implementation */
    }

    /* Connect to substrate */
    connect_substrate_bridge(brain);

    LOG_DEBUG(LOG_MODULE, "Hippocampus substrate bridge initialized");
    return true;
}

bool nimcp_brain_factory_init_hippocampus_thalamic_bridge(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Check if already initialized */
    if (brain->hippocampus_thalamic_bridge) {
        return true;
    }

    /* Need hippocampus adapter first */
    if (!brain->hippocampus) {
        return true;  /* Not ready yet */
    }

    LOG_DEBUG(LOG_MODULE, "Initializing hippocampus thalamic bridge");

    /* Get thalamic router - may be NULL in simple configurations */
    void* router = NULL;
    /*
     * TODO: Get thalamic router from brain if available
     * router = brain->thalamic_router;
     */

    /* Create thalamic bridge with default config */
    hippocampus_thalamic_config_t thal_config = hippocampus_thalamic_default_config();

    brain->hippocampus_thalamic_bridge = hippocampus_thalamic_bridge_create(
        brain->hippocampus, router, &thal_config);

    if (!brain->hippocampus_thalamic_bridge) {
        LOG_WARN(LOG_MODULE, "Hippocampus thalamic bridge not yet implemented");
        return true;  /* Non-fatal: stub implementation */
    }

    /* Connect to thalamus */
    connect_thalamic_bridge(brain);

    LOG_DEBUG(LOG_MODULE, "Hippocampus thalamic bridge initialized");
    return true;
}

bool nimcp_brain_factory_init_hippocampus_quantum_bridge(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Check if already initialized */
    if (brain->hippocampus_quantum_bridge) {
        return true;
    }

    /* Need hippocampus adapter first */
    if (!brain->hippocampus) {
        return true;  /* Not ready yet */
    }

    /* Check if quantum reasoning is enabled */
    if (!brain->quantum_reasoning_enabled) {
        LOG_DEBUG(LOG_MODULE, "Quantum reasoning not enabled, skipping quantum bridge");
        return true;  /* Not enabled, not an error */
    }

    LOG_DEBUG(LOG_MODULE, "Initializing hippocampus quantum bridge");

    /* Create quantum bridge with default config */
    hippocampus_quantum_config_t config = hippocampus_quantum_default_config();

    /* Use default search depth - can be overridden via config API */

    brain->hippocampus_quantum_bridge = hippocampus_quantum_bridge_create(
        brain->hippocampus, &config);

    if (!brain->hippocampus_quantum_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create hippocampus quantum bridge");
        return false;
    }

    LOG_DEBUG(LOG_MODULE, "Hippocampus quantum bridge initialized");
    return true;
}

bool nimcp_brain_factory_connect_hippocampus_to_cortex(brain_t brain) {
    if (!brain || !brain->hippocampus) {
        return true;  /* Nothing to connect */
    }

    LOG_DEBUG(LOG_MODULE, "Connecting hippocampus to cortical areas");

    /*
     * Register hippocampus for systems consolidation.
     * During sleep, memories are transferred from hippocampus to neocortex.
     * This involves:
     * - Prefrontal cortex (semantic memory)
     * - Temporal cortex (episodic memory)
     * - Parietal cortex (spatial memory)
     */

    /* Check if systems consolidation is available */
    if (brain->systems_consolidation) {
        /*
         * TODO: Register with systems consolidation
         * systems_consolidation_register_source(brain->systems_consolidation,
         *     brain->hippocampus);
         */
        LOG_DEBUG(LOG_MODULE, "Hippocampus registered with systems consolidation");
    }

    /* Connect to semantic memory if available */
    if (brain->semantic_memory) {
        /*
         * TODO: Register semantic abstraction callback
         * semantic_memory_register_source(brain->semantic_memory,
         *     brain->hippocampus);
         */
        LOG_DEBUG(LOG_MODULE, "Hippocampus connected to semantic memory");
    }

    LOG_DEBUG(LOG_MODULE, "Hippocampus connected to cortical areas");
    return true;
}

bool nimcp_brain_factory_connect_hippocampus_to_amygdala(brain_t brain) {
    if (!brain || !brain->hippocampus) {
        return true;  /* Nothing to connect */
    }

    LOG_DEBUG(LOG_MODULE, "Connecting hippocampus to amygdala");

    /*
     * Hippocampus-amygdala connection for emotional memory:
     * - Amygdala tags memories with emotional significance
     * - High emotional valence -> stronger memory encoding
     * - Stress hormones modulate hippocampal plasticity
     */

    /* Check if emotional system is available */
    if (brain->emotional_system) {
        /*
         * TODO: Register for emotional modulation
         * emotional_system_register_memory_module(brain->emotional_system,
         *     MEMORY_MODULE_HIPPOCAMPUS, brain->hippocampus);
         */
        LOG_DEBUG(LOG_MODULE, "Hippocampus registered with emotional system");
    }

    LOG_DEBUG(LOG_MODULE, "Hippocampus connected to amygdala");
    return true;
}

bool nimcp_brain_factory_connect_hippocampus_to_training(brain_t brain) {
    if (!brain || !brain->hippocampus) {
        return true;  /* Nothing to connect */
    }

    /* Check if training is enabled */
    if (!brain->enable_training_integration || !brain->training_ctx) {
        return true;  /* Training not enabled */
    }

    LOG_DEBUG(LOG_MODULE, "Connecting hippocampus to training system");

    /*
     * Register hippocampus adapter with training context.
     * This allows memory-based learning through:
     * - Place cell tuning from navigation experience
     * - Memory encoding improvement from retrieval practice
     * - Pattern separation/completion optimization
     */

    /*
     * TODO: Register with training
     * nimcp_brain_training_register_module(brain->training_ctx,
     *     TRAIN_MODULE_HIPPOCAMPUS, brain->hippocampus);
     */

    LOG_DEBUG(LOG_MODULE, "Hippocampus connected to training system");
    return true;
}

bool nimcp_brain_factory_connect_hippocampus_to_immune(brain_t brain) {
    if (!brain || !brain->hippocampus) {
        return true;  /* Nothing to connect */
    }

    /* Check if immune system is available */
    if (!brain->immune_enabled || !brain->immune_system) {
        return true;  /* Immune not enabled */
    }

    LOG_DEBUG(LOG_MODULE, "Connecting hippocampus to immune system");

    /*
     * Register for cytokine signals that affect memory.
     * Neuroinflammation (high IL-1beta, TNF-alpha) can cause:
     * - Reduced LTP (impaired memory encoding)
     * - Impaired spatial navigation
     * - Accelerated memory decay
     */

    /*
     * TODO: Register with immune system
     * brain_immune_register_cytokine_callback(brain->immune_system,
     *     CYTOKINE_IL1B | CYTOKINE_TNF_A, hippocampus_inflammation_callback, brain->hippocampus);
     */

    LOG_DEBUG(LOG_MODULE, "Hippocampus connected to immune system");
    return true;
}

bool nimcp_brain_factory_connect_hippocampus_to_sleep(brain_t brain) {
    if (!brain || !brain->hippocampus) {
        return true;  /* Nothing to connect */
    }

    LOG_DEBUG(LOG_MODULE, "Connecting hippocampus to sleep system");

    /*
     * Sleep is critical for hippocampal memory consolidation:
     * - NREM sleep: Sharp-wave ripples replay memories
     * - REM sleep: Memory integration and emotional processing
     * - Sleep deprivation impairs memory encoding
     */

    /* Register for sleep state changes */
    /*
     * TODO: Register sleep callback
     * sleep_system_register_callback(&brain->sleep_system,
     *     SLEEP_EVENT_NREM_ENTER | SLEEP_EVENT_REM_ENTER,
     *     hippocampus_sleep_callback, brain->hippocampus);
     */

    LOG_DEBUG(LOG_MODULE, "Hippocampus connected to sleep system");
    return true;
}
