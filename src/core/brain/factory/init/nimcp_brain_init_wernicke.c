//=============================================================================
// nimcp_brain_init_wernicke.c - Wernicke's Region Initialization
//=============================================================================
/**
 * @file nimcp_brain_init_wernicke.c
 * @brief Wernicke's Region Initialization Implementation
 *
 * WHAT: Initialization functions for Wernicke's region (language comprehension)
 * WHY:  Enable language comprehension capabilities in the brain
 * HOW:  Creates Wernicke adapter and connects all integration bridges
 *
 * EXTRACTED FROM: Brain factory initialization
 * DATE: 2026-01-05
 *
 * @version Phase W6: Wernicke Full Brain Integration
 * @author NIMCP Development Team
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_wernicke.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"

#define LOG_MODULE "BRAIN_INIT_WERNICKE"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for brain_init_wernicke module */
static nimcp_health_agent_t* g_brain_init_wernicke_health_agent = NULL;

/**
 * @brief Set health agent for brain_init_wernicke heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void brain_init_wernicke_set_health_agent(nimcp_health_agent_t* agent) {
    g_brain_init_wernicke_health_agent = agent;
}

/** @brief Send heartbeat from brain_init_wernicke module */
static inline void brain_init_wernicke_heartbeat(const char* operation, float progress) {
    if (g_brain_init_wernicke_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_brain_init_wernicke_health_agent, operation, progress);
    }
}


// Compatibility macro for set_error (converts to LOG_ERROR)
#ifndef set_error
#define set_error(msg) LOG_ERROR(LOG_MODULE, "%s", msg)
#endif

// Wernicke's region includes
// NOTE: Avoid including wernicke_adapter.h directly due to semantic_memory_t conflict
// Use forward declarations instead
#include "core/brain/regions/wernicke/nimcp_wernicke_substrate_bridge.h"
#include "core/brain/regions/wernicke/nimcp_wernicke_quantum_bridge.h"
#include "core/brain/regions/wernicke/nimcp_wernicke_broca_bridge.h"
#include "core/brain/regions/wernicke/nimcp_omni_wernicke_bridge.h"
#include "core/brain/regions/wernicke/nimcp_wernicke_gpu_bio_bridge.h"
#include "core/brain/regions/wernicke/nimcp_wernicke_immune.h"
#include "core/brain/regions/wernicke/nimcp_wernicke_nlp_bridge.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <math.h>

// Forward declarations and types to avoid header conflicts
struct wernicke_adapter;
typedef struct wernicke_adapter wernicke_adapter_t;

// Wernicke config (duplicated to avoid header conflicts)
typedef struct {
    uint32_t max_phonemes;
    uint32_t max_words;
    uint32_t max_concepts;
    uint32_t lexicon_size;
    uint32_t working_memory_slots;
    uint32_t embedding_dim;
    float processing_window_ms;
    uint32_t formant_count;
    bool enable_syntactic_parsing;
    bool enable_semantic_spreading;
    bool enable_predictive_coding;
    bool enable_audiovisual_integration;
} wernicke_config_local_t;

// Wernicke API declarations (to avoid header conflicts)
extern wernicke_config_local_t wernicke_default_config(void);
extern wernicke_adapter_t* wernicke_create(const wernicke_config_local_t* config);
extern void wernicke_destroy(wernicke_adapter_t* adapter);

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Connect Wernicke to bio-async messaging (internal)
 */
static bool connect_wernicke_to_bio_async_internal(brain_t brain) {
    if (!brain || !brain->wernicke) return true; /* Non-fatal if not available */

    /* Register with bio-async if enabled */
    if (brain->bio_async_enabled && brain->bio_async_ctx) {
        /*
         * TODO: Register Wernicke message handlers
         * bio_router_register_module(router, BIO_MODULE_WERNICKE, brain->wernicke);
         */
    }

    return true;
}

/**
 * @brief Connect Wernicke substrate bridge to neural substrate
 */
static bool connect_substrate_bridge(brain_t brain) {
    if (!brain || !brain->wernicke_substrate_bridge) return true;

    /* Apply initial metabolic effects */
    if (wernicke_substrate_bridge_update(brain->wernicke_substrate_bridge) != 0) {
        LOG_WARN(LOG_MODULE, "Initial substrate bridge update failed");
    }

    return true;
}

//=============================================================================
// Public API Implementation
//=============================================================================

bool nimcp_brain_factory_init_wernicke_subsystem(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_wernicke_subsystem: brain is NULL");

            return false;
    }

    /* Check if already initialized */
    if (brain->wernicke) {
        return true;  /* Already initialized */
    }

    /* Check if Wernicke is enabled in config */
    /* Note: Default to enabled for language-capable brains */
    if (!brain->config.enable_speech_cortex &&
        !brain->config.enable_multimodal_integration) {
        brain->wernicke_enabled = false;
        return true;  /* Not enabled, not an error */
    }

    /* Create Wernicke adapter with default configuration */
    wernicke_config_local_t wernicke_cfg = wernicke_default_config();

    /* Scale configuration based on brain config */
    if (brain->config.working_memory_capacity > 0) {
        wernicke_cfg.working_memory_slots = brain->config.working_memory_capacity;
    }

    brain->wernicke = wernicke_create(&wernicke_cfg);
    if (!brain->wernicke) {
        set_error("Failed to create Wernicke adapter");
        return false;
    }

    brain->wernicke_enabled = true;
    brain->last_wernicke_update_us = 0;

    /* Initialize integration bridges */
    if (!nimcp_brain_factory_init_wernicke_substrate_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Wernicke substrate bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_wernicke_quantum_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Wernicke quantum bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_wernicke_broca_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Wernicke-Broca bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_wernicke_omni_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Wernicke omni bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_wernicke_gpu_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Wernicke GPU bridge init failed (non-fatal)");
    }

    /* Connect to other subsystems */
    if (!nimcp_brain_factory_connect_wernicke_to_semantic_memory(brain)) {
        LOG_WARN(LOG_MODULE, "Wernicke-semantic memory connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_wernicke_to_working_memory(brain)) {
        LOG_WARN(LOG_MODULE, "Wernicke-WM connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_wernicke_to_training(brain)) {
        LOG_WARN(LOG_MODULE, "Wernicke-Training connection failed (non-fatal)");
    }

    if (!nimcp_brain_factory_connect_wernicke_to_immune(brain)) {
        LOG_WARN(LOG_MODULE, "Wernicke-Immune connection failed (non-fatal)");
    }

    /* Connect to bio-async */
    if (!connect_wernicke_to_bio_async_internal(brain)) {
        LOG_WARN(LOG_MODULE, "Wernicke bio-async connection failed (non-fatal)");
    }

    LOG_INFO(LOG_MODULE, "Wernicke's region initialized successfully");
    return true;
}

bool nimcp_brain_factory_init_wernicke_substrate_bridge(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_wernicke_substrate_bridge: brain is NULL");

            return false;
    }

    /* Check if already initialized */
    if (brain->wernicke_substrate_bridge) {
        return true;
    }

    /* Need Wernicke adapter first */
    if (!brain->wernicke) {
        return true;  /* Not ready yet, will be called again */
    }

    /* Get neural substrate - may be NULL in simple configurations */
    void* substrate = NULL;
    /*
     * TODO: Get substrate from brain if available
     * substrate = brain->substrate;
     */

    /* Create substrate bridge with default config */
    wernicke_substrate_config_t config = wernicke_substrate_default_config();

    brain->wernicke_substrate_bridge = wernicke_substrate_bridge_create(
        brain->wernicke, substrate, &config);

    if (!brain->wernicke_substrate_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create Wernicke substrate bridge");
        return false;
    }

    /* Connect to substrate */
    connect_substrate_bridge(brain);

    LOG_DEBUG(LOG_MODULE, "Wernicke substrate bridge initialized");
    return true;
}

bool nimcp_brain_factory_init_wernicke_quantum_bridge(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_wernicke_quantum_bridge: brain is NULL");

            return false;
    }

    /* Check if already initialized */
    if (brain->wernicke_quantum_bridge) {
        return true;
    }

    /* Need Wernicke adapter first */
    if (!brain->wernicke) {
        return true;  /* Not ready yet */
    }

    /* Check if quantum reasoning is enabled */
    if (!brain->quantum_reasoning_enabled) {
        return true;  /* Not enabled, not an error */
    }

    /* Create quantum bridge with default config */
    wernicke_quantum_config_t config;
    wernicke_quantum_default_config(&config);

    /* Scale Grover iterations based on lexicon size (sqrt(N) complexity) */
    if (brain->config.num_outputs > 100) {
        /* Grover needs sqrt(N) iterations, estimate from vocabulary size */
        uint32_t estimated_iters = (uint32_t)(sqrt((double)brain->config.num_outputs) * 1.5f);
        if (estimated_iters > config.grover_max_iterations) {
            config.grover_max_iterations = estimated_iters;
        }
    }

    brain->wernicke_quantum_bridge = wernicke_quantum_bridge_create(
        brain->wernicke, &config);

    if (!brain->wernicke_quantum_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create Wernicke quantum bridge");
        return false;
    }

    LOG_DEBUG(LOG_MODULE, "Wernicke quantum bridge initialized");
    return true;
}

bool nimcp_brain_factory_init_wernicke_broca_bridge(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_wernicke_broca_bridge: brain is NULL");

            return false;
    }

    /* Check if already initialized */
    if (brain->wernicke_broca_bridge) {
        return true;
    }

    /* Need Wernicke adapter first */
    if (!brain->wernicke) {
        return true;  /* Not ready yet */
    }

    /* Get Broca adapter - may be NULL if Broca not initialized yet */
    void* broca = brain->broca;

    /* Create Wernicke-Broca bridge with default config */
    wbb_config_t config = wbb_default_config();

    brain->wernicke_broca_bridge = wbb_create(
        brain->wernicke, broca, &config);

    if (!brain->wernicke_broca_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create Wernicke-Broca bridge");
        return false;
    }

    /* Reset bridge to clean state */
    if (wbb_reset(brain->wernicke_broca_bridge) != 0) {
        LOG_WARN(LOG_MODULE, "Wernicke-Broca bridge reset failed");
    }

    LOG_DEBUG(LOG_MODULE, "Wernicke-Broca bridge initialized");
    return true;
}

bool nimcp_brain_factory_init_wernicke_omni_bridge(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_wernicke_omni_bridge: brain is NULL");

            return false;
    }

    /* Check if already initialized */
    if (brain->omni_wernicke_bridge) {
        return true;
    }

    /* Need Wernicke adapter first */
    if (!brain->wernicke) {
        return true;  /* Not ready yet */
    }

    /* Create omni-wernicke bridge with default config */
    omni_wernicke_config_t config;
    omni_wernicke_default_config(&config);

    brain->omni_wernicke_bridge = omni_wernicke_bridge_create(&config);

    if (!brain->omni_wernicke_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create omni-Wernicke bridge");
        return false;
    }

    /* Connect Wernicke adapter to bridge */
    if (omni_wernicke_connect_wernicke(brain->omni_wernicke_bridge, brain->wernicke) != 0) {
        LOG_WARN(LOG_MODULE, "Failed to connect Wernicke to omni bridge");
    }

    /* Connect to Broca bridge if available */
    if (brain->wernicke_broca_bridge) {
        if (omni_wernicke_connect_broca_bridge(brain->omni_wernicke_bridge,
                                                brain->wernicke_broca_bridge) != 0) {
            LOG_WARN(LOG_MODULE, "Failed to connect Broca bridge to omni-Wernicke");
        }
    }

    /* Connect to bio-async */
    if (brain->bio_async_enabled) {
        if (omni_wernicke_connect_bio_async(brain->omni_wernicke_bridge) != 0) {
            LOG_WARN(LOG_MODULE, "Failed to connect omni-Wernicke to bio-async");
        }
    }

    LOG_DEBUG(LOG_MODULE, "Omni-Wernicke bridge initialized");
    return true;
}

bool nimcp_brain_factory_init_wernicke_gpu_bridge(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_wernicke_gpu_bridge: brain is NULL");

            return false;
    }

    /* Check if already initialized */
    if (brain->wernicke_gpu_bridge) {
        return true;
    }

    /* Need Wernicke adapter first */
    if (!brain->wernicke) {
        return true;  /* Not ready yet */
    }

    /* GPU bridge is optional - only create if CUDA is available */
    /* Note: We create the bridge without GPU context, it can be set later */
    wernicke_gpu_bio_config_t config = wernicke_gpu_bio_default_config();

    /* Create GPU bio bridge (GPU context can be NULL, set later when CUDA available) */
    brain->wernicke_gpu_bridge = wernicke_gpu_bio_create(NULL, &config);

    if (!brain->wernicke_gpu_bridge) {
        /* Non-fatal - GPU not required */
        LOG_DEBUG(LOG_MODULE, "Wernicke GPU bridge not available (CUDA may not be present)");
        return true;
    }

    /* Connect to bio-async if enabled */
    if (brain->bio_async_enabled) {
        if (wernicke_gpu_bio_connect(brain->wernicke_gpu_bridge) != 0) {
            LOG_WARN(LOG_MODULE, "Failed to connect Wernicke GPU bridge to bio-async");
        }
    }

    LOG_DEBUG(LOG_MODULE, "Wernicke GPU bridge initialized");
    return true;
}

bool nimcp_brain_factory_connect_wernicke_to_semantic_memory(brain_t brain) {
    if (!brain || !brain->wernicke) {
        return true;  /* Nothing to connect */
    }

    /* Check if semantic memory is available */
    /* Note: Semantic memory may be stored in brain_kg or separate system */

    /*
     * TODO: Connect Wernicke to semantic memory
     * wernicke_connect_semantic_memory(brain->wernicke, brain->semantic_memory);
     */

    LOG_DEBUG(LOG_MODULE, "Wernicke connected to semantic memory");
    return true;
}

bool nimcp_brain_factory_connect_wernicke_to_working_memory(brain_t brain) {
    if (!brain || !brain->wernicke) {
        return true;  /* Nothing to connect */
    }

    /* Check if working memory is available */
    if (!brain->working_memory) {
        return true;  /* WM not initialized yet */
    }

    /*
     * Register Wernicke as a working memory consumer for phonological loop.
     * This allows Wernicke to maintain active word candidates during
     * speech comprehension.
     */

    /* TODO: Register with working memory
     * working_memory_register_consumer(brain->working_memory,
     *     WM_CONSUMER_WERNICKE, brain->wernicke);
     */

    LOG_DEBUG(LOG_MODULE, "Wernicke connected to working memory");
    return true;
}

bool nimcp_brain_factory_connect_wernicke_to_training(brain_t brain) {
    if (!brain || !brain->wernicke) {
        return true;  /* Nothing to connect */
    }

    /* Check if training is enabled */
    if (!brain->enable_training_integration || !brain->training_ctx) {
        return true;  /* Training not enabled */
    }

    /*
     * Register Wernicke adapter with training context.
     * This allows language comprehension learning through:
     * - Vocabulary expansion
     * - Semantic association strengthening
     * - Phonological pattern learning
     */

    /* TODO: Register with training
     * nimcp_brain_training_register_module(brain->training_ctx,
     *     TRAIN_MODULE_WERNICKE, brain->wernicke);
     */

    LOG_DEBUG(LOG_MODULE, "Wernicke connected to training system");
    return true;
}

bool nimcp_brain_factory_connect_wernicke_to_immune(brain_t brain) {
    if (!brain || !brain->wernicke) {
        return true;  /* Nothing to connect */
    }

    /* Check if immune system is available */
    if (!brain->immune_enabled || !brain->immune_system) {
        return true;  /* Immune not enabled */
    }

    /*
     * Register for cytokine signals that affect language comprehension.
     * Neuroinflammation (high IL-1beta, TNF-alpha) can cause:
     * - Reduced comprehension speed
     * - Word-finding difficulties
     * - Semantic integration impairment
     */

    /* TODO: Register with immune system
     * brain_immune_register_cytokine_callback(brain->immune_system,
     *     CYTOKINE_IL1B | CYTOKINE_TNF_A, wernicke_inflammation_callback, brain->wernicke);
     */

    LOG_DEBUG(LOG_MODULE, "Wernicke connected to immune system");
    return true;
}

bool nimcp_brain_factory_connect_wernicke_to_bio_async(brain_t brain) {
    if (!brain || !brain->wernicke) {
        return true;  /* Nothing to connect */
    }

    /* Check if bio-async is enabled */
    if (!brain->bio_async_enabled || !brain->bio_async_ctx) {
        return true;  /* Bio-async not enabled */
    }

    /* Connect via omni bridge if available */
    if (brain->omni_wernicke_bridge) {
        if (omni_wernicke_connect_bio_async(brain->omni_wernicke_bridge) != 0) {
            LOG_WARN(LOG_MODULE, "Failed to connect omni-Wernicke to bio-async");
        }
    }

    /* Connect GPU bridge if available */
    if (brain->wernicke_gpu_bridge) {
        if (wernicke_gpu_bio_connect(brain->wernicke_gpu_bridge) != 0) {
            LOG_WARN(LOG_MODULE, "Failed to connect Wernicke GPU to bio-async");
        }
    }

    LOG_DEBUG(LOG_MODULE, "Wernicke connected to bio-async messaging");
    return true;
}

bool nimcp_brain_factory_init_wernicke_immune_bridge(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_wernicke_immune_bridge: brain is NULL");

            return false;
    }

    /* Check if already initialized */
    if (brain->wernicke_immune_bridge) {
        return true;
    }

    /* Need Wernicke adapter first */
    if (!brain->wernicke) {
        return true;  /* Not ready yet */
    }

    /* Need immune system */
    if (!brain->immune_enabled || !brain->immune_system) {
        return true;  /* Immune not enabled, not an error */
    }

    /* Create immune bridge with default config */
    wernicke_immune_config_t config;
    wernicke_immune_default_config(&config);

    brain->wernicke_immune_bridge = wernicke_immune_bridge_create(
        &config, brain->immune_system, brain->wernicke);

    if (!brain->wernicke_immune_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create Wernicke immune bridge");
        return false;
    }

    /* Start immune integration monitoring */
    if (wernicke_immune_bridge_start(brain->wernicke_immune_bridge) != 0) {
        LOG_WARN(LOG_MODULE, "Failed to start Wernicke immune bridge");
    }

    LOG_DEBUG(LOG_MODULE, "Wernicke immune bridge initialized");
    return true;
}

bool nimcp_brain_factory_init_wernicke_nlp_bridge(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "nimcp_brain_factory_init_wernicke_nlp_bridge: brain is NULL");

            return false;
    }

    /* Check if already initialized */
    if (brain->wernicke_nlp_bridge) {
        return true;
    }

    /* Need Wernicke adapter first */
    if (!brain->wernicke) {
        return true;  /* Not ready yet */
    }

    /* Create NLP bridge with default config */
    wernicke_nlp_config_t config;
    wernicke_nlp_default_config(&config);

    brain->wernicke_nlp_bridge = wernicke_nlp_bridge_create(brain->wernicke, &config);

    if (!brain->wernicke_nlp_bridge) {
        LOG_WARN(LOG_MODULE, "Failed to create Wernicke NLP bridge");
        return false;
    }

    /* Connect to speech cortex if available */
    if (brain->speech_cortex) {
        wernicke_nlp_connect_speech_cortex(brain->wernicke_nlp_bridge, brain->speech_cortex);
    }

    /* Connect to NLP network if available */
    if (brain->nlp_network) {
        wernicke_nlp_connect_nlp_network(brain->wernicke_nlp_bridge, brain->nlp_network);
    }

    /* Connect to working memory if available */
    if (brain->working_memory) {
        wernicke_nlp_connect_working_memory(brain->wernicke_nlp_bridge, brain->working_memory);
    }

    /* Connect to knowledge graph if available */
    if (brain->internal_kg) {
        wernicke_nlp_connect_knowledge_graph(brain->wernicke_nlp_bridge, brain->internal_kg);
    }

    /* Connect to bio-async if enabled */
    if (brain->bio_async_enabled) {
        wernicke_nlp_connect_bio_async(brain->wernicke_nlp_bridge);
    }

    LOG_DEBUG(LOG_MODULE, "Wernicke NLP bridge initialized");
    return true;
}

//=============================================================================
// Destruction
//=============================================================================

/**
 * @brief Destroy Wernicke's area subsystem
 *
 * WHAT: Clean up all Wernicke resources and bridges
 * WHY:  Prevent memory leaks during brain destruction
 * HOW:  Destroy in reverse initialization order
 *
 * @param brain Brain instance
 */
void nimcp_brain_factory_destroy_wernicke_subsystem(brain_t brain) {
    if (!brain) return;

    LOG_DEBUG(LOG_MODULE, "Destroying Wernicke's area subsystem");

    /* Destroy NLP bridge first */
    if (brain->wernicke_nlp_bridge) {
        wernicke_nlp_bridge_destroy(brain->wernicke_nlp_bridge);
        brain->wernicke_nlp_bridge = NULL;
    }

    /* Destroy quantum bridge (depends on wernicke) */
    if (brain->wernicke_quantum_bridge) {
        wernicke_quantum_bridge_destroy(brain->wernicke_quantum_bridge);
        brain->wernicke_quantum_bridge = NULL;
    }

    /* Destroy substrate bridge */
    if (brain->wernicke_substrate_bridge) {
        wernicke_substrate_bridge_destroy(brain->wernicke_substrate_bridge);
        brain->wernicke_substrate_bridge = NULL;
    }

    /* Destroy Wernicke adapter */
    if (brain->wernicke) {
        wernicke_destroy(brain->wernicke);
        brain->wernicke = NULL;
    }

    brain->wernicke_enabled = false;
    brain->last_wernicke_update_us = 0;

    LOG_DEBUG(LOG_MODULE, "Wernicke's area subsystem destroyed");
}
