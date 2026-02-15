//=============================================================================
// nimcp_brain_init_language.c - Language Layer Initialization
//=============================================================================
/**
 * @file nimcp_brain_init_language.c
 * @brief Language Layer Initialization Implementation
 *
 * WHAT: Initialization functions for unified Language Layer
 * WHY:  Enable unified language processing capabilities in the brain
 * HOW:  Creates language orchestrator and connects all integration bridges
 *
 * @version Phase L6: Language Layer Brain Integration
 * @author NIMCP Development Team
 * @date 2026-01-05
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_language.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"

/* Language Layer includes */
#include "language/nimcp_language_orchestrator.h"
#include "language/nimcp_language_config.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <stdlib.h>

#define LOG_MODULE "BRAIN_INIT_LANGUAGE"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_init_language)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_brain_init_language_mesh_id = 0;
static mesh_participant_registry_t* g_brain_init_language_mesh_registry = NULL;

nimcp_error_t brain_init_language_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_brain_init_language_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "brain_init_language", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "brain_init_language";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_brain_init_language_mesh_id);
    if (err == NIMCP_SUCCESS) g_brain_init_language_mesh_registry = registry;
    return err;
}

void brain_init_language_mesh_unregister(void) {
    if (g_brain_init_language_mesh_registry && g_brain_init_language_mesh_id != 0) {
        mesh_participant_unregister(g_brain_init_language_mesh_registry, g_brain_init_language_mesh_id);
        g_brain_init_language_mesh_id = 0;
        g_brain_init_language_mesh_registry = NULL;
    }
}


/* Compatibility macro for set_error */
#ifndef set_error
#define set_error(msg) LOG_ERROR(LOG_MODULE, "%s", msg)
#endif

/* Forward declarations for bridge types (avoid header conflicts) */
struct language_perception_bridge;
struct language_cognitive_bridge;
struct language_training_bridge;
struct language_omni_bridge;
struct language_immune_bridge;
struct language_gpu_bridge;
struct language_thalamic_bridge;
struct language_substrate_bridge;
struct language_logic_bridge;

/* Bio-async forward declarations */
struct bio_router;

/* Bridge API forward declarations - stub implementations */
static struct language_perception_bridge* language_perception_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_perception_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - returns NULL, actual implementation in bridge .c files */
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_cognitive_bridge* language_cognitive_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_cognitive_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_training_bridge* language_training_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_training_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_omni_bridge* language_omni_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_omni_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_immune_bridge* language_immune_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_immune_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_gpu_bridge* language_gpu_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_gpu_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_thalamic_bridge* language_thalamic_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_thalamic_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_substrate_bridge* language_substrate_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_substrate_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_logic_bridge* language_logic_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_logic_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Check prerequisites for language layer initialization
 */
static bool check_prerequisites(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Language layer requires at least speech processing capability */
    if (!brain->config.enable_speech_cortex &&
        !brain->config.enable_multimodal_integration) {
        LOG_DEBUG(LOG_MODULE, "Language layer not needed - no speech/multimodal config");
        return false;
    }

    return true;
}

/**
 * @brief Create default orchestrator configuration from brain config
 */
static void create_orchestrator_config(
    brain_t brain,
    language_orchestrator_config_t* config)
{
    if (!config) return;

    /* Get defaults first */
    language_orchestrator_default_config(config);

    /* Adjust based on brain capabilities */
    config->enable_wernicke = (brain->wernicke != NULL || brain->wernicke_enabled);
    config->enable_broca = (brain->broca != NULL || brain->broca_enabled);
    config->enable_nlp_core = true;
    config->enable_multimodal = brain->config.enable_multimodal_integration;

    /* Bridge enables */
    config->enable_perception_bridge = brain->config.enable_speech_cortex;
    config->enable_cognitive_bridge = true;
    config->enable_training_bridge = true;
    config->enable_omni_bridge = (brain->omni_wernicke_bridge != NULL);
    config->enable_immune_bridge = (brain->immune_system != NULL);
    config->enable_gpu_bridge = brain->gpu_enabled;

    /* Adjust working memory size */
    if (brain->config.working_memory_capacity > 0) {
        config->phoneme_buffer_size = brain->config.working_memory_capacity * 4;
    }
}

/**
 * @brief Connect orchestrator to Wernicke adapter
 */
static bool connect_wernicke(brain_t brain) {
    if (!brain || !brain->language_layer) return true;

    if (!brain->wernicke) {
        LOG_DEBUG(LOG_MODULE, "Wernicke not available, skipping connection");
        return true;
    }

    int result = language_orchestrator_connect_wernicke(
        brain->language_layer,
        brain->wernicke
    );

    if (result != 0) {
        LOG_WARN(LOG_MODULE, "Failed to connect Wernicke to language layer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "connect_wernicke: validation failed");
        return false;
    }

    return true;
}

/**
 * @brief Connect orchestrator to Broca adapter
 */
static bool connect_broca(brain_t brain) {
    if (!brain || !brain->language_layer) return true;

    if (!brain->broca) {
        LOG_DEBUG(LOG_MODULE, "Broca not available, skipping connection");
        return true;
    }

    int result = language_orchestrator_connect_broca(
        brain->language_layer,
        brain->broca
    );

    if (result != 0) {
        LOG_WARN(LOG_MODULE, "Failed to connect Broca to language layer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "connect_broca: validation failed");
        return false;
    }

    return true;
}

/**
 * @brief Connect orchestrator to NLP network
 */
static bool connect_nlp_network(brain_t brain) {
    if (!brain || !brain->language_layer) return true;

    /* nlp_network is a struct member, not a pointer */
    int result = language_orchestrator_connect_nlp(
        brain->language_layer,
        brain->nlp_network
    );

    if (result != 0) {
        LOG_WARN(LOG_MODULE, "Failed to connect NLP network to language layer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "connect_nlp_network: validation failed");
        return false;
    }

    return true;
}

/**
 * @brief Connect orchestrator to speech cortex
 */
static bool connect_speech_cortex(brain_t brain) {
    if (!brain || !brain->language_layer) return true;

    if (!brain->speech_cortex) {
        LOG_DEBUG(LOG_MODULE, "Speech cortex not available, skipping connection");
        return true;
    }

    int result = language_orchestrator_connect_speech_cortex(
        brain->language_layer,
        brain->speech_cortex
    );

    if (result != 0) {
        LOG_WARN(LOG_MODULE, "Failed to connect speech cortex to language layer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "connect_speech_cortex: validation failed");
        return false;
    }

    return true;
}

//=============================================================================
// Public API Implementation
//=============================================================================

bool nimcp_brain_factory_init_language_subsystem(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "nimcp_brain_factory_init_language_subsystem: brain is NULL");
        return false;
    }

    /* Check if already initialized */
    if (brain->language_layer) {
        return true;  /* Already initialized */
    }

    /* Check prerequisites */
    if (!check_prerequisites(brain)) {
        brain->language_layer_enabled = false;
        return true;  /* Not needed, not an error */
    }

    /* Create orchestrator configuration */
    language_orchestrator_config_t config;
    create_orchestrator_config(brain, &config);

    /* Create language orchestrator */
    brain->language_layer = language_orchestrator_create(&config);
    if (!brain->language_layer) {
        set_error("Failed to create language orchestrator");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_subsystem: brain->language_layer is NULL");
        return false;
    }

    brain->language_layer_enabled = true;
    brain->last_language_update_us = 0;

    /* Connect core subsystems */
    if (!connect_wernicke(brain)) {
        LOG_WARN(LOG_MODULE, "Wernicke connection failed (non-fatal)");
    }

    if (!connect_broca(brain)) {
        LOG_WARN(LOG_MODULE, "Broca connection failed (non-fatal)");
    }

    if (!connect_nlp_network(brain)) {
        LOG_WARN(LOG_MODULE, "NLP network connection failed (non-fatal)");
    }

    if (!connect_speech_cortex(brain)) {
        LOG_WARN(LOG_MODULE, "Speech cortex connection failed (non-fatal)");
    }

    /* Initialize bridges */
    if (!nimcp_brain_factory_init_language_perception_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Perception bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_cognitive_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Cognitive bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_training_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Training bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_omni_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Omni bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_immune_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Immune bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_gpu_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "GPU bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_thalamic_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Thalamic bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_substrate_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Substrate bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_logic_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Logic bridge init failed (non-fatal)");
    }

    /* Connect to bio-async */
    if (!nimcp_brain_factory_connect_language_to_bio_async(brain)) {
        LOG_WARN(LOG_MODULE, "Bio-async connection failed (non-fatal)");
    }

    /* Start the orchestrator */
    if (language_orchestrator_start(brain->language_layer) != 0) {
        LOG_WARN(LOG_MODULE, "Orchestrator start failed (non-fatal)");
    }

    LOG_INFO(LOG_MODULE, "Language layer initialized successfully");
    return true;
}

bool nimcp_brain_factory_init_language_perception_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_perception_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    /* Get default perception config */
    language_perception_config_t config;
    language_perception_default_config(&config);

    /* Adjust based on brain capabilities */
    config.enable_speech_cortex = (brain->speech_cortex != NULL);
    config.enable_audio_cortex = (brain->audio_cortex != NULL);
    config.enable_visual_cortex = (brain->visual_cortex != NULL);
    config.enable_omni_sensory = brain->config.enable_multimodal_integration;

    /* Create perception bridge (stub for now) */
    brain->language_perception_bridge = language_perception_bridge_create_stub(
        brain->language_layer,
        &config
    );

    /* Note: Stub returns NULL, but that's okay for now */
    if (!brain->language_perception_bridge) {
        LOG_DEBUG(LOG_MODULE, "Perception bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_cognitive_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_cognitive_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    /* Get default cognitive config */
    language_cognitive_config_t config;
    language_cognitive_default_config(&config);

    /* Adjust working memory based on brain config */
    if (brain->config.working_memory_capacity > 0) {
        config.phonological_buffer_size = brain->config.working_memory_capacity;
    }

    /* Create cognitive bridge (stub for now) */
    brain->language_cognitive_bridge = language_cognitive_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_cognitive_bridge) {
        LOG_DEBUG(LOG_MODULE, "Cognitive bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_training_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_training_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    /* Get default training config */
    language_training_config_t config;
    language_training_default_config(&config);

    /* Create training bridge (stub for now) */
    brain->language_training_bridge = language_training_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_training_bridge) {
        LOG_DEBUG(LOG_MODULE, "Training bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_omni_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_omni_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    if (!brain->omni_wernicke_bridge) {
        return true;  /* Not enabled */
    }

    /* Get default omni config */
    language_omni_config_t config;
    language_omni_default_config(&config);

    /* Create omni bridge (stub for now) */
    brain->language_omni_bridge = language_omni_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_omni_bridge) {
        LOG_DEBUG(LOG_MODULE, "Omni bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_immune_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_immune_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    if (!brain->immune_system) {
        return true;  /* Not enabled */
    }

    /* Get default immune config */
    language_immune_config_t config;
    language_immune_default_config(&config);

    /* Create immune bridge (stub for now) */
    brain->language_immune_bridge = language_immune_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_immune_bridge) {
        LOG_DEBUG(LOG_MODULE, "Immune bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_gpu_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_gpu_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    if (!brain->gpu_enabled) {
        return true;  /* Not enabled */
    }

    /* Get default GPU config */
    language_gpu_config_t config;
    language_gpu_default_config(&config);
    config.enable_gpu = true;

    /* Set device ID from brain config */
    config.device_id = brain->config.gpu_device_id;

    /* Create GPU bridge (stub for now) */
    brain->language_gpu_bridge = language_gpu_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_gpu_bridge) {
        LOG_DEBUG(LOG_MODULE, "GPU bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_thalamic_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_thalamic_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    /* Thalamic bridge doesn't require a specific prerequisite - always try to init */

    /* Get default thalamic config */
    language_thalamic_config_t config;
    language_thalamic_default_config(&config);

    /* Create thalamic bridge (stub for now) */
    brain->language_thalamic_bridge = language_thalamic_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_thalamic_bridge) {
        LOG_DEBUG(LOG_MODULE, "Thalamic bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_substrate_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_substrate_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    /* Substrate bridge doesn't require a specific prerequisite - always try to init */

    /* Get default substrate config */
    language_substrate_config_t config;
    language_substrate_default_config(&config);

    /* Create substrate bridge (stub for now) */
    brain->language_substrate_bridge = language_substrate_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_substrate_bridge) {
        LOG_DEBUG(LOG_MODULE, "Substrate bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_logic_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_logic_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    /* Logic bridge doesn't require a specific prerequisite - always try to init */

    /* Get default logic config */
    language_logic_config_t config;
    language_logic_default_config(&config);

    /* Create logic bridge (stub for now) */
    brain->language_logic_bridge = language_logic_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_logic_bridge) {
        LOG_DEBUG(LOG_MODULE, "Logic bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_connect_language_to_bio_async(brain_t brain) {
    if (!brain || !brain->language_layer) {
        return true;  /* Non-fatal */
    }

    if (!brain->bio_async_enabled || !brain->bio_async_ctx) {
        return true;  /* Bio-async not enabled */
    }

    /* Registration handled by orchestrator start */
    LOG_DEBUG(LOG_MODULE, "Language layer bio-async connection deferred to orchestrator");
    return true;
}

void nimcp_brain_factory_destroy_language_subsystem(brain_t brain) {
    if (!brain) {
        return;
    }

    /* Stop orchestrator */
    if (brain->language_layer) {
        language_orchestrator_stop(brain->language_layer);
    }

    /* Clear bridge pointers (stubs, no actual destroy needed yet) */
    brain->language_logic_bridge = NULL;
    brain->language_substrate_bridge = NULL;
    brain->language_thalamic_bridge = NULL;
    brain->language_gpu_bridge = NULL;
    brain->language_immune_bridge = NULL;
    brain->language_omni_bridge = NULL;
    brain->language_training_bridge = NULL;
    brain->language_cognitive_bridge = NULL;
    brain->language_perception_bridge = NULL;

    /* Destroy orchestrator */
    if (brain->language_layer) {
        language_orchestrator_destroy(brain->language_layer);
        brain->language_layer = NULL;
    }

    brain->language_layer_enabled = false;

    LOG_INFO(LOG_MODULE, "Language layer destroyed");
}

bool nimcp_brain_factory_language_is_initialized(brain_t brain) {
    if (!brain) {
        return false;
    }
    return brain->language_layer_enabled && brain->language_layer != NULL;
}

struct language_orchestrator* nimcp_brain_factory_get_language_orchestrator(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;
    }
    return brain->language_layer;
}
