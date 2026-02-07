//=============================================================================
// multimodal_integrator.c - Multimodal Feature Integration Implementation
//=============================================================================
/**
 * @file multimodal_integrator.c
 * @brief Single Responsibility: Fuse multi-modal features into unified representation
 *
 * REFACTORING NOTE:
 * Extracted from nimcp_brain.c brain_process_multimodal()
 * Reason: Apply Single Responsibility Principle - separate integration logic
 */

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

// Logging integration
#include "utils/logging/nimcp_logging.h"

// Unified memory integration
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_unified_memory.h"

#include "core/brain/processing/multimodal_integrator.h"
#include "core/brain/nimcp_brain.h"
#include "core/integration/nimcp_multimodal_integration.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>

#define LOG_MODULE "BRAIN_PROC_MM"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(multimodal_integrator)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_multimodal_integrator_mesh_id = 0;
static mesh_participant_registry_t* g_multimodal_integrator_mesh_registry = NULL;

nimcp_error_t multimodal_integrator_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_multimodal_integrator_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "multimodal_integrator", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "multimodal_integrator";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_multimodal_integrator_mesh_id);
    if (err == NIMCP_SUCCESS) g_multimodal_integrator_mesh_registry = registry;
    return err;
}

void multimodal_integrator_mesh_unregister(void) {
    if (g_multimodal_integrator_mesh_registry && g_multimodal_integrator_mesh_id != 0) {
        mesh_participant_unregister(g_multimodal_integrator_mesh_registry, g_multimodal_integrator_mesh_id);
        g_multimodal_integrator_mesh_id = 0;
        g_multimodal_integrator_mesh_registry = NULL;
    }
}


//=============================================================================
// Internal Brain Structure Access
//=============================================================================

// Forward declaration of brain structure (opaque type)
struct brain_struct {
    // Multimodal integration layer
    multimodal_integrator_t multimodal;
    float* integrated_feature_buffer;

    // Configuration
    struct {
        uint32_t visual_feature_dim;
        uint32_t audio_feature_dim;
        uint32_t speech_feature_dim;
        bool enable_multimodal_integration;
    } config;

    // ... other fields not needed by this module
};

//=============================================================================
// API Implementation
//=============================================================================

void integrated_features_init(integrated_features_t* output)
{
    if (!output) {
        return;
    }

    memset(output, 0, sizeof(integrated_features_t));

    output->integrated_features = NULL;
    output->integrated_dim = 0;
    output->visual_attention = 0.0f;
    output->audio_attention = 0.0f;
    output->speech_attention = 0.0f;
    output->direct_attention = 0.0f;
}

bool multimodal_integrate_features(
    const brain_t brain,
    const sensory_features_t* features,
    integrated_features_t* output)
{
    // =========================================================================
    // VALIDATION
    // =========================================================================

    if (!brain || !features || !output) {
        fprintf(stderr, "multimodal_integrator: Invalid parameters\n");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "multimodal_integrate_features: required parameter is NULL (brain, features, output)");
        return false;
    }

    if (!brain->config.enable_multimodal_integration) {
        fprintf(stderr, "multimodal_integrator: Multimodal integration not enabled\n");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "multimodal_integrate_features: brain->config is NULL");
        return false;
    }

    if (!brain->multimodal || !brain->integrated_feature_buffer) {
        fprintf(stderr, "multimodal_integrator: Multimodal integrator not initialized\n");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "multimodal_integrate_features: required parameter is NULL (brain->multimodal, brain->integrated_feature_buffer)");
        return false;
    }

    // Initialize output
    integrated_features_init(output);

    // =========================================================================
    // MULTIMODAL INTEGRATION
    // =========================================================================

    // Prepare input for multimodal integrator
    multimodal_input_t mm_input = {
        .visual_features = features->visual_features,
        .visual_dim = features->visual_dim,
        .audio_features = features->audio_features,
        .audio_dim = features->audio_dim,
        .speech_features = features->speech_features,
        .speech_dim = features->speech_dim,
        .direct_features = (float*)features->direct_features,
        .direct_dim = features->direct_dim,
        .timestamp = 0  // TODO: Get timestamp from input
    };

    // Integrate features using attention mechanism
    bool integrate_success = multimodal_integrate(
        brain->multimodal,
        &mm_input,
        brain->integrated_feature_buffer
    );

    if (!integrate_success) {
        fprintf(stderr, "multimodal_integrator: Integration failed\n");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "multimodal_integrate_features: integrate_success is NULL");
        return false;
    }

    // Set output integrated features
    output->integrated_features = brain->integrated_feature_buffer;
    output->integrated_dim = brain->config.visual_feature_dim +
                             brain->config.audio_feature_dim +
                             brain->config.speech_feature_dim;

    // Get attention weights for interpretability
    multimodal_get_attention(
        brain->multimodal,
        &output->visual_attention,
        &output->audio_attention,
        &output->speech_attention,
        &output->direct_attention
    );

    return true;
}
