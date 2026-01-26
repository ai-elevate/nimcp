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

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for multimodal_integrator module */
static nimcp_health_agent_t* g_multimodal_integrator_health_agent = NULL;

/**
 * @brief Set health agent for multimodal_integrator heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void multimodal_integrator_set_health_agent(nimcp_health_agent_t* agent) {
    g_multimodal_integrator_health_agent = agent;
}

/** @brief Send heartbeat from multimodal_integrator module */
static inline void multimodal_integrator_heartbeat(const char* operation, float progress) {
    if (g_multimodal_integrator_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_multimodal_integrator_health_agent, operation, progress);
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
        return false;
    }

    if (!brain->config.enable_multimodal_integration) {
        fprintf(stderr, "multimodal_integrator: Multimodal integration not enabled\n");
        return false;
    }

    if (!brain->multimodal || !brain->integrated_feature_buffer) {
        fprintf(stderr, "multimodal_integrator: Multimodal integrator not initialized\n");
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
