/**
 * @file nimcp_api_inference.c
 * @brief Brain inference and learning API implementation
 *
 * This module handles brain inference (prediction), learning operations,
 * and basic training through the learn_example interface.
 *
 * Responsibilities:
 * - Brain inference/prediction
 * - Brain learning from examples
 * - Input validation through Blood-Brain Barrier (BBB)
 */

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "API_INFERENCE"

#include <stddef.h>  /* for NULL */
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for api_inference module */
static nimcp_health_agent_t* g_api_inference_health_agent = NULL;

/**
 * @brief Set health agent for api_inference heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void api_inference_set_health_agent(nimcp_health_agent_t* agent) {
    g_api_inference_health_agent = agent;
}

/** @brief Send heartbeat from api_inference module */
static inline void api_inference_heartbeat(const char* operation, float progress) {
    if (g_api_inference_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_api_inference_health_agent, operation, progress);
    }
}


/* API Exception Integration (Phase 7) */
extern void set_error(const char* fmt, ...);
#define NIMCP_API_SET_ERROR(fmt, ...) set_error(fmt, ##__VA_ARGS__)
#include "api/nimcp_api_exception.h"

#include "nimcp.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/memory/nimcp_memory.h"
#include "security/nimcp_blood_brain_barrier.h"
#include <stdio.h>
#include <string.h>

//=============================================================================
// External References (from nimcp.c)
//=============================================================================

// These functions are defined in nimcp.c and shared across modules
extern void set_error(const char* fmt, ...);
extern const char* nimcp_get_error(void);

//=============================================================================
// Brain Inference and Learning API Implementation
//=============================================================================

nimcp_status_t nimcp_brain_learn_example(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    const char* label,
    float confidence)
{
    LOG_DEBUG("Learning example: label='%s', num_features=%u, confidence=%.3f",
              label ? label : "NULL", num_features, confidence);

    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL in learn_example");
    NIMCP_CHECK_THROW(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL in learn_example");
    NIMCP_CHECK_THROW(label, NIMCP_ERROR_NULL_ARG, "Label is NULL in learn_example");

    // === PHASE IS-1: BBB INPUT VALIDATION ===
    // Validate external input data through Blood-Brain Barrier before processing
    if (brain->internal_brain && brain->internal_brain->bbb_enabled &&
        brain->internal_brain->bbb_system) {
        LOG_DEBUG("BBB enabled, validating inputs");
        bbb_validation_result_t result;

        // Validate features array (external input data)
        if (!bbb_validate_input(brain->internal_brain->bbb_system,
                               features, num_features * sizeof(float), &result)) {
            LOG_WARN("BBB rejected features: %s", result.reason);
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_INVALID, "BBB rejected features: %s", result.reason);
        }

        // Validate label string (external string input)
        if (!bbb_validate_string(brain->internal_brain->bbb_system, label, &result)) {
            LOG_WARN("BBB rejected label: %s", result.reason);
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_INVALID, "BBB rejected label: %s", result.reason);
        }
        LOG_DEBUG("BBB validation passed");
    }

    // Call internal brain API
    LOG_DEBUG("Invoking internal brain_learn_example");
    float loss = brain_learn_example(brain->internal_brain, features, num_features, label, confidence);

    // brain_learn_example returns -1.0f on error, >= 0.0f on success (where value is the loss)
    if (loss < 0.0f) {
        LOG_ERROR("Brain learning failed for label '%s'", label);
        NIMCP_THROW_BRAIN(NIMCP_ERROR_OPERATION_FAILED, 0, "learning",
            "Brain learning failed for label '%s'", label);
        return NIMCP_ERROR;
    }

    LOG_DEBUG("Learning example completed successfully");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_predict(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    char* out_label,
    float* out_confidence)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL in brain_predict");
    NIMCP_CHECK_THROW(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL in brain_predict");
    NIMCP_CHECK_THROW(out_label, NIMCP_ERROR_NULL_ARG, "Output label buffer is NULL in brain_predict");
    NIMCP_CHECK_THROW(out_confidence, NIMCP_ERROR_NULL_ARG, "Output confidence pointer is NULL in brain_predict");

    // === PHASE IS-1: BBB INPUT VALIDATION ===
    // Validate external input data through Blood-Brain Barrier before processing
    if (brain->internal_brain && brain->internal_brain->bbb_enabled &&
        brain->internal_brain->bbb_system) {
        bbb_validation_result_t result;

        // Validate features array (external input data)
        if (!bbb_validate_input(brain->internal_brain->bbb_system,
                               features, num_features * sizeof(float), &result)) {
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "BBB rejected features: %s", result.reason);
        }
    }

    // Call internal brain API
    brain_decision_t* decision = brain_decide(brain->internal_brain, features, num_features);

    if (!decision) {
        NIMCP_THROW_BRAIN(NIMCP_ERROR_OPERATION_FAILED, 0, "prediction",
            "Brain prediction failed");
        return NIMCP_ERROR;
    }

    // Copy results
    strncpy(out_label, decision->label, 63);
    out_label[63] = '\0';
    *out_confidence = decision->confidence;

    // Free decision
    nimcp_free(decision);

    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_infer(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    float* outputs,
    uint32_t num_outputs)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL in brain_infer");
    NIMCP_CHECK_THROW(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL in brain_infer");
    NIMCP_CHECK_THROW(outputs, NIMCP_ERROR_NULL_ARG, "Outputs array is NULL in brain_infer");

    // Call internal brain API to get decision (which includes output vector)
    brain_decision_t* decision = brain_decide(brain->internal_brain, features, num_features);

    if (!decision) {
        NIMCP_THROW_BRAIN(NIMCP_ERROR_OPERATION_FAILED, 0, "inference",
            "Brain inference failed");
        return NIMCP_ERROR;
    }

    // Copy raw output vector
    uint32_t copy_size = (decision->output_size < num_outputs) ? decision->output_size : num_outputs;
    for (uint32_t i = 0; i < copy_size; i++) {
        outputs[i] = decision->output_vector[i];
    }

    // Fill remaining with zeros if requested more outputs than available
    for (uint32_t i = copy_size; i < num_outputs; i++) {
        outputs[i] = 0.0f;
    }

    // Free decision
    nimcp_free(decision);

    return NIMCP_OK;
}
