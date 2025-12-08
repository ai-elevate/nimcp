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

#define LOG_MODULE "API_INFERENCE"

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

    if (!brain) {
        LOG_ERROR("Brain handle is NULL");
        set_error("Brain handle is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!features) {
        LOG_ERROR("Features array is NULL");
        set_error("Features array is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!label) {
        LOG_ERROR("Label is NULL");
        set_error("Label is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

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
            set_error("BBB rejected features: %s", result.reason);
            return NIMCP_ERROR_INVALID;
        }

        // Validate label string (external string input)
        if (!bbb_validate_string(brain->internal_brain->bbb_system, label, &result)) {
            LOG_WARN("BBB rejected label: %s", result.reason);
            set_error("BBB rejected label: %s", result.reason);
            return NIMCP_ERROR_INVALID;
        }
        LOG_DEBUG("BBB validation passed");
    }

    // Call internal brain API
    LOG_DEBUG("Invoking internal brain_learn_example");
    float loss = brain_learn_example(brain->internal_brain, features, num_features, label, confidence);

    // brain_learn_example returns -1.0f on error, >= 0.0f on success (where value is the loss)
    if (loss < 0.0f) {
        LOG_ERROR("Brain learning failed for label '%s'", label);
        set_error("Brain learning failed");
        return NIMCP_ERROR;
    }

    set_error("No error");
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
    if (!brain) {
        set_error("Brain handle is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!features) {
        set_error("Features array is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!out_label) {
        set_error("Output label buffer is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!out_confidence) {
        set_error("Output confidence pointer is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    // === PHASE IS-1: BBB INPUT VALIDATION ===
    // Validate external input data through Blood-Brain Barrier before processing
    if (brain->internal_brain && brain->internal_brain->bbb_enabled &&
        brain->internal_brain->bbb_system) {
        bbb_validation_result_t result;

        // Validate features array (external input data)
        if (!bbb_validate_input(brain->internal_brain->bbb_system,
                               features, num_features * sizeof(float), &result)) {
            set_error("BBB rejected features: %s", result.reason);
            return NIMCP_ERROR_INVALID;
        }
    }

    // Call internal brain API
    brain_decision_t* decision = brain_decide(brain->internal_brain, features, num_features);

    if (!decision) {
        set_error("Brain prediction failed");
        return NIMCP_ERROR;
    }

    // Copy results
    strncpy(out_label, decision->label, 63);
    out_label[63] = '\0';
    *out_confidence = decision->confidence;

    // Free decision
    nimcp_free(decision);

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_infer(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    float* outputs,
    uint32_t num_outputs)
{
    if (!brain) {
        set_error("Brain handle is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!features) {
        set_error("Features array is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!outputs) {
        set_error("Outputs array is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    // Call internal brain API to get decision (which includes output vector)
    brain_decision_t* decision = brain_decide(brain->internal_brain, features, num_features);

    if (!decision) {
        set_error("Brain inference failed");
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

    set_error("No error");
    return NIMCP_OK;
}
