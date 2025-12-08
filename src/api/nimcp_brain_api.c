/**
 * @file nimcp_brain_api.c
 * @brief Brain core API implementation - create, destroy, learn, predict, save/load
 *
 * This file contains the primary brain lifecycle and inference functions.
 * Extracted from nimcp.c (SRP refactoring)
 */

#include "nimcp.h"
#include "api/nimcp_api_internal.h"
#include "core/brain/nimcp_brain.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/config/nimcp_config.h"
#include "utils/logging/nimcp_logging.h"
#include <stdio.h>
#include <string.h>

#define LOG_MODULE "API.BRAIN"

// External declarations from nimcp.c
extern void set_error(const char* fmt, ...);

//=============================================================================
// Brain API Implementation
//=============================================================================

NIMCP_EXPORT nimcp_brain_t nimcp_brain_create(
    const char* name,
    nimcp_brain_size_t size,
    nimcp_brain_task_t task,
    uint32_t num_inputs,
    uint32_t num_outputs)
{
    LOG_INFO("Creating brain: name='%s', size=%d, task=%d, inputs=%u, outputs=%u",
             name ? name : "NULL", size, task, num_inputs, num_outputs);

    if (!name) {
        LOG_ERROR("Brain name cannot be NULL");
        set_error("Brain name cannot be NULL");
        return NULL;
    }

    // Allocate handle
    LOG_DEBUG("Allocating brain handle (%zu bytes)", sizeof(struct nimcp_brain_handle));
    nimcp_brain_t handle = (nimcp_brain_t)nimcp_malloc(sizeof(struct nimcp_brain_handle));
    if (!handle) {
        LOG_ERROR("Failed to allocate brain handle");
        set_error("Failed to allocate brain handle");
        return NULL;
    }

    // Map public enums to internal enums
    brain_size_t internal_size = (brain_size_t)size;
    brain_task_t internal_task = (brain_task_t)task;
    LOG_DEBUG("Mapped enums: internal_size=%d, internal_task=%d", internal_size, internal_task);

    // Create internal brain
    LOG_DEBUG("Creating internal brain structure");
    handle->internal_brain = brain_create(name, internal_size, internal_task,
                                          num_inputs, num_outputs);

    if (!handle->internal_brain) {
        LOG_ERROR("Failed to create internal brain for '%s'", name);
        set_error("Failed to create internal brain");
        nimcp_free(handle);
        return NULL;
    }

    set_error("No error");
    LOG_INFO("Brain '%s' created successfully (handle=%p)", name, (void*)handle);
    return handle;
}

NIMCP_EXPORT void nimcp_brain_destroy(nimcp_brain_t brain) {
    if (!brain) {
        LOG_DEBUG("nimcp_brain_destroy called with NULL brain, ignoring");
        return;
    }

    LOG_INFO("Destroying brain (handle=%p)", (void*)brain);

    if (brain->internal_brain) {
        LOG_DEBUG("Destroying internal brain structure");
        brain_destroy(brain->internal_brain);
    } else {
        LOG_WARN("Brain handle has NULL internal_brain");
    }

    LOG_DEBUG("Freeing brain handle");
    nimcp_free(brain);
    LOG_DEBUG("Brain destroyed successfully");
}

NIMCP_EXPORT nimcp_status_t nimcp_brain_learn_example(
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

NIMCP_EXPORT nimcp_status_t nimcp_brain_predict(
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

NIMCP_EXPORT nimcp_status_t nimcp_brain_infer(
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

NIMCP_EXPORT nimcp_status_t nimcp_brain_save(nimcp_brain_t brain, const char* filepath) {
    if (!brain) {
        set_error("Brain handle is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!filepath) {
        set_error("Filepath is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    // Call internal brain API
    bool success = brain_save(brain->internal_brain, filepath);

    if (!success) {
        set_error("Failed to save brain");
        return NIMCP_ERROR_IO;
    }

    set_error("No error");
    return NIMCP_OK;
}

NIMCP_EXPORT nimcp_brain_t nimcp_brain_load(const char* filepath) {
    if (!filepath) {
        set_error("Filepath is NULL");
        return NULL;
    }

    // Allocate handle
    nimcp_brain_t handle = (nimcp_brain_t)nimcp_malloc(sizeof(struct nimcp_brain_handle));
    if (!handle) {
        set_error("Failed to allocate brain handle");
        return NULL;
    }

    // Load internal brain
    handle->internal_brain = brain_load(filepath);

    if (!handle->internal_brain) {
        set_error("Failed to load brain from file");
        nimcp_free(handle);
        return NULL;
    }

    set_error("No error");
    return handle;
}

NIMCP_EXPORT nimcp_brain_t nimcp_brain_create_from_config(const char* config_filepath) {
    if (!config_filepath) {
        set_error("Config filepath is NULL");
        return NULL;
    }

    // Load configuration from YAML/JSON
    nimcp_brain_config_t config;
    if (!nimcp_config_load(config_filepath, &config)) {
        set_error("Failed to load config from %s", config_filepath);
        return NULL;
    }

    // Create brain using loaded configuration
    nimcp_brain_t brain = nimcp_brain_create(
        config.name,
        (nimcp_brain_size_t)config.size,
        (nimcp_brain_task_t)config.task,
        config.num_inputs,
        config.num_outputs
    );

    if (!brain) {
        set_error("Failed to create brain from config");
        return NULL;
    }

    // Note: Additional configuration like BCM, STDP, ethics could be applied here
    // For now, we just create the basic brain structure
    // TODO: Apply plasticity settings, ethics config, etc.

    set_error("No error");
    return brain;
}

NIMCP_EXPORT nimcp_status_t nimcp_brain_probe(nimcp_brain_t brain, nimcp_brain_probe_t* probe) {
    if (!brain) {
        set_error("Brain is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    if (!probe) {
        set_error("Probe output structure is NULL");
        return NIMCP_ERROR_NULL_ARG;
    }

    // Get internal brain statistics
    brain_stats_t internal_stats;
    if (!brain_get_stats(brain->internal_brain, &internal_stats)) {
        set_error("Failed to get brain statistics");
        return NIMCP_ERROR;
    }

    // Map internal stats to public probe structure
    strncpy(probe->task_name, internal_stats.task_name, sizeof(probe->task_name) - 1);
    probe->task_name[sizeof(probe->task_name) - 1] = '\0';

    // Map internal size enum to public enum
    switch (internal_stats.size) {
        case BRAIN_SIZE_TINY:   probe->size = NIMCP_BRAIN_TINY; break;
        case BRAIN_SIZE_SMALL:  probe->size = NIMCP_BRAIN_SMALL; break;
        case BRAIN_SIZE_MEDIUM: probe->size = NIMCP_BRAIN_MEDIUM; break;
        case BRAIN_SIZE_LARGE:  probe->size = NIMCP_BRAIN_LARGE; break;
        default:                probe->size = NIMCP_BRAIN_SMALL; break;
    }

    // Get brain configuration to determine task type
    brain_config_t internal_config;
    memset(&internal_config, 0, sizeof(internal_config));
    // Note: We use size as a proxy for task type since internal API doesn't expose it directly
    probe->task = NIMCP_TASK_CLASSIFICATION; // Default

    probe->num_neurons = internal_stats.num_neurons;
    probe->num_synapses = internal_stats.num_synapses;
    probe->num_active_synapses = internal_stats.num_active_synapses;

    probe->total_inferences = internal_stats.total_inferences;
    probe->total_learning_steps = internal_stats.total_learning_steps;

    probe->avg_sparsity = internal_stats.avg_sparsity;
    probe->avg_inference_time_us = internal_stats.avg_inference_time_us;
    probe->current_learning_rate = internal_stats.current_learning_rate;

    probe->accuracy = internal_stats.accuracy;
    probe->memory_bytes = internal_stats.memory_bytes;

    // Note: Input/output sizes are not directly accessible from internal API
    // Set to 0 for now - could be extended if brain API exposes these
    probe->num_inputs = 0;
    probe->num_outputs = 0;

    // Get COW statistics from internal brain
    brain_cow_stats_t cow_stats;
    if (brain_get_cow_stats(brain->internal_brain, &cow_stats)) {
        probe->is_cow_clone = cow_stats.is_cow_clone;
        probe->cow_ref_count = cow_stats.cow_ref_count;
        probe->cow_shared_bytes = cow_stats.cow_shared_bytes;
        probe->cow_private_bytes = cow_stats.cow_private_bytes;
    } else {
        // Fallback to defaults if COW stats retrieval fails
        probe->is_cow_clone = false;
        probe->cow_ref_count = 0;
        probe->cow_shared_bytes = 0;
        probe->cow_private_bytes = internal_stats.memory_bytes;
    }

    set_error("No error");
    return NIMCP_OK;
}
