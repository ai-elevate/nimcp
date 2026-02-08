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
#include "utils/metrics/nimcp_metrics.h"
#include "utils/error/nimcp_error_codes.h"
#include "api/nimcp_api_exception.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#define LOG_MODULE "API.BRAIN"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_api)

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

    NIMCP_API_CHECK_NULL_RET_NULL(name, "Brain name cannot be NULL");

    // Allocate handle
    LOG_DEBUG("Allocating brain handle (%zu bytes)", sizeof(struct nimcp_brain_handle));
    nimcp_brain_t handle = (nimcp_brain_t)nimcp_malloc(sizeof(struct nimcp_brain_handle));
    NIMCP_API_CHECK_ALLOC_SIZE(handle, sizeof(struct nimcp_brain_handle),
        "Failed to allocate brain handle");

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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_BRAIN_CREATION, "Failed to create internal brain '%s'", name);
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

    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_API_CHECK_NULL(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    NIMCP_API_CHECK_NULL(label, NIMCP_ERROR_NULL_ARG, "Label is NULL");

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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_LEARNING_FAILED, "Brain learning failed for label '%s'", label);
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
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_API_CHECK_NULL(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    NIMCP_API_CHECK_NULL(out_label, NIMCP_ERROR_NULL_ARG, "Output label buffer is NULL");
    NIMCP_API_CHECK_NULL(out_confidence, NIMCP_ERROR_NULL_ARG, "Output confidence pointer is NULL");

    // === PHASE IS-1: BBB INPUT VALIDATION ===
    // Validate external input data through Blood-Brain Barrier before processing
    if (brain->internal_brain && brain->internal_brain->bbb_enabled &&
        brain->internal_brain->bbb_system) {
        bbb_validation_result_t result;

        // Validate features array (external input data)
        if (!bbb_validate_input(brain->internal_brain->bbb_system,
                               features, num_features * sizeof(float), &result)) {
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_INVALID, "BBB rejected features: %s", result.reason);
        }
    }

    // Call internal brain API
    brain_decision_t* decision = brain_decide(brain->internal_brain, features, num_features);

    if (!decision) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INFERENCE_FAILED, "Brain prediction failed");
        return NIMCP_ERROR;
    }

    // Copy results
    strncpy(out_label, decision->label, 63);
    out_label[63] = '\0';
    *out_confidence = decision->confidence;

    // Free decision
    brain_free_decision(decision);

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
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_API_CHECK_NULL(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    NIMCP_API_CHECK_NULL(outputs, NIMCP_ERROR_NULL_ARG, "Outputs array is NULL");
    if (num_features == 0) {
        set_error("num_features must be > 0");
        return NIMCP_ERROR_INVALID;
    }
    if (num_outputs == 0) {
        set_error("num_outputs must be > 0");
        return NIMCP_ERROR_INVALID;
    }

    // Call internal brain API to get decision (which includes output vector)
    brain_decision_t* decision = brain_decide(brain->internal_brain, features, num_features);

    if (!decision) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INFERENCE_FAILED, "Brain inference failed");
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
    brain_free_decision(decision);

    set_error("No error");
    return NIMCP_OK;
}

NIMCP_EXPORT nimcp_status_t nimcp_brain_save(nimcp_brain_t brain, const char* filepath) {
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_API_CHECK_NULL(filepath, NIMCP_ERROR_NULL_ARG, "Filepath is NULL");

    // Call internal brain API
    bool success = brain_save(brain->internal_brain, filepath);

    if (!success) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_FILE_WRITE, "Failed to save brain to '%s'", filepath);
        return NIMCP_ERROR_IO;
    }

    set_error("No error");
    return NIMCP_OK;
}

NIMCP_EXPORT nimcp_brain_t nimcp_brain_load(const char* filepath) {
    NIMCP_API_CHECK_NULL_RET_NULL(filepath, "Filepath is NULL");

    // Allocate handle
    nimcp_brain_t handle = (nimcp_brain_t)nimcp_malloc(sizeof(struct nimcp_brain_handle));
    NIMCP_API_CHECK_ALLOC_SIZE(handle, sizeof(struct nimcp_brain_handle),
        "Failed to allocate brain handle for load");

    // Load internal brain
    handle->internal_brain = brain_load(filepath);

    if (!handle->internal_brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_FILE_READ, "Failed to load brain from '%s'", filepath);
        set_error("Failed to load brain from file");
        nimcp_free(handle);
        return NULL;
    }

    set_error("No error");
    return handle;
}

NIMCP_EXPORT nimcp_brain_t nimcp_brain_create_from_config(const char* config_filepath) {
    NIMCP_API_CHECK_NULL_RET_NULL(config_filepath, "Config filepath is NULL");

    // Load configuration from YAML/JSON
    nimcp_brain_config_t config;
    if (!nimcp_config_load(config_filepath, &config)) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_FILE_READ, "Failed to load config from '%s'", config_filepath);
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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_BRAIN_CREATION, "Failed to create brain from config '%s'", config_filepath);
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
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain is NULL");
    NIMCP_API_CHECK_NULL(probe, NIMCP_ERROR_NULL_ARG, "Probe output structure is NULL");

    // Get internal brain statistics
    brain_stats_t internal_stats;
    if (!brain_get_stats(brain->internal_brain, &internal_stats)) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to get brain statistics");
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

    // Get input/output dimensions from internal brain
    probe->num_inputs = brain_get_num_inputs(brain->internal_brain);
    probe->num_outputs = brain->internal_brain->config.num_outputs;

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

//=============================================================================
// Bio-Async Brain Probe Broadcasting (Loose Coupling)
//=============================================================================

// Module context for brain bio-router registration (lazy init)
static bio_module_context_t g_brain_module_ctx = NULL;

/**
 * @brief Get or create the brain module context for bio-async messaging
 */
static bio_module_context_t get_brain_module_ctx(void) {
    if (!bio_router_is_initialized()) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "get_brain_module_ctx: bio_router_is_initialized is NULL");
        return NULL;
    }

    if (!g_brain_module_ctx) {
        bio_module_info_t info = {
            .module_id = BIO_MODULE_BRAIN,
            .module_name = "brain",
            .inbox_capacity = 64,
            .user_data = NULL
        };
        g_brain_module_ctx = bio_router_register_module(&info);
    }
    return g_brain_module_ctx;
}

/**
 * @brief Broadcast brain probe data via bio-async for decoupled metrics collection
 *
 * WHAT: Sends brain probe metrics to all interested subscribers via bio-async
 * WHY:  Enables loose coupling - metrics module receives data without direct dependency
 * HOW:  Fills bio_msg_brain_probe_data_t and broadcasts via BIO_MSG_BRAIN_PROBE_DATA
 *
 * @param brain Brain handle to probe and broadcast
 * @return NIMCP_OK on success, error code otherwise
 */
NIMCP_EXPORT nimcp_status_t nimcp_brain_broadcast_probe(nimcp_brain_t brain) {
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Invalid brain handle for probe broadcast");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_NULL_ARG, "Brain has NULL internal_brain");

    // Get probe data
    nimcp_brain_probe_t probe;
    nimcp_status_t status = nimcp_brain_probe(brain, &probe);
    if (status != NIMCP_OK) {
        LOG_ERROR("Failed to get brain probe data for broadcast");
        return status;
    }

    // Get module context for bio-async
    bio_module_context_t ctx = get_brain_module_ctx();
    if (!ctx) {
        LOG_DEBUG("Bio-router not available, skipping probe broadcast");
        return NIMCP_OK;  // Not an error - router may not be initialized
    }

    // Build bio-async message
    bio_msg_brain_probe_data_t msg;
    memset(&msg, 0, sizeof(msg));

    // Initialize header
    bio_msg_init_header(&msg.header, BIO_MSG_BRAIN_PROBE_DATA,
                        BIO_MODULE_BRAIN, BIO_MODULE_ALL,
                        sizeof(bio_msg_brain_probe_data_t));
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;

    // Fill probe data
    msg.brain_id = (uint64_t)(uintptr_t)brain;  // Use pointer as unique ID
    strncpy(msg.task_name, probe.task_name, sizeof(msg.task_name) - 1);
    msg.task_name[sizeof(msg.task_name) - 1] = '\0';
    msg.size = (uint32_t)probe.size;
    msg.task = (uint32_t)probe.task;
    msg.num_neurons = probe.num_neurons;
    msg.num_synapses = probe.num_synapses;
    msg.num_active_synapses = probe.num_active_synapses;
    msg.total_inferences = probe.total_inferences;
    msg.total_learning_steps = probe.total_learning_steps;
    msg.avg_sparsity = probe.avg_sparsity;
    msg.avg_inference_time_us = probe.avg_inference_time_us;
    msg.current_learning_rate = probe.current_learning_rate;
    msg.accuracy = probe.accuracy;
    msg.memory_bytes = probe.memory_bytes;
    msg.num_inputs = probe.num_inputs;
    msg.num_outputs = probe.num_outputs;
    msg.is_cow_clone = probe.is_cow_clone;
    msg.cow_ref_count = probe.cow_ref_count;
    msg.cow_shared_bytes = probe.cow_shared_bytes;
    msg.cow_private_bytes = probe.cow_private_bytes;

    // Broadcast to all subscribers
    nimcp_error_t err = bio_router_broadcast(ctx, &msg, sizeof(msg));
    if (err != NIMCP_SUCCESS) {
        LOG_WARN("Failed to broadcast brain probe: %d", err);
        return NIMCP_ERROR;
    }

    LOG_DEBUG("Brain probe broadcast: brain_id=%llu, neurons=%u, synapses=%u",
              (unsigned long long)msg.brain_id, msg.num_neurons, msg.num_synapses);

    set_error("No error");
    return NIMCP_OK;
}

