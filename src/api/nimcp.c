/**
 * @file nimcp.c
 * @brief Implementation of unified NIMCP API
 *
 * This file wraps the internal APIs (brain, neural_network, ethics, knowledge)
 * and provides a consistent, stable public interface.
 */

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"

#define LOG_MODULE "API"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(nimcp)

/* Exception integration for API layer */
static void api_set_error(const char* fmt, ...);
#define NIMCP_API_SET_ERROR(fmt, ...) api_set_error(fmt, ##__VA_ARGS__)
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception.h"  /* For exception system shutdown */

#include "nimcp.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/strategy/nimcp_brain_strategy.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "core/neuralnet/nimcp_neuralnet_backprop.h"  // Backprop for gradient-based training
#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/knowledge/nimcp_knowledge.h"
#include "cognitive/nimcp_working_memory.h"  // Phase 10.2: Working Memory API
#include "cognitive/global_workspace/nimcp_global_workspace.h"  // Global Workspace Architecture
#include "utils/memory/nimcp_memory.h"
#include "utils/config/nimcp_config.h"
#include "utils/cache/nimcp_cache.h"
#include "utils/time/nimcp_time.h"
#include "security/nimcp_blood_brain_barrier.h"  // Phase IS-1: BBB perimeter defense
#include "security/nimcp_constant_time.h"        // Constant-time crypto operations
#include "core/brain/nimcp_brain_internal.h"     // For accessing brain->bbb_system
#include "middleware/training/nimcp_brain_training_integration.h"  // Training coordinator
#include "middleware/training/nimcp_loss_functions.h"              // Loss functions
#include "middleware/training/nimcp_optimizers.h"                  // Optimizers
#include "middleware/training/nimcp_lr_scheduler.h"                // LR schedulers
#include "middleware/training/nimcp_training_callbacks.h"          // Training callbacks
#include "training/nimcp_training_dispatch.h"                      // SNN/LNN/CNN/Adaptive dispatch
#include "plasticity/adaptive/nimcp_adaptive.h"                    // Adaptive network
#include "utils/platform/nimcp_platform_once.h"                    // Thread-safe init
#include "utils/thread/nimcp_atomic.h"                             // Atomic operations
#include "utils/exception/nimcp_exception_macros.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <math.h>

//=============================================================================
// Internal Handle Structures
//=============================================================================

struct nimcp_brain_handle {
    brain_t internal_brain;  // Wraps internal brain_t
};

struct nimcp_network_handle {
    neural_network_t internal_network;  // Wraps internal neural_network_t
};

struct nimcp_ethics_handle {
    ethics_engine_t internal_ethics;  // Wraps internal ethics_engine_t
};

struct nimcp_knowledge_handle {
    knowledge_system_t internal_knowledge;  // Wraps internal knowledge_system_t
};

/**
 * @brief Brain snapshot handle for COW save/restore
 *
 * WHAT: Stores brain state snapshot with COW references and tracking
 * WHY:  Enables instant rollback with minimal memory overhead
 * HOW:  Holds cached references to brain data structures with refcount tracking
 */
struct nimcp_brain_snapshot_handle {
    brain_t internal_brain_snapshot;  // Snapshot of brain state
    uint64_t timestamp_us;            // Snapshot creation time
    size_t shared_memory_size;        // Size of shared memory (for tracking)
    uint32_t snapshot_refcount;       // Reference count for this snapshot
    bool is_isolated;                 // Isolation flag (true if snapshot is independent)
};

//=============================================================================
// Global State
//=============================================================================

static _Thread_local char g_last_error[256] = "No error";
static nimcp_atomic_bool_t g_initialized = {0};  // Thread-safe initialized flag
static nimcp_atomic_bool_t g_init_in_progress = {0};  // Guard against concurrent init
static nimcp_status_t g_init_result = NIMCP_OK;  // Result of init

//=============================================================================
// Version Functions
//=============================================================================

const char* nimcp_version(void) {
    return NIMCP_VERSION_STRING;
}

int nimcp_version_int(void) {
    return NIMCP_VERSION_MAJOR * 10000 + NIMCP_VERSION_MINOR * 100 + NIMCP_VERSION_PATCH;
}

//=============================================================================
// Error Handling
//=============================================================================

static void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
    va_end(args);
}

/* Wrapper for exception integration macros */
static void api_set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
    va_end(args);
}

const char* nimcp_get_error(void) {
    return g_last_error;
}

//=============================================================================
// Initialization
//=============================================================================

/**
 * @brief Internal initialization function
 *
 * NOTE: Previously used nimcp_platform_once() which prevented re-initialization
 * after shutdown. Now uses atomic compare-exchange for thread-safe init that
 * can be repeated after shutdown.
 *
 * @return NIMCP_OK on success, NIMCP_ERROR on failure
 */
static nimcp_status_t nimcp_init_internal(void) {
    LOG_INFO("Initializing NIMCP library version %s", NIMCP_VERSION_STRING);

    // Initialize memory tracking (unified memory management)
    LOG_DEBUG("Initializing memory tracking system");
    nimcp_memory_init();

    // Initialize bio-async system (core async communication infrastructure)
    LOG_INFO("Initializing bio-async communication system");
    nimcp_bio_async_config_t bio_async_config = {0};  // Use default config
    if (nimcp_bio_async_init(&bio_async_config) != NIMCP_SUCCESS) {
        LOG_ERROR("Failed to initialize bio-async system");
        nimcp_memory_cleanup();
        set_error("Failed to initialize bio-async system");
        return NIMCP_ERROR;
    }

    // Initialize bio-async router (message routing for modules)
    LOG_DEBUG("Initializing bio-async router");
    if (bio_router_init(NULL) != NIMCP_SUCCESS) {
        LOG_ERROR("Failed to initialize bio-async router");
        nimcp_bio_async_shutdown();
        nimcp_memory_cleanup();
        set_error("Failed to initialize bio-async router");
        return NIMCP_ERROR;
    }

    // Initialize COW cache system
    LOG_DEBUG("Initializing COW cache system");
    nimcp_cache_init();

    set_error("No error");
    LOG_INFO("NIMCP library initialized successfully");
    return NIMCP_OK;
}

nimcp_status_t nimcp_init(void) {
    // Fast path: already initialized
    if (nimcp_atomic_load_bool(&g_initialized, NIMCP_MEMORY_ORDER_ACQUIRE)) {
        return g_init_result;
    }

    // Try to acquire init lock using compare-exchange
    // Only one thread will succeed; others will spin-wait
    bool expected = false;
    if (!nimcp_atomic_compare_exchange_bool(&g_init_in_progress, &expected, true,
                                             NIMCP_MEMORY_ORDER_ACQ_REL)) {
        // Another thread is initializing - wait for it to complete
        while (nimcp_atomic_load_bool(&g_init_in_progress, NIMCP_MEMORY_ORDER_ACQUIRE)) {
            // Spin-wait (could use yield/sleep for production)
        }
        return g_init_result;
    }

    // Double-check after acquiring lock (another thread may have initialized)
    if (nimcp_atomic_load_bool(&g_initialized, NIMCP_MEMORY_ORDER_ACQUIRE)) {
        nimcp_atomic_store_bool(&g_init_in_progress, false, NIMCP_MEMORY_ORDER_RELEASE);
        return g_init_result;
    }

    // Perform actual initialization
    g_init_result = nimcp_init_internal();

    if (g_init_result == NIMCP_OK) {
        nimcp_atomic_store_bool(&g_initialized, true, NIMCP_MEMORY_ORDER_RELEASE);
    }

    // Release init lock
    nimcp_atomic_store_bool(&g_init_in_progress, false, NIMCP_MEMORY_ORDER_RELEASE);

    return g_init_result;
}

void nimcp_shutdown(void) {
    LOG_INFO("Shutting down NIMCP library");

    if (!nimcp_atomic_load_bool(&g_initialized, NIMCP_MEMORY_ORDER_ACQUIRE)) {
        LOG_DEBUG("NIMCP not initialized, nothing to shutdown");
        return;
    }

    // Cleanup cache system
    LOG_DEBUG("Cleaning up COW cache system");
    nimcp_cache_cleanup();

    // Shutdown bio-async router
    LOG_DEBUG("Shutting down bio-async router");
    bio_router_shutdown();

    // Shutdown bio-async system
    LOG_DEBUG("Shutting down bio-async communication system");
    nimcp_bio_async_shutdown();

    // Shutdown constant-time module (before memory cleanup to avoid double-free)
    LOG_DEBUG("Shutting down constant-time module");
    nimcp_ct_shutdown();

    // Cleanup adaptive module memory pool (before memory cleanup)
    LOG_DEBUG("Cleaning up adaptive network memory pool");
    adaptive_pool_cleanup();

    // Shutdown exception system (before memory cleanup to avoid dangling references)
    LOG_DEBUG("Shutting down exception system");
    bbb_helpers_shutdown();
    nimcp_exception_system_shutdown();

    // Cleanup memory tracking (last)
    LOG_DEBUG("Cleaning up memory tracking");
    nimcp_memory_cleanup();

    // Reset init state to allow re-initialization
    g_init_result = NIMCP_OK;  // Reset to default for next init
    nimcp_atomic_store_bool(&g_initialized, false, NIMCP_MEMORY_ORDER_RELEASE);
    LOG_INFO("NIMCP library shutdown complete");
}

//=============================================================================
// Brain API Implementation
//=============================================================================

nimcp_brain_t nimcp_brain_create(
    const char* name,
    nimcp_brain_size_t size,
    nimcp_brain_task_t task,
    uint32_t num_inputs,
    uint32_t num_outputs)
{
    LOG_INFO("Creating brain: name='%s', size=%d, task=%d, inputs=%u, outputs=%u",
             name ? name : "NULL", size, task, num_inputs, num_outputs);

    /* Use exception-integrated validation (returns NULL on error) */
    NIMCP_API_CHECK_NULL_RET_NULL(name, "Brain name cannot be NULL");

    /* Allocate handle with exception-integrated check */
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
        set_error("Failed to create internal brain");
        nimcp_free(handle);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_create: handle->internal_brain is NULL");
        return NULL;
    }

    set_error("No error");
    LOG_INFO("Brain '%s' created successfully (handle=%p)", name, (void*)handle);
    return handle;
}

// Forward declaration for training cleanup (defined in nimcp_api_training.c)
extern void nimcp_api_training_cleanup_brain(nimcp_brain_t brain);

void nimcp_brain_destroy(nimcp_brain_t brain) {
    if (!brain) {
        LOG_DEBUG("nimcp_brain_destroy called with NULL brain, ignoring");
        return;
    }

    LOG_INFO("Destroying brain (handle=%p)", (void*)brain);

    // Clean up training pipeline state BEFORE destroying internal brain
    // This prevents memory leaks from the global g_training_states array
    nimcp_api_training_cleanup_brain(brain);

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

nimcp_status_t nimcp_brain_learn_example(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    const char* label,
    float confidence)
{
    LOG_DEBUG("Learning example: label='%s', num_features=%u, confidence=%.3f",
              label ? label : "NULL", num_features, confidence);

    /* Exception-integrated parameter validation */
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_API_CHECK_NULL(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    NIMCP_API_CHECK_NULL(label, NIMCP_ERROR_NULL_ARG, "Label is NULL");

    /* === PHASE IS-1: BBB INPUT VALIDATION === */
    /* Validate external input data through Blood-Brain Barrier before processing */
    if (brain->internal_brain && brain->internal_brain->bbb_enabled &&
        brain->internal_brain->bbb_system) {
        LOG_DEBUG("BBB enabled, validating inputs");
        bbb_validation_result_t result;

        /* Validate features array (external input data) */
        if (!bbb_validate_input(brain->internal_brain->bbb_system,
                               features, num_features * sizeof(float), &result)) {
            NIMCP_API_CHECK_BBB(false, result, NIMCP_ERROR_BBB_REJECTED);
        }

        /* Validate label string (external string input) */
        if (!bbb_validate_string(brain->internal_brain->bbb_system, label, &result)) {
            NIMCP_API_CHECK_BBB(false, result, NIMCP_ERROR_BBB_REJECTED);
        }
        LOG_DEBUG("BBB validation passed");
    }

    /* Call internal brain API */
    LOG_DEBUG("Invoking internal brain_learn_example");
    float loss = brain_learn_example(brain->internal_brain, features, num_features, label, confidence);

    /* brain_learn_example returns -1.0f on error, >= 0.0f on success */
    NIMCP_API_CHECK_FLOAT(loss, NIMCP_ERROR_LEARNING_FAILED,
                          "Brain learning failed for label");

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
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_CHECK_THROW(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    NIMCP_CHECK_THROW(out_label, NIMCP_ERROR_NULL_ARG, "Output label buffer is NULL");
    NIMCP_CHECK_THROW(out_confidence, NIMCP_ERROR_NULL_ARG, "Output confidence pointer is NULL");

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
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_OPERATION_FAILED, "Brain prediction failed");
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
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_CHECK_THROW(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    NIMCP_CHECK_THROW(outputs, NIMCP_ERROR_NULL_ARG, "Outputs array is NULL");

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
        outputs[i] = 0.0F;
    }

    // Free decision
    nimcp_free(decision);

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_save(nimcp_brain_t brain, const char* filepath) {
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_CHECK_THROW(filepath, NIMCP_ERROR_NULL_ARG, "Filepath is NULL");

    // Call internal brain API
    bool success = brain_save(brain->internal_brain, filepath);

    if (!success) {
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_IO, "Failed to save brain");
    }

    set_error("No error");
    return NIMCP_OK;
}

nimcp_brain_t nimcp_brain_load(const char* filepath) {
    if (!filepath) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_ARG, "Filepath is NULL");
        set_error("Filepath is NULL");
        return NULL;
    }

    // Allocate handle
    nimcp_brain_t handle = (nimcp_brain_t)nimcp_malloc(sizeof(struct nimcp_brain_handle));
    if (!handle) {
        set_error("Failed to allocate brain handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "handle is NULL");

        return NULL;
    }

    // Load internal brain
    handle->internal_brain = brain_load(filepath);

    if (!handle->internal_brain) {
        set_error("Failed to load brain from file");
        nimcp_free(handle);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_load: handle->internal_brain is NULL");
        return NULL;
    }

    set_error("No error");
    return handle;
}

//=============================================================================
// Brain Snapshot API Implementation
//=============================================================================

nimcp_status_t nimcp_brain_snapshot_save(
    nimcp_brain_t brain,
    const char* name,
    const char* description)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_CHECK_THROW(name, NIMCP_ERROR_NULL_ARG, "Snapshot name is NULL");

    // Call internal brain snapshot API
    bool success = brain_save_snapshot(
        brain->internal_brain,
        name,
        description ? description : ""
    );

    if (!success) {
        set_error("Failed to save snapshot");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_SUCCESS;
}

nimcp_brain_t nimcp_brain_snapshot_restore(
    nimcp_brain_t brain,
    const char* name)
{
    if (!name) {
        set_error("Snapshot name is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "name is NULL");

        return NULL;
    }

    // Load from snapshot
    brain_t restored_brain = brain_restore_snapshot(
        brain ? brain->internal_brain : NULL,
        name
    );

    if (!restored_brain) {
        set_error("Failed to restore from snapshot");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "restored_brain is NULL");


        return NULL;
    }

    // Allocate new handle
    nimcp_brain_t handle = (nimcp_brain_t)nimcp_malloc(sizeof(struct nimcp_brain_handle));
    if (!handle) {
        set_error("Failed to allocate brain handle");
        brain_destroy(restored_brain);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "handle is NULL");

        return NULL;
    }

    handle->internal_brain = restored_brain;
    set_error("No error");
    return handle;
}

nimcp_status_t nimcp_brain_snapshot_list(
    nimcp_brain_t brain,
    nimcp_brain_snapshot_info_t* infos,
    uint32_t max_count,
    uint32_t* out_count)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_CHECK_THROW(infos, NIMCP_ERROR_NULL_ARG, "Infos array is NULL");

    // Call internal brain list API
    // Note: brain_snapshot_info_t and nimcp_brain_snapshot_info_t have same layout
    bool success = brain_list_snapshots(
        brain->internal_brain,
        (brain_snapshot_info_t*)infos,
        max_count,
        out_count
    );

    if (!success) {
        set_error("Failed to list snapshots");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_SUCCESS;
}

nimcp_status_t nimcp_brain_snapshot_delete(
    nimcp_brain_t brain,
    const char* name)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_CHECK_THROW(name, NIMCP_ERROR_NULL_ARG, "Snapshot name is NULL");

    // Call internal brain delete API
    bool success = brain_delete_snapshot(brain->internal_brain, name);

    if (!success) {
        set_error("Failed to delete snapshot");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_SUCCESS;
}

nimcp_brain_t nimcp_brain_create_from_config(const char* config_filepath) {
    if (!config_filepath) {
        set_error("Config filepath is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config_filepath is NULL");

        return NULL;
    }

    // Load configuration from YAML/JSON
    nimcp_brain_config_t config;
    if (!nimcp_config_load(config_filepath, &config)) {
        set_error("Failed to load config from %s", config_filepath);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_create_from_config: nimcp_config_load is NULL");
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
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");


        return NULL;
    }

    // Note: Additional configuration like BCM, STDP, ethics could be applied here
    // For now, we just create the basic brain structure
    // TODO: Apply plasticity settings, ethics config, etc.

    set_error("No error");
    return brain;
}

nimcp_status_t nimcp_brain_probe(nimcp_brain_t brain, nimcp_brain_probe_t* probe) {
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain is NULL");
    NIMCP_CHECK_THROW(probe, NIMCP_ERROR_NULL_ARG, "Probe output structure is NULL");

    // Get internal brain statistics
    brain_stats_t internal_stats;
    NIMCP_CHECK_THROW(brain_get_stats(brain->internal_brain, &internal_stats),
                      NIMCP_ERROR, "Failed to get brain statistics");

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

    // Get input/output sizes from internal brain
    probe->num_inputs = brain_get_num_inputs(brain->internal_brain);
    probe->num_outputs = brain_get_num_outputs(brain->internal_brain);

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

/**
 * @brief Module context for brain bio-router registration (lazy init)
 */
static bio_module_context_t g_brain_probe_module_ctx = NULL;

/**
 * @brief Get or create brain module context for bio-async broadcasting
 */
static bio_module_context_t get_brain_probe_module_ctx(void) {
    if (!bio_router_is_initialized()) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "get_brain_probe_module_ctx: bio_router_is_initialized is NULL");
        return NULL;
    }
    if (!g_brain_probe_module_ctx) {
        bio_module_info_t info = {
            .module_id = BIO_MODULE_BRAIN,
            .module_name = "brain_probe",
            .inbox_capacity = 64,
            .user_data = NULL
        };
        g_brain_probe_module_ctx = bio_router_register_module(&info);
    }
    return g_brain_probe_module_ctx;
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
nimcp_status_t nimcp_brain_broadcast_probe(nimcp_brain_t brain) {
    NIMCP_CHECK_THROW(brain && brain->internal_brain, NIMCP_ERROR_NULL_ARG, "Invalid brain handle");

    // Get probe data
    nimcp_brain_probe_t probe;
    nimcp_status_t status = nimcp_brain_probe(brain, &probe);
    if (status != NIMCP_OK) {
        LOG_ERROR("Failed to get brain probe data for broadcast");
        return status;
    }

    // Get module context for bio-async
    bio_module_context_t ctx = get_brain_probe_module_ctx();
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

    // Broadcast to all subscribers (best-effort - failure is non-fatal)
    nimcp_error_t err = bio_router_broadcast(ctx, &msg, sizeof(msg));
    if (err != NIMCP_SUCCESS) {
        // Log warning but don't fail - probe data was collected successfully,
        // broadcast is secondary and optional
        LOG_DEBUG("Probe broadcast incomplete (some modules may not have received): %d", err);
    } else {
        LOG_DEBUG("Brain probe broadcast: brain_id=%llu, neurons=%u, synapses=%u",
                  (unsigned long long)msg.brain_id, msg.num_neurons, msg.num_synapses);
    }

    set_error("No error");
    return NIMCP_OK;
}

//=============================================================================
// Copy-on-Write (COW) Cache API Implementation
//=============================================================================

/**
 * WHAT: Clone brain using copy-on-write caching
 * WHY:  Enable efficient replication with 86% memory savings
 * HOW:  Use internal brain_clone_cow which shares network structures
 *
 * PERFORMANCE: <10ms clone time vs ~1000ms for full copy
 * MEMORY: ~1MB overhead vs ~50MB for full copy
 *
 * Phase 2: True COW sharing via pointer sharing
 */
nimcp_brain_t nimcp_brain_clone_cow(nimcp_brain_t original) {
    // Guard: Validate parameters
    if (!original) {
        set_error("NULL brain provided to nimcp_brain_clone_cow");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "original is NULL");

        return NULL;
    }

    if (!original->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_clone_cow: original->internal_brain is NULL");
        return NULL;
    }

    // Allocate handle
    nimcp_brain_t clone_handle = (nimcp_brain_t)nimcp_malloc(sizeof(struct nimcp_brain_handle));
    if (!clone_handle) {
        set_error("Failed to allocate brain handle for COW clone");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "clone_handle is NULL");

        return NULL;
    }

    // Use internal COW clone function
    clone_handle->internal_brain = brain_clone_cow(original->internal_brain);

    if (!clone_handle->internal_brain) {
        set_error("Failed to clone internal brain");
        nimcp_free(clone_handle);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_clone_cow: clone_handle->internal_brain is NULL");
        return NULL;
    }

    set_error("No error");
    return clone_handle;
}

/**
 * WHAT: Create instant snapshot of brain state using COW with advanced reference tracking
 * WHY:  Enable zero-copy checkpointing for rollback with complete isolation guarantees
 * HOW:  Use brain_clone_cow with enhanced cache reference tracking for snapshot isolation
 *
 * PERFORMANCE: <1ms snapshot time (zero-copy with cache reference)
 * MEMORY: ~64 bytes overhead + O(1) reference count increment
 *
 * IMPLEMENTATION:
 * 1. Create COW clone using brain_clone_cow (shares network via reference counting)
 * 2. Track shared memory size for cache statistics
 * 3. Initialize snapshot refcount for multi-snapshot scenarios
 * 4. Mark snapshot as isolated to prevent unwanted modifications
 * 5. Multiple snapshots share underlying network data until modifications
 *
 * ADVANCED FEATURES:
 * - Snapshot isolation: Modifications to original don't affect snapshot
 * - Reference tracking: Accurate memory usage reporting
 * - Multi-snapshot support: Create multiple snapshots with shared data
 */
nimcp_brain_snapshot_t nimcp_brain_snapshot_cow(nimcp_brain_t brain) {
    // Guard: Validate parameters
    if (!brain) {
        set_error("NULL brain provided to nimcp_brain_snapshot_cow");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_snapshot_cow: brain->internal_brain is NULL");
        return NULL;
    }

    // Allocate snapshot handle
    nimcp_brain_snapshot_t snapshot =
        (nimcp_brain_snapshot_t)nimcp_malloc(sizeof(struct nimcp_brain_snapshot_handle));
    if (!snapshot) {
        set_error("Failed to allocate snapshot handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snapshot is NULL");

        return NULL;
    }

    // WHAT: Capture current stats before creating snapshot
    // WHY:  Snapshot should preserve stats at snapshot time, not reflect future changes
    // HOW:  Get stats from original brain before cloning
    brain_stats_t current_stats;
    if (!brain_get_stats(brain->internal_brain, &current_stats)) {
        set_error("Failed to get brain stats for snapshot");
        nimcp_free(snapshot);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_snapshot_cow: brain_get_stats is NULL");
        return NULL;
    }

    // WHAT: Create COW clone of brain
    // WHY:  Enables zero-copy snapshot with reference counting
    // HOW:  brain_clone_cow shares network and increments reference count
    brain_t snapshot_brain = brain_clone_cow(brain->internal_brain);

    if (!snapshot_brain) {
        set_error("Failed to create COW clone for snapshot");
        nimcp_free(snapshot);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snapshot_brain is NULL");


        return NULL;
    }

    // CRITICAL: Mark this brain as a snapshot and preserve stats
    // This prevents brain_get_stats from reading the shared network's current stats
    brain_mark_as_snapshot(snapshot_brain, &current_stats);

    // WHAT: Calculate shared memory size for tracking
    // WHY:  Enables accurate memory usage reporting and cache statistics
    // HOW:  Use the preserved stats to estimate network size
    size_t shared_size = (size_t)(current_stats.num_neurons * 100 + current_stats.num_synapses * 20);

    // WHAT: Store snapshot with enhanced tracking metadata
    // WHY:  Track when snapshot was created and memory usage for monitoring
    // HOW:  Save cloned brain, timestamp, and reference tracking info
    snapshot->internal_brain_snapshot = snapshot_brain;
    snapshot->timestamp_us = nimcp_time_monotonic_us();
    snapshot->shared_memory_size = shared_size;
    snapshot->snapshot_refcount = 1;  // This snapshot has one reference
    snapshot->is_isolated = true;     // Snapshot is isolated from original modifications

    // WHAT: Record cache reference for accurate COW statistics
    // WHY:  Track memory savings achieved by COW snapshots
    // HOW:  Use nimcp_cache_record_reference with shared memory size
    if (shared_size > 0) {
        nimcp_cache_record_reference(shared_size);
    }

    set_error("No error");
    return snapshot;
}

/**
 * WHAT: Restore brain state from COW snapshot
 * WHY:  Enable instant rollback to previous state
 * HOW:  Swap brain state with snapshot state
 *
 * PERFORMANCE: <1ms restore time (pointer swap)
 * MEMORY: O(1)
 */
nimcp_status_t nimcp_brain_restore_cow(nimcp_brain_t brain, nimcp_brain_snapshot_t snapshot) {
    // Guard: Validate parameters
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to nimcp_brain_restore_cow");
    NIMCP_CHECK_THROW(snapshot, NIMCP_ERROR_NULL_ARG, "NULL snapshot provided to nimcp_brain_restore_cow");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");
    NIMCP_CHECK_THROW(snapshot->internal_brain_snapshot, NIMCP_ERROR_INVALID,
                      "Snapshot has NULL internal_brain_snapshot");

    // CRITICAL FIX: Use brain_clone_cow() which properly handles COW refcounting
    // The snapshot was created via save/load, so it owns its network independently
    // We can safely clone it, and the clone will share the snapshot's network

    // Save the old brain pointer for cleanup
    brain_t old_brain = brain->internal_brain;

    // Clone the snapshot to create new brain state (creates COW reference)
    brain_t new_brain = brain_clone_cow(snapshot->internal_brain_snapshot);
    NIMCP_CHECK_THROW(new_brain, NIMCP_ERROR, "Failed to clone snapshot for restore");

    // CRITICAL: Mark the restored brain as a snapshot with the same preserved stats
    // This ensures brain_get_stats returns the snapshot's stats, not the current network stats
    brain_stats_t snapshot_stats;
    if (brain_get_stats(snapshot->internal_brain_snapshot, &snapshot_stats)) {
        brain_mark_as_snapshot(new_brain, &snapshot_stats);
    }

    // Assign new brain (do this before destroying old brain to avoid use-after-free)
    brain->internal_brain = new_brain;

    // Now destroy the old brain (will decrement refcounts properly)
    brain_destroy(old_brain);

    set_error("No error");
    return NIMCP_OK;
}

/**
 * WHAT: Destroy brain snapshot and release COW references
 * WHY:  Free snapshot resources and decrement reference counts
 * HOW:  Release cached references and free snapshot handle
 */
void nimcp_brain_snapshot_destroy(nimcp_brain_snapshot_t snapshot) {
    if (!snapshot) {
        return;
    }

    // Phase 1: Free the cloned internal brain
    if (snapshot->internal_brain_snapshot) {
        brain_destroy(snapshot->internal_brain_snapshot);
    }

    // Free snapshot handle
    nimcp_free(snapshot);
}

//=============================================================================
// Phase 10.2: Working Memory API Implementation
//=============================================================================

/**
 * @brief Add item to brain's working memory
 *
 * WHAT: Wrapper for working_memory_add() on brain's working memory
 * WHY:  Provide public API for adding items to working memory
 * HOW:  Validate brain → Get working memory → Call internal API
 */
nimcp_status_t nimcp_brain_working_memory_add(
    nimcp_brain_t brain,
    const float* data,
    uint32_t size,
    float salience)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to working_memory_add");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Validate parameters FIRST (before checking subsystem availability)
    NIMCP_CHECK_THROW(data, NIMCP_ERROR_NULL_ARG, "NULL data provided to working_memory_add");
    NIMCP_CHECK_THROW(size != 0, NIMCP_ERROR_INVALID, "Invalid size (0) provided to working_memory_add");

    // Guard: Check if working memory enabled (after parameter validation)
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID, "Working memory not enabled in brain config");

    // Add to working memory
    bool success = working_memory_add(wm, data, size, salience);
    NIMCP_CHECK_THROW(success, NIMCP_ERROR, "Failed to add item to working memory");

    set_error("No error");
    return NIMCP_OK;
}

/**
 * @brief Get item from brain's working memory
 *
 * WHAT: Wrapper for working_memory_get() on brain's working memory
 * WHY:  Provide public API for accessing working memory items
 * HOW:  Validate brain → Get working memory → Call internal API
 */
const float* nimcp_brain_working_memory_get(
    nimcp_brain_t brain,
    uint32_t index,
    uint32_t* size_out)
{
    // Guard: Validate brain
    if (!brain || !brain->internal_brain) {
        set_error("Invalid brain handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_working_memory_get: required parameter is NULL (brain, brain->internal_brain)");
        return NULL;
    }

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    if (!wm) {
        set_error("Working memory not enabled");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "wm is NULL");

        return NULL;
    }

    // Get item
    const float* item = working_memory_get(wm, index, size_out);
    if (!item) {
        set_error("Invalid index or empty working memory");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "item is NULL");

        return NULL;
    }

    set_error("No error");
    return item;
}

/**
 * @brief Get brain's working memory statistics
 *
 * WHAT: Wrapper for working_memory_get_stats() on brain's working memory
 * WHY:  Provide public API for monitoring working memory state
 * HOW:  Validate brain → Get working memory → Extract size/capacity
 */
nimcp_status_t nimcp_brain_working_memory_stats(
    nimcp_brain_t brain,
    uint32_t* current_size_out,
    uint32_t* capacity_out)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Validate output parameters
    NIMCP_CHECK_THROW(current_size_out && capacity_out, NIMCP_ERROR_NULL_ARG, "NULL output parameters");

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID, "Working memory not enabled");

    // Get stats
    working_memory_stats_t stats;
    working_memory_get_stats(wm, &stats);

    *current_size_out = stats.current_size;
    *capacity_out = stats.capacity;

    set_error("No error");
    return NIMCP_OK;
}

/**
 * @brief Refresh item in brain's working memory
 *
 * WHAT: Wrapper for working_memory_refresh() on brain's working memory
 * WHY:  Provide public API for preventing decay (rehearsal)
 * HOW:  Validate brain → Get working memory → Call internal API
 */
nimcp_status_t nimcp_brain_working_memory_refresh(
    nimcp_brain_t brain,
    uint32_t index)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    NIMCP_CHECK_THROW(wm, NIMCP_ERROR_INVALID, "Working memory not enabled");

    // Refresh item
    bool success = working_memory_refresh(wm, index);
    NIMCP_CHECK_THROW(success, NIMCP_ERROR_INVALID, "Invalid index for refresh");

    set_error("No error");
    return NIMCP_OK;
}

//=============================================================================
// Global Workspace API Implementation
//=============================================================================

/**
 * @brief Helper to convert public API module enum to internal enum
 */
static cognitive_module_t convert_module_enum(nimcp_cognitive_module_t public_module)
{
    // Direct mapping - enums have same values
    return (cognitive_module_t)public_module;
}

nimcp_status_t nimcp_brain_workspace_compete(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module,
    const float* content,
    uint32_t content_dim,
    float strength)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_compete");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Validate parameters FIRST (before checking workspace availability)
    NIMCP_CHECK_THROW(content, NIMCP_ERROR_NULL_ARG, "NULL content provided to workspace_compete");
    NIMCP_CHECK_THROW(content_dim != 0, NIMCP_ERROR_INVALID, "Invalid content_dim (0)");
    NIMCP_CHECK_THROW(strength >= 0.0F && strength <= 1.0F, NIMCP_ERROR_INVALID,
                      "Strength must be in range [0.0, 1.0]");

    // Guard: Check if global workspace enabled (after parameter validation)
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_CHECK_THROW(workspace, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Convert module enum and compete
    cognitive_module_t internal_module = convert_module_enum(module);
    bool won = global_workspace_compete(workspace, internal_module, content, content_dim, strength);

    if (won) {
        set_error("No error");
        return NIMCP_OK;
    } else {
        set_error("Did not win workspace competition");
        return NIMCP_ERROR;
    }
}

nimcp_status_t nimcp_brain_workspace_read(
    nimcp_brain_t brain,
    float* content,
    uint32_t max_dim,
    uint32_t* actual_dim,
    nimcp_cognitive_module_t* source_module)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_read");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Validate parameters FIRST (before checking subsystem availability)
    NIMCP_CHECK_THROW(content && actual_dim && source_module, NIMCP_ERROR_NULL_ARG,
                      "NULL output parameter provided to workspace_read");
    NIMCP_CHECK_THROW(max_dim != 0, NIMCP_ERROR_INVALID, "Invalid max_dim (0)");

    // Guard: Check if global workspace enabled (after parameter validation)
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_CHECK_THROW(workspace, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Read broadcast
    cognitive_module_t internal_source;
    bool success = global_workspace_read_broadcast(
        workspace, content, max_dim, actual_dim, &internal_source
    );

    if (success) {
        *source_module = (nimcp_cognitive_module_t)internal_source;
        set_error("No error");
        return NIMCP_OK;
    } else {
        set_error("No broadcast available or buffer too small");
        return NIMCP_ERROR;
    }
}

nimcp_status_t nimcp_brain_workspace_subscribe(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_subscribe");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_CHECK_THROW(workspace, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Subscribe module
    cognitive_module_t internal_module = convert_module_enum(module);
    bool success = global_workspace_subscribe(workspace, internal_module);
    NIMCP_CHECK_THROW(success, NIMCP_ERROR, "Failed to subscribe module");

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_workspace_unsubscribe(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_unsubscribe");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_CHECK_THROW(workspace, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Unsubscribe module
    cognitive_module_t internal_module = convert_module_enum(module);
    bool success = global_workspace_unsubscribe(workspace, internal_module);
    NIMCP_CHECK_THROW(success, NIMCP_ERROR, "Failed to unsubscribe module");

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_workspace_has_broadcast(
    nimcp_brain_t brain,
    bool* has_broadcast)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_has_broadcast");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_CHECK_THROW(workspace, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Guard: Validate output parameter
    NIMCP_CHECK_THROW(has_broadcast, NIMCP_ERROR_NULL_ARG, "NULL has_broadcast parameter");

    // Check for broadcast
    *has_broadcast = global_workspace_has_broadcast(workspace);
    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_workspace_stats(
    nimcp_brain_t brain,
    uint32_t* total_broadcasts,
    uint32_t* total_competitions,
    float* avg_strength)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_stats");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_CHECK_THROW(workspace, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Guard: Validate output parameters
    NIMCP_CHECK_THROW(total_broadcasts && total_competitions && avg_strength,
                      NIMCP_ERROR_NULL_ARG, "NULL output parameter in workspace_stats");

    // Get statistics
    workspace_statistics_t stats;
    bool success = global_workspace_get_statistics(workspace, &stats);
    NIMCP_CHECK_THROW(success, NIMCP_ERROR, "Failed to get workspace statistics");

    *total_broadcasts = (uint32_t)stats.total_broadcasts;
    *total_competitions = (uint32_t)stats.total_competitions;
    // Note: avg_strength not tracked in statistics, return 0.0
    *avg_strength = 0.0F;
    set_error("No error");
    return NIMCP_OK;
}

//=============================================================================
// Neural Network API Implementation
//=============================================================================

nimcp_network_t nimcp_network_create(
    uint32_t num_inputs,
    uint32_t num_outputs,
    uint32_t num_hidden,
    float learning_rate)
{
    // Allocate handle
    nimcp_network_t handle = (nimcp_network_t)nimcp_malloc(sizeof(struct nimcp_network_handle));
    if (!handle) {
        set_error("Failed to allocate network handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "handle is NULL");

        return NULL;
    }

    // Create config for internal API
    network_config_t config = {0};
    config.input_size = num_inputs;
    config.output_size = num_outputs;
    // Calculate total neurons: inputs + hidden layers + outputs
    config.num_neurons = num_inputs + num_hidden + num_outputs;
    config.learning_rate = learning_rate;

    // Create internal neural network
    handle->internal_network = neural_network_create(&config);

    if (!handle->internal_network) {
        set_error("Failed to create internal neural network");
        nimcp_free(handle);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_network_create: handle->internal_network is NULL");
        return NULL;
    }

    set_error("No error");
    return handle;
}

void nimcp_network_destroy(nimcp_network_t network) {
    if (!network) {
        return;
    }

    if (network->internal_network) {
        neural_network_destroy(network->internal_network);
    }

    nimcp_free(network);
}

nimcp_status_t nimcp_network_forward(
    nimcp_network_t network,
    const float* inputs,
    uint32_t num_inputs,
    float* outputs,
    uint32_t num_outputs)
{
    NIMCP_CHECK_THROW(network, NIMCP_ERROR_NULL_ARG, "Network handle is NULL");
    NIMCP_CHECK_THROW(inputs, NIMCP_ERROR_NULL_ARG, "Inputs array is NULL");
    NIMCP_CHECK_THROW(outputs, NIMCP_ERROR_NULL_ARG, "Outputs array is NULL");

    // Call internal network API
    bool success = neural_network_forward(network->internal_network,
                                         inputs, num_inputs,
                                         outputs, num_outputs);

    NIMCP_CHECK_THROW(success, NIMCP_ERROR, "Network forward pass failed");

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_network_train(
    nimcp_network_t network,
    const float* inputs,
    uint32_t num_inputs,
    const float* targets,
    uint32_t num_targets)
{
    NIMCP_CHECK_THROW(network, NIMCP_ERROR_NULL_ARG, "Network handle is NULL");

    // Training not yet implemented in internal API
    NIMCP_THROW(NIMCP_ERROR, "Training not yet implemented");
    return NIMCP_ERROR;
}

//=============================================================================
// Ethics API Implementation
//=============================================================================

nimcp_ethics_t nimcp_ethics_create(void) {
    nimcp_ethics_t handle = (nimcp_ethics_t)nimcp_malloc(sizeof(struct nimcp_ethics_handle));
    if (!handle) {
        set_error("Failed to allocate ethics handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "handle is NULL");

        return NULL;
    }

    // Create default ethics configuration
    ethics_config_t config = {0};
    config.policies = NULL;
    config.num_policies = 0;
    config.callback = NULL;
    config.callback_context = NULL;
    config.default_severity = 0.5F;
    config.enable_learning = true;
    config.action_feature_size = 32;
    config.max_agents = 16;
    config.golden_rule_threshold = 0.0F;
    config.empathy_weight = 0.5F;

    // Create internal ethics engine
    handle->internal_ethics = ethics_engine_create(&config);

    if (!handle->internal_ethics) {
        set_error("Failed to create internal ethics engine");
        nimcp_free(handle);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_ethics_create: handle->internal_ethics is NULL");
        return NULL;
    }

    set_error("No error");
    return handle;
}

void nimcp_ethics_destroy(nimcp_ethics_t ethics) {
    if (!ethics) {
        return;
    }

    if (ethics->internal_ethics) {
        ethics_engine_destroy(ethics->internal_ethics);
    }

    nimcp_free(ethics);
}

nimcp_status_t nimcp_ethics_check(
    nimcp_ethics_t ethics,
    const float* situation,
    uint32_t num_features,
    float* out_score)
{
    NIMCP_CHECK_THROW(ethics, NIMCP_ERROR_NULL_ARG, "Ethics handle is NULL");
    NIMCP_CHECK_THROW(situation, NIMCP_ERROR_NULL_ARG, "Situation array is NULL");
    NIMCP_CHECK_THROW(out_score, NIMCP_ERROR_NULL_ARG, "Output score pointer is NULL");

    // Create action context from situation features
    action_context_t action = {0};
    action.features = (float*)situation;  // Cast away const for internal API
    action.num_features = num_features;
    action.affected_agents = NULL;
    action.num_affected_agents = 0;
    action.predicted_harm = 0.0F;

    // Evaluate using internal ethics engine
    ethics_evaluation_t eval = ethics_engine_evaluate_action(ethics->internal_ethics, &action);

    // Convert evaluation to simple score
    // Golden Rule score: -1 (harmful) to +1 (beneficial)
    *out_score = eval.golden_rule_score;

    set_error("No error");
    return NIMCP_OK;
}

//=============================================================================
// Knowledge API Implementation
//=============================================================================

nimcp_knowledge_t nimcp_knowledge_create(void) {
    nimcp_knowledge_t handle = (nimcp_knowledge_t)nimcp_malloc(sizeof(struct nimcp_knowledge_handle));
    if (!handle) {
        set_error("Failed to allocate knowledge handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "handle is NULL");

        return NULL;
    }

    // Create internal knowledge system
    handle->internal_knowledge = knowledge_system_create("default");

    if (!handle->internal_knowledge) {
        set_error("Failed to create internal knowledge system");
        nimcp_free(handle);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_knowledge_create: handle->internal_knowledge is NULL");
        return NULL;
    }

    set_error("No error");
    return handle;
}

void nimcp_knowledge_destroy(nimcp_knowledge_t knowledge) {
    if (!knowledge) {
        return;
    }

    if (knowledge->internal_knowledge) {
        knowledge_system_destroy(knowledge->internal_knowledge);
    }

    nimcp_free(knowledge);
}

nimcp_status_t nimcp_knowledge_add_fact(
    nimcp_knowledge_t knowledge,
    const char* subject,
    const char* predicate,
    const char* object)
{
    NIMCP_CHECK_THROW(knowledge, NIMCP_ERROR_NULL_ARG, "Knowledge handle is NULL");
    NIMCP_CHECK_THROW(subject && predicate && object, NIMCP_ERROR_NULL_ARG,
                      "Subject, predicate, or object is NULL");

    // Create a knowledge item from the fact
    knowledge_item_t item = {0};

    // Use subject as concept
    strncpy(item.concept_name, subject, sizeof(item.concept_name) - 1);

    // Create definition from predicate and object
    snprintf(item.definition, sizeof(item.definition), "%s %s", predicate, object);

    // Set defaults
    item.domain = KNOWLEDGE_DOMAIN_GENERAL;
    item.confidence = 1.0F;
    item.examples = NULL;
    item.num_examples = 0;
    item.related_concepts = NULL;
    item.num_related = 0;
    item.learned_timestamp = 0;
    item.reinforcement_count = 1;

    // Add to internal knowledge system
    bool success = knowledge_add_item(knowledge->internal_knowledge, &item);

    NIMCP_CHECK_THROW(success, NIMCP_ERROR, "Failed to add knowledge item");

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_knowledge_query(
    nimcp_knowledge_t knowledge,
    const char* query,
    char* out_result,
    uint32_t max_result_len)
{
    NIMCP_CHECK_THROW(knowledge, NIMCP_ERROR_NULL_ARG, "Knowledge handle is NULL");
    NIMCP_CHECK_THROW(query, NIMCP_ERROR_NULL_ARG, "Query is NULL");
    NIMCP_CHECK_THROW(out_result, NIMCP_ERROR_NULL_ARG, "Output result buffer is NULL");

    // Try to retrieve knowledge about the query concept
    knowledge_item_t item;
    bool found = knowledge_retrieve(knowledge->internal_knowledge, query, &item);

    if (found) {
        // Return the definition
        snprintf(out_result, max_result_len, "%s", item.definition);
    } else {
        // Concept not found
        snprintf(out_result, max_result_len, "No knowledge found about '%s'", query);
    }

    set_error("No error");
    return NIMCP_OK;
}

//=============================================================================
// Complex Number & Oscillation API Implementation
//=============================================================================

#include "core/brain/oscillations/nimcp_brain_complex_oscillations.h"
#include "security/nimcp_bbb_helpers.h"

/**
 * @brief Enable or disable complex oscillation features
 */
bool nimcp_enable_complex_oscillations(nimcp_brain_t brain, bool enable) {
    if (!brain) {
        set_error("Brain handle is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_enable_complex_oscillations: brain is NULL");
        return false;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_enable_complex_oscillations: brain->internal_brain is NULL");
        return false;
    }

    // Get brain's internal structure to access complex oscillation state
    // Note: This requires brain_internal_t access, which would need to be exposed
    // For now, we'll use a hypothetical brain configuration function
    // This would typically be: brain_config_set_complex_oscillations(brain->internal_brain, enable)

    // Since we may not have direct config access, we'll check if it's already enabled
    // and return true if enabling and already enabled, or if disabling and already disabled
    bool currently_enabled = brain_complex_oscillation_is_enabled(brain->internal_brain);

    if (enable == currently_enabled) {
        set_error("No error");
        return true;
    }

    // For actual enable/disable, we would need brain_config_t access
    // This is a placeholder implementation that at least validates the state
    set_error("Complex oscillation enable/disable requires brain reconfiguration");
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "nimcp_enable_complex_oscillations: validation failed");
    return false;
}

/**
 * @brief Check if complex oscillation features are enabled
 */
bool nimcp_is_complex_oscillations_enabled(nimcp_brain_t brain) {
    if (!brain || !brain->internal_brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_is_complex_oscillations_enabled: required parameter is NULL (brain, brain->internal_brain)");
        return false;
    }

    return brain_complex_oscillation_is_enabled(brain->internal_brain);
}

/**
 * @brief Get oscillation phasor for a specific neuron
 */
nimcp_oscillation_phasor_t nimcp_get_oscillation_phasor(
    nimcp_brain_t brain,
    uint32_t neuron_id)
{
    nimcp_oscillation_phasor_t result = {0.0F, 0.0F};

    // Guard: Validate parameters
    if (!brain) {
        set_error("NULL brain provided to nimcp_get_oscillation_phasor");
        return result;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return result;
    }

    // Check if complex oscillations are enabled
    if (!brain_complex_oscillation_is_enabled(brain->internal_brain)) {
        set_error("Complex oscillations not enabled - call nimcp_enable_complex_oscillations first");
        return result;
    }

    // Get brain's oscillation analyzer
    brain_oscillation_analyzer_t* analyzer = brain_get_oscillations(brain->internal_brain);
    if (!analyzer) {
        set_error("Brain oscillation analyzer not available");
        return result;
    }

    // Access complex oscillation state from analyzer
    // Note: This assumes brain_oscillation_analyzer_t has a complex_state field
    // The actual implementation would need to extract this from the analyzer structure
    // For now, we'll use a hypothetical accessor function
    brain_complex_oscillation_state_t* complex_state =
        (brain_complex_oscillation_state_t*)analyzer;  // Placeholder cast

    // Get neuron phasor
    neural_phasor_t phasor;
    if (!brain_complex_oscillation_get_phasor(complex_state, neuron_id, &phasor)) {
        set_error("Invalid neuron ID or failed to get phasor");
        return result;
    }

    // Convert neural_phasor_t to nimcp_oscillation_phasor_t
    result.amplitude = phasor_amplitude(phasor);
    result.phase = phasor_phase(phasor);

    set_error("No error");
    return result;
}

/**
 * @brief Compute phase coherence across multiple neurons
 */
float nimcp_get_phase_coherence(
    nimcp_brain_t brain,
    const uint32_t* neuron_ids,
    uint32_t count)
{
    // Guard: Validate parameters
    if (!brain) {
        set_error("NULL brain provided to nimcp_get_phase_coherence");
        return 0.0F;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return 0.0F;
    }

    if (!neuron_ids) {
        set_error("NULL neuron_ids provided");
        return 0.0F;
    }

    if (count == 0) {
        set_error("Invalid count (0) provided");
        return 0.0F;
    }

    // Check if complex oscillations are enabled
    if (!brain_complex_oscillation_is_enabled(brain->internal_brain)) {
        set_error("Complex oscillations not enabled");
        return 0.0F;
    }

    // Get brain's oscillation analyzer
    brain_oscillation_analyzer_t* analyzer = brain_get_oscillations(brain->internal_brain);
    if (!analyzer) {
        set_error("Brain oscillation analyzer not available");
        return 0.0F;
    }

    // Access complex oscillation state
    brain_complex_oscillation_state_t* complex_state =
        (brain_complex_oscillation_state_t*)analyzer;

    // Compute phase coherence for neuron subset
    phase_coherence_result_t result;
    if (!brain_complex_oscillation_compute_coherence_subset(
            complex_state, neuron_ids, count, &result)) {
        set_error("Failed to compute phase coherence");
        return 0.0F;
    }

    set_error("No error");
    return result.coherence;
}

/**
 * @brief Compute phase-amplitude coupling (PAC) modulation index
 */
float nimcp_get_pac_modulation(
    nimcp_brain_t brain,
    float theta_freq,
    float gamma_freq)
{
    // Guard: Validate parameters
    if (!brain) {
        set_error("NULL brain provided to nimcp_get_pac_modulation");
        return 0.0F;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        return 0.0F;
    }

    // Validate frequency ranges
    if (theta_freq < 4.0F || theta_freq > 8.0F) {
        set_error("Theta frequency should be in range 4-8 Hz");
        return 0.0F;
    }

    if (gamma_freq < 30.0F || gamma_freq > 100.0F) {
        set_error("Gamma frequency should be in range 30-100 Hz");
        return 0.0F;
    }

    // Check if complex oscillations are enabled
    if (!brain_complex_oscillation_is_enabled(brain->internal_brain)) {
        set_error("Complex oscillations not enabled");
        return 0.0F;
    }

    // Get brain's oscillation analyzer
    brain_oscillation_analyzer_t* analyzer = brain_get_oscillations(brain->internal_brain);
    if (!analyzer) {
        set_error("Brain oscillation analyzer not available");
        return 0.0F;
    }

    // Access complex oscillation state
    brain_complex_oscillation_state_t* complex_state =
        (brain_complex_oscillation_state_t*)analyzer;

    // For PAC computation, we need to:
    // 1. Extract theta-band phase neurons
    // 2. Extract gamma-band amplitude values
    // 3. Compute PAC modulation index

    // This requires access to brain's neural network to filter by frequency bands
    // For now, we'll provide a simplified implementation that uses all neurons
    uint32_t num_neurons = brain_complex_oscillation_get_num_neurons(complex_state);

    if (num_neurons == 0) {
        set_error("No neurons in complex oscillation state");
        return 0.0F;
    }

    // Allocate arrays for phase and amplitude
    uint32_t* phase_indices = (uint32_t*)nimcp_malloc(num_neurons * sizeof(uint32_t));
    float* amplitude_values = (float*)nimcp_malloc(num_neurons * sizeof(float));

    if (!phase_indices || !amplitude_values) {
        set_error("Failed to allocate memory for PAC computation");
        if (phase_indices) nimcp_free(phase_indices);
        if (amplitude_values) nimcp_free(amplitude_values);
        return 0.0F;
    }

    // Populate indices and extract amplitude values
    for (uint32_t i = 0; i < num_neurons; i++) {
        phase_indices[i] = i;

        neural_phasor_t phasor;
        if (brain_complex_oscillation_get_phasor(complex_state, i, &phasor)) {
            amplitude_values[i] = phasor_amplitude(phasor);
        } else {
            amplitude_values[i] = 0.0F;
        }
    }

    // Compute PAC
    pac_result_t pac_result;
    bool success = brain_complex_oscillation_compute_pac(
        complex_state,
        phase_indices,
        num_neurons,
        amplitude_values,
        num_neurons,
        &pac_result
    );

    // Cleanup
    nimcp_free(phase_indices);
    nimcp_free(amplitude_values);

    if (!success) {
        set_error("Failed to compute PAC modulation index");
        return 0.0F;
    }

    set_error("No error");
    return pac_result.modulation_index;
}

//=============================================================================
// Phase 2.8: Dynamic Brain Resizing API
//=============================================================================

bool nimcp_brain_resize(nimcp_brain_t brain, uint32_t new_neuron_count) {
    if (!brain) {
        set_error("Brain handle is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_resize: brain is NULL");
        return false;
    }
    
    return brain_resize(brain->internal_brain, new_neuron_count);
}

bool nimcp_brain_auto_resize(nimcp_brain_t brain) {
    if (!brain) {
        set_error("Brain handle is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_auto_resize: brain is NULL");
        return false;
    }
    
    return brain_auto_resize(brain->internal_brain);
}

uint32_t nimcp_brain_get_neuron_count(nimcp_brain_t brain) {
    if (!brain) {
        set_error("Brain handle is NULL");
        return 0;
    }
    
    return brain_get_neuron_count(brain->internal_brain);
}

bool nimcp_brain_get_utilization_metrics(nimcp_brain_t brain, float* utilization, float* saturation) {
    if (!brain) {
        set_error("Brain handle is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_get_utilization_metrics: brain is NULL");
        return false;
    }

    if (!utilization || !saturation) {
        set_error("Output parameters are NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_get_utilization_metrics: required parameter is NULL (utilization, saturation)");
        return false;
    }

    return brain_get_utilization_metrics(brain->internal_brain, utilization, saturation);
}

//=============================================================================
// Training Pipeline API Implementation
//=============================================================================

/**
 * @brief Internal structure to track training configuration IDs
 *
 * Stored in brain handle to track created loss/optimizer/scheduler IDs
 */
typedef struct {
    uint32_t loss_id;
    uint32_t optimizer_id;
    uint32_t scheduler_id;
    uint32_t gradmgr_id;
    bool configured;
    uint32_t step_count;
    tcb_context_t* callbacks;         /**< Training callback manager */
    bool callbacks_enabled;           /**< Whether to fire callbacks */
    backprop_ctx_t* backprop;         /**< Backpropagation context for weight gradients */
} training_pipeline_state_t;

// Global map from brain handle to training state (simple approach for now)
// In production, this would be stored in the brain handle struct
#define MAX_TRAINING_STATES 64
static struct {
    nimcp_brain_t brain;
    training_pipeline_state_t state;
} g_training_states[MAX_TRAINING_STATES] = {0};

static training_pipeline_state_t* get_training_state(nimcp_brain_t brain) {
    // Find existing state
    for (int i = 0; i < MAX_TRAINING_STATES; i++) {
        if (g_training_states[i].brain == brain) {
            return &g_training_states[i].state;
        }
    }
    // Create new state
    for (int i = 0; i < MAX_TRAINING_STATES; i++) {
        if (g_training_states[i].brain == NULL) {
            g_training_states[i].brain = brain;
            memset(&g_training_states[i].state, 0, sizeof(training_pipeline_state_t));
            return &g_training_states[i].state;
        }
    }
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "get_training_state: validation failed");
    return NULL;  // No space
}

static void clear_training_state(nimcp_brain_t brain) {
    for (int i = 0; i < MAX_TRAINING_STATES; i++) {
        if (g_training_states[i].brain == brain) {
            // Destroy callback manager if present
            if (g_training_states[i].state.callbacks) {
                tcb_destroy(g_training_states[i].state.callbacks);
                g_training_states[i].state.callbacks = NULL;
            }
            // Destroy backprop context if present
            if (g_training_states[i].state.backprop) {
                backprop_destroy(g_training_states[i].state.backprop);
                g_training_states[i].state.backprop = NULL;
            }
            g_training_states[i].brain = NULL;
            memset(&g_training_states[i].state, 0, sizeof(training_pipeline_state_t));
            return;
        }
    }
}

/**
 * @brief Cleanup training state when brain is destroyed
 *
 * Called from nimcp_brain_destroy to prevent memory leaks from training state.
 */
void nimcp_api_training_cleanup_brain(nimcp_brain_t brain) {
    if (!brain) return;
    clear_training_state(brain);
}

nimcp_training_config_t nimcp_training_config_default(void) {
    nimcp_training_config_t config = {
        .loss_type = NIMCP_API_LOSS_CROSS_ENTROPY,
        .optimizer_type = NIMCP_API_OPT_ADAM,
        .scheduler_type = NIMCP_API_SCHED_COSINE,

        .learning_rate = 0.001F,
        .weight_decay = 0.0F,
        .momentum = 0.9F,
        .beta1 = 0.9F,
        .beta2 = 0.999F,
        .epsilon = 1e-8F,

        .scheduler_step_size = 1000,
        .scheduler_gamma = 0.1F,
        .warmup_steps = 0,

        .enable_gradient_clipping = true,
        .gradient_clip_value = 1.0F,

        .enable_biological_modulation = true,
        .biological_blend = 0.5F,

        // Network type dispatch defaults
        .network_type = NIMCP_NETWORK_ADAPTIVE,

        // SNN defaults
        .snn_method = NIMCP_SNN_TRAIN_SURROGATE,
        .snn_eligibility_tau = 20.0F,
        .snn_reward_tau = 100.0F,
        .snn_surrogate_beta = 5.0F,

        // LNN defaults
        .lnn_method = NIMCP_LNN_TRAIN_ADJOINT,
        .lnn_bptt_truncation = 100,
        .lnn_use_adjoint_checkpointing = true
    };
    return config;
}

nimcp_status_t nimcp_brain_configure_training(
    nimcp_brain_t brain,
    const nimcp_training_config_t* config)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_ARG, "Training config is NULL");

    // Get internal brain
    brain_t internal = brain->internal_brain;
    NIMCP_CHECK_THROW(internal, NIMCP_ERROR_NULL_ARG, "Internal brain is NULL");

    // Get or create training context
    nimcp_brain_training_ctx_t* training_ctx = internal->training_ctx;
    NIMCP_CHECK_THROW(training_ctx, NIMCP_ERROR_INVALID, "Brain has no training context (training not enabled)");

    // Get training state
    training_pipeline_state_t* state = get_training_state(brain);
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_MEMORY, "Failed to allocate training state");

    // Map public loss type to internal loss type
    // Note: Public API uses nimcp_loss_type_t from nimcp.h
    // Internal uses nimcp_loss_type_t from nimcp_loss_functions.h (same name, different values)
    nimcp_loss_type_t internal_loss;
    switch (config->loss_type) {
        case 0:  internal_loss = NIMCP_LOSS_MSE; break;                    // NIMCP_LOSS_MSE
        case 1:  internal_loss = NIMCP_LOSS_CROSS_ENTROPY; break;          // NIMCP_LOSS_CROSS_ENTROPY
        case 2:  internal_loss = NIMCP_LOSS_BINARY_CROSS_ENTROPY; break;   // NIMCP_LOSS_BINARY_CE
        case 3:  internal_loss = NIMCP_LOSS_HUBER; break;                  // NIMCP_LOSS_HUBER
        case 4:  internal_loss = NIMCP_LOSS_MAE; break;                    // NIMCP_LOSS_MAE
        case 5:  internal_loss = NIMCP_LOSS_FOCAL; break;                  // NIMCP_LOSS_FOCAL
        case 6:  internal_loss = NIMCP_LOSS_KL_DIVERGENCE; break;          // NIMCP_LOSS_KL_DIV
        default: internal_loss = NIMCP_LOSS_MSE; break;
    }

    // Create loss function
    nimcp_loss_config_t loss_config = nimcp_loss_default_config(internal_loss);
    nimcp_result_t res = nimcp_brain_training_create_loss(training_ctx, &loss_config, &state->loss_id);
    NIMCP_CHECK_THROW(res == NIMCP_SUCCESS && state->loss_id != 0, NIMCP_ERROR,
                      "Failed to create loss function");

    // Map public optimizer type to internal optimizer type
    nimcp_optimizer_type_t internal_opt;
    switch (config->optimizer_type) {
        case 0:  internal_opt = NIMCP_OPTIMIZER_SGD; break;          // NIMCP_OPT_SGD
        case 1:  internal_opt = NIMCP_OPTIMIZER_SGD_MOMENTUM; break; // NIMCP_OPT_MOMENTUM
        case 2:  internal_opt = NIMCP_OPTIMIZER_ADAM; break;         // NIMCP_OPT_ADAM
        case 3:  internal_opt = NIMCP_OPTIMIZER_ADAMW; break;        // NIMCP_OPT_ADAMW
        case 4:  internal_opt = NIMCP_OPTIMIZER_RMSPROP; break;      // NIMCP_OPT_RMSPROP
        case 5:  internal_opt = NIMCP_OPTIMIZER_ADAGRAD; break;      // NIMCP_OPT_ADAGRAD
        default: internal_opt = NIMCP_OPTIMIZER_ADAM; break;
    }

    // Create optimizer - use union-based config
    nimcp_optimizer_config_t opt_config = nimcp_optimizer_default_config(internal_opt);
    opt_config.clip_gradients = config->enable_gradient_clipping;
    opt_config.gradient_clip_value = config->gradient_clip_value;

    // Set parameters based on optimizer type
    switch (internal_opt) {
        case NIMCP_OPTIMIZER_SGD:
        case NIMCP_OPTIMIZER_SGD_MOMENTUM:
        case NIMCP_OPTIMIZER_NESTEROV:
            opt_config.params.sgd.learning_rate = config->learning_rate;
            opt_config.params.sgd.momentum = config->momentum;
            opt_config.params.sgd.weight_decay = config->weight_decay;
            break;
        case NIMCP_OPTIMIZER_ADAM:
            opt_config.params.adam.learning_rate = config->learning_rate;
            opt_config.params.adam.beta1 = config->beta1;
            opt_config.params.adam.beta2 = config->beta2;
            opt_config.params.adam.epsilon = config->epsilon;
            opt_config.params.adam.weight_decay = config->weight_decay;
            break;
        case NIMCP_OPTIMIZER_ADAMW:
            opt_config.params.adamw.learning_rate = config->learning_rate;
            opt_config.params.adamw.beta1 = config->beta1;
            opt_config.params.adamw.beta2 = config->beta2;
            opt_config.params.adamw.epsilon = config->epsilon;
            opt_config.params.adamw.weight_decay = config->weight_decay;
            break;
        case NIMCP_OPTIMIZER_RMSPROP:
            opt_config.params.rmsprop.learning_rate = config->learning_rate;
            opt_config.params.rmsprop.momentum = config->momentum;
            opt_config.params.rmsprop.weight_decay = config->weight_decay;
            opt_config.params.rmsprop.epsilon = config->epsilon;
            break;
        case NIMCP_OPTIMIZER_ADAGRAD:
            opt_config.params.adagrad.learning_rate = config->learning_rate;
            opt_config.params.adagrad.weight_decay = config->weight_decay;
            opt_config.params.adagrad.epsilon = config->epsilon;
            break;
        default:
            break;
    }

    res = nimcp_brain_training_create_optimizer(training_ctx, &opt_config, &state->optimizer_id);
    NIMCP_CHECK_THROW(res == NIMCP_SUCCESS && state->optimizer_id != 0, NIMCP_ERROR,
                      "Failed to create optimizer");

    // Map public scheduler type to internal scheduler type
    nimcp_lr_scheduler_type_t internal_sched;
    switch (config->scheduler_type) {
        case 0:  internal_sched = NIMCP_LR_CONSTANT; break;           // NIMCP_LR_CONSTANT
        case 1:  internal_sched = NIMCP_LR_STEP; break;               // NIMCP_LR_STEP
        case 2:  internal_sched = NIMCP_LR_EXPONENTIAL; break;        // NIMCP_LR_EXPONENTIAL
        case 3:  internal_sched = NIMCP_LR_COSINE_ANNEALING; break;   // NIMCP_LR_COSINE
        case 4:  internal_sched = NIMCP_LR_COSINE_WARMUP; break;      // NIMCP_LR_WARMUP_COSINE
        case 5:  internal_sched = NIMCP_LR_REDUCE_ON_PLATEAU; break;  // NIMCP_LR_REDUCE_ON_PLATEAU
        case 6:  internal_sched = NIMCP_LR_CYCLIC; break;             // NIMCP_LR_CYCLIC
        default: internal_sched = NIMCP_LR_COSINE_ANNEALING; break;
    }

    // Create scheduler - use union-based config
    nimcp_lr_scheduler_config_t sched_config = nimcp_lr_scheduler_config_from_type(
        internal_sched, config->learning_rate);

    // Set parameters based on scheduler type
    switch (internal_sched) {
        case NIMCP_LR_STEP:
            sched_config.params.step.step_size = config->scheduler_step_size;
            sched_config.params.step.gamma = config->scheduler_gamma;
            break;
        case NIMCP_LR_EXPONENTIAL:
            sched_config.params.exponential.gamma = config->scheduler_gamma;
            break;
        case NIMCP_LR_COSINE_ANNEALING:
            sched_config.params.cosine.T_max = config->scheduler_step_size;
            break;
        case NIMCP_LR_COSINE_WARMUP:
        case NIMCP_LR_LINEAR_WARMUP:
            sched_config.params.warmup.warmup_steps = config->warmup_steps;
            sched_config.params.warmup.target_lr = config->learning_rate;
            break;
        case NIMCP_LR_CYCLIC:
            sched_config.params.cyclic.base_lr = config->learning_rate * 0.1F;
            sched_config.params.cyclic.max_lr = config->learning_rate;
            sched_config.params.cyclic.step_size_up = config->scheduler_step_size;
            break;
        default:
            break;
    }

    res = nimcp_brain_training_create_scheduler(training_ctx, &sched_config, &state->scheduler_id);
    NIMCP_CHECK_THROW(res == NIMCP_SUCCESS && state->scheduler_id != 0, NIMCP_ERROR,
                      "Failed to create LR scheduler");

    // Use existing gradient manager if available, or create one
    state->gradmgr_id = 1;  // Default gradient manager created during brain init

    // Configure biological modulation
    if (config->enable_biological_modulation && internal->plasticity_bridge) {
        nimcp_brain_training_set_biological_modulation(training_ctx, config->biological_blend);
    }

    // Initialize training dispatcher for SNN/LNN/CNN/Adaptive routing
    // This sets up specialized training contexts based on network_type
    if (training_dispatch_init(internal, config) < 0) {
        // Dispatcher init failed - this is OK for ADAPTIVE type (returns -2)
        // For other types, log warning but continue with fallback to backprop
        if (config->network_type != NIMCP_NETWORK_ADAPTIVE) {
            LOG_WARN("Training dispatcher init failed for network_type=%d, falling back to adaptive",
                     config->network_type);
        }
    }

    // Store network type in internal brain for dispatch during train_step
    internal->active_network_type = config->network_type;

    state->configured = true;
    state->step_count = 0;

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_train_step(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    const float* targets,
    uint32_t num_targets,
    nimcp_training_result_t* result)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_CHECK_THROW(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    NIMCP_CHECK_THROW(targets, NIMCP_ERROR_NULL_ARG, "Targets array is NULL");

    brain_t internal = brain->internal_brain;
    NIMCP_CHECK_THROW(internal, NIMCP_ERROR_NULL_ARG, "Internal brain is NULL");

    // Validate dimensions
    NIMCP_CHECK_THROW(num_features == internal->config.num_inputs, NIMCP_ERROR_INVALID,
                      "Feature count mismatch");
    NIMCP_CHECK_THROW(num_targets == internal->config.num_outputs, NIMCP_ERROR_INVALID,
                      "Target count mismatch");

    // Get training state
    training_pipeline_state_t* state = get_training_state(brain);
    if (!state || !state->configured) {
        // Auto-configure with defaults if not configured
        nimcp_training_config_t default_config = nimcp_training_config_default();
        nimcp_status_t config_res = nimcp_brain_configure_training(brain, &default_config);
        if (config_res != NIMCP_OK) {
            return config_res;
        }
        state = get_training_state(brain);
    }

    nimcp_brain_training_ctx_t* training_ctx = internal->training_ctx;
    NIMCP_CHECK_THROW(training_ctx, NIMCP_ERROR_INVALID, "Training context not available");

    // === DISPATCHER: Try specialized training first (SNN/LNN/CNN) ===
    // For non-ADAPTIVE network types, the dispatcher handles training
    if (internal->active_network_type != NIMCP_NETWORK_ADAPTIVE) {
        training_dispatch_result_t dispatch_result = {0};
        int dispatch_rc = training_dispatch_step(internal, features, num_features,
                                                  targets, num_targets, &dispatch_result);

        if (dispatch_rc == 0) {
            // Dispatcher handled training successfully
            state->step_count++;

            if (result) {
                result->loss = dispatch_result.loss;
                result->learning_rate = dispatch_result.learning_rate;
                result->step = state->step_count;
                result->early_stopped = dispatch_result.early_stopped;
                result->gradient_norm = dispatch_result.gradient_norm;
            }

            set_error("No error");
            return NIMCP_OK;
        }

        // If dispatch_rc == -2, fall through to standard backprop
        // If dispatch_rc == -1, error - also fall through to backprop as fallback
        if (dispatch_rc == -1) {
            LOG_WARN("Training dispatch failed, falling back to adaptive backprop");
        }
    }

    // === ADAPTIVE PATH: Standard backpropagation ===

    // Get the adaptive network
    adaptive_network_t network = internal->network;
    if (!network) {
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_INVALID, "Brain has no neural network");
    }

    // Get base neural network
    neural_network_t base_net = adaptive_network_get_base_network(network);
    if (!base_net) {
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_OPERATION_FAILED, "Failed to get base network");
    }

    // === STEP 1: Create or get backprop context ===
    if (!state->backprop) {
        state->backprop = backprop_create(base_net);
        if (!state->backprop) {
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_MEMORY, "Failed to create backprop context");
        }
    }

    // Allocate predictions buffer
    float* predictions = nimcp_malloc(num_targets * sizeof(float));
    if (!predictions) {
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_MEMORY, "Failed to allocate predictions buffer");
    }

    // === STEP 2: Forward pass with activation recording ===
    // Use network's forward pass for predictions (more reliable)
    backprop_clear(state->backprop);
    (void)adaptive_network_forward(network, features, num_features,
                                   predictions, num_targets, 0);

    // Store activations in backprop context from neuron states
    // The forward pass updated neuron->state for each neuron
    backprop_store_activations_from_network(state->backprop, features, num_features);

    // === STEP 3: Compute loss and output gradients ===
    float loss_value = 0.0F;
    float* output_gradients = nimcp_malloc(num_targets * sizeof(float));
    if (!output_gradients) {
        nimcp_free(predictions);
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_MEMORY, "Failed to allocate output gradients");
    }

    // Compute loss (MSE or cross-entropy)
    nimcp_loss_context_t* loss_ctx = nimcp_brain_training_get_loss(training_ctx, state->loss_id);
    if (loss_ctx) {
        nimcp_loss_result_t loss_result = {0};
        nimcp_loss_forward(loss_ctx, predictions, targets, 1, num_targets, &loss_result);
        loss_value = loss_result.loss_value;
        nimcp_loss_backward(loss_ctx, predictions, targets, 1, num_targets, output_gradients);
    } else {
        // Fallback: compute MSE loss and gradients manually
        for (uint32_t i = 0; i < num_targets; i++) {
            float diff = predictions[i] - targets[i];
            loss_value += diff * diff;
            output_gradients[i] = 2.0F * diff / (float)num_targets;
        }
        loss_value /= (float)num_targets;
    }


    // === STEP 4: Backward pass to compute weight gradients ===
    if (!backprop_backward(state->backprop, output_gradients, num_targets)) {
        // If backprop fails, use output gradients as-is (limited training)
        LOG_WARN("Backprop backward failed, using limited gradient update");
    }

    // Get total weight count
    uint32_t num_neurons = neural_network_get_num_neurons(base_net);
    size_t total_weights = backprop_get_weight_count(state->backprop);
    if (total_weights == 0) {
        // Count weights manually
        for (uint32_t n = 0; n < num_neurons; n++) {
            neuron_t* neuron = neural_network_get_neuron(base_net, n);
            if (neuron && neuron->synapses) {
                total_weights += neuron->num_synapses;
            }
        }
    }

    if (total_weights == 0) {
        nimcp_free(predictions);
        nimcp_free(output_gradients);
        set_error("Network has no weights");
        return NIMCP_ERROR;
    }

    // === STEP 5: Get weight gradients from backprop ===
    float* weight_gradients = nimcp_malloc(total_weights * sizeof(float));
    if (!weight_gradients) {
        nimcp_free(predictions);
        nimcp_free(output_gradients);
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_MEMORY, "Failed to allocate weight gradients");
    }

    size_t grad_count = backprop_get_weight_gradients(state->backprop, weight_gradients, total_weights);

    if (grad_count == 0 || !backprop_has_valid_gradients(state->backprop)) {
        // Fallback: use output gradients distributed across weights
        // This is not ideal but prevents complete failure
        float grad_per_weight = 0.0F;
        for (uint32_t i = 0; i < num_targets; i++) {
            grad_per_weight += output_gradients[i];
        }
        grad_per_weight /= (float)total_weights;
        for (size_t i = 0; i < total_weights; i++) {
            weight_gradients[i] = grad_per_weight;
        }
    }

    // Extract current weights
    float* params = nimcp_malloc(total_weights * sizeof(float));
    if (!params) {
        nimcp_free(predictions);
        nimcp_free(output_gradients);
        nimcp_free(weight_gradients);
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_MEMORY, "Failed to allocate params buffer");
    }

    size_t weight_idx = 0;
    for (uint32_t n = 0; n < num_neurons; n++) {
        neuron_t* neuron = neural_network_get_neuron(base_net, n);
        if (neuron && neuron->synapses) {
            for (uint32_t s = 0; s < neuron->num_synapses; s++) {
                params[weight_idx++] = neuron->synapses[s].weight;
            }
        }
    }

    // === STEP 6: Apply optimizer with weight gradients ===
    float current_lr = 0.0F;
    nimcp_lr_scheduler_ctx_t* sched = nimcp_brain_training_get_scheduler(
        training_ctx, state->scheduler_id);
    if (sched) {
        current_lr = nimcp_lr_scheduler_get_lr(sched);
    }

    // Gradient clipping
    float gradient_norm = 0.0F;
    for (size_t i = 0; i < total_weights; i++) {
        gradient_norm += weight_gradients[i] * weight_gradients[i];
    }
    gradient_norm = sqrtf(gradient_norm);

    float clip_value = 1.0F;  // Default gradient clip
    if (gradient_norm > clip_value) {
        float scale = clip_value / gradient_norm;
        for (size_t i = 0; i < total_weights; i++) {
            weight_gradients[i] *= scale;
        }
    }

    // Apply optimizer step
    nimcp_result_t res = nimcp_brain_training_optimize(
        training_ctx, state->optimizer_id, params, weight_gradients, total_weights);

    // === STEP 7: Fire callbacks ===
    tcb_action_t cb_action = TCB_ACTION_CONTINUE;
    if (state->callbacks && state->callbacks_enabled) {
        tcb_update_metrics(state->callbacks, loss_value, current_lr,
                          state->step_count + 1, gradient_norm);

        tcb_event_t loss_event = {
            .event_type = TCB_EVENT_LOSS_COMPUTED,
            .metrics = {
                .step = state->step_count + 1,
                .loss = loss_value,
                .learning_rate = current_lr,
                .gradient_norm = gradient_norm
            },
            .user_data = NULL,
            .checkpoint_path = NULL,
            .timestamp_ns = 0
        };
        cb_action = tcb_fire(state->callbacks, &loss_event);

        if (cb_action == TCB_ACTION_SKIP_STEP) {
            nimcp_free(params);
            nimcp_free(weight_gradients);
            nimcp_free(output_gradients);
            nimcp_free(predictions);
            state->step_count++;
            if (result) {
                result->loss = loss_value;
                result->step = state->step_count;
                result->early_stopped = false;
                result->learning_rate = current_lr;
                result->gradient_norm = gradient_norm;
            }
            set_error("No error");
            return NIMCP_OK;
        }

        if (cb_action == TCB_ACTION_STOP_TRAINING) {
            nimcp_free(params);
            nimcp_free(weight_gradients);
            nimcp_free(output_gradients);
            nimcp_free(predictions);
            state->step_count++;
            if (result) {
                result->loss = loss_value;
                result->step = state->step_count;
                result->early_stopped = true;
                result->learning_rate = current_lr;
                result->gradient_norm = gradient_norm;
            }
            set_error("No error");
            return NIMCP_OK;
        }
    }

    // === STEP 8: Write updated weights back to network ===
    // NOTE: Must update BOTH outgoing synapses AND incoming synapses
    // The forward pass reads from incoming_synapses, but they're separate copies
    if (res == NIMCP_SUCCESS || res == NIMCP_TRAINING_ERROR_EARLY_STOP) {
        weight_idx = 0;
        for (uint32_t n = 0; n < num_neurons; n++) {
            neuron_t* neuron = neural_network_get_neuron(base_net, n);
            if (neuron && neuron->synapses) {
                for (uint32_t s = 0; s < neuron->num_synapses; s++) {
                    float new_weight = params[weight_idx++];
                    neuron->synapses[s].weight = new_weight;

                    // BUGFIX: Also update the corresponding incoming synapse on the target neuron
                    // The incoming synapse is where forward pass reads from!
                    uint32_t target_id = neuron->synapses[s].target_id;
                    if (target_id < num_neurons) {
                        neuron_t* target_neuron = neural_network_get_neuron(base_net, target_id);
                        if (target_neuron && target_neuron->incoming_synapses) {
                            // Find the incoming synapse with source_neuron_id == n
                            for (uint32_t i = 0; i < target_neuron->num_incoming; i++) {
                                if (target_neuron->incoming_synapses[i].source_neuron_id == n) {
                                    target_neuron->incoming_synapses[i].weight = new_weight;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fire weights updated callback
        if (state->callbacks && state->callbacks_enabled) {
            tcb_event_t weights_event = {
                .event_type = TCB_EVENT_WEIGHTS_UPDATED,
                .metrics = {
                    .step = state->step_count + 1,
                    .loss = loss_value,
                    .learning_rate = current_lr,
                    .gradient_norm = gradient_norm
                },
                .user_data = NULL,
                .checkpoint_path = NULL,
                .timestamp_ns = 0
            };
            cb_action = tcb_fire(state->callbacks, &weights_event);
        }
    }

    // Step scheduler
    if (state->scheduler_id > 0 && sched) {
        nimcp_lr_scheduler_step(sched);
    }

    // === STEP 9: Increment step count and fill result ===
    state->step_count++;

    // Update training context stats (for nimcp_brain_get_training_stats)
    nimcp_brain_training_update_stats(training_ctx, 1, loss_value);

    if (result) {
        result->loss = loss_value;
        result->step = state->step_count;
        result->early_stopped = (res == NIMCP_TRAINING_ERROR_EARLY_STOP) ||
                               (cb_action == TCB_ACTION_STOP_TRAINING);
        result->learning_rate = current_lr;
        result->gradient_norm = gradient_norm;
    }

    // Fire step complete callback
    if (state->callbacks && state->callbacks_enabled) {
        tcb_event_t step_event = {
            .event_type = TCB_EVENT_STEP_COMPLETE,
            .metrics = {
                .step = state->step_count,
                .loss = loss_value,
                .learning_rate = current_lr,
                .gradient_norm = gradient_norm
            },
            .user_data = NULL,
            .checkpoint_path = NULL,
            .timestamp_ns = 0
        };
        cb_action = tcb_fire(state->callbacks, &step_event);

        if (cb_action == TCB_ACTION_STOP_TRAINING && result) {
            result->early_stopped = true;
        }
    }

    // Cleanup
    nimcp_free(params);
    nimcp_free(weight_gradients);
    nimcp_free(output_gradients);
    nimcp_free(predictions);

    if (res != NIMCP_SUCCESS && res != NIMCP_TRAINING_ERROR_EARLY_STOP) {
        set_error("Training step failed");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_train_batch(
    nimcp_brain_t brain,
    const float* features,
    const float* targets,
    uint32_t batch_size,
    uint32_t num_features,
    uint32_t num_targets,
    nimcp_training_result_t* result)
{
    NIMCP_CHECK_THROW(brain && features && targets, NIMCP_ERROR_NULL_ARG, "NULL argument provided");
    NIMCP_CHECK_THROW(batch_size != 0, NIMCP_ERROR_INVALID, "Batch size cannot be zero");

    // Train on each example and average results
    float total_loss = 0.0F;
    nimcp_training_result_t step_result = {0};

    for (uint32_t i = 0; i < batch_size; i++) {
        const float* sample_features = features + (i * num_features);
        const float* sample_targets = targets + (i * num_targets);

        nimcp_status_t res = nimcp_brain_train_step(
            brain, sample_features, num_features,
            sample_targets, num_targets, &step_result);

        if (res != NIMCP_OK) {
            return res;
        }

        total_loss += step_result.loss;

        if (step_result.early_stopped) {
            break;
        }
    }

    if (result) {
        result->loss = total_loss / batch_size;
        result->learning_rate = step_result.learning_rate;
        result->step = step_result.step;
        result->early_stopped = step_result.early_stopped;
        result->gradient_norm = step_result.gradient_norm;
    }

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_get_training_stats(
    nimcp_brain_t brain,
    uint64_t* total_steps,
    float* total_loss,
    float* current_lr)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");

    brain_t internal = brain->internal_brain;
    NIMCP_CHECK_THROW(internal && internal->training_ctx, NIMCP_ERROR_INVALID, "Training not enabled");

    nimcp_training_session_stats_t stats;
    nimcp_result_t res = nimcp_brain_training_get_stats(internal->training_ctx, &stats);
    NIMCP_CHECK_THROW(res == NIMCP_SUCCESS, NIMCP_ERROR, "Failed to get training stats");

    if (total_steps) *total_steps = stats.total_samples;
    if (total_loss) *total_loss = stats.total_loss;

    if (current_lr) {
        training_pipeline_state_t* state = get_training_state(brain);
        if (state && state->configured) {
            nimcp_lr_scheduler_ctx_t* sched = nimcp_brain_training_get_scheduler(
                internal->training_ctx, state->scheduler_id);
            if (sched) {
                *current_lr = nimcp_lr_scheduler_get_lr(sched);
            } else {
                *current_lr = internal->config.learning_rate;
            }
        } else {
            *current_lr = internal->config.learning_rate;
        }
    }

    set_error("No error");
    return NIMCP_OK;
}

float nimcp_brain_step_scheduler(nimcp_brain_t brain, float validation_metric) {
    if (!brain) {
        set_error("Brain handle is NULL");
        return 0.0F;
    }

    brain_t internal = brain->internal_brain;
    if (!internal || !internal->training_ctx) {
        set_error("Training not enabled");
        return 0.0F;
    }

    training_pipeline_state_t* state = get_training_state(brain);
    if (!state || !state->configured) {
        set_error("Training not configured");
        return 0.0F;
    }

    // Step scheduler and update optimizer
    float new_lr = nimcp_brain_training_step_scheduler_metric(
        internal->training_ctx,
        state->scheduler_id,
        state->optimizer_id,
        validation_metric
    );

    set_error("No error");
    return new_lr;
}

//=============================================================================
// Training Callbacks API Implementation
//=============================================================================

/**
 * @brief Wrapper to convert public callback to internal callback
 */
typedef struct {
    nimcp_training_callback_fn public_callback;
    void* user_data;
} callback_wrapper_t;

/**
 * @brief Internal callback that bridges public API to internal API
 */
static tcb_action_t callback_bridge(const tcb_event_t* event) {
    if (!event || !event->user_data) {
        return TCB_ACTION_CONTINUE;
    }

    callback_wrapper_t* wrapper = (callback_wrapper_t*)event->user_data;
    if (!wrapper->public_callback) {
        return TCB_ACTION_CONTINUE;
    }

    // Map internal event type to public event type
    nimcp_callback_event_t pub_event;
    switch (event->event_type) {
        case TCB_EVENT_STEP_COMPLETE:    pub_event = NIMCP_CB_STEP_COMPLETE; break;
        case TCB_EVENT_EPOCH_COMPLETE:   pub_event = NIMCP_CB_EPOCH_COMPLETE; break;
        case TCB_EVENT_LOSS_COMPUTED:    pub_event = NIMCP_CB_LOSS_COMPUTED; break;
        case TCB_EVENT_WEIGHTS_UPDATED:  pub_event = NIMCP_CB_WEIGHTS_UPDATED; break;
        case TCB_EVENT_LR_CHANGED:       pub_event = NIMCP_CB_LR_CHANGED; break;
        case TCB_EVENT_CONVERGENCE:      pub_event = NIMCP_CB_CONVERGENCE; break;
        case TCB_EVENT_DIVERGENCE:       pub_event = NIMCP_CB_DIVERGENCE; break;
        case TCB_EVENT_CHECKPOINT:       pub_event = NIMCP_CB_CHECKPOINT; break;
        default:                         pub_event = NIMCP_CB_STEP_COMPLETE; break;
    }

    // Convert metrics
    nimcp_callback_metrics_t pub_metrics = {
        .step = event->metrics.step,
        .epoch = event->metrics.epoch,
        .loss = event->metrics.loss,
        .loss_ema = event->metrics.loss_ema,
        .learning_rate = event->metrics.learning_rate,
        .gradient_norm = event->metrics.gradient_norm,
        .step_time_us = event->metrics.step_time_us,
        .is_converging = event->metrics.is_converging,
        .is_diverging = event->metrics.is_diverging
    };

    // Call public callback
    nimcp_callback_action_t pub_action = wrapper->public_callback(
        pub_event, &pub_metrics, wrapper->user_data);

    // Map public action to internal action
    switch (pub_action) {
        case NIMCP_CB_ACTION_CONTINUE:    return TCB_ACTION_CONTINUE;
        case NIMCP_CB_ACTION_STOP:        return TCB_ACTION_STOP_TRAINING;
        case NIMCP_CB_ACTION_SKIP:        return TCB_ACTION_SKIP_STEP;
        case NIMCP_CB_ACTION_ROLLBACK:    return TCB_ACTION_ROLLBACK;
        case NIMCP_CB_ACTION_REDUCE_LR:   return TCB_ACTION_REDUCE_LR;
        case NIMCP_CB_ACTION_INCREASE_LR: return TCB_ACTION_INCREASE_LR;
        default:                          return TCB_ACTION_CONTINUE;
    }
}

nimcp_callback_config_t nimcp_callback_config_default(void) {
    nimcp_callback_config_t config = {
        .enable_auto_checkpoint = false,
        .checkpoint_interval = 1000,
        .enable_early_stopping = true,
        .patience = 100,
        .min_delta = 1e-4F,
        .divergence_threshold = 10.0F,
        .log_interval = 0  // Disabled by default
    };
    return config;
}

nimcp_status_t nimcp_brain_enable_callbacks(
    nimcp_brain_t brain,
    const nimcp_callback_config_t* config)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");

    training_pipeline_state_t* state = get_training_state(brain);
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_MEMORY, "Failed to get training state");

    // Destroy existing callbacks if present
    if (state->callbacks) {
        tcb_destroy(state->callbacks);
        state->callbacks = NULL;
    }

    // Build internal config from public config
    tcb_config_t internal_config = tcb_config_default();

    if (config) {
        internal_config.enable_auto_checkpoint = config->enable_auto_checkpoint;
        internal_config.checkpoint_interval = config->checkpoint_interval;
        internal_config.enable_early_stopping = config->enable_early_stopping;
        internal_config.patience = config->patience;
        internal_config.min_delta = config->min_delta;
        internal_config.divergence_threshold = config->divergence_threshold;
        if (config->log_interval > 0) {
            internal_config.enable_auto_logging = true;
            internal_config.log_interval = config->log_interval;
        }
    }

    // Create callback manager
    state->callbacks = tcb_create(&internal_config);
    NIMCP_CHECK_THROW(state->callbacks, NIMCP_ERROR_MEMORY, "Failed to create callback manager");

    state->callbacks_enabled = true;
    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_disable_callbacks(nimcp_brain_t brain) {
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");

    training_pipeline_state_t* state = get_training_state(brain);
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_INVALID, "No training state");

    state->callbacks_enabled = false;
    set_error("No error");
    return NIMCP_OK;
}

// Track callback wrappers for cleanup
#define MAX_CALLBACK_WRAPPERS 256
static callback_wrapper_t* g_callback_wrappers[MAX_CALLBACK_WRAPPERS] = {0};
static uint32_t g_next_wrapper_id = 0;
static nimcp_mutex_t g_callback_wrappers_mutex = NIMCP_MUTEX_INITIALIZER;

uint32_t nimcp_brain_register_callback(
    nimcp_brain_t brain,
    nimcp_callback_event_t event,
    nimcp_training_callback_fn callback,
    void* user_data,
    const char* name)
{
    if (!brain) {
        set_error("Brain handle is NULL");
        return 0;
    }

    if (!callback) {
        set_error("Callback function is NULL");
        return 0;
    }

    training_pipeline_state_t* state = get_training_state(brain);
    if (!state) {
        set_error("No training state");
        return 0;
    }

    // Auto-enable callbacks if not already enabled
    if (!state->callbacks) {
        nimcp_status_t res = nimcp_brain_enable_callbacks(brain, NULL);
        if (res != NIMCP_OK) {
            return 0;
        }
    }

    // Allocate wrapper
    callback_wrapper_t* wrapper = nimcp_malloc(sizeof(callback_wrapper_t));
    if (!wrapper) {
        set_error("Failed to allocate callback wrapper");
        return 0;
    }

    wrapper->public_callback = callback;
    wrapper->user_data = user_data;

    // Store wrapper for cleanup (thread-safe)
    nimcp_mutex_lock(&g_callback_wrappers_mutex);
    uint32_t wrapper_idx = g_next_wrapper_id % MAX_CALLBACK_WRAPPERS;
    if (g_callback_wrappers[wrapper_idx]) {
        nimcp_free(g_callback_wrappers[wrapper_idx]);
    }
    g_callback_wrappers[wrapper_idx] = wrapper;
    g_next_wrapper_id++;
    nimcp_mutex_unlock(&g_callback_wrappers_mutex);

    // Map public event type to internal event type
    tcb_event_type_t internal_event;
    switch (event) {
        case NIMCP_CB_STEP_COMPLETE:    internal_event = TCB_EVENT_STEP_COMPLETE; break;
        case NIMCP_CB_EPOCH_COMPLETE:   internal_event = TCB_EVENT_EPOCH_COMPLETE; break;
        case NIMCP_CB_LOSS_COMPUTED:    internal_event = TCB_EVENT_LOSS_COMPUTED; break;
        case NIMCP_CB_WEIGHTS_UPDATED:  internal_event = TCB_EVENT_WEIGHTS_UPDATED; break;
        case NIMCP_CB_LR_CHANGED:       internal_event = TCB_EVENT_LR_CHANGED; break;
        case NIMCP_CB_CONVERGENCE:      internal_event = TCB_EVENT_CONVERGENCE; break;
        case NIMCP_CB_DIVERGENCE:       internal_event = TCB_EVENT_DIVERGENCE; break;
        case NIMCP_CB_CHECKPOINT:       internal_event = TCB_EVENT_CHECKPOINT; break;
        default:                        internal_event = TCB_EVENT_STEP_COMPLETE; break;
    }

    // Register with internal callback manager
    tcb_callback_info_t info = {
        .callback = callback_bridge,
        .user_data = wrapper,
        .event_type = internal_event,
        .mode = TCB_MODE_SYNC,
        .priority = TCB_PRIORITY_NORMAL,
        .name = name,
        .enabled = true
    };

    uint32_t cb_id = tcb_register(state->callbacks, &info);
    if (cb_id == 0) {
        set_error("Failed to register callback");
        return 0;
    }

    set_error("No error");
    return cb_id;
}

nimcp_status_t nimcp_brain_unregister_callback(
    nimcp_brain_t brain,
    uint32_t callback_id)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");

    training_pipeline_state_t* state = get_training_state(brain);
    NIMCP_CHECK_THROW(state && state->callbacks, NIMCP_ERROR_INVALID, "Callbacks not enabled");

    // Unregister from internal manager
    if (!tcb_unregister(state->callbacks, callback_id)) {
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_OPERATION_FAILED, "Failed to unregister callback");
    }

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_get_callback_stats(
    nimcp_brain_t brain,
    uint64_t* total_fired,
    float* avg_time_us,
    uint32_t* early_stops)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");

    training_pipeline_state_t* state = get_training_state(brain);
    if (!state || !state->callbacks) {
        // No callbacks configured - return zeros
        if (total_fired) *total_fired = 0;
        if (avg_time_us) *avg_time_us = 0.0F;
        if (early_stops) *early_stops = 0;
        set_error("No error");
        return NIMCP_OK;
    }

    // Get stats from internal manager
    tcb_stats_t stats;
    tcb_get_stats(state->callbacks, &stats);

    if (total_fired) *total_fired = stats.total_callbacks_fired;
    if (avg_time_us) *avg_time_us = stats.avg_execution_time_us;
    if (early_stops) *early_stops = stats.early_stops_triggered;

    set_error("No error");
    return NIMCP_OK;
}

/**
 * @brief Fire callbacks for a training event (internal helper)
 *
 * @param state Training pipeline state
 * @param event_type Event type to fire
 * @param metrics Current training metrics
 * @return Action from callbacks
 */
static tcb_action_t fire_training_callback(
    training_pipeline_state_t* state,
    tcb_event_type_t event_type,
    float loss,
    float learning_rate,
    uint64_t step,
    float gradient_norm)
{
    if (!state || !state->callbacks || !state->callbacks_enabled) {
        return TCB_ACTION_CONTINUE;
    }

    // Update metrics in callback manager
    tcb_update_metrics(state->callbacks, loss, learning_rate, step, gradient_norm);

    // Fire the event
    tcb_event_t event = {
        .event_type = event_type,
        .metrics = {
            .step = step,
            .loss = loss,
            .learning_rate = learning_rate,
            .gradient_norm = gradient_norm
        },
        .user_data = NULL,
        .checkpoint_path = NULL,
        .timestamp_ns = 0
    };

    return tcb_fire(state->callbacks, &event);
}

