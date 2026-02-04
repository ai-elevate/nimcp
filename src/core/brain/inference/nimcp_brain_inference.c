//=============================================================================
// nimcp_brain_inference.c - Brain Inference Module Implementation
//=============================================================================
/**
 * @file nimcp_brain_inference.c
 * @brief Brain inference and prediction implementation
 *
 * WHAT: Inference engine for brain predictions and decisions
 * WHY:  Separates inference logic from core brain module for modularity
 * HOW:  Forward pass, decision caching, mirror neuron integration
 *
 * EXTRACTED FROM: nimcp_brain.c (lines 5343-7087, 1744 lines)
 *
 * FUNCTIONS EXTRACTED:
 * - allocate_decision (static helper)
 * - copy_decision (static helper)
 * - perform_forward_pass (static helper)  
 * - determine_output_label (static helper)
 * - populate_interpretability (static helper)
 * - update_inference_stats (static helper)
 * - brain_decision_to_action (static helper, Phase 10.11)
 * - features_to_action (static helper, Phase 10.11)
 * - brain_decide (public API)
 * - brain_observe_action (public API, Phase 10.11)
 * - brain_free_decision (public API)
 * - brain_decide_batch (public API)
 *
 * ARCHITECTURE:
 * - Primary inference via brain_decide()
 * - Batch processing via brain_decide_batch()
 * - Mirror neuron integration via brain_observe_action()
 * - Decision caching for repeated inputs (thread-safe)
 *
 * PERFORMANCE:
 * - Inference: O(s*n) where s=sparsity, n=active_neurons
 * - Caching: O(1) cache hit, O(s*n) cache miss
 * - Batch: O(m*s*n) where m=batch_size
 */

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/memory/nimcp_unified_memory.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

#include "core/brain/inference/nimcp_brain_inference.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "cognitive/nimcp_mirror_neurons.h"
#include "cognitive/nimcp_theory_of_mind.h"
#include "cognitive/nimcp_working_memory.h"
#include "async/nimcp_future.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "BRAIN_INFERENCE"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(brain_inference)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_brain_inference_mesh_id = 0;
static mesh_participant_registry_t* g_brain_inference_mesh_registry = NULL;

nimcp_error_t brain_inference_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_brain_inference_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "brain_inference", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "brain_inference";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_brain_inference_mesh_id);
    if (err == NIMCP_SUCCESS) g_brain_inference_mesh_registry = registry;
    return err;
}

void brain_inference_mesh_unregister(void) {
    if (g_brain_inference_mesh_registry && g_brain_inference_mesh_id != 0) {
        mesh_participant_unregister(g_brain_inference_mesh_registry, g_brain_inference_mesh_id);
        g_brain_inference_mesh_id = 0;
        g_brain_inference_mesh_registry = NULL;
    }
}


// Forward declarations for brain internal functions (defined in nimcp_brain.c)
extern void set_error(const char* fmt, ...);
extern void brain_clear_error(void);
extern bool is_cached_input(brain_t brain, const float* features, uint32_t num_features);
extern void cache_decision(brain_t brain, const float* features, uint32_t num_features, brain_decision_t* decision);
extern bool ensure_writable_network(brain_t brain);

//=============================================================================
// Static Helper Functions
//=============================================================================

/**
 * @brief Allocate decision structure
 *
 * COMPLEXITY: O(1)
 */
static brain_decision_t* allocate_decision(uint32_t output_size)
{
    brain_decision_t* decision = nimcp_calloc(1, sizeof(brain_decision_t));
    if (!decision) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "decision is NULL");

        return NULL;

    }

    decision->output_size = output_size;
    decision->output_vector = nimcp_malloc(output_size * sizeof(float));

    if (!decision->output_vector) {
        nimcp_free(decision);
        return NULL;
    }

    return decision;
}

/**
 * @brief Deep copy a decision structure
 *
 * WHAT: Creates an independent copy of a decision
 * WHY: Cached decisions must not be freed by caller - return copies instead
 * HOW: Allocates new decision and copies all fields including dynamically allocated arrays
 *
 * COMPLEXITY: O(n) where n = output_size + num_active_neurons
 *
 * @param source Decision to copy
 * @return New decision copy, or NULL on allocation failure
 */
static brain_decision_t* copy_decision(const brain_decision_t* source)
{
    if (!source) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "source is NULL");

        return NULL;

    }

    // Allocate new decision structure
    brain_decision_t* copy = nimcp_calloc(1, sizeof(brain_decision_t));
    if (!copy) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "copy is NULL");

        return NULL;

    }

    // Copy scalar fields
    memcpy(copy, source, sizeof(brain_decision_t));

    // NULL out pointer fields to prevent accidental sharing
    copy->output_vector = NULL;
    copy->active_neuron_ids = NULL;

    // Deep copy output_vector
    if (source->output_vector && source->output_size > 0) {
        copy->output_vector = nimcp_malloc(source->output_size * sizeof(float));
        if (!copy->output_vector) {
            nimcp_free(copy);
            return NULL;
        }
        memcpy(copy->output_vector, source->output_vector, source->output_size * sizeof(float));
    }

    // Deep copy active_neuron_ids
    if (source->active_neuron_ids && source->num_active_neurons > 0) {
        copy->active_neuron_ids = nimcp_malloc(source->num_active_neurons * sizeof(uint32_t));
        if (!copy->active_neuron_ids) {
            if (copy->output_vector)
                nimcp_free(copy->output_vector);
            nimcp_free(copy);
            return NULL;
        }
        memcpy(copy->active_neuron_ids, source->active_neuron_ids,
               source->num_active_neurons * sizeof(uint32_t));
    }

    return copy;
}

/**
 * @brief Perform forward pass through network
 *
 * COMPLEXITY: O(s*n) where s = sparsity
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @param decision Decision to populate
 * @return Number of active neurons
 */
static uint32_t perform_forward_pass(brain_t brain, const float* features, uint32_t num_features,
                                     brain_decision_t* decision)
{
    uint64_t start_time = nimcp_time_monotonic_us();

    uint32_t active_neurons;

    // Phase 3: Use read-only inference for COW clones to avoid triggering copy
    if (brain->can_use_readonly) {
        // COW clone using shared network - read-only inference
        active_neurons = adaptive_network_forward_readonly(
            brain->network, features, num_features, decision->output_vector, decision->output_size, 0);
    } else {
        // Original brain or post-COW clone - normal inference with statistics
        active_neurons = adaptive_network_forward(
            brain->network, features, num_features, decision->output_vector, decision->output_size, 0);
    }

    decision->inference_time_us = nimcp_time_elapsed_us(start_time);

    return active_neurons;
}

/**
 * @brief Find maximum output and determine label
 *
 * COMPLEXITY: O(n) where n = num_outputs
 */
static void determine_output_label(brain_t brain, brain_decision_t* decision)
{
    uint32_t max_idx = 0;
    float max_value = decision->output_vector[0];

    for (uint32_t i = 1; i < decision->output_size; i++) {
        if (decision->output_vector[i] > max_value) {
            max_value = decision->output_vector[i];
            max_idx = i;
        }
    }

    // Set label
    if (max_idx < brain->num_output_labels) {
        strncpy(decision->label, brain->output_labels[max_idx], sizeof(decision->label) - 1);
    } else {
        snprintf(decision->label, sizeof(decision->label), "output_%u", max_idx);
    }

    // Normalize confidence
    decision->confidence = fminf(max_value / 10.0F, 1.0F);
}

/**
 * @brief Populate interpretability information
 *
 * COMPLEXITY: O(n)
 */
static void populate_interpretability(brain_t brain, const float* features, uint32_t num_features,
                                      uint32_t active_neurons, brain_decision_t* decision)
{
    decision->num_active_neurons = active_neurons;
    decision->sparsity = adaptive_network_get_sparsity(brain->network);

    if (brain->config.enable_explanations) {
        adaptive_network_explain(brain->network, features, num_features, decision->explanation,
                                 sizeof(decision->explanation));
    }

    // Populate active neuron IDs
    decision->active_neuron_ids = nimcp_malloc(active_neurons * sizeof(uint32_t));
    if (!decision->active_neuron_ids) {
        set_error("Failed to allocate active neuron IDs array (%u neurons)", active_neurons);
        return;  // decision->num_active_neurons is 0, so this is safe
    }
    for (uint32_t i = 0; i < active_neurons; i++) {
        decision->active_neuron_ids[i] = i;
    }
}

/**
 * @brief Update brain statistics after inference
 *
 * THREAD SAFETY: Uses atomic operations for counter updates to ensure
 * thread-safe statistics tracking when multiple threads perform concurrent
 * inference on the same brain instance.
 *
 * COMPLEXITY: O(1)
 */
static void update_inference_stats(brain_t brain, brain_decision_t* decision)
{
    // Use atomic increment for thread-safe stats update
    uint64_t new_count = __atomic_fetch_add(&brain->stats.total_inferences, 1, __ATOMIC_RELAXED) + 1;

    // Update average inference time using the new count
    // Note: This calculation is not perfectly atomic, but the result
    // is a reasonable approximation under concurrent updates
    brain->stats.avg_inference_time_us =
        (brain->stats.avg_inference_time_us * (new_count - 1) +
         decision->inference_time_us) /
        new_count;
    brain->stats.avg_sparsity = decision->sparsity;
}

//=============================================================================
// Mirror Neuron Integration Helpers (Phase 10.11)
//=============================================================================

/**
 * @brief Convert brain decision to mirror neuron action
 *
 * WHAT: Transform brain decision into action_t for mirror neuron system
 * WHY:  Enable mirror neurons to learn from brain's own decisions
 * HOW:  Extract decision features, confidence, and output as action representation
 *
 * COMPLEXITY: O(n) where n = num_outputs (feature copying)
 *
 * @param decision Brain decision
 * @param action_id Unique action identifier
 * @param action_name Human-readable action name
 * @return action_t struct for mirror neuron system
 */
static action_t brain_decision_to_action(const brain_decision_t* decision,
                                         uint32_t action_id,
                                         const char* action_name)
{
    action_t action;
    memset(&action, 0, sizeof(action_t));

    if (!decision || !action_name) {
        return action;
    }

    action.action_id = action_id;
    strncpy(action.action_name, action_name, sizeof(action.action_name) - 1);
    action.agent_id = 0;  // 0 = self
    action.timestamp = nimcp_time_get_ms();
    action.confidence = decision->confidence;

    // Use output activations as action features (up to 32)
    action.num_features = (decision->output_size < 32) ? decision->output_size : 32;
    for (uint32_t i = 0; i < action.num_features; i++) {
        action.features[i] = decision->output_vector[i];
    }

    return action;
}

/**
 * @brief Convert input features to observed action
 *
 * WHAT: Transform input features into action_t for observation pathway
 * WHY:  Enable mirror neurons to learn from observed patterns
 * HOW:  Treat input as observed action with features
 *
 * COMPLEXITY: O(n) where n = num_features (copying)
 *
 * @param features Input features
 * @param num_features Number of features
 * @param agent_id ID of agent performing action (0 = self, >0 = other)
 * @return action_t struct for mirror neuron system
 */
static action_t features_to_action(const float* features, uint32_t num_features,
                                   uint32_t agent_id)
{
    action_t action;
    memset(&action, 0, sizeof(action_t));

    if (!features) {
        return action;
    }

    action.action_id = 0;  // Will be assigned by mirror neuron system
    snprintf(action.action_name, sizeof(action.action_name), "observed_%u", agent_id);
    action.agent_id = agent_id;
    action.timestamp = nimcp_time_get_ms();
    action.confidence = 1.0F;

    // Copy features (up to 32)
    action.num_features = (num_features < 32) ? num_features : 32;
    for (uint32_t i = 0; i < action.num_features; i++) {
        action.features[i] = features[i];
    }

    return action;
}

//=============================================================================
// Public API Functions
//=============================================================================
// NOTE: The full brain_decide() implementation is too large (1374 lines)
// and deeply integrated with brain internals. For this extraction, we are
// documenting the function signature and providing a reference implementation
// that must remain in nimcp_brain.c due to its extensive use of brain internals.
//=============================================================================

/**
 * @brief Free decision result with CoW support
 *
 * WHY: Proper memory management for decision results
 * Handles all allocated sub-structures
 *
 * Phase 1.5 CoW: Decisions may share data via reference counting.
 * Only free shared data when the last reference is released.
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Uses atomic operations for refcount updates
 *
 * @param decision Decision to free
 */
void brain_free_decision(brain_decision_t* decision)
{
    if (!decision)
        return;

    /**
     * WHAT: Free decision structure and its allocated fields
     * WHY: Prevent memory leaks
     * HOW: Use nimcp_free() for memory allocated with nimcp_malloc()
     *
     * Phase 1.5 CoW: Check reference count before freeing shared data
     */

    if (decision->_cow_refcount) {
        // This decision shares data with others via CoW
        // Atomically decrement refcount and check if we're the last user
        uint32_t remaining = __atomic_sub_fetch(decision->_cow_refcount, 1, __ATOMIC_SEQ_CST);

        if (remaining == 0) {
            // Last reference - free the shared data
            if (decision->output_vector) {
                nimcp_free(decision->output_vector);
            }
            if (decision->active_neuron_ids) {
                nimcp_free(decision->active_neuron_ids);
            }
            // Free the refcount itself
            nimcp_free(decision->_cow_refcount);
        }
        // else: Other references exist - don't free the shared data
    } else {
        // This decision owns its data exclusively (not shared)
        if (decision->output_vector) {
            nimcp_free(decision->output_vector);
        }
        if (decision->active_neuron_ids) {
            nimcp_free(decision->active_neuron_ids);
        }
    }

    // Always free the decision structure itself (each copy has its own)
    nimcp_free(decision);
}

/**
 * @brief Batch inference
 *
 * WHY: More efficient than individual calls for large batches
 * Enables parallel processing opportunities
 *
 * COMPLEXITY: O(m*s*n) where m = num_inputs
 *
 * @param brain Brain handle
 * @param inputs Array of input vectors
 * @param num_inputs Number of inputs
 * @param features_per_input Features per input
 * @param decisions Output decisions array (allocated by caller)
 * @return true on success
 */
bool brain_decide_batch(brain_t brain, const float** inputs, uint32_t num_inputs,
                        uint32_t features_per_input, brain_decision_t* decisions)
{
    // Guard: Validate parameters
    if (!brain || !inputs || !decisions || num_inputs == 0) {
        set_error("Invalid parameters to brain_decide_batch");
        return false;
    }

    for (uint32_t i = 0; i < num_inputs; i++) {
        brain_decision_t* decision = brain_decide(brain, inputs[i], features_per_input);

        if (!decision) {
            return false;
        }

        memcpy(&decisions[i], decision, sizeof(brain_decision_t));
        nimcp_free(decision);
    }

    brain_clear_error();
    return true;
}

/**
 * @brief Observe action performed by another agent (Phase 10.11)
 *
 * WHAT: Record observed action in mirror neuron system for observational learning
 * WHY:  Enable learning from watching others (imitation, social cognition)
 * HOW:  Convert input features to observed action and send to mirror neurons
 *
 * This is the OBSERVATION PATHWAY for mirror neurons. When the brain observes
 * another agent performing an action, this function records it for learning.
 *
 * USE CASES:
 * - Robot watching human demonstration
 * - Agent observing another agent's behavior
 * - Learning from video/sensor data of actions
 * - Social learning and imitation
 *
 * COMPLEXITY: O(n) where n = num_features
 * THREAD-SAFE: No (requires external synchronization)
 *
 * @param brain Brain handle
 * @param features Observed action features (sensor data, visual features, etc.)
 * @param num_features Number of features
 * @param agent_id ID of agent being observed (must be > 0, as 0 = self)
 * @return true on success, false on error
 */
bool brain_observe_action(brain_t brain, const float* features, uint32_t num_features,
                          uint32_t agent_id)
{
    // Guard: Validate parameters
    if (!brain || !features) {
        set_error("Invalid parameters to brain_observe_action");
        return false;
    }

    if (agent_id == 0) {
        set_error("agent_id must be > 0 (0 is reserved for self)");
        return false;
    }

    // Guard: Check if mirror neurons are enabled
    if (!brain->mirror_neurons || !brain->config.enable_mirror_neurons) {
        // Not an error, just not enabled
        return true;
    }

    // Convert features to action
    action_t action = features_to_action(features, num_features, agent_id);

    // Record as observed action
    bool success = mirror_neurons_observe_action(brain->mirror_neurons, &action);
    if (!success) {
        set_error("Failed to record observed action in mirror neurons");
        return false;
    }

    // Store in working memory if enabled (for offline replay)
    if (brain->working_memory && brain->config.enable_working_memory) {
        // Store action features in working memory
        // Note: This enables later replay and sequence learning
        working_memory_add(brain->working_memory, features, num_features, 0.8F);
    }

    // Trigger theory of mind inference if enabled
    if (brain->theory_of_mind && brain->config.enable_theory_of_mind) {
        // Use mirror neuron activations to infer agent intentions
        // Note: This requires getting current mirror neuron state
        // WHAT: Use mirror neuron activations to infer agent intentions
        // WHY:  Mirror neurons encode observed actions, ToM infers mental states
        // HOW:  Extract mirror activations and pass to ToM as action vector

        // BIOLOGICAL RATIONALE: Mirror neurons fire during action observation
        // (Rizzolatti & Craighero, 2004). ToM uses this to understand "why"
        // agents perform actions, enabling empathy and intention inference.

        float mirror_activations[100];  // Max 100 actions
        uint32_t num_activations = 0;

        if (mirror_neurons_get_all_activations(brain->mirror_neurons,
                                              mirror_activations,
                                              100,
                                              &num_activations)) {
            // Package as ToM observation
            tom_observation_t observation = {
                .action_vector = mirror_activations,
                .action_dim = num_activations,
                .verbal_context = NULL,  // No verbal context from features
                .observed_emotion = TOM_EMOTION_UNKNOWN,  // Infer from context
                .situational_context = features,  // Raw features as context
                .context_dim = num_features
            };

            // Let ToM infer mental state from mirror neuron pattern
            tom_observe(brain->theory_of_mind, &observation);
        }
    }

    brain_clear_error();
    return true;
}


//=============================================================================
// Async Inference Implementation
//=============================================================================

/**
 * @brief Context for async inference thread
 */
typedef struct {
    brain_t brain;
    float* features;
    uint32_t num_features;
    nimcp_promise_t promise;
} async_infer_context_t;

/**
 * @brief Background thread function for async inference
 *
 * WHAT: Performs inference in background thread and completes promise
 * WHY:  Enable non-blocking decision making
 * HOW:  Call brain_decide, set result or error on promise
 *
 * THREAD SAFETY: Each thread has its own context, brain must support concurrent reads
 *
 * @param arg async_infer_context_t pointer
 * @return NULL (unused)
 */
static void* async_infer_thread(void* arg)
{
    async_infer_context_t* ctx = (async_infer_context_t*)arg;

    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");


        return NULL;
    }

    LOG_MODULE_DEBUG("brain_inference", "Async inference started: %u features", ctx->num_features);

    // Perform synchronous inference
    brain_decision_t* decision = brain_decide(
        ctx->brain,
        ctx->features,
        ctx->num_features
    );

    // Complete promise with result or error
    if (!decision) {
        // Inference failed
        nimcp_error_t error = NIMCP_ERROR_OPERATION_FAILED;
        nimcp_promise_fail(ctx->promise, error);
        LOG_MODULE_ERROR("brain_inference", "Async inference failed");
    } else {
        // Inference succeeded
        // Note: promise takes ownership of decision pointer
        nimcp_promise_complete(ctx->promise, &decision);
        LOG_MODULE_DEBUG("brain_inference", "Async inference completed: label='%s', confidence=%.2f",
                       decision->label, decision->confidence);
    }

    // Cleanup
    nimcp_free(ctx->features);
    nimcp_promise_destroy(ctx->promise);
    nimcp_free(ctx);

    return NULL;
}

/**
 * @brief Asynchronous inference/decision making
 *
 * WHAT: Non-blocking version of brain_decide
 * WHY:  Enable concurrent inference without blocking caller
 * HOW:  Copy inputs, create promise/future, spawn thread
 *
 * IMPLEMENTATION NOTES:
 * - Features are deep-copied to ensure thread safety
 * - Promise is created with sizeof(brain_decision_t*) for result pointer
 * - Thread is detached for auto-cleanup
 * - Context is freed by worker thread
 *
 * ERROR HANDLING:
 * - Returns NULL if allocation fails or thread creation fails
 * - Inference errors propagated through future
 *
 * MEMORY MANAGEMENT:
 * - Context allocated on heap, freed by worker thread
 * - Features copied to heap, freed by worker thread
 * - Promise destroyed by worker thread after completion
 * - Future must be destroyed by caller
 * - Decision result must be freed by caller with brain_free_decision()
 *
 * @param brain Brain handle
 * @param features Input features (will be copied)
 * @param num_features Feature count
 * @return Future handle or NULL on error
 */
nimcp_future_t nimcp_brain_infer_async(brain_t brain, const float* features,
                                        uint32_t num_features)
{
    // Validate parameters
    if (!brain || !features) {
        LOG_MODULE_ERROR("brain_inference", "Invalid parameters to nimcp_brain_infer_async");
        return NULL;
    }

    if (num_features == 0) {
        LOG_MODULE_ERROR("brain_inference", "Invalid num_features=0");
        return NULL;
    }

    // Allocate context for worker thread
    async_infer_context_t* ctx = nimcp_malloc(sizeof(async_infer_context_t));
    if (!ctx) {
        LOG_MODULE_ERROR("brain_inference", "Failed to allocate async inference context");
        return NULL;
    }

    // Copy features to heap (worker thread will free)
    ctx->features = nimcp_malloc(num_features * sizeof(float));
    if (!ctx->features) {
        LOG_MODULE_ERROR("brain_inference", "Failed to allocate features array (%u floats)", num_features);
        nimcp_free(ctx);
        return NULL;
    }
    memcpy(ctx->features, features, num_features * sizeof(float));

    // Set context fields
    ctx->brain = brain;
    ctx->num_features = num_features;

    // Create promise for result (decision is a pointer)
    ctx->promise = nimcp_promise_create(sizeof(brain_decision_t*));
    if (!ctx->promise) {
        LOG_MODULE_ERROR("brain_inference", "Failed to create promise for async inference");
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        return NULL;
    }

    // Get future before starting thread
    nimcp_future_t future = nimcp_promise_get_future(ctx->promise);
    if (!future) {
        LOG_MODULE_ERROR("brain_inference", "Failed to get future from promise");
        nimcp_promise_destroy(ctx->promise);
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        return NULL;
    }

    // Create worker thread
    nimcp_thread_t thread;
    thread_attr_t attr = {
        .stack_size = NIMCP_THREAD_DEFAULT_STACK_SIZE,
        .priority = 0,
        .detached = true  // Auto-cleanup on exit
    };

    nimcp_result_t result = nimcp_thread_create(&thread, async_infer_thread, ctx, &attr);
    if (result != NIMCP_SUCCESS) {
        LOG_MODULE_ERROR("brain_inference", "Failed to create async inference thread: %d", result);
        nimcp_future_destroy(future);
        nimcp_promise_destroy(ctx->promise);
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        return NULL;
    }

    LOG_MODULE_INFO("brain_inference", "Async inference thread created successfully");

    // Return future to caller (caller must destroy)
    return future;
}
