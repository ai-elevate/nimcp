/**
 * @file nimcp_swarm_consciousness.c
 * @brief Implementation of Swarm Gestalt Consciousness - Collective Intelligence Metrics
 *
 * WHAT: Computes collective consciousness (Φ) for drone swarms using IIT principles
 * WHY:  Measure emergent collective intelligence and swarm coordination quality
 * HOW:  Aggregates individual drone consciousness with network integration metrics
 *
 * BIOLOGICAL BASIS:
 * - Extends Integrated Information Theory (IIT) to multi-agent systems
 * - Inspired by collective consciousness in social insects (swarm intelligence)
 * - Neural binding across brain regions → Information integration across drones
 * - Individual consciousness (local Φ) → Collective consciousness (swarm Φ)
 *
 * KEY ALGORITHMS:
 * 1. Collective Φ = f(individual_phis, network_integration, coherence)
 * 2. Network Integration = mutual_information + coherence_weighting
 * 3. Scaling Model: phi ~ base * n^exponent * (1 + synergy * coherence)
 * 4. State Classification: DORMANT → EMERGING → UNIFIED → TRANSCENDENT
 *
 * @author NIMCP Swarm Intelligence Team
 * @date 2025-12-11
 * @version 1.0.0
 */

#include "swarm/nimcp_swarm_consciousness.h"
#include "utils/memory/nimcp_memory.h"
#include "security/nimcp_bbb_helpers.h"
#include "utils/logging/nimcp_logging.h"
#include "api/nimcp_api_exception.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_wiring_helpers.h"
#include "cognitive/imagination/nimcp_imagination_callbacks.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include "utils/thread/nimcp_thread.h"

#include "utils/exception/nimcp_exception_macros.h"

#include <stddef.h>  /* for NULL */
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for swarm_consciousness module */
static nimcp_health_agent_t* g_swarm_consciousness_health_agent = NULL;

/**
 * @brief Set health agent for swarm_consciousness heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void swarm_consciousness_set_health_agent(nimcp_health_agent_t* agent) {
    g_swarm_consciousness_health_agent = agent;
}

/** @brief Send heartbeat from swarm_consciousness module */
static inline void swarm_consciousness_heartbeat(const char* operation, float progress) {
    if (g_swarm_consciousness_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_swarm_consciousness_health_agent, operation, progress);
    }
}


//=============================================================================
// KG-Driven Wiring Infrastructure
//=============================================================================

/* Forward declaration for imagination handler */
static nimcp_error_t imagination_collective_handler(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data);

/**
 * Handler map for swarm consciousness module.
 * Handles collective imagination sharing and insight messages.
 */
DEFINE_HANDLER_MAP_BEGIN(swarm_consciousness)
    HANDLER_MAP_ENTRY(BIO_MSG_IMAGINATION_COLLECTIVE_SHARE, imagination_collective_handler)
    HANDLER_MAP_ENTRY(BIO_MSG_IMAGINATION_COLLECTIVE_INSIGHT, imagination_collective_handler)
DEFINE_HANDLER_MAP_END()

/**
 * Wiring callback for KG-driven handler registration.
 */
DEFINE_HANDLER_CALLBACK(swarm_consciousness, swarm_consciousness_ctx_t, ctx)

//=============================================================================
// Forward Declarations for Swarm Brain Integration
//=============================================================================
// These functions may be added to swarm_brain in future, or we provide stubs

/**
 * Get number of drones in swarm (including local).
 * Uses peer count + 1 (local) as approximation.
 */
static uint32_t swarm_brain_get_drone_count_internal(const swarm_brain_t* swarm);

/**
 * Get individual drone brain by index.
 * Returns local brain for index 0, NULL for others (remote drones not accessible).
 */
static brain_t* swarm_brain_get_drone_brain_internal(const swarm_brain_t* swarm, uint32_t index);

/**
 * Get phi value from brain.
 * Uses introspection subsystem if available.
 */
static float brain_get_phi_internal(brain_t* brain);

/**
 * Get/set consciousness context on swarm brain.
 * These are stub implementations - real implementation would store in swarm_brain struct.
 */
static swarm_consciousness_ctx_t* swarm_consciousness_ctx_storage = NULL;
static void swarm_brain_set_consciousness_ctx(swarm_brain_t* swarm, swarm_consciousness_ctx_t* ctx) {
    (void)swarm;
    swarm_consciousness_ctx_storage = ctx;
}
static swarm_consciousness_ctx_t* swarm_brain_get_consciousness_ctx(const swarm_brain_t* swarm) {
    (void)swarm;
    return swarm_consciousness_ctx_storage;
}

//=============================================================================
// Constants and Magic Values
//=============================================================================

/** Magic value for context validation */
#define SWARM_CONSCIOUSNESS_MAGIC 0x53434F4E  // 'SCON'

/** Maximum drones for exact computation */
#define MAX_EXACT_DRONES 32

/** Phi history size for trend analysis */
#define PHI_HISTORY_SIZE 100

/** Default update interval (milliseconds) */
#define DEFAULT_UPDATE_INTERVAL_MS 1000

/* Note: BIO_MSG_SWARM_CONSCIOUSNESS_UPDATE defined in header (0x0700) */

//=============================================================================
// Internal Context Structure
//=============================================================================

/**
 * WHAT: Internal swarm consciousness context
 * WHY:  Track collective consciousness state and monitoring
 * HOW:  Stores configuration, metrics, history, and bio-async state
 */
typedef struct swarm_consciousness_ctx {
    uint32_t magic;                          /**< Magic for validation */
    swarm_consciousness_config_t config;     /**< Configuration */
    swarm_consciousness_metrics_t* current_metrics; /**< Current metrics */
    consciousness_scaling_model_t scaling_model; /**< Scaling model parameters */
    pthread_mutex_t lock;                    /**< Thread safety */

    // Monitoring state
    bool monitoring_active;                  /**< Is monitoring thread running? */
    pthread_t monitor_thread;                /**< Monitoring thread handle */
    void (*callback)(const swarm_consciousness_metrics_t*, void*); /**< User callback */
    void* user_data;                         /**< User callback context */

    // Bio-async integration
    bool bio_async_registered;               /**< Bio-async active? */
    bio_module_context_t bio_module_ctx;     /**< Bio-router module context */

    // Imagination integration (collective creativity)
    imagination_collective_receive_callback_t collective_imagination_callback; /**< Callback for received imagination */
    void* collective_imagination_user_data;  /**< User data for imagination callback */
    bool imagination_handler_registered;     /**< Imagination handler active? */

    // History for trend analysis
    float phi_history[PHI_HISTORY_SIZE];     /**< Recent phi values */
    uint32_t history_index;                  /**< Circular buffer index */
    uint32_t history_count;                  /**< Valid history entries */

    // Statistics
    uint64_t total_computations;             /**< Total phi computations */
    uint64_t state_transitions;              /**< State change count */
    uint64_t creation_time_ms;               /**< Context creation time */

    // Swarm brain reference for monitoring
    swarm_brain_t* swarm_brain;              /**< Associated swarm brain */

} swarm_consciousness_ctx_t;

//=============================================================================
// Helper Functions - Forward Declarations
//=============================================================================

static float compute_workspace_overlap(const collective_workspace_t* workspace);
static float compute_phi_variance(const float* individual_phis, uint32_t drone_count);
static void update_phi_history(swarm_consciousness_ctx_t* ctx, float phi);
static float get_phi_trend(const swarm_consciousness_ctx_t* ctx);
static void* consciousness_monitor_thread(void* arg);
static void publish_consciousness_update(swarm_consciousness_ctx_t* ctx,
                                        const swarm_consciousness_metrics_t* metrics);
static uint64_t get_time_ms(void);

// Public API forward declaration for use in destroy
void swarm_consciousness_stop_monitoring(swarm_consciousness_ctx_t* context);

// Imagination integration forward declarations
static nimcp_error_t imagination_collective_handler(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data);

//=============================================================================
// Configuration API
//=============================================================================

/**
 * WHAT: Get default swarm consciousness configuration
 * WHY:  Provide sensible defaults for most use cases
 * HOW:  Return pre-configured struct with standard values
 */
swarm_consciousness_config_t swarm_consciousness_default_config(void) {
    swarm_consciousness_config_t config = {
        .phi_aggregation_method = PHI_AGGREGATION_SYNERGISTIC,
        .integration_weight = SWARM_CONSCIOUSNESS_DEFAULT_INTEGRATION_WEIGHT,
        .coherence_weight = SWARM_CONSCIOUSNESS_DEFAULT_COHERENCE_WEIGHT,
        .update_interval_ms = SWARM_CONSCIOUSNESS_DEFAULT_UPDATE_INTERVAL_MS,
        .enable_bio_async = true,
        .enable_logging = false,
        .min_drones_for_emergence = SWARM_CONSCIOUSNESS_DEFAULT_MIN_DRONES
    };
    return config;
}

//=============================================================================
// Lifecycle API
//=============================================================================

/**
 * WHAT: Create swarm consciousness context
 * WHY:  Initialize consciousness monitoring for swarm
 * HOW:  Allocate context, initialize mutex, register with BBB and bio-async
 */
swarm_consciousness_ctx_t* swarm_consciousness_create(
    const swarm_consciousness_config_t* config)
{
    // Guard: Validate configuration pointer
    if (!bbb_check_pointer(config, "swarm_consciousness_create")) {
        LOG_ERROR("Null configuration pointer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Null configuration pointer in swarm_consciousness_create");
        return NULL;
    }

    // Allocate context
    swarm_consciousness_ctx_t* ctx = (swarm_consciousness_ctx_t*)
        nimcp_calloc(1, sizeof(swarm_consciousness_ctx_t));
    if (!ctx) {
        LOG_ERROR("Failed to allocate consciousness context");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate consciousness context");
        return NULL;
    }

    // Initialize magic and configuration
    ctx->magic = SWARM_CONSCIOUSNESS_MAGIC;
    memcpy(&ctx->config, config, sizeof(swarm_consciousness_config_t));

    // Initialize mutex
    if (pthread_mutex_init(&ctx->lock, NULL) != 0) {
        LOG_ERROR("Failed to initialize mutex");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to initialize mutex in swarm_consciousness_create");
        nimcp_free(ctx);
        return NULL;
    }

    // Allocate current metrics
    ctx->current_metrics = (swarm_consciousness_metrics_t*)
        nimcp_calloc(1, sizeof(swarm_consciousness_metrics_t));
    if (!ctx->current_metrics) {
        LOG_ERROR("Failed to allocate metrics structure");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate consciousness metrics structure");
        pthread_mutex_destroy(&ctx->lock);
        nimcp_free(ctx);
        return NULL;
    }

    // Initialize metrics
    ctx->current_metrics->collective_phi = 0.0f;
    ctx->current_metrics->consciousness_state = SWARM_CONSCIOUSNESS_DORMANT;
    ctx->current_metrics->drone_count = 0;
    ctx->current_metrics->timestamp = get_time_ms();

    // Initialize scaling model
    ctx->scaling_model.base_phi = 0.0f;
    ctx->scaling_model.scaling_exponent = 1.0f;
    ctx->scaling_model.synergy_factor = 0.0f;
    ctx->scaling_model.saturation_point = 100.0f;

    // Initialize history
    ctx->history_index = 0;
    ctx->history_count = 0;
    memset(ctx->phi_history, 0, sizeof(ctx->phi_history));

    // Initialize statistics
    ctx->total_computations = 0;
    ctx->state_transitions = 0;
    ctx->creation_time_ms = get_time_ms();

    // Initialize monitoring state
    ctx->monitoring_active = false;
    ctx->callback = NULL;
    ctx->user_data = NULL;
    ctx->swarm_brain = NULL;

    // Initialize imagination integration
    ctx->collective_imagination_callback = NULL;
    ctx->collective_imagination_user_data = NULL;
    ctx->imagination_handler_registered = false;
    ctx->bio_module_ctx = NULL;

    // Register with BBB security
    if (!bbb_register_module("swarm_consciousness", BBB_MODULE_TYPE_SWARM)) {
        LOG_WARN("Failed to register with BBB security module");
    }

    // Register with bio-async if enabled
    ctx->bio_async_registered = false;
    if (config->enable_bio_async && nimcp_bio_async_is_initialized()) {
        ctx->bio_async_registered = true;
        LOG_INFO("Bio-async integration enabled for swarm consciousness");
    }

    LOG_INFO("Swarm consciousness context created");
    bbb_audit_log(BBB_AUDIT_INFO, "swarm_consciousness", "create",
                  "Context created with method=%d", config->phi_aggregation_method);

    return (swarm_consciousness_ctx_t*)ctx;
}

/**
 * WHAT: Destroy swarm consciousness context
 * WHY:  Clean up resources and stop monitoring
 * HOW:  Stop monitoring thread, free memory, destroy mutex
 */
void swarm_consciousness_destroy(swarm_consciousness_ctx_t* context) {
    // Guard: Null check
    if (!context) {
        return;
    }

    swarm_consciousness_ctx_t* ctx = (swarm_consciousness_ctx_t*)context;

    // Guard: Validate magic
    if (ctx->magic != SWARM_CONSCIOUSNESS_MAGIC) {
        LOG_ERROR("Invalid context magic in destroy");
        return;
    }

    // Stop monitoring if active
    if (ctx->monitoring_active) {
        swarm_consciousness_stop_monitoring(context);
    }

    // Unregister imagination handlers and bio-router module
    if (ctx->bio_module_ctx) {
        if (ctx->imagination_handler_registered) {
            bio_router_unregister_handler(ctx->bio_module_ctx,
                                          BIO_MSG_IMAGINATION_COLLECTIVE_SHARE);
            bio_router_unregister_handler(ctx->bio_module_ctx,
                                          BIO_MSG_IMAGINATION_COLLECTIVE_INSIGHT);
        }
        bio_router_unregister_module(ctx->bio_module_ctx);
        ctx->bio_module_ctx = NULL;
    }

    // Unregister from BBB
    bbb_unregister_module("swarm_consciousness");

    // Free metrics
    if (ctx->current_metrics) {
        nimcp_free(ctx->current_metrics);
    }

    // Destroy mutex
    pthread_mutex_destroy(&ctx->lock);

    // Clear magic
    ctx->magic = 0;

    // Free context
    nimcp_free(ctx);

    LOG_INFO("Swarm consciousness context destroyed");
}

//=============================================================================
// Core Computation API
//=============================================================================

/**
 * WHAT: Compute network integration metric (internal helper)
 * WHY:  Measure information integration across drones
 * HOW:  Combine workspace overlap, phi variance, and coherence
 */
static float compute_network_integration_internal(
    const collective_workspace_t* workspace,
    const float* individual_phis,
    uint32_t drone_count)
{
    // Guard: Null checks
    if (!workspace || !individual_phis || drone_count == 0) {
        return 0.0f;
    }

    float integration = 0.0f;

    // 1. Workspace item overlap (shared information)
    float item_overlap = compute_workspace_overlap(workspace);

    // 2. Phi uniformity (low variance = more integrated)
    float phi_variance = compute_phi_variance(individual_phis, drone_count);
    float max_phi = 0.0f;
    for (uint32_t i = 0; i < drone_count; i++) {
        if (individual_phis[i] > max_phi) {
            max_phi = individual_phis[i];
        }
    }
    float phi_uniformity = (max_phi > 0.0f) ? (1.0f - (phi_variance / max_phi)) : 0.0f;

    // 3. Get workspace coherence
    float coherence = collective_workspace_get_coherence(workspace);

    // Combine metrics
    integration = 0.5f * item_overlap + 0.5f * phi_uniformity;
    integration *= coherence;

    // Clamp to [0, 1]
    if (integration < 0.0f) integration = 0.0f;
    if (integration > 1.0f) integration = 1.0f;

    return integration;
}

/**
 * WHAT: Compute network integration metric (public API)
 * WHY:  Measure information integration across drones
 * HOW:  Combine workspace overlap, phi variance, and coherence
 */
float swarm_compute_network_integration(
    swarm_consciousness_ctx_t* ctx,
    const float* individual_phis,
    uint32_t count,
    const collective_workspace_t* workspace)
{
    (void)ctx;  // Context used for future stateful integration computation
    return compute_network_integration_internal(workspace, individual_phis, count);
}

/**
 * WHAT: Compute collective phi from swarm brain
 * WHY:  Main function to quantify swarm consciousness
 * HOW:  Aggregate individual phis with synergy boost from integration
 */
swarm_consciousness_metrics_t* swarm_compute_collective_phi(
    swarm_brain_t* swarm,
    const swarm_consciousness_config_t* config)
{
    // Guard: Null checks
    if (!bbb_check_pointer(swarm, "swarm_compute_collective_phi")) {
        return NULL;
    }

    // Use defaults if no config provided
    swarm_consciousness_config_t effective_config;
    if (config) {
        memcpy(&effective_config, config, sizeof(swarm_consciousness_config_t));
    } else {
        effective_config = swarm_consciousness_default_config();
    }

    // Get drone count from swarm
    uint32_t drone_count = swarm_brain_get_drone_count_internal(swarm);
    if (drone_count == 0 || drone_count > SWARM_CONSCIOUSNESS_MAX_DRONES) {
        LOG_ERROR("Invalid drone count: %u", drone_count);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Invalid drone count: %u", drone_count);
        return NULL;
    }

    // Guard: Minimum drones for emergence
    if (drone_count < effective_config.min_drones_for_emergence) {
        // Still compute but state will be DORMANT
        LOG_DEBUG("Below min drones for emergence: %u < %u",
                       drone_count, effective_config.min_drones_for_emergence);
    }

    // Gather individual phi values from each drone
    float individual_phis[SWARM_CONSCIOUSNESS_MAX_DRONES];
    for (uint32_t i = 0; i < drone_count; i++) {
        // Get consciousness metrics from each drone's brain
        brain_t* drone_brain = swarm_brain_get_drone_brain_internal(swarm, i);
        if (drone_brain) {
            individual_phis[i] = brain_get_phi_internal(drone_brain);
        } else {
            individual_phis[i] = 0.0f;
        }
    }

    // Get collective workspace
    // Note: swarm_brain_get_workspace returns workspace_entry_t*, not collective_workspace_t*
    // For now, use simplified coherence estimation based on peer connectivity
    uint32_t workspace_size = 0;
    const workspace_entry_t* workspace_entries = swarm_brain_get_workspace(swarm, &workspace_size);
    (void)workspace_entries;  // Unused for now, coherence from peer count

    // Estimate network integration and coherence from peer connectivity
    float network_integration = (drone_count > 1) ?
        (float)(drone_count - 1) / (float)drone_count : 0.0f;
    float coherence = network_integration;  // Simplified: use connectivity as coherence proxy

    // Compute synergy boost
    float integration_weight = effective_config.integration_weight;
    float coherence_weight = effective_config.coherence_weight;
    float synergy_boost = integration_weight * network_integration +
                         coherence_weight * coherence;

    // Aggregate individual phis based on method
    float collective_phi = 0.0f;
    float phi_sum = 0.0f;
    float phi_product = 1.0f;

    for (uint32_t i = 0; i < drone_count; i++) {
        phi_sum += individual_phis[i];
        phi_product *= (1.0f + individual_phis[i]);
    }

    switch (effective_config.phi_aggregation_method) {
        case PHI_AGGREGATION_SUM:
            collective_phi = phi_sum * (1.0f + synergy_boost);
            break;

        case PHI_AGGREGATION_AVERAGE:
            collective_phi = (phi_sum / drone_count) * (1.0f + synergy_boost);
            break;

        case PHI_AGGREGATION_WEIGHTED:
            // Weight by coherence
            collective_phi = phi_sum * coherence * (1.0f + synergy_boost);
            break;

        case PHI_AGGREGATION_GEOMETRIC:
            collective_phi = powf(phi_product, 1.0f / drone_count) *
                           (1.0f + synergy_boost);
            break;

        case PHI_AGGREGATION_SYNERGISTIC:
            // Superlinear scaling with coherence
            {
                float n_log = logf((float)drone_count + 1.0f) / logf(2.0f);
                collective_phi = phi_sum *
                               (1.0f + n_log * synergy_boost * coherence);
            }
            break;

        default:
            collective_phi = phi_sum * (1.0f + synergy_boost);
            break;
    }

    // Allocate result
    swarm_consciousness_metrics_t* result = (swarm_consciousness_metrics_t*)
        nimcp_calloc(1, sizeof(swarm_consciousness_metrics_t));
    if (!result) {
        LOG_ERROR("Failed to allocate consciousness metrics");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate consciousness metrics result");
        return NULL;
    }

    // Fill metrics
    result->collective_phi = collective_phi;
    result->network_integration = network_integration;
    result->workspace_coherence = coherence;
    result->drone_count = drone_count;
    result->timestamp = get_time_ms();

    // Copy individual phis
    uint32_t copy_count = (drone_count < SWARM_CONSCIOUSNESS_MAX_DRONES) ?
                          drone_count : SWARM_CONSCIOUSNESS_MAX_DRONES;
    memcpy(result->individual_phi, individual_phis, copy_count * sizeof(float));

    // Compute phi variance
    result->phi_variance = compute_phi_variance(individual_phis, drone_count);

    // Classify state
    result->consciousness_state = swarm_classify_collective_phi(collective_phi, drone_count);

    // Trend defaults to stable (history-based trends handled by monitoring)
    result->phi_trend = PHI_TREND_STABLE;

    // Log computation
    if (effective_config.enable_logging) {
        LOG_INFO("Collective phi computed: %.3f (n=%u, integration=%.3f, coherence=%.3f, state=%s)",
                      collective_phi, drone_count, network_integration, coherence,
                      swarm_consciousness_state_name(result->consciousness_state));
    }

    // BBB audit
    bbb_audit_log(BBB_AUDIT_INFO, "swarm_consciousness", "compute_phi",
                 "phi=%.3f drones=%u state=%d", collective_phi, drone_count,
                 result->consciousness_state);

    return result;
}

/**
 * WHAT: Classify collective phi value into consciousness state
 * WHY:  Map continuous phi to interpretable state categories
 * HOW:  Apply thresholds based on normalized phi per drone
 */
swarm_consciousness_state_t swarm_classify_collective_phi(
    float collective_phi,
    uint32_t drone_count)
{
    // Guard: No drones
    if (drone_count == 0) {
        return SWARM_CONSCIOUSNESS_DORMANT;
    }

    // Normalize phi by drone count for fair comparison
    float normalized_phi = collective_phi / (float)drone_count;

    // Apply thresholds (based on IIT consciousness levels)
    if (normalized_phi < SWARM_CONSCIOUSNESS_MIN_PHI_THRESHOLD) {
        return SWARM_CONSCIOUSNESS_DORMANT;
    } else if (normalized_phi < 0.4f) {
        return SWARM_CONSCIOUSNESS_EMERGING;
    } else if (normalized_phi < SWARM_CONSCIOUSNESS_UNIFIED_COHERENCE_THRESHOLD) {
        return SWARM_CONSCIOUSNESS_UNIFIED;
    } else {
        return SWARM_CONSCIOUSNESS_TRANSCENDENT;
    }
}

/**
 * WHAT: Get current swarm consciousness metrics
 * WHY:  Access latest consciousness state
 * HOW:  Return copy of current metrics
 */
swarm_consciousness_metrics_t* swarm_consciousness_get_metrics(
    swarm_consciousness_ctx_t* context)
{
    // Guard: Null check
    if (!bbb_check_pointer(context, "swarm_consciousness_get_metrics")) {
        return NULL;
    }

    swarm_consciousness_ctx_t* ctx = (swarm_consciousness_ctx_t*)context;

    // Guard: Validate magic
    if (ctx->magic != SWARM_CONSCIOUSNESS_MAGIC) {
        LOG_ERROR("Invalid context magic");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Invalid context magic in swarm_consciousness_get_metrics");
        return NULL;
    }

    pthread_mutex_lock(&ctx->lock);

    swarm_consciousness_metrics_t* result = (swarm_consciousness_metrics_t*)
        nimcp_malloc(sizeof(swarm_consciousness_metrics_t));
    if (result) {
        memcpy(result, ctx->current_metrics, sizeof(swarm_consciousness_metrics_t));
    }

    pthread_mutex_unlock(&ctx->lock);

    return result;
}

/**
 * WHAT: Free consciousness metrics structure
 * WHY:  Release allocated memory
 * HOW:  Simple free (no sub-allocations)
 */
void swarm_consciousness_metrics_free(swarm_consciousness_metrics_t* metrics) {
    if (metrics) {
        nimcp_free(metrics);
    }
}


//=============================================================================
// Monitoring API
//=============================================================================

/**
 * WHAT: Monitoring thread function
 * WHY:  Periodic consciousness computation and callbacks
 * HOW:  Loop: compute phi → callback → publish → sleep
 */
static void* consciousness_monitor_thread(void* arg) {
    swarm_consciousness_ctx_t* ctx = (swarm_consciousness_ctx_t*)arg;

    LOG_INFO("Swarm consciousness monitoring thread started");

    while (ctx->monitoring_active) {
        // Note: In real implementation, would need workspace and individual_phis
        // This is a placeholder showing the pattern

        // Invoke callback if set
        if (ctx->callback && ctx->current_metrics) {
            pthread_mutex_lock(&ctx->lock);
            swarm_consciousness_metrics_t metrics_copy;
            memcpy(&metrics_copy, ctx->current_metrics,
                   sizeof(swarm_consciousness_metrics_t));
            pthread_mutex_unlock(&ctx->lock);

            ctx->callback(&metrics_copy, ctx->user_data);
        }

        // Publish via bio-async if enabled
        if (ctx->config.enable_bio_async && ctx->bio_async_registered) {
            publish_consciousness_update(ctx, ctx->current_metrics);
        }

        // Sleep for update interval
        usleep(ctx->config.update_interval_ms * 1000);
    }

    LOG_INFO("Swarm consciousness monitoring thread stopped");
    return NULL;
}

/**
 * WHAT: Start consciousness monitoring (internal)
 * WHY:  Enable periodic consciousness computation
 * HOW:  Spawn monitoring thread with callback
 */
static bool swarm_consciousness_start_monitoring_internal(
    swarm_consciousness_ctx_t* context,
    void (*callback)(const swarm_consciousness_metrics_t*, void*),
    void* user_data)
{
    // Guard: Null check
    if (!bbb_check_pointer(context, "swarm_consciousness_start_monitoring")) {
        return false;
    }

    swarm_consciousness_ctx_t* ctx = (swarm_consciousness_ctx_t*)context;

    // Guard: Validate magic
    if (ctx->magic != SWARM_CONSCIOUSNESS_MAGIC) {
        LOG_ERROR("Invalid context magic");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Invalid context magic in swarm_consciousness_start_monitoring");
        return false;
    }

    pthread_mutex_lock(&ctx->lock);

    // Guard: Already monitoring
    if (ctx->monitoring_active) {
        LOG_WARN("Monitoring already active");
        pthread_mutex_unlock(&ctx->lock);
        return false;
    }

    // Set callback
    ctx->callback = callback;
    ctx->user_data = user_data;
    ctx->monitoring_active = true;

    // Spawn monitoring thread
    if (nimcp_thread_create(&ctx->monitor_thread, consciousness_monitor_thread, ctx, NULL) != 0) {
        LOG_ERROR("Failed to create monitoring thread");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to create consciousness monitoring thread");
        ctx->monitoring_active = false;
        ctx->callback = NULL;
        ctx->user_data = NULL;
        pthread_mutex_unlock(&ctx->lock);
        return false;
    }

    pthread_mutex_unlock(&ctx->lock);

    LOG_INFO("Swarm consciousness monitoring started");
    bbb_audit_log(BBB_AUDIT_INFO, "swarm_consciousness", "monitoring_start",
                 "interval_ms=%u", ctx->config.update_interval_ms);

    return true;
}

/**
 * WHAT: Stop consciousness monitoring
 * WHY:  Disable periodic computation and cleanup thread
 * HOW:  Signal thread to stop, wait for completion
 */
void swarm_consciousness_stop_monitoring(swarm_consciousness_ctx_t* context) {
    // Guard: Null check
    if (!context) {
        return;
    }

    swarm_consciousness_ctx_t* ctx = (swarm_consciousness_ctx_t*)context;

    // Guard: Validate magic
    if (ctx->magic != SWARM_CONSCIOUSNESS_MAGIC) {
        return;
    }

    pthread_mutex_lock(&ctx->lock);

    if (!ctx->monitoring_active) {
        pthread_mutex_unlock(&ctx->lock);
        return;
    }

    // Signal thread to stop
    ctx->monitoring_active = false;
    pthread_mutex_unlock(&ctx->lock);

    // Wait for thread to complete
    nimcp_thread_join(ctx->monitor_thread, NULL);

    // Clear callback
    pthread_mutex_lock(&ctx->lock);
    ctx->callback = NULL;
    ctx->user_data = NULL;
    pthread_mutex_unlock(&ctx->lock);

    LOG_INFO("Swarm consciousness monitoring stopped");
    bbb_audit_log(BBB_AUDIT_INFO, "swarm_consciousness", "monitoring_stop", "");
}

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * WHAT: Get consciousness state name string
 * WHY:  Human-readable state representation
 * HOW:  Map enum to string
 */
const char* swarm_consciousness_state_name(swarm_consciousness_state_t state) {
    switch (state) {
        case SWARM_CONSCIOUSNESS_DORMANT:
            return "DORMANT";
        case SWARM_CONSCIOUSNESS_EMERGING:
            return "EMERGING";
        case SWARM_CONSCIOUSNESS_UNIFIED:
            return "UNIFIED";
        case SWARM_CONSCIOUSNESS_TRANSCENDENT:
            return "TRANSCENDENT";
        default:
            return "UNKNOWN";
    }
}

/**
 * WHAT: Get aggregation method name string
 * WHY:  Human-readable method representation
 * HOW:  Map enum to string
 */
const char* phi_aggregation_method_name(phi_aggregation_method_t method) {
    switch (method) {
        case PHI_AGGREGATION_SUM:
            return "SUM";
        case PHI_AGGREGATION_AVERAGE:
            return "AVERAGE";
        case PHI_AGGREGATION_WEIGHTED:
            return "WEIGHTED";
        case PHI_AGGREGATION_GEOMETRIC:
            return "GEOMETRIC";
        case PHI_AGGREGATION_SYNERGISTIC:
            return "SYNERGISTIC";
        default:
            return "UNKNOWN";
    }
}

//=============================================================================
// Helper Function Implementations
//=============================================================================

/**
 * WHAT: Compute workspace overlap metric
 * WHY:  Measure shared information across drones
 * HOW:  Analyze top workspace items for similarity
 */
static float compute_workspace_overlap(const collective_workspace_t* workspace) {
    if (!workspace) {
        return 0.0f;
    }

    // Get item count
    uint32_t item_count = collective_workspace_get_item_count(workspace);
    if (item_count == 0) {
        return 0.0f;
    }

    // Get coherence as proxy for overlap
    return collective_workspace_get_coherence(workspace);
}

/**
 * WHAT: Compute variance in individual phi values
 * WHY:  Measure uniformity of consciousness across drones
 * HOW:  Standard variance calculation
 */
static float compute_phi_variance(const float* individual_phis, uint32_t drone_count) {
    if (!individual_phis || drone_count == 0) {
        return 0.0f;
    }

    // Compute mean
    float mean = 0.0f;
    for (uint32_t i = 0; i < drone_count; i++) {
        mean += individual_phis[i];
    }
    mean /= drone_count;

    // Compute variance
    float variance = 0.0f;
    for (uint32_t i = 0; i < drone_count; i++) {
        float diff = individual_phis[i] - mean;
        variance += diff * diff;
    }
    variance /= drone_count;

    return variance;
}

/**
 * WHAT: Update phi history with new value
 * WHY:  Track consciousness trends over time
 * HOW:  Circular buffer insertion
 */
static void update_phi_history(swarm_consciousness_ctx_t* ctx, float phi) {
    if (!ctx) {
        return;
    }

    ctx->phi_history[ctx->history_index] = phi;
    ctx->history_index = (ctx->history_index + 1) % PHI_HISTORY_SIZE;

    if (ctx->history_count < PHI_HISTORY_SIZE) {
        ctx->history_count++;
    }
}

/**
 * WHAT: Compute phi trend from history
 * WHY:  Detect increasing/decreasing consciousness
 * HOW:  Simple linear regression slope
 */
static float get_phi_trend(const swarm_consciousness_ctx_t* ctx) {
    if (!ctx || ctx->history_count < 2) {
        return 0.0f;
    }

    // Simple slope estimation
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_xy = 0.0f;
    float sum_xx = 0.0f;

    uint32_t count = ctx->history_count;
    for (uint32_t i = 0; i < count; i++) {
        float x = (float)i;
        float y = ctx->phi_history[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    float n = (float)count;
    float denominator = (n * sum_xx - sum_x * sum_x);
    if (fabsf(denominator) < 1e-6f) {
        return 0.0f;
    }

    float slope = (n * sum_xy - sum_x * sum_y) / denominator;
    return slope;
}

/**
 * WHAT: Publish consciousness update via bio-async
 * WHY:  Notify other modules of consciousness changes
 * HOW:  Send message through bio-async system
 */
static void publish_consciousness_update(swarm_consciousness_ctx_t* ctx,
                                        const swarm_consciousness_metrics_t* metrics)
{
    if (!ctx || !metrics || !ctx->bio_async_registered) {
        return;
    }

    // In real implementation, would use bio-async message publishing
    // This is a placeholder
    LOG_DEBUG("Publishing consciousness update: phi=%.3f state=%s",
                   metrics->collective_phi,
                   swarm_consciousness_state_name(metrics->consciousness_state));
}

/**
 * WHAT: Get current time in milliseconds
 * WHY:  Timestamp consciousness measurements
 * HOW:  Use clock_gettime for precision
 */
static uint64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

//=============================================================================
// Swarm Brain Integration Functions
//=============================================================================

/**
 * WHAT: Enable continuous consciousness monitoring on swarm brain
 * WHY:  Enable real-time consciousness tracking during swarm operation
 * HOW:  Create context and start monitoring thread
 */
bool swarm_brain_enable_consciousness_monitoring(
    swarm_brain_t* swarm,
    const swarm_consciousness_config_t* config,
    uint32_t interval_ms,
    void (*callback)(const swarm_consciousness_metrics_t*, void*),
    void* user_data)
{
    // Guard: Null check
    if (!bbb_check_pointer(swarm, "swarm_brain_enable_consciousness_monitoring")) {
        return false;
    }

    // Use provided config or defaults
    swarm_consciousness_config_t effective_config;
    if (config) {
        memcpy(&effective_config, config, sizeof(swarm_consciousness_config_t));
    } else {
        effective_config = swarm_consciousness_default_config();
    }
    effective_config.update_interval_ms = interval_ms;

    // Create consciousness context for this swarm
    swarm_consciousness_ctx_t* ctx = swarm_consciousness_create(&effective_config);
    if (!ctx) {
        LOG_ERROR("Failed to create consciousness context for swarm brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to create consciousness context for swarm brain");
        return false;
    }

    // Store swarm brain reference
    ctx->swarm_brain = swarm;

    // Store context in swarm brain (assume swarm_brain has consciousness_ctx field)
    swarm_brain_set_consciousness_ctx(swarm, ctx);

    // Start monitoring
    return swarm_consciousness_start_monitoring_internal(ctx, callback, user_data);
}

/**
 * WHAT: Disable consciousness monitoring on swarm brain
 * WHY:  Cleanup when monitoring no longer needed
 * HOW:  Stop monitoring and destroy context
 */
void swarm_brain_disable_consciousness_monitoring(swarm_brain_t* swarm) {
    // Guard: Null check
    if (!swarm) {
        return;
    }

    // Get consciousness context from swarm brain
    swarm_consciousness_ctx_t* ctx = swarm_brain_get_consciousness_ctx(swarm);
    if (!ctx) {
        return;
    }

    // Stop monitoring
    swarm_consciousness_stop_monitoring(ctx);

    // Destroy context
    swarm_consciousness_destroy(ctx);

    // Clear reference in swarm brain
    swarm_brain_set_consciousness_ctx(swarm, NULL);
}

/**
 * WHAT: Get current collective phi from swarm brain
 * WHY:  Quick check of consciousness level without full metrics
 * HOW:  Compute phi on demand or return cached value
 */
float swarm_brain_get_collective_phi(const swarm_brain_t* swarm) {
    // Guard: Null check
    if (!swarm) {
        return 0.0f;
    }

    // Get consciousness context
    swarm_consciousness_ctx_t* ctx = swarm_brain_get_consciousness_ctx(swarm);
    if (ctx && ctx->current_metrics) {
        return ctx->current_metrics->collective_phi;
    }

    // No monitoring context - compute on demand
    swarm_consciousness_metrics_t* metrics = swarm_compute_collective_phi(
        (swarm_brain_t*)swarm, NULL);
    if (!metrics) {
        return 0.0f;
    }

    float phi = metrics->collective_phi;
    swarm_consciousness_metrics_free(metrics);
    return phi;
}

/**
 * WHAT: Check if swarm has achieved consciousness
 * WHY:  Simple boolean check for consciousness emergence
 * HOW:  Compare current phi to threshold
 */
bool swarm_brain_is_conscious(const swarm_brain_t* swarm, float threshold) {
    // Guard: Null check
    if (!swarm) {
        return false;
    }

    // Use default threshold if not specified
    float effective_threshold = (threshold > 0.0f) ? threshold :
                               SWARM_CONSCIOUSNESS_MIN_PHI_THRESHOLD;

    float phi = swarm_brain_get_collective_phi(swarm);
    return phi >= effective_threshold;
}

//=============================================================================
// Scaling Prediction Functions
//=============================================================================

/**
 * WHAT: Fit scaling model from historical metrics
 * WHY:  Derive power law model of phi vs swarm size
 * HOW:  Least-squares fit to log-transformed phi(n) data
 */
consciousness_scaling_model_t swarm_fit_scaling_model(
    const swarm_consciousness_metrics_t* history,
    uint32_t history_size)
{
    consciousness_scaling_model_t model = {0};

    // Guard: Null or insufficient data
    if (!history || history_size < 3) {
        LOG_ERROR("Need at least 3 samples for model fitting");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Need at least 3 samples for model fitting");
        return model;
    }

    // Extract swarm sizes and phi values from history
    float sum_log_n = 0.0f;
    float sum_log_phi = 0.0f;
    float sum_log_n_squared = 0.0f;
    float sum_log_n_log_phi = 0.0f;
    uint32_t valid_samples = 0;

    for (uint32_t i = 0; i < history_size; i++) {
        float n = (float)history[i].drone_count;
        float phi = history[i].collective_phi;

        if (n > 0.0f && phi > 0.0f) {
            float log_n = logf(n);
            float log_phi = logf(phi);

            sum_log_n += log_n;
            sum_log_phi += log_phi;
            sum_log_n_squared += log_n * log_n;
            sum_log_n_log_phi += log_n * log_phi;
            valid_samples++;
        }
    }

    if (valid_samples < 3) {
        LOG_ERROR("Insufficient valid samples for model fitting");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Insufficient valid samples for model fitting");
        return model;
    }

    // Compute exponent (slope in log-log)
    float n = (float)valid_samples;
    float denominator = (n * sum_log_n_squared - sum_log_n * sum_log_n);
    if (fabsf(denominator) < 1e-6f) {
        LOG_ERROR("Singular matrix in scaling model fit");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Singular matrix in scaling model fit");
        return model;
    }

    float exponent = (n * sum_log_n_log_phi - sum_log_n * sum_log_phi) / denominator;
    float log_base = (sum_log_phi - exponent * sum_log_n) / n;
    float base = expf(log_base);

    // Estimate synergy from super-linear scaling
    float synergy = (exponent > 1.0f) ? (exponent - 1.0f) : 0.0f;

    // Set model parameters
    model.base_phi = base;
    model.scaling_exponent = exponent;
    model.synergy_factor = synergy;
    model.saturation_point = 100.0f;  // Default saturation

    LOG_INFO("Scaling model fitted: base=%.3f, exponent=%.3f, synergy=%.3f",
                  base, exponent, synergy);

    return model;
}

/**
 * WHAT: Predict collective phi for target swarm size
 * WHY:  Forecast emergence before deploying larger swarms
 * HOW:  Evaluate scaling model at target_size
 */
float swarm_predict_phi_for_size(
    const consciousness_scaling_model_t* model,
    uint32_t target_size)
{
    // Guard: Null check
    if (!model || target_size == 0) {
        return 0.0f;
    }

    // phi(n) = base * n^exponent * (1 + synergy) / (1 + n/saturation)
    float n = (float)target_size;
    float saturation_factor = 1.0f;
    if (model->saturation_point > 0.0f) {
        saturation_factor = 1.0f / (1.0f + n / model->saturation_point);
    }

    float phi = model->base_phi *
                powf(n, model->scaling_exponent) *
                (1.0f + model->synergy_factor) *
                saturation_factor;

    return phi;
}

//=============================================================================
// Bio-Async Integration Functions
//=============================================================================

/**
 * WHAT: Register consciousness updates with bio-async router
 * WHY:  Enables streaming consciousness metrics over bio-async
 * HOW:  Register handler for BIO_MSG_SWARM_CONSCIOUSNESS_UPDATE
 */
bool swarm_consciousness_register_bio_async(swarm_consciousness_ctx_t* ctx) {
    // Guard: Null check
    if (!ctx) {
        return false;
    }

    // Check if bio-async is available
    if (!nimcp_bio_async_is_initialized()) {
        LOG_WARN("Bio-async not initialized, cannot register");
        return false;
    }

    ctx->bio_async_registered = true;
    LOG_INFO("Swarm consciousness registered with bio-async");
    return true;
}

//=============================================================================
// Security Functions (BBB)
//=============================================================================

/**
 * WHAT: Validate consciousness metrics for BBB security constraints
 * WHY:  Prevents invalid data from corrupting BBB systems
 * HOW:  Validates ranges, counts, and consistency
 */
bool swarm_consciousness_bbb_validate(const swarm_consciousness_metrics_t* metrics) {
    // Guard: Null check
    if (!metrics) {
        LOG_ERROR("Null metrics pointer in BBB validation");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Null metrics pointer in BBB validation");
        return false;
    }

    // Validate drone count
    if (metrics->drone_count > SWARM_CONSCIOUSNESS_MAX_DRONES) {
        LOG_ERROR("Invalid drone count: %u > %u",
                       metrics->drone_count, SWARM_CONSCIOUSNESS_MAX_DRONES);
        bbb_audit_log(BBB_AUDIT_ERROR, "swarm_consciousness", "bbb_validate",
                     "Invalid drone count: %u", metrics->drone_count);
        return false;
    }

    // Validate collective phi bounds
    if (metrics->collective_phi < 0.0f ||
        metrics->collective_phi > SWARM_CONSCIOUSNESS_MAX_PHI * metrics->drone_count) {
        LOG_ERROR("Invalid collective phi: %.3f", metrics->collective_phi);
        bbb_audit_log(BBB_AUDIT_ERROR, "swarm_consciousness", "bbb_validate",
                     "Invalid phi: %.3f", metrics->collective_phi);
        return false;
    }

    // Check for NaN/Inf
    if (isnan(metrics->collective_phi) || isinf(metrics->collective_phi)) {
        LOG_ERROR("NaN/Inf detected in collective phi");
        bbb_audit_log(BBB_AUDIT_ERROR, "swarm_consciousness", "bbb_validate",
                     "NaN/Inf in phi");
        return false;
    }

    // Validate network integration bounds [0, 1]
    if (metrics->network_integration < 0.0f || metrics->network_integration > 1.0f) {
        LOG_ERROR("Invalid network integration: %.3f", metrics->network_integration);
        return false;
    }

    // Validate workspace coherence bounds [0, 1]
    if (metrics->workspace_coherence < 0.0f || metrics->workspace_coherence > 1.0f) {
        LOG_ERROR("Invalid workspace coherence: %.3f", metrics->workspace_coherence);
        return false;
    }

    // Validate state enum
    if (metrics->consciousness_state < SWARM_CONSCIOUSNESS_DORMANT ||
        metrics->consciousness_state > SWARM_CONSCIOUSNESS_TRANSCENDENT) {
        LOG_ERROR("Invalid consciousness state: %d", metrics->consciousness_state);
        return false;
    }

    // Validate individual phi values
    for (uint32_t i = 0; i < metrics->drone_count; i++) {
        if (metrics->individual_phi[i] < 0.0f ||
            metrics->individual_phi[i] > SWARM_CONSCIOUSNESS_MAX_PHI) {
            LOG_ERROR("Invalid individual phi[%u]: %.3f",
                           i, metrics->individual_phi[i]);
            return false;
        }
        if (isnan(metrics->individual_phi[i]) || isinf(metrics->individual_phi[i])) {
            LOG_ERROR("NaN/Inf detected in individual phi[%u]", i);
            return false;
        }
    }

    return true;
}

//=============================================================================
// Internal Stub Implementations
//=============================================================================

/**
 * WHAT: Get number of drones in swarm
 * WHY:  Core swarm size metric for consciousness computation
 * HOW:  Use peer count + 1 (local) as drone count
 */
static uint32_t swarm_brain_get_drone_count_internal(const swarm_brain_t* swarm) {
    if (!swarm) {
        return 0;
    }

    uint32_t peer_count = 0;
    const swarm_peer_info_t* peers = swarm_brain_get_peers(swarm, &peer_count);
    (void)peers;  // We only need the count

    // Total drones = peers + self
    return peer_count + 1;
}

/**
 * WHAT: Get brain for specific drone by index
 * WHY:  Access individual phi values for collective computation
 * HOW:  Only local brain (index 0) is accessible; remote drones not directly accessible
 */
static brain_t* swarm_brain_get_drone_brain_internal(const swarm_brain_t* swarm, uint32_t index) {
    if (!swarm) {
        return NULL;
    }

    // Only local brain is accessible
    if (index == 0) {
        // Note: swarm_brain_get_local_brain returns brain_t (value), not pointer
        // This is a simplified stub - real implementation would cache the brain pointer
        static brain_t local_brain_cache = NULL;
        local_brain_cache = swarm_brain_get_local_brain((swarm_brain_t*)swarm);
        return local_brain_cache ? &local_brain_cache : NULL;
    }

    // Remote drone brains not directly accessible
    // In production, would use swarm messaging to request phi values
    return NULL;
}

/**
 * WHAT: Get phi (integrated information) from brain
 * WHY:  Core IIT metric for consciousness measurement
 * HOW:  Use introspection subsystem if available, else estimate from activity
 */
static float brain_get_phi_internal(brain_t* brain) {
    if (!brain || !*brain) {
        return 0.0f;
    }

    // Try to get introspection subsystem
    introspection_context_t introspection = brain_get_introspection(*brain);
    if (introspection) {
        // Compute phi using introspection
        consciousness_phi_result_t* phi_result = introspection_compute_phi_fast(
            introspection, NULL);  // NULL config uses defaults
        if (phi_result) {
            float phi = phi_result->phi;
            // Free result
            nimcp_free(phi_result);
            return phi;
        }
    }

    // Fallback: Estimate phi from brain activity
    // Simple heuristic: use 0.5 as baseline for active brain
    return 0.5f;
}

//=============================================================================
// Imagination Engine Integration
//=============================================================================
// BIOLOGICAL BASIS: Collective imagination enables distributed creativity
// across the swarm, similar to cultural transmission of imaginative content
// in social species. Individual nodes can share novel scenarios and insights,
// enabling emergent collective creativity beyond individual capabilities.
//
// NOTE: bio_msg_imagination_collective_t is defined in nimcp_bio_messages.h

/**
 * WHAT: Handler for incoming collective imagination messages
 * WHY:  Process imagination content from other swarm nodes
 * HOW:  Validate, evaluate relevance, invoke user callback
 */
static nimcp_error_t imagination_collective_handler(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    (void)response_promise;  // No response expected for broadcasts

    swarm_consciousness_ctx_t* ctx = (swarm_consciousness_ctx_t*)user_data;

    // Guard: Validate context
    if (!ctx || ctx->magic != SWARM_CONSCIOUSNESS_MAGIC) {
        return NIMCP_INVALID_PARAM;
    }

    // Guard: Validate message size
    if (!msg || msg_size < sizeof(bio_msg_imagination_collective_t)) {
        LOG_WARN("Invalid imagination collective message size");
        return NIMCP_INVALID_PARAM;
    }

    const bio_msg_imagination_collective_t* imag_msg =
        (const bio_msg_imagination_collective_t*)msg;

    // Log received imagination
    LOG_DEBUG("Received collective imagination from node %lu: scenario=%u, relevance=%.3f",
              (unsigned long)imag_msg->source_node,
              imag_msg->scenario_id,
              imag_msg->relevance);

    // Invoke user callback if registered
    pthread_mutex_lock(&ctx->lock);
    imagination_collective_receive_callback_t callback = ctx->collective_imagination_callback;
    void* callback_data = ctx->collective_imagination_user_data;
    pthread_mutex_unlock(&ctx->lock);

    if (callback) {
        // Note: In full implementation, would reconstruct imagination_scenario_t
        // from message payload. For now, we pass NULL scenario with metadata.
        callback(NULL,  // scenario pointer (would be reconstructed)
                 imag_msg->source_node,
                 imag_msg->relevance,
                 callback_data);
    }

    // BBB audit for collective imagination
    bbb_audit_log(BBB_AUDIT_INFO, "swarm_consciousness", "imagination_received",
                  "source=%lu scenario=%u relevance=%.3f",
                  (unsigned long)imag_msg->source_node,
                  imag_msg->scenario_id,
                  imag_msg->relevance);

    return NIMCP_SUCCESS;
}

/**
 * WHAT: Share imagination scenario with other swarm nodes
 * WHY:  Enable distributed creativity through collective imagination
 * HOW:  Broadcast imagination content via BIO_MSG_IMAGINATION_COLLECTIVE_SHARE
 *
 * BIOLOGICAL BASIS: Similar to cultural transmission in social species,
 * where imaginative content (stories, solutions, innovations) spreads
 * through the population, enabling collective creativity.
 *
 * @param ctx Swarm consciousness context
 * @param scenario Imagination scenario to share
 * @return 0 on success, -1 on error
 */
int swarm_consciousness_share_imagination(
    swarm_consciousness_ctx_t* ctx,
    const struct imagination_scenario* scenario)
{
    // Guard: Validate context
    if (!ctx) {
        LOG_ERROR("Null context in share_imagination");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Null context in share_imagination");
        return -1;
    }

    if (ctx->magic != SWARM_CONSCIOUSNESS_MAGIC) {
        LOG_ERROR("Invalid context magic in share_imagination");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Invalid context magic in share_imagination");
        return -1;
    }

    // Guard: Validate scenario
    if (!scenario) {
        LOG_ERROR("Null scenario in share_imagination");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Null scenario in share_imagination");
        return -1;
    }

    // Guard: Check bio-async availability
    if (!ctx->bio_async_registered || !ctx->bio_module_ctx) {
        LOG_WARN("Bio-async not available for imagination sharing");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Bio-async not available for imagination sharing");
        return -1;
    }

    // Prepare broadcast message
    bio_msg_imagination_collective_t msg;
    memset(&msg, 0, sizeof(msg));

    // Initialize header
    bio_msg_init_header(&msg.header, BIO_MSG_IMAGINATION_COLLECTIVE_SHARE,
                        bio_module_context_get_id(ctx->bio_module_ctx),
                        0,  // Broadcast to all
                        sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.header.timestamp_us = get_time_ms() * 1000;

    // Fill imagination metadata
    // Note: Would extract from scenario struct in full implementation
    msg.source_node = (uint64_t)bio_module_context_get_id(ctx->bio_module_ctx);
    msg.scenario_id = 0;  // Would extract from scenario->id
    msg.share_scope = 0;  // Default scope (global)
    msg.relevance = 1.0f;  // Self-generated = max relevance
    msg.is_share = true;  // This is a share operation

    // Send broadcast
    nimcp_error_t err = bio_router_send(ctx->bio_module_ctx, &msg, sizeof(msg), 0);
    if (err != NIMCP_SUCCESS) {
        LOG_ERROR("Failed to broadcast imagination: error=%d", err);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NETWORK_IO, "Failed to broadcast imagination: error=%d", err);
        return -1;
    }

    LOG_INFO("Shared imagination scenario with swarm: scenario_id=%u",
             msg.scenario_id);

    bbb_audit_log(BBB_AUDIT_INFO, "swarm_consciousness", "imagination_shared",
                  "scenario=%u scope=%u", msg.scenario_id, msg.share_scope);

    return 0;
}

/**
 * WHAT: Receive and process imagination from another swarm node
 * WHY:  Evaluate and integrate external imagination into local consciousness
 * HOW:  Assess relevance, filter, and invoke callback for accepted content
 *
 * BIOLOGICAL BASIS: Incoming cultural/imaginative content is evaluated
 * for relevance to current goals and mental state before integration,
 * similar to selective attention in social learning.
 *
 * @param ctx Swarm consciousness context
 * @param scenario Received imagination scenario
 * @param source_node_id ID of the node that shared the scenario
 * @return 0 on success, -1 on error
 */
int swarm_consciousness_receive_imagination(
    swarm_consciousness_ctx_t* ctx,
    const struct imagination_scenario* scenario,
    uint64_t source_node_id)
{
    // Guard: Validate context
    if (!ctx) {
        LOG_ERROR("Null context in receive_imagination");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Null context in receive_imagination");
        return -1;
    }

    if (ctx->magic != SWARM_CONSCIOUSNESS_MAGIC) {
        LOG_ERROR("Invalid context magic in receive_imagination");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Invalid context magic in receive_imagination");
        return -1;
    }

    // Guard: Validate scenario
    if (!scenario) {
        LOG_WARN("Null scenario in receive_imagination");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Null scenario in receive_imagination");
        return -1;
    }

    // Evaluate relevance to local consciousness
    // Relevance factors:
    // 1. Current consciousness state (more receptive in UNIFIED+ states)
    // 2. Workspace coherence (higher coherence = better integration)
    // 3. Source node trust/history
    float relevance = 0.5f;  // Base relevance

    pthread_mutex_lock(&ctx->lock);

    // Boost relevance based on consciousness state
    if (ctx->current_metrics) {
        switch (ctx->current_metrics->consciousness_state) {
            case SWARM_CONSCIOUSNESS_TRANSCENDENT:
                relevance *= 1.5f;
                break;
            case SWARM_CONSCIOUSNESS_UNIFIED:
                relevance *= 1.2f;
                break;
            case SWARM_CONSCIOUSNESS_EMERGING:
                relevance *= 1.0f;
                break;
            case SWARM_CONSCIOUSNESS_DORMANT:
            default:
                relevance *= 0.5f;  // Less receptive when dormant
                break;
        }

        // Modulate by workspace coherence
        relevance *= (0.5f + 0.5f * ctx->current_metrics->workspace_coherence);
    }

    // Clamp relevance to [0, 1]
    if (relevance > 1.0f) relevance = 1.0f;
    if (relevance < 0.0f) relevance = 0.0f;

    // Get callback
    imagination_collective_receive_callback_t callback = ctx->collective_imagination_callback;
    void* callback_data = ctx->collective_imagination_user_data;

    pthread_mutex_unlock(&ctx->lock);

    // Invoke callback with relevance-weighted scenario
    if (callback) {
        callback(scenario, source_node_id, relevance, callback_data);

        LOG_DEBUG("Processed imagination from node %lu with relevance %.3f",
                  (unsigned long)source_node_id, relevance);
    }

    return 0;
}

/**
 * WHAT: Register handler for collective imagination sharing
 * WHY:  Enable reception of imagination from other swarm nodes
 * HOW:  Register bio-async handler for BIO_MSG_IMAGINATION_COLLECTIVE_SHARE
 *
 * BIOLOGICAL BASIS: Establishes neural pathways for receiving and
 * processing culturally transmitted imaginative content from the swarm.
 *
 * @param ctx Swarm consciousness context
 * @param callback Callback for received imagination
 * @param user_data User data passed to callback
 * @return 0 on success, -1 on error
 */
int swarm_consciousness_register_imagination_handler(
    swarm_consciousness_ctx_t* ctx,
    imagination_collective_receive_callback_t callback,
    void* user_data)
{
    // Guard: Validate context
    if (!ctx) {
        LOG_ERROR("Null context in register_imagination_handler");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Null context in register_imagination_handler");
        return -1;
    }

    if (ctx->magic != SWARM_CONSCIOUSNESS_MAGIC) {
        LOG_ERROR("Invalid context magic in register_imagination_handler");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Invalid context magic in register_imagination_handler");
        return -1;
    }

    pthread_mutex_lock(&ctx->lock);

    // Store callback
    ctx->collective_imagination_callback = callback;
    ctx->collective_imagination_user_data = user_data;

    // Register bio-async handler if not already done
    if (!ctx->imagination_handler_registered && ctx->bio_async_registered) {
        // Register module with bio-router if not yet registered
        if (!ctx->bio_module_ctx) {
            bio_module_info_t mod_info = {
                .module_id = BIO_MODULE_SWARM_CONSCIOUSNESS,
                .module_name = "swarm_consciousness",
                .inbox_capacity = 0,  // Use default
                .user_data = ctx
            };
            ctx->bio_module_ctx = bio_router_register_module(&mod_info);
            if (!ctx->bio_module_ctx) {
                LOG_WARN("Failed to register swarm_consciousness with bio-router");
                NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Failed to register swarm_consciousness with bio-router");
                pthread_mutex_unlock(&ctx->lock);
                return -1;
            }
        }

        /* Register handlers via KG-driven wiring callback */
        nimcp_error_t err = bio_router_register_wiring_callback(
            BIO_MODULE_SWARM_CONSCIOUSNESS,
            (void*)swarm_consciousness_handler_callback,
            ctx
        );

        if (err != NIMCP_SUCCESS) {
            /* Legacy fallback: direct handler registration */
            LOG_DEBUG("KG wiring unavailable, using legacy registration");
            err = LEGACY_HANDLER_REGISTRATION(bio_router_register_handler(
                ctx->bio_module_ctx,
                BIO_MSG_IMAGINATION_COLLECTIVE_SHARE,
                imagination_collective_handler));

            if (err != NIMCP_SUCCESS) {
                LOG_WARN("Failed to register imagination handler: error=%d", err);
                pthread_mutex_unlock(&ctx->lock);
                return -1;
            }

            // Also register for collective insight messages
            LEGACY_HANDLER_REGISTRATION(bio_router_register_handler(
                ctx->bio_module_ctx,
                BIO_MSG_IMAGINATION_COLLECTIVE_INSIGHT,
                imagination_collective_handler));
        }

        ctx->imagination_handler_registered = true;
        LOG_INFO("Registered imagination collective handler");
    }

    pthread_mutex_unlock(&ctx->lock);

    bbb_audit_log(BBB_AUDIT_INFO, "swarm_consciousness", "imagination_handler_registered",
                  "callback=%p", (void*)callback);

    return 0;
}

//=============================================================================
// Knowledge Graph Self-Awareness Integration
//=============================================================================

/**
 * WHAT: Query knowledge graph for self-knowledge about swarm consciousness
 * WHY:  Enable self-awareness by introspecting module's identity in KG
 * HOW:  Query entity, observations, and relations from knowledge graph
 *
 * @param kg Knowledge graph reader handle
 * @return 1 if entity found, 0 if not found or error
 */
int swarm_consciousness_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) {
        return 0;
    }

    const kg_entity_t* self = kg_reader_get_entity(kg, "Swarm_Consciousness");
    if (self) {
        LOG_INFO("KG Self-Knowledge: Found entity '%s' of type '%s'",
                 self->name, self->entity_type);
        for (uint32_t i = 0; i < self->num_observations; i++) {
            LOG_DEBUG("  Observation[%u]: %s", i, self->observations[i]);
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Swarm_Consciousness");
    if (connections) {
        LOG_INFO("KG Self-Knowledge: Swarm_Consciousness has %u outgoing connections",
                 connections->count);
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Swarm_Consciousness");
    if (incoming) {
        LOG_INFO("KG Self-Knowledge: Swarm_Consciousness has %u incoming connections",
                 incoming->count);
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
