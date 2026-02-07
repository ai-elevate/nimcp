/**
 * @file nimcp_systems_consolidation.c
 * @brief Phase M2: Systems consolidation implementation
 *
 * WHAT: Implements hippocampus → cortex memory transfer during sleep
 * WHY:  Models complementary learning systems (McClelland et al., 1995)
 * HOW:  Sleep replay drives cortical plasticity and semantic abstraction
 *
 * BIO-ASYNC INTEGRATION:
 * - Module ID: 0x0332 (BIO_MODULE_SYSTEMS_CONSOLIDATION)
 * - Publishes: replay events, consolidation progress
 * - Subscribes: sleep state changes, engram updates
 *
 * @version Phase M2
 * @date 2025-11-13
 */

#define LOG_MODULE "systems_consolidation"

#include "cognitive/memory/nimcp_systems_consolidation.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "cognitive/memory/nimcp_engram.h"
#include <stddef.h>  /* for NULL */
#include "nimcp.h"  // For NIMCP_ERROR_* codes
#include "utils/exception/nimcp_exception_macros.h"  // Phase 7: Exception integration
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_wiring_helpers.h"
#include "utils/platform/nimcp_platform_time.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/memory/nimcp_memory_pool.h"
#include "utils/memory/nimcp_memory_guards.h"  // For nimcp_calloc/nimcp_free
#include "utils/logging/nimcp_logging.h"
#include "utils/containers/nimcp_vector.h"
#include <string.h>
#include <math.h>

/*=============================================================================
 * Health Agent Forward Declarations (Phase 8: Heartbeat for Long Operations)
 *===========================================================================*/
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

// Use unique prefix to avoid conflict with cognitive/consolidation/nimcp_consolidation.c
NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(systems_consolidation)

//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_systems_consolidation_mesh_id = 0;
static mesh_participant_registry_t* g_systems_consolidation_mesh_registry = NULL;

static nimcp_error_t systems_consolidation_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_systems_consolidation_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "systems_consolidation", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_MEMORY);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "systems_consolidation";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_systems_consolidation_mesh_id);
    if (err == NIMCP_SUCCESS) g_systems_consolidation_mesh_registry = registry;
    return err;
}

static void systems_consolidation_mesh_unregister(void) {
    if (g_systems_consolidation_mesh_registry && g_systems_consolidation_mesh_id != 0) {
        mesh_participant_unregister(g_systems_consolidation_mesh_registry, g_systems_consolidation_mesh_id);
        g_systems_consolidation_mesh_id = 0;
        g_systems_consolidation_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat from systems_consolidation module (instance-level) */
static inline void systems_consolidation_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_systems_consolidation_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_systems_consolidation_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_systems_consolidation_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}


//=============================================================================
// BIO-ASYNC MODULE REGISTRATION
//=============================================================================

#define BIO_MODULE_SYSTEMS_CONSOLIDATION 0x0332

//=============================================================================
// BIO-ASYNC HANDLERS (Forward declarations)
//=============================================================================

static nimcp_error_t handle_consolidation_trigger(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data);

static void bio_broadcast_consolidation_complete(systems_consolidation_system_t* system, uint32_t engram_id, float strength);

/*=============================================================================
 * KG-Driven Wiring Callback
 *============================================================================*/

/**
 * @brief KG-driven wiring handler callback
 *
 * WHAT: Register message handlers based on discovered wiring from KG
 * WHY:  Enables runtime assembly - module discovers its handlers from KG
 * HOW:  Orchestrator invokes this with message types from HANDLES_MESSAGE relations
 */
static int systems_consolidation_wiring_handler_callback(
    bio_module_context_t ctx,
    const bio_message_type_t* message_types,
    uint32_t message_count,
    void* user_data
) {
    if (!ctx || !message_types || message_count == 0) {
        return 0;  /* No handlers to register */
    }

    LOG_INFO(LOG_MODULE, "systems_consolidation_wiring_handler_callback: registering %u handlers from KG",
             message_count);

    for (uint32_t i = 0; i < message_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && message_count > 256) {
            systems_consolidation_heartbeat("consolidatio_loop",
                             (float)(i + 1) / (float)message_count);
        }

        switch (message_types[i]) {
            case BIO_MSG_CONSOLIDATION_TRIGGER:
                bio_router_register_handler(ctx, message_types[i], handle_consolidation_trigger);
                LOG_DEBUG(LOG_MODULE, "  Registered handler for BIO_MSG_CONSOLIDATION_TRIGGER");
                break;

            default:
                LOG_DEBUG(LOG_MODULE, "  Unknown message type %u - skipping", message_types[i]);
                break;
        }
    }

    return 0;
}

//=============================================================================
// Internal Helper Functions
//=============================================================================

/* Cosine similarity uses nimcp_vector_cosine_similarity from utils/containers/nimcp_vector.h */

/**
 * @brief Generate unique ID for cortical memory node
 *
 * WHAT: Creates unique identifier for cortical nodes
 * WHY:  Needed to track and reference nodes
 * HOW:  Uses timestamp + counter for uniqueness
 *
 * @return Unique 64-bit ID
 */
static uint64_t generate_node_id(void)
{
    static uint64_t counter = 0;
    uint64_t timestamp = nimcp_platform_time_monotonic_ms();
    return (timestamp << 16) | (counter++ & 0xFFFF);
}

//=============================================================================
// Cortical Node Management
//=============================================================================

/**
 * @brief Create a new cortical memory node
 *
 * WHAT: Allocates and initializes a cortical memory node
 * WHY:  Represents abstracted memory in cortex
 * HOW:  Allocates structure and feature vector
 *
 * @param features Semantic feature vector (will be copied)
 * @param feature_dim Dimensionality of features
 * @param source_engram_id Original hippocampal engram
 * @return Pointer to new node, or NULL on failure
 */
static cortical_memory_node_t* cortical_node_create(
    const float* features,
    uint32_t feature_dim,
    uint64_t source_engram_id)
{
    // WHAT: Guard against invalid input
    if (!features || feature_dim == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "cortical_node_create: features is NULL");
        return NULL;
    }

    // WHAT: Allocate node structure
    cortical_memory_node_t* node = nimcp_calloc(1, sizeof(cortical_memory_node_t));
    if (!node) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate node");

        return NULL;
    }

    // WHAT: Allocate and copy feature vector
    node->features = nimcp_malloc(feature_dim * sizeof(float));
    if (!node->features) {
        nimcp_free(node);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "cortical_node_create: node->features is NULL");
        return NULL;
    }
    memcpy(node->features, features, feature_dim * sizeof(float));

    // WHAT: Allocate neighbor arrays
    node->neighbor_capacity = CONSOLIDATION_DEFAULT_NEIGHBORS_PER_NODE;
    node->neighbors = nimcp_calloc(node->neighbor_capacity, sizeof(cortical_memory_node_t*));
    node->neighbor_strengths = nimcp_calloc(node->neighbor_capacity, sizeof(float));

    if (!node->neighbors || !node->neighbor_strengths) {
        nimcp_free(node->features);
        nimcp_free(node->neighbors);
        nimcp_free(node->neighbor_strengths);
        nimcp_free(node);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "cortical_node_create: required parameter is NULL (node->neighbors, node->neighbor_strengths)");
        return NULL;
    }

    // WHAT: Initialize node state
    node->id = generate_node_id();
    node->feature_dim = feature_dim;
    node->type = CORTICAL_MEMORY_EPISODIC;  // Starts episodic
    node->consolidation_strength = 0.0F;    // Starts weak
    node->hippocampal_dependency = 1.0F;    // Fully dependent on hippocampus initially
    node->creation_time_ms = nimcp_platform_time_monotonic_ms();
    node->last_activation_ms = node->creation_time_ms;
    node->source_engram_id = source_engram_id;
    node->is_transferred = false;
    node->neighbor_count = 0;

    // Note: Bio-async is now handled at the system level, not per-node

    return node;
}

/*=============================================================================
 * BIO-ASYNC HANDLER IMPLEMENTATIONS
 *============================================================================*/

static nimcp_error_t handle_consolidation_trigger(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    (void)msg_size;
    (void)response_promise;
    NIMCP_CHECK_THROW(msg && user_data, NIMCP_ERROR_NULL_ARG, "msg or user_data is NULL");
    const bio_msg_salience_response_t* trigger = (const bio_msg_salience_response_t*)msg;
    systems_consolidation_system_t* system = (systems_consolidation_system_t*)user_data;
    LOG_DEBUG(LOG_MODULE, "Received consolidation trigger: stimulus_id=%u, strength=%.2f, node_count=%u",
              trigger->stimulus_id, trigger->salience_score, system->node_count);

    // Schedule replay for the triggered engram
    systems_consolidation_schedule_replay(system, trigger->stimulus_id, trigger->salience_score);

    return NIMCP_SUCCESS;
}

static void bio_broadcast_consolidation_complete(systems_consolidation_system_t* system, uint32_t engram_id, float strength) {
    if (!system || !system->bio_async_enabled || !system->bio_ctx) { return; }
    // Use salience response for consolidation complete notification
    bio_msg_salience_response_t msg = {0};
    bio_msg_init_header(&msg.header, BIO_MSG_SALIENCE_RESPONSE,
                        bio_module_context_get_id(system->bio_ctx), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.stimulus_id = engram_id;
    msg.salience_score = strength;
    msg.attention_priority = strength;
    msg.requires_immediate_attention = false;
    bio_router_broadcast(system->bio_ctx, &msg, sizeof(msg));
    LOG_DEBUG(LOG_MODULE, "Broadcast consolidation complete: engram=%u, strength=%.2f", engram_id, strength);
}

/**
 * @brief Destroy cortical memory node and free resources
 *
 * WHAT: Frees all memory associated with cortical node
 * WHY:  Prevents memory leaks
 * HOW:  Frees arrays and structure in correct order
 *
 * @param node Node to destroy (can be NULL)
 */
static void cortical_node_destroy(cortical_memory_node_t* node)
{
    // WHAT: Guard against NULL
    if (!node) {
        return;
    }

    // WHAT: Free allocated arrays
    if (node->features) {
        nimcp_free(node->features);
    }
    if (node->neighbors) {
        nimcp_free(node->neighbors);
    }
    if (node->neighbor_strengths) {
        nimcp_free(node->neighbor_strengths);
    }

    // Note: Bio-async is now handled at the system level, not per-node

    // WHAT: Free node structure
    nimcp_free(node);
}

/**
 * @brief Add a neighbor link to cortical node
 *
 * WHAT: Creates semantic similarity connection between nodes
 * WHY:  Models lateral cortical connections (concepts cluster)
 * HOW:  Adds neighbor pointer and connection strength
 *
 * @param node Node to add neighbor to
 * @param neighbor Neighboring node
 * @param strength Connection strength (0.0-1.0)
 * @return true if added, false if capacity reached
 */
static bool cortical_node_add_neighbor(
    cortical_memory_node_t* node,
    cortical_memory_node_t* neighbor,
    float strength)
{
    // WHAT: Guard against invalid input
    if (!node || !neighbor) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "cortical_node_add_neighbor: required parameter is NULL (node, neighbor)");
        return false;
    }

    // WHAT: Check capacity
    if (node->neighbor_count >= node->neighbor_capacity) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_BUFFER_OVERFLOW, "cortical_node_add_neighbor: capacity exceeded");
        return false;  // Max neighbors reached
    }

    // WHAT: Add neighbor and strength
    node->neighbors[node->neighbor_count] = neighbor;
    node->neighbor_strengths[node->neighbor_count] = strength;
    node->neighbor_count++;

    return true;
}

//=============================================================================
// System Management API
//=============================================================================

systems_consolidation_system_t* systems_consolidation_create(void)
{
    /* Phase 8: Heartbeat at operation start */
    systems_consolidation_heartbeat("consolidatio_systems_consolidatio", 0.0f);


    LOG_INFO("Creating systems consolidation system");

    // WHAT: Allocate main system structure
    systems_consolidation_system_t* system =
        nimcp_calloc(1, sizeof(systems_consolidation_system_t));
    if (!system) {
        LOG_ERROR("Failed to allocate systems consolidation system (%zu bytes)", sizeof(systems_consolidation_system_t));
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "systems_consolidation_create: system is NULL");
        return NULL;
    }

    // WHAT: Allocate cortical node storage
    system->node_capacity = CONSOLIDATION_DEFAULT_CORTICAL_CAPACITY;
    system->cortical_nodes =
        nimcp_calloc(system->node_capacity, sizeof(cortical_memory_node_t*));
    if (!system->cortical_nodes) {
        nimcp_free(system);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "systems_consolidation_create: system->cortical_nodes is NULL");
        return NULL;
    }

    // WHAT: Allocate replay queue
    system->replay_queue_capacity = CONSOLIDATION_DEFAULT_REPLAY_QUEUE_SIZE;
    system->replay_queue =
        nimcp_calloc(system->replay_queue_capacity, sizeof(replay_event_t));
    if (!system->replay_queue) {
        nimcp_free(system->cortical_nodes);
        nimcp_free(system);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "systems_consolidation_create: system->replay_queue is NULL");
        return NULL;
    }

    // WHAT: Initialize parameters with biological defaults
    system->node_count = 0;
    system->replay_queue_size = 0;
    system->replay_frequency_hz = CONSOLIDATION_REPLAY_FREQUENCY_SWS;
    system->last_replay_time_ms = nimcp_platform_time_monotonic_ms();
    system->transfer_rate = CONSOLIDATION_TRANSFER_RATE_SWS;
    system->forgetting_rate = CONSOLIDATION_FORGETTING_RATE;
    system->semantic_threshold = CONSOLIDATION_SEMANTIC_THRESHOLD;
    system->engram_system = NULL;  // Set via integration API
    system->sleep_system = NULL;   // Set via integration API
    system->total_replays = 0;
    system->total_transfers = 0;
    system->total_forgotten = 0;

    // Phase 1.5: Initialize memory pools for hot-path allocations
    memory_pool_config_t node_pool_config = {
        .block_size = sizeof(cortical_memory_node_t),
        .num_blocks = 64,  // Pre-allocate typical usage
        .alignment = 16,   // SIMD alignment
        .enable_tracking = false,
        .enable_guard_pages = false
    };
    system->node_pool = memory_pool_create(&node_pool_config);

    memory_pool_config_t feature_pool_config = {
        .block_size = 32 * sizeof(float),  // Default feature dimension
        .num_blocks = 64,
        .alignment = 16,
        .enable_tracking = false,
        .enable_guard_pages = false
    };
    system->feature_pool = memory_pool_create(&feature_pool_config);

    memory_pool_config_t neighbor_pool_config = {
        .block_size = CONSOLIDATION_DEFAULT_NEIGHBORS_PER_NODE * sizeof(void*),
        .num_blocks = 64,
        .alignment = 16,
        .enable_tracking = false,
        .enable_guard_pages = false
    };
    system->neighbor_pool = memory_pool_create(&neighbor_pool_config);

    // Bio-async registration at system level
    system->bio_ctx = NULL;
    system->bio_async_enabled = false;
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_SYSTEMS_CONSOLIDATION,
            .module_name = "systems_consolidation",
            .inbox_capacity = 64,
            .user_data = system
        };
        system->bio_ctx = bio_router_register_module(&bio_info);
        if (system->bio_ctx) {
            system->bio_async_enabled = true;

            // Try KG-driven wiring callback first
            nimcp_error_t wiring_result = bio_router_register_wiring_callback(
                BIO_MODULE_SYSTEMS_CONSOLIDATION,
                (void*)systems_consolidation_wiring_handler_callback,
                system
            );

            if (wiring_result != NIMCP_SUCCESS) {
                // Fallback to legacy hardcoded registration
                LEGACY_HANDLER_REGISTRATION(
                    bio_router_register_handler(system->bio_ctx, BIO_MSG_CONSOLIDATION_TRIGGER,
                                                handle_consolidation_trigger)
                );
                LOG_INFO("Bio-async registered with legacy handlers (module_id=0x%04X)",
                         BIO_MODULE_SYSTEMS_CONSOLIDATION);
            } else {
                LOG_INFO("Bio-async registered with KG wiring callback (module_id=0x%04X)",
                         BIO_MODULE_SYSTEMS_CONSOLIDATION);
            }
        } else {
            LOG_WARN("Failed to register with bio_router (async disabled)");
        }
    }

    return system;
}

void systems_consolidation_destroy(systems_consolidation_system_t* system)
{
    // WHAT: Guard against NULL
    if (!system) {
        return;
    }

    // Unregister from bio-router
    /* Phase 8: Heartbeat at operation start */
    systems_consolidation_heartbeat("consolidatio_systems_consolidatio", 0.0f);


    if (system->bio_async_enabled && system->bio_ctx) {
        bio_router_unregister_module(system->bio_ctx);
        system->bio_ctx = NULL;
        system->bio_async_enabled = false;
        LOG_DEBUG("Unregistered from bio_router");
    }

    // WHAT: Free all cortical nodes
    if (system->cortical_nodes) {
        for (uint32_t i = 0; i < system->node_count; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && system->node_count > 256) {
                systems_consolidation_heartbeat("consolidatio_loop",
                                 (float)(i + 1) / (float)system->node_count);
            }

            cortical_node_destroy(system->cortical_nodes[i]);
        }
        nimcp_free(system->cortical_nodes);
    }

    // WHAT: Free replay queue
    if (system->replay_queue) {
        nimcp_free(system->replay_queue);
    }

    // Phase 1.5: Destroy memory pools
    if (system->node_pool) {
        memory_pool_destroy(system->node_pool);
    }
    if (system->feature_pool) {
        memory_pool_destroy(system->feature_pool);
    }
    if (system->neighbor_pool) {
        memory_pool_destroy(system->neighbor_pool);
    }

    // WHAT: Free system structure
    nimcp_free(system);
}

void systems_consolidation_reset(systems_consolidation_system_t* system)
{
    // WHAT: Guard against NULL
    if (!system) {
        return;
    }

    // WHAT: Free all cortical nodes
    /* Phase 8: Heartbeat at operation start */
    systems_consolidation_heartbeat("consolidatio_systems_consolidatio", 0.0f);


    for (uint32_t i = 0; i < system->node_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->node_count > 256) {
            systems_consolidation_heartbeat("consolidatio_loop",
                             (float)(i + 1) / (float)system->node_count);
        }

        cortical_node_destroy(system->cortical_nodes[i]);
        system->cortical_nodes[i] = NULL;
    }

    // WHAT: Reset counters (keep allocated capacity)
    system->node_count = 0;
    system->replay_queue_size = 0;
    system->total_replays = 0;
    system->total_transfers = 0;
    system->total_forgotten = 0;
    system->last_replay_time_ms = nimcp_platform_time_monotonic_ms();
}

//=============================================================================
// Sleep Replay API
//=============================================================================

bool systems_consolidation_schedule_replay(
    systems_consolidation_system_t* system,
    uint64_t engram_id,
    float priority)
{
    // WHAT: Guard against invalid input
    if (!system || engram_id == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "systems_consolidation_schedule_replay: system is NULL");
        return false;
    }

    // WHAT: Check queue capacity
    /* Phase 8: Heartbeat at operation start */
    systems_consolidation_heartbeat("consolidatio_systems_consolidatio", 0.0f);


    if (system->replay_queue_size >= system->replay_queue_capacity) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_BUFFER_OVERFLOW, "systems_consolidation_schedule_replay: capacity exceeded");
        return false;  // Queue full
    }

    // WHAT: Create replay event
    replay_event_t event = {
        .engram_id = engram_id,
        .cortical_node_id = 0,  // Will be determined during replay
        .priority = priority,
        .emotional_salience = priority,  // Simplified: use priority as salience
        .scheduled_time_ms = nimcp_platform_time_monotonic_ms(),
        .is_completed = false
    };

    // WHAT: Add to queue
    system->replay_queue[system->replay_queue_size] = event;
    system->replay_queue_size++;

    return true;
}

uint32_t systems_consolidation_execute_replays(
    systems_consolidation_system_t* system,
    float time_delta_seconds,
    bool is_sws,
    bool is_rem)
{
    // WHAT: Guard against invalid input
    if (!system || time_delta_seconds <= 0.0F) {
        return 0;
    }

    /* Phase 8: Send heartbeat at start of replay execution */
    systems_consolidation_heartbeat("consolidation_replays", 0.0f);

    // WHAT: Determine replay rate based on sleep state
    // WHY: SWS has highest replay rate (Born & Wilhelm, 2012)
    float replay_rate_hz;
    if (is_sws) {
        replay_rate_hz = CONSOLIDATION_REPLAY_FREQUENCY_SWS;  // 10 Hz in SWS
    } else if (is_rem) {
        replay_rate_hz = CONSOLIDATION_REPLAY_FREQUENCY_SWS * 0.5F;  // 5 Hz in REM
    } else {
        replay_rate_hz = 0.1F;  // Minimal awake replay
    }

    // WHAT: Calculate number of replays to execute this cycle
    // WHY: Replay frequency determines consolidation speed
    float replays_to_execute_float = replay_rate_hz * time_delta_seconds;
    uint32_t replays_to_execute = (uint32_t)replays_to_execute_float;

    // WHAT: Cap by queue size
    if (replays_to_execute > system->replay_queue_size) {
        replays_to_execute = system->replay_queue_size;
    }

    // WHAT: Execute replays
    uint32_t executed_count = 0;
    for (uint32_t i = 0; i < replays_to_execute && i < system->replay_queue_size; i++) {
        /* Phase 8: Send progress heartbeat for long replay sequences */
        if (replays_to_execute > 0) {
            systems_consolidation_heartbeat("consolidation_replays", (float)i / (float)replays_to_execute);
        }
        replay_event_t* event = &system->replay_queue[i];

        // WHAT: Transfer engram to cortex
        // WHY: Replay drives cortical plasticity
        float replay_strength = is_sws ? 0.8F : (is_rem ? 0.5F : 0.1F);
        uint64_t cortical_node_id = systems_consolidation_transfer_to_cortex(
            system,
            event->engram_id,
            replay_strength
        );

        if (cortical_node_id != 0) {
            event->is_completed = true;
            executed_count++;
            system->total_replays++;
        }
    }

    // WHAT: Remove completed events from queue
    // WHY: Keep queue compact
    uint32_t write_idx = 0;
    for (uint32_t read_idx = 0; read_idx < system->replay_queue_size; read_idx++) {
        /* Phase 8: Loop progress heartbeat */
        if ((read_idx & 0xFF) == 0 && system->replay_queue_size > 256) {
            systems_consolidation_heartbeat("consolidatio_loop",
                             (float)(read_idx + 1) / (float)system->replay_queue_size);
        }

        if (!system->replay_queue[read_idx].is_completed) {
            system->replay_queue[write_idx] = system->replay_queue[read_idx];
            write_idx++;
        }
    }
    system->replay_queue_size = write_idx;

    system->last_replay_time_ms = nimcp_platform_time_monotonic_ms();
    return executed_count;
}

//=============================================================================
// Cortical Transfer API
//=============================================================================

uint64_t systems_consolidation_transfer_to_cortex(
    systems_consolidation_system_t* system,
    uint64_t engram_id,
    float replay_strength)
{
    // WHAT: Guard against invalid input
    if (!system || engram_id == 0 || replay_strength <= 0.0F) {
        return 0;
    }

    // WHAT: Guard against missing engram system (currently optional for testing)
    // WHY: During testing, we may not have a full engram system
    // TODO: Make required once integration phase is complete
    /* Phase 8: Heartbeat at operation start */
    systems_consolidation_heartbeat("consolidatio_systems_consolidatio", 0.0f);


    bool have_engram_system = (system->engram_system != NULL);

    // WHAT: Extract semantic features (simplified implementation)
    // WHY: Cortex stores abstracted/semantic representations
    // HOW: Create semantic feature vector from engram ID
    // NOTE: In full integration, would query engram_system for actual neurons/activations
    const uint32_t SEMANTIC_DIM = 32;  // Semantic feature dimensionality
    float semantic_features[SEMANTIC_DIM];

    if (have_engram_system) {
        // TODO: Replace with actual engram_get_neurons() call
        // For now, use deterministic features based on engram ID
        for (uint32_t i = 0; i < SEMANTIC_DIM; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && SEMANTIC_DIM > 256) {
                systems_consolidation_heartbeat("consolidatio_loop",
                                 (float)(i + 1) / (float)SEMANTIC_DIM);
            }

            semantic_features[i] = ((float)(engram_id % 100) / 100.0F) + (i * 0.01F);
        }
    } else {
        // Simplified mode for testing without engram system
        for (uint32_t i = 0; i < SEMANTIC_DIM; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && SEMANTIC_DIM > 256) {
                systems_consolidation_heartbeat("consolidatio_loop",
                                 (float)(i + 1) / (float)SEMANTIC_DIM);
            }

            semantic_features[i] = ((float)(engram_id % 100) / 100.0F) + (i * 0.01F);
        }
    }

    // WHAT: Check if cortical node already exists for this engram
    cortical_memory_node_t* existing_node = NULL;
    for (uint32_t i = 0; i < system->node_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->node_count > 256) {
            systems_consolidation_heartbeat("consolidatio_loop",
                             (float)(i + 1) / (float)system->node_count);
        }

        if (system->cortical_nodes[i]->source_engram_id == engram_id) {
            existing_node = system->cortical_nodes[i];
            break;
        }
    }

    if (existing_node) {
        // WHAT: Update existing node (strengthening)
        // WHY: Repeated replay consolidates memory
        existing_node->consolidation_strength += replay_strength * 0.1F;
        if (existing_node->consolidation_strength > 1.0F) {
            existing_node->consolidation_strength = 1.0F;
        }

        existing_node->hippocampal_dependency -= replay_strength * 0.05F;
        if (existing_node->hippocampal_dependency < 0.0F) {
            existing_node->hippocampal_dependency = 0.0F;
            existing_node->is_transferred = true;
            system->total_transfers++;
        }

        existing_node->last_activation_ms = nimcp_platform_time_monotonic_ms();
        return existing_node->id;
    }

    // WHAT: Create new cortical node
    // WHY: First transfer of this engram to cortex
    cortical_memory_node_t* new_node = cortical_node_create(
        semantic_features,
        SEMANTIC_DIM,
        engram_id
    );

    if (!new_node) {
        return 0;  // Allocation failed
    }

    // WHAT: Check capacity
    if (system->node_count >= system->node_capacity) {
        cortical_node_destroy(new_node);
        return 0;  // Cortex full
    }

    // WHAT: Apply initial replay strength to new node
    // WHY: First replay establishes initial consolidation
    new_node->consolidation_strength = replay_strength * 0.1F;
    new_node->hippocampal_dependency -= replay_strength * 0.05F;

    // WHAT: Add to system
    system->cortical_nodes[system->node_count] = new_node;
    system->node_count++;

    // WHAT: Link to similar nodes (semantic clustering)
    // WHY: Cortex groups related concepts (lateral connections)
    for (uint32_t i = 0; i < system->node_count - 1; i++) {
        cortical_memory_node_t* other_node = system->cortical_nodes[i];
        float similarity = nimcp_vector_cosine_similarity(
            new_node->features,
            other_node->features,
            SEMANTIC_DIM
        );

        // WHAT: Link if sufficiently similar
        if (similarity > 0.7F) {
            cortical_node_add_neighbor(new_node, other_node, similarity);
            cortical_node_add_neighbor(other_node, new_node, similarity);
        }
    }

    return new_node->id;
}

void systems_consolidation_update(
    systems_consolidation_system_t* system,
    float time_delta_seconds,
    bool is_sleeping)
{
    // WHAT: Guard against invalid input
    if (!system || time_delta_seconds <= 0.0F) {
        return;
    }

    /* Phase 8: Send heartbeat at start of consolidation update */
    systems_consolidation_heartbeat("consolidation_update", 0.0f);

    // Process pending bio-async messages
    if (system->bio_async_enabled && system->bio_ctx) {
        bio_router_process_inbox(system->bio_ctx, 5);
    }

    // WHAT: Determine consolidation rate based on sleep state
    // WHY: Sleep accelerates consolidation (Born & Wilhelm, 2012)
    float consolidation_rate = is_sleeping
        ? CONSOLIDATION_TRANSFER_RATE_SWS
        : CONSOLIDATION_TRANSFER_RATE_AWAKE;

    float time_delta_hours = time_delta_seconds / 3600.0F;
    float consolidation_increment = consolidation_rate * time_delta_hours;
    float forgetting_decrement = system->forgetting_rate * time_delta_hours;

    // WHAT: Update all cortical nodes
    for (uint32_t i = 0; i < system->node_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->node_count > 256) {
            systems_consolidation_heartbeat("consolidatio_loop",
                             (float)(i + 1) / (float)system->node_count);
        }

        cortical_memory_node_t* node = system->cortical_nodes[i];

        // WHAT: Increase consolidation strength
        // WHY: Gradual strengthening over time
        node->consolidation_strength += consolidation_increment;
        if (node->consolidation_strength > 1.0F) {
            node->consolidation_strength = 1.0F;
        }

        // WHAT: Decrease hippocampal dependency
        // WHY: Cortex becomes independent over time
        node->hippocampal_dependency -= consolidation_increment;
        if (node->hippocampal_dependency < 0.0F) {
            node->hippocampal_dependency = 0.0F;
            if (!node->is_transferred) {
                node->is_transferred = true;
                system->total_transfers++;
            }
        }

        // WHAT: Check for episodic → semantic transition
        // WHY: Details fade, gist remains (Winocur & Moscovitch, 2011)
        if (node->consolidation_strength >= system->semantic_threshold &&
            node->type == CORTICAL_MEMORY_EPISODIC) {
            node->type = CORTICAL_MEMORY_SEMANTIC;
        }

        // WHAT: Apply forgetting to unrehearsed memories
        // WHY: Memories decay without rehearsal (Ebbinghaus forgetting curve)
        uint64_t time_since_activation = nimcp_platform_time_monotonic_ms() - node->last_activation_ms;
        if (time_since_activation > 3600000) {  // >1 hour
            node->consolidation_strength -= forgetting_decrement;
            if (node->consolidation_strength < 0.0F) {
                node->consolidation_strength = 0.0F;
            }
        }
    }
}

//=============================================================================
// Query API
//=============================================================================

cortical_memory_node_t* systems_consolidation_get_node(
    const systems_consolidation_system_t* system,
    uint64_t node_id)
{
    // WHAT: Guard against invalid input
    if (!system || node_id == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "systems_consolidation_get_node: system is NULL");
        return NULL;
    }

    // WHAT: Linear search for node
    /* Phase 8: Heartbeat at operation start */
    systems_consolidation_heartbeat("consolidatio_systems_consolidatio", 0.0f);


    for (uint32_t i = 0; i < system->node_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && system->node_count > 256) {
            systems_consolidation_heartbeat("consolidatio_loop",
                             (float)(i + 1) / (float)system->node_count);
        }

        if (system->cortical_nodes[i]->id == node_id) {
            return system->cortical_nodes[i];
        }
    }

    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "systems_consolidation_get_node: validation failed");
    return NULL;  // Not found
}

uint32_t systems_consolidation_find_similar(
    const systems_consolidation_system_t* system,
    const float* query_features,
    uint32_t feature_dim,
    uint32_t max_results,
    uint64_t* results_out,
    float* similarities_out)
{
    // WHAT: Guard against invalid input
    if (!system || !query_features || feature_dim == 0 || max_results == 0 ||
        !results_out || !similarities_out) {
        return 0;
    }

    /* Phase 8: Heartbeat at operation start */
    systems_consolidation_heartbeat("consolidatio_systems_consolidatio", 0.0f);

    // WHAT: Compute similarities for all nodes
    uint32_t results_found = 0;

    for (uint32_t i = 0; i < system->node_count && results_found < max_results; i++) {
        cortical_memory_node_t* node = system->cortical_nodes[i];

        // WHAT: Skip if dimensionality mismatch
        if (node->feature_dim != feature_dim) {
            continue;
        }

        // WHAT: Compute similarity
        float similarity = nimcp_vector_cosine_similarity(
            query_features,
            node->features,
            feature_dim
        );

        // WHAT: Insert into results (simple linear insertion)
        // TODO: Use heap for large result sets
        uint32_t insert_pos = results_found;
        for (uint32_t j = 0; j < results_found; j++) {
            /* Phase 8: Loop progress heartbeat */
            if ((j & 0xFF) == 0 && results_found > 256) {
                systems_consolidation_heartbeat("consolidatio_loop",
                                 (float)(j + 1) / (float)results_found);
            }

            if (similarity > similarities_out[j]) {
                insert_pos = j;
                break;
            }
        }

        // WHAT: Shift results if needed
        if (insert_pos < max_results) {
            for (uint32_t j = results_found; j > insert_pos && j > 0; j--) {
                if (j < max_results) {
                    results_out[j] = results_out[j - 1];
                    similarities_out[j] = similarities_out[j - 1];
                }
            }

            results_out[insert_pos] = node->id;
            similarities_out[insert_pos] = similarity;

            if (results_found < max_results) {
                results_found++;
            }
        }
    }

    return results_found;
}

void systems_consolidation_get_statistics(
    const systems_consolidation_system_t* system,
    uint32_t* total_nodes_out,
    uint64_t* total_replays_out,
    uint64_t* total_transfers_out,
    uint64_t* total_forgotten_out,
    uint32_t* pending_replays_out)
{
    // WHAT: Guard against NULL system
    if (!system) {
        return;
    }

    // WHAT: Return statistics
    /* Phase 8: Heartbeat at operation start */
    systems_consolidation_heartbeat("consolidatio_systems_consolidatio", 0.0f);


    if (total_nodes_out) *total_nodes_out = system->node_count;
    if (total_replays_out) *total_replays_out = system->total_replays;
    if (total_transfers_out) *total_transfers_out = system->total_transfers;
    if (total_forgotten_out) *total_forgotten_out = system->total_forgotten;
    if (pending_replays_out) *pending_replays_out = system->replay_queue_size;
}

//=============================================================================
// Integration API
//=============================================================================

void systems_consolidation_set_engram_system(
    systems_consolidation_system_t* system,
    engram_system_t* engram_system)
{
    // WHAT: Guard against NULL system
    if (!system) {
        return;
    }

    // WHAT: Store non-owning pointer
    /* Phase 8: Heartbeat at operation start */
    systems_consolidation_heartbeat("consolidatio_systems_consolidatio", 0.0f);


    system->engram_system = engram_system;
}

void systems_consolidation_set_sleep_system(
    systems_consolidation_system_t* system,
    void* sleep_system)
{
    // WHAT: Guard against NULL system
    if (!system) {
        return;
    }

    // WHAT: Store opaque pointer
    /* Phase 8: Heartbeat at operation start */
    systems_consolidation_heartbeat("consolidatio_systems_consolidatio", 0.0f);


    system->sleep_system = sleep_system;
}

//=============================================================================
// KG Self-Awareness Integration
//=============================================================================

/**
 * @brief Query self-knowledge from knowledge graph
 * WHAT: Retrieve module's self-awareness information from KG
 * WHY:  Enable introspection about module capabilities and connections
 * HOW:  Query KG reader for entity and relations
 */
int systems_consolidation_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    /* Phase 8: Heartbeat at operation start */
    systems_consolidation_heartbeat("consolidatio_systems_consolidatio", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Systems_Consolidation_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                systems_consolidation_heartbeat("consolidatio_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            LOG_DEBUG("Systems consolidation self-knowledge: %s", self->observations[i]);
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Systems_Consolidation_Module");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Systems_Consolidation_Module");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void systems_consolidation_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_systems_consolidation_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int systems_consolidation_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "systems_consolidation_training_begin: NULL argument");
        return -1;
    }
    systems_consolidation_heartbeat_instance(NULL, "systems_consolidation_training_begin", 0.0f);
    return 0;
}

int systems_consolidation_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "systems_consolidation_training_end: NULL argument");
        return -1;
    }
    systems_consolidation_heartbeat_instance(NULL, "systems_consolidation_training_end", 1.0f);
    return 0;
}

int systems_consolidation_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "systems_consolidation_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    systems_consolidation_heartbeat_instance(NULL, "systems_consolidation_training_step", progress);
    return 0;
}
