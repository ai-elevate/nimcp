/**
 * @file nimcp_mirror_neurons.c
 * @brief Mirror Neuron System Implementation
 * @version 1.0.0
 * @date 2025-01-09
 *
 * Implementation follows NIMCP coding standards:
 * - Proper WHAT/WHY/HOW documentation
 * - Error validation on all inputs
 * - Memory safety and leak prevention
 * - Thread-safety where applicable
 * - Performance optimization
 */

#include "cognitive/nimcp_mirror_neurons.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "cognitive/mirror_neurons/nimcp_mirror_substrate.h"  // Substrate integration (Phase 10.11.2)
#include "cognitive/mirror_neurons/nimcp_mirror_stdp.h"       // STDP learning (Phase 10.11.4)
#include "cognitive/mirror_neurons/nimcp_mirror_resonance.h"  // Motor resonance (Phase 10.11.5)
#include "cognitive/mirror_neurons/nimcp_mirror_hierarchy.h"  // Hierarchical goals (Phase 10.11.6)
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"  // Neuromodulator integration
#include "core/brain/nimcp_brain.h"  // Brain reference
#include <string.h>
#include <math.h>
#include <stdlib.h>

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_wiring_helpers.h"
#include "nimcp.h"  // For error codes

// SNN and Plasticity bridge integration
#include "cognitive/mirror_neurons/nimcp_mirror_snn_bridge.h"
#include "cognitive/mirror_neurons/nimcp_mirror_plasticity_bridge.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "mirror_neurons"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for mirror_neurons module */
static nimcp_health_agent_t* g_mirror_neurons_health_agent = NULL;

/**
 * @brief Set health agent for mirror_neurons heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void mirror_neurons_set_health_agent(nimcp_health_agent_t* agent) {
    g_mirror_neurons_health_agent = agent;
}

/** @brief Send heartbeat from mirror_neurons module */
static inline void mirror_neurons_heartbeat(const char* operation, float progress) {
    if (g_mirror_neurons_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_mirror_neurons_health_agent, operation, progress);
    }
}


// Logging macros
#define MIRROR_LOG_ERROR NIMCP_LOGGING_ERROR
#define MIRROR_LOG_WARN NIMCP_LOGGING_WARN
#define MIRROR_LOG_INFO NIMCP_LOGGING_INFO

//=============================================================================
// Internal Data Structures
//=============================================================================

/**
 * @brief Single mirror neuron unit
 *
 * WHAT: Represents one mirror neuron with dual observation/execution pathways
 * WHY:  Enable shared representation for observed and executed actions
 * HOW:  Combines cognitive state with optional biological substrate backing
 */
typedef struct {
    uint32_t neuron_id;                /**< Unique neuron identifier */
    uint32_t action_id;                /**< Action this neuron represents */

    // Dual pathways
    float observation_activation;      /**< Current observation activation */
    float execution_activation;        /**< Current execution activation */

    // Learning state
    float association_weight;          /**< Obs-exec association strength */
    uint32_t observation_count;        /**< Times activated by observation */
    uint32_t execution_count;          /**< Times activated by execution */

    // Feature representation
    float action_features[32];         /**< Learned action features */
    uint32_t num_features;             /**< Number of features */

    // Temporal state
    uint64_t last_observation_time;    /**< Last observation activation */
    uint64_t last_execution_time;      /**< Last execution activation */

    // Phase 10.11.2: Substrate integration
    mirror_substrate_backing_t* substrate; /**< Biological substrate backing (NULL = abstract mode) */
    bool has_substrate;                /**< True if substrate backing is active */

} mirror_neuron_unit_t;

/**
 * @brief Action-to-neuron mapping entry
 *
 * WHAT: Maps action IDs to corresponding mirror neuron populations
 * WHY:  Enable fast lookup of neurons for a given action
 */
typedef struct {
    uint32_t action_id;
    char action_name[64];
    uint32_t* neuron_indices;          /**< Array of neuron indices */
    uint32_t num_neurons;              /**< Number of neurons for this action */
    uint32_t capacity;                 /**< Allocated capacity */

    // Statistics
    uint32_t total_observations;
    uint32_t total_executions;
    float avg_similarity;              /**< Average obs-exec similarity */

} action_mapping_t;

/**
 * @brief Observed agent tracking
 *
 * WHAT: Track information about observed agents
 * WHY:  Enable multi-agent learning and agent-specific adaptation
 */
typedef struct {
    uint32_t agent_id;
    uint32_t observation_count;
    uint64_t last_observation_time;
    float trust_score;                 /**< Agent reliability (0.0-1.0) */
} agent_info_t;

/**
 * @brief Main mirror neuron system structure
 *
 * WHAT: Complete mirror neuron system state
 * WHY:  Encapsulate all data for observation-based learning
 * HOW:  Combines cognitive infrastructure with biological substrate integration
 */
struct mirror_neurons_system {
    // Configuration
    mirror_neuron_config_t config;

    // Neuron population
    mirror_neuron_unit_t* neurons;
    uint32_t num_neurons;

    // Action mappings
    action_mapping_t* actions;
    uint32_t num_actions;
    uint32_t actions_capacity;

    // Agent tracking
    agent_info_t* agents;
    uint32_t num_agents;
    uint32_t agents_capacity;

    // Statistics (atomic for thread-safety)
    mirror_neuron_stats_t stats;

    // Integration handles
    void* working_memory;              /**< Working memory system */
    void* theory_of_mind;              /**< Theory of mind system */
    void* predictive_network;          /**< Predictive network */
    void* glial_integration;           /**< Glial cell integration (Phase 10.11.1) */
    brain_t brain;                     /**< Brain reference for neuromodulation */

    // Temporal state
    uint64_t creation_time;
    uint64_t last_update_time;

    // Memory management
    bool initialized;

    // Phase 10.11.2: Substrate integration
    mirror_substrate_config_t substrate_config;   /**< Substrate configuration */
    mirror_substrate_pool_t* substrate_pool;      /**< Memory pool for substrate backings */
    bool substrate_enabled;                       /**< True if substrate mode active */

    // Phase 10.11.2: Network integration handles
    void* axon_network;                /**< Axon network for propagation delays */
    void* dendrite_network;            /**< Dendrite network for spine plasticity */
    void* myelin_network;              /**< Myelin sheath network for myelination */

    // Phase 10.11.4-6: Enhancement systems
    void* stdp_system;                 /**< STDP learning system */
    void* resonance_system;            /**< Motor resonance system */
    void* hierarchy_system;            /**< Goal-motor hierarchy system */
    bool stdp_enabled;                 /**< True if STDP enabled */
    bool resonance_enabled;            /**< True if resonance enabled */
    bool hierarchy_enabled;            /**< True if hierarchy enabled */

    // Bio-async integration
    bio_module_context_t bio_ctx;      /**< Bio-async module context */
    bool bio_async_enabled;            /**< Bio-async registration status */

    // SNN and Plasticity bridge integration
    mirror_snn_bridge_t* snn_bridge;           /**< SNN bridge for spike-based computation */
    mirror_plasticity_bridge_t* plasticity_bridge; /**< Plasticity bridge for learning rules */
    bool bridges_enabled;                      /**< True if bridges are active */
};

//=============================================================================
// Forward Declarations
//=============================================================================

static float compute_feature_similarity(const float* f1, const float* f2, uint32_t n);
static uint32_t find_or_create_action(mirror_neurons_t mirror, const action_t* action);
static uint32_t find_or_create_agent(mirror_neurons_t mirror, uint32_t agent_id);
static void activate_neurons_for_action(mirror_neurons_t mirror, uint32_t action_idx,
                                       float strength, bool is_observation);
static void update_action_statistics(mirror_neurons_t mirror, uint32_t action_idx);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get default configuration
 *
 * WHAT: Return sensible default configuration
 * WHY:  Provide good starting point for most use cases
 * HOW:  Return pre-configured struct
 */
mirror_neuron_config_t mirror_neurons_get_default_config(void)
{
    mirror_neuron_config_t config = {
        .num_mirror_neurons = 1000,
        .max_actions = 100,
        .max_agents = 10,
        .learning_rate = 0.01F,
        .decay_rate = 0.05F,
        .match_threshold = 0.7F,
        .enable_working_memory = true,
        .enable_theory_of_mind = true,
        .enable_prediction = true,
        .observation_window = 1000,
        .replay_capacity = 100,
        // Phase 10.11.1: Glial cell support (default: all enabled)
        .enable_glial_modulation = true,
        .enable_astrocytes = true,
        .enable_oligodendrocytes = true,
        .enable_microglia = true,
        // Phase 10.11.2: Substrate integration (default: disabled for backward compat)
        .enable_substrate = false,
        .enable_myelination = true,
        .enable_dendrite_plasticity = true,
        .enable_axon_timing = true,
        .enable_substrate_pool = true,
        .substrate_pool_size = NIMCP_MIRROR_SUBSTRATE_POOL_SIZE
    };
    return config;
}

/**
 * @brief Create action from features
 *
 * WHAT: Helper to construct action_t from components
 * WHY:  Simplify action creation in tests and applications
 * HOW:  Fill struct with provided data
 */
action_t mirror_neurons_create_action(
    uint32_t action_id,
    const char* action_name,
    const float* features,
    uint32_t num_features,
    uint32_t agent_id)
{
    action_t action = {0};
    action.action_id = action_id;
    action.agent_id = agent_id;
    action.num_features = (num_features > 32) ? 32 : num_features;
    action.timestamp = nimcp_time_get_ms();
    action.confidence = 1.0F;

    if (action_name) {
        strncpy(action.action_name, action_name, sizeof(action.action_name) - 1);
    }

    if (features && num_features > 0) {
        memcpy(action.features, features, action.num_features * sizeof(float));
    }

    return action;
}

/**
 * @brief Compute cosine similarity between feature vectors
 *
 * WHAT: Calculate similarity between two feature vectors
 * WHY:  Determine how similar two actions are
 * HOW:  Compute dot product / (norm1 * norm2)
 *
 * @return Similarity in range [0.0, 1.0], or -1.0 on error
 */
static float compute_feature_similarity(const float* f1, const float* f2, uint32_t n)
{
    if (!f1 || !f2 || n == 0) {
        return -1.0F;
    }

    float dot_product = 0.0F;
    float norm1 = 0.0F;
    float norm2 = 0.0F;

    for (uint32_t i = 0; i < n; i++) {
        dot_product += f1[i] * f2[i];
        norm1 += f1[i] * f1[i];
        norm2 += f2[i] * f2[i];
    }

    if (norm1 == 0.0F || norm2 == 0.0F) {
        return 0.0F;
    }

    float similarity = dot_product / (sqrtf(norm1) * sqrtf(norm2));

    // Normalize to [0, 1] from [-1, 1]
    similarity = (similarity + 1.0F) / 2.0F;

    return (similarity < 0.0F) ? 0.0F : ((similarity > 1.0F) ? 1.0F : similarity);
}

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Find or create action mapping
 *
 * WHAT: Get index of action in actions array, create if not exists
 * WHY:  Maintain action-to-neuron mappings
 * HOW:  Linear search, then append if not found
 *
 * @return Action index, or UINT32_MAX on error
 */
static uint32_t find_or_create_action(mirror_neurons_t mirror, const action_t* action)
{
    if (!mirror || !action) {
        return UINT32_MAX;
    }

    // Search for existing action
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        if (mirror->actions[i].action_id == action->action_id) {
            return i;
        }
    }

    // Check capacity
    if (mirror->num_actions >= mirror->config.max_actions) {
        MIRROR_LOG_ERROR("Mirror neurons: max actions limit reached (%u)",
                       mirror->config.max_actions);
        return UINT32_MAX;
    }

    // Create new action mapping
    uint32_t idx = mirror->num_actions++;
    action_mapping_t* mapping = &mirror->actions[idx];

    mapping->action_id = action->action_id;
    strncpy(mapping->action_name, action->action_name, sizeof(mapping->action_name) - 1);
    mapping->num_neurons = 0;
    mapping->capacity = 10;  // Initial capacity
    mapping->neuron_indices = (uint32_t*)nimcp_malloc(mapping->capacity * sizeof(uint32_t));
    mapping->total_observations = 0;
    mapping->total_executions = 0;
    mapping->avg_similarity = 0.0F;

    if (!mapping->neuron_indices) {
        MIRROR_LOG_ERROR("Mirror neurons: failed to allocate neuron indices");
        mirror->num_actions--;
        return UINT32_MAX;
    }

    return idx;
}

/**
 * @brief Find or create agent tracking entry
 *
 * WHAT: Get index of agent in agents array, create if not exists
 * WHY:  Track which agents we've observed
 * HOW:  Linear search, then append if not found
 *
 * @return Agent index, or UINT32_MAX on error
 */
static uint32_t find_or_create_agent(mirror_neurons_t mirror, uint32_t agent_id)
{
    if (!mirror) {
        return UINT32_MAX;
    }

    // Search for existing agent
    for (uint32_t i = 0; i < mirror->num_agents; i++) {
        if (mirror->agents[i].agent_id == agent_id) {
            return i;
        }
    }

    // Check capacity
    if (mirror->num_agents >= mirror->config.max_agents) {
        MIRROR_LOG_WARN("Mirror neurons: max agents limit reached (%u)",
                      mirror->config.max_agents);
        return UINT32_MAX;
    }

    // Create new agent entry
    uint32_t idx = mirror->num_agents++;
    agent_info_t* agent = &mirror->agents[idx];

    agent->agent_id = agent_id;
    agent->observation_count = 0;
    agent->last_observation_time = nimcp_time_get_ms();
    agent->trust_score = 0.5F;  // Neutral trust initially

    return idx;
}

/**
 * @brief Activate neurons for an action
 *
 * WHAT: Set activation levels for neurons representing an action
 * WHY:  Implement observation or execution pathway
 * HOW:  Find neurons for action, update activations
 */
/**
 * @brief Get acetylcholine modulation for mirror neuron observation
 *
 * WHAT: Compute ACh-based gating factor for observed actions
 * WHY:  Acetylcholine gates social learning and action observation
 * HOW:  Read ACh level, map to modulation factor [0.6, 1.4]
 *
 * BIOLOGY: ACh enhances attention to observed actions
 *          High ACh (0.7) → 1.4× observation strength (focused social learning)
 *          Low ACh (0.3) → 0.6× observation strength (inattentive, autism-like)
 *
 * COMPLEXITY: O(1)
 *
 * @param mirror Mirror neuron system
 * @return Modulation factor [0.6, 1.4], or 1.0 if no brain
 */
static float get_mirror_ach_modulation(mirror_neurons_t mirror)
{
    // Guard: Early return if no brain
    if (!mirror || !mirror->brain) {
        return 1.0F;
    }

    neuromodulator_system_t neuromod = brain_get_neuromodulator_system(mirror->brain);
    if (!neuromod) {
        return 1.0F;
    }

    // Read acetylcholine level
    float ach = neuromodulator_get_level(neuromod, NEUROMOD_ACETYLCHOLINE);

    // Map ACh range [0.3, 0.7] to modulation [0.6, 1.4]
    float modulation = 0.6F + (ach - 0.3F) * 2.0F;

    return modulation;
}

static void activate_neurons_for_action(
    mirror_neurons_t mirror,
    uint32_t action_idx,
    float strength,
    bool is_observation)
{
    if (!mirror || action_idx >= mirror->num_actions) {
        return;
    }

    action_mapping_t* mapping = &mirror->actions[action_idx];
    uint64_t current_time = nimcp_time_get_ms();

    // Allocate neurons if this is first activation
    if (mapping->num_neurons == 0) {
        // Assign a small population of neurons to this action
        uint32_t neurons_per_action = mirror->config.num_mirror_neurons /
                                     (mirror->config.max_actions + 1);
        neurons_per_action = (neurons_per_action < 5) ? 5 : neurons_per_action;

        for (uint32_t i = 0; i < neurons_per_action && mapping->num_neurons < mapping->capacity; i++) {
            // Find an available neuron
            uint32_t neuron_idx = (action_idx * neurons_per_action + i) % mirror->num_neurons;
            if (mirror->neurons[neuron_idx].action_id == 0 ||
                mirror->neurons[neuron_idx].action_id == mapping->action_id) {

                mirror->neurons[neuron_idx].action_id = mapping->action_id;
                mirror->neurons[neuron_idx].neuron_id = neuron_idx;
                mapping->neuron_indices[mapping->num_neurons++] = neuron_idx;
            }
        }
    }

    // Apply acetylcholine modulation to observation pathway
    float ach_modulation = is_observation ? get_mirror_ach_modulation(mirror) : 1.0F;

    // Activate neurons
    for (uint32_t i = 0; i < mapping->num_neurons; i++) {
        uint32_t neuron_idx = mapping->neuron_indices[i];
        mirror_neuron_unit_t* neuron = &mirror->neurons[neuron_idx];

        if (is_observation) {
            // Apply ACh gating: enhances social learning when attentive
            neuron->observation_activation += strength * ach_modulation;
            if (neuron->observation_activation > 1.0F) {
                neuron->observation_activation = 1.0F;
            }
            neuron->observation_count++;
            neuron->last_observation_time = current_time;
        } else {
            neuron->execution_activation += strength;
            if (neuron->execution_activation > 1.0F) {
                neuron->execution_activation = 1.0F;
            }
            neuron->execution_count++;
            neuron->last_execution_time = current_time;
        }
    }
}

/**
 * @brief Update action statistics
 *
 * WHAT: Recalculate statistics for an action
 * WHY:  Track learning progress and matching quality
 * HOW:  Aggregate neuron-level metrics
 */
static void update_action_statistics(mirror_neurons_t mirror, uint32_t action_idx)
{
    // Process pending bio-async messages
    if (mirror && mirror->bio_async_enabled && mirror->bio_ctx) {
        bio_router_process_inbox(mirror->bio_ctx, 5);
    }

    if (!mirror || action_idx >= mirror->num_actions) {
        return;
    }

    action_mapping_t* mapping = &mirror->actions[action_idx];

    if (mapping->num_neurons == 0) {
        return;
    }

    // Calculate average similarity across neurons
    float total_similarity = 0.0F;
    uint32_t count = 0;

    for (uint32_t i = 0; i < mapping->num_neurons; i++) {
        uint32_t neuron_idx = mapping->neuron_indices[i];
        mirror_neuron_unit_t* neuron = &mirror->neurons[neuron_idx];

        if (neuron->observation_count > 0 && neuron->execution_count > 0) {
            // Similarity based on co-activation
            float obs_norm = neuron->observation_activation;
            float exec_norm = neuron->execution_activation;
            float similarity = (obs_norm * exec_norm) /
                             (obs_norm + exec_norm + 0.001F);  // Prevent div by 0
            total_similarity += similarity;
            count++;
        }
    }

    if (count > 0) {
        mapping->avg_similarity = total_similarity / count;
    }
}

//=============================================================================
// BIO-ASYNC MESSAGE HANDLERS
//=============================================================================

/**
 * @brief Bio-async message handler: Handle mirror neuron activation
 */
static nimcp_error_t handle_mirror_activation(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    (void)msg_size;
    (void)response_promise;

    if (!msg || !user_data) {
        return NIMCP_ERROR_NULL_ARG;
    }

    const bio_msg_introspection_query_t* activation = (const bio_msg_introspection_query_t*)msg;
    mirror_neurons_t mirror = (mirror_neurons_t)user_data;
    (void)mirror;  // Will be used for actual processing
    (void)activation;

    LOG_DEBUG(LOG_MODULE, "Received mirror neuron activation via bio-async");

    return NIMCP_SUCCESS;
}

/**
 * @brief Broadcast mirror neuron fire event via bio-async
 */
static void bio_broadcast_mirror_fire(mirror_neurons_t mirror,
                                       uint32_t action_id,
                                       float activation) {
    if (!mirror || !mirror->bio_async_enabled || !mirror->bio_ctx) {
        return;
    }

    bio_msg_introspection_response_t msg = {0};
    bio_msg_init_header(&msg.header, BIO_MSG_MIRROR_NEURON_ACTIVATION,
                        bio_module_context_get_id(mirror->bio_ctx), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.query_type = action_id;
    msg.confidence = activation;
    msg.matched_pattern_count = 1;

    bio_router_broadcast(mirror->bio_ctx, &msg, sizeof(msg));
    LOG_DEBUG(LOG_MODULE, "Broadcast mirror fire: action=%u, activation=%.2f",
              action_id, activation);
}

//=============================================================================
// KG-Driven Wiring Callback
//=============================================================================

/**
 * @brief Wiring callback for KG-driven handler registration
 *
 * WHAT: Callback invoked by orchestrator with discovered message types
 * WHY:  Enable dynamic handler registration based on KG wiring diagram
 * HOW:  Register handlers for discovered message types from handler map
 *
 * @param ctx Bio-async module context
 * @param message_types Array of discovered message types from KG
 * @param message_count Number of message types
 * @param user_data User-provided context (mirror_neurons_t)
 * @return 0 on success, -1 on error
 */
static int mirror_neurons_wiring_handler_callback(
    bio_module_context_t ctx,
    const bio_message_type_t* message_types,
    uint32_t message_count,
    void* user_data
) {
    (void)user_data;

    int registered = 0;
    for (uint32_t i = 0; i < message_count; i++) {
        switch (message_types[i]) {
            case BIO_MSG_MIRROR_NEURON_ACTIVATION:
                bio_router_register_handler(ctx, message_types[i], handle_mirror_activation);
                registered++;
                break;
            default:
                LOG_DEBUG(LOG_MODULE, "Unknown message type %d in wiring callback", message_types[i]);
                break;
        }
    }

    MIRROR_LOG_INFO("KG-driven wiring callback registered %d handlers", registered);
    return (registered > 0) ? 0 : -1;
}

//=============================================================================
// Core API Implementation - Lifecycle
//=============================================================================

/**
 * @brief Create mirror neuron system
 */
mirror_neurons_t mirror_neurons_create(const mirror_neuron_config_t* config)
{
    // Allocate system structure
    mirror_neurons_t mirror = (mirror_neurons_t)nimcp_malloc(sizeof(struct mirror_neurons_system));
    if (!mirror) {
        MIRROR_LOG_ERROR("Mirror neurons: failed to allocate system structure");
        return NULL;
    }

    memset(mirror, 0, sizeof(struct mirror_neurons_system));

    // Use default config if none provided
    if (config) {
        memcpy(&mirror->config, config, sizeof(mirror_neuron_config_t));
    } else {
        mirror->config = mirror_neurons_get_default_config();
    }

    // Validate config
    if (mirror->config.num_mirror_neurons == 0 || mirror->config.max_actions == 0) {
        MIRROR_LOG_ERROR("Mirror neurons: invalid configuration");
        nimcp_free(mirror);
        return NULL;
    }

    // Allocate neurons
    mirror->neurons = (mirror_neuron_unit_t*)nimcp_calloc(
        mirror->config.num_mirror_neurons,
        sizeof(mirror_neuron_unit_t)
    );
    if (!mirror->neurons) {
        MIRROR_LOG_ERROR("Mirror neurons: failed to allocate neurons");
        nimcp_free(mirror);
        return NULL;
    }
    mirror->num_neurons = mirror->config.num_mirror_neurons;

    // Allocate action mappings
    mirror->actions_capacity = mirror->config.max_actions;
    mirror->actions = (action_mapping_t*)nimcp_calloc(
        mirror->actions_capacity,
        sizeof(action_mapping_t)
    );
    if (!mirror->actions) {
        MIRROR_LOG_ERROR("Mirror neurons: failed to allocate actions");
        nimcp_free(mirror->neurons);
        nimcp_free(mirror);
        return NULL;
    }

    // Allocate agent tracking
    mirror->agents_capacity = mirror->config.max_agents;
    mirror->agents = (agent_info_t*)nimcp_calloc(
        mirror->agents_capacity,
        sizeof(agent_info_t)
    );
    if (!mirror->agents) {
        MIRROR_LOG_ERROR("Mirror neurons: failed to allocate agents");
        nimcp_free(mirror->actions);
        nimcp_free(mirror->neurons);
        nimcp_free(mirror);
        return NULL;
    }

    // Initialize temporal state
    mirror->creation_time = nimcp_time_get_ms();
    mirror->last_update_time = 0;  // Set to 0 until first observation
    mirror->brain = NULL;  // Initialize brain reference
    mirror->initialized = true;

    MIRROR_LOG_INFO("Mirror neurons: created system with %u neurons, max %u actions",
                  mirror->num_neurons, mirror->config.max_actions);

    // Initialize bio-async fields
    mirror->bio_ctx = NULL;
    mirror->bio_async_enabled = false;

    // Register with bio-async router if available
    MIRROR_LOG_INFO("mirror_neurons: Checking bio-async router initialization...");
    if (bio_router_is_initialized()) {
        MIRROR_LOG_INFO("mirror_neurons: Bio-router initialized, registering module (id=%d, inbox_capacity=32)...",
                       BIO_MODULE_MIRROR_NEURONS);
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_MIRROR_NEURONS,
            .module_name = "mirror_neurons",
            .inbox_capacity = 32,
            .user_data = mirror
        };
        mirror->bio_ctx = bio_router_register_module(&bio_info);
        if (mirror->bio_ctx) {
            mirror->bio_async_enabled = true;

            /* Register handlers via KG-driven wiring callback */
            nimcp_error_t wiring_result = bio_router_register_wiring_callback(
                BIO_MODULE_MIRROR_NEURONS,
                (void*)mirror_neurons_wiring_handler_callback,
                mirror
            );

            if (wiring_result != NIMCP_SUCCESS) {
                /* Legacy fallback: direct handler registration */
                MIRROR_LOG_INFO("mirror_neurons: KG wiring unavailable, using legacy registration");
                LEGACY_HANDLER_REGISTRATION(
                    bio_router_register_handler(mirror->bio_ctx, BIO_MSG_MIRROR_NEURON_ACTIVATION, handle_mirror_activation));
            }

            MIRROR_LOG_INFO("mirror_neurons: Bio-async communication enabled with handlers (module_id=%d)",
                           BIO_MODULE_MIRROR_NEURONS);
        } else {
            MIRROR_LOG_WARN("mirror_neurons: Bio-async registration failed - module will operate without async messaging");
        }
    } else {
        MIRROR_LOG_INFO("mirror_neurons: Bio-router not initialized, skipping async registration");
    }

    // Initialize SNN and Plasticity bridges
    mirror->snn_bridge = NULL;
    mirror->plasticity_bridge = NULL;
    mirror->bridges_enabled = false;

    // Create SNN bridge with default config
    mirror_snn_config_t snn_config = mirror_snn_config_default();
    mirror->snn_bridge = mirror_snn_create(&snn_config);
    if (!mirror->snn_bridge) {
        MIRROR_LOG_WARN("mirror_neurons: Failed to create SNN bridge - continuing without spike-based computation");
    }

    // Create Plasticity bridge with default config
    mirror_plasticity_config_t plasticity_config = mirror_plasticity_config_default();
    mirror->plasticity_bridge = mirror_plasticity_create(&plasticity_config);
    if (!mirror->plasticity_bridge) {
        MIRROR_LOG_WARN("mirror_neurons: Failed to create Plasticity bridge - continuing without unified learning");
    }

    // Mark bridges as enabled if at least one succeeded
    if (mirror->snn_bridge || mirror->plasticity_bridge) {
        mirror->bridges_enabled = true;
        MIRROR_LOG_INFO("mirror_neurons: Bridge integration enabled (SNN=%s, Plasticity=%s)",
                       mirror->snn_bridge ? "yes" : "no",
                       mirror->plasticity_bridge ? "yes" : "no");
    }

    return mirror;
}

/**
 * @brief Set brain reference for neuromodulator integration
 *
 * WHAT: Associate mirror neurons with brain for ACh modulation
 * WHY:  Enable acetylcholine-gated social learning
 * HOW:  Store brain reference in system structure
 *
 * COMPLEXITY: O(1)
 *
 * @param mirror Mirror neuron system
 * @param brain Brain handle (can be NULL to disable neuromodulation)
 */
void mirror_neurons_set_brain(mirror_neurons_t mirror, brain_t brain)
{
    // Guard: Validate mirror system
    if (!mirror) {
        MIRROR_LOG_ERROR("Mirror neurons: NULL system in set_brain");
        return;
    }

    mirror->brain = brain;
}

/**
 * @brief Destroy mirror neuron system
 */
void mirror_neurons_destroy(mirror_neurons_t mirror)
{
    if (!mirror) {
        return;
    }

    // Destroy SNN and Plasticity bridges first (before other cleanup)
    if (mirror->snn_bridge) {
        mirror_snn_destroy(mirror->snn_bridge);
        mirror->snn_bridge = NULL;
        MIRROR_LOG_INFO("mirror_neurons: SNN bridge destroyed");
    }

    if (mirror->plasticity_bridge) {
        mirror_plasticity_destroy(mirror->plasticity_bridge);
        mirror->plasticity_bridge = NULL;
        MIRROR_LOG_INFO("mirror_neurons: Plasticity bridge destroyed");
    }
    mirror->bridges_enabled = false;

    // Unregister from bio-async router
    if (mirror->bio_async_enabled && mirror->bio_ctx) {
        bio_router_unregister_module(mirror->bio_ctx);
        mirror->bio_ctx = NULL;
        mirror->bio_async_enabled = false;
        MIRROR_LOG_INFO("Bio-async communication disabled for mirror_neurons");
    }

    // Free action neuron indices
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        if (mirror->actions[i].neuron_indices) {
            nimcp_free(mirror->actions[i].neuron_indices);
        }
    }

    // Free all arrays
    nimcp_free(mirror->agents);
    nimcp_free(mirror->actions);
    nimcp_free(mirror->neurons);
    nimcp_free(mirror);

    MIRROR_LOG_INFO("Mirror neurons: system destroyed");
}

//=============================================================================
// Core API Implementation - Action Processing
//=============================================================================

/**
 * @brief Process observed action
 */
bool mirror_neurons_observe_action(mirror_neurons_t mirror, const action_t* action)
{
    if (!mirror || !mirror->initialized) {
        MIRROR_LOG_ERROR("Mirror neurons: invalid system handle");
        return false;
    }

    if (!action) {
        MIRROR_LOG_ERROR("Mirror neurons: null action");
        return false;
    }

    // Find or create action mapping
    uint32_t action_idx = find_or_create_action(mirror, action);
    if (action_idx == UINT32_MAX) {
        return false;
    }

    // Track agent
    if (action->agent_id != 0) {  // 0 = self
        uint32_t agent_idx = find_or_create_agent(mirror, action->agent_id);
        if (agent_idx != UINT32_MAX) {
            mirror->agents[agent_idx].observation_count++;
            mirror->agents[agent_idx].last_observation_time = nimcp_time_get_ms();
        }
    }

    // Activate observation pathway
    float activation_strength = action->confidence;
    activate_neurons_for_action(mirror, action_idx, activation_strength, true);

    // Update statistics
    mirror->actions[action_idx].total_observations++;
    mirror->stats.total_observations++;
    mirror->stats.num_observed_agents = mirror->num_agents;
    mirror->stats.num_learned_actions = mirror->num_actions;

    update_action_statistics(mirror, action_idx);

    mirror->last_update_time = nimcp_time_get_ms();

    // Broadcast mirror neuron activation via bio-async
    if (activation_strength > 0.5F) {
        bio_broadcast_mirror_fire(mirror, action->action_id, activation_strength);
    }

    return true;
}

/**
 * @brief Process self-executed action
 */
bool mirror_neurons_execute_action(mirror_neurons_t mirror, const action_t* action)
{
    if (!mirror || !mirror->initialized) {
        MIRROR_LOG_ERROR("Mirror neurons: invalid system handle");
        return false;
    }

    if (!action) {
        MIRROR_LOG_ERROR("Mirror neurons: null action");
        return false;
    }

    // Find or create action mapping
    uint32_t action_idx = find_or_create_action(mirror, action);
    if (action_idx == UINT32_MAX) {
        return false;
    }

    // Activate execution pathway
    float activation_strength = action->confidence;
    activate_neurons_for_action(mirror, action_idx, activation_strength, false);

    // Update statistics
    mirror->actions[action_idx].total_executions++;
    mirror->stats.total_executions++;
    mirror->stats.num_learned_actions = mirror->num_actions;

    update_action_statistics(mirror, action_idx);

    mirror->last_update_time = nimcp_time_get_ms();

    return true;
}

/**
 * @brief Get activation for action
 */
float mirror_neurons_get_activation(mirror_neurons_t mirror, uint32_t action_id)
{
    if (!mirror || !mirror->initialized) {
        return -1.0F;
    }

    // Find action
    uint32_t action_idx = UINT32_MAX;
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        if (mirror->actions[i].action_id == action_id) {
            action_idx = i;
            break;
        }
    }

    if (action_idx == UINT32_MAX) {
        return -1.0F;  // Action not found
    }

    // Sum activations across neurons
    float total_activation = 0.0F;
    action_mapping_t* mapping = &mirror->actions[action_idx];

    for (uint32_t i = 0; i < mapping->num_neurons; i++) {
        uint32_t neuron_idx = mapping->neuron_indices[i];
        mirror_neuron_unit_t* neuron = &mirror->neurons[neuron_idx];

        // Combined activation (max of observation and execution)
        float activation = (neuron->observation_activation > neuron->execution_activation) ?
                          neuron->observation_activation : neuron->execution_activation;
        total_activation += activation;
    }

    // Average activation
    if (mapping->num_neurons > 0) {
        return total_activation / mapping->num_neurons;
    }

    return 0.0F;
}

/**
 * @brief Match actions
 */
bool mirror_neurons_match_actions(
    mirror_neurons_t mirror,
    const action_t* observed_action,
    const action_t* executed_action,
    float* out_similarity)
{
    if (!mirror || !observed_action || !executed_action) {
        if (out_similarity) *out_similarity = 0.0F;
        return false;
    }

    // Compute feature similarity
    uint32_t num_features = (observed_action->num_features < executed_action->num_features) ?
                           observed_action->num_features : executed_action->num_features;

    if (num_features == 0) {
        if (out_similarity) *out_similarity = 0.0F;
        return false;
    }

    float similarity = compute_feature_similarity(
        observed_action->features,
        executed_action->features,
        num_features
    );

    if (out_similarity) {
        *out_similarity = similarity;
    }

    // Match if above threshold
    return (similarity >= mirror->config.match_threshold);
}

//=============================================================================
// Learning & Adaptation API Implementation
//=============================================================================

/**
 * @brief Learn from action demonstration
 */
bool mirror_neurons_learn_demonstration(
    mirror_neurons_t mirror,
    const action_t* actions,
    uint32_t num_actions,
    uint32_t demonstrator_id)
{
    (void)demonstrator_id;  // TODO: Use for agent-specific learning

    if (!mirror || !actions || num_actions == 0) {
        return false;
    }

    // Process each action in sequence
    for (uint32_t i = 0; i < num_actions; i++) {
        if (!mirror_neurons_observe_action(mirror, &actions[i])) {
            MIRROR_LOG_WARN("Mirror neurons: failed to observe action %u in demonstration", i);
        }
    }

    // TODO: Store sequence in working memory if enabled
    if (mirror->config.enable_working_memory && mirror->working_memory) {
        // Integration with working memory would go here
    }

    // Update associations across the sequence
    mirror_neurons_update_associations(mirror);

    return true;
}

/**
 * @brief Update association strengths
 */
bool mirror_neurons_update_associations(mirror_neurons_t mirror)
{
    if (!mirror || !mirror->initialized) {
        return false;
    }

    // Apply Hebbian-like learning: neurons that fire together, wire together
    for (uint32_t i = 0; i < mirror->num_neurons; i++) {
        mirror_neuron_unit_t* neuron = &mirror->neurons[i];

        if (neuron->action_id == 0) {
            continue;  // Unassigned neuron
        }

        // Calculate co-activation
        float obs_act = neuron->observation_activation;
        float exec_act = neuron->execution_activation;

        if (obs_act > 0.0F && exec_act > 0.0F) {
            // Both pathways active - strengthen association
            float delta = mirror->config.learning_rate * obs_act * exec_act;
            neuron->association_weight += delta;

            // Bound to [0, 1]
            if (neuron->association_weight > 1.0F) {
                neuron->association_weight = 1.0F;
            }
        } else {
            // Weak decay if no co-activation
            neuron->association_weight *= 0.99F;
        }
    }

    // Update action-level statistics
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        update_action_statistics(mirror, i);
    }

    return true;
}

/**
 * @brief Decay activations
 */
bool mirror_neurons_decay_activations(mirror_neurons_t mirror, uint32_t delta_time_ms)
{
    if (!mirror || !mirror->initialized) {
        return false;
    }

    // Calculate decay factor based on time
    float decay_factor = expf(-mirror->config.decay_rate * (delta_time_ms / 1000.0F));

    // Apply decay to all neurons
    for (uint32_t i = 0; i < mirror->num_neurons; i++) {
        mirror_neuron_unit_t* neuron = &mirror->neurons[i];

        neuron->observation_activation *= decay_factor;
        neuron->execution_activation *= decay_factor;

        // Threshold to zero if very small
        if (neuron->observation_activation < 0.001F) {
            neuron->observation_activation = 0.0F;
        }
        if (neuron->execution_activation < 0.001F) {
            neuron->execution_activation = 0.0F;
        }
    }

    return true;
}

//=============================================================================
// Query & Analysis API Implementation
//=============================================================================

/**
 * @brief Get system statistics
 */
bool mirror_neurons_get_stats(mirror_neurons_t mirror, mirror_neuron_stats_t* stats)
{
    if (!mirror || !stats) {
        return false;
    }

    // Copy current statistics
    memcpy(stats, &mirror->stats, sizeof(mirror_neuron_stats_t));

    // Update dynamic metrics
    stats->num_active_neurons = 0;
    for (uint32_t i = 0; i < mirror->num_neurons; i++) {
        if (mirror->neurons[i].observation_activation > 0.01F ||
            mirror->neurons[i].execution_activation > 0.01F) {
            stats->num_active_neurons++;
        }
    }

    stats->num_learned_actions = mirror->num_actions;
    stats->num_observed_agents = mirror->num_agents;

    // Calculate average match quality
    float total_quality = 0.0F;
    uint32_t count = 0;
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        if (mirror->actions[i].avg_similarity > 0.0F) {
            total_quality += mirror->actions[i].avg_similarity;
            count++;
        }
    }
    stats->avg_match_quality = (count > 0) ? (total_quality / count) : 0.0F;

    stats->last_update_time = mirror->last_update_time;

    return true;
}

/**
 * @brief Get activation record for action
 */
bool mirror_neurons_get_activation_record(
    mirror_neurons_t mirror,
    uint32_t action_id,
    mirror_activation_t* activation)
{
    if (!mirror || !activation) {
        return false;
    }

    // Find action
    uint32_t action_idx = UINT32_MAX;
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        if (mirror->actions[i].action_id == action_id) {
            action_idx = i;
            break;
        }
    }

    if (action_idx == UINT32_MAX) {
        return false;
    }

    action_mapping_t* mapping = &mirror->actions[action_idx];

    // Aggregate activation across neurons
    activation->action_id = action_id;
    activation->observation_activation = 0.0F;
    activation->execution_activation = 0.0F;
    activation->association_strength = 0.0F;
    activation->observation_count = mapping->total_observations;
    activation->execution_count = mapping->total_executions;
    activation->last_activation = 0;

    if (mapping->num_neurons == 0) {
        return true;  // Valid but no neurons yet
    }

    // Average across neurons
    for (uint32_t i = 0; i < mapping->num_neurons; i++) {
        uint32_t neuron_idx = mapping->neuron_indices[i];
        mirror_neuron_unit_t* neuron = &mirror->neurons[neuron_idx];

        activation->observation_activation += neuron->observation_activation;
        activation->execution_activation += neuron->execution_activation;
        activation->association_strength += neuron->association_weight;

        if (neuron->last_observation_time > activation->last_activation) {
            activation->last_activation = neuron->last_observation_time;
        }
        if (neuron->last_execution_time > activation->last_activation) {
            activation->last_activation = neuron->last_execution_time;
        }
    }

    activation->observation_activation /= mapping->num_neurons;
    activation->execution_activation /= mapping->num_neurons;
    activation->association_strength /= mapping->num_neurons;

    return true;
}

/**
 * @brief Predict next action in sequence
 */
bool mirror_neurons_predict_next_action(
    mirror_neurons_t mirror,
    const action_t* previous_actions,
    uint32_t num_previous,
    action_t* predicted_action,
    float* confidence)
{
    if (!mirror || !previous_actions || num_previous == 0 || !predicted_action) {
        if (confidence) *confidence = 0.0F;
        return false;
    }

    // Simple prediction: find action with highest activation that follows the sequence
    // In future, this could use more sophisticated sequence learning

    // Get last action in sequence
    const action_t* last_action = &previous_actions[num_previous - 1];

    // Find action with highest activation that's different from last
    float max_activation = 0.0F;
    uint32_t best_action_id = 0;

    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        if (mirror->actions[i].action_id == last_action->action_id) {
            continue;  // Skip same action
        }

        float activation = mirror_neurons_get_activation(mirror, mirror->actions[i].action_id);
        if (activation > max_activation) {
            max_activation = activation;
            best_action_id = mirror->actions[i].action_id;
        }
    }

    if (best_action_id == 0) {
        if (confidence) *confidence = 0.0F;
        return false;
    }

    // Create predicted action (simplified - use stored features)
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        if (mirror->actions[i].action_id == best_action_id) {
            predicted_action->action_id = best_action_id;
            strncpy(predicted_action->action_name, mirror->actions[i].action_name,
                   sizeof(predicted_action->action_name) - 1);
            predicted_action->num_features = 0;  // Would need to reconstruct features
            predicted_action->agent_id = 0;
            predicted_action->timestamp = nimcp_time_get_ms();

            if (confidence) {
                *confidence = max_activation;
            }

            return true;
        }
    }

    if (confidence) *confidence = 0.0F;
    return false;
}

//=============================================================================
// Integration API Implementation
//=============================================================================

/**
 * @brief Integrate with working memory
 */
bool mirror_neurons_integrate_working_memory(
    mirror_neurons_t mirror,
    void* working_memory)
{
    if (!mirror) {
        return false;
    }

    mirror->working_memory = working_memory;

    if (working_memory) {
        MIRROR_LOG_INFO("Mirror neurons: integrated with working memory");
    }

    return true;
}

/**
 * @brief Integrate with theory of mind
 */
bool mirror_neurons_integrate_theory_of_mind(
    mirror_neurons_t mirror,
    void* theory_of_mind)
{
    if (!mirror) {
        return false;
    }

    mirror->theory_of_mind = theory_of_mind;

    if (theory_of_mind) {
        MIRROR_LOG_INFO("Mirror neurons: integrated with theory of mind");
    }

    return true;
}

/**
 * @brief Integrate with predictive processing
 */
bool mirror_neurons_integrate_predictive(
    mirror_neurons_t mirror,
    void* predictive_network)
{
    if (!mirror) {
        return false;
    }

    mirror->predictive_network = predictive_network;

    if (predictive_network) {
        MIRROR_LOG_INFO("Mirror neurons: integrated with predictive network");
    }

    return true;
}

/**
 * @brief Integrate with glial cell system (Phase 10.11.1)
 *
 * WHAT: Enable glial cell modulation of mirror neurons
 * WHY:
 * - Astrocytes: Modulate association learning strength (Ca2+ dependent plasticity)
 * - Oligodendrocytes: Speed up action recognition (myelination reduces delays)
 * - Microglia: Prune weak/unused mirror neuron associations (synaptic homeostasis)
 * HOW:  Store glial integration handle, glial cells will modulate mirror neuron activity
 *
 * BIOLOGICAL RATIONALE:
 * Mirror neurons show dense glial coverage in premotor and parietal cortex.
 * Astrocytes modulate mirror neuron plasticity during observational learning.
 * Oligodendrocytes enhance temporal precision of action recognition.
 * Microglia maintain network efficiency by pruning non-matching associations.
 *
 * @param mirror Mirror neuron system
 * @param glial_integration Glial integration system handle
 * @return true on success, false on error
 *
 * COMPLEXITY: O(1) - just stores pointer
 */
bool mirror_neurons_integrate_glial(
    mirror_neurons_t mirror,
    void* glial_integration)
{
    if (!mirror) {
        return false;
    }

    mirror->glial_integration = glial_integration;

    if (glial_integration) {
        MIRROR_LOG_INFO("Mirror neurons: integrated with glial cell system (astrocytes, oligodendrocytes, microglia)");

        if (mirror->config.enable_astrocytes) {
            MIRROR_LOG_INFO("  - Astrocytes enabled: modulate association strength");
        }
        if (mirror->config.enable_oligodendrocytes) {
            MIRROR_LOG_INFO("  - Oligodendrocytes enabled: speed up recognition");
        }
        if (mirror->config.enable_microglia) {
            MIRROR_LOG_INFO("  - Microglia enabled: prune weak associations");
        }
    }

    return true;
}

//=============================================================================
// Bidirectional Feedback Functions (Phase 10.11.3)
//=============================================================================

/**
 * @brief Get social salience for current context
 *
 * WHAT: Query importance of social cues
 * WHY:  Visual cortex can boost attention to social stimuli
 * HOW:  Return agent detection confidence × observation activation
 *
 * BIOLOGY: STS (superior temporal sulcus) modulates V1 for social stimuli
 *          High social salience → enhanced processing of faces, biological motion
 *
 * COMPLEXITY: O(n) where n = number of actions
 *
 * @param mirror Mirror neuron system
 * @return Social salience [0, 1]
 */
float mirror_neurons_get_social_salience(mirror_neurons_t mirror)
{
    // Guard: Validate mirror system
    if (!mirror || !mirror->initialized) {
        return 0.0F;
    }

    // WHAT: Compute average activation across all mirror neurons
    // WHY:  High activation suggests social context is salient
    // HOW:  Average observation activation across neurons
    float total_activation = 0.0F;
    uint32_t active_count = 0;

    for (uint32_t i = 0; i < mirror->num_neurons; i++) {
        mirror_neuron_unit_t* neuron = &mirror->neurons[i];
        if (neuron->observation_activation > 0.01F) {
            total_activation += neuron->observation_activation;
            active_count++;
        }
    }

    if (active_count == 0) {
        return 0.0F;
    }

    float avg_activation = total_activation / active_count;

    // Scale by number of observed agents
    float agent_factor = (mirror->num_agents > 0) ? 1.0F : 0.5F;

    // Combine activation and agent presence
    float social_salience = avg_activation * agent_factor;

    // Clamp to [0, 1]
    return fminf(fmaxf(social_salience, 0.0F), 1.0F);
}

/**
 * @brief Activate observation mode
 *
 * WHAT: Signal that agent detected, prepare for observation learning
 * WHY:  Visual detection of agents triggers mirror neuron activation
 * HOW:  Prime observation pathways for incoming action features
 *
 * COMPLEXITY: O(1)
 *
 * @param mirror Mirror neuron system
 */
void mirror_neurons_activate_observation_mode(mirror_neurons_t mirror)
{
    // Guard: Validate mirror system
    if (!mirror || !mirror->initialized) {
        MIRROR_LOG_ERROR("Mirror neurons: invalid system in activate_observation_mode");
        return;
    }

    // WHAT: Boost baseline activation for all neurons
    // WHY:  Prepare for incoming social observation
    // HOW:  Small activation boost to make system more sensitive
    for (uint32_t i = 0; i < mirror->num_neurons; i++) {
        mirror_neuron_unit_t* neuron = &mirror->neurons[i];
        if (neuron->action_id != 0) {  // Skip unassigned neurons
            neuron->observation_activation = fminf(
                neuron->observation_activation + 0.1F,  // Small boost
                1.0F
            );
        }
    }

    MIRROR_LOG_INFO("Mirror neurons: observation mode activated (primed %u neurons)",
                   mirror->num_neurons);
}

/**
 * @brief Check for recent observations
 *
 * WHAT: Determine if observations occurred recently
 * WHY:  Enable Theory of Mind predictions
 * HOW:  Compare last update time with current time
 *
 * COMPLEXITY: O(1)
 */
bool mirror_neurons_has_recent_observations(mirror_neurons_t mirror)
{
    // Guard: Validate mirror system
    if (!mirror) {
        return false;
    }

    // Check if never updated
    if (mirror->last_update_time == 0) {
        return false;
    }

    // Check if observation within last 5 seconds (5000ms)
    uint64_t current_time = nimcp_time_get_ms();
    uint64_t time_since_last = current_time - mirror->last_update_time;

    return time_since_last < 5000;
}

/**
 * @brief Get all mirror neuron activations for ToM integration
 *
 * WHAT: Extract activation patterns across all mirror neurons
 * WHY:  Enable Theory of Mind to infer intentions from observed actions
 * HOW:  Aggregate observation + execution activations per action
 *
 * BIOLOGICAL RATIONALE:
 * Mirror neurons encode both observed and executed actions, enabling
 * understanding of others' intentions (Rizzolatti & Craighero, 2004).
 * ToM uses this shared representation to infer "why" from "what."
 *
 * COMPLEXITY: O(n) where n = num_actions
 */
bool mirror_neurons_get_all_activations(
    mirror_neurons_t mirror,
    float* activations,
    uint32_t max_size,
    uint32_t* out_size)
{
    // Guard: Validate inputs
    if (!mirror || !activations || !out_size) {
        if (out_size) *out_size = 0;
        return false;
    }

    // Guard: Check initialized
    if (!mirror->initialized) {
        MIRROR_LOG_ERROR("Mirror neurons: system not initialized");
        *out_size = 0;
        return false;
    }

    // WHAT: Compute combined activation for each action
    // WHY:  ToM needs overall action strength, not separate obs/exec
    // HOW:  Average observation and execution activations per action

    uint32_t count = 0;
    for (uint32_t i = 0; i < mirror->num_actions && count < max_size; i++) {
        action_mapping_t* action = &mirror->actions[i];

        // Sum activations across all neurons for this action
        float obs_activation = 0.0F;
        float exec_activation = 0.0F;
        uint32_t neuron_count = 0;

        for (uint32_t j = 0; j < action->num_neurons; j++) {
            uint32_t neuron_idx = action->neuron_indices[j];
            if (neuron_idx < mirror->num_neurons) {
                mirror_neuron_unit_t* neuron = &mirror->neurons[neuron_idx];
                obs_activation += neuron->observation_activation;
                exec_activation += neuron->execution_activation;
                neuron_count++;
            }
        }

        // Average and combine (observation + execution weighted)
        // Higher weight on observation since ToM focuses on observing others
        if (neuron_count > 0) {
            float avg_obs = obs_activation / neuron_count;
            float avg_exec = exec_activation / neuron_count;
            // Weight: 70% observation, 30% execution (empathy focus)
            activations[count] = 0.7F * avg_obs + 0.3F * avg_exec;
            count++;
        }
    }

    *out_size = count;
    return true;
}

//=============================================================================
// Persistence API (Save/Load) - Phase 10.11
//=============================================================================

/**
 * @brief Save mirror neuron system state to file
 *
 * WHAT: Serialize mirror neuron system to binary file
 * WHY:  Enable persistence of learned action associations and statistics
 * HOW:  Write version, config, neurons, actions, agents, and statistics
 *
 * Binary format:
 *   uint32_t version (1)
 *   mirror_neuron_config_t config
 *   uint32_t num_neurons
 *   For each neuron:
 *     mirror_neuron_unit_t neuron (excluding integration pointers)
 *   uint32_t num_actions
 *   For each action:
 *     action_mapping_t action (with neuron_indices array)
 *   uint32_t num_agents
 *   For each agent:
 *     agent_info_t agent
 *   mirror_neuron_stats_t stats
 *   uint64_t creation_time
 *   uint64_t last_update_time
 *
 * Note: Integration handles (working_memory, theory_of_mind, etc.) are not saved
 *       They must be re-established after loading
 *
 * COMPLEXITY: O(n + a + g) where n=neurons, a=actions, g=agents
 * THREAD-SAFE: No (caller must ensure exclusive access)
 *
 * @param mirror Mirror neuron system
 * @param file Open file handle for writing
 * @return true on success, false on error
 */
bool mirror_neurons_save(mirror_neurons_t mirror, FILE* file)
{
    // Guard: Validate parameters
    if (!mirror || !file) {
        return false;
    }

    // WHAT: Write version marker for backward compatibility
    // WHY:  Enable future format changes while supporting old saves
    // HOW:  Write uint32_t version = 1
    uint32_t version = 1;
    if (fwrite(&version, sizeof(uint32_t), 1, file) != 1) {
        return false;
    }

    // WHAT: Write configuration
    // WHY:  Restore mirror neuron behavior on load
    // HOW:  Binary write of config struct
    if (fwrite(&mirror->config, sizeof(mirror_neuron_config_t), 1, file) != 1) {
        return false;
    }

    // WHAT: Write neuron count and neurons
    // WHY:  Restore learned neuron representations
    // HOW:  Write count, then each neuron unit
    if (fwrite(&mirror->num_neurons, sizeof(uint32_t), 1, file) != 1) {
        return false;
    }

    for (uint32_t i = 0; i < mirror->num_neurons; i++) {
        if (fwrite(&mirror->neurons[i], sizeof(mirror_neuron_unit_t), 1, file) != 1) {
            return false;
        }
    }

    // WHAT: Write action mappings
    // WHY:  Restore action-to-neuron associations
    // HOW:  Write count, then each action with its neuron indices
    if (fwrite(&mirror->num_actions, sizeof(uint32_t), 1, file) != 1) {
        return false;
    }

    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        action_mapping_t* action = &mirror->actions[i];

        // Write action metadata (excluding neuron_indices pointer)
        if (fwrite(&action->action_id, sizeof(uint32_t), 1, file) != 1) return false;
        if (fwrite(action->action_name, sizeof(char), 64, file) != 64) return false;
        if (fwrite(&action->num_neurons, sizeof(uint32_t), 1, file) != 1) return false;
        if (fwrite(&action->capacity, sizeof(uint32_t), 1, file) != 1) return false;
        if (fwrite(&action->total_observations, sizeof(uint32_t), 1, file) != 1) return false;
        if (fwrite(&action->total_executions, sizeof(uint32_t), 1, file) != 1) return false;
        if (fwrite(&action->avg_similarity, sizeof(float), 1, file) != 1) return false;

        // Write neuron indices array
        if (action->num_neurons > 0 && action->neuron_indices) {
            if (fwrite(action->neuron_indices, sizeof(uint32_t), action->num_neurons, file) != action->num_neurons) {
                return false;
            }
        }
    }

    // WHAT: Write agent tracking data
    // WHY:  Restore multi-agent learning state
    // HOW:  Write count, then each agent info
    if (fwrite(&mirror->num_agents, sizeof(uint32_t), 1, file) != 1) {
        return false;
    }

    for (uint32_t i = 0; i < mirror->num_agents; i++) {
        if (fwrite(&mirror->agents[i], sizeof(agent_info_t), 1, file) != 1) {
            return false;
        }
    }

    // WHAT: Write statistics
    // WHY:  Preserve performance metrics
    // HOW:  Binary write of stats struct
    if (fwrite(&mirror->stats, sizeof(mirror_neuron_stats_t), 1, file) != 1) {
        return false;
    }

    // WHAT: Write temporal state
    // WHY:  Restore timing information
    // HOW:  Write creation_time and last_update_time
    if (fwrite(&mirror->creation_time, sizeof(uint64_t), 1, file) != 1) {
        return false;
    }

    if (fwrite(&mirror->last_update_time, sizeof(uint64_t), 1, file) != 1) {
        return false;
    }

    return true;
}

/**
 * @brief Load mirror neuron system state from file
 *
 * WHAT: Deserialize mirror neuron system from binary file
 * WHY:  Restore saved action associations and learning state
 * HOW:  Read version, validate, reconstruct state
 *
 * Note: Integration handles must be set separately via integration functions
 * Note: Brain reference must be set via mirror_neurons_set_brain()
 *
 * COMPLEXITY: O(n + a + g) where n=neurons, a=actions, g=agents
 * THREAD-SAFE: Yes (creates new instance)
 *
 * @param file Open file handle for reading
 * @return Mirror neuron system handle or NULL on error
 */
mirror_neurons_t mirror_neurons_load(FILE* file)
{
    // Guard: Validate parameter
    if (!file) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "file is NULL");

        return NULL;
    }

    // WHAT: Read and validate version
    // WHY:  Ensure format compatibility
    // HOW:  Read version, check against current version
    uint32_t version = 0;
    if (fread(&version, sizeof(uint32_t), 1, file) != 1) {
        return NULL;
    }

    if (version != 1) {
        return NULL;
    }

    // WHAT: Allocate mirror neuron system structure
    // WHY:  Need structure to hold loaded data
    // HOW:  Use nimcp_calloc for zero-initialization
    mirror_neurons_t mirror = (mirror_neurons_t)nimcp_calloc(1, sizeof(struct mirror_neurons_system));
    if (!mirror) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "mirror is NULL");

        return NULL;
    }

    // WHAT: Read configuration
    // WHY:  Restore mirror neuron behavior
    // HOW:  Binary read into config struct
    if (fread(&mirror->config, sizeof(mirror_neuron_config_t), 1, file) != 1) {
        goto cleanup;
    }

    // WHAT: Read neuron count and allocate neurons
    // WHY:  Restore learned neuron representations
    // HOW:  Read count, allocate array, read each neuron
    if (fread(&mirror->num_neurons, sizeof(uint32_t), 1, file) != 1) {
        goto cleanup;
    }

    mirror->neurons = (mirror_neuron_unit_t*)nimcp_calloc(mirror->num_neurons, sizeof(mirror_neuron_unit_t));
    if (!mirror->neurons) {
        goto cleanup;
    }

    for (uint32_t i = 0; i < mirror->num_neurons; i++) {
        if (fread(&mirror->neurons[i], sizeof(mirror_neuron_unit_t), 1, file) != 1) {
            goto cleanup;
        }
    }

    // WHAT: Read action mappings
    // WHY:  Restore action-to-neuron associations
    // HOW:  Read count, allocate array, read each action with indices
    if (fread(&mirror->num_actions, sizeof(uint32_t), 1, file) != 1) {
        goto cleanup;
    }

    mirror->actions_capacity = mirror->num_actions;
    mirror->actions = (action_mapping_t*)nimcp_calloc(mirror->actions_capacity, sizeof(action_mapping_t));
    if (!mirror->actions) {
        goto cleanup;
    }

    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        action_mapping_t* action = &mirror->actions[i];

        // Read action metadata
        if (fread(&action->action_id, sizeof(uint32_t), 1, file) != 1) goto cleanup;
        if (fread(action->action_name, sizeof(char), 64, file) != 64) goto cleanup;
        if (fread(&action->num_neurons, sizeof(uint32_t), 1, file) != 1) goto cleanup;
        if (fread(&action->capacity, sizeof(uint32_t), 1, file) != 1) goto cleanup;
        if (fread(&action->total_observations, sizeof(uint32_t), 1, file) != 1) goto cleanup;
        if (fread(&action->total_executions, sizeof(uint32_t), 1, file) != 1) goto cleanup;
        if (fread(&action->avg_similarity, sizeof(float), 1, file) != 1) goto cleanup;

        // Allocate and read neuron indices array
        if (action->num_neurons > 0) {
            action->neuron_indices = (uint32_t*)nimcp_calloc(action->capacity, sizeof(uint32_t));
            if (!action->neuron_indices) {
                goto cleanup;
            }

            if (fread(action->neuron_indices, sizeof(uint32_t), action->num_neurons, file) != action->num_neurons) {
                goto cleanup;
            }
        } else {
            action->neuron_indices = NULL;
        }
    }

    // WHAT: Read agent tracking data
    // WHY:  Restore multi-agent learning state
    // HOW:  Read count, allocate array, read each agent
    if (fread(&mirror->num_agents, sizeof(uint32_t), 1, file) != 1) {
        goto cleanup;
    }

    mirror->agents_capacity = mirror->num_agents;
    mirror->agents = (agent_info_t*)nimcp_calloc(mirror->agents_capacity, sizeof(agent_info_t));
    if (!mirror->agents) {
        goto cleanup;
    }

    for (uint32_t i = 0; i < mirror->num_agents; i++) {
        if (fread(&mirror->agents[i], sizeof(agent_info_t), 1, file) != 1) {
            goto cleanup;
        }
    }

    // WHAT: Read statistics
    // WHY:  Restore performance metrics
    // HOW:  Binary read into stats struct
    if (fread(&mirror->stats, sizeof(mirror_neuron_stats_t), 1, file) != 1) {
        goto cleanup;
    }

    // WHAT: Read temporal state
    // WHY:  Restore timing information
    // HOW:  Read creation_time and last_update_time
    if (fread(&mirror->creation_time, sizeof(uint64_t), 1, file) != 1) {
        goto cleanup;
    }

    if (fread(&mirror->last_update_time, sizeof(uint64_t), 1, file) != 1) {
        goto cleanup;
    }

    // WHAT: Initialize integration handles to NULL
    // WHY:  Must be set separately by caller
    // HOW:  Zero-initialization already done by calloc
    mirror->working_memory = NULL;
    mirror->theory_of_mind = NULL;
    mirror->predictive_network = NULL;
    mirror->glial_integration = NULL;
    mirror->brain = NULL;

    mirror->initialized = true;

    return mirror;

cleanup:
    // WHAT: Cleanup on error
    // WHY:  Prevent memory leaks
    // HOW:  Free allocated resources
    if (mirror) {
        if (mirror->neurons) {
            nimcp_free(mirror->neurons);
        }
        if (mirror->actions) {
            for (uint32_t i = 0; i < mirror->num_actions; i++) {
                if (mirror->actions[i].neuron_indices) {
                    nimcp_free(mirror->actions[i].neuron_indices);
                }
            }
            nimcp_free(mirror->actions);
        }
        if (mirror->agents) {
            nimcp_free(mirror->agents);
        }
        nimcp_free(mirror);
    }
    return NULL;
}

//=============================================================================
// Phase 10.11.2: Substrate Integration API Implementation
//=============================================================================

/**
 * @brief Enable substrate integration for mirror neurons
 *
 * WHAT: Enable biological substrate backing for all mirror neuron units
 * WHY:  Provides myelination timing, dendrite plasticity, glial modulation
 * HOW:  Create memory pool and substrate backings for each unit
 */
bool mirror_neurons_enable_substrate(mirror_neurons_t mirror)
{
    if (!mirror) return false;

    /* Already enabled? */
    if (mirror->substrate_enabled) {
        MIRROR_LOG_INFO("Mirror neurons: substrate already enabled");
        return true;
    }

    /* Initialize substrate config from mirror config */
    mirror->substrate_config = mirror_substrate_get_default_config();
    mirror->substrate_config.enable_myelination = mirror->config.enable_myelination;
    mirror->substrate_config.enable_dendrites = mirror->config.enable_dendrite_plasticity;
    mirror->substrate_config.enable_axons = mirror->config.enable_axon_timing;
    mirror->substrate_config.enable_astrocytes = mirror->config.enable_astrocytes;
    mirror->substrate_config.enable_oligodendrocytes = mirror->config.enable_oligodendrocytes;
    mirror->substrate_config.enable_microglia = mirror->config.enable_microglia;
    mirror->substrate_config.enable_memory_pool = mirror->config.enable_substrate_pool;
    mirror->substrate_config.pool_capacity = mirror->config.substrate_pool_size;

    /* Create memory pool if enabled */
    if (mirror->config.enable_substrate_pool) {
        mirror->substrate_pool = mirror_substrate_pool_create(
            mirror->config.substrate_pool_size);
        if (!mirror->substrate_pool) {
            MIRROR_LOG_ERROR("Mirror neurons: failed to create substrate pool");
            return false;
        }
    }

    /* Create substrate backings for all neurons */
    for (uint32_t i = 0; i < mirror->num_neurons; i++) {
        mirror_neuron_unit_t* unit = &mirror->neurons[i];

        unit->substrate = mirror_substrate_backing_create(
            unit->neuron_id,
            &mirror->substrate_config,
            mirror->substrate_pool);

        if (unit->substrate) {
            unit->has_substrate = true;
        } else {
            MIRROR_LOG_WARN("Mirror neurons: failed to create substrate for unit %u", i);
        }
    }

    mirror->substrate_enabled = true;
    MIRROR_LOG_INFO("Mirror neurons: substrate enabled for %u units", mirror->num_neurons);

    return true;
}

/**
 * @brief Connect substrate to axon network
 */
bool mirror_neurons_connect_axon_network(
    mirror_neurons_t mirror,
    void* axon_network)
{
    if (!mirror) return false;

    mirror->axon_network = axon_network;

    if (axon_network) {
        nimcp_result_t result = mirror_substrate_connect_axon_network(
            mirror, axon_network);
        return result == NIMCP_SUCCESS;
    }

    return true;
}

/**
 * @brief Connect substrate to dendrite network
 */
bool mirror_neurons_connect_dendrite_network(
    mirror_neurons_t mirror,
    void* dendrite_network)
{
    if (!mirror) return false;

    mirror->dendrite_network = dendrite_network;

    if (dendrite_network) {
        nimcp_result_t result = mirror_substrate_connect_dendrite_network(
            mirror, dendrite_network);
        return result == NIMCP_SUCCESS;
    }

    return true;
}

/**
 * @brief Connect substrate to myelin sheath network
 */
bool mirror_neurons_connect_myelin_network(
    mirror_neurons_t mirror,
    void* myelin_network)
{
    if (!mirror) return false;

    mirror->myelin_network = myelin_network;

    if (myelin_network) {
        nimcp_result_t result = mirror_substrate_connect_myelin_network(
            mirror, myelin_network);
        return result == NIMCP_SUCCESS;
    }

    return true;
}

/**
 * @brief Get recognition delay for action (substrate-aware)
 *
 * WHAT: Calculate delay for action recognition including substrate effects
 * WHY:  Myelination and axon properties affect recognition speed
 * HOW:  Query substrate backing for delay, fall back to base if no substrate
 */
float mirror_neurons_get_recognition_delay(mirror_neurons_t mirror, uint32_t action_id)
{
    if (!mirror) return NIMCP_MIRROR_BASE_DELAY_MS;

    /* Find action mapping */
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        if (mirror->actions[i].action_id == action_id) {
            /* Get average delay across neurons for this action */
            float total_delay = 0.0F;
            uint32_t count = 0;

            for (uint32_t j = 0; j < mirror->actions[i].num_neurons; j++) {
                uint32_t neuron_idx = mirror->actions[i].neuron_indices[j];
                if (neuron_idx < mirror->num_neurons) {
                    mirror_neuron_unit_t* unit = &mirror->neurons[neuron_idx];

                    if (unit->has_substrate && unit->substrate) {
                        total_delay += mirror_substrate_get_observation_delay(unit->substrate);
                    } else {
                        total_delay += NIMCP_MIRROR_BASE_DELAY_MS;
                    }
                    count++;
                }
            }

            return (count > 0) ? (total_delay / count) : NIMCP_MIRROR_BASE_DELAY_MS;
        }
    }

    return NIMCP_MIRROR_BASE_DELAY_MS;
}

/**
 * @brief Get association strength from spine weights
 *
 * WHAT: Query observation-execution association based on spine plasticity
 * WHY:  Spines encode learned associations in substrate mode
 * HOW:  Sum spine weights for the action's mirror units
 */
float mirror_neurons_get_spine_association(mirror_neurons_t mirror, uint32_t action_id)
{
    if (!mirror || !mirror->substrate_enabled) return 0.0F;

    /* Find action mapping */
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        if (mirror->actions[i].action_id == action_id) {
            float total_weight = 0.0F;

            for (uint32_t j = 0; j < mirror->actions[i].num_neurons; j++) {
                uint32_t neuron_idx = mirror->actions[i].neuron_indices[j];
                if (neuron_idx < mirror->num_neurons) {
                    mirror_neuron_unit_t* unit = &mirror->neurons[neuron_idx];

                    if (unit->has_substrate && unit->substrate) {
                        total_weight += mirror_substrate_get_total_spine_weight(unit->substrate);
                    }
                }
            }

            return total_weight;
        }
    }

    return 0.0F;
}

/**
 * @brief Step substrate simulation forward
 *
 * WHAT: Advance all substrate states by one timestep
 * WHY:  Keep substrate synchronized with simulation time
 * HOW:  Update myelination, spine plasticity, glial states for each unit
 */
bool mirror_neurons_step_substrate(mirror_neurons_t mirror, float dt_ms)
{
    if (!mirror || !mirror->substrate_enabled) return false;

    uint64_t current_time = nimcp_time_get_us();
    float dt_seconds = dt_ms / 1000.0F;

    /* Step each neuron's substrate */
    for (uint32_t i = 0; i < mirror->num_neurons; i++) {
        mirror_neuron_unit_t* unit = &mirror->neurons[i];

        if (unit->has_substrate && unit->substrate) {
            mirror_substrate_step(unit->substrate, current_time, dt_seconds);
        }
    }

    return true;
}

/**
 * @brief Check if substrate is enabled
 */
bool mirror_neurons_has_substrate(mirror_neurons_t mirror)
{
    if (!mirror) return false;
    return mirror->substrate_enabled;
}

//=============================================================================
// Enhancement Systems Implementation (Phase 10.11.4-6)
//=============================================================================

/**
 * @brief Enable STDP learning for mirror neurons
 */
bool mirror_neurons_enable_stdp(mirror_neurons_t mirror, uint32_t max_synapses)
{
    if (!mirror) return false;

    // Already enabled?
    if (mirror->stdp_enabled && mirror->stdp_system) {
        return true;
    }

    // Use max_actions if max_synapses not specified
    if (max_synapses == 0) {
        max_synapses = mirror->config.max_actions;
    }

    // Create STDP system with default configuration
    mirror_stdp_t stdp = mirror_stdp_create(NULL, max_synapses);
    if (!stdp) {
        MIRROR_LOG_ERROR("Failed to create STDP system\n");
        return false;
    }

    // Create synapses for existing actions
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        uint32_t action_id = mirror->actions[i].action_id;
        mirror_stdp_create_synapse(stdp, action_id, 0.5F);  // Initial weight 0.5
    }

    mirror->stdp_system = stdp;
    mirror->stdp_enabled = true;

    MIRROR_LOG_INFO("STDP learning enabled with %u synapses\n", max_synapses);
    return true;
}

/**
 * @brief Get STDP system handle
 */
mirror_stdp_t mirror_neurons_get_stdp(mirror_neurons_t mirror)
{
    if (!mirror || !mirror->stdp_enabled) return NULL;
    return (mirror_stdp_t)mirror->stdp_system;
}

/**
 * @brief Set dopamine level for STDP learning
 */
void mirror_neurons_set_stdp_dopamine(mirror_neurons_t mirror, float level)
{
    if (!mirror || !mirror->stdp_enabled || !mirror->stdp_system) return;

    mirror_stdp_set_dopamine((mirror_stdp_t)mirror->stdp_system, level);
}

/**
 * @brief Enable motor resonance for mirror neurons
 */
bool mirror_neurons_enable_resonance(mirror_neurons_t mirror, uint32_t max_channels)
{
    if (!mirror) return false;

    // Already enabled?
    if (mirror->resonance_enabled && mirror->resonance_system) {
        return true;
    }

    // Use max_actions if max_channels not specified
    if (max_channels == 0) {
        max_channels = mirror->config.max_actions;
    }

    // Create resonance system with default configuration
    motor_resonance_t resonance = motor_resonance_create(NULL, max_channels);
    if (!resonance) {
        MIRROR_LOG_ERROR("Failed to create motor resonance system\n");
        return false;
    }

    // Create channels for existing actions
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        uint32_t action_id = mirror->actions[i].action_id;
        motor_resonance_create_channel(resonance, action_id);
    }

    mirror->resonance_system = resonance;
    mirror->resonance_enabled = true;

    MIRROR_LOG_INFO("Motor resonance enabled with %u channels\n", max_channels);
    return true;
}

/**
 * @brief Get motor resonance system handle
 */
motor_resonance_t mirror_neurons_get_resonance(mirror_neurons_t mirror)
{
    if (!mirror || !mirror->resonance_enabled) return NULL;
    return (motor_resonance_t)mirror->resonance_system;
}

/**
 * @brief Set learning context for motor resonance
 */
void mirror_neurons_set_learning_context(mirror_neurons_t mirror, float learning_strength)
{
    if (!mirror || !mirror->resonance_enabled || !mirror->resonance_system) return;

    motor_resonance_release_for_learning((motor_resonance_t)mirror->resonance_system,
                                          -1, learning_strength);
}

/**
 * @brief Set social context for motor resonance
 */
void mirror_neurons_set_social_context(mirror_neurons_t mirror, float social_strength)
{
    if (!mirror || !mirror->resonance_enabled || !mirror->resonance_system) return;

    motor_resonance_release_for_social((motor_resonance_t)mirror->resonance_system,
                                        -1, social_strength);
}

/**
 * @brief Check if action is above execution threshold
 */
bool mirror_neurons_should_imitate(mirror_neurons_t mirror, uint32_t action_id)
{
    if (!mirror || !mirror->resonance_enabled || !mirror->resonance_system) return false;

    motor_resonance_t resonance = (motor_resonance_t)mirror->resonance_system;
    uint32_t channel_id = motor_resonance_find_channel(resonance, action_id);

    if (channel_id == UINT32_MAX) return false;

    return motor_resonance_above_threshold(resonance, channel_id);
}

/**
 * @brief Enable hierarchical goal representation
 */
bool mirror_neurons_enable_hierarchy(mirror_neurons_t mirror)
{
    if (!mirror) return false;

    // Already enabled?
    if (mirror->hierarchy_enabled && mirror->hierarchy_system) {
        return true;
    }

    // Create hierarchy system with default configuration
    mirror_hierarchy_t hierarchy = mirror_hierarchy_create(NULL);
    if (!hierarchy) {
        MIRROR_LOG_ERROR("Failed to create hierarchy system\n");
        return false;
    }

    // Create motor representations for existing actions
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        mirror_hierarchy_create_motor(hierarchy, mirror->actions[i].action_name,
                                       MOTOR_TYPE_UNKNOWN);
    }

    mirror->hierarchy_system = hierarchy;
    mirror->hierarchy_enabled = true;

    MIRROR_LOG_INFO("Hierarchical goal representation enabled\n");
    return true;
}

/**
 * @brief Get hierarchy system handle
 */
mirror_hierarchy_t mirror_neurons_get_hierarchy(mirror_neurons_t mirror)
{
    if (!mirror || !mirror->hierarchy_enabled) return NULL;
    return (mirror_hierarchy_t)mirror->hierarchy_system;
}

/**
 * @brief Infer goal from observed action
 */
bool mirror_neurons_infer_goal(mirror_neurons_t mirror, uint32_t action_id,
                                uint32_t* out_goal, float* out_confidence)
{
    if (!mirror || !mirror->hierarchy_enabled || !mirror->hierarchy_system) return false;
    if (!out_goal || !out_confidence) return false;

    mirror_hierarchy_t hierarchy = (mirror_hierarchy_t)mirror->hierarchy_system;

    // Find motor representation for this action
    uint32_t motor_id = UINT32_MAX;
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        if (mirror->actions[i].action_id == action_id) {
            motor_id = i;  // Motor ID corresponds to action index
            break;
        }
    }

    if (motor_id == UINT32_MAX) return false;

    // Infer goal
    uint32_t goal_ids[4];
    float probs[4];
    uint32_t num_goals = mirror_hierarchy_infer_goal(hierarchy, motor_id,
                                                      goal_ids, probs, 4);

    if (num_goals == 0) return false;

    *out_goal = goal_ids[0];
    *out_confidence = probs[0];

    return true;
}

/**
 * @brief Select goal for top-down motor control
 */
void mirror_neurons_select_goal(mirror_neurons_t mirror, int32_t goal_id)
{
    if (!mirror || !mirror->hierarchy_enabled || !mirror->hierarchy_system) return;

    mirror_hierarchy_select_goal((mirror_hierarchy_t)mirror->hierarchy_system, goal_id);
}

/**
 * @brief Step all enhancement systems
 */
bool mirror_neurons_step_enhancements(mirror_neurons_t mirror, float dt_ms)
{
    if (!mirror) return false;

    // Step STDP system
    if (mirror->stdp_enabled && mirror->stdp_system) {
        mirror_stdp_step((mirror_stdp_t)mirror->stdp_system, dt_ms);
    }

    // Step resonance system
    if (mirror->resonance_enabled && mirror->resonance_system) {
        motor_resonance_step((motor_resonance_t)mirror->resonance_system, dt_ms);
    }

    // Step hierarchy system
    if (mirror->hierarchy_enabled && mirror->hierarchy_system) {
        mirror_hierarchy_step((mirror_hierarchy_t)mirror->hierarchy_system, dt_ms);
    }

    return true;
}

//=============================================================================
// KG Self-Awareness Integration
//=============================================================================

int mirror_neurons_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Mirror_Neurons");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            MIRROR_LOG_INFO("Mirror neurons self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Mirror_Neurons");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Mirror_Neurons");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
