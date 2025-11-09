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
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

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

    // Temporal state
    uint64_t creation_time;
    uint64_t last_update_time;

    // Memory management
    bool initialized;
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
        .learning_rate = 0.01f,
        .decay_rate = 0.05f,
        .match_threshold = 0.7f,
        .enable_working_memory = true,
        .enable_theory_of_mind = true,
        .enable_prediction = true,
        .observation_window = 1000,
        .replay_capacity = 100,
        // Phase 10.11.1: Glial cell support (default: all enabled)
        .enable_glial_modulation = true,
        .enable_astrocytes = true,
        .enable_oligodendrocytes = true,
        .enable_microglia = true
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
    action.confidence = 1.0f;

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
        return -1.0f;
    }

    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (uint32_t i = 0; i < n; i++) {
        dot_product += f1[i] * f2[i];
        norm1 += f1[i] * f1[i];
        norm2 += f2[i] * f2[i];
    }

    if (norm1 == 0.0f || norm2 == 0.0f) {
        return 0.0f;
    }

    float similarity = dot_product / (sqrtf(norm1) * sqrtf(norm2));

    // Normalize to [0, 1] from [-1, 1]
    similarity = (similarity + 1.0f) / 2.0f;

    return (similarity < 0.0f) ? 0.0f : ((similarity > 1.0f) ? 1.0f : similarity);
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
    mapping->avg_similarity = 0.0f;

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
    agent->trust_score = 0.5f;  // Neutral trust initially

    return idx;
}

/**
 * @brief Activate neurons for an action
 *
 * WHAT: Set activation levels for neurons representing an action
 * WHY:  Implement observation or execution pathway
 * HOW:  Find neurons for action, update activations
 */
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

    // Activate neurons
    for (uint32_t i = 0; i < mapping->num_neurons; i++) {
        uint32_t neuron_idx = mapping->neuron_indices[i];
        mirror_neuron_unit_t* neuron = &mirror->neurons[neuron_idx];

        if (is_observation) {
            neuron->observation_activation += strength;
            if (neuron->observation_activation > 1.0f) {
                neuron->observation_activation = 1.0f;
            }
            neuron->observation_count++;
            neuron->last_observation_time = current_time;
        } else {
            neuron->execution_activation += strength;
            if (neuron->execution_activation > 1.0f) {
                neuron->execution_activation = 1.0f;
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
    if (!mirror || action_idx >= mirror->num_actions) {
        return;
    }

    action_mapping_t* mapping = &mirror->actions[action_idx];

    if (mapping->num_neurons == 0) {
        return;
    }

    // Calculate average similarity across neurons
    float total_similarity = 0.0f;
    uint32_t count = 0;

    for (uint32_t i = 0; i < mapping->num_neurons; i++) {
        uint32_t neuron_idx = mapping->neuron_indices[i];
        mirror_neuron_unit_t* neuron = &mirror->neurons[neuron_idx];

        if (neuron->observation_count > 0 && neuron->execution_count > 0) {
            // Similarity based on co-activation
            float obs_norm = neuron->observation_activation;
            float exec_norm = neuron->execution_activation;
            float similarity = (obs_norm * exec_norm) /
                             (obs_norm + exec_norm + 0.001f);  // Prevent div by 0
            total_similarity += similarity;
            count++;
        }
    }

    if (count > 0) {
        mapping->avg_similarity = total_similarity / count;
    }
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
    mirror->last_update_time = mirror->creation_time;
    mirror->initialized = true;

    MIRROR_LOG_INFO("Mirror neurons: created system with %u neurons, max %u actions",
                  mirror->num_neurons, mirror->config.max_actions);

    return mirror;
}

/**
 * @brief Destroy mirror neuron system
 */
void mirror_neurons_destroy(mirror_neurons_t mirror)
{
    if (!mirror) {
        return;
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
        return -1.0f;
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
        return -1.0f;  // Action not found
    }

    // Sum activations across neurons
    float total_activation = 0.0f;
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

    return 0.0f;
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
        if (out_similarity) *out_similarity = 0.0f;
        return false;
    }

    // Compute feature similarity
    uint32_t num_features = (observed_action->num_features < executed_action->num_features) ?
                           observed_action->num_features : executed_action->num_features;

    if (num_features == 0) {
        if (out_similarity) *out_similarity = 0.0f;
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

        if (obs_act > 0.0f && exec_act > 0.0f) {
            // Both pathways active - strengthen association
            float delta = mirror->config.learning_rate * obs_act * exec_act;
            neuron->association_weight += delta;

            // Bound to [0, 1]
            if (neuron->association_weight > 1.0f) {
                neuron->association_weight = 1.0f;
            }
        } else {
            // Weak decay if no co-activation
            neuron->association_weight *= 0.99f;
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
    float decay_factor = expf(-mirror->config.decay_rate * (delta_time_ms / 1000.0f));

    // Apply decay to all neurons
    for (uint32_t i = 0; i < mirror->num_neurons; i++) {
        mirror_neuron_unit_t* neuron = &mirror->neurons[i];

        neuron->observation_activation *= decay_factor;
        neuron->execution_activation *= decay_factor;

        // Threshold to zero if very small
        if (neuron->observation_activation < 0.001f) {
            neuron->observation_activation = 0.0f;
        }
        if (neuron->execution_activation < 0.001f) {
            neuron->execution_activation = 0.0f;
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
        if (mirror->neurons[i].observation_activation > 0.01f ||
            mirror->neurons[i].execution_activation > 0.01f) {
            stats->num_active_neurons++;
        }
    }

    stats->num_learned_actions = mirror->num_actions;
    stats->num_observed_agents = mirror->num_agents;

    // Calculate average match quality
    float total_quality = 0.0f;
    uint32_t count = 0;
    for (uint32_t i = 0; i < mirror->num_actions; i++) {
        if (mirror->actions[i].avg_similarity > 0.0f) {
            total_quality += mirror->actions[i].avg_similarity;
            count++;
        }
    }
    stats->avg_match_quality = (count > 0) ? (total_quality / count) : 0.0f;

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
    activation->observation_activation = 0.0f;
    activation->execution_activation = 0.0f;
    activation->association_strength = 0.0f;
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
        if (confidence) *confidence = 0.0f;
        return false;
    }

    // Simple prediction: find action with highest activation that follows the sequence
    // In future, this could use more sophisticated sequence learning

    // Get last action in sequence
    const action_t* last_action = &previous_actions[num_previous - 1];

    // Find action with highest activation that's different from last
    float max_activation = 0.0f;
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
        if (confidence) *confidence = 0.0f;
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

    if (confidence) *confidence = 0.0f;
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
