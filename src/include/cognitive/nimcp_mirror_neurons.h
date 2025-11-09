/**
 * @file nimcp_mirror_neurons.h
 * @brief Mirror Neuron System - Observation-based learning and social cognition
 * @version 1.0.0
 * @date 2025-01-09
 *
 * WHAT: Biologically-inspired mirror neuron system for imitation learning
 * WHY:  Enable learning by observation, social cognition, and action understanding
 * HOW:  Neurons that fire for both self-action and observed-action, with associative learning
 *
 * Mirror neurons are a class of neurons that activate both when:
 * 1. An agent performs an action (motor execution)
 * 2. An agent observes another performing the same action (motor observation)
 *
 * This dual representation enables:
 * - Imitation learning (learn by watching)
 * - Action understanding (predict others' goals)
 * - Empathy and theory of mind (understand others' mental states)
 * - Reduced training time (observation > trial-and-error)
 *
 * Key Features:
 * - Action representation in shared observation/execution space
 * - Association learning between observed and performed actions
 * - Integration with working memory for action replay
 * - Connection to theory of mind for intent understanding
 * - Multi-agent learning support
 *
 * Integration Points:
 * - Working Memory: Store observed action sequences
 * - Theory of Mind: Understand agent intentions
 * - Executive Functions: Plan imitation strategies
 * - Predictive Processing: Predict action outcomes
 *
 * @see Phase 10.11 - Mirror Neurons & Social Cognition
 */

#ifndef NIMCP_MIRROR_NEURONS_H
#define NIMCP_MIRROR_NEURONS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Forward Declarations
//=============================================================================

/**
 * @brief Opaque handle to mirror neuron system
 */
typedef struct mirror_neurons_system mirror_neurons_system_t;
typedef mirror_neurons_system_t* mirror_neurons_t;

//=============================================================================
// Configuration & Parameters
//=============================================================================

/**
 * @brief Mirror neuron configuration
 *
 * WHAT: Configuration for mirror neuron system creation
 * WHY:  Allow customization of mirror neuron behavior and capacity
 */
typedef struct {
    // System capacity
    uint32_t num_mirror_neurons;      /**< Number of mirror neurons (default: 1000) */
    uint32_t max_actions;             /**< Max distinct actions to track (default: 100) */
    uint32_t max_agents;              /**< Max agents to observe (default: 10) */

    // Learning parameters
    float learning_rate;              /**< Association learning rate (0.0-1.0, default: 0.01) */
    float decay_rate;                 /**< Activation decay rate (0.0-1.0, default: 0.05) */
    float match_threshold;            /**< Action matching threshold (0.0-1.0, default: 0.7) */

    // Integration flags
    bool enable_working_memory;       /**< Store observations in working memory (default: true) */
    bool enable_theory_of_mind;       /**< Infer agent intentions (default: true) */
    bool enable_prediction;           /**< Predict action outcomes (default: true) */

    // Performance tuning
    uint32_t observation_window;      /**< Time window for observation (ms, default: 1000) */
    uint32_t replay_capacity;         /**< Max stored observations for replay (default: 100) */

} mirror_neuron_config_t;

/**
 * @brief Action representation
 *
 * WHAT: Represents an action in the mirror neuron system
 * WHY:  Common representation for both observed and executed actions
 */
typedef struct {
    uint32_t action_id;               /**< Unique action identifier */
    char action_name[64];             /**< Human-readable action name */
    float features[32];               /**< Action feature vector */
    uint32_t num_features;            /**< Number of valid features */
    uint64_t timestamp;               /**< When action occurred */
    uint32_t agent_id;                /**< Which agent performed it (0 = self) */
    float confidence;                 /**< Confidence in action recognition (0.0-1.0) */
} action_t;

/**
 * @brief Mirror neuron activation record
 *
 * WHAT: Records activation pattern for observation-execution pair
 * WHY:  Track associations between observed and executed actions
 */
typedef struct {
    uint32_t action_id;               /**< Action that triggered activation */
    float observation_activation;     /**< Activation from observation */
    float execution_activation;       /**< Activation from execution */
    float association_strength;       /**< Learned association (0.0-1.0) */
    uint32_t observation_count;       /**< Times observed */
    uint32_t execution_count;         /**< Times executed */
    uint64_t last_activation;         /**< Last activation timestamp */
} mirror_activation_t;

/**
 * @brief Mirror neuron system statistics
 *
 * WHAT: Runtime statistics for mirror neuron system
 * WHY:  Monitor performance and learning progress
 */
typedef struct {
    // Capacity metrics
    uint32_t num_active_neurons;      /**< Currently active mirror neurons */
    uint32_t num_learned_actions;     /**< Number of distinct learned actions */
    uint32_t num_observed_agents;     /**< Number of agents observed */

    // Learning metrics
    uint32_t total_observations;      /**< Total observations processed */
    uint32_t total_executions;        /**< Total self-actions executed */
    uint32_t successful_matches;      /**< Successful observation-execution matches */
    float avg_match_quality;          /**< Average matching quality (0.0-1.0) */

    // Performance metrics
    float avg_observation_latency_ms; /**< Average observation processing time */
    float avg_execution_latency_ms;   /**< Average execution processing time */
    uint64_t last_update_time;        /**< Last system update timestamp */

    // Integration metrics
    uint32_t working_memory_stores;   /**< Observations stored in working memory */
    uint32_t theory_of_mind_inferences; /**< Intent inferences made */
    uint32_t predictions_made;        /**< Action predictions made */

} mirror_neuron_stats_t;

//=============================================================================
// Core API - Lifecycle Management
//=============================================================================

/**
 * @brief Create mirror neuron system
 *
 * WHAT: Initialize mirror neuron system with configuration
 * WHY:  Enable observation-based learning and social cognition
 * HOW:  Allocate neurons, initialize learning parameters, setup tracking
 *
 * COMPLEXITY: O(n) where n = num_mirror_neurons
 * THREAD-SAFE: Yes (creates new instance)
 *
 * @param config Configuration parameters (NULL = use defaults)
 * @return Mirror neuron system handle or NULL on error
 */
mirror_neurons_t mirror_neurons_create(const mirror_neuron_config_t* config);

/**
 * @brief Destroy mirror neuron system
 *
 * WHAT: Free all resources associated with mirror neuron system
 * WHY:  Prevent memory leaks
 * HOW:  Release neurons, clear observations, free memory
 *
 * COMPLEXITY: O(n) where n = num_mirror_neurons
 * THREAD-SAFE: No (caller must ensure exclusive access)
 *
 * @param mirror Mirror neuron system to destroy (NULL-safe)
 */
void mirror_neurons_destroy(mirror_neurons_t mirror);

//=============================================================================
// Core API - Action Processing
//=============================================================================

/**
 * @brief Process observed action from another agent
 *
 * WHAT: Activate mirror neurons in response to observing an action
 * WHY:  Learn from observation, build action associations
 * HOW:  Match action features to neurons, update activations, strengthen associations
 *
 * This is the core observation pathway. Mirror neurons will:
 * 1. Recognize the observed action
 * 2. Activate corresponding mirror neurons
 * 3. Update observation statistics
 * 4. Store in working memory (if enabled)
 * 5. Trigger theory of mind inference (if enabled)
 *
 * COMPLEXITY: O(n) where n = num_mirror_neurons
 * THREAD-SAFE: No (requires external synchronization)
 *
 * @param mirror Mirror neuron system
 * @param action Observed action
 * @return true on success, false on error
 */
bool mirror_neurons_observe_action(mirror_neurons_t mirror, const action_t* action);

/**
 * @brief Process self-executed action
 *
 * WHAT: Activate mirror neurons when agent performs an action
 * WHY:  Build execution representation, strengthen observation-execution associations
 * HOW:  Match action to neurons, update execution statistics, learn associations
 *
 * This is the core execution pathway. Mirror neurons will:
 * 1. Recognize the executed action
 * 2. Activate corresponding mirror neurons
 * 3. Update execution statistics
 * 4. Strengthen associations with previously observed similar actions
 *
 * COMPLEXITY: O(n) where n = num_mirror_neurons
 * THREAD-SAFE: No (requires external synchronization)
 *
 * @param mirror Mirror neuron system
 * @param action Executed action
 * @return true on success, false on error
 */
bool mirror_neurons_execute_action(mirror_neurons_t mirror, const action_t* action);

/**
 * @brief Get mirror neuron activation for specific action
 *
 * WHAT: Query current activation level for an action
 * WHY:  Determine how strongly an action is represented
 * HOW:  Sum activations across all neurons responding to action
 *
 * COMPLEXITY: O(n) where n = num_mirror_neurons
 * THREAD-SAFE: Yes (read-only)
 *
 * @param mirror Mirror neuron system
 * @param action_id Action to query
 * @return Activation level (0.0-1.0), or -1.0 on error
 */
float mirror_neurons_get_activation(mirror_neurons_t mirror, uint32_t action_id);

/**
 * @brief Match observed action to executed action
 *
 * WHAT: Determine if observed and executed actions are similar
 * WHY:  Enable action recognition and imitation learning
 * HOW:  Compare feature vectors, check association strength
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: Yes (read-only)
 *
 * @param mirror Mirror neuron system
 * @param observed_action Observed action
 * @param executed_action Executed action
 * @param out_similarity Pointer to store similarity score (0.0-1.0, can be NULL)
 * @return true if actions match (above threshold), false otherwise
 */
bool mirror_neurons_match_actions(
    mirror_neurons_t mirror,
    const action_t* observed_action,
    const action_t* executed_action,
    float* out_similarity
);

//=============================================================================
// Learning & Adaptation API
//=============================================================================

/**
 * @brief Learn from action demonstration
 *
 * WHAT: Process a complete action demonstration for imitation learning
 * WHY:  Enable one-shot or few-shot learning from demonstration
 * HOW:  Observe action sequence, build internal representation, prepare for reproduction
 *
 * This high-level function processes a demonstration by:
 * 1. Observing each action in sequence
 * 2. Building associations between actions
 * 3. Storing in working memory for replay
 * 4. Creating executable representation
 *
 * COMPLEXITY: O(m*n) where m = sequence length, n = num_mirror_neurons
 * THREAD-SAFE: No (requires external synchronization)
 *
 * @param mirror Mirror neuron system
 * @param actions Array of actions in demonstration
 * @param num_actions Number of actions in sequence
 * @param demonstrator_id ID of demonstrating agent
 * @return true on success, false on error
 */
bool mirror_neurons_learn_demonstration(
    mirror_neurons_t mirror,
    const action_t* actions,
    uint32_t num_actions,
    uint32_t demonstrator_id
);

/**
 * @brief Update association strengths through learning
 *
 * WHAT: Apply learning rule to update observation-execution associations
 * WHY:  Strengthen associations based on repeated co-activation
 * HOW:  Hebbian-like learning: neurons that fire together, wire together
 *
 * COMPLEXITY: O(n) where n = num_mirror_neurons
 * THREAD-SAFE: No (requires external synchronization)
 *
 * @param mirror Mirror neuron system
 * @return true on success, false on error
 */
bool mirror_neurons_update_associations(mirror_neurons_t mirror);

/**
 * @brief Decay inactive activations
 *
 * WHAT: Reduce activation levels of inactive neurons
 * WHY:  Prevent saturation, maintain dynamic responsiveness
 * HOW:  Apply exponential decay to activation levels
 *
 * COMPLEXITY: O(n) where n = num_mirror_neurons
 * THREAD-SAFE: No (requires external synchronization)
 *
 * @param mirror Mirror neuron system
 * @param delta_time_ms Time since last decay (milliseconds)
 * @return true on success, false on error
 */
bool mirror_neurons_decay_activations(mirror_neurons_t mirror, uint32_t delta_time_ms);

//=============================================================================
// Query & Analysis API
//=============================================================================

/**
 * @brief Get system statistics
 *
 * WHAT: Retrieve runtime statistics for mirror neuron system
 * WHY:  Monitor performance, learning progress, integration health
 * HOW:  Aggregate metrics from internal state
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: Yes (read-only with atomic reads)
 *
 * @param mirror Mirror neuron system
 * @param stats Pointer to store statistics
 * @return true on success, false on error
 */
bool mirror_neurons_get_stats(mirror_neurons_t mirror, mirror_neuron_stats_t* stats);

/**
 * @brief Get activation record for specific action
 *
 * WHAT: Retrieve detailed activation information for an action
 * WHY:  Analyze observation-execution associations
 * HOW:  Look up activation record by action ID
 *
 * COMPLEXITY: O(1) average (hash table lookup)
 * THREAD-SAFE: Yes (read-only)
 *
 * @param mirror Mirror neuron system
 * @param action_id Action to query
 * @param activation Pointer to store activation record
 * @return true if action found, false otherwise
 */
bool mirror_neurons_get_activation_record(
    mirror_neurons_t mirror,
    uint32_t action_id,
    mirror_activation_t* activation
);

/**
 * @brief Predict next action in sequence
 *
 * WHAT: Predict likely next action based on observed sequence
 * WHY:  Enable proactive behavior, action anticipation
 * HOW:  Use learned action sequences and associations
 *
 * COMPLEXITY: O(n) where n = num_learned_actions
 * THREAD-SAFE: Yes (read-only)
 *
 * @param mirror Mirror neuron system
 * @param previous_actions Recent action sequence
 * @param num_previous Number of previous actions
 * @param predicted_action Pointer to store predicted action
 * @param confidence Pointer to store prediction confidence (can be NULL)
 * @return true if prediction made, false otherwise
 */
bool mirror_neurons_predict_next_action(
    mirror_neurons_t mirror,
    const action_t* previous_actions,
    uint32_t num_previous,
    action_t* predicted_action,
    float* confidence
);

//=============================================================================
// Integration API - Cognitive Pipeline
//=============================================================================

/**
 * @brief Integrate with working memory
 *
 * WHAT: Store observed actions in working memory for replay
 * WHY:  Enable offline learning, action replay, sequence learning
 * HOW:  Convert observations to working memory items
 *
 * @param mirror Mirror neuron system
 * @param working_memory Working memory system handle
 * @return true on success, false on error
 */
bool mirror_neurons_integrate_working_memory(
    mirror_neurons_t mirror,
    void* working_memory  // working_memory_t* (avoid circular dependency)
);

/**
 * @brief Integrate with theory of mind
 *
 * WHAT: Use mirror neurons to infer agent intentions
 * WHY:  Understand why agents perform actions (goal inference)
 * HOW:  Map observed actions to likely goals via theory of mind
 *
 * @param mirror Mirror neuron system
 * @param theory_of_mind Theory of mind system handle
 * @return true on success, false on error
 */
bool mirror_neurons_integrate_theory_of_mind(
    mirror_neurons_t mirror,
    void* theory_of_mind  // tom_t* (avoid circular dependency)
);

/**
 * @brief Integrate with predictive processing
 *
 * WHAT: Use mirror neurons for action prediction
 * WHY:  Anticipate others' actions, generate predictions
 * HOW:  Feed mirror neuron activations to predictive network
 *
 * @param mirror Mirror neuron system
 * @param predictive_network Predictive network handle
 * @return true on success, false on error
 */
bool mirror_neurons_integrate_predictive(
    mirror_neurons_t mirror,
    void* predictive_network  // predictive_t* (avoid circular dependency)
);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get default configuration
 *
 * WHAT: Retrieve default mirror neuron configuration
 * WHY:  Provide sensible defaults for most use cases
 * HOW:  Return pre-configured struct
 *
 * @return Default configuration
 */
mirror_neuron_config_t mirror_neurons_get_default_config(void);

/**
 * @brief Create action from features
 *
 * WHAT: Helper to create action_t from feature vector
 * WHY:  Simplify action creation
 * HOW:  Fill in action_t struct with provided data
 *
 * @param action_id Unique action identifier
 * @param action_name Human-readable name
 * @param features Feature vector
 * @param num_features Number of features
 * @param agent_id Performing agent (0 = self)
 * @return action_t struct
 */
action_t mirror_neurons_create_action(
    uint32_t action_id,
    const char* action_name,
    const float* features,
    uint32_t num_features,
    uint32_t agent_id
);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_MIRROR_NEURONS_H
