//=============================================================================
// nimcp_ethics.h - Ethics and Empathy Layer
//=============================================================================

#ifndef NIMCP_ETHICS_H
#define NIMCP_ETHICS_H

#include "nimcp_events.h"
#include "nimcp_export.h"
#include "nimcp_neuralnet.h"
#include "nimcp_protocol.h"

/**
 * @file nimcp_ethics.h
 * @brief Ethics and empathy mechanisms for NIMCP 2.0
 *
 * Implements value-based filtering, empathy networks (mirror neurons),
 * and ethical regulation of neural activity.
 */

//=============================================================================
// Ethics Constants and Types
//=============================================================================

/**
 * @brief Ethical violation types
 */
typedef enum {
    ETHICS_VIOLATION_NONE = 0x00,
    ETHICS_VIOLATION_HARM = 0x01,       /**< Potential harm detected */
    ETHICS_VIOLATION_UNFAIRNESS = 0x02, /**< Unfair action detected */
    ETHICS_VIOLATION_DECEPTION = 0x03,  /**< Deceptive pattern detected */
    ETHICS_VIOLATION_PRIVACY = 0x04,    /**< Privacy violation */
    ETHICS_VIOLATION_AUTONOMY = 0x05,   /**< Autonomy violation */
    ETHICS_VIOLATION_CONSENT = 0x06,    /**< Consent violation */
    ETHICS_VIOLATION_DIGNITY = 0x07,    /**< Dignity violation */
    ETHICS_VIOLATION_RIGHTS = 0x08,     /**< Rights violation */
    ETHICS_VIOLATION_CUSTOM = 0x80      /**< User-defined violation */
} ethics_violation_enum_t;

// Backward compatibility
typedef ethics_violation_enum_t ethics_violation_t;

/**
 * @brief Ethical action types
 */
typedef enum {
    ETHICS_ACTION_ALLOW,  /**< Allow event to proceed */
    ETHICS_ACTION_BLOCK,  /**< Block event completely */
    ETHICS_ACTION_MODIFY, /**< Modify event before forwarding */
    ETHICS_ACTION_DEFER,  /**< Defer to higher authority */
    ETHICS_ACTION_LOG     /**< Log but allow */
} ethics_action_t;

/**
 * @brief Ethical policy rule
 */
typedef struct {
    feature_code_t feature_pattern;    /**< Feature code pattern to match */
    uint32_t feature_mask;             /**< Mask for pattern matching */
    ethics_violation_t violation_type; /**< Type of violation */
    float severity_threshold;          /**< Minimum severity to trigger */
    ethics_action_t action;            /**< Action to take */
    bool requires_consensus;           /**< Requires multi-node agreement */

    // NIMCP 2.5 extensions
    uint32_t policy_id;        /**< Unique policy ID */
    char name[128];            /**< Policy name */
    char description[256];     /**< Policy description */
    float confidence_required; /**< Required confidence level */
    bool enabled;              /**< Whether policy is active */
    bool learned;              /**< Whether policy was learned */
} ethics_policy_t;

/**
 * @brief Empathy (mirror neuron) state
 */
typedef struct {
    uint32_t observed_node_id;       /**< Node being observed */
    feature_code_t observed_pattern; /**< Observed activity pattern */
    float mirror_activation;         /**< Mirror neuron activation */
    float predicted_impact;          /**< Predicted impact of action */
    uint64_t timestamp;              /**< When observation occurred */
} empathy_state_t;

//=============================================================================
// Ethics Engine
//=============================================================================

/**
 * @brief Ethics evaluation callback
 *
 * Called when ethical evaluation is needed.
 *
 * @param packet Event packet being evaluated
 * @param violation Detected violation type
 * @param severity Violation severity (0.0-1.0)
 * @param context User context
 * @return Recommended action
 */
typedef ethics_action_t (*ethics_eval_callback_t)(const event_packet_t* packet,
                                                  ethics_violation_t violation, float severity,
                                                  void* context);

/**
 * @brief Ethics engine configuration
 */
typedef struct {
    ethics_policy_t* policies;       /**< Array of ethical policies */
    uint32_t num_policies;           /**< Number of policies */
    ethics_eval_callback_t callback; /**< Evaluation callback */
    void* callback_context;          /**< Callback context */
    float default_severity;          /**< Default violation severity */
    bool enable_learning;            /**< Learn from decisions */

    // NIMCP 2.5 extensions for Golden Rule
    uint32_t action_feature_size; /**< Size of action feature vector */
    uint32_t max_agents;          /**< Maximum number of agents */
    float golden_rule_threshold;  /**< Threshold for Golden Rule (default 0.0) */
    float empathy_weight;         /**< Weight for empathy signals (0-1) */
} ethics_config_t;

/**
 * @brief Ethics engine state
 */
typedef struct ethics_engine_struct* ethics_engine_t;

/**
 * @brief Create an ethics engine
 *
 * @param config Engine configuration
 * @return Ethics engine handle, or NULL on error
 */
ethics_engine_t ethics_engine_create(const ethics_config_t* config);

/**
 * @brief Destroy an ethics engine
 *
 * @param engine Engine to destroy
 */
void ethics_engine_destroy(ethics_engine_t engine);

/**
 * @brief Evaluate an event packet for ethical violations
 *
 * @param engine Ethics engine
 * @param packet Event packet to evaluate
 * @param violation Output: detected violation type
 * @param severity Output: violation severity (0.0-1.0)
 * @return Recommended action
 */
ethics_action_t ethics_engine_evaluate(ethics_engine_t engine, const event_packet_t* packet,
                                       ethics_violation_t* violation, float* severity);

/**
 * @brief Add ethical policy
 *
 * @param engine Ethics engine
 * @param policy Policy to add
 * @return true on success
 */
bool ethics_engine_add_policy(ethics_engine_t engine, const ethics_policy_t* policy);

/**
 * @brief Remove ethical policy
 *
 * @param engine Ethics engine
 * @param index Policy index to remove
 * @return true on success
 */
bool ethics_engine_remove_policy(ethics_engine_t engine, uint32_t index);

/**
 * @brief Update policy weights based on feedback
 *
 * @param engine Ethics engine
 * @param violation Violation type that occurred
 * @param was_correct Whether the decision was correct
 * @param learning_rate Learning rate for update
 * @return true on success
 */
bool ethics_engine_update_policy(ethics_engine_t engine, ethics_violation_t violation,
                                 bool was_correct, float learning_rate);

//=============================================================================
// Empathy Network (Mirror Neurons)
//=============================================================================

/**
 * @brief Empathy network configuration
 */
typedef struct {
    neural_network_t mirror_network; /**< Mirror neuron network */
    uint32_t observation_window_ms;  /**< Time window for observations */
    float empathy_threshold;         /**< Minimum activation for empathy */
} empathy_config_t;

/**
 * @brief Empathy network state
 */
typedef struct empathy_network_struct* empathy_network_t;

/**
 * @brief Create an empathy network
 *
 * @param config Network configuration
 * @return Empathy network handle, or NULL on error
 */
empathy_network_t empathy_network_create(const empathy_config_t* config);

/**
 * @brief Destroy an empathy network
 *
 * @param network Network to destroy
 */
void empathy_network_destroy(empathy_network_t network);

/**
 * @brief Observe another node's activity
 *
 * @param network Empathy network
 * @param packet Event packet from observed node
 * @param timestamp Current time
 * @return true if empathy was triggered
 */
bool empathy_network_observe(empathy_network_t network, const event_packet_t* packet,
                             uint64_t timestamp);

/**
 * @brief Get current empathy state
 *
 * @param network Empathy network
 * @param state Output: current empathy state
 * @return true on success
 */
bool empathy_network_get_state(empathy_network_t network, empathy_state_t* state);

/**
 * @brief Predict impact of proposed action
 *
 * @param network Empathy network
 * @param packet Proposed event packet
 * @return Predicted impact (0.0-1.0, negative = harmful)
 */
float empathy_network_predict_impact(empathy_network_t network, const event_packet_t* packet);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Create ETHICS_POLICY control message
 *
 * @param policy Policy to encode
 * @param buffer Output buffer
 * @param buffer_size Buffer size
 * @return Bytes written, or -1 on error
 */
int ethics_create_policy_message(const ethics_policy_t* policy, uint8_t* buffer,
                                 uint32_t buffer_size);

/**
 * @brief Parse ETHICS_POLICY control message
 *
 * @param buffer Input buffer
 * @param buffer_size Buffer size
 * @param policy Output: parsed policy
 * @return Bytes read, or -1 on error
 */
int ethics_parse_policy_message(const uint8_t* buffer, uint32_t buffer_size,
                                ethics_policy_t* policy);

/**
 * @brief Get human-readable name for violation type
 *
 * @param violation Violation type
 * @return Violation name string
 */
const char* ethics_violation_name(ethics_violation_t violation);

//=============================================================================
// NIMCP 2.5 - Golden Rule Ethics API
//=============================================================================

/**
 * @brief Agent identifier type
 */
typedef uint32_t agent_id_t;

/**
 * @brief Violation types for Golden Rule evaluation
 */
typedef enum {
    ETHICS_VIOLATION_TYPE_NONE = 0,
    ETHICS_VIOLATION_TYPE_HARM,
    ETHICS_VIOLATION_TYPE_UNFAIRNESS,
    ETHICS_VIOLATION_TYPE_DECEPTION,
    ETHICS_VIOLATION_TYPE_AUTONOMY,
    ETHICS_VIOLATION_TYPE_PRIVACY,
    ETHICS_VIOLATION_TYPE_CONSENT,
    ETHICS_VIOLATION_TYPE_DIGNITY,
    ETHICS_VIOLATION_TYPE_GOLDEN_RULE
} ethics_violation_type_t;

/**
 * @brief Action context for Golden Rule evaluation
 */
typedef struct {
    float* features;              /**< Action feature vector */
    uint32_t num_features;        /**< Number of features */
    agent_id_t* affected_agents;  /**< IDs of affected agents */
    uint32_t num_affected_agents; /**< Number of affected agents */
    float predicted_harm;         /**< Predicted harm level (0-1) */

    // Specific violation metrics
    float fairness_violation; /**< Fairness violation level (0-1) */
    float deception_level;    /**< Deception level (0-1) */
    float autonomy_violation; /**< Autonomy violation level (0-1) */
    float privacy_violation;  /**< Privacy violation level (0-1) */
    float consent_violation;  /**< Consent violation level (0-1) */
} action_context_t;

/**
 * @brief Action outcome for learning
 */
typedef struct {
    agent_id_t affected_agent; /**< Agent that was affected */
    float actual_harm;         /**< Actual harm that occurred (0-1) */
    float actual_benefit;      /**< Actual benefit that occurred (0-1) */
    float emotional_impact;    /**< Emotional impact (-1 to +1) */
    float material_impact;     /**< Material impact (-1 to +1) */
    float autonomy_impact;     /**< Autonomy impact (-1 to +1) */
    float impact_magnitude;    /**< Overall magnitude (0-1) */
    float uncertainty;         /**< Uncertainty in assessment (0-1) */
} action_outcome_t;

/**
 * @brief Ethics evaluation result
 */
typedef struct {
    bool allowed;                              /**< Whether action is allowed */
    float confidence;                          /**< Confidence in decision (0-1) */
    float golden_rule_score;                   /**< Golden Rule score (-1 to +1) */
    ethics_action_t recommended_action;        /**< Recommended action */
    ethics_violation_type_t primary_violation; /**< Primary violation type */
    char explanation[512];                     /**< Human-readable explanation */
} ethics_evaluation_t;

/**
 * @brief Extended empathy state for Golden Rule evaluation
 */
typedef struct {
    agent_id_t agent_id;     /**< Agent being simulated */
    float emotional_valence; /**< Emotional state (-1 to +1) */
    float material_impact;   /**< Material impact (-1 to +1) */
    float autonomy_impact;   /**< Autonomy impact (-1 to +1) */
    float impact_magnitude;  /**< Overall magnitude (0-1) */
    float uncertainty;       /**< Uncertainty (0-1) */
    bool active;             /**< Whether state is active */
} empathy_state_extended_t;

/**
 * @brief Extended ethics configuration for Golden Rule
 */
typedef struct {
    uint32_t action_feature_size; /**< Size of action feature vector */
    uint32_t max_agents;          /**< Maximum number of agents */
    float golden_rule_threshold;  /**< Threshold for Golden Rule (default 0.0) */
    float empathy_weight;         /**< Weight for empathy signals (0-1) */
    bool enable_learning;         /**< Enable learning from outcomes */
} ethics_config_extended_t;

/**
 * @brief Violation record for logging (NIMCP 2.5)
 */
typedef struct {
    ethics_violation_type_t violation_type; /**< Type of violation */
    float severity;                         /**< Violation severity (0-1) */
    uint64_t timestamp;                     /**< When violation occurred */
    char action_description[256];           /**< Description of action */
    agent_id_t affected_agent;              /**< Agent that was affected */
    float golden_rule_score;                /**< Golden Rule evaluation score */
} violation_record_t;

/**
 * @brief Evaluate action using Golden Rule
 *
 * This is the core NIMCP 2.5 function that implements the hard-wired
 * Golden Rule: "Do unto others as you would have them done unto you"
 *
 * @param engine Ethics engine
 * @param action Action to evaluate
 * @return Evaluation result
 */
ethics_evaluation_t ethics_engine_evaluate_action(ethics_engine_t engine,
                                                  const action_context_t* action);

/**
 * @brief Learn from action outcome
 *
 * Updates empathy networks based on observed outcomes
 *
 * @param engine Ethics engine
 * @param action Action that was taken
 * @param outcome Observed outcome
 * @return true on success
 */
bool ethics_learn_from_outcome(ethics_engine_t engine, const action_context_t* action,
                               const action_outcome_t* outcome);

/**
 * @brief Simulate agent's perspective
 *
 * Uses empathy network to predict how an agent would experience an action
 *
 * @param network Empathy network
 * @param agent Agent to simulate
 * @param action Action to evaluate
 * @return Simulated empathy state
 */
empathy_state_extended_t empathy_network_simulate_agent(empathy_network_t network, agent_id_t agent,
                                                        const action_context_t* action);

/**
 * @brief Print evaluation result
 *
 * @param eval Evaluation to print
 */
void ethics_print_evaluation(const ethics_evaluation_t* eval);

/**
 * @brief Get violation type name
 *
 * @param type Violation type
 * @return Human-readable name
 */
const char* ethics_violation_type_name(ethics_violation_type_t type);

/**
 * @brief Ethics statistics
 */
typedef struct {
    uint64_t total_evaluations;      /**< Total evaluations performed */
    uint64_t violations_detected;    /**< Total violations detected */
    uint64_t actions_blocked;        /**< Actions blocked */
    uint64_t learning_updates;       /**< Learning updates performed */
    float average_golden_rule_score; /**< Average Golden Rule score */
    float avg_golden_rule_score;     /**< Alias for average_golden_rule_score */
    uint32_t num_policies;           /**< Number of active policies */
    uint32_t num_violations_logged;  /**< Number of logged violations */
} ethics_statistics_t;

/**
 * @brief Get ethics engine statistics
 *
 * @param engine Ethics engine
 * @param stats Output statistics
 * @return true on success
 */
bool ethics_get_statistics(ethics_engine_t engine, ethics_statistics_t* stats);

#endif  // NIMCP_ETHICS_H
