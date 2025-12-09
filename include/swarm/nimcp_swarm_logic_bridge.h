/**
 * @file nimcp_swarm_logic_bridge.h
 * @brief Logic-Swarm Bridge: Neural Logic Gates for Swarm Intelligence
 *
 * WHAT: Bridges neural logic gate system with swarm intelligence modules
 * WHY:  Enables swarm agents to use symbolic logic for consensus, validation, and inference
 * HOW:  Combines logic gates (AND/OR/NOT/XOR/IMPLIES) with swarm agent states
 *
 * ARCHITECTURE:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                   Logic-Swarm Bridge                            │
 * ├─────────────────────────────────────────────────────────────────┤
 * │                                                                 │
 * │  Swarm Agent States  ──┐                                       │
 * │  (beliefs, votes)      │                                       │
 * │                        ├──►  Logic Gate Network                │
 * │  Consensus Rules  ─────┘     (AND/OR/NOT/XOR/IMPLIES)         │
 * │                                      │                         │
 * │                                      ▼                         │
 * │                               Validated Results                │
 * │                              (consensus, beliefs)              │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * USE CASES:
 * 1. Consensus Validation: AND/OR voting across agents
 * 2. Threat Detection: IMPLIES rules for security
 * 3. Belief Validation: XOR for contradiction detection
 * 4. Multi-agent Reasoning: Distributed inference
 *
 * BIOLOGICAL INSPIRATION:
 * - Honeybee waggle dance consensus (AND gates for agreement)
 * - Ant pheromone trail logic (OR gates for path finding)
 * - Neural coincidence detection (temporal AND gates)
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 * @version 1.0.0
 */

#ifndef NIMCP_SWARM_LOGIC_BRIDGE_H
#define NIMCP_SWARM_LOGIC_BRIDGE_H

#include "utils/validation/nimcp_common.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "core/neuron_types/nimcp_neural_logic.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/platform/nimcp_platform.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * FORWARD DECLARATIONS
 *============================================================================*/

typedef struct swarm_logic_bridge swarm_logic_bridge_t;

/*=============================================================================
 * CONFIGURATION STRUCTURES
 *============================================================================*/

/**
 * @brief Logic-Swarm Bridge configuration
 */
typedef struct {
    uint32_t max_rules;              /**< Maximum logic rules (default: 1000) */
    uint32_t rule_cache_size;        /**< Cached rule evaluations (default: 256) */
    bool enable_bio_async;           /**< Enable bio-async messaging (default: true) */
    float inference_threshold;       /**< Min confidence for inference (default: 0.7) */
    uint32_t max_agents;             /**< Maximum swarm agents (default: 100) */
    float timeout_ms;                /**< Rule evaluation timeout (default: 1000.0) */
} swarm_logic_bridge_config_t;

/**
 * @brief Logic rule for swarm validation
 */
typedef struct {
    uint32_t rule_id;                /**< Unique rule identifier */
    logic_gate_type_t gate_type;     /**< Logic gate type (AND/OR/NOT/XOR/IMPLIES) */
    uint32_t* input_agent_ids;       /**< Input agent IDs */
    uint32_t num_inputs;             /**< Number of input agents */
    float confidence_weight;         /**< Confidence weighting (0-1) */
    float threshold;                 /**< Activation threshold (0-1) */
    char description[128];           /**< Human-readable description */
} swarm_logic_rule_t;

/**
 * @brief Agent state for logic evaluation
 */
typedef struct {
    uint32_t agent_id;               /**< Agent identifier */
    float belief_value;              /**< Belief/vote value (0-1) */
    float confidence;                /**< Confidence in belief (0-1) */
    uint64_t timestamp_us;           /**< Timestamp (microseconds) */
    bool is_active;                  /**< Agent active flag */
} swarm_agent_state_t;

/**
 * @brief Logic evaluation result
 */
typedef struct {
    uint32_t rule_id;                /**< Rule that was evaluated */
    bool result;                     /**< Logical result (true/false) */
    float confidence;                /**< Result confidence (0-1) */
    uint32_t num_inputs_used;        /**< Number of inputs evaluated */
    uint64_t evaluation_time_us;     /**< Evaluation time (microseconds) */
    char explanation[256];           /**< Result explanation */
} swarm_logic_result_t;

/**
 * @brief Bridge statistics
 */
typedef struct {
    uint64_t total_evaluations;      /**< Total rule evaluations */
    uint64_t successful_evaluations; /**< Successful evaluations */
    uint64_t failed_evaluations;     /**< Failed evaluations */
    uint64_t cache_hits;             /**< Cache hits */
    uint64_t cache_misses;           /**< Cache misses */
    float avg_evaluation_time_us;    /**< Average evaluation time */
    uint32_t active_rules;           /**< Currently active rules */
    uint32_t active_agents;          /**< Currently active agents */
} swarm_logic_bridge_stats_t;

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *============================================================================*/

/**
 * @brief Get default bridge configuration
 *
 * WHAT: Provides default configuration for logic-swarm bridge
 * WHY:  Ensures safe defaults for all parameters
 * HOW:  Returns struct with sensible defaults
 *
 * @param config Configuration to fill with defaults
 */
void swarm_logic_bridge_get_default_config(swarm_logic_bridge_config_t* config);

/**
 * @brief Create logic-swarm bridge
 *
 * WHAT: Creates and initializes logic-swarm bridge
 * WHY:  Enables symbolic logic for swarm agents
 * HOW:  Allocates memory, initializes neural logic network, registers with bio-async
 *
 * @param config Configuration (NULL for defaults)
 * @return Bridge handle or NULL on failure
 */
swarm_logic_bridge_t* swarm_logic_bridge_create(const swarm_logic_bridge_config_t* config);

/**
 * @brief Destroy logic-swarm bridge
 *
 * WHAT: Cleans up and destroys logic-swarm bridge
 * WHY:  Prevents memory leaks
 * HOW:  Unregisters from bio-async, frees memory
 *
 * @param bridge Bridge to destroy (NULL-safe)
 */
void swarm_logic_bridge_destroy(swarm_logic_bridge_t* bridge);

/*=============================================================================
 * RULE MANAGEMENT
 *============================================================================*/

/**
 * @brief Add logic rule
 *
 * WHAT: Adds new logic rule to bridge
 * WHY:  Enables custom validation logic
 * HOW:  Creates rule, validates parameters, adds to rule table
 *
 * @param bridge Bridge handle
 * @param rule Rule to add
 * @return NIMCP_SUCCESS or error code
 */
nimcp_error_t swarm_logic_bridge_add_rule(swarm_logic_bridge_t* bridge,
                                           const swarm_logic_rule_t* rule);

/**
 * @brief Remove logic rule
 *
 * WHAT: Removes logic rule by ID
 * WHY:  Allows dynamic rule management
 * HOW:  Finds rule, removes from table, frees memory
 *
 * @param bridge Bridge handle
 * @param rule_id Rule ID to remove
 * @return NIMCP_SUCCESS or error code
 */
nimcp_error_t swarm_logic_bridge_remove_rule(swarm_logic_bridge_t* bridge,
                                              uint32_t rule_id);

/**
 * @brief Get rule by ID
 *
 * WHAT: Retrieves logic rule by ID
 * WHY:  Allows rule inspection
 * HOW:  Searches rule table, returns pointer
 *
 * @param bridge Bridge handle
 * @param rule_id Rule ID to find
 * @return Rule pointer or NULL if not found
 */
const swarm_logic_rule_t* swarm_logic_bridge_get_rule(swarm_logic_bridge_t* bridge,
                                                       uint32_t rule_id);

/**
 * @brief Get all rules
 *
 * WHAT: Retrieves all active rules
 * WHY:  Enables rule enumeration
 * HOW:  Returns array of rule pointers
 *
 * @param bridge Bridge handle
 * @param rules Output array of rule pointers
 * @param max_rules Maximum rules to return
 * @return Number of rules returned
 */
uint32_t swarm_logic_bridge_get_all_rules(swarm_logic_bridge_t* bridge,
                                           const swarm_logic_rule_t** rules,
                                           uint32_t max_rules);

/*=============================================================================
 * EVALUATION FUNCTIONS
 *============================================================================*/

/**
 * @brief Evaluate logic rules over agent states
 *
 * WHAT: Evaluates all logic rules given current agent states
 * WHY:  Core function for swarm consensus and validation
 * HOW:  Iterates rules, evaluates logic gates, returns results
 *
 * @param bridge Bridge handle
 * @param agent_states Array of agent states
 * @param num_agents Number of agents
 * @param results Output array for results
 * @param max_results Maximum results to return
 * @return Number of results returned, -1 on error
 */
int swarm_logic_bridge_evaluate(swarm_logic_bridge_t* bridge,
                                 const swarm_agent_state_t* agent_states,
                                 uint32_t num_agents,
                                 swarm_logic_result_t* results,
                                 uint32_t max_results);

/**
 * @brief Evaluate single rule
 *
 * WHAT: Evaluates specific rule given agent states
 * WHY:  Allows targeted rule evaluation
 * HOW:  Finds rule, evaluates gate, returns result
 *
 * @param bridge Bridge handle
 * @param rule_id Rule to evaluate
 * @param agent_states Array of agent states
 * @param num_agents Number of agents
 * @param result Output result
 * @return NIMCP_SUCCESS or error code
 */
nimcp_error_t swarm_logic_bridge_evaluate_rule(swarm_logic_bridge_t* bridge,
                                                uint32_t rule_id,
                                                const swarm_agent_state_t* agent_states,
                                                uint32_t num_agents,
                                                swarm_logic_result_t* result);

/*=============================================================================
 * CONSENSUS VALIDATION
 *============================================================================*/

/**
 * @brief Validate consensus across agents
 *
 * WHAT: Validates consensus using AND/OR voting
 * WHY:  Enables democratic decision making
 * HOW:  Applies logic gates to agent votes
 *
 * @param bridge Bridge handle
 * @param votes Array of agent votes (0-1)
 * @param num_votes Number of votes
 * @param consensus_type Gate type (LOGIC_GATE_AND for unanimous, LOGIC_GATE_OR for majority)
 * @param result Output consensus result
 * @return NIMCP_SUCCESS or error code
 */
nimcp_error_t swarm_logic_bridge_validate_consensus(swarm_logic_bridge_t* bridge,
                                                     const float* votes,
                                                     uint32_t num_votes,
                                                     logic_gate_type_t consensus_type,
                                                     swarm_logic_result_t* result);

/**
 * @brief Detect belief contradictions
 *
 * WHAT: Detects contradictions in agent beliefs using XOR
 * WHY:  Ensures logical consistency
 * HOW:  Applies XOR gates to belief pairs
 *
 * @param bridge Bridge handle
 * @param beliefs Array of belief states
 * @param num_beliefs Number of beliefs
 * @param contradictions Output array of contradiction pairs
 * @param max_contradictions Maximum contradictions to return
 * @return Number of contradictions found, -1 on error
 */
int swarm_logic_bridge_detect_contradiction(swarm_logic_bridge_t* bridge,
                                            const swarm_agent_state_t* beliefs,
                                            uint32_t num_beliefs,
                                            uint32_t (*contradictions)[2],
                                            uint32_t max_contradictions);

/**
 * @brief Validate implication rule
 *
 * WHAT: Validates if A implies B for agent states
 * WHY:  Enables conditional reasoning
 * HOW:  Applies IMPLIES gate to agent pairs
 *
 * @param bridge Bridge handle
 * @param antecedent_agent Agent ID for antecedent (if A...)
 * @param consequent_agent Agent ID for consequent (then B...)
 * @param agent_states Array of agent states
 * @param num_agents Number of agents
 * @param result Output implication result
 * @return NIMCP_SUCCESS or error code
 */
nimcp_error_t swarm_logic_bridge_validate_implication(swarm_logic_bridge_t* bridge,
                                                       uint32_t antecedent_agent,
                                                       uint32_t consequent_agent,
                                                       const swarm_agent_state_t* agent_states,
                                                       uint32_t num_agents,
                                                       swarm_logic_result_t* result);

/*=============================================================================
 * BIO-ASYNC INTEGRATION
 *============================================================================*/

/**
 * @brief Process bio-async inbox
 *
 * WHAT: Processes pending bio-async messages
 * WHY:  Enables asynchronous communication with other modules
 * HOW:  Calls bio_router_process_inbox with bridge context
 *
 * @param bridge Bridge handle
 * @return Number of messages processed, -1 on error
 */
int swarm_logic_bridge_process_inbox(swarm_logic_bridge_t* bridge);

/**
 * @brief Send logic evaluation result
 *
 * WHAT: Sends logic result to target module via bio-async
 * WHY:  Enables distributed logic evaluation
 * HOW:  Constructs message, sends via bio_router
 *
 * @param bridge Bridge handle
 * @param target_module Target module ID
 * @param result Result to send
 * @return NIMCP_SUCCESS or error code
 */
nimcp_error_t swarm_logic_bridge_send_result(swarm_logic_bridge_t* bridge,
                                              bio_module_id_t target_module,
                                              const swarm_logic_result_t* result);

/**
 * @brief Broadcast consensus result
 *
 * WHAT: Broadcasts consensus to all swarm modules
 * WHY:  Enables swarm-wide synchronization
 * HOW:  Constructs broadcast message, sends via bio_router
 *
 * @param bridge Bridge handle
 * @param result Consensus result to broadcast
 * @return NIMCP_SUCCESS or error code
 */
nimcp_error_t swarm_logic_bridge_broadcast_consensus(swarm_logic_bridge_t* bridge,
                                                      const swarm_logic_result_t* result);

/*=============================================================================
 * STATISTICS AND DEBUGGING
 *============================================================================*/

/**
 * @brief Get bridge statistics
 *
 * WHAT: Retrieves bridge performance statistics
 * WHY:  Enables monitoring and debugging
 * HOW:  Returns snapshot of internal counters
 *
 * @param bridge Bridge handle
 * @param stats Output statistics structure
 * @return NIMCP_SUCCESS or error code
 */
nimcp_error_t swarm_logic_bridge_get_stats(swarm_logic_bridge_t* bridge,
                                            swarm_logic_bridge_stats_t* stats);

/**
 * @brief Reset statistics
 *
 * WHAT: Resets all statistics counters to zero
 * WHY:  Enables periodic monitoring
 * HOW:  Zeros all counters in stats structure
 *
 * @param bridge Bridge handle
 */
void swarm_logic_bridge_reset_stats(swarm_logic_bridge_t* bridge);

/**
 * @brief Clear evaluation cache
 *
 * WHAT: Clears cached evaluation results
 * WHY:  Forces re-evaluation of all rules
 * HOW:  Invalidates all cache entries
 *
 * @param bridge Bridge handle
 */
void swarm_logic_bridge_clear_cache(swarm_logic_bridge_t* bridge);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SWARM_LOGIC_BRIDGE_H */
