//=============================================================================
// nimcp_ethics_policies.c - Policy Management and Evaluation
//=============================================================================
// RESPONSIBILITY: Policy storage, evaluation strategies, and management
//
// This module implements the Strategy Pattern for policy evaluation,
// hash table-based policy storage, and policy management operations.
//=============================================================================

#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/ethics/nimcp_ethics_internal.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/error/nimcp_error_codes.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>

#define LOG_MODULE "ethics_policies"

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for ethics_policies module */
static nimcp_health_agent_t* g_ethics_policies_health_agent = NULL;

/**
 * @brief Set health agent for ethics_policies heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void ethics_policies_set_health_agent(nimcp_health_agent_t* agent) {
    g_ethics_policies_health_agent = agent;
}

/** @brief Send heartbeat from ethics_policies module */
static inline void ethics_policies_heartbeat(const char* operation, float progress) {
    if (g_ethics_policies_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_ethics_policies_health_agent, operation, progress);
    }
}


//=============================================================================
// Strategy Pattern - Policy Evaluator Functions
//=============================================================================

/**
 * @brief Evaluates harm violation severity
 *
 * WHY: Isolates harm evaluation logic. Makes code testable and maintainable.
 *
 * COMPLEXITY: O(1)
 */
float ethics_evaluate_harm_policy(const action_context_t* action)
{
    return action ? action->predicted_harm : 0.0F;
}

/**
 * @brief Evaluates unfairness violation severity
 *
 * WHY: Separates fairness logic from other policies for clarity.
 *
 * COMPLEXITY: O(1)
 */
float ethics_evaluate_unfairness_policy(const action_context_t* action)
{
    return action ? action->fairness_violation : 0.0F;
}

/**
 * @brief Evaluates deception violation severity
 *
 * COMPLEXITY: O(1)
 */
float ethics_evaluate_deception_policy(const action_context_t* action)
{
    return action ? action->deception_level : 0.0F;
}

/**
 * @brief Evaluates autonomy violation severity
 *
 * COMPLEXITY: O(1)
 */
float ethics_evaluate_autonomy_policy(const action_context_t* action)
{
    return action ? action->autonomy_violation : 0.0F;
}

/**
 * @brief Evaluates privacy violation severity
 *
 * COMPLEXITY: O(1)
 */
float ethics_evaluate_privacy_policy(const action_context_t* action)
{
    return action ? action->privacy_violation : 0.0F;
}

/**
 * @brief Evaluates consent violation severity
 *
 * COMPLEXITY: O(1)
 */
float ethics_evaluate_consent_policy(const action_context_t* action)
{
    return action ? action->consent_violation : 0.0F;
}

/**
 * @brief Initializes policy evaluation strategy table
 *
 * WHY: Replaces switch statement with O(1) function pointer lookup.
 * Follows Open/Closed Principle - new strategies can be added without
 * modifying existing code.
 *
 * COMPLEXITY: O(1)
 */
void ethics_init_strategy_table(policy_strategy_table_t* table)
{
    // Guard clause: Validate input
    if (!table)
        return;

    memset(table->evaluators, 0, sizeof(table->evaluators));

    // Register evaluation strategies
    table->evaluators[ETHICS_VIOLATION_HARM] = ethics_evaluate_harm_policy;
    table->evaluators[ETHICS_VIOLATION_UNFAIRNESS] = ethics_evaluate_unfairness_policy;
    table->evaluators[ETHICS_VIOLATION_DECEPTION] = ethics_evaluate_deception_policy;
    table->evaluators[ETHICS_VIOLATION_AUTONOMY] = ethics_evaluate_autonomy_policy;
    table->evaluators[ETHICS_VIOLATION_PRIVACY] = ethics_evaluate_privacy_policy;
    table->evaluators[ETHICS_VIOLATION_CONSENT] = ethics_evaluate_consent_policy;
}

/**
 * @brief Evaluates a policy using the strategy table
 *
 * WHY: O(1) dispatch to appropriate evaluator vs O(n) switch statement.
 * Makes adding new violation types trivial - just register a new function.
 *
 * COMPLEXITY: O(1)
 *
 * @return Severity score [0.0, 1.0] or 0.0 if invalid
 */
float ethics_evaluate_policy_strategy(const policy_strategy_table_t* table,
                                      const ethics_policy_t* policy, const action_context_t* action)
{
    // Guard clause: Validate inputs
    if (!table || !policy || !action)
        return 0.0F;

    // Guard clause: Check bounds
    if (policy->violation_type >= 16)
        return 0.0F;

    policy_evaluator_fn evaluator = table->evaluators[policy->violation_type];
    // Guard clause: Check if evaluator exists
    if (!evaluator)
        return 0.0F;

    return evaluator(action);
}

/**
 * @brief Evaluates all enabled policies against action
 *
 * WHY: Extracted policy evaluation loop. No nested ifs - guard clauses only.
 * Uses strategy pattern for O(1) policy dispatch.
 *
 * ALGORITHM:
 * 1. Iterate through all policies once (O(n))
 * 2. For each enabled policy, evaluate using strategy table (O(1))
 * 3. Track worst violation severity (O(1))
 *
 * COMPLEXITY: O(n) where n = num_policies
 *
 * @param worst_violation Output parameter for worst violation type
 * @param worst_severity Output parameter for worst severity
 * @return Overall policy compliance score [0.0, 1.0]
 */
float ethics_evaluate_all_policies(ethics_engine_t engine, const action_context_t* action,
                                   ethics_violation_type_t* worst_violation, float* worst_severity)
{
    // Guard clause: Validate inputs
    if (!engine || !action || !worst_violation || !worst_severity) {
        return 1.0F;
    }

    float policy_score = 1.0F;
    *worst_violation = ETHICS_VIOLATION_TYPE_NONE;
    *worst_severity = 0.0F;

    uint32_t num_policies = ethics_engine_get_num_policies(engine);
    const policy_strategy_table_t* strategy_table = ethics_engine_get_strategy_table(engine);

    // Single pass through policies - O(n)
    for (uint32_t i = 0; i < num_policies; i++) {
        const ethics_policy_t* policy = ethics_engine_get_policy(engine, i);

        // Guard clause: Skip disabled policies
        if (!policy || !policy->enabled)
            continue;

        // O(1) evaluation using strategy pattern
        float severity = ethics_evaluate_policy_strategy(strategy_table, policy, action);

        // Guard clause: Check threshold
        if (severity <= policy->severity_threshold)
            continue;

        // Update worst violation
        if (severity > *worst_severity) {
            *worst_severity = severity;
            *worst_violation = (ethics_violation_type_t)policy->violation_type;
        }

        policy_score = fminf(policy_score, 1.0F - severity);
    }

    return policy_score;
}

//=============================================================================
// Policy Management Functions
//=============================================================================

/**
 * @brief Adds policy to engine using hash table
 *
 * WHY: O(1) insertion via hash table vs O(n) array append.
 * Maintains both hash table (for lookups) and array (for iteration).
 *
 * COMPLEXITY: O(1) average case
 *
 * @return true on success, false on failure
 */
bool ethics_add_policy(ethics_engine_t engine, const ethics_policy_t* policy)
{
    // Guard clause: Validate inputs
    if (!engine || !policy)
        return false;

    // Add to internal storage
    return ethics_engine_add_policy_internal(engine, policy);
}

/**
 * @brief Removes policy using hash table and array compaction
 *
 * WHY: O(1) hash removal + O(n) array compaction.
 * Must maintain both structures in sync.
 *
 * COMPLEXITY: O(n) for array compaction
 *
 * @return true if removed, false if not found
 */
bool ethics_remove_policy(ethics_engine_t engine, uint32_t policy_id)
{
    // Guard clause: Validate input
    if (!engine)
        {

            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "ethics_remove_policy: engine is NULL");

            return false;

        }

    return ethics_engine_remove_policy_internal(engine, policy_id);
}

/**
 * @brief Retrieves all policies
 *
 * WHY: Provides access to policy array for iteration/display.
 *
 * COMPLEXITY: O(n) for memcpy
 *
 * @return Number of policies copied
 */
uint32_t ethics_get_policies(ethics_engine_t engine, ethics_policy_t* policies,
                             uint32_t max_policies)
{
    // Guard clause: Validate inputs
    if (!engine || !policies)
        return 0;

    uint32_t num_policies = ethics_engine_get_num_policies(engine);
    uint32_t count = (num_policies < max_policies) ? num_policies : max_policies;

    for (uint32_t i = 0; i < count; i++) {
        const ethics_policy_t* policy = ethics_engine_get_policy(engine, i);
        if (policy) {
            policies[i] = *policy;
        }
    }

    return count;
}

/**
 * @brief Add ethical policy (public API wrapper)
 *
 * WHY: Allows dynamic policy management, enabling systems to add new
 * ethical constraints at runtime.
 *
 * COMPLEXITY: O(1) amortized (O(n) when array needs realloc)
 * THREAD SAFETY: Not thread-safe, caller must synchronize
 *
 * @param engine Ethics engine
 * @param policy Policy to add
 * @return true on success, false on error
 */
bool ethics_engine_add_policy(ethics_engine_t engine, const ethics_policy_t* policy)
{
    return ethics_add_policy(engine, policy);
}

/**
 * @brief Remove ethical policy by policy_id (public API wrapper)
 *
 * WHY: Allows dynamic policy management, enabling systems to remove
 * outdated or incorrect ethical constraints.
 *
 * COMPLEXITY: O(n) where n = number of policies
 * THREAD SAFETY: Not thread-safe, caller must synchronize
 *
 * @param engine Ethics engine
 * @param policy_id Policy ID to remove
 * @return true if policy was found and removed, false otherwise
 */
bool ethics_engine_remove_policy(ethics_engine_t engine, uint32_t policy_id)
{
    return ethics_remove_policy(engine, policy_id);
}

/**
 * @brief Get violation type name
 *
 * @param type Violation type
 * @return Human-readable name
 */
const char* ethics_violation_type_name(ethics_violation_type_t type)
{
    switch (type) {
        case ETHICS_VIOLATION_TYPE_NONE:
            return "None";
        case ETHICS_VIOLATION_TYPE_HARM:
            return "Harm";
        case ETHICS_VIOLATION_TYPE_UNFAIRNESS:
            return "Unfairness";
        case ETHICS_VIOLATION_TYPE_DECEPTION:
            return "Deception";
        case ETHICS_VIOLATION_TYPE_AUTONOMY:
            return "Autonomy Violation";
        case ETHICS_VIOLATION_TYPE_PRIVACY:
            return "Privacy Violation";
        case ETHICS_VIOLATION_TYPE_CONSENT:
            return "Consent Violation";
        case ETHICS_VIOLATION_TYPE_DIGNITY:
            return "Dignity Violation";
        case ETHICS_VIOLATION_TYPE_GOLDEN_RULE:
            return "Golden Rule Violation";
        default:
            return "Unknown";
    }
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

/**
 * WHAT: Query knowledge graph for Ethics Policies self-knowledge
 * WHY:  Enable self-awareness about module's role and connections
 * HOW:  Query KG for entity observations and relations
 */
int ethics_policies_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Ethics_Policies_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            LOG_DEBUG("Ethics policies self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Ethics_Policies_Module");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Ethics_Policies_Module");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
